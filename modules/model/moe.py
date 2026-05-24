import math
import torch
from torch import nn
import torch.nn.functional as F

from modules.model.router import Router, compute_aux_loss
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm
from modules.model.experts import SelfAttention



class ParallelSparseMoELayer(nn.Module):
    """a sparse MoE layer that routes tokens to experts in parallel using matrix multiplication for efficiency"""
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        
        # vectorized expert parameters (without bias)
        # param layout (num_experts, inputs, outputs)
        self.gate_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.up_proj   = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        
        # init weights
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
        
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        # expects x of shape [batch_size, seq_len, hidden_size]
        # topk_weights and topk_indices of shape [batch_size, seq_len, top_k]
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])  # [total_tokens, hidden_size]
        
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])  # [total_tokens, top_k]
        topk_indices = topk_indices.view(-1, topk_indices.shape[-1])  # [total_tokens, top_k]

        # map each token to its top k experts (one-hot)
        b_matrix = F.one_hot(topk_indices, num_classes=self.num_experts).to(x_flat.dtype) # [total_tokens, top_k, num_experts]
        
        # scale the one hot matrix by the corresponding topk weights
        scaled_b_matrix = b_matrix * topk_weights.unsqueeze(-1)
        
        # collapse the top_k dimension to get a routing map per token
        routing_map = scaled_b_matrix.sum(dim=1) # [total_tokens, num_experts]

        # parallel token gathering (grouping tokens by expert)
        gathered_tokens = torch.einsum("td,te->etd", x_flat, routing_map) # [num_experts, total_tokens, hidden_size]


        # expert (MLP) computations in parallel
        up = torch.bmm(gathered_tokens, self.up_proj)     # [num_experts, total_tokens, intermediate_size]
        gate = torch.bmm(gathered_tokens, self.gate_proj) # [num_experts, total_tokens, intermediate_size]
        act = self.activation(gate) * up                  # [num_experts, total_tokens, intermediate_size]
        expert_outputs = torch.bmm(act, self.down_proj)   # [num_experts, total_tokens, hidden_size]

       
        # recombining the distributed representations
        # multiply expert outputs back by the routing weights and sum them together
        combined_output = torch.einsum("etd,te->td", expert_outputs, routing_map) # [total_tokens, hidden_size]

        return combined_output.view(*orig_shape)


class LoopMixtureOfExperts(nn.Module):
    """a Mixture of Experts module that routes tokens to a mixture of attention and feedforward experts in multiple loops"""
    def __init__(
        self,
        hidden_size: int, 
        intermediate_size: int, 
        num_experts: int, 
        top_k: int = 2,
        n_loops: int = 8, 
        dropout: float = 0.0,
        num_attn_experts: int = 4,
        temperature: float = 1.0,
    ):
        """Mixture of Experts module with multiple loops of routing to a mixture of attention and feedforward experts

        Args:
            hidden_size (int): hidden size of the input and output representations
            intermediate_size (int): intermediate size of the feedforward layers
            num_experts (int): number of MLP experts in the mixture
            top_k (int, optional): number of top experts to route each token to. Defaults to 2.
            n_loops (int, optional): number of routing loops. Defaults to 8.
            dropout (float, optional): dropout probability. Defaults to 0.0.
            num_attn_experts (int, optional): number of attention experts. Defaults to 4.
            temperature (float, optional): temperature for the router. Defaults to 1.0.
        """
        super().__init__()
        self._num_experts = num_experts
        self._num_attn_experts = num_attn_experts
        self.top_k = top_k
        self.n_loops = n_loops
        self.temperature = temperature
        
        # experts
        experts = []
        experts.extend([
            SelfAttention(input_size=hidden_size, dropout=dropout, num_heads=16)
            for _ in range(num_attn_experts)
        ])
        experts.append(nn.Identity()) # identity expert for skipping
        self.experts = nn.ModuleList(experts)
        
        self.parallel_experts = ParallelSparseMoELayer(
            hidden_size=hidden_size, 
            intermediate_size=intermediate_size, 
            num_experts=num_experts
        )
        
        # router
        self.router = Router(hidden_size=hidden_size, num_experts=self.num_experts)
        
        self.post_norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.identity_scalar = nn.Parameter(torch.ones(1))
        
        # note: the identity expert acts both as a skip connection and a way to indicate that the current representation is sufficient and doesnt need to be modified by any expert again
        # during inference, landing on the identity expert could idicate that the model is confident in its current representation and can stop routing early. On the contrary, if the model never routes to the identity expert, 
        # it may indicate that the model is not confident or does not know how to improve further (might not have an answer at all)
        # adjust the identity skew encourages the model to end routing early, thus shortening the internal "reasoning path" and possible producing lower quality outputs.
        
    @property
    def num_experts(self):
        return self._num_experts + self._num_attn_experts + 1
    
    @property
    def identity_expert_index(self):
        return self._num_attn_experts
    
    def route(self, hidden_states: torch.Tensor, temperature: float = 1.0, identity_skew: float = 1.0, on_loop: int = 0):
        """routes tokens to experts and computes the load balancing loss

        Args:
            hidden_states (torch.Tensor): [batch_size, seq_len, hidden_size]
            temperature (float, optional): temperature for the router. Defaults to 1.0.
            identity_skew (float, optional): skew for the identity expert. Defaults to 1.0.
            on_loop (int, optional): current loop iteration. Defaults to 0.

        Returns:
            topk_scores (torch.Tensor): [batch_size, seq_len, top_k] normalized scores for the selected experts
            
            topk_indices (torch.Tensor): [batch_size, seq_len, top_k] indices of the selected experts
            
            load_balancing_loss (torch.Tensor): auxiliary loss to encourage balanced routing
        """
        # note: on_loop = 0 disables the identity bias
        
        expert_scores = self.router(hidden_states, temperature=temperature)       # [batch_size, seq_len, num_experts]
        
        # compute load balancing loss before applying
        load_balancing_loss = compute_aux_loss(torch.topk(expert_scores, self.top_k, dim=-1)[1], expert_scores, self.num_experts)
        
        # apply identity skew to encourage the router to select the identity towards the end of the loop
        id_skew = 1 + torch.exp(identity_skew * self.identity_scalar)
        id_skew = id_skew ** (on_loop / self.n_loops)
        expert_scores = expert_scores.clone() # avoid inplace modification to preserve scores for loss computation (funny error with torch.autograd.set_detect_anomaly(True) if this is removed)
        expert_scores[..., self.identity_expert_index] += id_skew - 1.0
        
        # select topk experts
        topk_scores, topk_indices = torch.topk(expert_scores, self.top_k, dim=-1) # [batch_size, seq_len, top_k], [batch_size, seq_len, top_k]
        
        # normalize the topk scores
        topk_scores = topk_scores / torch.sum(topk_scores, dim=-1, keepdim=True)
        
        return topk_scores, topk_indices, load_balancing_loss

    def forward_step(self, hidden_states: torch.Tensor, on_loop: int = 0, identity_skew: float = 1.0, attn_mask: torch.Tensor = None):
        topk_scores, topk_indices, load_balancing_loss = self.route(
            hidden_states, 
            temperature=self.temperature,
            on_loop=on_loop, 
            identity_skew=identity_skew
        )
        
        # index placement in the scores would be:
        # [attn_experts..., identity, ff_experts...]
        
        output = torch.zeros_like(hidden_states)
        
        # sparsely route tokens to experts in self.experts
        for k in range(self.top_k):
            expert_indices = topk_indices[..., k] # [batch_size, seq_len]
            expert_scores = topk_scores[..., k]   # [batch_size, seq_len]
            
            for i in range(self.identity_expert_index + 1): # loop through attention experts and identity
                mask = (expert_indices == i)      # [batch_size, seq_len]
                if mask.sum() == 0:
                    continue
                                
                if isinstance(self.experts[i], SelfAttention):
                    expert_output = self.experts[i](hidden_states, attn_mask)
                    # not sparse due to attention
                else:
                    expert_output = self.experts[i](hidden_states)
                
                # mask after attention
                output[mask] += expert_scores[mask].unsqueeze(-1) * expert_output[mask]
        
        # routed tokens to parallel experts
        mlp_mask = topk_indices > self.identity_expert_index
        mlp_indices = torch.where(mlp_mask, topk_indices - (self.identity_expert_index + 1), torch.zeros_like(topk_indices))
        mlp_scores = torch.where(mlp_mask, topk_scores, torch.zeros_like(topk_scores))


        parallel_output = self.parallel_experts(
            hidden_states, 
            mlp_scores, 
            mlp_indices
        )
        output += parallel_output
        
        output = self.post_norm(output)
        output = self.dropout(output)
        
        return output, load_balancing_loss
    

    def forward(self, hidden_states: torch.Tensor, return_loss: bool = False, attention_mask: torch.Tensor = None):
        total_load_balancing_loss = 0.0
        
        for loop in range(self.n_loops):
            hidden_states, load_balancing_loss = self.forward_step(hidden_states, on_loop=loop, attn_mask=attention_mask)
            total_load_balancing_loss += load_balancing_loss
            
        if return_loss:
            return hidden_states, total_load_balancing_loss / self.n_loops
        else:
            return hidden_states

    def set_temperature(self, temperature: float):
        self.temperature = temperature




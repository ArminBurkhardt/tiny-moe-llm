import torch
from torch import nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
# TE checkpoint required for quantized (FP8/NVFP4) layers; see transformer.py note.
from transformer_engine.pytorch import checkpoint

from modules.model.router import Router, compute_aux_loss
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm
from modules.model.experts import CrossAttention, InformationRetrievalExpert, SelfAttention
from modules.model.embeddings import RotaryPositionEmbeddingsFrequency



class ParallelSparseMoELayer(nn.Module):
    """a sparse MoE layer that dispatches each token only to its routed experts via a
    grouped GEMM (Transformer Engine ``GroupedLinear``).

    The previous implementation gathered tokens into a dense ``[num_experts, total_tokens, ...]``
    tensor and ran every expert over every token (masking afterwards), so with ``top_k`` of
    ``num_experts`` it spent ``num_experts / top_k`` more matmul FLOPs than necessary. Here we
    sort the (token, slot) assignments by expert, run one variable-sized GEMM per expert group,
    then scatter the weighted results back -- so the FF experts only do work for the tokens
    actually routed to them.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size

        # fuse gate + up into a single grouped GEMM
        # one weight per expert (Linear layout [out, in])
        self.gate_up = te.GroupedLinear(num_experts, hidden_size, 2 * intermediate_size, bias=False)
        self.down = te.GroupedLinear(num_experts, intermediate_size, hidden_size, bias=False)

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_size]
        # topk_weights, topk_indices: [batch_size, seq_len, top_k]
        # null (non-MLP) slots arrive as index 0 with weight 0 -> contribute nothing after scaling
        B, S, H = x.shape
        top_k = topk_indices.shape[-1]
        x_flat = x.reshape(-1, H)                       # [T, H]
        T = x_flat.shape[0]

        idx = topk_indices.reshape(-1)                  # [N] expert id per (token, slot), N = T*top_k
        wgt = topk_weights.reshape(-1)                  # [N] routing weight per slot
        tok = torch.arange(T, device=x.device).repeat_interleave(top_k)  # [N] owning token per slot

        # sort assignments so each experts rows are contiguous (required by GroupedLinear)
        # stable=True keeps the permutation deterministic across the checkpoint recompute pass
        order = torch.argsort(idx, stable=True)         # [N]
        idx_sorted = idx[order]
        tok_sorted = tok[order]
        wgt_sorted = wgt[order]
        x_sorted = x_flat.index_select(0, tok_sorted)   # [N, H] each slot sees its (unscaled) token

        # per expert group sizes
        m_splits = torch.bincount(idx_sorted, minlength=self.num_experts).tolist() # .tolist() is the only host sync (E small ints)

        # run the experts in BF16: NVFP4 requires each GEMMs row count (a dynamic per expert group size here) 
        # to be divisible by 16, which cant be guaranteed without padding every group.
        # => sparsity
        with te.autocast(enabled=False):
            gate_up = self.gate_up(x_sorted, m_splits)  # [N, 2*intermediate]
            gate, up = gate_up.chunk(2, dim=-1)
            act = self.activation(gate) * up            # [N, intermediate]
            out_sorted = self.down(act, m_splits)       # [N, H]

        # scale each expert output by its routing weight (null slots -> *0), then scatter back
        out_sorted = out_sorted * wgt_sorted.unsqueeze(-1)
        combined = torch.zeros(T, H, device=x.device, dtype=out_sorted.dtype)
        combined.index_add_(0, tok_sorted, out_sorted)

        return combined.view(B, S, H)


class _ExpertTracking():
    def __init__(self, id_idx: int, num_experts: int):
        self._num_experts = num_experts
        self.prob_dist = None      # lazily placed on device on first update
        self.post_skew_dist = None
        self.choices = None
        self.id_idx = id_idx
        self.sliding_window_size = 256
        # recompute guard: under activation checkpointing route() runs again during backward, which would double count every update
        self._expected_updates = None
        self._seen_updates = 0

    def begin_forward(self, expected_updates: int):
        self._expected_updates = expected_updates
        self._seen_updates = 0

    def update(self, topk_indices: torch.Tensor, topk_scores: torch.Tensor, expert_scores: torch.Tensor):
        if self._expected_updates is not None:
            if self._seen_updates >= self._expected_updates:
                return  # checkpoint recompute pass, already counted
            self._seen_updates += 1
        n = self._num_experts
        device = topk_indices.device
        if self.prob_dist is None or self.prob_dist.device != device:
            self.prob_dist = torch.zeros(n, device=device)
            self.post_skew_dist = torch.zeros(n, device=device)
            self.choices = torch.zeros(n, device=device)

        topk_indices_flat = topk_indices.detach().view(-1, topk_indices.size(-1))  # [T, k]
        topk_scores_flat = topk_scores.detach().view(-1, topk_scores.size(-1))     # [T, k]
        expert_scores_flat = expert_scores.detach().view(-1, n)                    # [T, n]
        num_tokens = topk_indices_flat.size(0)

        # compute once (not inside a per expert loop)
        one_hot_mask = F.one_hot(topk_indices_flat, num_classes=n).any(dim=1)      # [T, n] bool

        prob_updates = torch.zeros(n, device=device, dtype=topk_scores.dtype)
        prob_updates.scatter_add_(0, topk_indices_flat.reshape(-1), topk_scores_flat.reshape(-1))

        post_skew_updates = (expert_scores_flat * one_hot_mask.to(expert_scores.dtype)).sum(dim=0)
        choices_updates = one_hot_mask.sum(dim=0).float()

        # normalize to per-token quantities so the EMAs are interpretable
        decay = 1.0 - 1.0 / self.sliding_window_size
        inv_w = (1.0 / self.sliding_window_size) / max(num_tokens, 1)
        self.prob_dist = self.prob_dist * decay + prob_updates * inv_w
        self.post_skew_dist = self.post_skew_dist * decay + post_skew_updates * inv_w
        self.choices = self.choices * decay + choices_updates * inv_w

    def get_stats(self):
        return {
            "prob_dist": self.prob_dist.cpu().tolist() if self.prob_dist is not None else [0.0] * self._num_experts,
            "post_skew_dist": self.post_skew_dist.cpu().tolist() if self.post_skew_dist is not None else [0.0] * self._num_experts,
            "choices": self.choices.cpu().tolist() if self.choices is not None else [0.0] * self._num_experts,
            "id_idx": self.id_idx,
        }

    def reset_stats(self):
        if self.prob_dist is not None:
            self.prob_dist.zero_()
            self.post_skew_dist.zero_()
            self.choices.zero_()

class LoopMixtureOfExperts(nn.Module):
    """a Mixture of Experts module that routes tokens to a mixture of attention and feedforward experts in multiple loops"""
    def __init__(
        self,
        hidden_size: int, 
        intermediate_size: int, 
        num_mlp_experts: int, 
        num_attn_experts: int = 4,
        num_ir_experts: int = 0,
        top_k: int = 2,
        n_loops: int = 8, 
        dropout: float = 0.0,
        temperature: float = 1.0,
        num_ir_entries: int = 1024,
        ir_dim: int = 128,
        ir_residual: bool = False,
        max_seq_len: int = 4096,
        rope_theta: float = 100000.0,
    ):
        """Mixture of Experts module with multiple loops of routing to a mixture of attention and feedforward experts

        Args:
            hidden_size (int): hidden size of the input and output representations
            intermediate_size (int): intermediate size of the feedforward layers (MLP)
            num_mlp_experts (int): number of MLP experts in the mixture
            num_attn_experts (int, optional): number of attention experts. Defaults to 4.
            num_ir_experts (int, optional): number of intermediate representation experts. Defaults to 0.
            top_k (int, optional): number of top experts to route each token to. Defaults to 2.
            n_loops (int, optional): number of routing loops. Defaults to 8.
            dropout (float, optional): dropout probability. Defaults to 0.0.
            temperature (float, optional): temperature for the router. Defaults to 1.0.
            num_ir_entries (int, optional): number of entries in the information retrieval expert. Defaults to 1024.
            ir_dim (int, optional): dimension of the information retrieval experts latent space. Defaults to 128.
            ir_residual (bool, optional): whether the information retrieval expert should have a residual connection. Defaults to False.
        """
        super().__init__()
        self._num_mlp_experts = num_mlp_experts
        self._num_attn_experts = num_attn_experts * 2 # account for both self and cross attention experts
        self._num_ir_experts = num_ir_experts
        self.top_k = top_k
        self.n_loops = n_loops
        self.temperature = temperature
        
        # num expert heads
        n_heads = 16
        n_kv_heads = 4

        # rotary embeddings for the attention experts. the experts run with their own head count
        # (n_heads above), so their head_dim differs from the Gemma decoders
        # they need a separately sized RoPE cache :( 
        # Positions are global over the packed sequence (0..S-1),
        # matching the decoders convention so cross document packing stays consistent
        self.rotary_emb = RotaryPositionEmbeddingsFrequency(
            dim=hidden_size // n_heads,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
        )

        # experts
        experts = []
        experts.extend([
            SelfAttention(input_size=hidden_size, dropout=dropout, num_heads=n_heads, num_kv_heads=n_kv_heads)
            for _ in range(num_attn_experts)
        ])
        experts.extend([
            CrossAttention(input_size=hidden_size, dropout=dropout, num_heads=n_heads, num_kv_heads=n_kv_heads)
            for _ in range(num_attn_experts)
        ])
        experts.extend([
            InformationRetrievalExpert(
                input_size=hidden_size, 
                num_entries=num_ir_entries, 
                ir_dim=ir_dim,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                dropout=dropout,
                residual=ir_residual
            ) for _ in range(num_ir_experts)
        ])
        experts.append(nn.Identity()) # identity expert for skipping
        self.experts = nn.ModuleList(experts)
        
        self.parallel_experts = ParallelSparseMoELayer(
            hidden_size=hidden_size, 
            intermediate_size=intermediate_size, 
            num_experts=num_mlp_experts
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
        
        self.expert_tracker = _ExpertTracking(id_idx=self.identity_expert_index, num_experts=self.num_experts)
        
    @property
    def num_experts(self):
        return self._num_attn_experts + self._num_ir_experts + 1 + self._num_mlp_experts
    
    @property
    def identity_expert_index(self):
        return self._num_attn_experts + self._num_ir_experts
    
    def route(self, hidden_states: torch.Tensor, temperature: float = 1.0, identity_skew: float = 1.0, on_loop: int = 0):
        """routes tokens to experts and computes the load balancing loss

        Args:
            hidden_states (torch.Tensor): [batch_size, seq_len, hidden_size]
            temperature (float, optional): temperature for the router. Defaults to 1.0.
            identity_skew (float, optional): skew for the identity expert; higher values push harder
                towards the identity on later loops, <= 0 disables the bias. Defaults to 1.0.
            on_loop (int, optional): current loop iteration. Defaults to 0.

        Returns:
            topk_scores (torch.Tensor): [batch_size, seq_len, top_k] normalized scores for the selected experts
            
            topk_indices (torch.Tensor): [batch_size, seq_len, top_k] indices of the selected experts
            
            load_balancing_loss (torch.Tensor): auxiliary loss to encourage balanced routing
        """
        # note: on_loop = 0 disables the identity bias

        expert_logits = self.router(hidden_states, temperature=temperature)       # [batch_size, seq_len, num_experts] raw logits

        # load balancing loss on the PRE-skew distribution
        pre_skew_probs = F.softmax(expert_logits / temperature, dim=-1)
        load_balancing_loss = compute_aux_loss(torch.topk(pre_skew_probs, self.top_k, dim=-1)[1], pre_skew_probs, self.num_experts)

        # apply identity skew (additive logit bias) to encourage the router to select the identity
        # towards the end of the loop. identity_skew <= 0 disables the bias entirely
        expert_logits = expert_logits.clone() # avoid inplace modification to preserve scores for loss computation (funny error with torch.autograd.set_detect_anomaly(True) if this is removed)
        if identity_skew > 0:
            id_skew = 1 + torch.exp(-self.identity_scalar.abs() / identity_skew)
            id_skew = id_skew ** (on_loop / self.n_loops)
            expert_logits[..., self.identity_expert_index] += id_skew - 1.0

        # single softmax over the (skewed) logits gives the selection distribution
        expert_scores = F.softmax(expert_logits / temperature, dim=-1)
        
        # select topk experts
        topk_scores, topk_indices = torch.topk(expert_scores, self.top_k, dim=-1) # [batch_size, seq_len, top_k], [batch_size, seq_len, top_k]
        
        # normalize the topk scores
        topk_scores = topk_scores / torch.sum(topk_scores, dim=-1, keepdim=True)
        
        # track expert selection stats
        self.expert_tracker.update(topk_indices, topk_scores, expert_scores)
        
        return topk_scores, topk_indices, load_balancing_loss

    def forward_step(self, hidden_states: torch.Tensor, on_loop: int = 0, identity_skew: float = 1.0, cu_seqlens: torch.Tensor = None, max_seqlen: int = None, other: torch.Tensor = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] = None):
        topk_scores, topk_indices, load_balancing_loss = self.route(
            hidden_states, 
            temperature=self.temperature,
            on_loop=on_loop, 
            identity_skew=identity_skew
        )
        
        # index placement in the scores would be:
        # [attn_experts..., num_ir_experts..., identity, ff_experts...]
        
        output = torch.zeros_like(hidden_states)
        _other = other if other is not None else hidden_states

        # compute each non-MLP expert exactly once per forward_step, then cache across k slots
        # (attention runs over the full sequence regardless of routing, so recomputing per k-slot wastes compute)
        expert_cache = []
        for i in range(self.identity_expert_index + 1):
            if isinstance(self.experts[i], (SelfAttention, InformationRetrievalExpert)):
                expert_cache.append(self.experts[i](hidden_states, cu_seqlens, max_seqlen, position_embeddings))
            elif isinstance(self.experts[i], CrossAttention):
                expert_cache.append(self.experts[i](hidden_states, _other, cu_seqlens, max_seqlen, position_embeddings))
            else:  # identity
                expert_cache.append(hidden_states)

        # accumulate weighted outputs. mask multiply avoids mask.sum() device syncs. yay more tokens per second
        for k in range(self.top_k):
            expert_indices_k = topk_indices[..., k]   # [B, S]
            expert_scores_k = topk_scores[..., k]     # [B, S]
            for i in range(self.identity_expert_index + 1):
                mask = (expert_indices_k == i).unsqueeze(-1)                         # [B, S, 1]
                output = output + mask * expert_scores_k.unsqueeze(-1) * expert_cache[i]
        
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
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        other: torch.Tensor = None,
        return_loss: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        identity_skew: float = 1.0,
        use_checkpointing: bool = False,
    ):
        total_load_balancing_loss = 0.0

        self.expert_tracker.begin_forward(self.n_loops)

        # rotary cos/sin for the expert attention, computed once and reused across loops/experts
        position_embeddings = self.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])

        for loop in range(self.n_loops):
            if self.training and use_checkpointing:
                hidden_states, load_balancing_loss = checkpoint(self.forward_step, hidden_states, loop, identity_skew, cu_seqlens, max_seqlen, other, position_embeddings, use_reentrant=False)
            else:
                hidden_states, load_balancing_loss = self.forward_step(hidden_states, on_loop=loop, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, identity_skew=identity_skew, other=other, position_embeddings=position_embeddings)
            total_load_balancing_loss += load_balancing_loss
            
        if return_loss:
            return hidden_states, total_load_balancing_loss / self.n_loops
        else:
            return hidden_states

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def set_router_noise(self, noise_factor: float):
        """set the global multiplier on the router's exploration noise (annealed 1 -> 0 by the trainer over training). 0 disables the noise entirely."""
        self.router.noise_factor = noise_factor




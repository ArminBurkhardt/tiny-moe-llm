import torch
from torch import nn
from torch.nn import functional as F
import transformer_engine.pytorch as te

from modules.model.gemma4 import GemmaRMSNorm as RMSNorm


class Router(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, num_experts, bias=False),
        )
        
        self.noise_proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # global multiplier on the exploration noise, annealed 1 -> 0 over training 
        # high early noise encourages to explore experts 
        # once routing has specialized the noise only adds grad variance => decayed away
        self.noise_factor = 1.0

    def forward(self, hidden_states, temperature: float = 1.0):
        expert_scores = self.router(hidden_states)       # [batch_size, seq_len, num_experts]

        # add (annealed) noise for exploration
        if self.training and self.noise_factor > 0.0:
            noise = torch.randn_like(expert_scores)
            noise_scale = F.softplus(self.noise_proj(hidden_states))
            expert_scores = expert_scores + self.noise_factor * noise_scale * noise

        # raw logits: the single softmax happens in LoopMixtureOfExperts.route()
        return expert_scores


def compute_aux_loss(indices: torch.Tensor, router_probs: torch.Tensor, num_experts: int) -> torch.Tensor:
    """computes a load balancing auxiliary loss to prevent routing collapse"""
    num_tokens = indices.numel()
    
    # f_i: Hard fraction of tokens routed to expert i
    # flatten assignments and count frequencies using a one-hot vector
    flat_indices = indices.view(-1)
    hard_counts = torch.zeros(num_experts, device=indices.device)
    hard_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
    f_i = hard_counts / num_tokens

    # P_i: Soft average probability assigned to expert i across the batch
    P_i = router_probs.view(-1, num_experts).mean(dim=0)

    # Dot product optimization function minimizes when vectors are uniform
    aux_loss = num_experts * torch.dot(f_i.type_as(P_i), P_i)
    return aux_loss


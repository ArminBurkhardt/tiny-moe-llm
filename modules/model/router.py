import torch
from torch import nn
from torch.nn import functional as F
import transformer_engine.pytorch as te

from modules.model.gemma4 import GemmaRMSNorm as RMSNorm


# ceiling on the exploration noise, as a fraction of the router logits' own std.
#
# the learned `softplus(noise_proj(h))` scale lands around ~0.7 at init while the clean logits
# (RMSNorm -> default-init Linear over hidden_size=768) have std ~0.33 -- i.e. un-scaled, the noise
# is ~2x the signal, so early routing, the routing *weights* that scale expert outputs, and the
# load-balance loss are all measuring noise rather than the router. 0.3 keeps exploration well
# below the signal while still perturbing the argmax; it is a ceiling on the *initial* level, since
# noise_factor anneals this to 0 over noise_anneal_tokens anyway.
ROUTER_NOISE_SCALE = 0.3


class Router(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, noise_scale: float = ROUTER_NOISE_SCALE):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, num_experts, bias=False),
        )

        self.noise_proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.noise_scale = noise_scale

        # global multiplier on the exploration noise, annealed 1 -> 0 over training
        # high early noise encourages to explore experts
        # once routing has specialized the noise only adds grad variance => decayed away
        self.noise_factor = 1.0

    def forward(self, hidden_states, temperature: float = 1.0):
        expert_scores = self.router(hidden_states)       # [batch_size, seq_len, num_experts]

        # add (annealed) noise for exploration
        if self.training and self.noise_factor > 0.0:
            noise = torch.randn_like(expert_scores)
            noise_scale = F.softplus(self.noise_proj(hidden_states)) * self.noise_scale
            expert_scores = expert_scores + self.noise_factor * noise_scale * noise

        # raw logits: the single softmax happens in LoopMixtureOfExperts.route()
        return expert_scores


def compute_aux_loss(
    indices: torch.Tensor, router_probs: torch.Tensor, num_experts: int, token_mask: torch.Tensor = None
) -> torch.Tensor:
    """computes a load balancing auxiliary loss to prevent routing collapse

    Args:
        indices: [.., top_k] selected expert ids, e.g. [B, S, top_k].
        router_probs: [.., num_experts] the router's softmax, e.g. [B, S, num_experts].
        num_experts: size of the expert pool both of the above index into.
        token_mask: optional [B, S] (broadcastable to ``indices``/``router_probs`` minus their last
            axis), True/1 for a real (non pad) token. None reproduces today's numerics exactly,
            shape derived denominators and all -- every position counts, padding included, which is
            a REAL and measured distortion, not a theoretical one: a packed batch that is mostly
            padding reads a near uniform routing signal for the pad rows regardless of what the real
            tokens did, moving the aux loss purely with row fill (step 0 measured 3.06 with one
            packing bug and 1.22 with it fixed, on an otherwise identical batch). When given, both
            terms are weighted by it and their denominators become the mask's own (device tensor,
            clamped) sum instead of the tensor's raw element count, with no host sync either way.
    """
    top_k = indices.shape[-1]
    flat_indices = indices.reshape(-1)

    if token_mask is None:
        # f_i: Hard fraction of tokens routed to expert i
        # flatten assignments and count frequencies using a one-hot vector
        num_tokens = indices.numel()
        hard_counts = torch.zeros(num_experts, device=indices.device)
        hard_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
        f_i = hard_counts / num_tokens

        # P_i: Soft average probability assigned to expert i across the batch
        P_i = router_probs.reshape(-1, num_experts).mean(dim=0)
    else:
        # each token contributes top_k (token, slot) events; every slot inherits its own token's
        # mask so a padded token's top_k selections do not count toward any expert's hard fraction
        mask_flat = token_mask.reshape(-1).to(router_probs.dtype)                       # [T]
        slot_weight = mask_flat.unsqueeze(-1).expand(-1, top_k).reshape(-1)              # [T * top_k]
        hard_counts = torch.zeros(num_experts, device=indices.device, dtype=router_probs.dtype)
        hard_counts.scatter_add_(0, flat_indices, slot_weight)
        f_i = hard_counts / slot_weight.sum().clamp_min(1.0)

        P_i = (router_probs.reshape(-1, num_experts) * mask_flat.unsqueeze(-1)).sum(dim=0)
        P_i = P_i / mask_flat.sum().clamp_min(1.0)

    # Dot product optimization function minimizes when vectors are uniform
    aux_loss = num_experts * torch.dot(f_i.type_as(P_i), P_i)
    return aux_loss


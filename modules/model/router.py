import torch
import torch.nn.functional as F
from modules.model.modules import MLP, LinearAttention
from torch import nn


class LatentRouter(nn.Module):
    """Routes a latent ``z`` to one of ``num_experts`` plus a special OUTPUT expert.

    Training behavior:
      - If ``is_final`` is False, the OUTPUT expert is masked out (never chosen).
      - If ``is_final`` is True, only the OUTPUT expert is allowed.

    Inference behavior:
      - The router decides between all experts (including OUTPUT) based on logits.

    Experts can be appended during training via ``add_experts``; the OUTPUT head is
    kept as the last logit.
    """

    def __init__(
        self,
        input_size: int,
        num_experts: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size

        self.backbone = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            LinearAttention(hidden_size, hidden_size),
        )
        # +1 for the special OUTPUT expert (always the last index)
        self.head = nn.Linear(hidden_size, num_experts + 1)

    @property
    def output_index(self) -> int:
        """Index of the special OUTPUT expert in the logits/probabilities."""
        return self.num_experts

    def add_experts(self, k: int):
        """Append ``k`` new experts while preserving the OUTPUT head weights."""
        if k <= 0:
            return
        old_head = self.head
        old_num = self.num_experts
        new_num = old_num + k
        new_head = nn.Linear(self.hidden_size, new_num + 1)

        with torch.no_grad():
            # Copy old expert rows
            new_head.weight[:old_num] = old_head.weight[:old_num]
            new_head.bias[:old_num] = old_head.bias[:old_num]
            # Copy OUTPUT row to new last position
            new_head.weight[new_num] = old_head.weight[old_num]
            new_head.bias[new_num] = old_head.bias[old_num]

        self.head = new_head
        self.num_experts = new_num

    def forward(self, z: torch.Tensor, is_final: bool | None = None) -> torch.Tensor:
        """Return probabilities over experts including the OUTPUT expert.

        Args:
            z: Latent tensor of shape [..., input_size].
            is_final: Training-time flag. Required during training to control whether
                      the OUTPUT expert is allowed. Ignored during eval/inference.
        """

        h = self.backbone(z)
        logits = self.head(h)

        if self.training:
            if is_final is None:
                raise ValueError("is_final must be provided during training")
            # Masking logic keeps computation simple and GPU-friendly
            if is_final:
                # Enable only the OUTPUT expert
                mask = torch.full_like(logits, float("-inf"))
                mask[..., self.output_index] = 0.0
                logits = logits + mask
            else:
                # Disable the OUTPUT expert
                logits = logits.masked_fill(
                    torch.arange(logits.size(-1), device=logits.device) == self.output_index,
                    float("-inf"),
                )

        probs = F.softmax(logits, dim=-1)
        return probs


# Backward compatibility alias
Router = LatentRouter








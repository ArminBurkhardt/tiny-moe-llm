import torch
from torch import nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

class SmallLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, factor: int = 8):
        super().__init__()
        self.projection = te.Linear(hidden_size, hidden_size, bias=False)
        self.lm_heads = nn.ModuleList([nn.Linear(hidden_size // factor, vocab_size // factor, bias=False) for _ in range(factor)])

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        slices = torch.chunk(x, len(self.lm_heads), dim=-1)
        head_outputs = [lm_head(slice) for lm_head, slice in zip(self.lm_heads, slices)]
        return torch.cat(head_outputs, dim=-1)


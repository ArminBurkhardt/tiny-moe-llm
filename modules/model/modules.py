import torch
from torch import nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

class SmallLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, factor: int = 8):
        super().__init__()
        self.projection = te.Linear(hidden_size, hidden_size, bias=False)
        # te.Linear, not nn.Linear: these are the single largest GEMMs in the model, and an
        # nn.Linear inside te.autocast stays BF16 -- i.e. the biggest matmul would opt out of FP8
        # exactly where it matters most. (the router / MTP-head NVFP4 divisibility caveat noted in
        # scripts/pretrain.py is about NVFP4 block scaling, not FP8 DelayedScaling.)
        self.lm_heads = nn.ModuleList([te.Linear(hidden_size // factor, vocab_size // factor, bias=False) for _ in range(factor)])

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        slices = torch.chunk(x, len(self.lm_heads), dim=-1)
        head_outputs = [lm_head(slice) for lm_head, slice in zip(self.lm_heads, slices)]
        return torch.cat(head_outputs, dim=-1)


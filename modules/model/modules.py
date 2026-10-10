import math

import torch
from torch import nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from utils import logger

# TE refuses an FP8/NVFP4 GEMM whose last input dim isn't divisible by 16. Since the sub-heads
# became te.Linear they are subject to that, and hidden_size // factor is easy to push under it.
FP8_DIM_MULTIPLE = 16


def _torch_default_init(weight: torch.Tensor):
    """nn.Linear's own weight init, for te.Linear layers that replaced an nn.Linear.

    te.Linear's default init is a FIXED normal(0, 0.023) regardless of fan_in (the Megatron
    convention), while nn.Linear uses fan-in-scaled kaiming_uniform. Swapping the layer type
    without this would silently shrink the init: 1.8x for the main head's 192->16384 blocks and
    3.6x for the MTP head's 48->8192 blocks, which measurably slows early convergence. The swap
    is meant to be a kernel/precision change only, not an init change.
    """
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class SmallLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, factor: int = 8):
        super().__init__()
        self.projection = te.Linear(hidden_size, hidden_size, bias=False)
        # te.Linear, not nn.Linear: these are the single largest GEMMs in the model, and an
        # nn.Linear inside te.autocast stays BF16 -- i.e. the biggest matmul would opt out of FP8
        # exactly where it matters most. (the router / MTP-head NVFP4 divisibility caveat noted in
        # scripts/pretrain.py is about NVFP4 block scaling, not FP8 DelayedScaling.)
        # init_method keeps the pre-swap nn.Linear init -- see _torch_default_init.
        self.lm_heads = nn.ModuleList([
            te.Linear(hidden_size // factor, vocab_size // factor, bias=False,
                      init_method=_torch_default_init)
            for _ in range(factor)
        ])

        # warn rather than assert: BF16 runs these dims fine, FP8/NVFP4 does not, and whether a
        # quantized recipe is active is a runtime (USE_FP8) decision this module can't see. Failing
        # here would break BF16-only configs; saying nothing means the failure surfaces as a bare
        # "dims=[N, 8]" ValueError from deep inside TE's assert_dim_for_fp8_exec.
        in_w, out_w = hidden_size // factor, vocab_size // factor
        if in_w % FP8_DIM_MULTIPLE or out_w % FP8_DIM_MULTIPLE:
            logger.warning(
                f"SmallLMHead({hidden_size}, {vocab_size}, factor={factor}): sub-head dims "
                f"{in_w}->{out_w} are not both divisible by {FP8_DIM_MULTIPLE}; this runs in BF16 "
                f"but WILL fail under te.autocast with an FP8/NVFP4 recipe. Lower lm_head_factor "
                f"or raise hidden_size/vocab_size."
            )

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        slices = torch.chunk(x, len(self.lm_heads), dim=-1)
        head_outputs = [lm_head(slice) for lm_head, slice in zip(self.lm_heads, slices)]
        return torch.cat(head_outputs, dim=-1)


"""Left padding must be invisible to the real tokens (scripts/eval_abstention.py's decode path).

Batched generation there left-pads every prompt to a common width and gives the pad run its own
``document_ids`` segment, so flash's block-diagonal causal mask keeps real tokens from ever reading
a pad -- the same mechanism that keeps packed documents apart during training, just used for a
different purpose. ``tests/test_attention_equiv.py`` already proves the mask is correct at the
kernel level; this proves the whole model is wired to it, including the MoE's attention experts and
``shared_attn``, which take their own ``cu_seqlens`` argument and would be easy to regress.

The assertion is exact and it is on the **dense decoder**: hold the real tokens fixed, change only
what sits in the pad region, and the decoder's output for the real tokens must be bit-identical. An
unsegmented control runs the same comparison with one segment covering pads and real tokens, where
the outputs *must* differ -- without it a broken forward that ignored its input entirely would pass.

Deliberately NOT asserted on the full model or on a decoded string. ``ParallelSparseMoELayer`` tiles
its grouped GEMM by ``m_splits``, the per-expert row counts over every token in the batch (pads
included), so a different batch composition changes the bf16 accumulation order for the real tokens'
rows too -- ~0.5-1% of hidden-state magnitude, deterministic (the same input twice is bit-identical)
but real. That is reduction order, not information flow, and equality on greedy output would flap.
The consequence for the eval script is documented there: compare runs only at a fixed --batch-size.

GPU required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from config import ModelConfig
from utils import BF16

torch.manual_seed(0)
dev = "cuda"
REAL_N, PAD_N, PAD_ID = 48, 96, 1

model = TinyMoETransformer(**ModelConfig.Params).to(dev).to(BF16)
model.set_checkpointing(False, False)
model.delayed_mtp_loss(True)
model.eval()
# router exploration noise is gated on self.training; eval() must have switched it off, or every
# comparison below would be measuring torch.randn_like rather than the attention mask
assert not model.moe.router.training

real = torch.randint(100, 60000, (1, REAL_N), device=dev)


@torch.inference_mode()
def decoder_out(fill: torch.Tensor, segmented: bool) -> torch.Tensor:
    """Dense-decoder states for the real tokens, with ``fill`` prepended as left padding.

    The decoder is used rather than the full model because it is pure attention + dense MLP: a
    token's output there is a mathematical function of its own segment alone, so any difference is
    a mask failure rather than a rounding artefact.
    """
    ids = torch.cat([fill, real], dim=1)
    doc = (torch.cat([torch.zeros_like(fill), torch.ones_like(real)], dim=1)
           if segmented else torch.ones_like(ids))
    cu, max_seqlen = cu_seqlens_from_doc_ids(doc)
    hidden = model.gemma_decoder(ids, cu, max_seqlen).last_hidden_state
    return hidden[0, -REAL_N:, :]


pads = torch.full((1, PAD_N), PAD_ID, dtype=torch.long, device=dev)
junk_a = torch.randint(100, 60000, (1, PAD_N), device=dev)
junk_b = torch.randint(100, 60000, (1, PAD_N), device=dev)

base = decoder_out(pads, segmented=True)

repeat = decoder_out(pads, segmented=True)
print(f"same input twice:            max abs diff = {(base - repeat).abs().max().item()} (must be 0)")
assert torch.equal(base, repeat), "the forward is not deterministic -- nothing below is meaningful"

for name, fill in (("pad -> random ids", junk_a), ("random ids -> other ids", junk_b)):
    other = decoder_out(fill, segmented=True)
    diff = (base - other).abs().max().item()
    print(f"{name:<28} max abs diff = {diff} (must be 0)")
    assert torch.equal(base, other), (
        f"changing the pad region ({name}) changed the real tokens' decoder states -- a real token "
        "is attending across its segment boundary into the padding"
    )

# control: with one segment spanning pads + real tokens the pads ARE visible, so the checks above
# are actually constraining something
unseg_pad = decoder_out(pads, segmented=False)
unseg_junk = decoder_out(junk_a, segmented=False)
control = (unseg_pad - unseg_junk).abs().max().item()
print(f"UNSEGMENTED control:         max abs diff = {control:.4f} (must be > 0)")
assert not torch.equal(unseg_pad, unseg_junk), "control failed: the checks above are vacuous"

print("\nPAD ISOLATION CHECKS PASSED")

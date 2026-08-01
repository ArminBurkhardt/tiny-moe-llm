"""Regression test for PLAN.md Step 3b's ponder-warmup correctness requirement.

The claim (see the comment on LoopMixtureOfExperts.loop_scale in moe.py): at loop_scale == 0,
CE loss has EXACTLY zero gradient wrt p_halt (the update `(1 - p_halt) * loop_scale * delta` is
multiplied by loop_scale, so its derivative wrt p_halt is `-loop_scale * delta`, an exact zero
when loop_scale == 0 regardless of delta). An un-warmed ponder loss would then be p_halt's ONLY
gradient source, constant-sign, and AdamW would climb the halt bias regardless of how small
lambda_ponder is -- deadlocking the halt head before loop_scale (starting at the safe 0.1 init,
not 0) has any chance to grow. This is tested directly and deterministically via autograd rather
than via a noisy multi-step training simulation. GPU required (TE layers).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.mtp import compute_mtp_loss
from config import TrainingConfig

torch.manual_seed(0)
dev = "cuda"
P = dict(vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
         head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
         top_k=2, n_loops=2, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
         dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2)

B, S = 2, 32
input_ids = torch.randint(2, P["vocab_size"], (B, S), device=dev)
pad_mask = torch.zeros(B, S, dtype=torch.bool, device=dev)
labels = input_ids.clone()
labels[:, 0] = -100


def ce_loss_and_p_halt(model):
    hidden, aux, p_halt, mtp = model(input_ids=input_ids, return_aux_loss=True, return_hidden=True)
    loss, _ = compute_mtp_loss(hidden, labels, mtp_outputs=mtp, lm_head=model.mtp_head.lm_head,
                            lambda_mtp=0.1, main_lm_head=model.lm_head, pad_mask=pad_mask)
    return loss, p_halt


# --- 1. at loop_scale == 0, CE gives exactly zero gradient into the halt head ---
model = TinyMoETransformer(**P).to(dev).to(torch.bfloat16).train()
with torch.no_grad():
    model.moe.loop_scale.zero_()
loss, p_halt = ce_loss_and_p_halt(model)
loss.backward()
g = model.moe.halt_proj.weight.grad
assert g is not None and torch.count_nonzero(g).item() == 0, \
    f"expected exactly-zero halt_proj gradient from CE at loop_scale=0, got nonzero entries: {g}"
print("[ok] loop_scale=0: CE loss gives exactly zero gradient into halt_proj (the deadlock precondition)")

# --- 2. with loop_scale == 0, the ponder loss is the ONLY thing that can move the halt head ---
model.zero_grad(set_to_none=True)
loss, p_halt = ce_loss_and_p_halt(model)
valid_mask = (~pad_mask).to(p_halt.dtype)
ponder = ((1.0 - p_halt) * valid_mask).sum() / valid_mask.sum().clamp(min=1)
(loss + 0.1 * ponder).backward()
g = model.moe.halt_proj.weight.grad
assert g is not None and torch.count_nonzero(g).item() > 0, \
    "expected the ponder term to give halt_proj a nonzero gradient even at loop_scale=0"
print("[ok] loop_scale=0: adding the (un-warmed) ponder term gives halt_proj its only gradient -- "
      "constant-sign, so AdamW would climb the halt bias regardless of lambda_ponder's magnitude")

# --- 3. away from the pathological loop_scale=0 case (the real 0.1 init), CE already gives the
#         halt head a real gradient -- warmup is defense-in-depth, not the only thing keeping it alive
model2 = TinyMoETransformer(**P).to(dev).to(torch.bfloat16).train()
loss, p_halt = ce_loss_and_p_halt(model2)
loss.backward()
g = model2.moe.halt_proj.weight.grad
assert g is not None and torch.count_nonzero(g).item() > 0, \
    "expected a nonzero CE gradient into halt_proj at the real loop_scale=0.1 init"
print("[ok] loop_scale=0.1 (real init): CE loss already gives halt_proj a nonzero gradient")

# --- 4. the warmup ramp formula itself (as used in scripts/pretrain.py's train_step) ---
def lambda_ponder_now(tokens, warm, ramp, target):
    return target * min(1.0, max(0.0, (tokens - warm) / ramp))

warm, ramp = TrainingConfig.ponder_warmup_tokens, TrainingConfig.ponder_ramp_tokens
target = TrainingConfig.lambda_ponder
assert lambda_ponder_now(0, warm, ramp, target) == 0.0
assert lambda_ponder_now(warm, warm, ramp, target) == 0.0
assert abs(lambda_ponder_now(warm + ramp / 2, warm, ramp, target) - target / 2) < 1e-12
assert lambda_ponder_now(warm + ramp, warm, ramp, target) == target
assert lambda_ponder_now(warm + 10 * ramp, warm, ramp, target) == target  # clamped, doesn't overshoot
print(f"[ok] ponder ramp formula: 0 before warmup ({warm:,} tok), "
      f"-> target {target} over the next {ramp:,} tok, clamped after")

print("\nPONDER DEADLOCK CHECK PASSED")

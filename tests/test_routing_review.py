"""Verifies the routing behavior of LoopMixtureOfExperts.route() after the identity expert was
removed (PLAN.md Step 3c):
- router returns logits, exactly one softmax decides selection
- num_experts == 2A + I + M (no identity slot)
- load balancing loss has its proper scale (>= ~1.0, computed from real probabilities)
Uses the production config sizes (39 experts, top_k=2, n_loops=4).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.moe import LoopMixtureOfExperts

torch.manual_seed(0)
dev = "cuda"

moe = LoopMixtureOfExperts(
    hidden_size=512, intermediate_size=2048,
    num_mlp_experts=36, num_attn_experts=1, num_ir_experts=1,
    top_k=2, n_loops=4, dropout=0.0,
    num_ir_entries=1024, ir_dim=128, max_seq_len=4096,
).to(dev).to(torch.bfloat16).eval()  # eval: no router noise, deterministic

A = moe._num_attn_experts   # self + cross
I = moe._num_ir_experts
M = moe._num_mlp_experts
assert moe.num_experts == A + I + M, (moe.num_experts, A, I, M)
assert moe.first_mlp_index == A + I, (moe.first_mlp_index, A, I)
print(f"num_experts={moe.num_experts} (A={A} I={I} M={M}, no identity slot) first_mlp_index={moe.first_mlp_index}")

x = torch.randn(2, 256, 512, device=dev, dtype=torch.bfloat16)

with torch.no_grad():
    logits = moe.router(x)
s = logits.float().sum(-1)
assert (s - 1.0).abs().mean() > 0.1, "router output looks softmaxed; it must be raw logits"
print(f"router returns raw logits (range {logits.min().item():.3f}..{logits.max().item():.3f})")

with torch.no_grad():
    tk_scores, tk_idx, aux = moe.route(x)
assert tk_idx.max().item() < moe.num_experts, "topk index out of range"
assert torch.allclose(tk_scores.sum(-1), torch.ones_like(tk_scores.sum(-1)), atol=1e-2), \
    "topk scores must renormalize to sum to 1"
print(f"topk indices in [0, {moe.num_experts}), scores renormalize to 1")

# aux loss scale: probabilities-based, minimum 1.0 at perfect balance
assert 0.9 < float(aux) < 5.0, f"aux loss out of expected scale: {float(aux)}"
print(f"aux loss = {float(aux):.3f} (proper scale, min 1.0 at perfect balance)")

print("\nROUTING CHECKS PASSED")

"""Verifies the fixed routing behavior of LoopMixtureOfExperts.route():
- router returns logits, exactly one softmax decides selection
- identity skew is a mild, loop-increasing logit bias (no more 100% identity hijack)
- identity_skew <= 0 fully disables the bias
- load balancing loss has its proper scale (>= ~1.0, computed from real probabilities)
Uses the production config sizes (40 experts, top_k=2, n_loops=4, identity_skew=0.4).
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

print(f"num_experts={moe.num_experts} identity_idx={moe.identity_expert_index}")

x = torch.randn(2, 256, 512, device=dev, dtype=torch.bfloat16)

with torch.no_grad():
    logits = moe.router(x)
s = logits.float().sum(-1)
assert (s - 1.0).abs().mean() > 0.1, "router output looks softmaxed; it must be raw logits"
print(f"router returns raw logits (range {logits.min().item():.3f}..{logits.max().item():.3f})")

def frac_identity(skew, loop):
    with torch.no_grad():
        tk_scores, tk_idx, aux = moe.route(x, identity_skew=skew, on_loop=loop)
    return (tk_idx == moe.identity_expert_index).any(dim=-1).float().mean().item(), aux

# skew disabled -> identical selection on every loop
f0, aux0 = frac_identity(0.0, 0)
f3, _ = frac_identity(0.0, 3)
assert abs(f0 - f3) < 1e-6, (f0, f3)
print(f"identity_skew=0: identity selection {f0*100:.1f}% on loop 0 and loop 3 (bias fully disabled)")

# production skew: bias grows with the loop but must not hijack routing
print("\n--- identity_skew=0.4 (config) ---")
fracs = []
for loop in range(moe.n_loops):
    f, aux = frac_identity(0.4, loop)
    fracs.append(f)
    with torch.no_grad():
        id_skew = 1 + torch.exp(-moe.identity_scalar.abs() / 0.4)
        bias = (id_skew ** (loop / moe.n_loops) - 1.0).item()
    print(f" loop {loop}: logit bias = {bias:.4f} | identity in top{moe.top_k}: {f*100:5.1f}% | aux loss = {float(aux):.3f}")

assert fracs[0] == f0, "loop 0 must be bias-free"
assert fracs[-1] >= fracs[0], "bias should not decrease identity selection"
assert fracs[-1] < 0.5, f"identity must not dominate routing anymore, got {fracs[-1]*100:.0f}% on the last loop"

# aux loss scale: probabilities-based, minimum 1.0 at perfect balance
_, aux = frac_identity(0.4, 0)
assert 0.9 < float(aux) < 5.0, f"aux loss out of expected scale: {float(aux)}"
print(f"\naux loss = {float(aux):.3f} (proper scale, min 1.0 at perfect balance)")

print("\nROUTING CHECKS PASSED")

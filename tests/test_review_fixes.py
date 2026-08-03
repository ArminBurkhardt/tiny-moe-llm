"""Regression checks for the code-review fixes.

Covers, in one pass: the p_halt/ponder loop-axis normalization, the weight-decay param grouping,
per-loop loop_scale, the sinusoidal loop-index router bias (including running a trained model at a
loop count it was not trained at), the cheap p_max identity, per-loop CE token subsampling being
unbiased, the split FLOP components, and the te.Linear LM heads.

NOT YET RUN -- written alongside the fixes but never executed. GPU + TE required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch

from modules.model.transformer import TinyMoETransformer
from modules.model.mtp import compute_mtp_loss
from modules.model.attention import cu_seqlens_from_doc_ids

torch.manual_seed(0)
dev = "cuda"
N_LOOPS = 3
P = dict(vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
         head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
         top_k=2, n_loops=N_LOOPS, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
         dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2, lm_head_factor=4)

B, S = 2, 64
model = TinyMoETransformer(**P).to(dev).to(torch.bfloat16).train()
input_ids = torch.randint(2, P["vocab_size"], (B, S), device=dev)
labels = input_ids.clone(); labels[:, 0] = -100
pad_mask = torch.zeros(B, S, dtype=torch.bool, device=dev)
doc_ids = torch.tensor([[0] * (S // 2) + [1] * (S // 2)] * B, device=dev)
cu, ms = cu_seqlens_from_doc_ids(doc_ids)

# --- per-loop loop_scale, init 1/sqrt(n_loops) ---
ls = model.moe.loop_scale
assert ls.shape == (N_LOOPS,), ls.shape
assert all(abs(v - 1 / math.sqrt(N_LOOPS)) < 1e-6 for v in ls.tolist()), ls.tolist()
print(f"[ok] loop_scale shape {tuple(ls.shape)}, init {ls.tolist()[0]:.4f} == 1/sqrt({N_LOOPS})")

# --- loop-index router bias: zero-init no-op, distinct per loop, clamps past the table ---
assert torch.count_nonzero(model.moe.loop_router_bias.weight).item() == 0
b0, b1 = model.moe.loop_bias(0), model.moe.loop_bias(1)
assert torch.equal(b0, b1) and torch.count_nonzero(b0).item() == 0, "zero-init bias must be a no-op"
with torch.no_grad():
    model.moe.loop_router_bias.weight.normal_(0, 0.5)
b0, b1, b2 = model.moe.loop_bias(0), model.moe.loop_bias(1), model.moe.loop_bias(2)
assert not torch.allclose(b0, b1) and not torch.allclose(b1, b2), "loops must get distinct biases"
assert torch.allclose(model.moe.loop_bias(10_000), model.moe.loop_bias(model.moe.loop_enc.size(0) - 1))
with torch.no_grad():
    model.moe.loop_router_bias.weight.zero_()
print("[ok] loop router bias: zero-init no-op, distinct per loop, clamps past the encoding table")

# --- the whole point of the sinusoidal encoding: loop count is a runtime choice ---
model.eval()
with torch.no_grad():
    for n in (1, 2, 3, 5, 8):
        h, aux, ph, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=ms,
                                return_aux_loss=True, return_hidden=True, n_loops=n)
        assert h.shape == (n, B, S, P["hidden_size"]), (n, h.shape)
        assert ph.shape == (n, B, S), (n, ph.shape)
print("[ok] n_loops override runs at 1/2/3/5/8 loops with correct shapes")
model.train()

# --- p_halt / ponder normalization: the old form was exactly n_loops x too large ---
h, aux, p_halt, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=ms,
                            return_aux_loss=True, return_hidden=True)
valid = (~pad_mask).to(p_halt.dtype)
old = ((p_halt * valid).sum() / valid.sum().clamp(min=1)).item()
new = ((p_halt * valid).sum() / (valid.sum().clamp(min=1) * p_halt.size(0))).item()
true_mean = p_halt.float().mean().item()
assert abs(new - true_mean) < 2e-2, (new, true_mean)
assert abs(old / max(new, 1e-9) - N_LOOPS) < 0.05, (old, new)
print(f"[ok] p_halt mean: fixed={new:.4f} matches true {true_mean:.4f}; old form read {old:.4f} ({old/new:.2f}x)")

# --- p_max via 1/sum(exp(l - l_max)) == fp32 softmax max, without the fp32 copy ---
hid = torch.randn(300, P["hidden_size"], device=dev, dtype=torch.bfloat16)
lg = model.lm_head(hid)
mx = lg.max(-1).values
cheap = 1.0 / (lg - mx.unsqueeze(-1)).exp().sum(-1, dtype=torch.float32)
ref = lg.float().softmax(-1).max(-1).values
assert torch.allclose(cheap, ref, atol=1e-5, rtol=1e-4), (cheap[:4], ref[:4])
print(f"[ok] p_max: max abs diff vs fp32 softmax = {(cheap - ref).abs().max().item():.2e}")

# --- per-loop CE subsampling is an unbiased estimate of the full-token CE ---
kw = dict(mtp_outputs=mtp, lm_head=model.mtp_head.lm_head, main_lm_head=model.lm_head,
          pad_mask=pad_mask, loop_ce_weights=[0.2, 0.3, 1.0], correct_proj=model.correct_proj,
          lambda_conf=0.05, return_metrics=True)
full = compute_mtp_loss(h, labels, loop_ce_subsample=1.0, **kw)
subs = [compute_mtp_loss(h, labels, loop_ce_subsample=0.25, **kw) for _ in range(12)]
for li in range(N_LOOPS - 1):
    ref_ce = full[2]["per_loop_ce"][li].item()
    est = sum(s[2]["per_loop_ce"][li].item() for s in subs) / len(subs)
    assert abs(est - ref_ce) / ref_ce < 0.05, (li, est, ref_ce)
    print(f"[ok] loop {li} CE: full={ref_ce:.4f}, mean of 12 subsampled={est:.4f}")
assert abs(full[2]["per_loop_ce"][-1].item() - subs[0][2]["per_loop_ce"][-1].item()) < 1e-4, \
    "final loop must never be subsampled"
print("[ok] final loop CE unaffected by subsampling")

model.zero_grad(set_to_none=True)
subs[0][0].backward(retain_graph=True)
assert model.lm_head.projection.weight.grad is not None
assert torch.count_nonzero(model.moe.loop_scale.grad).item() == N_LOOPS, model.moe.loop_scale.grad
print(f"[ok] backward through subsampled CE; per-loop loop_scale grads {model.moe.loop_scale.grad.tolist()}")

# --- weight decay param grouping ---
from scripts.pretrain import build_param_groups
groups = build_param_groups(model, 0.02)
assert groups[0]["weight_decay"] == 0.02 and groups[1]["weight_decay"] == 0.0
no_decay = {id(p) for p in groups[1]["params"]}
for name, p in [("moe.loop_scale", model.moe.loop_scale),
                ("layer_scalar", model.gemma_decoder.layers[0].layer_scalar),
                ("halt_proj.bias", model.moe.halt_proj.bias)]:
    assert id(p) in no_decay, f"{name} must not be weight-decayed"
assert id(model.gemma_decoder.embed_tokens.weight) in {id(p) for p in groups[0]["params"]}
assert sum(len(g["params"]) for g in groups) == len([p for p in model.parameters() if p.requires_grad])
print(f"[ok] param groups: {len(groups[0]['params'])} decayed / {len(groups[1]['params'])} undecayed, all covered")

# --- router exploration noise is scaled down ---
from modules.model.router import ROUTER_NOISE_SCALE
assert model.moe.router.noise_scale == ROUTER_NOISE_SCALE == 0.3
print(f"[ok] router noise_scale = {model.moe.router.noise_scale}")

# --- FLOP components ---
expected_attn_layers = P["num_layers"] + N_LOOPS * (1 + 2 * P["num_attn_experts"] + P["num_ir_experts"])
assert model.attn_flops_per_seqsq == 2 * P["hidden_size"] * expected_attn_layers
assert model.lm_head_flops_per_token == 2 * sum(p.numel() for p in model.lm_head.parameters())
assert model.flops_per_token_fwd > model.body_flops_per_token
print(f"[ok] FLOPs: body={model.body_flops_per_token/1e6:.1f}M lm_head/pass="
      f"{model.lm_head_flops_per_token/1e6:.1f}M mtp={model.mtp_flops_per_token/1e6:.1f}M "
      f"attn_coeff={model.attn_flops_per_seqsq} ({expected_attn_layers} attn layers)")

# --- LM head sub-heads are TE layers (so they participate in FP8) ---
import transformer_engine.pytorch as te
assert all(isinstance(head, te.Linear) for head in model.lm_head.lm_heads)
assert all(isinstance(head, te.Linear) for head in model.mtp_head.lm_head.lm_heads)
print("[ok] SmallLMHead sub-heads are te.Linear")

print("\nREVIEW FIX CHECKS PASSED")

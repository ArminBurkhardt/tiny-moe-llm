"""Regression checks for the loop machinery, in one GPU pass.

Covers: per-loop ``loop_scale``, the sinusoidal loop-index router bias (including running a model
at a loop count it was not trained at), the parameter-free convergence exit, the cheap ``p_max``
identity, per-loop CE token subsampling being unbiased, stochastic loop depth, the weight-decay
param grouping and its fp32-master mechanism, the split FLOP components, and the te.Linear LM heads.

GPU + TE required.
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

# --- expert pool shape and routing: exactly one softmax decides selection, no identity slot ---
moe = model.moe
A, I, M = moe._num_attn_experts, moe._num_ir_experts, moe._num_mlp_experts
assert moe.num_experts == A + I + M, (moe.num_experts, A, I, M)
assert moe.first_mlp_index == A + I, (moe.first_mlp_index, A, I)
model.eval()   # eval: no router noise, deterministic
with torch.no_grad():
    x = torch.randn(B, S, P["hidden_size"], device=dev, dtype=torch.bfloat16)
    router_logits = moe.router(x)
    assert (router_logits.float().sum(-1) - 1.0).abs().mean() > 0.1, \
        "router output looks softmaxed; it must be raw logits"
    tk_scores, tk_idx, aux_only = moe.route(x)
assert tk_idx.max().item() < moe.num_experts, "topk index out of range"
assert torch.allclose(tk_scores.sum(-1), torch.ones_like(tk_scores.sum(-1)), atol=1e-2), \
    "topk scores must renormalize to sum to 1"
# aux loss is probability-based, so its floor is 1.0 at perfect balance, not 0
assert 0.9 < float(aux_only) < 5.0, f"aux loss out of expected scale: {float(aux_only)}"
model.train()
print(f"[ok] routing: num_experts={moe.num_experts} (A={A} I={I} M={M}), first_mlp_index="
      f"{moe.first_mlp_index}, raw logits, scores renormalize, aux={float(aux_only):.3f}")

# --- per-loop loop_scale, init 1/sqrt(n_loops) ---
ls = model.moe.loop_scale
assert ls.shape == (N_LOOPS,), ls.shape
# tolerance is bf16-sized, not fp32: the model is cast to bf16, so 1/sqrt(3)=0.57735 stores as 0.578125
assert all(abs(v - 1 / math.sqrt(N_LOOPS)) < 1e-2 for v in ls.tolist()), ls.tolist()
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
        h, aux, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=ms,
                            return_aux_loss=True, return_hidden=True, n_loops=n)
        assert h.shape == (n, B, S, P["hidden_size"]), (n, h.shape)
print("[ok] n_loops override runs at 1/2/3/5/8 loops with correct shapes")

# --- convergence exit: the parameter-free depth policy that replaced the halt head ---
with torch.no_grad():
    # tol=0 can never be satisfied (|dlogp| < 0 is false), so the full depth always runs
    h_full, _, _ = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=ms,
                         return_aux_loss=True, return_hidden=True, converge_tol=0.0)
    assert h_full.shape[0] == N_LOOPS, h_full.shape
    # a huge tol reduces it to "top-1 unchanged", and min_loops is a hard floor either way
    h_min, _, _ = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=ms,
                        return_aux_loss=True, return_hidden=True,
                        converge_tol=1e9, min_loops=2)
    assert 2 <= h_min.shape[0] <= N_LOOPS, h_min.shape
    # the un-exited prefix must be bit-identical to the full run: exiting changes nothing about
    # the loops that did run, it only stops adding more
    assert torch.equal(h_min[0], h_full[0]), "early exit perturbed loop 0"
print(f"[ok] convergence exit: tol=0 -> {h_full.shape[0]} loops, tol=inf/min_loops=2 -> "
      f"{h_min.shape[0]} loops, prefix bit-identical")

# an exit_check must never be combined with training or with a KV cache
try:
    model.train()
    model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=ms, return_hidden=True, converge_tol=0.1)
    raise SystemExit("converge_tol must be rejected in training mode")
except AssertionError:
    pass
finally:
    model.eval()
from modules.model.kv_cache import KVCache
try:
    with torch.no_grad():
        model(input_ids=input_ids[:1, :4], return_hidden=True, converge_tol=0.1,
              kv_cache=KVCache.for_model(model))
    raise SystemExit("converge_tol + kv_cache must be rejected")
except AssertionError:
    pass
print("[ok] convergence exit refuses training mode and the KV cache")
model.train()

h, aux, mtp = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=ms,
                    return_aux_loss=True, return_hidden=True)

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
          pad_mask=pad_mask, loop_ce_weights=[0.2, 0.3, 1.0], return_metrics=True)
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

# --- stochastic loop depth ---
import random
from scripts.pretrain import sample_n_loops, loop_ce_weights_for

rng = random.Random(0)
assert all(sample_n_loops(rng, N_LOOPS, 0.0) == N_LOOPS for _ in range(50)), "p=0 must never reduce"
assert all(sample_n_loops(rng, 1, 1.0) == 1 for _ in range(20)), "n_loops=1 has nothing to reduce to"
draws = [sample_n_loops(rng, N_LOOPS, 0.3) for _ in range(4000)]
assert set(draws) == set(range(1, N_LOOPS + 1)), sorted(set(draws))
frac_full = draws.count(N_LOOPS) / len(draws)
assert 0.65 < frac_full < 0.75, frac_full
print(f"[ok] sample_n_loops: depths {sorted(set(draws))}, full-depth fraction {frac_full:.3f} (~0.70)")

# truncated weights must keep the deepest loop actually run at weight 1.0
import config as _cfg
_cfg.TrainingConfig.loop_ce_weights = [0.2, 0.3, 1.0]
assert loop_ce_weights_for(3) == [0.2, 0.3, 1.0]
assert loop_ce_weights_for(2) == [0.2 / 0.3, 1.0]
assert loop_ce_weights_for(1) == [1.0]
print(f"[ok] loop_ce_weights_for: 3->{loop_ce_weights_for(3)} 2->{[round(w,3) for w in loop_ce_weights_for(2)]} 1->{loop_ce_weights_for(1)}")

# a reduced-depth step must produce a finite loss and real gradients end to end
for n in (1, 2, 3):
    model.zero_grad(set_to_none=True)
    h_n, aux_n, mtp_n = model(input_ids=input_ids, cu_seqlens=cu, max_seqlen=ms,
                              return_aux_loss=True, return_hidden=True, n_loops=n)
    loss_n, _ = compute_mtp_loss(h_n, labels, mtp_outputs=mtp_n, lm_head=model.mtp_head.lm_head,
                                 main_lm_head=model.lm_head, pad_mask=pad_mask,
                                 loop_ce_weights=loop_ce_weights_for(n),
                                 loop_ce_subsample=0.25)
    (loss_n + aux_n).backward()
    assert torch.isfinite(loss_n), (n, loss_n)
    g = model.moe.loop_scale.grad
    assert torch.count_nonzero(g[:n]).item() == n, (n, g)
    assert torch.count_nonzero(g[n:]).item() == 0, f"loops beyond {n} must get no gradient: {g}"
    print(f"[ok] depth {n}: loss={loss_n.item():.4f}, loop_scale grad touches exactly loops 0..{n-1}")

# --- weight decay param grouping ---
from scripts.pretrain import build_param_groups, sync_master_grads_, sync_master_values_
groups, no_decay_master_pairs = build_param_groups(model, 0.02)
assert groups[0]["weight_decay"] == 0.02 and groups[1]["weight_decay"] == 0.0
# the no_decay group's optimizer entries are fp32 masters, not the bf16 model params directly
# (see build_param_groups' docstring) -- membership is checked via the pairing, not group identity
no_decay_ids = {id(bf16_p) for bf16_p, _ in no_decay_master_pairs}
for name, p in [("moe.loop_scale", model.moe.loop_scale),
                ("layer_scalar", model.gemma_decoder.layers[0].layer_scalar)]:
    assert id(p) in no_decay_ids, f"{name} must not be weight-decayed"
assert id(model.gemma_decoder.embed_tokens.weight) in {id(p) for p in groups[0]["params"]}
assert sum(len(g["params"]) for g in groups) == len([p for p in model.parameters() if p.requires_grad])
for bf16_p, master in no_decay_master_pairs:
    assert master.dtype == torch.float32 and bf16_p.dtype == torch.bfloat16
    assert master.data_ptr() != bf16_p.data_ptr(), "master must be a separate tensor, not an alias"
    assert torch.allclose(master.float(), bf16_p.float(), atol=1e-3), "master must start at the bf16 value"
print(f"[ok] param groups: {len(groups[0]['params'])} decayed / {len(groups[1]['params'])} undecayed, all covered")

# --- bf16-native AdamW on a no_decay tensor silently discards a steady-state-sized step; the
# fp32-master mechanism must not (this is the bug that left loop_scale pinned at its init) ---
ls = model.moe.loop_scale
ls_master = dict(no_decay_master_pairs)[ls]
before = ls.detach().clone()
lr_like_step = 2e-4  # this run's peak lr; ls sits at 1/sqrt(3)=0.578, bf16 ulp there is 0.0039
naive = before.clone()
naive -= lr_like_step
assert torch.equal(naive, before), "sanity check: a naive bf16 in-place step should round away to nothing"
opt = torch.optim.AdamW([ls_master], lr=lr_like_step)
for _ in range(50):
    ls.grad = torch.full_like(ls, -1.0)  # consistent-sign synthetic gradient, like a real steady state
    sync_master_grads_(no_decay_master_pairs)
    opt.step()
    sync_master_values_(no_decay_master_pairs)
    ls.grad = None
assert not torch.equal(ls, before), (
    f"loop_scale must move via the fp32 master after 50 steps at lr={lr_like_step}, "
    f"still exactly {ls.tolist()} == init -- the bf16-freeze bug is back"
)
print(f"[ok] fp32-master step: loop_scale moved {before.tolist()} -> {ls.tolist()} (naive bf16 step would not have)")

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

print("\nLOOP MACHINERY CHECKS PASSED")

# Looped Mixture-of-Experts

`LoopMixtureOfExperts` ([modules/model/moe.py](../modules/model/moe.py)) routes each token to a
mixture of heterogeneous experts, and repeats this `n_loops` times over a shared pool. A recurrent
(LoopLM-style) refinement of the representation rather than a single MoE pass

## Expert pool

The router indexes a single flat expert list, ordered:

```
[ self-attn × A | cross-attn × A | IR × I | MLP × M ]
                                    ▲
                              first_mlp_index
```

With the default config: `A=1`, `I=1`, `M=36` -> **39 experts** (`num_attn_experts` counts self *and*
cross, so it contributes `2A`). Types:

- **Self-attention** ([experts.py](../modules/model/experts.py)) - GQA over the sequence, its own
  head count (16 heads / 4 KV heads) and RoPE cache
- **Cross-attention** - same, but keys/values come from the `other` stream (the projected MoE
  per-layer embedding), letting tokens attend to a side channel
- **Information-retrieval (IR)** ([information_retrieval.py](../modules/model/information_retrieval.py)) -
  down-projects the token, does a cosine-similarity lookup over a learned key/value table
  (`num_ir_entries` x `ir_dim`), up-projects, and feeds the result as cross-attention values. A
  differentiable key-value memory realized with cross attention
- **MLP** - SwiGLU FFNs, run sparsely as one grouped GEMM (see Sparse MLP dispatch).

There is no identity expert (removed, see Halt head below); a plain always-on `shared_mlp` +
`shared_attn` pair seeds every loop's output unconditionally instead (outside the router pool
entirely — see the "Shared experts" note in [CLAUDE.md](../CLAUDE.md)).

Attention/IR experts run over the **full sequence regardless of routing**, so each is computed
**once per loop** and cached across the `top_k` slots (recomputing per slot would waste compute).
Only the MLP experts are dispatched sparsely (as thats more "easily" (not really) implementable)

## Routing

`route()` per loop:

1. Router ([router.py](../modules/model/router.py)) = `RMSNorm -> Linear` produces logits. During
   training it adds **annealed exploration noise** `noise_factor * softplus(noise_proj(x)) * ε`.
2. Single softmax over the logits gives the selection distribution.
3. **Load-balancing aux loss** is computed directly from that softmax: `num_experts * Σ f_i * P_i`
   (hard token fraction `f_i` * mean soft prob `P_i`), minimized at a uniform distribution. This
   prevents routing collapse. Returned and weighted by `aux_loss_weight` in the trainer.
4. `top_k` selection -> renormalize the selected weights to sum to 1.

Experts other than MLP are applied by masked accumulation; MLP slots are remapped to expert-local
indices and dispatched to the sparse layer. All expert outputs (plus the always-on shared experts)
are summed, then `RMSNorm` + dropout. The mean aux loss over loops is returned.

## Halt head

There is no identity expert anymore. In its place, `forward_step` computes a **halt probability**
each loop from a small linear head on the *incoming* hidden state:

```
p_halt = sigmoid(halt_proj(hidden_states))         # [B, S, 1], zero-init weight, bias -2.0
hidden_states = hidden_states + (1 - p_halt) * loop_scale * delta
```

`p_halt -> 1` means "this token doesn't need further refinement" and directly gates how much of
the loop's update lands — a **compute-allocation signal**, not a correctness score. It's greedy
and per-loop (recomputed from scratch each loop, not a cumulative ACT-style accumulator), so a
token can halt at one loop and un-halt at the next. `LoopMixtureOfExperts.forward` stacks `p_halt`
over loops into `[n_loops, B, S]` and returns it alongside the hidden states.

The trainer adds a **ponder loss** on `(1 - p_halt)` over real (non-pad) tokens, weighted by
`TrainingConfig.lambda_ponder`, ramped from 0 over `ponder_warmup_tokens` / `ponder_ramp_tokens`.
The warmup is load-bearing, not tunable: see the ponder-deadlock note in
[CLAUDE.md](../CLAUDE.md) and `tests/test_ponder_deadlock.py`.

## Per-loop CE supervision

Using `p_halt` for early exit at inference means `lm_head` must be able to read *any* loop's
hidden state, not just the last one -- otherwise thresholding `p_halt` and exiting early reads
from an interface never trained (PLAN.md Step 4a). `LoopMixtureOfExperts.forward` returns a 4th
value alongside `hidden_states`/`aux_loss`/`p_halt`: `hidden_states_all`, the stack of every
loop's (pre-norm) hidden state, `[n_loops, B, S, H]`. `TinyMoETransformer.forward` applies the
final `RMSNorm` to the whole stack (not just the last loop) before returning it as `x` when
`return_hidden=True` -- `lm_head` never reads the raw residual stream, so skipping this makes
per-loop losses meaningless.

`compute_mtp_loss` (`modules/model/mtp.py`) takes a `loop_ce_weights` list (one entry per loop,
`TrainingConfig.loop_ce_weights`, ascending -- e.g. `[0.1, 0.2, 0.3, 1.0]`) and computes the
chunked CE once per loop, summing the weighted results. Each loop's CE is independently chunked
(`CE_CHUNK_SIZE`), so at most one loop's one chunk of `[chunk, vocab]` logits is ever live --
looping over loops does not multiply peak logit memory the way materializing all loops' logits at
once would. The returned `loss_ce` (used for logging) is always the *final* loop's raw, unweighted
CE, matching its pre-Step-4a meaning. MTP heads still apply to the final loop only, never per loop.

Missing or wrong-length `loop_ce_weights` is a hard error (`compute_mtp_loss` asserts
`len(loop_ce_weights) == n_loops`); `config.py` also asserts this against `ModelConfig.Params`
at import time so a config typo fails fast instead of at the first training step.

## Correctness head

`p_halt` answers "is more compute useful here" (a compute-allocation signal); it is trained by LM
loss + ponder cost, so it learns whether refinement *changes* the prediction, not whether the
prediction is *right*. A confidently hallucinated fact is stable under refinement -- it halts early
and reads as maximum certainty. `TinyMoETransformer.correct_proj` (PLAN.md Step 4b) is a second,
independent `Linear(hidden_size, 1)` head (zero-init) that asks the right question directly: is
this prediction correct.

Like `lm_head`/`mtp_head`, `correct_proj` is applied externally rather than inside `forward()` --
`compute_mtp_loss` calls it only on the **final loop's** hidden states (never per loop, unlike the
CE terms above) when passed `correct_proj=` and `lambda_conf > 0`. Its target is free: inside the
same chunked pass that already computes the final loop's CE, `is_correct = (logits.argmax(-1) ==
labels)` under `torch.no_grad()` -- no extra forward pass, no labels beyond what CE already uses.
The `no_grad` is load-bearing: without it, the "correct" target would itself carry gradient back
through the shared CE logits, on top of `correct_proj`'s own gradient, quietly corrupting the LM
loss. `tests/test_correctness_head.py` checks this directly by comparing `lm_head`'s gradient with
the correctness term on vs. off -- it must be bit-identical.

Unlike the ponder loss, `lambda_conf` (`TrainingConfig.lambda_conf`, default `0.05`) is **not**
warmup-ramped -- there's no `loop_scale`-style deadlock precondition here, since `correct_proj`
doesn't gate anything else's gradient the way `p_halt` gates the loop's residual update.

This head is provisional. `p_max = softmax(logits).max()` is already a strong, free per-token
correctness predictor with no extra parameters. Gate 5 (`scripts/eval_calibration.py`, not written
yet) compares `p_correct = sigmoid(correct_proj(hidden))`'s ECE and abstention AUROC against
`p_max`'s on a held-out slice; if `p_correct` doesn't win on both, the plan is to revert this head
entirely (see PLAN.md's Step 4b for the exact revert scope) and use `p_max` for abstention
throughout post-training instead.

## Sparse MLP dispatch - `ParallelSparseMoELayer`

A naive MoE gathers a dense `[num_experts, tokens, ...]` tensor and runs every expert over every
token, wasting `num_experts / top_k` in matmul FLOPs. Instead:

1. Flatten `(token, slot)` assignments, **sort by expert id** (stable, for deterministic checkpoint
   recompute).
2. `bincount` gives per-expert group sizes (the only host sync)
3. One variable-sized grouped GEMM per expert via TE `GroupedLinear` (fused gate+up, then down).
4. Scale each output by its routing weight (null/identity slots carry weight 0), `index_add_` back.

MLP experts run in **BF16 even under low-precision autocast**: NVFP4 requires each groups row count
divisible by 16, which cant be guaranteed for dynamic per-expert group sizes without padding every
group, so sparsity and NVFP4 are mutually exclusive here.

## Expert-selection tracking

`_ExpertTracking` maintains per-token EMA statistics, like selection fraction, mean routed weight, mean
raw softmax probability when selected (`post_skew_dist`, a name left over from the deleted identity
skew — now just the un-renormalized selection probability) and is plotted every `sliding_window_size`
(256) steps during training. A
recompute guard (`begin_forward` / `_expected_updates`) prevents double counting when
`route()` reruns on the gradient-checkpointing backward pass, under checkpointing the guard covers
the sub-checkpointed case, though stats can still be double-counted in some configurations

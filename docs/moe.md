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

## Depth policy

There is no identity expert and, since Phase 0, no halt head either. `forward_step`'s update is
now just the per-loop gain:

```
hidden_states = hidden_states + loop_scale[loop] * delta
```

**What used to be here.** A learned `p_halt = sigmoid(halt_proj(hidden_states))` gated that update
as `(1 - p_halt) * loop_scale * delta`, trained by a "ponder" loss on `(1 - p_halt)` with a runtime
controller nudging its weight. It failed structurally, not by mistuning: `p_halt` collapsed to
~0.004 during the zero-λ warmup, overshot to ~0.78 when the ramp engaged, and pinned there for 14B
tokens while the controller cut the weight 11 times with no measurable effect. A saturated sigmoid
has no gradient, so λ was not a control knob. `loop_scale` grew to `[1.73, 1.81, 1.32]` to
compensate for the pinned gate. Full post-mortem in [CONCLUSION.md](CONCLUSION.md).

Deleting the gate naively would have multiplied every loop's delta by ~1/(1 − p_halt) and broken
the checkpoint, so `scripts/migrate_phase0.py` **folds the gate's measured per-loop mean into
`loop_scale`**: `loop_scale_new[k] = loop_scale_old[k] * mean_k(1 - p_halt)`. A migrated
checkpoint's `loop_scale` is therefore much smaller than a fresh `1/sqrt(n_loops)` init — that is
correct, not a bug. The gate had to be *measured*: the training log only recorded `p_halt` averaged
over all loops, hiding a strong decreasing trend (`[0.290, 0.134, 0.084]` on the SFT checkpoint).

**What replaced it.** A parameter-free convergence criterion, `converge_tol` on
`TinyMoETransformer.forward`. After each loop it reads out the **last position only** (one token ×
vocab — free next to a loop of the MoE block) and stops when the top-1 token is unchanged *and* its
log-probability moved less than `converge_tol`. Nothing is learned, so nothing can saturate.

Two things about it are load-bearing:

- It is measured in the **readout**, not in `‖Δh‖`. `loop_scale` still injects a sizeable hidden
  delta on the last loop while the prediction is already stationary, so a hidden-state criterion
  would never fire.
- It is **mutually exclusive with the KV cache** (asserted in both `LoopMixtureOfExperts.forward`
  and `TinyMoETransformer.forward`). An exited loop appends no K/V for that token, so a later
  full-depth step would attend over a cache with a hole in it. `scripts/inference.py` resolves this
  by turning the cache off when `--converge-tol` is set.

`scripts/eval_calibration.py` prints per-transition top-1 agreement and mean `|Δ log p_top|`, which
is how the threshold gets picked, alongside the early-exit CE curve that says what it costs.

## Per-loop CE supervision

Exiting early at inference means `lm_head` must be able to read *any* loop's hidden state, not just
the last one -- otherwise an early exit reads from an interface never trained.
`LoopMixtureOfExperts.forward` returns a third value alongside `hidden_states`/`aux_loss`:
`hidden_states_all`, the stack of every loop's (pre-norm) hidden state, `[loops_run, B, S, H]`.
`TinyMoETransformer.forward` applies the final `RMSNorm` to the whole stack (not just the last
loop) before returning it as `x` when `return_hidden=True` -- `lm_head` never reads the raw
residual stream, so skipping this makes per-loop losses meaningless.

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

## Confidence signal

`p_max = softmax(logits).max()` is the confidence signal everywhere downstream — training logs,
`sft.py`'s validation pass, `eval_calibration.py`, `eval_abstention.py`. It costs nothing and has
no parameters.

There used to be a learned alternative: `correct_proj`, a `Linear(hidden_size, 1)` head asking "is
this specific prediction correct", supervised by BCE against `is_correct = (logits.argmax(-1) ==
labels)`. It lost. The target was derived from `lm_head`'s own argmax on the same hidden state the
head reads, so "reproduce `p_max`" was the reachable optimum by construction — and that is exactly
what it did, tracking `p_max` to within 0.005 across the whole run. On the real checkpoint `p_max`
beat it on ECE *and* AUROC, and `(1 - p_correct)` scored 0.457 AUROC — **below chance** — at
flagging unanswerable questions, with mean `p_correct` *higher* on abstentions (0.835) than on real
answers (0.739). It was deleted in Phase 0; `expected_calibration_error` / `roc_auc` in
`eval_calibration.py` survive it as shared code.

Anything that replaces it has to *add* information over `p_max` rather than reproduce it: target
sampled continuations rather than teacher-forced tokens, make the target sequence-level, and feed
it the logit features explicitly. See [CONCLUSION.md](CONCLUSION.md).

`p_max` is computed as `1 / Σ_j exp(l_j - l_max)` rather than `logits.float().softmax(-1).max(-1)`
— the identity is exact, and it avoids two ~2GB fp32 transients per chunk at `chunk=8192 /
vocab=65536`, allocated on every step and again on the checkpoint recompute.

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

## Retrieval-entropy tracking

`RetrievalEntropyTracking` (in `modules/model/information_retrieval.py`) does the same job for the IR
experts table: an EMA, one slot per loop, of the retrieval softmax entropy divided by
`ln(num_ir_entries)`. It reads the weights already computed inside `InformationRetrievalModule.forward`
under `no_grad`, upcasting to fp32 in 1024-row chunks so the fp32 copy of a `[B*S, num_ir_entries]`
tensor never exists whole (67MB transient instead of ~0.5GB), and carries the same recompute guard
and every-8th-forward sampling as `_ExpertTracking`. The reduction is `torch.special.entr`, one
kernel for `-x log x`, which measured 3x faster than the written-out clamp/log/multiply form at the
real `[16384, 8192]` shape: 1.5ms per (loop, IR expert), so ~0.5ms amortized over a ~400ms step. No
host sync happens in the step path; `get_stats()` is the only one, at log cadence. `LoopMixtureOfExperts` owns one instance (`moe.ir_tracker`, `None` when there are
no IR experts) and hands it to every IR expert, so the trainer reads a single per-loop vector; it is
logged as `IR E/lnN: [...]` at the training log interval. The normalization is what makes it directly
comparable to the diagnostic 1 that `scripts/eval_stage0.py` prints: 1.0 is a uniform read over the
whole table, i.e. the table stores nothing.

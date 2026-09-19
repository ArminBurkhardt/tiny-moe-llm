# Looped Mixture-of-Experts

`LoopMixtureOfExperts` ([modules/model/moe.py](../modules/model/moe.py)) routes each token to a
mixture of heterogeneous experts, and repeats this `n_loops` times over a shared pool — a recurrent
(LoopLM-style) refinement of the representation rather than a single MoE pass.

## Expert pool

The router indexes a single flat expert list, ordered:

```
[ self-attn × A | cross-attn × A | IR × I | MLP × M ]
                                    ▲
                              first_mlp_index = 2A + I
```

With the default config: `A=1`, `I=1`, `M=32` → **35 experts** (`num_attn_experts` counts self *and*
cross, so it contributes `2A`). Types:

- **Self-attention** ([experts.py](../modules/model/experts.py)) — GQA over the sequence, its own
  head count (16 heads / 4 KV heads) and RoPE cache.
- **Cross-attention** — same, but keys/values come from the `other` stream (the projected MoE
  per-layer embedding of the same tokens).
- **Information-retrieval (IR)** — the selector; see below.
- **MLP** — SwiGLU FFNs, run sparsely as one grouped GEMM (see Sparse MLP dispatch).

There is no identity expert. Three always-on modules seed every loop's accumulator unconditionally,
outside the router pool entirely (not in `Router`'s output dim, not in the aux loss): `shared_mlp`, a
dense SwiGLU MLP; `shared_attn`, a `SelfAttention` reused for its RoPE/varlen path; and, when a
checkpoint carries the evidence port *and* a batch attaches evidence, `shared_evidence`, the
evidence reader. Static row count, so they run inside `te.autocast`.

Attention/IR experts run over the **full sequence regardless of routing**, so each is computed
**once per loop** and cached across the `top_k` slots; the top-k mask only scales their output.
Only the MLP experts are dispatched sparsely.

## Routing

`route()` per loop:

1. Router ([router.py](../modules/model/router.py)) = `RMSNorm -> Linear` produces logits. During
   training it adds **annealed exploration noise** `noise_factor * 0.3 * softplus(noise_proj(x)) * ε`
   (the 0.3 caps the initial level, which otherwise swamped the clean logits' ~0.33 std).
2. `loop_router_bias(loop_enc[loop])` is added — a zero-init linear over a **sinusoidal encoding of
   the absolute loop index**, not a learned `[n_loops, num_experts]` table, so `n_loops` stays a
   runtime choice and indices past the table reuse its last row.
3. Single softmax over the logits gives the selection distribution.
4. **Load-balancing aux loss** is computed directly from that softmax: `num_experts * Σ f_i * P_i`
   (hard token fraction `f_i` × mean soft prob `P_i`), minimized at a uniform distribution.
   Normalized by the loops actually run. It takes an optional `token_mask` so padded positions can
   be excluded (a mostly-padding batch reads a near-uniform routing signal on its pad rows and
   moved the aux loss 3x with row fill); `None` is bit-identical to the unmasked form, and the
   trainer does not pass it yet.
5. `top_k` selection → renormalize the selected weights to sum to 1. One `torch.topk` feeds both the
   aux loss and the selection.

Non-MLP experts are applied through one `[B, S, first_mlp_index]` gate built by `scatter_add_` (a
mask multiply, never boolean indexing, which is a device sync per expert); MLP slots are remapped to
expert-local indices and dispatched to the sparse layer. All expert outputs plus the always-on seed
are summed, then `RMSNorm` + dropout, then the residual update:

```
hidden_states = hidden_states + loop_scale[loop] * dropout(post_norm(output))
```

`loop_scale` is an `nn.Parameter` of shape `[n_loops]`, init `1/sqrt(n_loops)`, excluded from weight
decay; indices past `n_loops - 1` reuse the last entry. A checkpoint migrated by
`scripts/migrate_phase0.py` carries much smaller values (`[0.63, 0.32, 0.11]` on the phase2
lineage) because the deleted halt gate's measured per-loop mean was folded in — correct, not a
collapsed loop.

`loop_inject` (arm D, `scripts/migrate_loop_inject.py`) adds `inject(e)` of the block's input to
what the router and experts *read* at every loop, without touching the residual. It is zero-init,
inferred from the state dict, and measured not to help (Gate G2c failed); the code stays for the
record and the control.

## The IR expert: a selector over two stores

[information_retrieval.py](../modules/model/information_retrieval.py). The token is normed,
down-projected to `ir_dim` (384), given a zero-init **per-loop query bias** (same sinusoidal
encoding as the router's, so the query can differ between loops — Stage 0 measured
`cos(q2, q3) = 0.99` without it), and unit-normalized. It then reads:

- **The learned table**, `z_keys`/`y_values` `[65536, 384]`, through a **two stage read**: score 256
  centroids, open the top 8 clusters, score their 2048 members exactly, keep the global top 32,
  softmax over those divided by a learned `log_temperature` and a persistent anneal multiplier.
  Clusters are exactly equal in size, which is what makes the candidate scoring one `bmm` with no
  ragged gather and no host sync. `balanced_spherical_kmeans` refreshes the partition on a token
  cadence, warm-started, measuring candidate recall@32 on a reservoir of real queries and recycling
  entries that are dead by both a quantile cap and an absolute usage floor. `ir_num_clusters: 0` is
  the exact full-table read every pre-reshape checkpoint was trained under. Value rows are
  unit-normalized in the forward, so a read lands in `[1/√k, 1]` and its magnitude is its confidence.
- **The external store**, when a batch carries `chunk_keys` (bge-small vectors of the retrieved
  chunks): `key_adapter` rotates them into the query space, `exp(log_memory_scale)` puts them on the
  table's logit scale, and **one softmax over the union** decides the split. Two independently
  normalized reads summed would have no notion of which store won. A `[tokens, chunks]` visibility
  mask keeps every document on its own chunks. The summed external share, `last_memory_mass`
  (kept per loop in `memory_mass_by_loop`), is the groundedness signal Gate G3b reads;
  `last_memory_weights` keeps the per-chunk breakdown with its gradient for the selection loss.

The read goes `g_proj` → `up_proj` → the output stage. `ir_direct_read: true` (default, inferred
from the state dict) writes each token's **own** read through a zero-init `direct_gate`. The
original stage, an inner attention with keys/values from every position's own read, is kept
loadable for the A/B but measurably averaged a document's reads over its prefix: replacing the read
by its batch mean cost 0.0000 nats.

**What three arms found.** The table sharpens on loop 1 and the model does not use what it
retrieves: zeroing the read costs 0.0002 nats across three key inits and two widths, trained or not
([ir_sharpening.md](measurements/ir_sharpening.md), [ir_scale_fix.md](measurements/ir_scale_fix.md)).
A table trained on the trunk's own corpus can only offer content the trunk already holds. Its size
is frozen out of the real run spec; the external store — content the trunk provably lacks, worth
+3.23 nats in context ([evidence_ceiling.md](measurements/evidence_ceiling.md)) — is the mechanism.

## The evidence reader

`shared_evidence` is a `CrossAttention` whose key/value side is the encoded evidence
(`EvidenceBatch.states`, built by `TinyMoETransformer.build_evidence`; see
[architecture.md](architecture.md) §3), with its own `cu_seqlens_k`, per-chunk RoPE and
`causal=False`. At every loop it reads `step_input + evidence_query_bias(loop_enc[loop])` and its
output is scaled by `evidence_loop_scale[loop]` (one-init) before joining the accumulator, so
`loop_scale`'s own shrinkage does not stunt a later re-read. Before the projections, the evidence
states are multiplied per chunk by `1 + evidence_gate_scale * sigmoid(chunk mass)`, where the chunk
mass is the selector's mean external weight on that chunk over the document's own tokens — a
zero-init scale makes the gate *exactly* 1, and it gives the selector a dense, always-on gradient
through the reader. Its `o_proj` is zero-init. All of these tensors are in `is_fresh_loop_param`
and train at the from-scratch rate under `sft.py --evidence`.

[evidence.py](../modules/model/evidence.py) also holds `GroundednessHead` + `groundedness_loss`
(BCE against the corpus label "gold chunk present AND answerable", on the reader's output) and
`information_retrieval.evidence_selection_loss` (BCE between the per-chunk external share and the
gold flag). Both exist and are tested; neither is wired into `train_step` yet — see
[NEXT.md](plans/NEXT.md) Phase 4.

## Depth policy

There is no identity expert and, since Phase 0, no halt head either. **What used to be here:** a
learned `p_halt = sigmoid(halt_proj(hidden_states))` gated the update as
`(1 - p_halt) * loop_scale * delta`, trained by a "ponder" loss. It failed structurally: `p_halt`
pinned at ~0.78 (0.92 on the final loop) for 14B tokens while a runtime controller cut its weight
11 times with no effect — a saturated sigmoid has no gradient. Full post-mortem in
[CONCLUSION.md](CONCLUSION.md); the fold into `loop_scale` is above.

**What replaced it.** Two things, neither learned:

- **Stochastic loop depth during training** (`loop_count_sampling`): 30% of steps run a uniformly
  random depth in `1..n_loops-1`, with `loop_ce_weights` truncated and rescaled so the deepest loop
  run carries weight 1.0. Every depth becomes a real operating point.
- **The convergence exit at inference** (`converge_tol` on `TinyMoETransformer.forward`). After each
  loop it reads out the **last position only** and stops when the top-1 token is unchanged *and* its
  log-probability moved less than `converge_tol`. It reads the **readout**, not `‖Δh‖` (`loop_scale`
  still injects a sizeable hidden delta while the prediction is stationary); it is asserted
  inference-only; and it is asserted **mutually exclusive with the KV cache** — an exited loop
  appends no K/V for that token. `scripts/eval_calibration.py` prints per-transition top-1
  agreement and mean `|Δ log p_top|`, which is how the threshold gets picked.

A learned depth mechanism is parked until halting actually skips compute (NEXT.md, Parked).

## Per-loop CE supervision

Exiting early means `lm_head` must be able to read *any* loop's hidden state.
`LoopMixtureOfExperts.forward` returns `hidden_states_all`, the stack of every loop's hidden state,
`[loops_run, B, S, H]`; `TinyMoETransformer.forward` applies the final `RMSNorm` to the whole stack
before returning it under `return_hidden=True`.

`compute_mtp_loss` ([mtp.py](../modules/model/mtp.py)) takes `loop_ce_weights` (one per loop,
ascending, `[0.2, 0.3, 1.0]`) and computes the chunked CE once per loop, summing the weighted
results; the non-final loops are token-subsampled by `loop_ce_subsample` (0.25), an unbiased
estimate of the full mean. At most one loop's one chunk of `[chunk, vocab]` logits is ever live.
`loss_ce` (for logging) is always the *final* loop's raw, unweighted CE. MTP heads apply to the
final loop only. A wrong-length `loop_ce_weights` fails at config-load time.

Whether dense per-loop supervision is itself what makes loop 3 redundant is an open question with a
from-scratch test in [NEXT.md](plans/NEXT.md) 7c.

## Confidence signal

`p_max = softmax(logits).max()` is the confidence signal everywhere — training logs, `sft.py`'s
validation pass, `eval_calibration.py`, `eval_abstention.py`. Computed as `1 / Σ_j exp(l_j - l_max)`
to avoid two ~2GB fp32 transients per chunk. It carries no answerability signal (AUROC at or below
chance on every checkpoint); a linear probe of the final loop's last-position hidden state reads
0.584 on the trunk, the SFT and the repair checkpoints alike
([answerability_probe.md](measurements/answerability_probe.md)).

There used to be a learned alternative, `correct_proj`, supervised by BCE against `lm_head`'s own
argmax; "reproduce `p_max`" was its reachable optimum by construction and that is what it learned.
Deleted in Phase 0. `GroundednessHead` is the successor with an *external* label, which is what
keeps it from collapsing the same way.

## Sparse MLP dispatch - `ParallelSparseMoELayer`

1. Flatten `(token, slot)` assignments, **sort by expert id** (`stable=True`, for deterministic
   checkpoint recompute).
2. `bincount(...).tolist()` gives per-expert group sizes — the one accepted host sync per loop.
3. One variable-sized grouped GEMM per expert via TE `GroupedLinear` (fused gate+up, then down).
4. Scale each output by its routing weight (non-MLP slots carry weight 0), `index_add_` back.

MLP experts run in **BF16 even under low-precision autocast**: NVFP4 requires each group's row
count divisible by 16, which dynamic routing cannot guarantee.

## Tracking

`_ExpertTracking` keeps per-token EMAs of selection fraction, mean routed weight and mean softmax
probability when selected, sampled every 8th forward with a recompute guard against
activation-checkpoint double counting, plotted to `expert_selection.png`. Note the mean routed
weight is flat across experts by construction (top-2 scores renormalize to sum to 1); the selection
*fraction* is the signal.

`RetrievalEntropyTracking` does the same for the IR read: an EMA per loop of the softmax entropy
divided by `ln(width)`, where width is the softmax the module actually takes (`read_top_k` on the
two stage path, the table size on the exact path). 1.0 means the read is uniform over everything it
looked at. Logged as `IR E/ln32: [...]`, the same units `scripts/eval_stage0.py` prints. Reads the
weights already inside `forward` under `no_grad`, fp32 in 1024-row chunks, `torch.special.entr`,
~1.5ms per (loop, expert) on every 8th forward, no host sync in the step path.

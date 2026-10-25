# Looped Mixture-of-Experts

`LoopMixtureOfExperts` ([modules/model/moe.py](../modules/model/moe.py)) routes each token to a
mixture of heterogeneous experts, and repeats this `n_loops` times over a shared pool: a recurrent
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

- **Self-attention** ([experts.py](../modules/model/experts.py)): GQA over the sequence, its own
  head count (16 heads / 4 KV heads) and RoPE cache.
- **Cross-attention**: same, but keys/values come from the `other` stream (the projected MoE
  per-layer embedding of the same tokens).
- **Information-retrieval (IR)**: the selector; see below.
- **MLP**: SwiGLU FFNs, run sparsely as one grouped GEMM (see Sparse MLP dispatch).

There is no identity expert. Always-on modules seed every loop's accumulator unconditionally,
outside the router pool entirely (not in `Router`'s output dim, not in the aux loss): `shared_mlp`, a
dense SwiGLU MLP; `shared_attn`, a `SelfAttention` reused for its RoPE/varlen path (under the
key/value reader it also reads the evidence, below); and, when a checkpoint carries the cross
reader *and* a batch attaches evidence, `shared_evidence`, the cross evidence reader. Static row
count, so they run inside `te.autocast`.

Attention/IR experts run over the **full sequence regardless of routing**, so each is computed
**once per loop** and cached across the `top_k` slots; the top-k mask only scales their output.
Only the MLP experts are dispatched sparsely.

## Routing

`route()` per loop:

1. Router ([router.py](../modules/model/router.py)) = `RMSNorm -> Linear` produces logits. During
   training it adds **annealed exploration noise** `noise_factor * 0.3 * softplus(noise_proj(x)) * ε`
   (the 0.3 caps the initial level, which otherwise swamped the clean logits' ~0.33 std).
2. `loop_router_bias(loop_enc[loop])` is added: a zero-init linear over a **sinusoidal encoding of
   the absolute loop index**, not a learned `[n_loops, num_experts]` table, so `n_loops` stays a
   runtime choice and indices past the table reuse its last row.
3. Single softmax over the logits gives the routing probabilities `expert_scores`.
4. **Selection**: one `torch.topk` over `expert_scores + balance_bias[loop]` picks the `top_k`
   experts; the weights are gathered from the **unbiased** `expert_scores` at those indices and
   renormalized to sum to 1. `balance_bias` is the loss-free balancing buffer (below); at zero, which
   is every checkpoint before 2026-10-19 and every run under `moe_balancing: aux`, the selection is
   exactly the old router's. Under loss-free balancing the selection scores (the
   probabilities plus `balance_bias`) are formed in fp32 for the topk; a zero bias is bit-identical,
   and at bias 0.30 the bf16 sum picked differently on 2% of tokens.
5. **Load-balancing aux loss** is computed from the same softmax and the same top-k indices:
   `num_experts * Σ f_i * P_i` (hard token fraction `f_i` times mean soft prob `P_i`), minimized at a
   uniform distribution. It is computed in every mode but added to the objective only under
   `moe_balancing: aux` (weight `aux_loss_weight`); under `loss_free` it is logged and not trained
   on. Normalized by the loops actually run; under `detach_loops=k` the passes of the detached
   prefix contribute nothing and the sum is divided by the live passes alone
   (`loops_run - k`), so the per-pass gradient keeps its size. It takes an optional `token_mask` so
   padded positions can be excluded (a mostly-padding batch reads a near-uniform routing signal on
   its pad rows and moved the aux loss 3x with row fill); `None` is bit-identical to the unmasked
   form. Both trainers pass `input_ids != pad` through `TinyMoETransformer.forward`'s `token_mask`,
   so numbers from a run before that are not comparable to one after it. The masked branch counts,
   masks and reduces in fp32 since 2026-10-20: a bf16 count stops at 256, so every masked value
   logged before then was a constant and the term balanced nothing.

**Loss-free balancing** (`moe_balancing: loss_free`, arXiv 2408.15664). `balance_bias` is a
persistent buffer `[n_loops, num_experts]`, zero at init. Both trainers turn on per-pass hard load
counting (`set_balance_tracking(True)`, in every mode): `route` adds each selection's count into an
fp32 per-pass counter on device, a padded token counting for nothing. After each optimizer step the
trainers call `update_balance_bias(rate)`: for every pass and every MLP slot the step is
`sign(mean MLP load - load)`, the signs are **centred** over the MLP slots (their mean subtracted,
so the update is zero sum), and the bias moves by `rate` times that. The attention and IR slots
keep a zero bias, and centring keeps every MLP bias from drifting up together against them under a
skewed load, so the MLP against dense split stays the router's. A pass whose counts are all zero
(a shallower sampled depth did not run it) is left alone. The update runs on an fp32 shadow of the
buffer (the buffer follows the model's bf16 cast) and copies back; the rate is
`moe_balance_bias_rate` under `loss_free` and 0 under `aux`, which folds the counts into the load
EMA and leaves the bias where it is. `balance_load_stats` reads that EMA for the log line:
`load max/mean` (the most loaded MLP expert's share over the mean MLP share, 1.0 even) and
`dense share` (the attention and IR slots' share of the selections) per pass, logged in every mode
since 2026-10-20. A checkpoint without the buffer loads as zero (`BALANCE_KEYS`).

Non-MLP experts are applied through one `[B, S, first_mlp_index]` gate built by `scatter_add_` (a
mask multiply, never boolean indexing, which is a device sync per expert); MLP slots are remapped to
expert-local indices and dispatched to the sparse layer. All expert outputs plus the always-on seed
are summed, then `RMSNorm` + dropout, then the residual update:

```
hidden_states = hidden_states + loop_scale[loop] * dropout(post_norm(output))
```

`loop_scale` is an `nn.Parameter` of shape `[n_loops]`, init `1/sqrt(n_loops)`, excluded from weight
decay; indices past `n_loops - 1` reuse the last entry. A checkpoint migrated by
`scripts/migrate_phase0.py` carries much smaller values because the deleted halt gate's measured
per-loop mean was folded in: `[0.637, 0.326, 0.098]` on `checkpoint_phase2_final` and
`[0.501, 0.242, 0.115]` on `checkpoint_sft_final`
([phase0_migration.md](measurements/phase0_migration.md)). Correct, not a collapsed loop.

`loop_inject` (arm D, `scripts/migrate_loop_inject.py`) adds `inject(e)` of the block's input to
what the router and experts *read* at every loop, without touching the residual. It is zero-init,
inferred from the state dict, and measured not to help (Gate G2c failed); the code stays for the
record and the control.

## The IR expert: a selector over two stores

[information_retrieval.py](../modules/model/information_retrieval.py). The token is normed,
down-projected to `ir_dim` (384), given a zero-init **per-loop query bias** (same sinusoidal
encoding as the router's, so the query can differ between loops; Stage 0 measured
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
is frozen out of the real run spec; the external store, content the trunk provably lacks, worth
+3.23 nats in context ([evidence_ceiling.md](measurements/evidence_ceiling.md)), is the mechanism.

## The evidence reader

Two readers exist; the checkpoint's marker buffers decide which one a model builds (see
[architecture.md](architecture.md) §3 for both).

**The key/value reader** (`evidence_reader="kv"`, marker `moe.evidence_reader_kv`, `sft.py
--reader-kv`), the measured reader since 2026-10-03 and the pilot's. No reader module and no
`evidence_query_bias`: inside `forward_step` the encoded evidence (`EvidenceBatch.states`) goes to
`shared_attn`, which passes it through its own norm, multiplies it per token by the chunk gate
(after the norm), projects it with its own `k_proj`/`v_proj`, scales the values by
`evidence_loop_scale[loop]`, and reads it as leading keys of each query segment under one softmax
with the segment's own causal keys (`attention.prefix_varlen_attention`), the prefix rotated at
`start_s - E_s + i` (`evidence.prefix_position_ids`). The read shares the softmax with the
segment's own keys, so this reader is **not neutral with evidence attached**, even at init; without
evidence it is bit-identical to the model without the port.

**The cross reader** (`shared_evidence`, the legacy mode) is a `CrossAttention` whose key/value
side is the encoded evidence (built by `TinyMoETransformer.build_evidence`), with its own
`cu_seqlens_k`, per-chunk RoPE and `causal=False`. At every loop it reads
`step_input + evidence_query_bias(loop_enc[loop])` and its output is scaled by
`evidence_loop_scale[loop]` (one-init) before joining the accumulator, so `loop_scale`'s own
shrinkage does not stunt a later re-read. Its `o_proj` is zero-init, which makes a fresh cross
reader neutral even with evidence attached; that neutrality argument holds for the cross reader
only.

**The chunk gate** is shared by both in the code: the evidence states are multiplied per chunk by
`1 + evidence_gate_scale * sigmoid(chunk mass)`, where the chunk mass is this loop's selector mean
external weight on that chunk over the document's own tokens. A zero-init scale makes the gate
*exactly* 1, and a nonzero one gives the selector a dense, always-on gradient through the reader.
In every config the scale is frozen at exactly 0 (`freeze_evidence_gate`, `config.py`), so the
gate is the identity and, since 2026-10-24, it is skipped under both readers: no selector mass,
scatter or multiply by ones per loop (the zero test is cached against the scale's `_version`, no
per-step sync; `tests/test_evidence_gate_skip.py`). All of these tensors are in `is_fresh_loop_param` and train at the from-scratch rate under
`sft.py --evidence`.

`information_retrieval.evidence_selection_loss` (BCE between the per-chunk external share and the
gold flag) is the selector's supervised half, reached through
`LoopMixtureOfExperts.evidence_selection_term`: it reads each IR module's `last_memory_weights`
(the per-chunk breakdown before the sum-and-detach) against the mask the recurrence built once in
`last_memory_visible`, averages over every loop and IR expert, and `train_step` adds it at
`TrainingConfig.evidence_selection_weight`. Since 2026-10-24 the BCE is in log space with no clamp on
the renormalized share, rows with fewer than two visible chunks are excluded, and
`evidence_selection_weighting` (`token`, default; `segment`, each query segment averaged first from
`cu_seqlens`; passed through `evidence_selection_term(segment_ids=, num_segments=)`) sets the average. Uniform loop weights, not `loop_ce_weights`' ascending
ones: a relevant chunk is relevant at every depth. `None` whenever the batch had no external
store or the corpus no gold flag, which is every profile but `--evidence`.

[evidence.py](../modules/model/evidence.py) also holds `GroundednessHead` + `groundedness_loss`
(BCE against the corpus label "gold chunk present AND answerable", on the reader's output), added
to a checkpoint by `migrate_groundedness_head.py` and inferred from the state dict.
`TinyMoETransformer.groundedness_term` reads `moe.last_reader_output` (under the cross reader, the
read before the per-loop gain and before the shared/routed sum, because the question is about the
read rather than about what the accumulator did with it; under the key/value reader, `shared_attn`'s
whole output, the attention that did the read) at the last prompt token of each supervised span, and
`train_step` adds it at `TrainingConfig.groundedness_weight`. `gold_present` comes from
`last_memory_visible @ chunk_gold`, `answerable` from the corpus's `.ans` sidecar.

Under the cross reader its weight gradient is **exactly zero until the reader's `o_proj` leaves
zero**, because the head reads a zero vector until then and only its bias can move (the key/value
reader's output is never zero, so this does not apply there). Since 2026-10-24 the head's input is
detached under the key/value reader: the head still learns, no trunk parameter gets a gradient from it. So an early falling `grounded:` is the
base rate being learned; the held-out AUROC in the `[eval]` line is what says otherwise.

## Depth policy

There is no identity expert and, since Phase 0, no halt head either. **What used to be here:** a
learned `p_halt = sigmoid(halt_proj(hidden_states))` gated the update as
`(1 - p_halt) * loop_scale * delta`, trained by a "ponder" loss. It failed structurally: `p_halt`
pinned at ~0.78 (0.92 on the final loop) for 14B tokens while a runtime controller cut its weight
11 times with no effect: a saturated sigmoid has no gradient. Full post-mortem in
[CONCLUSION.md](CONCLUSION.md); the fold into `loop_scale` is above.

**What replaced it.** Two things, neither learned:

- **Stochastic loop depth during training** (`loop_count_sampling`): 30% of steps run a uniformly
  random depth in `1..n_loops-1`, with `loop_ce_weights` truncated and rescaled so the deepest loop
  run carries weight 1.0. Every depth becomes a real operating point.
- **The convergence exit at inference** (`converge_tol` on `TinyMoETransformer.forward`). After each
  loop it reads out the **last position only** and stops when the top-1 token is unchanged *and* its
  log-probability moved less than `converge_tol`. It reads the **readout**, not `‖Δh‖` (`loop_scale`
  still injects a sizeable hidden delta while the prediction is stationary); it is asserted
  inference-only; and it is asserted **mutually exclusive with the KV cache**: an exited loop
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

`p_max = softmax(logits).max()` (its sum, like the CE and top-1 sums, accumulates in fp32 since 2026-10-24; the evidence encoder's padding is cut into 128-token `EVIDENCE_PAD_SEGMENT` segments) is the confidence signal everywhere: training logs, `sft.py`'s
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
2. `bincount(...).tolist()` gives per-expert group sizes, the one accepted host sync per loop. Since
   2026-10-24 the layer delegates to `experts.grouped_mlp_forward` with `null_mask=~mlp_mask`: the
   dense-slot picks (`(index 0, weight 0)`) sort after the real experts, are counted in the same
   `bincount().tolist()` sync and sliced off before the grouped GEMM (exact zeros as before; outputs
   differ only by bf16 summation order; `tests/test_efficiency_2026_10_24.py`).
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

# NEXT.md

The plan. The record of how we got here is [docs/CONCLUSION.md](../CONCLUSION.md) (the 16B-token
run and everything it measured); read [CLAUDE.md](../../CLAUDE.md) first, it is authoritative for
everything already built.

> **Phase 0 is done** — see the note at the end of that section for what the code actually looks
> like now and where this document's own numbers turned out to be wrong.

**The plan in one line:** remove both learned heads, then turn the IR expert + CrossAttention
expert into a real retriever/reader pair over external evidence — all as finetunes of the existing
16B-token checkpoint.

## Rules (carried over)

- Commit per logical change (`feat:` / `chore:`, branch `train-build`), single-line subjects.
- Lowercase explanatory comments that justify *why*; Google-style docstrings with `Args:`.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, `moe` in the training loop
  goes through `accelerator.unwrap_model(model)`.
- **Never add `.item()` / `.tolist()` / `.cpu()` / boolean mask indexing to the per-step path.**

## Decisions already made

| decision | choice |
|---|---|
| run shape | **Finetune the existing checkpoint.** No fresh pretrain. Every shape change must stay loadable. |
| evidence port | **Option C** — IR expert = selector, CrossAttention expert = reader |
| query/key space | **B2** — external embedder (bge-small, 384-d) + adapter. Revisit B1 after Stage 3. |
| abstention owner | **Both, sequenced and measured separately** — data repair first, Stage 2 on top |
| depth policy | **Convergence exit + "evidence still arriving"** — no learned head, two criteria |
| IR table | **256-d, 65536 entries, two-stage centroid scoring** |
| MTP | **Keep on.** Drop for the finetune only if throughput or loss goes badly (Phase 4 fallback). |
| evidence RoPE | **Per-chunk position basis restarting at 0** |
| ANN corpus | **phase1.bin for Stage 3 InfoNCE, Wikipedia for Stage 4/5 and all reported numbers** |

---

## Phase 0 — Heads out ✅

Both heads failed for structural reasons, not tuning ones. This is a subtraction.

### 0a. Correctness head — delete

Gate 5 resolved against it on the real checkpoint: `p_max` beat `p_correct` on ECE (0.371 vs
0.378) and AUROC, and `(1 - p_correct)` scored **0.457 AUROC — below chance** — for flagging
unanswerable questions, with mean `p_correct` *higher* on abstentions (0.835) than on real answers
(0.739). The BCE target was derived from `lm_head`'s own argmax on the same hidden state the head
reads, so "reproduce `p_max`" was the reachable optimum by construction.

Remove: `correct_proj` (`transformer.py`), the BCE term in `compute_mtp_loss` (`mtp.py`),
`TrainingConfig.lambda_conf` + `config.yaml`'s `lambda_conf`, `tests/test_correctness_head.py`.
Substitute `p_max` at every read site: `sft.py` val logging, `eval_abstention.py`,
`eval_calibration.py`. Keep `expected_calibration_error` / `roc_auc` — they are shared code.

### 0b. Halt head — delete, fold the gate into `loop_scale`

`p_halt` collapsed to ~0.004 during the zero-λ warmup, overshot to ~0.78 when the ramp engaged,
and pinned there for 14B tokens while the auto-adjust controller cut `lambda_ponder` 11 times with
no measurable effect. A saturated sigmoid has no gradient; magnitude nudges cannot recover it.

**The trap:** `forward_step` is
`h = h + (1 - p_halt) * loop_scale[loop] * dropout(post_norm(output))`. With `(1 - p_halt) ≈ 0.22`
pinned, `loop_scale` grew to `[1.73, 1.81, 1.32]` to compensate. Deleting the gate naively
multiplies every loop's delta by ~4.5x and the checkpoint stops working.

**Fold it:** rewrite `loop_scale` to the measured effective gate `[0.38, 0.40, 0.29]` at load time
(measure the actual per-loop mean `(1 - p_halt)` on held-out data first; don't trust the logged
scalar). Then delete `halt_proj`, the `p_halt` return value and its `[n_loops, B, S]` plumbing, the
ponder loss, `modules/runtime/ponder.py` + `PonderController`, all `ponder_*` config keys, the
`ponder_state` checkpoint field, and `tests/test_ponder_deadlock.py` /
`tests/test_ponder_autoadjust.py`. `load_checkpoint`'s tuple arity changes — no live run depends on
it now, but bump it in one commit.

### 0c. Depth without a head

Two parameter-free criteria, both evaluated at the **last position only** (1 token × 65536 vocab —
free at generation time; under teacher forcing the per-loop readouts already exist for per-loop CE):

1. **Convergence** — `KL(p_k ‖ p_{k-1})` or top-1 agreement between consecutive loops' `lm_head`
   readouts. Measure in the *readout*, not `‖Δh‖`: `loop_scale[2] = 1.32` means the hidden delta is
   large while the prediction is stationary, so a hidden-state criterion never fires.
2. **Evidence still arriving** — keep looping while the append-only evidence buffer is still
   growing; stop when a loop's retrieval adds nothing new.

One threshold each, tunable post-hoc, no loss term, nothing that can saturate.

**Gate P0:** re-run `eval_abstention.py` and `eval_calibration.py` on the folded checkpoint. Loss
and top-1 must be within noise of the pre-removal numbers — this phase is meant to change nothing
behaviorally, only to remove dead machinery. Any regression means the fold constant is wrong.

### What actually shipped (2026-08-19)

`scripts/migrate_phase0.py` does the fold and the strip. It **measures** the gate per checkpoint
rather than taking a constant, because the guess above was wrong:

| | 0b's guess | measured, `checkpoint_sft_final` | measured, `checkpoint_phase2_final` |
|---|---|---|---|
| mean `(1 - p_halt)` per loop | `[0.22, 0.22, 0.22]` (implied) | `[0.290, 0.134, 0.084]` | `[0.367, 0.179, 0.074]` |
| folded `loop_scale` | `[0.38, 0.40, 0.29]` | `[0.501, 0.242, 0.115]` | `[0.637, 0.326, 0.098]` |

The single logged `p_halt` scalar was a mean over all three loops and hid a strong decreasing
trend, so a flat fold would have left loop 1 ~25% too weak and loop 3 ~2.6x too strong. The gate is
stable across corpora within a checkpoint (~5%) but differs a lot between checkpoints — hence
"measure it, per checkpoint". Its per-token std is small (≈0.03–0.06), which is what makes folding
to a mean legitimate at all.

**Gate P0 passed on both** (local `phase1` slice, 654k tokens, identical settings before/after):

| | `sft_final` before → after | `phase2_final` before → after |
|---|---|---|
| final-loop CE | 3.7564 → 3.7604 | 3.3720 → 3.3843 |
| top-1 | 0.3644 → 0.3628 | 0.3979 → 0.3963 |
| ECE(`p_max`) | 0.0925 → 0.0887 | 0.0088 → 0.0089 |
| AUROC(`p_max`) | 0.8236 → 0.8230 | 0.8390 → 0.8376 |

`p_correct` lost to `p_max` again on both checkpoints, so 0a's revert is confirmed a third time.

Two deviations from the text above, both deliberate:

- **Only the convergence criterion of 0c is implemented** (`converge_tol` / `min_loops` on
  `TinyMoETransformer.forward`, surfaced in `inference.py` and the Gradio app). The "evidence still
  arriving" criterion reads an append-only evidence buffer that does not exist until Phase 4.
- **The convergence exit and the KV cache are mutually exclusive.** An exited loop appends no K/V
  for that token, so a later full-depth step would attend over a cache with a hole in it. Filling
  those caches cheaply (K/V projections only, skipping the skipped loops' experts) is real plumbing
  through every attention expert and was left out of a subtraction-shaped change.

Measured convergence statistics, for picking `converge_tol` (`eval_calibration.py` prints these):

| transition | top-1 agreement | mean `\|Δ log p_top\|` |
|---|---|---|
| loop 1 → 2 | 0.808 | 0.232 |
| loop 2 → 3 | 0.925–0.944 | 0.067–0.091 |

A migrated checkpoint drops its optimizer/scheduler state — it is a finetune **seed**, not a resume
point.

---

## Phase 1 — Stage 0 diagnostics (no training, hours)

Non-negotiable. Every design below is conditioned on numbers that don't exist yet and that the
existing checkpoint can produce in minutes. **One script against the SFT checkpoint.**

1. **Retrieval entropy** of the IR softmax. Expect ≈ `ln 8192 = 9.01` nats (i.e. maximal — the
   table stores nothing addressable).
2. **IR ablation** — zero the expert's output, measure ΔCE on held-out. → **Gate G1**.
3. **Query drift** — `cos(down_proj(h_loop1), down_proj(h_loop3))`. If ≈ 1, the loop-conditioned
   query bias (Phase 5 item 1) is mandatory, not optional.
4. **Loop-3 flip rate** — top-1 flip rate between loop 2's and loop 3's readouts, plus
   `‖Δh‖/‖h‖` and `cos(Δ_k, Δ_{k-1})`. Sets the convergence threshold in 0c and tells you whether
   loop 3 is idle (ship `n_loops=2`) or churning.
5. **Oracle minimum sufficient depth** — per token, the smallest `n_loops ∈ {1,2,3}` whose argmax
   matches the label, "never correct" bucketed separately. This histogram is the honest ceiling on
   what any adaptive-depth scheme could ever buy.

**Gate G1: IR ablation ΔCE > ~0.02 nats.** If ~0, the expert is a bias term today — that's expected
and fine (it's what makes re-initializing it free), but it means the router's 7–9% preference is
*not* evidence the pathway works, and the Phase 3 ablation becomes the real test.

### What Stage 0 measured (2026-08-20)

`scripts/eval_stage0.py`, both migrated checkpoints, the same held-out slice Gate P0 used (local
`phase1`, doc 0 onward, 40 x 4 x 4096 = 654,128 supervised tokens), run at `--max-loops 6`. Full
reports in [docs/measurements/](../measurements/). Loop-3 CE and top-1 reproduce the Gate P0 table
to the last digit on both checkpoints, so this harness is reading the same quantity that one did.

| | `sft_final` | `phase2_final` |
|---|---|---|
| 1. retrieval entropy, loop 3 (max `ln 8192` = 9.011) | 8.9671 (99.51%) | 8.9650 (99.49%) |
| max weight / top-32 mass (uniform: 0.000122 / 0.0039) | 0.000229 / 0.0069 | 0.000245 / 0.0073 |
| read dispersion, loop 1 / loops 2–6 | 0.53 / 0.056–0.065 | 0.20 / 0.023–0.034 |
| 2. **ΔCE with the read zeroed (Gate G1 wants > 0.02)** | **0.0004** | **0.0002** |
| ΔCE with the read replaced by its own batch mean | 0.0000 | −0.0000 |
| IR routed weight / selection rate, best loop (uniform: 0.0286 / 0.0571) | 0.0249 / 0.070 | 0.0368 / 0.081 |
| 3. `cos(q1, q2)` / `cos(q2, q3)` | 0.078 / 0.988 | −0.659 / 0.989 |
| 4. top-1 flip 1→2 / 2→3 | 0.189 / 0.073 | 0.191 / 0.054 |
| `‖Δh‖/‖h‖`, loops 1 / 2 / 3 | 0.80 / 0.32 / 0.14 | 0.87 / 0.36 / 0.10 |
| `cos(Δ2, Δ1)` / `cos(Δ3, Δ2)` | 0.371 / 0.733 | 0.299 / 0.634 |
| 5. earliest-correct-loop histogram, loops 1/2/3 | 0.349 / 0.032 / 0.009 | 0.383 / 0.035 / 0.006 |
| never correct at any depth | 0.594 | 0.562 |
| oracle vs. actual top-1 at loop 3 (headroom) | 0.389 vs 0.363 (0.026) | 0.424 vs 0.396 (0.028) |
| 6. CE at loops 3 / 4 / 5 / 6 | 3.7604 / 3.7594 / 3.7724 / 3.7965 | 3.3843 / 3.3889 / 3.4026 / 3.4231 |

**1 + 2 — the table stores nothing, and the read is a constant.** Entropy is 99.5% of `ln 8192` on
every loop of both checkpoints, and the top 32 of 8192 entries carry 0.7% of the mass against a
uniform 0.39%. Zeroing the read costs 0.0004 nats; replacing it with its own batch mean costs
0.0000, so *all* of that already-negligible contribution is content-free. The read-dispersion column
says the same thing in vector form — past loop 1 a token's read deviates from the mean read by 2–6%
of its norm. **Gate G1 fails by ~50x**, which is the branch this section predicted: re-initializing
`z_keys`/`y_values` in Phase 3 costs nothing, and the router's above-uniform interest in the IR slot
(`phase2_final`: 0.037 mean routed weight vs 0.029 uniform) is now confirmed to be a preference for
a bias term. The Phase 3 ablation is the real test.

**3 — query drift is zero after loop 1, so 5c item 1 is mandatory, not conditional.** `cos(q2, q3)`
is 0.988/0.989 and every later pair is ≥ 0.96. Loop 1's query is the only distinct one, and on the
pretrained checkpoint it points the other way entirely (−0.659) before the query snaps to a fixed
direction and stays there. Re-executed retrieval cannot make loop 3 differ from loop 2 without the
loop-conditioned query bias.

**4 — loop 3 is not idle, it is redundant.** It still moves the residual stream by 14%/10% of `‖h‖`
and flips 7.3%/5.4% of top-1 predictions, but 73%/63% of that movement is aligned with loop 2's
direction, and it buys 0.021/0.008 nats. "Ship `n_loops=2`" is not the reading; "later loops repeat
the previous update because nothing feeds them anything new" is.

**5 — oracle headroom is 2.6/2.8 points of top-1, and each loop's slice of it shrinks.** Read this
as a floor on today's checkpoint, not a depth recommendation. It is measured on plain LM
continuation, where most next tokens are settled by loop 1 in *any* looped LM, and with none of the
mechanisms that are supposed to give a later loop something new to work on (evidence buffer,
loop-conditioned query, novelty pressure) in existence yet. The number that decides depth is G5's
EM-vs-`n_loops` curve with a corpus attached, and this histogram is expected to change under Phase
4/5 — if it doesn't, that is the finding.

**6 — depth past the trained count degrades gracefully but buys nothing.** Loops 4–6 run without
incident (the sinusoidal loop encoding and `loop_scale`'s clamp hold), and the CE curve is
flat-then-slightly-worse rather than divergent. That flat curve is the baseline the Phase 5c depth
curriculum has to beat, and the reason "minimum depth well above 3" is a training goal rather than a
config change.

**Consequences for the plan:** 5c item 1 (loop-conditioned IR query) is promoted from conditional to
required and is a precondition for any depth story, not a Stage-5 refinement; Phase 3's re-init is
confirmed free; `n_loops=3` stays the shipping depth until something actually feeds later loops new
information.

---

## Phase 2 — Abstention repair (parallel track, independent subsystem) ✅

Runs alongside Phase 1/3; touches only data. **Owns the first half of the abstention fix**;
Phase 4's no-evidence condition owns the second, and each is measured separately so the movement is
attributable.

Root cause: SQuAD v2 is 7.5% of SFT tokens but **25.6% of conversations**, and its unanswerable
third is a ~6-token, extremely low-entropy target. Per-token CE makes a memorized refusal the
cheapest available loss reduction — 7,786 of 11,873 completions are literally
`"The passage doesn't say."`

Fixes, in order of expected effect:

1. Down-sample SQuAD v2's unanswerable rows to ~10–15% of the QA subset (from ~33–50%).
2. Add answerable-only extractive QA (SQuAD 1.1, NQ-open, TriviaQA, HotpotQA) so answerable
   QA-shaped volume isn't dwarfed.
3. **Weight the loss per conversation, not per token**, so a 6-token refusal stops mechanically
   out-earning a multi-sentence real answer.
4. Vary abstention *training* phrasings; keep the closed set in `modules/data/abstention.py` for
   eval-side `is_abstention` detection only.

Delivery: a short repair finetune on the existing SFT checkpoint (~20–50M tokens, `lr=1e-5`,
1 epoch) — hours on a rented box, not a 708.9M-token rerun.

**Gate P2:** answerable-half false-abstention rate materially below the current **78.4%**;
abstention precision clearly above the ~0.50 base rate. Step 13's eventual bar is <10%.

### What shipped (2026-08-20)

All four fixes, plus the repair finetune. Full report and the corpus composition behind it in
[docs/measurements/abstention_repair.md](../measurements/abstention_repair.md).

- **1 — down-sampling** is `SFTSource.unanswerable_keep` / `--squad-unanswerable-fraction`, applied
  before the shard's conversation list exists so it never occupies a resume `row_idx`. 0.40 for the
  repair profile, 1.0 (no-op) for the plain SFT one.
- **2 — answerable extractive QA**: `rajpurkar/squad` (SQuAD 1.1) and `hotpotqa/hotpot_qa`
  (distractor), both rendered under `SQUAD_INSTRUCTION` **verbatim**, because the quantity that has
  to move is P(answer | this exact prompt shape). **NQ-open and TriviaQA were deliberately left
  out**: their usable configs are closed-book, so they share neither the prompt shape Gate P2
  measures nor the extraction task, and at 332M params closed-book QA targets teach guessing — the
  opposite of the calibration goal.
- **3 — per-conversation loss weighting**: `SFTDataset` emits `loss_weights` = `1/(supervised tokens
  in this conversation)`, plumbed through `train_step` → `compute_mtp_loss` → `_chunked_linear_ce`,
  which becomes `sum(w*ce)/sum(w)`. `p_max`/top-1 stay token-level so they remain comparable across
  every checkpoint measured here. Off for `SFTConfig`, on for `RepairConfig`.
- **4 — 15 abstention phrasings** (`ABSTENTIONS_PASSAGE_TRAIN`) for training, with
  `ABSTENTIONS_PASSAGE` (the original 5) still the eval's forced reference. `is_abstention` matches
  the union — a detector that knew only the old five would under-report the very rate this gate
  checks.

**Gate P2 passed** (2,000 SQuAD v2 validation questions, identical flags on both checkpoints):

| | `sft_final_phase0` | `repair_final` |
|---|---|---|
| **false abstention (answerable half)** | **0.7832** | **0.1358** |
| abstention precision (base rate 0.5065) | 0.5154 ± 0.0125 | 0.5759 ± 0.0278 |
| abstention recall | 0.8115 | 0.1797 |
| EM / token F1 (answerable half) | 0.0648 / 0.0806 | 0.1611 / 0.2162 |
| most common completion | 1,398 / 2,000 | 259 / 2,000 |
| distinct completions | 320 | 1,293 |

**Read the recall row.** The collapse is gone — one string went from 70% of all completions to 13%,
and answerable-half EM tripled — but the model now over-answers: it fails to refuse 82% of
unanswerable questions, where before it refused 81% of them. That is the half of the abstention fix
this document assigns to **Phase 4's no-evidence condition**, and this is the first checkpoint where
the two halves are measured separately rather than confounded. The nearer lever is data: the
realized unanswerable share came out at **9.6% of QA conversations**, just under the 10-15% target,
so `--squad-unanswerable-fraction 0.55` is the obvious retune before anything architectural.

Two things that did **not** move, and matter:

- **`p_max` still carries no signal about answerability.** AUROC of `1 - p_max` for detecting an
  unanswerable question: 0.462 before, 0.471 after — below chance both times. It predicts whether
  *this answer* is right (0.619 → 0.660) and nothing about whether the question could be answered.
  Same conclusion Phase 0 reached about `p_correct`, unchanged by repairing the data.
- **`repair_val` CE fell 2.088 → 1.910 with `p_max` and top-1 flat** across all nine eval points.
  The trunk did not move; the abstention decision did.

One operational finding worth carrying into every future local run: at `sft:`'s batch size of 8 the
finetune peaks at 29.55GB allocated on the 32GB 5090, pushing the driver to ~32.0GB and spilling
into **shared system memory** at 9.4k tokens/sec. At 4 x 4096 with accumulation 4 — identical tokens
per optimizer step, identical objective — it is 21.36GB, resident, at 22-37k tokens/sec.

### Early stopping is not the recall lever (2026-08-20)

Checked first, because it would have been free. The rolling checkpoints, all at the same eval flags
as the table above:

| | SFT seed | 30M | 40M | final (49.3M) |
|---|---|---|---|---|
| false abstention (answerable half) | 0.7832 | 0.1429 | 0.1226 | 0.1358 |
| abstention precision | 0.5154 | 0.5778 | 0.5784 | 0.5759 |
| abstention recall | 0.8115 | 0.1905 | 0.1639 | 0.1797 |
| EM (answerable half) | 0.0648 | 0.1520 | 0.1641 | 0.1611 |
| abstentions made (of 2,000) | 1,595 | 334 | 287 | 316 |

**The flip is complete well before 30M** and precision is flat at ~0.578 across all of them, so
there is no intermediate checkpoint that trades back into balance — it is one policy sampled at
three times, not a trajectory through a better one. The corpus ratio is the lever.

### Retune at 0.55, and archiving the corpus (2026-08-20)

Ordered, because steps 1 and 3 are both about not destroying the thing being replaced:

```bash
# 1. keep the 0.40 build. prepare_sft_data.py deletes each source shard right after appending it,
#    so a corpus overwritten in place costs a full re-download and re-tokenize to get back
mkdir -p data/prepared_frac040
mv data/prepared/repair_{train,val}.{bin,idx,mask} \
   data/prepared/_prepare_state_repair.json data/prepared_frac040/

# 2. rebuild at the top of the 10-15% band. check the realized share the run prints at the end
#    BEFORE training on it -- it is a share of QA conversations, not of squad_v2's rows
python scripts/prepare_sft_data.py --profile repair --squad-unanswerable-fraction 0.55

# 3. a fresh run directory. sft.py resumes from the newest checkpoint in ckpts/repair, so leaving
#    the previous run there means adopting its optimizer state and LR schedule, not replacing it
mv ckpts/repair ckpts/repair_frac040
python scripts/sft.py --repair -c ckpts/trained/checkpoint_sft_final_phase0.pt

# 4. the gate, at the exact flags every number above was measured at (batch size is part of the
#    measurement -- the MoE's grouped GEMM tiles by the batch's per-expert row counts)
python scripts/eval_abstention.py -c ckpts/repair/checkpoint_repair_final.pt \
  --max-examples 2000 --batch-size 16 --skip-forced
```

**Archive a corpus rather than rebuilding it.** `scripts/archive_corpus.py` packs a split's
`.bin`/`.idx`/`.mask` plus the prep resume sidecar into one `.tar.gz` under `data/archives/`, with a
JSON sidecar holding per-file sha256, the token/document counts read off the files themselves, and
the matching `manifest.json` slice. Restore is always explicit, so an archive can never silently
shadow a fresh build:

```bash
python scripts/archive_corpus.py pack --all        # every split in data/prepared
python scripts/archive_corpus.py list              # measured counts vs. what the builder claimed
python scripts/archive_corpus.py restore repair_train --force
```

The measured-vs-claimed split in `list` is load-bearing on this box: the dev machine's
`phase1.bin` is a small local stand-in while `manifest.json` describes the rented box's 24.7B-token
build, and the two must not be read for each other.

---

## Phase 3 — IR expert reshape and sharpening

The table holds no information, so re-initializing it costs nothing. Do the reshape and the
temperature anneal in one finetune.

### Shape

| tensor | before | after |
|---|---|---|
| `z_keys` / `y_values` | `[8192, 128]` | `[65536, 256]` |
| `down_proj` / `up_proj` | `768↔128` | `768↔256` (re-init, `strict=False`) |
| `g_proj` | `128→128` | `256→256` (re-init) |
| centroids (new) | — | `[256, 256]` |
| `temperature` | `1.0` hardcoded | learned `log_temperature`, annealed |

~33.5M new params (~67MB bf16), taking the model from 332M to ~366M total. **These are active
params** — the IR expert runs densely every token every loop — so re-measure the FLOP/token
estimate `TinyMoETransformer.__init__` prints and re-derive the throughput budget from it.

### Two-stage scoring

Full scoring at 65536×256 is ~137 GFLOP/loop at 8192 tokens — top-k sparsifies the *read* but not
the *scoring*, so PLAN's "roughly flat FLOPs" claim does not hold. Instead:

```
score 256 centroids            ->  pick ~4 clusters
score ~1024 keys exactly       ->  top-k = 32
gather y_values, softmax read  ->  ~1.3 GFLOP/loop
```

Differentiable through the selected entries. Cluster assignments drift as keys train — refresh them
on a fixed token cadence (checkpoint the assignment alongside the keys) and log the refresh so a
loss step at refresh boundaries is attributable. This mirrors the ANN structure used for the
external corpus, which is the point: one retrieval mechanism, two backing stores.

### Temperature

**Do not just lower the constant.** `y_values` have only ever been read as a near-uniform mixture,
so dropping temperature at inference reads out vectors that were never individually trained — you
get a loss spike, not a signal. The "free look at inference" is not informative. Anneal
`1.0 → ~0.05` *during* the finetune.

**Gate G2:** post-anneal entropy well below `ln N` **and** held-out CE not regressed.

---

## Phase 4 — Oracle evidence (the main training spend)

The key trick: before any index exists, hand the model evidence you already know is relevant — the
gold passage for QA, a held-out span from the same document for web text. Three conditions mixed
roughly evenly:

| condition | what it teaches |
|---|---|
| gold evidence | read the buffer (large, immediate CE gradient) |
| distractor evidence | don't blindly trust the buffer |
| **no evidence** | **abstain — grounded in retrieval, not memorized as a string** |

That third row is why this phase is worth doing even if the RAG project stalls: it is a principled
fix for the false-abstention collapse that doesn't depend on SQuAD-v2 ratios. Measure the
answerable-half false-abstention rate again here, separately from Phase 2's number.

### Plumbing built in this phase

**Option A — IR reads external memory.** Add `memory=(K_ext, V_ext)` to
`InformationRetrievalModule.forward`; concatenate `K = [z_keys ; K_ext]`, `V = [y_values ; V_ext]`.
Two properties fall out free: **no corpus attached → bit-identical to today** (one checkpoint serves
both modes), and **softmax mass on external vs. parametric entries is a groundedness signal** —
"I retrieved nothing relevant" becomes measurable instead of guessed.

**Option B — CrossAttention reads evidence tokens.** `other` in `transformer.py` is already a
per-call injection port re-read at every loop. Swap `_moe_ple(input_ids)` for embedded retrieved
chunks. Three concrete blockers:

- `attention.py` passes the *same* `cu_seqlens` for q and k, so `o_len` must equal `S`. Plumb
  `cu_seqlens_k` / `max_seqlen_k` (flash supports it natively) and `causal=False`.
- **RoPE**: per-chunk position basis restarting at 0 for each retrieved chunk. Within-chunk order is
  preserved (needed for span copying); cross-chunk geometry is meaningless and correctly absent.
  Build a second `cos/sin` for the evidence set.
- **Evidence read is always-on, not routed.** The router never specialized in the real run (aux loss
  pinned at its balanced value from step 0, mean routed weight flat across all 35 experts). Making
  "does this token need a fact?" depend on the weakest measured component is a bad bet. When a
  corpus is attached, seed the accumulator with the evidence read alongside `shared_mlp` /
  `shared_attn`, and gate the *content* by retrieval scores instead.

**Append-only evidence buffer.** New retrievals extend the KV set, never rewrite it. Keeps the KV
cache valid mid-generation, and hands you the accumulating buffer multi-hop needs anyway.

**Gate G3:** gold-vs-no-evidence CE gap ≥ ~0.3 nats on the answer span, and abstention rate under
no-evidence ≫ under gold.

---

## Phase 5 — Retriever alignment and the real index

### 5a. InfoNCE (Stage 3)

Train the **query side only**, document encoder frozen — this also sidesteps index staleness
entirely. Pairs mined from `phase1.bin`: context → the chunk containing the continuation. One
hard-negative mining round.

**Gate G4:** recall@k beats BM25. If it doesn't, keep B2's off-the-shelf embedder and stop training
the retriever.

### 5b. End-to-end with the Wikipedia ANN index (Stage 4)

Swap in the DPR/KILT 21M-passage Wikipedia index — it's what G5's benchmarks assume, and PopQA's
long-tail entities are Wikipedia entities by construction. `phase1.bin` overlaps training data, so
it stays a Stage-3 training resource and never backs a reported number.

**Granularity — where the cost lands:** per-token ANN during generation is infeasible;
per-sequence-at-prefill kills multi-hop. The design that works:

> **ANN retrieves k ≈ 32–64 candidates per sequence per loop. The IR module's soft, differentiable
> read over those candidates stays per-token.**

ANN cost scales with loops × sequences, not tokens, and it is exactly the two-stage retrieve/read
structure the module already implements.

### 5c. Depth curriculum (Stage 5) — the only thing that makes >3 loops pay

Loop 3 buys ~0 nats today (per-loop CE 3.109 / 2.969 / 2.969). ">3 loops" is not a config change;
later loops need a *reason* to differ, and re-executed retrieval with a moving query is that reason.
`max_enc_loops=64` and the sinusoidal loop encoding already make `forward(n_loops=8)` run today —
nothing structural blocks depth, only training does.

1. **Loop-conditioned IR query.** Zero-init per-loop bias, mirroring `loop_router_bias` exactly
   (sinusoidal in absolute loop index, clamped past the last entry). No-op at init → the checkpoint
   loads unchanged. Guarantees loop 3 doesn't re-issue loop 1's query. **Required** — Phase 1
   measured `cos(q2, q3) = 0.99`, i.e. the query stops moving after loop 1, so this is a
   precondition for the rest of this list rather than a refinement on top of it.
2. **Loop L reads the union of retrievals from loops 1..L** (the append-only buffer). Makes depth
   monotonically informative.
3. **Novelty pressure** — mask already-retrieved ids from the next loop's ANN result, or an MMR
   term. Without it three loops fetch the same top-1 three times.
4. **Extend `sample_n_loops` upward** (max 6–8) on retrieval-augmented batches, with a
   **back-loaded** `loop_ce_weights` for those batches (e.g. `[0,0,0.1,0.2,0.3,1.0]`) while plain-LM
   batches keep `[0.2,0.3,1.0]`. `loop_ce_weights_for(n)` already truncates and rescales so the
   deepest loop run carries weight 1.0. This resolves the "loop 1 reads out well vs. later loops do
   a lot" tension **per-task** instead of globally — which is the whole reason loop-index
   conditioning exists.
5. **Retrieval-utility diagnostic** — per-loop CE-with-evidence minus CE-with-evidence-zeroed. At
   minimum it says whether depth buys grounding or churn.

### 5d. RAG SFT (Stage 6)

`modules/data/chat.py` has system/user/assistant only. Evidence needs either a new segment or the
system turn — decide when you get here; a new control token is cheap (the Step 8 prune kept every
special/added token unconditionally) but changes the template for every existing SFT checkpoint.

---

## Acceptance

- **G1** — IR ablation ΔCE > ~0.02 nats (Phase 1). **Measured 2026-08-20: 0.0004 / 0.0002 nats —
  FAIL**, on the expected branch. The table is a bias term today, so Phase 3's re-init is free and
  the Phase 3 ablation is the real test.
- **G2** — post-anneal entropy ≪ `ln N`, held-out CE not regressed (Phase 3).
- **G3** — gold-vs-no-evidence CE gap ≥ ~0.3 nats; abstention under no-evidence ≫ under gold
  (Phase 4).
- **G4** — recall@k beats BM25 (Phase 5a).
- **G5** — EM/F1 on NQ-open / TriviaQA / **PopQA**, corpus attached vs. not. Then the depth
  ablation: EM at `n_loops` = 2, 3, 4, 6, 8 with corpus attached. **Flat past 3 → the depth story is
  dead. Ship 3 and don't rationalize it.**
- **G6 — the honest baseline: put the retrieved passages in the prompt as text.** If side-channel
  RAG only matches that, the architecture claim is unproven. The claim worth aiming at is the one
  in-context evidence *cannot* make: **attach far more evidence than 4096 tokens can hold, at a cost
  that doesn't grow quadratically with evidence.** That is the actual reason to do this in the
  architecture rather than the prompt.
- **P0** — head removal is behaviorally neutral (loss/top-1 within noise).
- **P2** — answerable-half false abstention well below 78.4%, precision above the ~0.50 base rate.
  **Measured 2026-08-20: 0.1358 and 0.5759 (base rate 0.5065) — PASS.** Abstention *recall* fell
  0.81 → 0.18 in exchange; that half belongs to G3's no-evidence condition, and the data lever
  (unanswerable share landed at 9.6% of QA conversations, under the 10-15% target) is untried.

## Risks

- **332M / 16B tokens is a weak reader.** But grounded *extraction* is the easiest thing to teach at
  this scale — copying beats recalling. That is the real argument that RAG suits *this* model: it
  converts a knowledge problem it can't solve into a copying problem it can.
- **The router may be using the IR slot as a bias term.** ~~7–9% routed weight against a 5.7%
  uniform share is suggestive, not evidence.~~ Settled: it is. Those figures were the *selection
  rate*; the mean routed weight is 0.025–0.037 against a 0.029 uniform share, and ablating the read
  entirely costs 0.0002–0.0004 nats.
- **Finetune-only constrains everything.** Re-initialized `down_proj`/`up_proj`/`g_proj` have to
  relearn at `lr=1e-5` against a frozen-ish trunk. If Phase 3's CE doesn't recover, the reshape
  needs either a higher LR on just those tensors (à la an IR-only finetune, freezing the rest) or a
  fresh pretrain — decide then, don't pre-commit.
- **Centroid staleness.** Two-stage scoring is only exact if assignments track the keys. Refresh
  cadence is a real hyperparameter, not a detail.
- **MTP stays on** (Plan A). If Phase 3/4 throughput or loss goes badly, the fallback is dropping it
  *for the finetunes only*: note that `lambda_mtp: 0.0` does **not** skip the compute —
  `compute_mtp_loss` gates on `mtp_outputs is not None`, so a zero weight still pays the head plus
  its chunked vocab projections at 4x. Pass `mtp_outputs=None` / skip `_mtp_forward`, keep the
  weights on disk. Independently worth fixing either way: `_mtp_forward` runs unconditionally and
  `inference.py` / `eval_abstention.py` discard the result, paid over the whole prefix every
  generated token.

---

## Parked

- **Step 13 — self-labelled calibration set.** Sample the repaired checkpoint N=16 at T=0.8 on
  short-answer QA, label by empirical pass rate, rewrite targets (>0.8 → answer, <0.2 → abstention,
  between → hedge), hold out 10%, second SFT pass. **Unparks when** Phase 2 + Phase 4 land a
  non-degenerate baseline — building it from a 78%-refusing checkpoint would re-encode the collapse.
- **Step 16 — RL.** Do not start. A 135M RLVR study on GSM8K went *backwards* under GRPO
  (24/1319 → 21/1319 → 16/1319); Qwen2.5-0.5B base stayed below 0.1 format reward after 300 steps.
  Under 0/1 reward a base model that can't sample correct solutions produces no gradient.
  **Unparks when** pass@8 on the target task exceeds ~15% on the post-Phase-4 checkpoint. If it ever
  does: vanilla GRPO is mismatched to a looped model (it credits output tokens while the computation
  is latent) — read LoopRPT (arXiv 2603.19714) and RLTT first, which assign reward to per-loop latent
  states.
- **Step 17 — reasoning training.** Gated on Step 16's pass@8. Below ~15%, reasoning training isn't
  worth attempting by any method at this size — the Small Model Learnability Gap says long-CoT
  imitation teaches fluent filler before a wrong answer. Above it, the cheap probe first (add back
  smoltalk2's `_think` splits with a distinct reasoning segment in the chat template), then
  loop-aware RLVR — which for *this* architecture is closer to the truth anyway: `n_loops` and
  `loop_scale` are the reasoning substrate here, not the token stream.
- **`num_ir_experts > 1`.** The one invasive option — shifts `first_mlp_index` and the router's
  output dim, needs a surgical remap of the router weight rather than a plain load. Only once a
  single sharpened, grown table demonstrably helps and is saturating.

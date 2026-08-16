# NEXT.md

Successor to [PLAN.md](PLAN.md) and [IR.md](IR.md) — those two stay as the record of how we got
here; this file is the plan. Read [CLAUDE.md](../../CLAUDE.md) first: it is authoritative for
everything already built.

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

## Phase 0 — Heads out

Both heads failed for structural reasons, not tuning ones (PLAN 12b-i / 12b-ii). This is a
subtraction.

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

---

## Phase 2 — Abstention repair (parallel track, independent subsystem)

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
   loads unchanged. Guarantees loop 3 doesn't re-issue loop 1's query. Mandatory if Phase 1 item 3
   found query drift ≈ 0.
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

- **G1** — IR ablation ΔCE > ~0.02 nats (Phase 1).
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

## Risks

- **332M / 16B tokens is a weak reader.** But grounded *extraction* is the easiest thing to teach at
  this scale — copying beats recalling. That is the real argument that RAG suits *this* model: it
  converts a knowledge problem it can't solve into a copying problem it can.
- **The router may be using the IR slot as a bias term.** 7–9% routed weight against a 5.7% uniform
  share is suggestive, not evidence. G1 settles it.
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

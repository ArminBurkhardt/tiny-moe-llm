# NEXT.md

The plan. The record of how we got here is [docs/CONCLUSION.md](../CONCLUSION.md) (the 16B-token
run and everything it measured); read [CLAUDE.md](../../CLAUDE.md) first, it is authoritative for
everything already built. Phases 0, 1 and 2 are **done** — their headline results are under
"Where this stands" below and the full measurement records live in
[docs/measurements/](../measurements/).

**The plan in one line:** prove every mechanism against its own ablation on a fixed benchmark
suite — reshape the IR expert, feed the IR/CrossAttention pair real external evidence, align the
retriever, attach the real index, train depth to pay — then consolidate with one full SFT plus a
preference pass, and only after that spend the real budget (a ~500M model, ≤5k H100-hours) on the
components that survived their gates.

## What the POC must prove

The POC checkpoint cannot itself be SOTA in the 300–500M class and this plan does not pretend
otherwise: 16B pretraining tokens against SmolLM2-360M's 11T is a ~700x data gap, and no finetune
closes it. What the POC can prove — and what the real run's budget is committed on the strength
of — is three claims, each with a gate:

1. **Every mechanism beats its own ablation at matched compute.** Retrieval-on vs
   retrieval-zeroed, evidence-attached vs not, depth N vs depth 3, preference pass vs none — same
   checkpoint, same eval flags, so the delta is attributable to the mechanism and nothing else
   (Gates G2–G7).
2. **The trunk is on-trend for its token budget.** The fixed benchmark suite (Phase 1b) against
   token-parity and token-rich peers separates "the architecture wastes capacity" from "it hasn't
   seen the tokens yet". The real run fixes the latter; it must not inherit the former.
3. **The evidence architecture delivers something in-context RAG cannot** (Gate G6): more
   evidence than the 4096-token context can physically hold, at a cost that does not grow
   quadratically with evidence size. At ≤5k hours the real run will still see 10–20x fewer tokens
   than its class peers — the honest pitch for SOTA-in-class is that retrieval supplies at eval
   time the knowledge peers had to store in weights. This claim *is* the thesis, which is why no
   phase below shortcuts the evidence pathway.

Every gate result lands in Phase 7's go/no-go table; a component that fails its gate does not
ship in the real run. "Flat past 3 → ship 3 and don't rationalize it" generalizes to every row.

## Rules (carried over)

- Commit per logical change (`feat:` / `docs:` / `chore:`, branch `ir-train-build`, PRs to
  `prototype`), single-line subjects.
- Lowercase explanatory comments that justify *why*; Google-style docstrings with `Args:`.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, `moe` in the training loop
  goes through `accelerator.unwrap_model(model)`.
- **Never add `.item()` / `.tolist()` / `.cpu()` / boolean mask indexing to the per-step path.**
- **Every reported number is measured at fixed eval flags.** Batch size is part of the measurement
  (the MoE's grouped GEMM tiles by per-expert row counts), and from Phase 1b on, gates are quoted
  against the measured noise floor, not as raw deltas.

## Decisions already made

| decision | choice |
|---|---|
| run shape | **Finetune the existing checkpoint for the whole POC.** No fresh pretrain before Phase 7's real run. Every shape change must stay loadable. |
| evidence port | **Option C** — IR expert = selector, CrossAttention expert = reader |
| query/key space | **B2** — external embedder (bge-small, 384-d) + adapter. Revisit B1 after 5a. |
| abstention owner | **Sequenced and measured separately.** Data repair done — lever exhausted with precision pinned at ~0.578. Phase 4's groundedness signal and Phase 6's preference pass are the two levers left. |
| depth policy | **Convergence exit + "evidence still arriving"** — no learned head, two criteria |
| IR table | **384-d (bge-small's width), 65536 entries, two stage centroid scoring**; unit-norm value rows, and the neutrality zero on `g_proj` rather than on `y_values` — see Phase 3b for why the zero moved |
| IR key init | **Random, provisionally.** Phase 3's A/B tied to four decimals and the warm start fit the clustering worse — but it ran while the read was worth 0.0002 nats, so no init difference *could* have shown. Re-open only if Phase 3b's scale fix passes |
| MTP | **Keep on** for training. Stop *computing* it where its output is discarded (Phase 1b) — a compute fix, not a removal. |
| evidence RoPE | **Per-chunk position basis restarting at 0** |
| ANN corpus | **phase1.bin for 5a InfoNCE, Wikipedia (KILT) for Stage 4/5 and all reported numbers** |
| benchmarks | **Fixed log-likelihood suite + 4 peers, frozen at Phase 1b.** Every later finetune gates on it — CE on a local slice is a health check, not a quality claim. |
| replay floor | **Every finetune carries ≥20% general LM/chat replay.** Phase 4/5 are the big spends and a narrow corpus that quietly costs trunk quality is the classic way to lose the final model. |
| final POC model | **One consolidation SFT + preference pass** (Phase 6), not the current stack of narrow repairs on an SFT that predates everything learned since. |
| real run | **500M-class, ≤5k H100-hours, target ~1k.** FP8 recipe and MFU work are validated on short runs *before* any long-run tokens are spent (Phase 7) — the 16B run left FP8 unused at MFU ~11%, and throughput multiplies the budget directly. |
| trunk objective | **AR LoopLM, POC and real run.** A masked-diffusion trunk is parked with an explicit unpark condition (see Parked): swapping the objective replaces the measurement instruments, not just the model, and its one real advantage here is already installed by 5c item 1. |

---

## Where this stands (Phases 0–2 and 1b ✅)

- **Phase 0 — heads out.** Both learned heads deleted; the halt gate's *measured* per-loop mean
  folded into `loop_scale` (`scripts/migrate_phase0.py`). Gate P0 passed on both checkpoints —
  behaviorally neutral to within noise — and `p_max` beat `p_correct` for the third time. Full
  record: [measurements/phase0_migration.md](../measurements/phase0_migration.md).
- **Phase 1 — Stage 0 diagnostics.** `scripts/eval_stage0.py` on both migrated checkpoints. Gate
  G1 **failed by ~50x**: the IR table stores nothing (entropy 99.5% of max, zeroing the read costs
  0.0002–0.0004 nats), so the Phase 3 re-init is free and the router's interest in the slot is a
  preference for a bias term. Query drift is zero after loop 1 (`cos(q2,q3) = 0.99`), making the
  loop-conditioned IR query **mandatory**. Loop 3 is redundant, not idle — nothing feeds later
  loops new information. Full record:
  [measurements/stage0_diagnostics.md](../measurements/stage0_diagnostics.md).
- **Phase 2 — abstention repair.** Gate P2 passed: false abstention 0.783 → 0.136 (0.161 at the
  0.55 retune), answerable EM tripled, the one-string collapse gone. But recall fell 0.81 → 0.22
  and **precision has now been read at ~0.578 six times while everything else moved: the corpus
  ratio slides the operating point along a fixed curve and does not bend the curve.** The missing
  quantity is discrimination, and `p_max` does not carry it (AUROC below chance on three
  checkpoints running). The data lever is exhausted. Full record:
  [measurements/abstention_repair.md](../measurements/abstention_repair.md).

- **Phase 1b — the instrument.** Thirteen-task suite validated against Pythia-410m's published
  numbers, four peers frozen, eval-sampling noise floor measured, MTP skipped where its output is
  discarded, and the three-checkpoint snapshot recorded: mean MC headroom **+0.088 / +0.081 /
  +0.084** for the trunk / SFT / repair, against gpt2-medium's +0.193. Gate G0 **passed**. The
  answerability probe reads **0.584 on all three checkpoints alike** — the trunk carries a weak
  answerability signal that no finetune moved. Records:
  [benchmark_suite.md](../measurements/benchmark_suite.md),
  [noise_floor.md](../measurements/noise_floor.md),
  [benchmark_snapshot.md](../measurements/benchmark_snapshot.md),
  [answerability_probe.md](../measurements/answerability_probe.md).

Three findings bind everything below:

1. **The loop-conditioned IR query (5c item 1) is a precondition, not a refinement** — without it,
   re-executed retrieval cannot make loop 3 differ from loop 2.
2. **The evidence read must be always-on, not routed.** The router never specialized in the real
   run (aux loss balanced from step 0); making "does this token need a fact?" depend on the
   weakest measured component is a bad bet.
3. **Abstention discrimination has to come from a new signal.** Phase 4's
   external-vs-parametric retrieval mass is the principled candidate; Phase 6's preference pass is
   the behavioral one. **1b.3 settled the third possibility**: a linear probe of the trunk reads
   0.584 identically on all three checkpoints, so the representation carries a weak signal the
   policy is not using — enough to make the preference pass an amplification rather than an
   invention, nowhere near enough to ship. Each is measured separately so the movement stays
   attributable.

---

## Phase 1b — benchmark foundation and the noise floor ✅ (Gate G0 passed 2026-08-26)

Runs **before Phase 3 touches a weight**. Phases 3–6 are trunk surgery followed by the largest
finetunes this project has run; today the only trunk-health instrument is CE on a stale local
slice. That is a health check, not a quality claim, and it cannot see a capability regression that
leaves average CE flat. This phase builds the instrument the rest of the plan reports against.

### 1b.1 The suite (`scripts/eval_benchmarks.py`) ✅

**Done 2026-08-23** — thirteen tasks, one scoring path for this model and the peers, harness
validated against Pythia-410m's published zero-shot numbers (11 of 11 anchors inside 0.4 points
against a 1.5-point tolerance) and all four peers measured and frozen. Full record:
[measurements/benchmark_suite.md](../measurements/benchmark_suite.md). The spec below is what was
built; two scoring conventions it does not mention (byte normalization excludes the joining space,
LAMBADA's published perplexity is per document) were caught by the validation and are recorded
there.

Log-likelihood multiple-choice scoring (length-normalized where that is the published convention)
plus a small generative set. **One script scores both this model and the HF peers**, so the
scoring code cannot differ between them:

- **MC, log-likelihood:** HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande, OpenBookQA, SciQ,
  BoolQ, LAMBADA (last-token), MMLU (recorded even at chance — it is the "did knowledge arrive"
  axis for the real run).
- **Generative, greedy:** TriviaQA and NQ-open closed-book EM (these two later become G5's
  corpus-attached-vs-not delta), GSM8K (expect ~0; it is Step 16's unpark metric, so it gets a
  standing measurement rather than a guess).
- Fixed flags throughout: batch size, BF16, prompt formats frozen in-repo.
- **Validate the harness before trusting it:** reproduce one peer's published numbers to within
  ~1 point. A harness that can't reproduce known numbers produces unknown numbers.

Peers, chosen to bracket the token axis: **gpt2-medium** (355M, the nearest token-parity anchor),
**Pythia-410M** (300B tokens, the scaling-trend anchor), **SmolLM2-360M** (**4T**, not 11T as
written here originally — 11T is the 1.7B model; the data ceiling either way), **Qwen2.5-0.5B**
(18T, the practical upper bound of the class). Peer numbers are measured once and frozen with the
suite.

### 1b.2 The noise floor — slice noise done, seed noise open

Gates need honest thresholds more than they need ambition. Two cheap measurements, written to
[`docs/measurements/noise_floor.md`](../measurements/noise_floor.md):

- **Seed noise:** re-run the 0.55 repair finetune with a different `sft.seed` (identical corpus,
  ~22 min) and re-run the abstention eval. The per-metric spread between the two finals is the
  training-seed σ on every abstention metric. **Not run.**
- **Slice noise:** the same checkpoint on SQuAD v2 validation questions 2000–4000. The spread
  against the standard slice is the eval-sampling σ. ✅ **Done 2026-08-23** via
  `eval_abstention.py --example-offset`. The decision metrics move ≤0.004 and are dominated by
  their own binomial error (precision ±0.026 on ~370 abstentions); answerable-half EM moves 0.015;
  AUROC moves 0.029 and crosses chance, so no argument may rest on the abstention signal being
  specifically *below* 0.5. The standard slice reproduced the recorded 0.55 numbers to four
  decimals on every metric.

From here on, a gate that says "must not regress" means "within the documented noise", and a gate
that says "must improve" means "by ≥3σ".

### 1b.3 The answerability probe ✅

**Done 2026-08-26** — `scripts/eval_probe.py`, run on all three checkpoints rather than only the
repaired one, which is what made the result interpretable. Full record:
[measurements/answerability_probe.md](../measurements/answerability_probe.md).

Neither branch below fired cleanly. **The hidden-state probe reads AUROC 0.5804 / 0.5850 / 0.5839 on
the pretrained trunk, the SFT checkpoint and the repair checkpoint** — agreeing to within 0.005, an
order of magnitude inside the ±0.030 slice noise. So the signal is real (it beats the best free
scalar by 0.032–0.060, outside that noise), it is weak (0.58, not 0.7), and **it is a property of
the pretrained trunk that 758M tokens of finetuning did not move in either direction.** That is the
representation-level form of "the corpus ratio slides the operating point along a fixed curve": the
finetunes were re-reading a fixed representation, not changing it. `p_max`, entropy and the top-1
margin carry nothing on any checkpoint, fitted or not (0.454–0.548) — the fourth independent reading
of that.

Consequences, which the phases below now assume:

- **Phase 4 keeps the burden.** Three checkpoints of finetuning put no discrimination into the
  trunk, so it has to come from information the trunk does not have.
- **G3b's comparison changes, its threshold does not.** "Beat `p_max`'s 0.462–0.478" was a
  chance-level bar. The real bar is **0.584** — what a free linear read of the existing trunk
  already gives — and G3b's 0.65 clears it by 0.066, so the gate stands as written, now against an
  honest baseline.
- **Phase 6 is asked for something easier than it was.** The preference pass no longer has to
  invent a distinction the model cannot represent; it has to make the policy use one that
  measurably exists. At matched recall a probe threshold is already 0.13 more precise than the
  shipped policy at half its false-abstention rate — a better-shaped curve that is still not bent
  (precision at recall 0.55 is 0.557, barely over the 0.507 base rate).

The branches this was run to decide, kept for the record:

features = final-loop last-position hidden state plus the free scalars (`p_max`, entropy, top-1
margin); target = answerable/unanswerable, AUROC read on the standard 2,000-question slice.

- **AUROC ≈ 0.5** → the signal is not in the trunk's representation at all. Only new information
  (Phase 4's evidence) can create it; no readout trick will.
- **AUROC ≥ ~0.7** → it is a *readout* problem: the trunk knows, the policy doesn't. Then a
  thresholded probe is a legitimate shipping mechanism (tunable operating point, minutes to
  refit), and Phase 6's preference pass has something real to amplify.

### 1b.4 Stop paying for MTP where it is discarded ✅

**Done 2026-08-26** — `TinyMoETransformer.forward` takes `skip_mtp`, which skips the head and drops
`extra_token_outputs` from the return; `inference.py` (when not drafting), `eval_abstention.py`,
`eval_benchmarks.py`, `eval_calibration.py` and `eval_stage0.py` all pass it.
`tests/test_mtp_skip.py` is the structural check this item asked for: logits, per-loop hidden states
and every step of a KV-cached decode are **bit-identical** with the head on and off — equality, not
a tolerance, since a tolerance would hide exactly the bug it checks for.

`_mtp_forward` used to run unconditionally while those callers threw the result away — paid over the
whole prefix, on every generated token, in every generative eval this plan runs dozens of times.

### 1b.5 Baseline snapshot ✅

**Done 2026-08-26** — all three checkpoints through all thirteen tasks at the frozen flags. Full
record: [measurements/benchmark_snapshot.md](../measurements/benchmark_snapshot.md).

Mean MC headroom **+0.088 / +0.081 / +0.084** (pretrained trunk / SFT / repair @ 0.55) against
gpt2-medium's +0.193, Pythia-410M's +0.208 and SmolLM2-360M's +0.345. Three readings that bind
what follows:

- **758M tokens of post-training moved the benchmark position by 0.007 of headroom**, and the
  repair column sits *above* the SFT one. The narrow finetunes cost no general capability and
  bought none — the same "policy change on a fixed representation" the probe found one level down.
- **Below gpt2-medium at ~half its headroom** is the honest starting position, and nothing in the
  suite separates "wastes capacity" from "hasn't seen the tokens". That separation is exactly what
  Phases 3–6 buy by running each mechanism against its own ablation at matched compute.
- **BoolQ regressed monotonically under post-training** (0.4599 → 0.4416 → 0.4190 `acc`, all below
  the 0.50 chance of a two-option task; `acc_norm` on the same scores reads 0.6162 → 0.5920). A
  yes/no answer-policy bias that deepened under the finetunes teaching the model to decline, and
  invisible to average CE. Re-read it after Phase 4, whose no-evidence condition trains a different
  decline behaviour.

TriviaQA 0.000 → 0.014 and NQ-open 0.000 → 0.002 are now the recorded floor G5's corpus-attached
delta is measured against; MMLU sits at chance on every column, as does GSM8K (0.018 / 0.005 /
0.009), so Step 16 stays parked.

The repair column also reproduced the 2026-08-23 shakedown **to four decimals on all thirteen
tasks** while running in 7.3 min against 36.0 — 1b.4's `skip_mtp` confirmed free at full scale, not
just in the unit test.

**Gate G0: PASS.** Harness reproduced Pythia-410m on 11 of 11 published anchors inside 0.4 points
(1b.1); noise floor documented for the eval-sampling half, with the training-seed half still open
(1b.2); three-checkpoint snapshot recorded (1b.5).

---

## Phase 3 — IR expert reshape and sharpening

The table holds no information (Stage 0), so re-initializing it costs nothing. Do the reshape,
the temperature anneal, and the init A/B in one short phase.

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

### Init: behaviorally neutral at step 0, keys A/B'd

- **`y_values` zero-init, both arms.** A zero value table makes the read contribute nothing
  regardless of the softmax, so the reshaped checkpoint scores *identically* to the migrated seed
  at load — the same zero-init-neutrality pattern as `loop_router_bias`, and it makes the
  post-reshape baseline free instead of a separate measurement.
- **Arm A: random `z_keys`** (current init scheme).
- **Arm B: warm-started `z_keys`** — 65,536 diverse chunks sampled from the phase-1 mix, embedded
  with bge-small, mapped 384→256 through a linear adapter that trains with the run. The hypothesis
  is that semantically pre-clustered keys give the temperature anneal somewhere to attach;
  the A/B is two ~150M-token finetunes (~1.5h each on the 5090), which is cheap enough that
  guessing would be the expensive option — this choice propagates into the real run.

### LR: per-tensor groups, not a compromise

The old risk note said re-initialized tensors at `lr=1e-5` against a frozen-ish trunk may never
recover, "decide then". Decided now, properly: **two LR groups** via the existing
`build_param_groups` mechanism — new tensors (`z_keys`, `y_values`, projections, centroids,
`log_temperature`, the arm-B adapter) at ~3e-4 with their own warmup, everything else at 1e-5.
Fresh params get a fresh-param LR; the trunk gets a finetune LR. No third option needed.

### Two-stage scoring

Full scoring at 65536×256 is ~137 GFLOP/loop at 8192 tokens — top-k sparsifies the *read* but not
the *scoring*. Instead:

```
score 256 centroids            ->  pick ~4 clusters
score ~1024 keys exactly       ->  top-k = 32
gather y_values, softmax read  ->  ~1.3 GFLOP/loop
```

Differentiable through the selected entries. Cluster assignments drift as keys train — refresh
them on a fixed token cadence (checkpoint the assignment alongside the keys) and log the refresh
so a loss step at refresh boundaries is attributable. This mirrors the ANN structure used for the
external corpus, which is the point: one retrieval mechanism, two backing stores.

Clustering decisions, made now rather than at implementation time:

- **Spherical k-means with a balance cap.** Scoring is cosine over L2-normalized keys, so
  centroids are means of normalized members, renormalized — plain Euclidean Lloyd's optimizes the
  wrong metric. Cap cluster size at ~2x the mean during refresh: unbalanced clusters make the
  probe cost per token variable and leave whole key regions unreachable, i.e. untrainable.
- **Measure candidate recall at every refresh.** Recall@32 of the two-stage path against exact
  full scoring on a fixed query sample — one cheap matmul. Below ~0.9, raise the probed clusters
  from 4 before blaming the training; a centroid path that silently misses the true top-k turns
  the anneal into noise.
- **Recycle dead entries at refresh.** A sharpened softmax trains only what it selects; entries
  whose selection EMA stays ~0 over a refresh window get their key re-seeded to a recent
  underserved query and their value zeroed — the same zero-init neutrality pattern as everything
  else here. The per-entry EMA rides the existing tracker machinery.
- **Fallback, named now: product-key memory.** Two 256-entry half-key codebooks give *exact*
  top-k over all 65536 composed keys at 2×256 scoring cost, fully differentiable, no centroids,
  no refresh, no staleness (Lample et al. 2019; PEER 2024). It is the fallback rather than the
  default only because it cannot back the external corpus (FAISS needs real centroids) and one
  shared mechanism is the design bet. If G2's dynamics show refresh churn or cluster collapse,
  PKM is the measured alternative for the parametric table — a swap, not a redesign.

If arm B wins, one refinement worth its own short measurement: a lower LR on `z_keys` than on
`y_values`, so the key space stays a semantic index (what the warm start bought) while the values
learn content — key drift erasing the warm start would show up as the B arm converging to A.

### Temperature

**Do not just lower the constant.** `y_values` have only ever been read as a near-uniform
mixture, so dropping temperature at inference reads out vectors that were never individually
trained — a loss spike, not a signal. Anneal `1.0 → ~0.05` *during* the finetune.

### Corpus

~150–200M tokens per arm, replay-shaped: ~70% phase-2-style LM mix, ~30% chat/QA replay. This is
a pretraining-style finetune whose job is to give the sharpening table general text to store —
not another QA-shaped repair.

**Gate G2:** post-anneal retrieval entropy well below `ln 65536 = 11.09` with top-32 mass far off
uniform; held-out CE and the benchmark suite within noise of the seed; the winning arm chosen on
those numbers plus sharpening dynamics. **Also re-measure the read-zeroed ablation** and record
it: it is *not* required to clear G1's 0.02-nat bar yet — a table trained only by LM CE may
legitimately stay near-bias — but it must move well off the 0.0004-nat floor, and the 0.02 bar
comes due after Phase 4/5 has given the mechanism something to retrieve. **Branch, stated now:**
if the parametric table still ablates to ~0 after Phases 4–5, its size is frozen out of the real
run spec — external memory then carries the mechanism, and the model does not pay 33M params for
a bias term.

### Outcome (2026-08-29) — measured in [ir_sharpening.md](../measurements/ir_sharpening.md)

Both arms ran at 208M tokens off the same seed, ~72 min each on the 5090.
[ir_reshape.md](../measurements/ir_reshape.md) covers the reshape itself: total params 332M → 364M,
forward FLOP/token 490M → **484M** (the two stage read is cheaper than the old exact read of a table
an eighth the size), and the migrated checkpoint confirmed behaviourally identical to its source
(+0.0002 nats, i.e. exactly what deleting the read costs). `ir_probe_clusters` is **8**, not the 4
guessed above — at 4, candidate recall@32 sat at 0.86–0.95 and drifted down as the anneal sharpened.

**Gate G2: FAIL**, on the ablation, on both arms.

- Retrieval entropy did fall: loop 1 reads `E / ln 32 = 0.911` (A) / `0.923` (B) with a max weight
  4.4–4.6x uniform. Loops 2 and 3 stayed at 0.984–0.987, i.e. uniform — the same split the query
  drift matrix shows, `cos(q1, q2) ≈ 0.32–0.37` against `cos(q2, q3) ≈ 0.98`. The table can only be
  sharp for a query that varies, and the recurrence asks one question and then repeats it.
- Held-out CE and the benchmark suite stayed within a finetune's worth of the seed (per-loop CE
  3.3843 → 3.4109; mean MC headroom +0.088 → +0.072 / +0.076, with BoolQ again the outlier at
  −4.6 / −5.0 points).
- **The read-zeroed ablation did not move off the floor: 0.0002 nats, the same number the untrained
  seed measured.** Per-loop CE is identical whether the trained checkpoint reads at temperature
  scale 1.0 or the annealed 0.05, which is only possible if the retrieved value reaches the logits
  with ~zero weight.
- **The A/B is a tie**, on CE to four decimals, on ablation content, and on routed weight per loop.
  Arm A (random keys) is the arm to keep, on the tiebreak that its clustering fit better — arm B
  tripped the recall warning three times out of ten refreshes, arm A never. **Read the tie as
  uninformative rather than as a fact about key geometry**: Phase 3b found the read contributes
  ~1e-3 of the residual on both arms, and an experiment in that regime cannot detect an init
  difference. The "lower LR on `z_keys` than `y_values`" refinement above is parked, not refuted.

**Phase 3b supersedes the diagnosis.** The bottleneck is not the model declining to use a good read;
the read arrives ~250x below the query because the value side is never normalized, and the same
attenuation is present in the pre-reshape trunk after 16B tokens. Nothing here recommends the
product-key fallback either — refresh churn and cluster collapse were not the failure mode.

---

## Phase 3b — The read never reached a usable scale

Self-contained. Phase 3 concluded the read was worth 0.0002 nats and left *why* open; a pathway
probe answered it, and the answer is a scale bug rather than a verdict on the mechanism.

### What was measured

RMS over tokens, along the IR pathway, on both arms and on the pre-reshape trunk:

| stage | arm A | arm B | pre-reshape trunk (16B tokens) |
|---|---|---|---|
| `x_norm` (the residual) | 0.992 | 0.992 | 0.991 |
| `down` (the query) | 0.83–1.35 | 0.76–1.21 | 1.28–2.08 |
| **`retrieved_y`** | **0.0034–0.0044** | **0.0032–0.0049** | **0.0020–0.0043** |
| `information` (the expert's K/V) | 0.004–0.006 | 0.003–0.008 | 0.001–0.003 |
| `expert_out` | 0.026–0.094 | 0.025–0.104 | 0.017–0.069 |

The read arrives ~250x below the query it was issued from, so the IR expert puts 3e-4 to 1.6e-3 of
magnitude into a unit-RMS residual. Zeroing it changes the expert's output by exactly 100% on every
loop — nothing is gating it off, there is nothing there to gate.

**The cause is that the value side is never normalized.** The query is `F.normalize`d and scored by
cosine; the read is a convex combination of value rows, so `‖retrieved_y‖ ≤ max‖y_row‖ ≈ 0.05`, and
a near-uniform read over 32 near-orthogonal rows divides that by another ~`√32`.

Three facts that follow, each of which changes an earlier conclusion:

- **This is not a "grafted on too late" problem.** The pre-reshape table had the IR expert present
  for all 16B pretraining tokens and shows the same 0.002–0.004 read magnitude — and its `y_values`
  RMS is **0.0204**, still exactly its `randn * 0.02` init. The mechanism has been suppressed the
  whole time; it was never actually tested.
- **The table did learn, at the wrong rate for the wrong reason.** Both arms took `y_values` from
  exactly 0 to row norm 0.0507 in 208M tokens, so `fresh_lr` worked. But per-element movement was
  0.0037 against 0.034 for a pure random walk at that step size — the steps cancel. Entries are
  selected by a near-uniform read, so each receives averaged gradient from unrelated contexts, and
  weight decay (`y_values` is `ndim=2`, so it is in the decayed group) reels in what has no
  consistent direction. Selectivity and content are each waiting on the other.
- **The gradient that teaches selectivity is proportional to how much the read moves the loss**, so
  at 1e-3 of the residual the keys receive no supervision worth accumulating: `z_keys` relative
  gradient 4.6e-6 against the trunk's 7.0e-4, 150x weaker, and `z_rms` unmoved from its init after
  16B tokens. **This is why Phase 3's key-init A/B tied** — it ran in a regime where no init
  difference could have been detected. That tie is evidence about the scale bug, not about key
  geometry.

**The router weight is not part of this.** 1 self-attn + 1 cross-attn + 1 IR + 32 MLP at `top_k: 2`
makes uniform selection 2/35 = 5.7%; IR measures 5.7% / 11.8% / 8.6% and takes 0.35–0.48 of the gate
when selected. A mean routed weight of 0.02–0.05 is what the arithmetic produces for any expert in
this pool, not a router that learned to avoid retrieval.

### The changes

Ordered by whether they preserve loadability, which is what decides where they go.

**1. Move the neutrality zero from the values to the output projection.** Zero-initializing
`y_values` puts the zero on the *content* — 33.5M parameters that must travel ~250x in norm, under
weight decay, with no pressure to make the trip. Zero-initializing `g_proj.weight` instead is the
standard residual-branch pattern (ReZero / LayerScale): migration neutrality is preserved exactly,
but the zero now sits on a `ir_dim x ir_dim` tensor that trains fast at `fresh_lr`. Note
`∂L/∂y_values` is
zero at step 0 because it flows through `g_proj` — `g_proj` moves first and the table learns behind
it, so **log `‖g_proj‖` early to confirm it leaves zero** rather than assuming it.

**2. Unit-normalize the value rows, and initialize each cluster to a rotated orthonormal set.**
Normalizing in the forward makes the table store *directions* and removes magnitude as a
degree of freedom weight decay can eat; read magnitude then lands in `[1/√k, 1]` — ~0.177 for a
uniform read over 32 rows, 1.0 for a peaked one — which is the confidence signal at a usable scale,
with no constant to tune. It is also symmetric with the key side, which is already normalized by
construction, and it is the convention the external corpus path arrives in anyway (bge outputs are
L2-normalized), so one rule covers both backing stores.

For the init: each cluster holds `65536 / 256 = 256` entries, so seed each one with **256 mutually
orthonormal vectors, rotated per cluster** (one QR per cluster at migration — constructive, not an
iterative repulsion). Within a probed cluster the candidates are then exactly mutually orthogonal,
which is maximum discriminability where the read actually looks, and the per-cluster rotation keeps
cross-cluster correlation low. Random unit vectors are already near-optimal *on average* in a few
hundred dimensions (`E|cos| = 1/√d`); this buys the tail, which is the part that matters among
candidates that were selected for being close.

**3. `ir_dim` 256 → 384.** The reshape rebuilds these tensors anyway, so the width is free to
revisit exactly once, here. **384 is bge-small's native dimension**, and the evidence path is
already committed to bge-small plus an adapter — at 256 that adapter compresses 384 → 256 and
discards information before the table ever sees it; at 384 it is a rotation. Neither cost binds:
+16.8M parameters (33.5M → 50.3M table, 364M → 381M total), ~+0.3GB peak against a budget that only
spills past ~29.6GB, and ~2.5M FLOP/token/loop against 1.72M in a 484M-FLOP model.

1:1 (768) is the alternative and it is worse: 67M extra parameters, 18% of the model, to *upsample*
a 384-d embedding into a width carrying no information the external path has. It buys only geometry
(`E|cos|` 0.036 vs 0.051), which is not what is broken.

One casualty: at 256 the identity `num_ir_entries / ir_num_clusters = ir_dim` made each cluster
exactly one orthonormal basis. At 384 a cluster seeds 256 orthonormal vectors into a 384-d space —
still exactly orthogonal, with slack, so better conditioned and less tidy.

**4. Independent gates for the non-MLP experts — deferred to the real run.** Non-MLP experts run
unconditionally (attention must see the whole sequence), so for IR the top-k mask buys **zero
compute** and costs ~17x in signal. Underneath that, one softmax over a heterogeneous pool makes a
memory read compete for probability mass against 32 MLP experts it is not a substitute for; the
codebase already concedes the point by keeping `shared_mlp` / `shared_attn` outside the router. The
consistent design is a per-expert sigmoid gate for the non-MLP slots and the top-k softmax only over
the MLP experts, where sparsity actually saves FLOPs.

It is **not** loadable-neutral — it changes all three non-MLP experts' contribution for every token
and the trunk has to re-equilibrate — and it is not yet justified. Arm C decides: if the read starts
contributing and the router raises IR's selection rate on its own, the gate structure was never
binding. If the read contributes when selected but selection stays pinned at uniform, that is the
evidence, and it belongs in Phase 7's architecture rather than bolted onto this checkpoint.

### Arm C

Changes 1–3, all of which preserve loadability: the migrated checkpoint scores identically to its
source at step 0 (`g_proj = 0`), so any movement in the ablation is real. Same 208M corpus and
settings as arms A and B: ~72 min plus ~15 min of evals.

It tests the scale fix and the width together, which is a confound worth naming. It is accepted
because the risk is asymmetric — a wider table has more capacity and better key separation under the
same scale fix, so it cannot plausibly *cause* a failure. What is lost is the ability to attribute
credit if it passes, which nothing downstream needs.

**Staged, with an early kill.** Read the ablation at 50M tokens (~18 min). Off the floor → run to
208M and take the full gate. Still 0.0002 → stop, and the finding is that the scale was not the
binding constraint after all, which is worth more than the remaining 54 minutes.

**Gate G2b:** final-loop `dCE` with the read zeroed **> 0.01 nats** — half of G1's bar, since LM CE
alone is not what the table is ultimately for, but 50x off the floor it has sat at through 16B + 208M
tokens. Held-out CE and the benchmark suite within a finetune's noise of the seed, as in G2. Report
IR selection rate per loop, `‖g_proj‖` against its zero init, and `‖y_values‖`, because those three
say *which* of the changes did the work.

If G2b passes, the parametric table is live and Phase 4 inherits a mechanism that works. If it
fails, Phase 3's branch stands: the table's size is frozen out of the real run spec and external
memory carries the mechanism alone.

---

## Phase 4 — Oracle evidence (the main POC training spend)

The key trick: before any index exists, hand the model evidence you already know is relevant —
the gold passage for QA, a held-out span from the same document for web text. Four conditions
(the old plan's three, plus the one real retrieval actually produces):

| condition | what it teaches |
|---|---|
| gold evidence | read the buffer (large, immediate CE gradient) |
| **gold among distractors** | **select before reading — the condition real retrieval delivers** |
| distractors only | don't trust the buffer; the answer isn't in it |
| **no evidence** | **abstain — grounded in retrieval, not memorized as a string** |

The fourth row is why this phase is worth doing even if the RAG project stalls: it is the
principled fix for the abstention discrimination gap that Phase 2 proved no data ratio can
supply. Measure its effect separately from Phase 2's numbers.

### Sources

- **SQuAD v2** — the paragraph is gold for answerable rows; for unanswerable rows it is
  distractor-shaped *by construction* (relevant, non-answering). The dataset is the condition
  table in miniature.
- **HotpotQA (distractor setting)** — 2 gold + 8 distractor paragraphs per question: multi-hop
  selection, exactly the gold-among-distractors condition.
- **Plain web text** from the phase-2-style mix with a held-out same-document span attached as
  pseudo-gold — evidence-conditioned *language modeling*, so the mechanism generalizes past QA
  formatting.
- **≥20% general chat/LM replay with no evidence attached.** With no corpus attached the forward
  pass is bit-identical to today (Option A's defining property), so replay genuinely protects the
  trunk rather than training the port.

### Plumbing built in this phase

**Option A — IR reads external memory.** Add `memory=(K_ext, V_ext)` to
`InformationRetrievalModule.forward`; concatenate `K = [z_keys ; K_ext]`,
`V = [y_values ; V_ext]`. `K_ext`/`V_ext` come from the external embedder through two new
384→256 adapters — the V-adapter zero-init so attaching an empty or irrelevant memory is neutral
at step 0. Two properties fall out free: **no corpus attached → bit-identical to today** (one
checkpoint serves both modes), and **softmax mass on external vs. parametric entries is a
groundedness signal** — "I retrieved nothing relevant" becomes measurable instead of guessed.
One added tensor, decided now: a **learned per-source logit scale on the external half** of the
concatenated scores. Post-anneal `z_keys` and adapter-mapped bge keys land at different logit
scales, and without the scale the mass split — the G3b signal itself — measures that mismatch
rather than relevance. Init to match the parametric scale; G3b is read after it.

**Option B — CrossAttention reads evidence tokens.** `other` in `transformer.py` is already a
per-call injection port re-read at every loop. Swap `_moe_ple(input_ids)` for embedded retrieved
chunks. Three concrete blockers:

- `attention.py` passes the *same* `cu_seqlens` for q and k, so `o_len` must equal `S`. Plumb
  `cu_seqlens_k` / `max_seqlen_k` (flash supports it natively) and `causal=False`.
- **RoPE**: per-chunk position basis restarting at 0 for each retrieved chunk. Within-chunk order
  is preserved (needed for span copying); cross-chunk geometry is meaningless and correctly
  absent. Build a second `cos/sin` for the evidence set.
- **Evidence read is always-on, not routed.** The router never specialized in the real run.
  When a corpus is attached, seed the accumulator with the evidence read alongside `shared_mlp` /
  `shared_attn`, and gate the *content* by retrieval scores instead.

**Append-only evidence buffer.** New retrievals extend the KV set, never rewrite it. Keeps the KV
cache valid mid-generation, hands multi-hop its accumulating state, and finally makes 0c's
"evidence still arriving" depth criterion implementable.

### The eval flip

`eval_abstention.py` gains an `--evidence-port` mode: the passage comes *out of the prompt* and
in through the evidence port, so the eval measures the port doing the reading rather than
in-context attention. The in-prompt form is kept unchanged — it is G6's baseline — and both run
at the standard flags.

### Size and schedule

300–500M tokens mixed across the four conditions plus replay, local on the 5090 (4 × 4096 ×
accum 4, BF16), ~4–7h. Trunk at `lr=1e-5`; the new adapters get the fresh-param LR group from
Phase 3. Conversation weighting on (it is what fixed the last policy collapse).

**Gate G3:** gold-vs-no-evidence CE gap ≥ ~0.3 nats on the answer span; abstention rate under
no-evidence ≫ under gold; benchmark suite within noise.

**Gate G3b (the discrimination gate):** AUROC of the external-mass groundedness signal for
detecting unanswerable questions **≥ 0.65** on the standard SQuAD v2 slice. The bar to beat is
**not** `p_max`'s 0.462–0.478 (that is chance) but the **0.584** a free linear probe of the
existing trunk already reads (1b.3) — 0.65 clears it by 0.066, which is why the threshold stands. First reading of the bent curve: precision above
0.578 by ≥3σ at recall ≥ 0.5 (the hard bar lands at G7). If the groundedness signal *also* fails
to discriminate, Phase 6's probe + preference pass is the only lever left and the Phase 7
go/no-go table says so explicitly.

---

## Phase 5 — Retriever alignment and the real index

### 5a. InfoNCE (Stage 3)

Train the **query side only**, document encoder frozen — this sidesteps index staleness entirely.
The query adapter maps the model's hidden state into bge space; pairs mined from `phase1.bin`:
context → the chunk containing the continuation. One hard-negative mining round.

**Gate G4:** recall@k must beat **both** BM25 **and** the no-training baseline (bge-small
embedding of the context tail as the query). Beating BM25 alone is the old bar; if training the
adapter can't beat the frozen embedder it was initialized from, the training added nothing — keep
B2 off-the-shelf and stop training the retriever. **Gated extension, not default:** if G4 passes
but the margin is thin, unfreeze the document side with periodic index refresh (embedding 21M
passages with bge-small is hours, not days, on the 5090) — staleness machinery is real
complexity, bought only if the measured margin says the query side is the bottleneck.

### 5b. End-to-end with the Wikipedia ANN index (Stage 4)

Swap in the KILT/DPR 21M-passage Wikipedia index — it is what G5's benchmarks assume, and PopQA's
long-tail entities are Wikipedia entities by construction. `phase1.bin` overlaps training data,
so it stays a Stage-3 training resource and never backs a reported number.

Index build: bge-small 384-d embeddings, FAISS IVF-PQ (raw fp16 is ~16GB; PQ brings it to a size
that coexists with the model on the 5090). **Measure recall@64 of the ANN against exact search
once** — if quantization costs more than ~2 points of recall, loosen the PQ before blaming the
model for retrieval misses.

**Granularity — where the cost lands:** per-token ANN during generation is infeasible;
per-sequence-at-prefill kills multi-hop. The design that works:

> **ANN retrieves k ≈ 32–64 candidates per sequence per loop. The IR module's soft,
> differentiable read over those candidates stays per-token.**

ANN cost scales with loops × sequences, not tokens, and it is exactly the two-stage
retrieve/read structure the module already implements.

**Key granularity vs. reader granularity — the needle decision.** A single 384-d vector is
faithful to ~100–300 tokens; a needle sentence inside a 4096-token chunk dilutes out of the
chunk's pooled embedding, and no later stage can recover a candidate the ANN never surfaced. So
the two granularities are decoupled: **the ANN indexes fine keys** (sentence windows, ~32–128
tokens — KILT's 100-word passages already sit in this band), each mapped to its parent chunk,
and **the reader gets the parent chunk's tokens** around the hit. A chunk's score is its best
sentence hit (late-interaction MaxSim, structurally), so the needle's own key is what gets
found. This is the two-stage shape from the parametric table again — chunk as cluster, its
sentence keys as members — and the IR expert's exact stage scores the candidates' precomputed
fine keys (~64 chunks × ~8 sentences per loop, trivial) in the same softmax as the parametric
candidates, behind the per-source scale; nothing about the learned table changes. Two
boundaries, stated now: verbatim needles are lexical, not semantic, so the ANN candidates are
**unioned with BM25's** rather than asked to beat them (G4 keeps BM25 as the bar on the semantic
side only); and **no learned encoder touches raw chunk tokens at selection time** — that is a
cross-encoder, it cannot scale past the candidate set, and extracting the relevant tokens is
already the reader's job. The learned surface stays the query adapter and the logit scale.

### 5c. Depth curriculum (Stage 5) — the only thing that makes >3 loops pay

Loop 3 buys ~0 nats today. ">3 loops" is not a config change; later loops need a *reason* to
differ, and re-executed retrieval with a moving query is that reason. `max_enc_loops=64` and the
sinusoidal loop encoding already make `forward(n_loops=8)` run today — nothing structural blocks
depth, only training does.

1. **Loop-conditioned IR query.** Zero-init per-loop bias, mirroring `loop_router_bias` exactly
   (sinusoidal in absolute loop index, clamped past the last entry). No-op at init → the
   checkpoint loads unchanged. **Required** — Stage 0 measured `cos(q2, q3) = 0.99`: the query
   stops moving after loop 1, so this is a precondition for the rest of this list.
2. **Loop L reads the union of retrievals from loops 1..L** (the append-only buffer). Makes depth
   monotonically informative.
3. **Novelty pressure** — mask already-retrieved ids from the next loop's ANN result, or an MMR
   term. Without it three loops fetch the same top-1 three times.
4. **Extend `sample_n_loops` upward** (max 6–8) on retrieval-augmented batches, with a
   **back-loaded** `loop_ce_weights` for those batches (e.g. `[0,0,0.1,0.2,0.3,1.0]`) while
   plain-LM batches keep `[0.2,0.3,1.0]`. `loop_ce_weights_for(n)` already truncates and rescales
   so the deepest loop run carries weight 1.0. This resolves the "loop 1 reads out well vs. later
   loops do a lot" tension **per-task** instead of globally — which is the whole reason
   loop-index conditioning exists.
5. **Retrieval-utility diagnostic** — per-loop CE-with-evidence minus CE-with-evidence-zeroed. At
   minimum it says whether depth buys grounding or churn.

### 5d. RAG SFT data and the template (Stage 6)

Decide the template mechanism **now**, not "when you get here": evidence gets its **own control
token / segment** (the vocab prune kept every special/added token unconditionally, so the token
is free) rather than being smuggled through the system turn — a distinct segment is what lets the
loss mask, the eval prompts and the groundedness measurement all address evidence unambiguously.
The template change invalidates existing SFT checkpoints' chat formatting, and that is fine
*because Phase 6 re-runs the full SFT anyway* — this is the one moment in the plan where a
template migration is free. Build the RAG SFT corpus (evidence-attached conversations, including
abstention-with-empty-retrieval rows) here; Phase 6 consumes it.

### 5e. The needle eval (what "retrieval at this scale" is allowed to claim)

Plant one unique fact in one of N attached chunks (N = 4, 16, 64, 256; distractors drawn from the
same document distribution) and ask for it. Report **two numbers per N, never one**: selector hit
rate (the needle chunk carries the top read mass) and end-to-end EM — so a miss is attributable
to selection vs. reading, the same stage separation the whole architecture is built on. This is
G6's beyond-context claim in eval form; it runs at the standard flags and its N=64+ rows are
quoted alongside G6.

The honest scope, stated once: **the needle lives in the evidence pathway, not the parametric
table.** ANN recall gets the chunk into the buffer, the IR read ranks it, CrossAttention copies
from its actual tokens — copying is the operation a 332M model can do reliably. A 256-d value
cannot store a verbatim fact; 65536 of them are a learned cache of frequent atoms, not a document
store, and no gate asks the table to pass a needle test. "RAG over more tokens than any context
window holds" is a claim about the buffer + reader, and this eval is what licenses it.

**Gate G5:** EM/F1 on NQ-open / TriviaQA / **PopQA**, corpus attached vs. not — the attach delta
must be ≥3σ. Then the depth ablation: EM at `n_loops` = 2, 3, 4, 6, 8 with corpus attached.
**Flat past 3 → the depth story is dead. Ship 3 and don't rationalize it.**

**Gate G6 — the honest baseline: put the retrieved passages in the prompt as text.** If
side-channel RAG only matches that, the architecture claim is unproven. The claim worth aiming at
is the one in-context evidence *cannot* make: attach 64+ chunks (8–16k tokens of evidence)
against the best 4096-token in-prompt packing, at a cost that grows linearly in evidence. The
side channel must win where in-context physically can't play.

---

## Phase 6 — Consolidation SFT and the preference pass

The current "final model" is a 708.9M-token SFT that predates every finding since, patched by a
49M-token repair. Highest-quality means one clean rebuild from the best trunk, not a stack of
narrow repairs.

### 6a. Full SFT rebuild

From the post-Phase-5 trunk, with everything learned baked into one corpus: the 0.55 unanswerable
ratio, the answerable extractive QA sources, the 15 training phrasings, the RAG SFT data from 5d
under the new evidence segment, the general chat mix as before, and the smoltalk2 holdout
honored as always. **Conversation-loss weighting is measured at full scale, not assumed:** the
original SFT ran without it, the repair with it, and they were never compared on the same corpus.
Two runs (~6h each local), pick by the gates. Same fp32-master machinery, same
`pretrain.train_step`.

### 6b. Preference pass (DPO) — the tool CE structurally lacks

Twice now, per-token CE failed to see a *policy-level* failure it was creating (the refusal
collapse, then the over-answering). A pairwise objective sees exactly that, and this is its
standard use case. On the 6a checkpoint:

- **Pairs:** answerable → (correct extractive answer ≻ abstention); unanswerable → (abstention ≻
  hallucinated answer). Mined from the model's own samples (N=16, T=0.8, labeled by empirical
  pass rate — this is parked Step 13's machinery, unparked here now that the baseline is
  non-degenerate) plus gold-constructed pairs. 10% held out.
- Reference model is the frozen 6a checkpoint (0.7GB bf16 alongside the policy — trivial on the
  5090). Small β sweep; benchmark suite and `repair_val` CE watched for trunk drift.

**Gate G7 (the final POC gate):** benchmark suite within noise of 6a's seed; abstention **curve
bent, not slid** — false abstention < 10% *with* recall ≥ 0.5 *and* precision ≥ 0.65
(vs. the 0.578 flatline); answerable EM/F1 not regressed vs. the Phase 4 reading; G5's
attach-delta and G6's beyond-context result re-confirmed **on this checkpoint** — it is the model
every claim is made about, so every claim gets re-measured on it.

---

## Phase 7 — Throughput, FP8, and the real run spec

The 16B run measured **MFU ~11%, BF16 only, on an H100 NVL**. At a fixed hour budget, throughput
*is* token budget: every 1.5x here is 1.5x more pretraining data for the same money. This phase
spends a little compute to multiply the big spend, then writes the run down before it is bought.

### 7a. Throughput engineering (measured, on a short rented-box calibration)

- **FP8 via Transformer Engine** — the machinery already exists in-repo and went unused. Validate
  with an A/B at 332M: ~2B tokens BF16 vs FP8, loss curves within ~1%. A recipe that diverges at
  2B tokens does not get 5,000 hours of trust.
- Micro-batch / packing / grad-accum retune at the real shape (the local 4×4096 finding — same
  tokens per step, 3x throughput — says the defaults are not to be trusted).
- Profile the step: dataloader, aux/metric overhead, checkpointing levels, TE fused paths.
- **Inference-side:** the KV-cache ↔ convergence-exit exclusivity gets its real fix (K/V
  projections only for exited loops) **iff** G5 shipped depth > 3 or the exit is part of the
  shipped story; otherwise it stays a documented limitation rather than speculative plumbing.

**Gate G8:** ≥2x sustained tokens/s over the 97k/s baseline at equivalent shape, measured over
≥1h with checkpointing and upload machinery live, FP8 A/B within tolerance.

### 7b. Budget math (anchored to measurement, recomputed after 7a)

At the *measured* 97k tok/s: 1k hours ≈ 350B tokens, 5k hours ≈ 1.75T — for the 332M shape; a
500M-class shape lands around 2/3 of that, and the 7a multiplier scales whatever survives. Even
the 1k-hour floor is ~20x the POC's pretraining data. Consequences:

- **Data is the binding constraint, not compute.** The prepared corpus is 24.7B tokens. Target a
  ≥100B-token unique corpus from the same seven-source mix (fineweb-edu alone supports it) and
  cap repetition at ~4 epochs — the repeated-data literature puts up to ~4 epochs at near-fresh
  value, and beyond that the returns decay. Corpus prep is its own budgeted, interruption-safe
  job (`prepare_data.py` was designed for exactly this).
- **Keep the two-phase curriculum.** The phase-2 step change (CE 3.359 → 3.046 in 200M tokens)
  was a real distribution effect; the real run keeps a reasoning-weighted tail phase and the
  cosine that spans both.

### 7c. The run spec (`docs/plans/RUN2.md`)

Written before anything is rented, containing:

- **The go/no-go component table** — one row per mechanism, filled by gates G2–G7: sharpened
  parametric table (or frozen out, per the Phase 3 branch), evidence port, retriever adapter,
  shipping `n_loops`, depth curriculum, preference pass. Nothing ships on promise.
- **Shape** for the ~500M model, derived from the POC's measured FLOP components (the
  construction-time estimate is the budget's anchor — recompute it, don't reuse it) and from
  where the POC was capacity-bound vs data-bound on the benchmark snapshot.
- **LR transfer**: a 3-point LR sweep at reduced width, ~1B tokens each, local (days, not rented
  hours) — or a µP-style parameterization if it can be adopted cheaply; decided in the spec, not
  mid-run.
- Token budget from 7a's measured throughput × purchased hours, schedule, eval cadence (the
  Phase 1b suite at every checkpoint sync), and the SFT/preference plan (Phase 6's recipe re-run
  at scale).

---

## Acceptance (all gates)

- **G0** — benchmark harness reproduces a published peer; noise floor documented; three-checkpoint
  snapshot recorded (Phase 1b). **PASS** (2026-08-26,
  [record](../measurements/benchmark_snapshot.md)). The seed-noise half of 1b.2 is still open and
  is the one thing G0 passed without.
- **G1** — IR ablation ΔCE > ~0.02 nats (Phase 1). **Measured 2026-08-20: 0.0004 / 0.0002 nats —
  FAIL**, on the expected branch: the table is a bias term, the re-init is free, and the bar
  comes due again after the mechanism has something to retrieve.
- **G2** — post-anneal entropy ≪ `ln 65536`; held-out CE and benchmarks within noise; read-zeroed
  ablation well off the 0.0004 floor; A/B arm chosen (Phase 3). **Measured 2026-08-29: FAIL**
  ([record](../measurements/ir_sharpening.md)). Entropy and CE/benchmarks passed; the ablation stayed
  at 0.0002 nats, unmoved by 208M tokens on either arm. The A/B tied, so random keys are kept
  provisionally.
- **G2b** — the same question after the read is put at a usable scale: final-loop read-zeroed `dCE`
  **> 0.01 nats**, CE and benchmarks within a finetune's noise, with IR selection rate per loop,
  `‖g_proj‖` against its zero init and `‖y_values‖` reported alongside so the cause is attributable
  (Phase 3b). Passing makes the parametric table live and hands Phase 4 a working mechanism; failing
  freezes its size out of the real run spec.
- **G3** — gold-vs-no-evidence CE gap ≥ ~0.3 nats; abstention under no-evidence ≫ under gold;
  benchmarks within noise (Phase 4).
- **G3b** — groundedness AUROC ≥ 0.65 for unanswerable detection, against the trunk probe's
  **0.584** rather than `p_max`'s below-chance record (Phase 4).
- **G4** — recall@k beats BM25 **and** the frozen-embedder baseline (Phase 5a).
- **G5** — corpus-attached EM/F1 delta ≥3σ on NQ-open / TriviaQA / PopQA; depth ablation at
  `n_loops` = 2, 3, 4, 6, 8. **Flat past 3 → ship 3 and don't rationalize it** (Phase 5b/5c).
- **G6** — side-channel evidence beats the best 4096-token in-prompt packing when the evidence
  exceeds the context, at linear cost in evidence (Phase 5).
- **G7** — final POC model: benchmarks within noise; abstention curve bent (false abstention
  < 10%, recall ≥ 0.5, precision ≥ 0.65); G5/G6 re-confirmed on the shipped checkpoint (Phase 6).
- **G8** — ≥2x sustained tokens/s at equivalent shape; FP8 A/B loss-matched (Phase 7).
- **P0** — head removal behaviorally neutral. **PASS** (2026-08-19,
  [record](../measurements/phase0_migration.md)).
- **P2** — false abstention well below 78.4%, precision above base rate. **PASS** (2026-08-20:
  0.1358 / 0.5759; retuned 2026-08-22,
  [record](../measurements/abstention_repair.md)). Recall 0.81 → 0.22 is the open half, assigned
  to G3b/G7.

## Risks

- **332M / 16B tokens is a weak reader.** But grounded *extraction* is the easiest thing to teach
  at this scale — copying beats recalling. That is the real argument that RAG suits *this* model:
  it converts a knowledge problem it can't solve into a copying problem it can.
- **Forgetting during the Phase 4–6 spends.** These are the largest finetunes the project has run,
  on evidence-shaped corpora. The ≥20% replay floor and the benchmark gate on every phase exist
  because a narrow finetune that quietly costs trunk quality is invisible to a QA-only eval.
- **The oracle-format shortcut.** A model trained mostly on gold evidence can learn "evidence
  attached → copy something" and faceplant on imperfect retrieval. The gold-among-distractors and
  distractors-only conditions exist to break that shortcut; watch the distractor-condition EM as
  its own number, not folded into an average.
- **Both discrimination signals could fail.** `p_max` already has (three checkpoints); if Phase
  4's groundedness mass also can't separate answerable from unanswerable, the fallback is Phase
  1b's probe + Phase 6's preference pass, and the calibration story ships weaker. **1b.3 measured
  how weak that fallback is: AUROC 0.584, identical on all three checkpoints**, so the fallback is
  a better-shaped curve at low recall, not a discriminator. The class claim survives; the gate
  table must say which story shipped.
- **Benchmark contamination.** fineweb-edu's decontamination against these suites is not
  guaranteed. Standard val/test splits only, and a single suspiciously-strong benchmark is
  treated as suspect, not as a win.
- **Re-initialized tensors vs. a finetune LR** — addressed by the per-tensor LR groups in Phase
  3; the residual risk is interference between a hot IR pathway and a cold trunk, which the
  benchmark gate is there to catch.
- **Centroid staleness.** Two-stage scoring is only exact if assignments track the keys. Refresh
  cadence is a real hyperparameter, not a detail — same for the ANN index if 5a's doc-side
  extension is ever bought.
- **FP8 divergence risk.** Validated on a 2B-token A/B before the real run trusts it; BF16 is the
  fallback and the budget math is honest either way.
- **MTP stays on** (Plan A). If Phase 3/4 throughput or loss goes badly, the fallback is dropping
  it *for the finetunes only*: note that `lambda_mtp: 0.0` does **not** skip the compute —
  `compute_mtp_loss` gates on `mtp_outputs is not None`, so a zero weight still pays the head
  plus its chunked vocab projections at 4x. Pass `mtp_outputs=None` / skip `_mtp_forward`, keep
  the weights on disk. (The inference-side half of this is fixed in Phase 1b.)
- **Data prep at 100B+ tokens is a real job**, not a preamble — schedule it on the box with the
  same interruption-safety it was designed for, and budget its wall clock explicitly in RUN2.md.

## Parked

- **Step 13 — self-labelled calibration set.** **Unparked into Phase 6b** — the repaired baseline
  is non-degenerate, so pass-rate labeling no longer re-encodes the collapse.
- **Step 16 — RL.** Do not start. A 135M RLVR study on GSM8K went *backwards* under GRPO
  (24/1319 → 21/1319 → 16/1319); Qwen2.5-0.5B base stayed below 0.1 format reward after 300
  steps. Under 0/1 reward a base model that can't sample correct solutions produces no gradient.
  **Unparks when** pass@8 on the target task exceeds ~15% on the post-Phase-4 checkpoint (GSM8K
  pass@8 now has a standing measurement via the Phase 1b suite). If it ever does: vanilla GRPO is
  mismatched to a looped model (it credits output tokens while the computation is latent) — read
  LoopRPT (arXiv 2603.19714) and RLTT first, which assign reward to per-loop latent states.
- **Step 17 — reasoning training.** Gated on Step 16's pass@8. Below ~15%, reasoning training
  isn't worth attempting by any method at this size — the Small Model Learnability Gap says
  long-CoT imitation teaches fluent filler before a wrong answer. Above it, the cheap probe first
  (add back smoltalk2's `_think` splits with a distinct reasoning segment in the chat template),
  then loop-aware RLVR — which for *this* architecture is closer to the truth anyway: `n_loops`
  and `loop_scale` are the reasoning substrate here, not the token stream.
- **B1 (self-embedded query/key space).** Revisit after 5a — if the external-embedder pipeline
  wins its gates, B1 is a real-run refinement, not a POC question.
- **`num_ir_experts > 1`.** The one invasive option — shifts `first_mlp_index` and the router's
  output dim, needs a surgical remap of the router weight rather than a plain load. Only once a
  single sharpened, grown table demonstrably helps and is saturating.
- **Masked-diffusion trunk.** Not a stage swap — a different project: the benchmark suite scores
  log-likelihoods (diffusion yields ELBO bounds), `p_max` and the entire calibration/abstention
  record assume per-token AR confidence, MTP / KV cache / the convergence exit are AR machinery,
  and extractive span copying — the one thing 332M provably does — is what left-to-right decoding
  with a reader is shaped for. At 16B single-epoch tokens the public record (LLaDA-class models
  needing trillions of tokens to reach AR peers) puts diffusion behind AR at matched compute at
  this scale; its genuine case is the **data-constrained multi-epoch regime** ("Diffusion Beats
  Autoregressive in Data-Constrained Settings", Prabhudesai et al. 2025), where repetition decays
  slower for diffusion than for AR. That regime is a live possibility for the real run: at 7a's
  target throughput, 5k hours can consume 2–4x the capped 4-epoch supply of a 100B-token unique
  corpus. **Unparks iff 7b's budget math lands data-bound** (capacity tokens ≥ ~2x the 4-epoch
  supply) — then buy one ~16B-token matched-compute *multi-epoch* A/B (AR vs. masked diffusion,
  same corpus, same suite) with an ELBO scoring path validated against a published diffusion
  peer's numbers first, same rule as G0; a harness that can't reproduce LLaDA/Dream can't rank
  the A/B. What LoopLM keeps either way: diffusion's actual mechanism advantage — iterative
  refinement conditioned on the step index — is exactly what the loop-conditioned query/router
  bias installs, without abandoning the objective every instrument is built on.

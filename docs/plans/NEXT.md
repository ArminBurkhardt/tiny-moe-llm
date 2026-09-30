# NEXT.md

The plan (older notes and commit messages call it `PLAN.md`). Rewritten 2026-09-30 around the goal
stated below; the previous version is in git history (`git show aea3372:docs/plans/NEXT.md`). The
record of the 16B-token run is [docs/CONCLUSION.md](../CONCLUSION.md). Read
[CLAUDE.md](../../CLAUDE.md) first, it is authoritative for everything already built. Every
measurement lives in [docs/measurements/](../measurements/). The two reviews this plan rests on
are [docs/review_2026-09-18.md](../review_2026-09-18.md) (the pre-run code review) and
[docs/review_2026-09-29.md](../review_2026-09-29.md) (the direction review, with the loop section
and the new references). The design itself, with blueprints of both runs, is
[docs/evidence_path_design.html](../evidence_path_design.html).

## Now (2026-09-30)

Nothing has trained since 2026-09-20. The run's instruments were fixed on 2026-09-30 (Phase 4,
"Before the launch", all eight items) and R0 was read. In order:

1. **Launch** arm A, then arm B, each under a Monitor watch:
   ```
   python scripts/sft.py --evidence -c ckpts/repair/checkpoint_repair_final_irrandom_evidence_grounded.pt
   python scripts/sft.py --evidence --reader-no-rotary --run-name norope -c <same seed>
   ```
   The kill is automatic: the first eval past 10M tokens with a fixed-target gold gain under 0.1
   nats saves and exits 10. Otherwise stop each arm by hand at 10M (`touch ckpts/evidence*/STOP`)
   and read the five `[eval fixed]` blocks: the gain, the per-loop chunk AUROC, the per-loop mass.
   This run is a mechanism check: it can show that the port reads, not where facts live.
2. **Build the separation instruments** (Phase 4b): counterfactual and closed-book conditions in
   `eval_abstention.py`, the synthetic chain generator and its eval, an evidence path in
   `eval_benchmarks.py`.
3. **Then the pilot** (Phase 5).

Seed baselines, read before step 1 on the new splits: fixed-target gold gain **-0.0020 nats**
(noise from packing, the reader is exactly neutral: `eval_abstention.py --evidence-port` reads
+0.0000 on 400 SQuAD rows), chunk AUROC 0.434 / 0.415 / 0.415 by loop (the seed's selector ranks
gold chunks slightly below distractors), gold share of the external mass 0.334 (uniform over the
buffer), grounded AUROC 0.5000 (zero head). **R0 failed on both checkpoints**
([loop_scale_probe.md](../measurements/loop_scale_probe.md)): every multiplier makes loop 3 worse,
so `loop_scale` stays out of the fresh group in graft arms. The seed's `g_proj` and `direct_gate`
are both exactly zero and hold each other there: the IR value path is dead for this run, which
matches the real-run decision (selector without a value read); the selector trains from the
selection loss alone (`value_adapter` gradient 0 at step 10, `key_adapter` 1.4e-3).

Corpus `evidence_train`: 963,011 conversations, 140.1M prompt / 525.6M evidence tokens, ratio
3.75. Held-out `evidence_dev` 18,966 rows (ratio 11.2, QA only) and `evidence_fixed` 13,333
questions x 4 conditions (ratio 6.25), from `prepare_evidence_data.py --heldout`. `evidence_val`
is a train-loss slice. `EvidenceConfig`: batch 2, accumulate 8, `max_evidence_tokens: 14336`,
`eval_every_tokens: 2500000`; smoke peak 26.4 GB, 8.6 to 14.8k tok/s. `ckpts/evidence/` does not
exist, so the first launch honours `-c`.

## The goal

> All world facts come from an external store through the retrieval pathway: the selector chooses
> chunks by embedding, the evidence reader carries their exact tokens into the model. The trunk
> holds language, common sense and reasoning, and the looped block is where reasoning over the
> retrieved facts happens.

Two things the wording settles:

- **The IR expert is the selector, not the information channel.** Its external value is a
  projection of one pooled 384-d vector, so no span can be copied through it. Content flows through
  the cross-attention reader. Making the selector carry content would compress chunks into one
  vector, which published work shows recovers gist, not facts. The goal is stated over the pathway.
- **"No facts in the weights" is the strong form and nobody has demonstrated it.** The reachable
  target is no long-tail or episodic facts in the weights, with a language and common-sense core
  kept. What the trunk keeps and what the store holds:

| information type | lives in |
|---|---|
| entity, episodic and long-tail facts, numbers, dates | store |
| rare-word and domain definitions | store |
| syntax, common word meaning, coreference | trunk |
| common sense and basic priors | trunk |
| procedural and reasoning skill, code, arithmetic | trunk |

Two mechanisms carry the goal and both are measured separately:

- **Externalization**: the trunk is never rewarded for producing a fact the buffer does not supply,
  and it is penalized where recall from weights and evidence disagree. This is a pretraining
  property. Every graft onto the converged checkpoint measured zero, and the published systems
  that externalize facts impose it from token 0.
- **Reasoning over evidence**: the selector and the reader both query the state a loop starts with,
  and chunks are encoded independently, so a chunk that depends on a fact in another chunk is
  unreachable until that fact is in the state. k hops over evidence need at least k loops. The
  architecture forces depth; the data and the objective have to ask for it.

## What must be proven

The gates below replace the POC's older G-gates for everything after Phase 4. Each is a metric, a
dataset, a bar against a noise floor, and a script that holds it. Thresholds are relative to
[noise_floor.md](../measurements/noise_floor.md); it has no CE sigma yet, so a paired-bootstrap CE
sigma is recorded before any CE gate is quoted.

| gate | metric | bar | home |
|---|---|---|---|
| A1 closed-book floor | EM on TriviaQA, NQ-open, PopQA and an obscurity-tier probe with the pathway off and a neutral prompt | tail tiers near chance, no more than 2x the seed | `eval_benchmarks.py` |
| A2 evidence recovery | fixed-target answer-span CE, gold minus none, held-out rows | at least 50% of the 3.23-nat in-context ceiling; distractor minus none at most 0 | `eval_abstention.py --evidence-port` |
| A3 counterfactual faithfulness | answer follows an entity-swapped gold chunk, stratified by entity frequency | follow rate at least 90%, parametric-answer rate at most 5% | new condition in `eval_abstention.py` |
| A4 store edit | change or delete the gold chunk and its near neighbours; the answer flips or disappears | at least 70% of edits propagate | two-store swap in `eval_abstention.py` |
| A5 selector | mass on the gold chunk in `mixed`, per loop; groundedness-head AUROC | AUROC at least 0.674, three sigma over the 0.584 probe | `eval_abstention.py` |
| A6 reasoning retained | HotpotQA, BoolQ and SciQ through the port against in-prompt; language benchmarks within noise | port matches in-prompt | `eval_benchmarks.py` with an evidence path |
| L1 composition | synthetic k-hop chains with fictional entities, accuracy by hops and by loops run | every cell with fewer loops than hops at chance; held-out-template 2-hop above the control | synthetic eval, Phase 4b |
| L2 depth pays | held-out 4-hop at 4 loops; HotpotQA bridge answer CE from 1 to 3 loops | rises with loops; at least 0.05 nats | synthetic eval plus `eval_abstention.py` |

A1 is already met by the seed (TriviaQA 0.014, NQ-open 0.002), so it says nothing until A3 and A4
exist. It guards against facts coming back in. A3 is read at every checkpoint, because context
reliance is known to rise early and then decay under finetuning on context that agrees with the
weights. L1's validity rule is mechanistic: the read structure makes accuracy with fewer loops than
hops impossible, so anything above chance there is a leak in the data.

## Rules (carried over)

- Commit per logical change (`feat:` / `docs:` / `chore:`, branch `ir-train-build`, PRs to
  `prototype`), single-line subjects.
- Lowercase explanatory comments that justify why; Google-style docstrings with `Args:`.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, `moe` in the training loop
  goes through `accelerator.unwrap_model(model)`.
- **Never add `.item()` / `.tolist()` / `.cpu()` / boolean mask indexing to the per-step path.**
- **Every reported number is measured at fixed eval flags**, quoted against the noise floor.
- **Looping is a requirement.** A later loop that does not earn its compute is a defect to fix,
  never a reason to ship shallower.

## Decisions

| decision | choice |
|---|---|
| goal | The reworded goal above. Strong form is not a gate; A1 with A3 and A4 is. |
| roles | Selector = the IR path, chooses by embedding. Reader = cross-attention over chunk tokens, carries content. |
| learned IR table | Dropped from the real run. Zeroing its read costs 0.0002 nats on every arm. |
| selector in the real run | Always on, not routed; scores raw bge keys plus one learned null key; no value read, no key adapter. |
| selector to reader coupling | The selector's own logit enters the reader's attention logit through a one-hot chunk channel, per token, causal. Replaces the document-mean sigmoid gate. |
| reader positions | No rotary in the reader; the encoder already positioned chunk tokens. Arm B of Phase 4 tests it. |
| reader output | `o_proj`, then after `post_norm`: `h = h + g_loop[k] * read`; `g_loop` zero-init on a graft, one-init from scratch. Not a norm after a zero-init projection. |
| evidence encoder | The trunk's own dense decoder, chunk-causal. All 8 layers in the POC; the first 4 prelude layers under TE checkpoint from scratch. No cached encoder states in the store. |
| readout | A 2-layer coda after the loop, own KV slots. The loop refines, the coda reads out. |
| loss placement | One CE on the last pass run, through the coda. Per-loop CE is the control arm, not the default. |
| depth | Drawn per step from {3: 0.7, 4: 0.2, 5: 0.1}; the first D minus 3 passes run under `no_grad` and are detached; minimum 3, so no pass is an exit. Inference 3, at most 4. |
| loop tensors | `loop_scale` and `evidence_loop_scale` length 5, fresh `1/sqrt(5)`; `loop_inject` on; loop code clamped to the trained depth. |
| selection supervision | Union over the passes in the gradient window: `u_c = 1 - prod(1 - p_hat[k, c])`, BCE against the gold flag. Never a target per hop and per loop. |
| query training | InfoNCE on the in-model query against the bge key of the gold chunk, in-batch negatives, same-document and near-duplicate negatives masked. Keys stay bge, so the index is never rebuilt. |
| token tables | One table, tied to the LM head and the MTP head through small adapters; the per-layer and router tables become projections of it, gated on a zero-ablation of the per-layer read (G9). |
| shape | Shape L: width 1024, 16 heads x 64, 4 KV heads; 8 prelude layers; looped core of 3 sublayers per pass with 8 MLP experts each, top 2; 2 coda layers. About 470M total, 300M active per token. Sublayer count and expert count are pilot arms. |
| context | 4096 for the pilot and the main run; a short 8k to 16k extension phase at the end on long documents with retrieval attached. Never 32k from the start. |
| real run ordering | Retrieval-augmented pretraining from token 0 with fact spans kept out of the loss. No plain pretraining plus a graft. |
| real run budget | About 2.5k H100 hours for shape L, about 350B tokens with evidence; the plan's 5k-hour ceiling stands. Recomputed after Phase 6a's throughput work. |
| abstention | The groundedness head reads the reader output and the null mass; the preference pass stays deferred until an A-gate has a pilot reading. |
| POC role | Phase 4 is a mechanism check of the reader, read per loop. It decides nothing about externalization. |
| matched compute | SMELT's definition: equal compute per token, equal non-embedding parameters and equal KV-cache size, or the comparison is not quoted as compute-matched. |
| IR key init, `num_ir_experts > 1`, B1 | Moot: the table is dropped. |

## Where this stands

Done, with the full records linked:

- **Phase 0**: both learned heads removed, the halt gate's measured pass-through folded into
  `loop_scale` ([phase0_migration.md](../measurements/phase0_migration.md)). `p_max` beat
  `p_correct` a third time.
- **Phase 1**: Stage 0 diagnostics ([stage0_diagnostics.md](../measurements/stage0_diagnostics.md)).
  The IR table stores nothing (entropy 99.5% of max, zeroing the read 0.0002 to 0.0004 nats). Query
  drift after loop 1 is about zero. Loop 3 is redundant, not idle.
- **Phase 1b**: the benchmark suite validated against Pythia-410m on 11 of 11 anchors, four peers
  frozen, slice noise measured, the three-checkpoint snapshot recorded
  ([benchmark_suite.md](../measurements/benchmark_suite.md),
  [noise_floor.md](../measurements/noise_floor.md),
  [benchmark_snapshot.md](../measurements/benchmark_snapshot.md)). The answerability probe reads
  0.584 on all three checkpoints ([answerability_probe.md](../measurements/answerability_probe.md)).
  Mean MC headroom +0.088 / +0.081 / +0.084 against gpt2-medium's +0.193. BoolQ with the passage in
  the prompt reads 0.46 / 0.44 / 0.42, below chance.
- **Phase 2**: abstention repair ([abstention_repair.md](../measurements/abstention_repair.md)).
  False abstention 0.783 to 0.136; recall 0.81 to 0.22; precision pinned at 0.578 six readings
  running. The data lever is exhausted.
- **Phase 3, 3b, 3c**: table reshape and sharpening, the value-scale fix, loop input injection
  ([ir_sharpening.md](../measurements/ir_sharpening.md),
  [ir_scale_fix.md](../measurements/ir_scale_fix.md),
  [loop_injection.md](../measurements/loop_injection.md)). All three gates failed on the same
  number: the read is worth 0.0002 nats whatever the init, width or scale, and the injection made
  later loops marginally more alike. The graft record is 0 for 4.
- **The evidence ceiling** ([evidence_ceiling.md](../measurements/evidence_ceiling.md)): in-context
  gold evidence is worth +3.23 nats on the answer span; a distractor costs 0.63 nats more than
  nothing.
- **Phase 4 build**: port, corpus builder, `--evidence` profile, selection loss, groundedness head,
  `eval_abstention.py --evidence-port`, all built; the 2026-09-18 review's thirteen findings acted
  on; corpus built and seed migrated 2026-09-20; batch and cap retuned. Not trained.
- **The direction review** (2026-09-29): seven audits, five literature sweeps, three judges. It
  found the kill metric, the validation leak, the non-causal gate and the reader's rotary phase, and
  established that the corpus and the old real-run recipe both keep rewarding recall from weights.
  Its loop section found the objective and the migrated `loop_scale` working against depth, and
  merged three loop designs into the one in Decisions.

## Binding measurements

Each measured, each recorded; a retry must move the ablation, not the entropy.

- The learned IR table stores nothing the trunk lacks: zeroing its read costs 0.0002 nats at any
  init, width or scale. Dropped from the real run.
- In-context gold evidence is worth +3.23 nats on the answer span; a distractor costs 0.63 nats more
  than nothing. External evidence is the mechanism, and 3.23 is the ceiling every reader reading
  is a fraction of.
- Every graft onto the converged checkpoint measured zero for four. The real run pretrains with
  retrieval from token 0.
- `p_max` carries no answerability signal; a linear trunk probe reads 0.584 on every checkpoint.
  Abstention precision is pinned at 0.578 by the data lever.
- Loops buy computation, not storage. Looping is a requirement.
- No per-loop number on record is free of the halt gate: it passed 8% of the loop-3 update for all
  16B pretraining tokens, and every later arm grafted onto that lineage. Only the from-scratch
  pilot can separate "loops cannot reason" from "loops were never allowed to".
- The trunk is already near-empty of facts (TriviaQA 0.014, NQ-open 0.002). The pending run tests
  reader capacity, not fact removal.
- Depth past 3 degrades on plain text: CE 3.4112 at loop 3, 3.4115 at loop 4, 3.4900 at loop 8 on
  `ir_c`, with no evidence attached.

---

## Phase 4: the evidence finetune as a mechanism check

One question: can the port be read at all. It runs the architecture as built, on the grafted
checkpoint, with the gate frozen. It cannot show where facts live: the seed scores near zero
closed-book, and 92% of the corpus's supervised tokens carry content the buffer does not supply.

### Before the launch

**Done 2026-09-30**, all eight: `[eval fixed]` in `sft.py` over `evidence_fixed` with the kill at
10M tokens, `evidence_dev` / `evidence_fixed` from `prepare_evidence_data.py --heldout`, the 2.5M
cadence plus a step-0 eval, `freeze_evidence_gate`, `--reader-no-rotary` (inferred from
`moe.evidence_reader_rotary_off`), one chunk per passage and same-passage exclusion in
`eval_abstention.py`, the seed's value path printed, and the tests (`test_port_backward`,
`test_gate_causality`, `test_evidence_decode_cache`, `test_fresh_param_routing`,
`test_reader_rotary`, `test_prepare_evidence_heldout`). The gate test measured the leak it exists
for: at scale 5 earlier positions move 4.2e-3 against an exact 0, and cached against uncached
decode differs by 8.1e-3.

Ordered by how badly each would mislead the run. None needs the GPU.

1. **The kill number.** `sft.py`'s per-condition `[eval]` CE uses each row's own target, the answer
   under `gold` and a refusal under `none`, and reads minus 3.25 nats on a seed with a dead reader.
   The `none` bucket also holds replay rows. Add a fixed-target pass to `evaluate()`: teacher-force
   the real answer under every condition over the natively answerable QA rows, print gold minus
   none, and kill on that. The 0.0000 seed baseline is this number, read by
   `eval_abstention.py --evidence-port`.
2. **The validation split.** `prepare_evidence_data.py` splits per rendered row after the QA
   sources repeat, so every QA validation question is also in train (7,440 of 7,440). Build
   `evidence_val` from SQuAD dev and HotpotQA dev, split by source id before the condition draw.
   Treat the current `evidence_val` as a train-loss slice.
3. **Cadence.** `eval_every_tokens: 2500000`, and an eval before step 1, so the decision at 10M
   tokens has five readings and a baseline instead of one reading on the decision itself. Warmup
   covers the first ~4.2M tokens.
4. **The chunk gate.** `chunk_mean_mass` averages the selector's weight over every token of the
   document and that number gates the states every position reads, so training sees future tokens
   and cached, uncached and training gates differ. Freeze `evidence_gate_scale` at zero:
   `requires_grad = False` and out of the fresh group.
5. **Reader rotary.** The query is rotated by its prompt position and the key by its position in
   the chunk; the offset is unrelated to content. Add a flag that passes `position_embeddings=None`
   to the reader. Arm B.
6. **Eval chunking.** The primary G3 reading uses one chunk per passage, as trained; the 128-token
   re-chunking stays as a secondary robustness row. Exclude same-context passages from the eval
   distractor pool.
7. **Seed check.** Print `|g_proj|rms` once. If it is zero, `g_proj` and the zero-born
   `direct_gate` hold each other at zero and the IR value path is dead for the run.
8. **Tests.** Three: backward on a zero-init port (`o_proj` gradient nonzero at step 0, q/k/v zero,
   then nonzero after one step), causality of the gate when its scale is nonzero, cached against
   uncached decode with evidence attached. Plus a GPU-free assert on the `is_fresh_loop_param`
   predicate and the by-hand gradient clear in `train_step`, which route exactly the tensors this run
   trains and have no test.

### The run

- Arm A: as built, gate frozen. Arm B: same, reader without rotary. 10M tokens each, about 20
  minutes each, under a Monitor watch.
- Kill at 10M tokens if the fixed-target gap is under 0.1 nats.
- Pass at 50% of the in-context ceiling, about 1.6 of 3.23 nats, with distractor minus none at or
  below 0. Report the fraction of the ceiling closed, on the same rows as
  `evidence_ceiling_probe.py`.
- Read per loop: selector AUROC (`memory_mass_by_loop`), external mass per condition, and the
  reader gain. A rising `|shared_evidence.o_proj|rms` with a flat gap means the reader learned a
  bias from a random reader, not retrieval; RMS alone is not the signal.
- Log gradient norms of `key_adapter`, `value_adapter`, `down_proj` and `loop_query_bias` at steps
  10 and 100.
- Under 50% of the ceiling: the next arm is a reader per loop or reads in dense layers, before more
  data. Arm C, only after a pass: the one-hot chunk channel with `gamma = 0`.

### What it decides

A pass says the grafted reader can carry a span. A fail on the graft is weak evidence, given the
record. Neither outcome moves the real run's design, which is why Phase 5 does not wait for it.

**Gate G3** (restated): fixed-target gold-minus-none gap on held-out rows, at least 50% of the
ceiling; distractor minus none at most 0; benchmarks within noise. **Gate G3b**: selector AUROC at
least 0.674 per loop, which is three sigma over the 0.584 probe (0.65 was 2.2 sigma); the
groundedness head's AUROC scored in the eval script, beside the mass.

---

## Phase 4b: instruments

Everything the goal is measured with. Most of it is eval-only code and runs on existing
checkpoints; it is on the critical path because the pilot cannot be read without it.

- **R0, the loop scale probe.** `eval_stage0.py` gets a flag that multiplies `loop_scale` at eval.
  Run on `ir_c` and `phase2_final` with factors 1, 2, 3.5 and 5.9. Read the CE gain from loop 2 to
  3. Pass at 0.02 nats with loops 1 and 2 not worse. Decides whether `loop_scale` joins the fresh
  LR group in graft arms. **FAIL 2026-09-30** on both checkpoints, monotone in the multiplier
  ([loop_scale_probe.md](../measurements/loop_scale_probe.md)): it does not join.
- **A3 and A1 conditions in `eval_abstention.py`.** A counterfactual condition following the
  Faithfulness-QA recipe: swap the answer entity in the gold chunk for a same-type entity, score EM
  against the swapped answer and report the parametric-answer rate, stratified by entity frequency.
  A closed-book condition with a neutral prompt that does not licence abstention. Per-record JSON
  in evidence mode, EM and non-abstain rate under distractors and none.
- **A4, the store swap.** Two stores on the same questions: the gold chunk edited or deleted with
  its near neighbours; report the flip rate.
- **A5 readouts.** Gold-chunk mass in `mixed` per loop and per IR expert (the eval reads
  `ir_modules[0]` only today); the groundedness head's AUROC.
- **The synthetic chain generator and eval** (L1, L2). Fictional entities, hops 1 / 2 / 3, 6 to 14
  distractors of three kinds (same relation with other entities, bridge relation with other
  entities, one full decoy chain), shuffled chunk order, a hop index per gold chunk written to an
  `.evhop` sidecar for evaluation only, 4-hop chains and unseen hop-2 templates held out. Score
  accuracy by hops and by loops run. The validity rule: every cell with fewer loops than hops must
  sit at chance.
- **An evidence path in `eval_benchmarks.py`.** SciQ and BoolQ carry their passage; put it through
  the port and diff against the in-prompt score (A6). TriviaQA-rc and NQ with DPR gold passages for
  the attach delta. HotpotQA validation with both gold paragraphs in the port, by loop count.
- **Per-loop readouts in the training log.** `cos(q_k, q_k+1)`, per-loop selector mass, per-loop
  reader gain, and `||h||` per loop (the readout blind spot from the first review).
- **The oracle head (G9).** Train a dense `768 -> 65536` head on frozen final-loop states for
  100 to 200M tokens; `CE_factored - CE_oracle` is the damage the block-diagonal head does. Tie if
  at least 0.02 nats. Plus the free reading: least-squares fit of `E A^T` to the trained head.
- **A paired-bootstrap CE sigma** on the standard slice, so "within noise" is defined for CE.

---

## Phase 5: the from-scratch pilot

The first experiment that can falsify the goal. About 120M parameters at width 512, 1 to 2B
tokens per arm on the 5090, with a matched-token control without retrieval. Run times are
unmeasured; the R1b memory probe comes first.

### The model, from scratch

Every item is in the design page's from-scratch blueprint. Grouped by what it touches.

**Token tables and readout.**
- One table `E`, init std `1/sqrt(hidden)`, decayed in its input role only. LM head
  `norm(h) @ A @ E^T` with `A` hidden x hidden; MTP head through a `hidden/2 -> hidden` adapter
  onto `E^T`, CE on a 25% token subsample. Fused linear plus cross-entropy in
  `_chunked_linear_ce`. The per-layer and router tables become `E W_ple` and `E W_moe` if the
  per-layer zero-ablation lands near the IR table's 0.0002 nats; otherwise the per-layer table stays.
- A 2-layer coda between the loop and the norm, a separate `ModuleList`, zero-init residual
  branch, own KV slots allocated in `KVCache.__init__`, keys in `NEUTRAL_LOOP_KEYS` and
  `is_fresh_loop_param`. MTP reads the coda output. The convergence exit stays off.

**Selector.**
- Always on, outside the router pool. `q = normalize(down_proj(norm(h)) + loop_query_bias[k])`,
  384-d. `z[t, c] = cos(q_t, key_c) / tau` over the candidates plus one learned null key. Keys are
  raw bge; `key_adapter`, `value_adapter`, `g_proj`, `up_proj`, `direct_gate` and the learned table
  are gone. The null mass feeds the groundedness head.
- The selection loss is the union over the passes in the gradient window on the renormalized
  per-pass share. Returned from `forward_step` like the aux loss, so gradient checkpointing can
  stay on.

**Reader.**
- `k'_j = [W_k e_j ; onehot(rank(chunk(j)))]`, `q'_t = [W_q x_t ; sqrt(d) * b_t]`, head dim
  `d + M` rounded to a multiple of 8, `softmax_scale = d^-0.5` passed explicitly, so
  `logit(t, j) = content + b[t, chunk(j)]`. `b[t, c] = gamma_h z[t, c] - lambda log n_c + vis[t, c]`,
  `gamma` per head init 1, `lambda` init 0, `vis` the visibility mask at minus 1e4. One sink slot
  per segment with a learned key and a zero value. No rotary. 4 KV heads. K and V are
  pass-invariant, one cache per buffer. `M` up to 40. Check first that the installed flash-attn
  build accepts the head size.
- Output: `o_proj`, then after `post_norm`, `h = h + g_loop[k] * read`, `g_loop` init 1.
- Read sites: sublayers 1 and 3 of every pass. Reads in dense prelude layers 3 and 6 are a pilot
  arm.

**Loop.**
- 3 sublayers per pass: self-attention (shared), always-on MLP plus 8 routed experts at top 2, the
  reader in sublayers 1 and 3, `post_norm` and `loop_scale[k]` per sublayer.
- Depth `D` from {3: 0.7, 4: 0.2, 5: 0.1}; the first `D - 3` passes under `no_grad`, detached;
  the last 3 carry gradient; log steps at `D = 3`. `loop_inject` on, because a detached prefix
  otherwise leaves the decoder without main-CE gradient on those steps. `loop_scale` and
  `evidence_loop_scale` length 5, fresh `1/sqrt(5)`. The loop code clamped to the trained depth.
- `loop_ce_weights` are gone; the loss is `CE(head(coda(h_D)))`. The control arm keeps per-loop
  CE and `loop_count_sampling: 0.3`. Note that `loop_ce_weights: [0, 0, 1]` alone is a trap:
  `loop_ce_weights_for` returns the zeros unrescaled, so 30% of steps carry no CE.

**Encoder.** The first 4 prelude layers, chunk-causal, TE checkpoint, encoded once per forward.

### The pilot corpus (a new builder mode)

Retrieval is attached to every slice. The only evidence-free case is the abstention condition.

| slice | tokens | sources | evidence | supervision |
|---|---|---|---|---|
| web and edu text | 45% | the pretrain mix, fineweb-edu weighted up | top 2 chunks per 256-token window, staircase visibility | span weights, counterfactual swaps, tail anonymization |
| code and math | 10% | stackv2 edu, Nemotron math | same | plain CE |
| evidence QA | 15% | SQuAD v2, HotpotQA, NQ, TriviaQA with passages, MuSiQue, 2Wiki; at most 2 passes each | gold, mixed, many, distractors, none, counterfactual, partial hop; buffer capped at 8 chunks | answer or abstain |
| synthetic chains | 10% | generated | hops 1 / 2 / 3 at 25 / 50 / 25 | answer |
| chat replay | 20% | smoltalk2, instruction and format rows | same per-window retrieval | assistant text, span weights |

- **Windows.** Each 256-token window is queried with the bge embedding of the preceding window's
  text, offline. Top 2 from the store, the document itself excluded. Window 0 is queried from its
  first 64 tokens and readable from token 64. Tokens in window w see the chunks retrieved for
  windows up to w; the mask lives in `vis[t, c]`.
- **Near-duplicate filter.** Drop a candidate whose longest common token run with the document is
  32 or more, or whose 8-gram Jaccard is 0.3 or more. Hash sets per document; the cost is
  negligible. Report the language-model gain by overlap tier, because a paraphrased mirror passes
  both thresholds.
- **Span weights.** Entities, numbers and dates are tagged (numbers need two tokens or a matching
  context n-gram). A span is supported when it occurs in a visible chunk: weight 1, chunk flagged
  gold. Unsupported: weight 0. Language tokens: weight 1. On 15% of supported spans the entity is
  swapped for a same-type entity in the chunk and the target alike. In 50% of documents,
  unsupported entities below the head-frequency threshold get a typed placeholder in input and
  target. The swap and anonymization rates are first guesses and get a sweep.
- **QA.** Distractors half bge hard negatives, half random. Partial hop: a HotpotQA row with one
  supporting paragraph withheld, target abstains. Hop labels from MuSiQue and 2Wiki are for
  evaluation only. A wider abstention phrasing set.
- **Store.** Training corpus plus Wikipedia, at least 0.4B tokens, 128-token sentence-aligned
  chunks, MinHash deduplicated, raw bge keys. Closed-book probe answers held out of the pretraining
  text. No cached encoder states (600 GB at 0.4B tokens).
- **Build metrics.** Share of tagged spans supported by a visible chunk (under about 20% fails
  the build); gold recall of bge top 2 on the QA slices; near-duplicate drop rate per source;
  evidence ratio and weighted gradient share per slice; the contamination check.

### The ladder

| rung | tokens | runs | read | pass | decides |
|---|---|---|---|---|---|
| R0 | 0 | loop scale probe on existing checkpoints | CE gain loop 2 to 3 | 0.02 nats, loops 1 and 2 not worse | `loop_scale` in the fresh group for grafts |
| R1 | 10M | the Phase 4 run | selector AUROC per loop | loop 2 or 3 beats loop 1 by 0.02 | whether reads differ by loop at all |
| R1b | 0 | one micro step at depth 3 and at depth 5 with the prefix, pilot shape | peak memory | fits | whether the depth schedule is trainable |
| R2 | 0 | synthetic chains scored on the R1 checkpoint | accuracy by hops and loops | chance wherever loops are fewer than hops | whether the instrument is valid |
| R3 | 2 x 30M | graft: proposed objective against the current recipe, 15% synthetic, `loop_scale` reset | 2-hop accuracy at 3 loops minus 1 loop | +0.15 and proposed beats current | a pass helps; a null decides nothing |
| R4 | 4 x 1B | from scratch: loss placement by coda, 2 x 2, shared fresh `loop_scale`, inject and data | held-out-template 2-hop at 3 loops | best arm 10 points over per-loop loss without coda; 3-hop above chance | objective and coda |
| R4b | 2 x 1B | sublayers per pass 2 against 3 at matched compute; experts 4 against 8 | L1 plus closed-book recall plus language benchmarks | see gates | shape L's core |
| R5 | 0 | the winner, eval only | held-out 4-hop at 4 loops; HotpotQA bridge by loops; A1 to A6 | L2 and the A gates | whether the design enters the real run |
| R6 | 1 to 2B | hop 1 seen only in pretraining text (about 100 exposures), hop 2 retrieved | 2-hop accuracy at 2 or more loops | well above chance | whether weights and store compose |

A 1-hop curve must saturate before any null is read. R4 replaces the previous plan's per-loop CE
test, which changed loss placement, depth sampling and the coda in one arm. Pilot arms that do not
fit the budget are dropped from the bottom of this table, not the top.

### Later, after the pilot passes

- Reader-attention distilled into the selector where no gold label exists.
- The append-only buffer and per-pass re-query at serving, with a staged two-hop eval. Untrained
  today, so deferred.
- A learned bridge-entity span as a small auxiliary, as an arm.

---

## Phase 6: the real run

### 6a. Throughput first

The 16B run reached about 11% MFU on an H100. The step is overhead-bound, not GEMM-bound: routing
with host syncs through `m_splits` every loop, several attention passes per loop, a chunked and
checkpointed head at 4x, no `torch.compile`, no CUDA graphs. In order: profile the step;
`torch.compile` the dense decoder and the shared MLP; remove the per-loop host syncs; a bigger
micro-batch on the 80 GB card; the fused linear plus cross-entropy head and the subsampled MTP
head; then FP8 through Transformer Engine, validated by a 2B-token A/B within about 1% of BF16.

**Gate G8**: at least 2x sustained tokens per second over the 97k/s baseline at equivalent shape,
measured over an hour with checkpointing and upload live; FP8 A/B loss-matched.

### 6b. Budget

Shape L costs about 1.0 GFLOP per token at 3 passes, twice today, and evidence multiplies it by
about 1.25 at the corpus ratio of 1.3 (QA buffer capped). At the plan's base rate that is about
350B tokens in 2.5k H100 hours; every G8 multiplier scales it. The data cap is 100B unique tokens
at 4 epochs. Recompute after 6a; the FLOP line the constructor prints is the anchor, and it goes
stale silently.

### 6c. The corpus at 100B tokens

The pilot builder mode at scale. Offline cost estimates at 100B tokens: the fact tagger about
15 H100 hours; bge window queries about 5; about 400M ANN queries against the store, CPU hours with
a flat index on GPU; the near-duplicate filter negligible; store deduplication minutes. Corpus
prep is its own interruption-safe job with its own budget line in the run spec.

### 6d. The run spec (`docs/plans/RUN2.md`)

Written before anything is rented. It names, per component, the pilot rung that admitted it and
its margin. Shape L from the pilot's R4b reading. An LR transfer sweep at reduced width, or muP if
it can be adopted cheaply. The gate table A1 to A6 and L1, L2, with the seed noise measured. The
two-phase curriculum with a reasoning-weighted tail. The per-loop and per-pass readouts in the log
from step 0. The early signals for a run whose kill rules no longer have zero-init tensors to
watch: the selector's gold mass per pass, the reader gain per pass, the supported-span share of the
loss, and A3 at every checkpoint.

### 6e. After the main run

- **Context extension**: 8k or 16k on long documents with retrieval attached, rotary base raised or
  YaRN-scaled, a few billion tokens. A context change is also a corpus rebuild, because packing,
  fill and the per-row evidence cap all change with the row length. Sliding-window attention in
  some prelude layers only if this phase shows the cost matters.
- **Consolidation SFT** on the real trunk with the evidence segment in the template, then the
  preference pass on abstention pairs mined from the model's own samples, if the groundedness
  signal did not bend the abstention curve on its own. Gate G7 as before: benchmarks within noise,
  false abstention under 10% with recall at least 0.5 and precision at least 0.65, A-gates
  re-confirmed on the shipped checkpoint.

---

## Acceptance (all gates)

Kept results:

- **G0** benchmark harness and noise floor: PASS 2026-08-26. The seed-noise half is still open.
- **G1** IR ablation: FAIL, 0.0004 / 0.0002 nats.
- **G2** sharpening: FAIL, ablation at 0.0002 nats on both arms.
- **G2b** scale fix: FAIL, 0.0002 nats; `g_proj` stalled at RMS 0.0047.
- **G2c** loop input injection: FAIL, +0.0007 / +0.0009 against a 0.01 bar, stopped at 123M tokens.
- **P0** head removal neutral: PASS. **P2** abstention repair: PASS on false abstention, recall
  open.

Live:

- **G3** fixed-target gold-minus-none gap at least 50% of the 3.23-nat ceiling on held-out rows;
  distractor minus none at most 0; benchmarks within noise (Phase 4).
- **G3b** selector AUROC at least 0.674 per loop; groundedness-head AUROC scored beside it (Phase 4).
- **G9** the oracle-head reading recorded; tie the head if the factoring costs at least 0.02 nats
  (Phase 4b).
- **A1 to A6, L1, L2** as in "What must be proven" (Phase 5 for the pilot reading, Phase 6 on the
  real run).
- **G8** throughput (Phase 6a).
- **G7** the shipped checkpoint (Phase 6e).

Retired: G4 (retriever alignment is now the InfoNCE term inside the pilot, read by A5), G5 and G6
(subsumed by A6 and the benchmark evidence path; the beyond-context claim is read as HotpotQA with
64-plus chunks through the port against the best in-prompt packing, at linear cost), R1 to R5 (A1
to A4 are their measurable forms).

## Risks

- **Facts reach the weights through the input side.** Masking unsupported spans is necessary and
  not sufficient; later tokens still learn entity-to-attribute links from the input. The
  counterfactual swaps and the tail anonymization are the levers, and their rates are guesses.
  Without them A1 is expected to fail.
- **Small models ignore evidence.** Models of 7B and under have been measured ignoring oracle
  passages 85 to 100% of the time on questions they cannot answer alone. At 120M the pilot may
  come out flat. The matched control and A6 decide, and the 1-hop curve must saturate first.
- **Template learning without transfer.** Synthetic chains can pass while HotpotQA stays flat;
  published second hops fail off template, and BoolQ here is below chance. R5 catches it.
- **The bridge never reaches the query.** At loop 1 the read is one summand under `post_norm`; the
  answer position may not carry the bridge at loop 2. If loops collapse again (`cos(d3, d2)`
  high) the explicit reader-to-query path comes back as an arm.
- **Retrieval mismatch and leakage.** The reader trains on bge's errors and serves on its own;
  near-duplicate continuations inflate the language-model gain while the selector learns topic
  overlap. The candidate refresh in the last 15% of the pilot, the overlap-tier report and the
  same-document exclusion are the mitigations.
- **Context reliance decays.** It rises early and then falls under finetuning on context that
  agrees with the weights. A3 is read at every checkpoint, with a kill on the trend.
- **Coverage.** With the document itself excluded, a fact that appears in one document only is
  never supported: masked, never memorized, never practised. The store must be at least 0.4B
  tokens and its coverage is a build metric.
- **Reader bandwidth.** One read per sublayer in two of three sublayers is still narrower than
  per-layer designs. The dense-layer reads are the prepared arm.
- **Ambiguous pilot null.** 1B tokens at 120M with 10% synthetic may undertrain. The prefix and
  inject interaction and the depth-5 memory are untested. R1b and the saturating 1-hop curve are
  the guards.
- **Benchmark contamination.** Standard val and test splits only; a single suspiciously strong
  benchmark is suspect, not a win; the A1 probe answers are held out of the pretraining text.
- **FP8 divergence.** Validated on the 2B-token A/B before the real run trusts it.
- **Data prep at 100B tokens is a real job**, budgeted in the run spec.
- **Stale numbers.** Every FLOP, parameter and ratio figure in this plan is an estimate until the
  constructor and the builder print it. Budget math keyed to an estimate goes stale silently.

## Parked

- **Learned halting.** Only trainable if halting skips real compute; unparks after the exit and KV
  cache exclusion is fixed and a halt decision breaks the loop during training.
- **RL and reasoning training.** Unparks when pass@8 on the target task exceeds about 15%. A
  looped model wants per-loop latent credit (LoopRPT, RLTT), not token-level GRPO.
- **Per-pass re-query and the append-only buffer at serving.** Untrained; after the pilot.
- **Explicit reader-to-query feedback.** Query drift was measured without evidence; revisit only
  if it stays flat with evidence attached.
- **Evidence compression** (chunk embeddings in place of token states). A fallback for the memory
  wall; high compression costs 30 to 40% of capability in the published results.
- **Masked-diffusion trunk.** Unparks only if the budget math lands data-bound by 2x or more.
- **Hierarchical or adaptive softmax.** Changes the normalizer every calibration number assumes.

## What changed from the previous plan

- The goal is stated over the pathway and the strong form is replaced by A1 with A3 and A4.
- The learned IR table, the selector's value read and adapters, the chunk gate and the factored
  heads leave the real run. The selector becomes always on; the head is tied.
- Phase 4 is a mechanism check with fixed instruments, not the main POC spend.
- Phase 5 (retriever alignment and the real index) is folded into the pilot: the InfoNCE term,
  the store, the near-duplicate filter and the evidence path in the benchmarks.
- The loop gets a readout stage, one loss on the last pass, a depth schedule to 5 with a detached
  prefix, fresh loop scales, input re-presentation and a union selection loss.
- The from-scratch corpus attaches retrieval to every slice, masks unsupported fact spans, swaps
  entities, anonymizes the tail, caps the QA buffer, and includes synthetic chains.
- The real run targets shape L at about 2.5k H100 hours with a context extension phase after it.

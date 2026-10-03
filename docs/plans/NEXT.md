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

## Now (2026-10-03)

**The from-scratch reader reads, once it is the model's own attention.** R0b ran on 2026-10-02 and
2026-10-03 ([r0b_micro_pilot.md](../measurements/r0b_micro_pilot.md)), all at the micro shape (35M,
`config_micro.yaml`), from one seed, on `data/prepared_inject`:

- **Arm (a), full CE, 300M tokens: the instrument reads.** Closed-book entity `delta` in
  distribution 0.009 / 0.063 / 0.378 / 0.467 at tiers 1 / 10 / 100 / 1000 (tier 1000 rank 0,
  top-1 1.000); on the held-out template only 0.001 / 0.009 / 0.023 / 0.104. Memorization is mostly
  surface form. In context, the same checkpoint copies a card from its prompt at 0.29 / 0.26 / 0.12
  / 0.003 (in distribution) and 0.20 flat over tiers (held out).
- **The 20M arm (c) smoke was read before any copy circuit existed.** The new prompt read
  (`closed_book_rank.py bios --evidence prompt`, the card as prompt text, no port) equals the
  closed-book read for entities at every tier on that save (0.491 / 0.496 / 0.480 / 0.394), so the
  smoke said nothing about the reader.
- **At matched 100M tokens the cross attention reader does not copy and the key/value reader
  does.** The key/value reader (`sft.py --reader-kv`: the evidence states become leading keys and
  values of `shared_attn` in every loop, through its own projections, rotated as if the evidence
  were the text right before the document) ranks the gold card's value at 0.003 / 0.003 / 0.002 /
  0.000 in distribution (top-1 0.92, flat in exposure) and 0.02 held out, and follows a swapped card
  92% of the time with a memorization ratio under 0.03. The cross reader, resumed to 100M as the
  matched control, ranks it at 0.47 / 0.46 / 0.35 / 0.02 and follows a swapped card 2% of the time,
  although the same model copies from its prompt at 0.10 to 0.13. Training the key/value read also
  trained in-context copying (prompt 0.003). The selector separated in both runs by 100M on the
  selection loss alone (gate frozen): external mass 0.39 to 0.46 with gold, 0.02 to 0.12 without.
- **Arm (c) already leaks at 100M under either reader.** Closed-book `delta` in distribution at
  tier 100 / 1000: cross 0.037 (z 10.5) / 0.203 (z 20.3), key/value 0.026 (z 5.2) / 0.149 (z 12.3),
  arm (a) at the same tokens 0.099 / 0.332. The reader that copies leaks least but still fails the
  pass rule at tier 100. These are mid-cosine readings; the full arms decide.

Earlier, still binding: Phase 4's graft arms were killed at 10.08M tokens on the content gain
(+0.058 / +0.081, [evidence_arm_a.md](../measurements/evidence_arm_a.md),
[evidence_arm_b.md](../measurements/evidence_arm_b.md)); the graft branch is closed (seven failed
gates, one graft condition). The in-context ceiling on `evidence_fixed` is 3.18 nats pooled
([evidence_ceiling.md](../measurements/evidence_ceiling.md) section 5), so the 10M arms reached a
sixth of it on gold gain and 2% on content gain. R0 failed: `loop_scale` stays at the trunk's rate
in graft arms.

**The ladder, in order** (done steps kept so the order reads whole):

1. **Done 2026-10-02: arm (a) in full.** The instrument reads (above).
2. **Done 2026-10-03: the copy control.** `--evidence prompt` on the 20M arm (c) save and on arm
   (a) final. It showed the 20M smoke was read before copying existed.
3. **Done 2026-10-03: the reader redesign and its matched control.** The key/value reader built and
   tested (`tests/test_evidence_kv_reader.py`), a 100M arm (c) smoke with it, and the cross reader
   resumed to the same 100M. The key/value reader copies; the cross reader does not.
4. **Arm (b), `inject_masked`, in full** (0.3B tokens, about 90 minutes, no reader involved). The
   input-side leak test, and the third arm `compare` needs.
5. **Arm (c) in full with the key/value reader.** Resume `inject_retrieval_kv` from its 100M save
   with the same flags (`--reader-kv` included; the mode is in the checkpoint and a relaunch
   without the flag fails the strict load). Every arm (c) save gets four reads, `--evidence gold`,
   `swapped`, `none` and `prompt`, at `--batch-size 1024`, json to
   `ckpts/inject/rank_<run><tokens>_<mode>.json`. Then `compare full masked retrieval_kv`, one
   verdict per form, under the pass rule (R0b in the Phase 5 ladder: tier 1000 in distribution
   read alongside tier 100). The cross reader is closed at micro scale; its 100M save stays as the control.
6. **If (c) still climbs at 300M**, the plan's lever for "only (c) climbs" applies: one sweep arm
   at the higher swap rate (`--suffix s30a50`), key/value reader. If (b) climbs, the anonymization
   rate (`s15a90`). One sweep arm per leak.
7. **The loop reading on arm (a), cheap and eval only.** The storage half of the loop axiom
   (Decisions): closed-book rank by loop count, `n_loops` 1, 2, 3 on arm (a) final (needs an
   `--n-loops` flag on `closed_book_rank.py bios`; the micro arms trained every exit through
   `loop_count_sampling`, so depths 1 and 2 are trained readouts).
8. **Learned depth allocation on the loop axis**, after the reader works on the full arm (c):
   an entropy-regularized exit distribution over depths, as in Ouro (arXiv 2510.25741), in place of
   the fixed depth draw, as a micro arm against the fixed schedule.
9. **The seed-side instruments still open**: the counterfactual eval and PopQA on the seed and both
   10M checkpoints. Both 10M arms abstain on 99.9% of answerable rows and the seed on 46.5%, so the
   counterfactual is read with `eval_abstention.py --counterfactual-likelihood-only` (mr_ll by
   stratum, no generation, no gold condition needed). The store build, the edit store and the
   chain splits are CPU work that can run beside any GPU arm.
10. **Then the pilot** (Phase 5), with the key/value reader and its rotation, after R0b passes;
    the ladder is cut to the budget before launch.

Batch caveat: `config_micro.yaml` moved from batch 16 x accumulate 2 to 32 x 1 for throughput.
That changes nothing the induction-head result keys on: formation is set by tokens per update x
updates (batch x context, arXiv 2511.16893), and both settings put 32 rows of 1024 tokens into
every optimizer step. Copying emerged between 20M and 100M here, so nothing is blocking. If a later
arm copies late or weakly, the lever is fewer tokens per optimizer step (16 x 1, twice the updates
per token, a larger batch giving weaker heads), not the accumulation split.

Relaunching: a stopped arm is relaunched with the same `-c` and every flag (`--data-dir`,
`--train-split`, `--val-split`, `--reader-kv`, `--run-name`). The STOP file stays in the run
directory after a stop (`ckpts/evidence_inject_retrieval/STOP`, `.../evidence_inject_retrieval_kv/STOP`);
delete it before a resume or the run exits at its first poll. Throughput at the micro shape: 55k to
60k tok/s typical, 13.3 GB peak with the card buffer, under either reader.

Seed baselines for the graft lineage (2026-09-30, unchanged): fixed-target gold gain -0.0020 nats,
per-token chunk AUROC 0.515 / 0.484 / 0.490 by loop, mass/chunk AUROC 0.481 / 0.470 / 0.469,
grounded AUROC 0.5000; `g_proj` and `direct_gate` exactly zero. Corpus `evidence_train` 963,011
conversations, ratio 3.75; held-out `evidence_dev` and `evidence_fixed` from `--heldout`.

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
- **Reasoning over evidence**: the selector and the reader query the state at their read site,
  and chunks are encoded independently, so a chunk that depends on a fact in another chunk is
  unreachable until that fact is in the state. k hops over evidence need at least k sequential
  read sites before readout. In the built POC that is one read per loop from `step_input`, so
  there k hops need k loops. In the pilot design the reader sits in sublayers 1 and 3 of every
  pass, so 2 hops fit in one pass (4 with the prelude-read arm). The architecture permits depth;
  the data and the objective have to ask for it, and whether the loop, rather than the sublayers
  of one pass, does the composing is an axiom with a falsifier (Decisions, L1).

## What must be proven

The gates below replace the POC's older G-gates for everything after Phase 4. Each is a metric, a
dataset, a bar against a noise floor, and a script that holds it. Thresholds are relative to
[noise_floor.md](../measurements/noise_floor.md); it has no CE sigma yet, so a paired-bootstrap CE
sigma is recorded before any CE gate is quoted.

| gate | metric | bar | home |
|---|---|---|---|
| A1 closed-book externalization | (a) closed-book likelihood rank (or MC accuracy) of the gold answer among at least 20 same-type candidates, pathway off, neutral prompt, head and tail obscurity tiers, entity-valued relations; (b) fact injection: fictional biographies (bioS style, Allen-Zhu and Li, Physics of Language Models 3.1) inserted at 1 / 10 / 100 / 1000 exposures with their chunks in the store | (a) tail-tier rank worse than the matched no-retrieval control by at least 3 sigma (bootstrap); (b) closed-book rank at chance up to 100 exposures where the control memorizes | new closed-book rank scorer, PopQA and the tiers (Phase 4b build item); R0b |
| A2 evidence recovery | fixed-target answer-span CE on held-out rows, gain = CE(none) - CE(cond) | gold gain at least 1.6 nats (50% of the 3.23-nat in-context ceiling); distractors gain at least 0 | `sft.py` `[eval fixed]`; `eval_abstention.py --evidence-port` cross-check |
| A3 counterfactual faithfulness | memorization ratio p_orig / (p_orig + p_swap) (Longpre et al. 2021) under an entity-swapped gold chunk, on items answered correctly with the unswapped chunk, stratified by entity frequency, injected high-exposure facts included so conflicts exist at pilot scale; n reported per stratum | memorization ratio at most 0.05 and follow rate at least 90% in every stratum | new condition in `eval_abstention.py` |
| A4 store edit | edit or delete the gold chunk and its near neighbours in the store, re-retrieve with the model's own query, answer; not the oracle buffer | flip rate at least 70% on items answered correctly before the edit | two-store swap through the model-query retrieval path |
| A5 selector | chunk AUROC (per token, chance 0.5) per loop; mass/chunk AUROC gold-present vs distractors; grounded AUROC (evidence rows) | chunk AUROC at least 0.59 per loop (3 sigma of 0.030 over 0.5); mass/chunk and grounded AUROC at least 0.674 (0.584 + 3 x 0.030) | `sft.py` `[eval fixed]` (Phase 4 home); `eval_abstention.py` cross-check |
| A6 reasoning retained | HotpotQA, BoolQ and SciQ through the port against in-prompt, counted only on tasks whose in-prompt score is at least 3 sigma above chance; language benchmarks within noise | port at least in-prompt minus 2 points, judged at the binomial sigma; control is the pilot's own in-prompt score | `eval_benchmarks.py` with an evidence path |
| A7 store-level retrieval | held-out NQ, TriviaQA and HotpotQA with gold passages in the store: recall@k of the model's query; open-book EM through the whole pathway | model-query recall@k at least bge-query recall@k; pathway EM read against oracle-buffer EM | new, Phase 4b |
| L1 composition | synthetic k-hop chains with fictional entities, accuracy by hops and by sequential read sites, at fixed D = 3 with the reader output zeroed at every site after site j (read ablation) | every cell with fewer read sites than hops at chance, 1 / (type-matched candidate answers in the buffer); held-out-template 2-hop above the control (the read-ablated model or the per-loop-CE arm) | synthetic eval, Phase 4b |
| L2 depth pays | held-out 4-hop and HotpotQA bridge answer CE by read sites at D = 3 (read ablation), plus D = 4 against D = 3 | rises with read sites past the first pass; at least 0.05 nats on bridge CE | synthetic eval plus `eval_abstention.py` |

The old A1 bar (tail tiers near chance, at most 2x the seed) could not separate "facts
externalized" from "too small to know": pythia-410m at 300B tokens already reads TriviaQA 0.021
and NQ 0.000, under that bar, and every 120M pilot control would pass it. The rank bar is relative
to the matched control, and the injection probe makes exposure count the variable. The A1 claim
covers entity-valued relations only, because the fact tagger covers only those; extending it to
attribute nouns (occupation, nationality, genre) extends the claim. A3 is read at every
checkpoint, because context reliance is known to rise early and then decay under finetuning on
context that agrees with the weights; the injected high-exposure facts are what makes a
weights-against-chunk conflict exist at pilot scale.

L1's validity rule: hops are bounded by sequential read sites before readout, so anything above
chance in a cell with fewer sites than hops is a leak in the data. It is scored by read ablation at
fixed D = 3, not by truncating depth: the minimum depth is 3, so 1- and 2-loop exits are untrained
and a truncated cell at chance would be readout failure. Chance is 1 over the number of
type-matched candidate answers in the buffer. The control is never the no-retrieval arm, which
cannot see the chains.

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
| goal | The reworded goal above. Strong form is not a gate; A1 with A3, A4 and A7 is. |
| roles | Selector = the IR path, chooses by embedding. Reader = the evidence tokens as leading keys and values of the shared self-attention in every loop, carries content. Measured 2026-10-03 at micro scale: at matched 100M tokens this read copies (gold card top-1 0.92, swapped card followed 92%) where the separate cross-attention reader does not (top-1 0.01, followed 2%). The cross reader stays loadable as the legacy mode. |
| learned IR table | Dropped from the real run. Zeroing its read costs 0.0002 nats on every arm. |
| selector in the real run | Always on, not routed; scores raw bge keys plus one learned null key; no value read, no key adapter. |
| selector to reader coupling | The selector's own logit enters the reader's attention logit through a one-hot chunk channel, per token, causal. Replaces the document-mean sigmoid gate. Carries over to the key/value read as a head-dim extension: the channel on the evidence keys, zeros on the segment's own keys. |
| reader positions | Prefix rotation: each segment's evidence keys are rotated at the positions the evidence would hold as the text right before the segment, so a query sees the offsets of in-context copying. Supersedes "no rotary", which was a cross-reader decision (Phase 4 arm B: +40% content gain without rotary, still an order under the bar). |
| reader output | No separate output: the read lands through the shared self-attention's own `o_proj`, inside the always-on seed; `evidence_loop_scale[k]` (init 1) scales the evidence values per loop. Not neutral when evidence is attached (the read shares the softmax), which from-scratch training does not need; with no evidence attached the forward is bit-identical to the model without the port. |
| evidence encoder | The trunk's own dense decoder, chunk-causal. All 8 layers in the POC; the first 4 prelude layers under TE checkpoint from scratch. No cached encoder states in the store. |
| readout | A 2-layer coda after the loop, own KV slots. The loop refines, the coda reads out. |
| loss placement | One CE on the last pass run, through the coda. Per-loop CE is the control arm, not the default. |
| depth | Drawn per step from {3: 0.7, 4: 0.2, 5: 0.1}; the first D minus 3 passes run under `no_grad` and are detached; minimum 3, so no pass is an exit. Inference 3, at most 4. |
| loop tensors | `loop_scale` and `evidence_loop_scale` length 5, fresh `1/sqrt(5)`; `loop_inject` on; loop code clamped to the trained depth. |
| selection supervision | Union over the passes in the gradient window: `u_c = 1 - prod(1 - p_hat[k, c])`, BCE against the gold flag. Never a target per hop and per loop. |
| query training | InfoNCE on the in-model query against the bge key of the gold chunk, in-batch negatives plus store-mined hard negatives, same-document and near-duplicate negatives masked. Keys stay bge, so the index is never rebuilt; Atlas (Izacard et al. 2022) justifies query-side-only tuning against a fixed index. |
| token tables | One table, tied to the LM head and the MTP head through small adapters; no weight decay on it. The per-layer and router tables become projections of it, gated on the per-layer table zero-ablation on the seed (Phase 4b). G9 gates the head tie only. |
| shape | Shape L: width 1024, 16 heads x 64, 4 KV heads; 8 prelude layers; looped core of 3 sublayers per pass with 8 MLP experts each, top 2; 2 coda layers. About 470M total, 300M active per token. Sublayer count and expert count are pilot arms. |
| context | 4096 for the pilot and the main run; a short 8k to 16k extension phase at the end on long documents with retrieval attached. Never 32k from the start. |
| real run ordering | Retrieval-augmented pretraining from token 0 with fact spans kept out of the loss. No plain pretraining plus a graft. |
| real run budget | A token target is fixed and the hours derived from it (6b). About 2.5k H100 hours planned for shape L, which buys about 270 to 290B tokens with evidence at today's MFU; the 5k-hour ceiling stands. Estimates until the constructor prints them; recomputed after Phase 6a's throughput work. |
| loops (axiom) | "Loops buy computation, not storage" is an axiom, not a measurement: the record is 0.008 to 0.012 nats from loop 2 to 3, confounded by the halt gate. Ouro (arXiv 2510.25741) measures looped and non-looped models at the same ~2 bits per parameter, with the loop gain in knowledge manipulation. Two falsifiers. Storage half: closed-book rank by loop count on arm (a) (depth 1, 2, 3; every exit was trained through `loop_count_sampling`); if the tier 100 and 1000 `delta` grows with depth beyond its paired sigma, later loops carry stored facts and the axiom is wrong for this design. Computation half (L1): if read sites past the first pass add under 5 points on held-out 2-hop at D = 3, the loop clause of the goal is wrong. Looping stays a requirement either way. |
| depth allocation | The planned loop fix, after the reader works on the full arm (c): learned depth allocation on the loop axis, an entropy-regularized exit distribution over depths (Ouro), tried as a micro arm against the fixed depth draw before it replaces the draw in the pilot. |
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
  The 0.55 arm (the shipped repair): false abstention 0.783 to 0.161, recall 0.81 to 0.22
  (the 0.40 arm reached 0.136 false abstention at recall 0.18); precision pinned at 0.578 six
  readings running. The data lever is exhausted.
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
- **R0b, first from-scratch runs** (2026-10-02 and 03,
  [r0b_micro_pilot.md](../measurements/r0b_micro_pilot.md)): arm (a) reads the instrument; the
  key/value reader copies from the card at 100M where the cross reader does not; arm (c) leaks at
  tier 100 at 100M under both. Arms (b) and (c) in full are open.

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
- No per-loop number on record is free of the halt gate: it passed 8% of the loop-3 update for all
  16B pretraining tokens, and every later arm grafted onto that lineage. Only the from-scratch
  pilot can separate "loops cannot reason" from "loops were never allowed to".
- The trunk is already near-empty of facts (TriviaQA 0.014, NQ-open 0.002). The pending run tests
  reader capacity, not fact removal.
- Depth past 3 degrades on plain text: CE 3.4112 / 3.4115 / 3.4213 / 3.4384 / 3.4618 / 3.4900 at
  depth 3 to 8 on `ir_c`, with no evidence attached
  ([loop_scale_probe.md](../measurements/loop_scale_probe.md), "Depth past the trained 3").

"Loops buy computation, not storage" is no longer listed here: nothing measured supports it, so it
is an axiom under Decisions, next to its falsifier.

---

## Phase 4: the evidence finetune as a mechanism check

One question: can the port be read at all. It runs the architecture as built, on the grafted
checkpoint, with the gate frozen. It cannot show where facts live: the seed scores near zero
closed-book, and between 54 and 92% of the corpus's supervised tokens (depending on the weighting)
carry content the buffer does not supply.

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

**Follow-up, 2026-09-30.** The kill checks two numbers (gold gain and content gain, gold minus
distractors) and is armed again when the fixed pass read nothing; `kill_checked` is persisted in
the checkpoint payload. The fixed pass prints a per-loop reader gain from the per-loop CE of the
same forward and a per-token chunk AUROC (chance 0.5) beside the old pooled one. The selection
loss supervises the positions that produce a supervised token (`labels[:, 1:] != -100` shifted to
`:-1`), the positions `answer_start_positions` and the readouts use; the old `labels != -100` was
off by one. `checkpoint_every_tokens: 10000000` puts a save at the decision.

The eight items as specified, in the order of how badly each would have misled the run:

1. **The kill number.** `sft.py`'s per-condition `[eval]` CE uses each row's own target, the answer
   under `gold` and a refusal under `none`, and read minus 3.25 nats on a seed with a dead reader
   (on the old leaked split; `evidence_dev` reads about -2.3). The `none` bucket also holds replay
   rows. The fixed-target pass teacher-forces the real answer under every condition over the
   natively answerable QA rows and prints gain = CE(none) - CE(cond); the kill reads it. The seed
   reads -0.0020 there, and `eval_abstention.py --evidence-port` reads +0.0000 as a cross-check.
2. **The validation split.** `prepare_evidence_data.py` split per rendered row after the QA
   sources repeat, so every QA validation question was also in train (7,440 of 7,440). The
   held-out splits `evidence_dev` and `evidence_fixed` come from SQuAD dev and HotpotQA dev, split
   by source id before the condition draw. The current `evidence_val` is a train-loss slice.
3. **Cadence.** `eval_every_tokens: 2500000`, and an eval before step 1, so the decision at 10M
   tokens has a step-0 baseline plus four readings at 2.5 / 5 / 7.5 / 10M, the fourth being the
   decision. Warmup is 99 optimizer steps, about 4.2M tokens.
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
   uncached decode with evidence attached. Plus an assert on the `is_fresh_loop_param` predicate
   and the by-hand gradient clear in `train_step`, which route exactly the tensors this run trains
   (`test_fresh_param_routing`, a GPU test: it builds the model).

### The run

- Arm A: as built, gate frozen. Arm B: same, reader without rotary (`--reader-no-rotary
  --run-name norope`). 10M tokens each, 11 to 19 minutes each at 6.0k to 14.8k tok/s (about 10k
  typical), in the background under a Monitor watch teed to a gitignored log.
- Sign convention, stated once: gain = CE(none) - CE(cond) on the fixed-target answer span;
  positive means the evidence helped.
- Kill: at the first fixed-target eval past 10M tokens the run saves and exits 10 if the gold gain
  or the content gain (gold minus distractors) is under 0.1 nats. The check is armed again when
  the fixed pass read nothing; `kill_checked` is persisted, so a resume does not decide twice.
- Stop by hand after the 10M reading: `touch ckpts/evidence/STOP` (arm A) or
  `touch ckpts/evidence_norope/STOP` (arm B). Polled every 10 micro steps; the run saves and exits
  10. The glob form does not work. Relaunch a killed or stopped arm with the same `-c` and the
  same flags: without `-c` the yaml builds a model without the port and the strict load fails.
- Pass: gold gain at least 1.6 nats and distractors gain at least 0. The printed ceiling fraction
  is against the SQuAD-only 3.23 nats, measured on other rows and another checkpoint, so it is
  indicative until the Phase 4b ceiling on the `evidence_fixed` rows exists.
- Read per loop, all from the same `[eval fixed]` forward: the reader gain lines (final loop equals
  the headline), external mass and mass per chunk per condition, mass/chunk AUROC gold-present vs
  distractors, chunk AUROC (per token, chance 0.5) and gold share. A rising
  `|shared_evidence.o_proj|rms` with a flat gain means the reader learned a bias, not retrieval;
  RMS alone is not the signal.
- Log gradient norms of `key_adapter`, `value_adapter`, `down_proj` and `loop_query_bias` at steps
  10 and 100.
- Under 50% of the ceiling: the next arm is a reader per loop or reads in dense layers, before more
  data. Arm C, only after a pass: the one-hot chunk channel with `gamma = 0` (arm C only; from
  scratch `gamma` starts at 1).

### What it decides

A pass says the grafted reader can carry a span; it does not change the real run's design. A fail
on the graft is weak evidence, given the record, but it changes which reader arm the pilot runs
first (a reader per loop or dense-layer reads). Phase 5 does not wait for it either way.

**Gate G3** (restated): gold gain on the held-out fixed-target rows at least 1.6 nats, 50% of the
ceiling; distractors gain at least 0; benchmarks within noise. **Gate G3b**: chunk AUROC (per
token, chance 0.5) at least 0.59 per loop (3 sigma of 0.030 over 0.5); mass/chunk AUROC and
grounded AUROC (evidence rows) at least 0.674, three sigma over the 0.584 probe (0.65 was 2.2
sigma). `sft.py`'s `[eval fixed]` is the home; `eval_abstention.py --evidence-port` is the
cross-check.

---

## Phase 4b: instruments

Everything the goal is measured with. Most of it is eval-only code and runs on existing
checkpoints; it is on the critical path because the pilot cannot be read without it.

- **R0, the loop scale probe.** `eval_stage0.py` gets a flag that multiplies `loop_scale` at eval.
  Run on `ir_c` and `phase2_final` with factors 1, 2, 3.5 and 5.9. Read the CE gain from loop 2 to
  3. Pass at 0.02 nats with loops 1 and 2 not worse. Decides whether `loop_scale` joins the fresh
  LR group in graft arms. **FAIL 2026-09-30** on both checkpoints, monotone in the multiplier
  ([loop_scale_probe.md](../measurements/loop_scale_probe.md)): it does not join.
- **R1b, the depth memory probe.** One micro step at depth 3 and at depth 5 with the detached
  prefix, pilot shape; read peak memory. Decides whether the depth schedule is trainable.
- **The in-context ceiling on the `evidence_fixed` rows**, read with the seed, and a source
  sidecar on `evidence_fixed` so every gain splits per source (SQuAD, HotpotQA). The printed
  ceiling fraction is against the SQuAD-only 3.23 nats on other rows and another checkpoint until
  then.
- **A1, the closed-book rank scorer.** Likelihood rank (or MC accuracy) of the gold answer among
  at least 20 same-type candidates (100 in R0b), pathway off, neutral prompt that does not licence
  abstention, head and tail obscurity tiers, bootstrap sigma. PopQA and the tiers are not in
  `eval_benchmarks.py` today: a build item. Plus the fictional biography generator (bioS style) with
  per-fact exposure counts 1 / 10 / 100 / 1000 and the matching store chunks, shared with R0b.
- **A3 in `eval_abstention.py`.** A counterfactual condition following the Faithfulness-QA recipe:
  swap the answer entity in the gold chunk for a same-type entity, and report the memorization
  ratio p_orig / (p_orig + p_swap) (Longpre et al. 2021) and the follow rate on items answered
  correctly with the unswapped chunk, stratified by entity frequency, with n per stratum. Injected
  high-exposure facts are included so conflicts exist at pilot scale. Per-record JSON in evidence
  mode, EM and non-abstain rate under distractors and none.
- **A4, the store edit.** Two stores on the same questions: the gold chunk edited or deleted with
  its near neighbours; re-retrieve with the model's own query, not the oracle buffer; report the
  flip rate on items answered correctly before the edit.
- **A5 readouts.** Built in `sft.py`'s `[eval fixed]` (chunk AUROC per token, mass/chunk AUROC,
  grounded AUROC, per loop). Left for the cross-check: gold-chunk mass per IR expert in
  `eval_abstention.py` (it reads `ir_modules[0]` only today).
- **A7, store-level retrieval.** Held-out NQ, TriviaQA and HotpotQA with the gold passages in the
  store; recall@k of the model's query against bge's; open-book EM through the whole pathway
  against the oracle-buffer EM.
- **The synthetic chain generator and eval** (L1, L2). Fictional entities, hops 1 / 2 / 3, 6 to 14
  distractors of three kinds (same relation with other entities, bridge relation with other
  entities, one full decoy chain), shuffled chunk order, a hop index per gold chunk written to an
  `.evhop` sidecar for evaluation only, 4-hop chains and unseen hop-2 templates held out. Score
  accuracy by hops and by sequential read sites at fixed D = 3, zeroing the reader output at every
  site after site j. The validity rule: every cell with fewer read sites than hops sits at chance,
  1 / (type-matched candidate answers in the buffer).
- **An evidence path in `eval_benchmarks.py`.** SciQ and BoolQ carry their passage; put it through
  the port and diff against the in-prompt score (A6, only where in-prompt is 3 sigma over chance).
  TriviaQA-rc and NQ with DPR gold passages for the attach delta. HotpotQA validation with both
  gold paragraphs in the port, by read site.
- **Per-loop readouts in the training log.** `cos(q_k, q_k+1)`, per-loop selector mass, per-loop
  reader gain, and `||h||` per loop (the readout blind spot from the first review).
- **The oracle head (G9).** Train a dense `768 -> 65536` head on frozen final-loop states for
  100 to 200M tokens; `CE_factored - CE_oracle` is the damage the block-diagonal head does. Tie if
  at least 0.02 nats. Plus the free reading: least-squares fit of `E A^T` to the trained head. G9 is
  the head tie only.
- **The per-layer table zero-ablation on the seed** (the PLE ablation). Zero the per-layer table's
  read and measure CE; near the IR table's 0.0002 nats, the per-layer and router tables become
  projections of `E`; otherwise the per-layer table stays.
- **A paired-bootstrap CE sigma** on the standard slice, so "within noise" is defined for CE.

---

## Phase 5: the from-scratch pilot

The first experiment that can falsify the goal. About 120M parameters at width 512, 1 to 2B
tokens per arm on the 5090, with a matched-token control without retrieval. Run times are
unmeasured; the R1b memory probe comes first, and the R0b micro-pilot (ladder below) reads the
externalization recipe before any 1B-token arm.

### The model, from scratch

Every item is in the design page's from-scratch blueprint. Grouped by what it touches.

**Token tables and readout.**
- One table `E`, init std `1/sqrt(hidden)`, no weight decay (it is tied, so its input and output
  roles cannot be decayed separately). LM head `norm(h) @ A @ E^T` with `A` hidden x hidden; MTP
  head through a `hidden/2 -> hidden` adapter onto `E^T`, CE on a 25% token subsample. Fused
  linear plus cross-entropy in `_chunked_linear_ce`. The per-layer and router tables become
  `E W_ple` and `E W_moe` if the PLE ablation (Phase 4b) lands near the IR table's 0.0002 nats;
  otherwise the per-layer table stays.
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

**Reader.** Superseded in form on 2026-10-03: the measured reader is the key/value read in the shared
self-attention (Decisions, roles and reader positions). The items below carry over to it where they
are about the keys (the one-hot channel, the sink slot, 4 KV heads) and lapse where they are about a
separate module (its own `o_proj`, its own output gate).
- `k'_j = [W_k e_j ; onehot(rank(chunk(j)))]`, `q'_t = [W_q x_t ; sqrt(d) * b_t]`, head dim
  `d + M` rounded to a multiple of 8, `softmax_scale = d^-0.5` passed explicitly, so
  `logit(t, j) = content + b[t, chunk(j)]`. `b[t, c] = gamma_h z[t, c] - lambda log n_c + vis[t, c]`,
  `gamma` per head init 1, `lambda` init 0, `vis` the visibility mask at minus 1e4. One sink slot
  per segment with a learned key and a zero value. No rotary. 4 KV heads. K and V are
  pass-invariant, one cache per buffer. `M` up to 40. Check first that the installed flash-attn
  build accepts the head size.
- Output: after `post_norm`, `h = h + g_loop[k] * RMSNorm(o_proj(read))`, `o_proj` at default
  init, `g_loop` init 1.
- Read sites: sublayers 1 and 3 of every pass, so 2 sequential read sites per pass and 2 hops can
  fit in one pass. Reads in dense prelude layers 3 and 6 are a pilot arm (4 sites before the
  first pass ends).

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

**Objective.** `L = CE x span_weight + 0.1 MTP + 0.01 aux + 0.1 L_sel + 0.1 contrastive + 0.1
groundedness`, with the MTP targets span-weighted like the main CE. The control arm swaps the first
term for per-loop CE.

**Optimizer (pilot).** Its own spec, in tokens: warmup 2 to 5% of the pilot's tokens (the design's
earlier "1000 steps" is 524M tokens at the pretrain batch), the router noise anneal and the
checkpoint cadence scaled to the pilot's budget, LR from a width-512 sweep, cosine to 0.1 lr, no
weight decay on `E` or the loop scales.

### The pilot corpus (a new builder mode)

Retrieval is attached to every slice. The only evidence-free case is the abstention condition.

| slice | tokens | sources | evidence | supervision |
|---|---|---|---|---|
| web and edu text | 45% | the pretrain mix, fineweb-edu weighted up | top 2 chunks per 256-token window, staircase visibility | span weights, counterfactual swaps, tail anonymization |
| code and math | 10% | stackv2 edu, Nemotron math | same | plain CE |
| evidence QA | 15% | SQuAD v2, HotpotQA, NQ, TriviaQA with passages, MuSiQue, 2Wiki; at most 2 passes each | gold, mixed, distractors, none, counterfactual, partial hop; buffer capped at 8 chunks, so no `many` | answer or abstain |
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
  gold. Unsupported: weight 0. Language tokens: weight 1. The MTP targets get the same span
  weight, or the MTP head relearns what the main loss masks. The tagger covers entity-valued
  relations only; attribute nouns (occupation, nationality, genre) stay in the loss, so the A1
  claim is restricted to entity-valued relations unless the tagger is extended to them.
- **Counterfactual swaps.** On 15% of supported spans the entity is swapped for a same-type entity
  in every visible chunk that holds it, in the target, and in every later mention in the document.
- **Tail anonymization.** In 50% of documents, unsupported entities below the head-frequency
  threshold get a typed placeholder in input and target, re-drawn per epoch. The 15% and 50% rates
  are first guesses and get a sweep in the micro-pilot (R0b).
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
| R0 | 0 | loop scale probe on existing checkpoints | CE gain loop 2 to 3 | 0.02 nats, loops 1 and 2 not worse | `loop_scale` in the fresh group for grafts. **FAIL**: it stays at the trunk's rate |
| R1 | 10M | the Phase 4 run | chunk AUROC (per token) and reader gain per loop | descriptive: read, not decided (a 0.02 bar is under one sigma of the 0.030 floor, and the POC objective gives every loop the same target); a decided form needs a 0.09 bar | nothing; describes whether reads differ by loop. **Read 2026-10-01**: both arms killed on the content gain (0.058 / 0.081); chunk AUROC 0.650 / 0.643 / 0.644 by loop and the per-loop content gain 0.070 / 0.078 / 0.081 (arm B), so loop 1 reads best and later loops add little |
| R0b | 3 x 0.3B | externalization micro-pilot, 35M params, about 1.5 5090 hours per arm: (a) full CE, no retrieval; (b) span weights with facts masked, no retrieval (the input-side leak test); (c) span weights plus swaps plus anonymization, with store retrieval through the key/value reader. Fictional biographies injected at 1 / 10 / 100 / 1000 exposures; swap and anonymization rates swept in (c) | closed-book likelihood rank among 100 same-type candidates, by exposure, against a fresh-name prior; on every (c) save also gold, swapped and prompt reads | (b) and (c) within 3 sigma of their prior through tier 100 while (a) is 3 sigma above at tier 100 or 1000; tier 1000 in distribution read alongside: a (b) or (c) arm 3 sigma off its prior there is a leak even when tier 100 holds, because the 95% filler weakens tier-100 storage (Physics 3.3, arXiv 2404.05405), so tier 1000 is where (a) stores reliably and a leak shows first. **(a) read 2026-10-02**: 0.378 (z 51) at tier 100, 0.467 at tier 1000. **(c) at 100M**: 0.026 (z 5.2) at tier 100, 0.149 (z 12.3) at tier 1000 with the key/value reader | if (b) or (c) climb with exposure like (a), the recipe fails before any 1B-token spend |
| R1b | 0 | one micro step at depth 3 and at depth 5 with the prefix, pilot shape | peak memory | fits | whether the depth schedule is trainable |
| R2 | 0 | synthetic chains scored on the R1 checkpoint | accuracy by hops and read sites (loops, on the POC) | after the 1-hop curve saturates: chance wherever read sites are fewer than hops | whether the instrument is valid |
| R3 | 2 x 30M | graft: proposed objective against the current recipe, 15% synthetic, `loop_scale` kept at the migrated values and the trunk's rate | 2-hop accuracy at 3 loops minus 1 loop | +0.15 and proposed beats current | a pass helps; a null decides nothing |
| R4 | 4 x 2 x 1B | from scratch: loss placement by coda, 2 x 2, two seeds per compared arm (training-seed sigma is unmeasured), shared fresh `loop_scale`, inject and data | held-out-template 2-hop at D = 3 | best arm 10 points over per-loop loss without coda, outside the seed spread; 3-hop above chance | objective and coda |
| R4b | 2 x 1B | sublayers per pass 2 against 3 at matched compute; experts 4 against 8 | L1 plus closed-book rank plus language benchmarks | see gates | shape L's core |
| R5 | 0 | the winner, eval only | held-out 4-hop by read sites and at D = 4; HotpotQA bridge by read sites; A1 to A7 | L2 and the A gates | whether the design enters the real run |
| R6 | 1 to 2B | hop 1 seen only in pretraining text (about 100 exposures), hop-1 documents excluded from the store, hop 2 retrieved | closed-book hop-1 accuracy and 2-hop accuracy | at chance on both (the A1 leak probe: above chance means pretraining-text facts reached the weights) | whether the recipe keeps pretraining-text facts out of the weights |

A 1-hop curve must saturate before any null is read. R4 replaces the previous plan's per-loop CE
test, which changed loss placement, depth sampling and the coda in one arm. Pilot arms that do not
fit the budget are dropped from the bottom of this table, not the top; R0b runs before all of them.

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

**Gate G8**: at least 2x the sustained tokens per second that shape L reaches at today's MFU
(about 30 to 32k/s with evidence, the rate behind the 6b estimate; an estimate until measured),
defined in tokens per second at shape L, measured over an hour with checkpointing and upload live;
FP8 A/B loss-matched. "Equivalent shape" is not a definition and is not used.

### 6b. Budget

Shape L costs about 1.3 GFLOP per token forward at 3 passes: body 2 x (109 + 3 x 98 + 27)M =
860M, tied head 134M, MTP at 25% x 2 tokens 67M, attention 160M, reader sites 80M; the expected
depth of 3.4 passes adds to the body. Evidence multiplies it by about 1.29 at the slice-mix ratio
of 1.55 (0.75 x 1.0 + 0.15 x 2 + 0.10 x 5, QA buffer capped). At today's MFU, 2.5k H100 hours buy
about 270 to 290B tokens; shape M buys about 235B per 1k H100 hours.

The data cap is 400B tokens (100B unique at 4 epochs). Passing G8 at 2x doubles the tokens an
hour buys, which makes a 2.5k-hour run data-bound at that cap, so the hours and the token target
cannot both hold. The run spec fixes a token target and derives the hours from the measured rate.
Every figure here is an estimate until the constructor prints it; recompute after 6a, because
the FLOP line is the anchor and it goes stale silently.

### 6c. The corpus at 100B tokens

The pilot builder mode at scale. Offline cost estimates at 100B tokens: the fact tagger about
15 H100 hours; bge window queries about 5; about 400M ANN queries against the store, CPU hours with
a flat index on GPU; the near-duplicate filter negligible; store deduplication minutes. Corpus
prep is its own interruption-safe job with its own budget line in the run spec.

### 6d. The run spec (`docs/plans/RUN2.md`)

Written before anything is rented. It names, per component, the pilot rung that admitted it and
its margin. Shape L from the pilot's R4b reading. A token target, with the hours derived from it.
An LR transfer sweep at reduced width, or muP if it can be adopted cheaply. The gate table A1 to
A7 and L1, L2, with the seed noise measured. The
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

- **G3** fixed-target gold gain, gain = CE(none) - CE(cond), at least 1.6 nats (50% of the
  3.23-nat ceiling) on held-out rows; distractors gain at least 0; benchmarks within noise
  (Phase 4).
- **G3b** chunk AUROC (per token, chance 0.5) at least 0.59 per loop; mass/chunk AUROC and
  grounded AUROC (evidence rows) at least 0.674 (Phase 4).
- **G9** the oracle-head reading recorded; tie the head if the factoring costs at least 0.02 nats
  (Phase 4b). The PLE ablation is a separate reading and gates the per-layer and router tables.
- **A1 to A7, L1, L2** as in "What must be proven" (R0b and Phase 5 for the pilot readings, Phase
  6 on the real run).
- **G8** throughput (Phase 6a).
- **G7** the shipped checkpoint (Phase 6e).

Retired: G4 (retriever alignment is now the InfoNCE term inside the pilot, read by A5), G5 and G6
(subsumed by A6 and the benchmark evidence path; the beyond-context claim is read as HotpotQA with
64-plus chunks through the port against the best in-prompt packing, at linear cost), the previous
plan's retrieval gates R1 to R5 (A1 to A4 are their measurable forms; the ladder rungs R0 to R6
above are a different, live series).

## Risks

- **Facts reach the weights through the input side.** Masking unsupported spans is necessary and
  not sufficient; later tokens still learn entity-to-attribute links from the input. The
  counterfactual swaps and the tail anonymization are the levers, and their rates are guesses.
  Without them A1 is expected to fail. The levers only work if they are complete: MTP targets carry
  the same span weight, a swap reaches every visible chunk and every later mention, the
  anonymization is re-drawn per epoch, and attribute nouns the tagger misses stay learnable. R0b's
  arm (b) is the input-side leak test and sweeps the rates before any 1B-token spend.
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

## Literature behind the 2026-10-03 changes

- Ouro, "Scaling Latent Reasoning via Looped Language Models" (arXiv 2510.25741): looped and
  non-looped models store about 2 bits per parameter alike, the loop gain is knowledge
  manipulation, and depth is allocated by an entropy-regularized learned exit. The storage half of
  the loop falsifier and the depth allocation arm.
- Allen-Zhu and Li, Physics of Language Models 3.3 (arXiv 2404.05405): about 1000 exposures reach
  2 bits per parameter; junk data without domain tags cuts capacity. Why tier 1000 is read
  alongside tier 100 on a 95% filler corpus.
- Allen-Zhu and Li, Physics of Language Models 3.1 (arXiv 2309.14316): unaugmented facts are
  memorized but not extractable. Matches arm (a): tier 1000 perfect on the training templates, 0.10
  on a held-out one.
- "Predicting the Emergence of Induction Heads in Language Model Pretraining" (arXiv 2511.16893):
  the formation point is set by batch size x context size in updates, independent of model size,
  and a larger batch gives weaker heads. The batch caveat in "Now", and the reason a reader that
  reuses the model's own attention inherits copying instead of relearning it.
- Ram et al., "In-Context Retrieval-Augmented Language Models" (arXiv 2302.00083): documents
  prepended to an unchanged LM are read. The prompt copy control, and the design the key/value
  read imitates inside the loop.
- Memorization Sinks (arXiv 2507.09937) and the goldfish loss (Hans et al., NeurIPS 2024, arXiv
  2406.10209): alternative levers against memorization, held for the case where the swap and
  anonymization rates do not stop arm (c) from leaking.

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
- The real run targets shape L with a fixed token target and derived hours (about 2.5k H100 hours
  planned), with a context extension phase after it.
- 2026-09-30 revision after the review: A1 becomes a closed-book rank against the matched control
  plus a fact-injection probe, A3 a memorization ratio, A4 goes through the store, A7 is added,
  L1 and L2 are scored by read ablation at fixed depth, R0b is added, and the budget is recomputed
  at about 1.3 GFLOP per token.
- 2026-10-03: R0b arm (a) read the instrument; the copy control showed the 20M arm (c) smoke was
  read before copying existed; the key/value reader (evidence as leading keys of the shared
  self-attention, prefix rotation) copies at 100M where the cross reader does not at matched
  tokens, and replaces it in the plan; arm (c) leaks at tier 100 at 100M under both readers; the
  pass rule reads tier 1000 in distribution alongside tier 100; the loop axiom gets a storage-half
  falsifier (closed-book rank by loop count on arm (a)) and learned depth allocation is the planned
  loop fix; `eval_abstention.py --counterfactual-likelihood-only` reads abstaining checkpoints.
- 2026-10-01: Phase 4 ran and both arms were killed on the content gain; the pilot's first reader
  arm is a reader per loop or dense prelude reads, without rotary. The graft branch is closed:
  all seven failed gates measured the same graft condition, and no from-scratch result exists yet
  in either direction. Phase 4b and the R0b tooling are built (see "Now"); R0b is the next spend,
  arm (a) first, a 20M smoke of arm (c) as the reader precondition, and the ladder below R0b is
  cut to the budget before the pilot launches. Decisions taken in the build: the micro shape is
  selected by `TINY_LLM_CONFIG` (shape is not inferable from a state dict); the span weight
  travels through `.mask`, so the MTP targets carry it; biography values are fictional and drawn
  uniformly per person, but exposure weighting makes the corpus marginals non-uniform, so the
  closed-book reading is paired against a fresh-name prior control (PopQA likewise against a
  subject-blind control); chain chance is one over the distinct objects of the question's
  final-relation chunks, since the wh-frame names that relation; read ablation cuts the reader
  and the IR expert's external read together at a site; the model-side store index on this
  lineage goes through `key_adapter`, so it is per checkpoint.

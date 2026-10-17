# The from-scratch pilot (Phase 5): the spec, written 2026-10-09 before the build

Status 2026-10-11: spec written 2026-10-09, build list done 2026-10-10 (the "Build"
section), arm P launched 2026-10-10 19:03, switched to the main split at the 300M save the same
evening, complete at 860.61M (one epoch) on 2026-10-11 and read; arm C launched the same day,
complete at 909.72M on 2026-10-14 and read. Arm P's final read: 1-hop 0.85 (the gate holds), L1
Delta +0.002 +- 0.005 (the spec's "falsified" cell), tier 1000 leak 0.064 / 0.046, tier 100 at
z 3.2 held out and z 2.2 in distribution (the arm P half of A1(b) not met on one form), copying
intact; A2 and A5 clear on the trainer's numbers. Arm C's final read: the control stores tier 100
at 0.467 in distribution and 0.049 held out, 47x and 4x arm P; the paired compare is at z -81 and
-6.4 there. The ladder's remaining reads (2026-10-14): A5 holds (port gain 1.56x the in-context
ceiling), A6 holds (within noise of the control), A3 holds in likelihood (+3.75 nats gold against
none, the swapped answer wins) and is unreadable in generation (96% abstention under gold), A4
fails (the model's query at recall@5 0.004 against bge 0.829), A7 unreadable (oracle EM 2.6%);
the exit and depth reads repeat the micro shape on both arms. Decided 2026-10-17 (the user, after
the research round in [review_2026-10-16.md](../review_2026-10-16.md)): A1(b) passes with the
leak recorded as known and the criterion amended to the paired reading (the read table above); L1
is recorded as failed at this shape unless the chain continuation of arm P (one pass over the
chain split from the final save, saves every 10M, read for the abrupt 2-hop transition) shows
composition, in which case it is under-dosed; the retriever fix (in-batch InfoNCE on the model's
own query, Co-LMLM's recipe) is built and goes to a micro arm; A3's behaviour is left to 6f's RL
and marked the one RL step that must not fail; the queue is the chain arm, the InfoNCE arm, then
R7, R8, R9. **The chain continuation ran and was read the same day
([chain_continuation_pilot.md](../measurements/chain_continuation_pilot.md)): one epoch of the
chain split (102.66M tokens, 5.1 hours) takes 1-hop lookup to 0.99 and leaves every composition
cell at chance (held-out 2-hop paired Delta +0.004 +- 0.008, composed answers at a uniform pick
among candidates), so L1 is failed at this shape and the decision table's "falsified" row
applies.** The pilot runs on today's code: the R0b recipe (the
key/value reader, the copy-first warm-up, swap rate 0.15) at a larger shape and on a mixed corpus,
two arms of 1B prompt tokens each, the pathway arm P against a matched-token control C without
retrieval. Most of the Phase 5 blueprint in NEXT.md (coda, tied token table, always-on selector,
sublayer reads, the depth draw) is not built and is deferred: it becomes arms after this pilot reads,
or goes to the real-run spec. Numbers come from the code, from the split files on disk and from three
throughput probes on the pilot shape run today; where a number is an estimate it says so. Everything
down to "Limitations" is fixed before the build; the build and the read go in sections appended below.

## Question, and what it decides

Does the R0b recipe keep facts out of the weights, and put reading and lookup in the loop, at three
times the micro parameters, on a corpus that mixes prose, QA and chains? The two claims are read on
the instruments R0b and the chain arm already validated:

- **Externalization (A1(b))**: closed-book `delta` on the injected biographies, arm P against arm C.
- **Lookup and composition (the 1-hop gate, then L1)**: `eval_chains.py` by kept read sites. The
  micro chain arm never learned 1-hop lookup ([chain_depth_micro.md](../measurements/chain_depth_micro.md)),
  so the pilot reads the precondition first and L1 only once it holds.

A pass sends the plan to the blueprint items as arms, selector rewrite first; an A1(b) fail at tier
100 stops the recipe at shape before any blueprint work (Decision table).

## Scope: today's code, the blueprint deferred

A code inventory on 2026-10-09 against the Phase 5 blueprint:

| blueprint item | in the code today |
|---|---|
| 2-layer coda, tied token table `E` | absent |
| always-on selector, null key, one-hot chunk channel, sink slot | absent; the selector is the routed IR expert with a 256-entry learned table |
| reads in sublayers 1 and 3 | absent; one read site per loop, no sublayers |
| depth draw {3: 0.7, 4: 0.2, 5: 0.1}, detached prefix | absent; `loop_count_sampling` draws uniformly over 1 to n_loops minus 1 |
| last-pass loss through the coda | absent; per-loop CE (`loop_ce_weights [0, 0, 1]` is a trap: `loop_ce_weights_for` returns the zeros unrescaled at sampled depths) |
| selection union over passes | absent; per-(token, chunk) BCE averaged over reads, asserts gradient checkpointing off |
| InfoNCE query term | absent |
| groundedness head on the null mass | the head and loss exist; there is no null mass to read |
| per-window bge retrieval on natural text, fact tagger, near-duplicate filter | absent |
| MuSiQue, 2Wiki, counterfactual and partial-hop training conditions | absent from `prepare_evidence_data.py` |
| retrieval on replay rows; a store of at least 0.4B tokens | absent; `build_store.py` is sized by question count |

What exists and is used: `sft.py --evidence --reader-kv` (prefix rotation, the chunk gate, the
selection BCE, the groundedness head and loss, the `[eval fixed]` pass), `prepare_injection_data.py`
(filler from the pretrain bins plus fictional biographies with store cards, swaps, anonymization,
placeholders, `--seq-length`), `prepare_evidence_data.py` (the `evidence_nomany_*` splits: SQuAD v2,
HotpotQA, web text, smoltalk2 replay; conditions gold, mixed, distractors, none), `prepare_chain_data.py`
(`chains_train` built), `build_store.py` and `eval_store.py` (`data/index/openqa`, `openqa_edit`),
`closed_book_rank.py`, `eval_chains.py`, `eval_exit.py`, `eval_abstention.py --evidence-port` with the
counterfactual condition, `eval_benchmarks.py`, `evidence_ceiling_probe.py --fixed-split`.

**Decision (2026-10-09).** The pilot runs on today's code. The ladder is cut from the bottom, as the
plan allows:

| rung | in this pilot |
|---|---|
| R0, R1, R0b | done; R0b accepted on the warm-up arm (2026-10-05) |
| R1b | reduced to the batch probes of the Cost section; there is no depth draw to probe |
| R2 | becomes the 1-hop gate, read at every save |
| R3 | closed with the graft branch |
| R4 (loss placement by coda), R4b (sublayers and experts), R6 (pretraining-text leak probe) | cut; R4 and R4b return as blueprint arms after a pass |
| R5 | the pilot's reads, without L2 at D = 4 and without A6 through the port |

**What this pilot cannot read.**

- **A1(a)** (tail-tier rank against the control on natural-text facts): no retrieval is attached to
  natural text, so arm P has no way to externalize those facts and both arms would store them
  alike. A1 is read on the injected biographies (A1(b)) only.
- **A6 through the port**: `eval_benchmarks.py` has no evidence path. A6 is read as language
  benchmarks within noise between the two arms.
- **L2 at D = 4 against D = 3**: the model trains at `n_loops` 3 and depth past training is untrained.
  The read-site half of L2 (held-out 4-hop answer CE by kept sites at D = 3) is read as secondary.
- **L1 as the blueprint states it** (sublayer sites against passes): one site per loop here, so
  "past the first pass" means passes 2 and 3, as at micro.

## Model

`config_pilot.yaml` at the repo root, like `config_micro.yaml`; its draft is
`ckpts/inject/config_pilot_probe.yaml`.

| key | value |
|---|---|
| width, intermediate (dense and MoE) | 512, 1536 |
| prelude layers, heads | 6; 8 heads x 64, 2 KV heads |
| experts | 8 MLP experts top 2, 1 attention expert, 1 IR expert with a 256-entry exact-read table |
| IR | `ir_dim` 384, `ir_num_clusters` 0, `ir_direct_read` true; the table is the selector here and is dropped from the real run |
| evidence | `evidence_encoder` true (all 6 prelude layers), `--reader-kv` |
| MTP, LM head | 2 extra tokens; `lm_head_factor` 4; `per_layer_embeddings_size` 16 |
| depth, context | `n_loops` 3, `max_seq_length` 4096 |

Constructor print, as given for this layout: total 99.1M, active 84.9M, 44.0M excluding embeddings;
forward about 249M FLOP per token at seq 4096 (body 110M, heads 64M, attention 75M); the evidence
encoder about 39.4M FLOP per evidence token over 6 layers. The probe logs, after `--reader-kv` has
dropped the cross reader's `shared_evidence` and `evidence_query_bias`, print total 98.4M, active
84.2M, 43.3M excluding embeddings, 245M FLOP per token (body 106M), so the 0.7M difference is the cross
reader the seed carries. The design said about 120M at width 512; 99M is what the constructor prints
with this layout.

**Loss weights** as `config_micro.yaml`: `lambda_mtp` 0.1, `aux_loss_weight` 0.01,
`evidence_selection_weight` 0.1, `loop_ce_weights [0.2, 0.3, 1.0]`, `loop_ce_subsample` 0.25,
`loop_count_sampling` 0.3; `groundedness_weight` 0.1 (0.0 in the micro config; QA rows with `.ans`
exist here). `conversation_loss_weighting: false`, `freeze_evidence_gate: true`, `max_evidence_tokens`
4608 per row (Cost).

**Optimizer.** AdamW, lr 6e-4, `fresh_lr` equal to lr (from scratch), weight decay 0.1 as today
(`ndim >= 2`), clip 1.0, warmup 2% of tokens, cosine floor 0.1, BF16, never FP8. The micro arms ran
1e-3 at width 256; 6e-4 is scaled down for width 512 without a sweep and is recorded as a guess.

**Batch.** 8 x 4096 x 1 (about 32k tokens per optimizer step, the same as the micro arms; probe 3 in
the Cost section fits it at 20.4 GB). Checkpoint every 50M tokens; `[eval]` every 25M; `[eval fixed]` on
`evidence_fixed` with `kill_tokens: 0` (the kill is off; the pilot reads, it does not kill on the
fixed pass). With uploads off the trainer keeps every save locally.

## Corpus

Token target 1B prompt tokens per arm. One merged training split per form, built from existing
builders by a new tool, `scripts/merge_evidence_splits.py`. It interleaves documents from several
evidence-format splits at given token shares, seeded, re-indexes the evidence chunk tables, and writes
one split. Common file set: `bin idx mask ans cond ev evchunk evgold evidx evkey evkeyidx`; the
`factspan`, `evhop` and `chains.jsonl` sidecars are eval-only and are not merged. It also emits an
evidence-free form of the same document stream (for arm C), and prints fill and realized share per
slice.

| slice | share, target | source | evidence | supervision |
|---|---|---|---|---|
| text filler plus injected biographies | 60%, 600M | `prepare_injection_data.py --seq-length 4096 --target-tokens 600000000`, arms retrieval and full | store cards, swap rate 0.15, anon rate 0.5, gold-drop rate 0.2, 3 distractors, placeholder rate 0 | span weights as R0b |
| evidence QA, web text, replay | 25%, 250M | `evidence_nomany_train`, 105M prompt tokens, at most 2 passes | gold, mixed, distractors, none; no counterfactual condition (a gap) | answer or abstain |
| synthetic chains | 10%, 100M | `chains_train`, 104.66M prompt tokens, one pass | hops 1 / 2 / 3 at 25 / 50 / 25 | answer |

- **Against the design.** 60 / 25 / 10 against the design's 45 / 10 / 15 / 10 / 20 (web and edu, code
  and math, evidence QA, chains, chat replay): what exists decides it. The filler bins `ir`, `phase1`,
  `phase2` already carry the pretrain mix's code and math; chat replay exists only inside
  `evidence_nomany_train`; no natural-text slice carries retrieval.
- **Biography dose.** The biographies stay at 3,600 people at 1 / 10 / 100 / 1000 exposures, so their
  share of the slice falls from 4.6% at 300M to about 2.3% at 600M, about 1.4% of the merged split.
- **What the files hold** (sizes on disk, 2 bytes per token): `evidence_nomany_train` 105.30M prompt
  and 101.35M evidence tokens (ratio 0.96; `evidence_train`, which has `many`, is 3.75); `chains_train`
  104.66M and 439.09M (ratio 4.20); the micro biography split 0.10, so about 0.05 at 600M with the
  card count fixed (estimate). At 2 passes the QA slice holds 210.6M, not 250M, so the merged split
  holds about 915M prompt tokens at the shares above. `sft.py` fits the cosine to one epoch, so the
  split is the budget: the merge prints the realized total and shares, and the gap to 1B is closed
  or recorded at the build.
- **Warm-up form.** The same merge with the biographies at swap rate 1.0 (the `inject_retrieval_s100a50`
  form rebuilt at 4096 and the 600M target); QA and chains unchanged.
- **The QA exception.** QA gold rows supervise real answers from the start: their answers are in the
  buffer by construction, and the closed-book probe does not read them. The copy-first requirement is
  enforced on the slice the A1 probe reads.
- **Control form.** The same merge with the `inject_full` form of the biography slice (fact spans in
  the loss, no cards) and with QA and chain rows stripped of their evidence, so both arms see the same
  token stream.
- **One data dir.** `sft.py` reads the train, val and fixed splits from one `--data-dir`, so the
  pilot's dir, `data/prepared_pilot`, holds the merges, the builder's `inject_val` (filler only, at
  4096) and a copy of `evidence_fixed.*` with its `.src`. The biography builds write there and to their
  own store dir, so the micro splits in `data/prepared_inject` and `data/index/inject_bios` stay as
  recorded.
- **Checks.** Each form is built once. What must be identical: the QA and chain documents between
  the warm-up and main merges (same seed, same order; read off `md5_by_slice` in
  `{split}.merge.json`, since the whole files differ in the biography documents), the `inject_val`
  and the facts and pools files between the biography builds, `evidence_fixed.*` against
  `data/prepared`. The three merges share one document order (`--order-from`), so a run resumed by
  document position on the main split continues where it stopped on the warm-up split.

## Schedule

**Arm P (pathway).** From a fresh seed (`init_scratch_seed.py` with `TINY_LLM_CONFIG=config_pilot.yaml`,
then `migrate_groundedness_head.py`, since the seed carries no head), `--evidence --reader-kv` on the
warm-up split until the switch criterion holds, then relaunched with `--train-split` on the main split
to 1B, in the same run directory (data overrides are not stored in the checkpoint).

```
TINY_LLM_CONFIG=config_pilot.yaml python scripts/sft.py --evidence --reader-kv -c ckpts/inject/seed_pilot_grounded.pt --run-name pilot_p --data-dir data/prepared_pilot --train-split pilot_warm_train --val-split inject_val 2>&1 | tee -a ckpts/inject/pilot_p.log
```

- **Switch criterion**, checked at every 50M save with `closed_book_rank.py bios --evidence gold` and
  `--evidence none`: gold card top-1 in distribution at least 0.9 at every tier, and closed-book entity
  `delta` within 3 sigma of the prior at every tier on both forms.
- **Expected** at 100M to 200M (the micro arm switched at 100M at top-1 0.95 to 0.97). Estimate on
  the copy dose: at micro copying emerged between 50M and 100M of 300M, so between about one sixth
  and one third of the 647,555 biography spans; spread over this merge the same counts arrive at
  about 170M to 330M. QA gold rows also train copying, so the window may come earlier; the copy
  curve at 50M to 150M says which.
- **Stop.** If the criterion is not met by the 300M save the run stops and that is the finding: the
  reader does not copy at this shape and dose.

**Arm C (control).** The control split, the `--evidence` profile without cards (as micro arm (a)),
full CE, the same token count, one launch, from the same seed.

Both launch in the background under a Monitor watch, filter
`Traceback|Error|Killed|OOM|assert|device not ready|Step|Tokens/sec|\[eval|packing`, teed to
`ckpts/inject/pilot_{p,c}.log`. Minute one: read the packing line (fill, rows closed by the
evidence budget) and the peak.

## Reads and gates

Gate names as NEXT.md. Every closed-book read at full depth. `closed_book_rank.py` takes `--facts`,
`--pools` and `--store` from the pilot build, `--batch-size` set from the first read.

| gate | script, split | bar | cadence | a fail means |
|---|---|---|---|---|
| copy (arm P) | `closed_book_rank.py bios --evidence gold\|swapped\|prompt` | switch criterion; afterwards descriptive | every save | before 300M: Stop; after: copying regressed, read with A3 |
| A1(b) | `closed_book_rank.py bios --evidence none --form both`, both arms; `compare` arm C against arm P | arm C at least 3 sigma above its prior at tier 100 or 1000, and the paired `compare` puts arm P under arm C at tier 100 and 1000 on both forms by at least 3 sigma; arm P's own residual at tier 100 is recorded as a known leak when it is under a quarter of the control's (amended 2026-10-17 by the user after the pilot read: the original "arm P within 3 sigma of its prior per form" was written for micro power and flips on noise at pilot power; the published relatives, LMLM and Co-LMLM, judge the same residual against the standard model) | every save | the paired compare fails at tier 100 on either form, or arm P's tier 100 residual is above a quarter of the control's: the externalization claim fails |
| 1-hop gate | `eval_chains.py`, `chains_eval` 1-hop, D 3, all sites | at least 3 sigma above chance (1 over type-matched candidates; 0.146 against 0.115 at n 1,000) and rising, by the 500M save | every save | at the final save: L1 inconclusive at this shape too |
| L1 | `eval_chains.py`, `chains_heldout_tmpl` 2-hop, D 3, kept 1, 2, 3 | Delta = kept 3 minus kept 1 at least 0.05 and at least 3 sigma | once the 1-hop gate holds, and final | falsified: Delta + 2 sigma under 0.05 with the 1-hop gate passed; inconclusive: the gate failed, or Delta between the two |
| A2, A5 | `[eval fixed]` on `evidence_fixed`; `--ceiling-json` from `evidence_ceiling_probe.py --fixed-split evidence_fixed` on arm P's final checkpoint | gold gain at least half that ceiling, distractors gain at least 0; chunk AUROC per loop at least 0.59; grounded AUROC at least 0.674 | every 25M in the trainer; final verdict | the port does not recover evidence on held-out QA |
| A3 | `eval_abstention.py --evidence-port --evidence-condition gold,counterfactual,distractors,none --json-out`; the swapped bios read for the injected facts | memorization ratio at most 0.05, follow at least 0.9 per stratum, ratio-binned or baseline-paired | swapped bios at every save; `eval_abstention.py` at final | weights override a chunk |
| A4, A7 | `eval_store.py recall\|pathway\|edit`, `data/index/openqa`, `openqa_edit` | flip rate at least 70%; model-query recall@k at least bge-query recall@k; pathway EM against oracle-buffer EM | final | store edits do not reach the answer; the query loses to bge |
| A6 | `eval_benchmarks.py`, both arms, against `docs/measurements/benchmark_snapshot.md` | arm P within noise of arm C | final | the pathway costs language |

**A1(b) detail.** Tier 1000 is read alongside; the leak is known from R0b (0.085 at micro, z 6.4)
and the expectation recorded is at or below the micro level. The NEXT.md method holds: `delta` against
the subject-blind prior, never the raw rank, and arm C must sit above its prior before arm P is read
against it.

**1-hop gate detail.** New; the precondition the plan calls "a 1-hop curve must saturate before any
null is read", which the micro chain arm failed. It must hold by the 500M save: about 50M chain
tokens seen, half the micro arm's dose at three times the parameters. Instrument fixes before launch:
per-question records in the JSON, per-answer gold NLL against ln K (the uniform-pick bound) for every
split, the `*` marker glued to the previous column. If the gate fails at the final save, the next arm
is a selector fix (the selection BCE stayed near uniform at micro, `selection:` 0.495 to 0.440), not a
bigger run.

**L1 detail.** `reader_sites_kept` 1, 2, 3, one site per loop in this code. Paired bootstrap on 2,000
held-out questions, sigma about 0.011 (estimate; the micro read was unpaired at 0.013). Every cell
with fewer sites than hops must sit at chance (validity, as at micro). `chains_hop4` is read
alongside, its answer CE by kept sites at D = 3 standing in for L2's read-site half.

**Loop reads carried over.** `eval_exit.py` on `inject_val` (per-exit CE, both arms, final), and
`closed_book_rank.py --n-loops` at 1, 2 and 3 at tier 100 on both arms (the recall-depth reading; one
run per depth, the flag takes one integer). `loop_scale` per save from the trainer log.

## Cost

Measured 2026-10-09 on the probe config, RTX 5090, GPU 99%, 307 W, from `seed_pilot_probe.pt`:

| probe | split | batch | fill | prompt tokens/s | peak |
|---|---|---|---|---|---|
| 1 | `evidence_train` (QA-heavy, evidence ratio 3.75, cap 4608) | 4 x 4096 x 2 | 17% to 30%, rows closed by the evidence budget | 2.7k to 3.9k | 15.0 GB |
| 2 | `inject_retrieval_train` (cards only, evidence about 8% of the cap) | 4 x 4096 x 2 | 88% to 92% | 41k to 45k | 12.7 GB |
| 3 | `inject_retrieval_train` | 8 x 4096 x 1 | 88% to 92% | 44k to 50k | 20.4 GB |

Batch 8 x 1 fits with room (20.4 GB of 32) and runs a little faster than 4 x 2 at the same tokens
per update, so the pilot runs 8 x 4096 x 1. The step is GPU bound at this width (99%), not host
bound as the micro shape was.

**The mix estimate (estimate).** Probe 1 read `evidence_train`, ratio 3.75 with `many`; the pilot's
QA slice is `evidence_nomany_train`, ratio 0.96, and the chains are the high-ratio slice (4.20, about
439M of the merged split's roughly 670M evidence tokens). Rows mix slices, so per-slice rates do not
add. A two-point linear fit of seconds per row against evidence tokens per row through probes 1 and 2
(evidence 78 to 83% of the cap in probe 1's packing lines), at the merged ratio of about 0.73 and fill
about 85%, gives about 16k prompt tokens per second, about 16 hours for 915M. A slice-by-slice sum
with the QA rows taken at probe 1's rate (a worst case, since their ratio is a quarter of probe 1's)
gives 26 to 28 hours; the truth sits between. The first hour of each arm decides. The chain slice is
the cost lever if one is needed: its evidence ratio is what closes rows on the evidence budget, and
its per-document chunk count (6 to 14 distractors) is a build parameter; `max_evidence_tokens` is the
per-row cap for every slice and lowering it lowers fill, so it is not the lever. The merge tool
reports fill per slice and the packing line in minute one is read against this estimate.

| item | time | disk |
|---|---|---|
| biography builds (3) and merges (3) | hours, unmeasured | about 25 GB per evidence merge (the chain keys are 18.5 GB), under 5 GB for the control; 239 GB free on D: |
| arm P | about 16 hours (fit) to 28 hours (slice sum) | about 1 GB per save, 20 saves (estimate from the micro 352 MB at 35M) |
| arm C | about 6 hours (without evidence, at about the probe 3 rate or faster) | the same |
| reads | about 1 hour per save pair; every save through the switch, then every 100M: about 12 pairs | small |
| total | roughly 35 to 45 GPU hours, estimate until the first hour of each arm prints its rate | |

The per-save read set is timed on the first save; if a pair runs over an hour the closed-book reads
keep every save and the chain and fixed reads move to every 100M. The micro reference: 35M
parameters, 55k to 60k tokens/s, 300M in 90 minutes.

## Build list before launch, in order

1. `scripts/merge_evidence_splits.py` and `tests/test_merge_evidence_splits.py` (plain asserts: the
   chunk tables re-indexed, the shares, determinism under the seed, the evidence-free form equal to the
   evidence form in `bin idx mask`, the per-document chunk cap).
2. `eval_chains.py` instrument fixes (per-question records, per-answer gold NLL against ln K for every
   split, the `*` column), with `tests/test_chain_generator.py` extended or a new test as fits.
3. `config_pilot.yaml` and the seed:
   `TINY_LLM_CONFIG=config_pilot.yaml python scripts/init_scratch_seed.py --out ckpts/inject/seed_pilot.pt --seed 0`,
   then `migrate_groundedness_head.py -c ckpts/inject/seed_pilot.pt`. Whether arm C keeps
   `groundedness_weight` 0.1 (without evidence every label is 0) is settled when the yaml is written.
4. The three biography builds at 4096 into `data/prepared_pilot` with their own `--store-dir`
   (retrieval s15 and full in one build, retrieval s100 with `--suffix s100a50`), then the three
   merges (warm-up, main, control), each packing line read and the `md5sum` checks of the Corpus
   section run.
5. Launcher scripts under `ckpts/inject/` (`launch_pilot_p.sh`, `launch_pilot_c.sh`, as the probe
   launchers), each started under a Monitor watch.
6. The read scripts per save, each run once on a probe save (`ckpts/evidence_pilot_probe3/`) to check
   that every read script loads a `--reader-kv` pilot checkpoint and to time it.

No change to `modules/`.

## Decision table

| copies by 300M | A1(b) tier 100 | 1-hop gate | L1 | next |
|---|---|---|---|---|
| no | not read | not read | not read | arm P stops at 300M; the copy curve is the finding; a decision on the copy recipe at this shape comes before any blueprint work |
| yes | fails on either form | any | any | the recipe fails at shape; no blueprint work until the leak is understood |
| yes | arm C not above its prior | any | any | A1(b) unreadable (the control stores nothing at this dose), not passed; the biography dose is fixed in the next arm |
| yes | holds | fails at final | inconclusive | the next arm is a selector fix, not a bigger run |
| yes | holds | passes | passes | the blueprint items as arms, selector rewrite first, then the real-run spec |
| yes | holds | passes | inconclusive | as the pass row; L1 is read again on the sublayer arm (two sites per pass) |
| yes | holds | passes | falsified | the loop clause is recorded as falsified at this shape; looping stays a requirement; the goal's reasoning clause is reworded before the real-run spec; blueprint arms as the pass row |

A2 to A7 failing with the rows above passing: recorded per gate, each a named target of the
blueprint arm that touches it (A5 the selector, A3 the counterfactual training condition, A4 and A7
the query training).

## Limitations

- One seed per arm; the seed sigma is unmeasured.
- No retrieval on natural text, so A1 is read on the biographies only.
- The control's QA rows have no passage, so arm C's QA answers train as text: a matched token
  stream, not a matched task.
- lr 6e-4 is a guess, not a sweep.
- 1B tokens against the plan's 1 to 2B; at the shares above the split holds about 915M.
- The IR table is the selector here and is dropped from the real run; a selector result does not
  transfer as is.
- `loop_scale` of the third pass collapsed at micro on chains (0.07); it is read per save.
- No counterfactual or partial-hop training condition; A3 is read without training for it.
- The in-context ceiling for A2 comes from arm P itself, which never sees a QA passage in the prompt
  during training; NEXT.md's absolute bar (1.6 nats, half the 3.23-nat ceiling of the full model) is
  printed beside it.

## Build (2026-10-10)

The build list ran in order on 2026-10-10; nothing in the spec above changed, and no file under
`modules/` was touched.

1. `scripts/merge_evidence_splits.py` with `tests/test_merge_evidence_splits.py` (six checks, all
   pass). Per slice a seeded permutation that depends only on the seed, the label and the document
   count; the output order drawn by remaining tokens; `evidx`, `evkeyidx` rebuilt; `--no-evidence`
   writes `bin idx mask` only; `--max-chunks-per-doc` keeps every gold chunk; `{split}.slice` (uint8
   per document) and `{split}.merge.json` (stats and `md5_by_slice`) ride along. One addition over
   the spec: `--order-from`, since the interleave weights are token counts and the two biography
   forms differ in length, so the warm-up and main merges would not share an order and the switch
   (a resume by document position on the main split) would repeat and skip documents.
2. `eval_chains.py` with `tests/test_eval_chains.py`: `records` per question and cell in the JSON
   (candidate scores, gold rank, margin, gold NLL, ln K, the divergent position and the NLL there),
   gold NLL against ln K per split, depth, kept and hops, `readings.paired_delta` (kept max minus
   kept 1, paired bootstrap, 2,000 resamples), the `*` marker in its own column.
3. `config_pilot.yaml` at the repo root (the probe shape; lr 6e-4 for every group, batch 8 x 4096
   x 1, checkpoint every 50M, eval every 25M, `[eval fixed]` on `evidence_fixed` with `kill_tokens`
   0, `groundedness_weight` 0.1). One file for both arms: the groundedness and selection terms are
   gated on evidence labels in the batch and the control split carries none, so both weights are
   inert there and every other number is shared by construction. Seed
   `ckpts/inject/seed_pilot.pt` (seed 0, 99.1M parameters) and `seed_pilot_grounded.pt` (the head
   added, 1.0K parameters).
4. Biography builds into `data/prepared_pilot` with the store in `data/index/pilot_bios`
   (`ckpts/inject/build_pilot_bios.sh`, log `build_pilot_bios.log`, 10 minutes for both):
   `--seq-length 4096 --target-tokens 600000000`, arms full and retrieval, then retrieval with
   `--swap-rate 1.0 --suffix s100a50`. 974,525 documents per arm (812,525 filler, 162,000
   biographies), filler repeat 2.18x over 373,317 train windows (the 270M-token filler supply is
   the limit), bio token share 2.3%, gold in buffer 79.9%, supported spans 81.1%, swapped spans
   97,395 (s15) and 647,555 (s100), evidence ratio 0.05. `inject_val`, `inject_facts.jsonl`,
   `inject_pools.json`, the store's `keys.npy` and `chunks.jsonl` are md5 identical to the micro
   build's, so the closed-book reads use the same people, pools and cards as every micro arm.
   Merges (`merge_pilot.sh`, `merge_pilot2.sh`, log `merge_pilot.log`, about 10 minutes each):
   `pilot_warm_train` (bios s100a50 at 0.60, QA `evidence_nomany_train` at 0.25 and 2 passes,
   chains `chains_train` at 0.10, target 1B, seed 42), `pilot_main_train` (bios s15, the same
   slices, `--order-from pilot_warm_train.slice`) and `pilot_control_train` (bios full at 0.6001 so
   its 600,000,752 tokens are taken whole and the counts match the order, `--no-evidence`,
   `--order-from` the same). Realized per merge: 3,488,744 documents, 910.2M prompt tokens (bios
   599.6M, QA 210,607,046 at exactly 2 passes, chains 99,999,982 in 1,910,913 of 2M questions);
   evidence 652.3M tokens at ratio 0.717 (bios 30.0M at 0.050, QA 202.7M at 0.962, chains 419.6M
   at 4.196; 24.1M chunks), no document over the 4608 cap or the 256-chunk limit. The QA and chain
   `md5_by_slice` are identical across the three merges, the `.slice` files are identical, and
   `evidence_fixed.*` with its `.src` is md5 identical to `data/prepared`. **The budget is 910M
   prompt tokens per arm**, not 1B: `sft.py` fits the cosine to the split, the QA slice at 2 passes
   holds 210.6M, and the gap is recorded here rather than closed (a third QA pass or more chains
   would change the shares). Disk: 25.6 GB per evidence merge, 2.8 GB for the control, 194 GB free.
5. `ckpts/inject/launch_pilot_p.sh` (`SPLIT` defaults to `pilot_warm_train`; the switch relaunch is
   the same script with `SPLIT=pilot_main_train`, resuming `ckpts/evidence_pilot_p/`) and
   `launch_pilot_c.sh`, both teeing to `ckpts/inject/pilot_{p,c}.log`.
6. `ckpts/inject/read_pilot_save.sh CKPT p|c TAG [CHAINS]`: the four card conditions of
   `closed_book_rank.py bios` (arm C: `none` only) with `--facts`, `--pools` and `--store` from the
   pilot build, then `eval_chains.py` at depths 1, 2, 3 on the three chain splits and at 3, 4 on
   `chains_hop4`, each timed. Not dry-run: no probe save exists (the probe run directories were
   removed on 2026-10-09), so the first 50M save of arm P is the dry run and the timing.

## Run (2026-10-10, arm P warm-up and switch)

Arm P launched 2026-10-10 19:03 on `pilot_warm_train` (`launch_pilot_p.sh`, log
`ckpts/inject/pilot_p.log`), stopped by STOP right after the 300M save at 21:17 (saves at 300M and
301M, exit 10), read, and relaunched on `pilot_main_train` from the 301M save at 21:50 (resumed at
document 1,221,989 of 3,488,744; the shared order makes that position the same document on both
splits). Nothing in the spec changed.

- **Rate.** 42k to 48k prompt tokens per second steady (bursts to 80k), peak 27.1 GB at batch 8 x
  4096 x 1 with 2.5 GB already held by other processes; packing over 256 rows at fill 87 to 89%,
  evidence 57 to 60% of its cap, 26 to 40 rows of 256 closed by the evidence budget. 50M per 21
  minutes, so arm P is about 6.5 hours of training in all, not the 16 to 28 the fit gave: the
  ratio-0.72 corpus runs at the low-evidence rate, and the 3.75-ratio probe rows did not transfer.
- **Trainer reads through 300M.** Val CE on `inject_val` 5.75 / 5.04 / 4.46 / 4.21 / 4.07 / 3.96 /
  3.89 / 3.82 / 3.77 / 3.68 at 25M steps from 25M to 275M. `[eval fixed]` gold gain +0.30 at 25M,
  +2.02 at 75M, +3.08 at 125M, +3.25 at 150M (the 3.23-nat ceiling, measured on different rows),
  +3.73 at 275M; content gain +3.39 at 275M; grounded AUROC 0.84 at 25M, 0.95 at 275M. The
  selector stays weak: chunk AUROC per token 0.61 at 125M, gold mass per chunk 0.011 against
  0.009 for distractors. `loop_scale` [1.19, 0.40, 0.13] at 300M (init 0.577 each), the third
  pass faint as on the chain arm.
- **Switch read on the 300M save** (`read_pilot_save.sh ... p 300M 1`, 26 minutes for the six
  reads: rank 346 / 307 / 296 / 275 s for none / gold / swapped / prompt, chains 216 + 123 s;
  outputs `ckpts/inject/pilot_p_300M_*`). Copying: gold card top-1 1.000 in every tier and class
  in distribution, held out 0.953 to 1.000 (entities 0.98 / 0.98 / 0.98 / 0.95 by tier); swapped
  card followed 1.000 in distribution, 0.979 to 0.986 held out, `mr_ll` 0.000 to 0.001; prompt
  copy 1.000 in distribution, 0.967 to 1.000 held out. Closed book (`none`): every entity, date
  and noun cell at chance on both forms, `delta` within 3 sigma everywhere (largest |z| 2.3, at
  tier 10 dates in distribution; entity tier 1000 -0.013, z -1.3 / -0.016, z -1.9). Both halves
  of the switch criterion met; the micro arm's 50M leak (0.010, z 4.0 at tier 100) has no
  counterpart here.
- **1-hop gate at 300M: passed, three saves ahead of the 500M bar.** `chains_eval` 1-hop with one
  site kept 0.324 +- 0.015 against chance 0.115 (z +20.7) at depth 1, 0.319 to 0.325 at depths 2
  and 3; kept 0 at 0.117 (chance). 2-hop 0.181 to 0.193 (chance 0.185, z at most 0.7), 3-hop 0.225
  to 0.232 (chance 0.206, z 1.5 to 2.1), `chains_heldout_tmpl` 2-hop 0.198 to 0.201 (chance 0.191),
  `chains_hop4` at chance; paired Delta kept 3 minus kept 1 on the held-out 2-hop +0.000 +- 0.003,
  seen -0.004 +- 0.006; "sub-hop cells outside 3 sigma of chance: 0, instrument valid". Gold NLL
  per answer 4.0 to 4.2 against ln K 1.6 to 2.2 with any site kept (46 to 48 with none): the read
  carries the answer type and the candidates compete, as on the chain arm, but now 1-hop lookup
  picks the right one a third of the time. Nothing composes yet; L1 is readable from here and
  reads flat at 300M of warm-up.
- **Read cadence.** 26 minutes per full read set (20 without the chain reads) against 21 minutes
  per 50M of training, and the trainer keeps the GPU during reads, so each read set is a STOP,
  read, relaunch (a pause costs the dataloader prefetch, under 4 documents, plus a 2-minute
  restart). The micro arms wrote their leak within 50M of the switch, so the rank reads run at
  350M and 400M; from 500M every 100M with the chain reads, and at the final save. Saves count
  50M from the last resume point, so after each stop the grid shifts by about 1M (301M, 351M,
  376M, ...); `stop_after_save.sh ARM PATTERN` takes a regex on the token tag.
- **351M read, 50M after the switch** (rank reads only, 19.5 minutes; the stop landed late at
  376M because the save notice reached the watcher buffered, so the run resumed from the 376M
  save and nothing was lost). Copying intact: gold top-1 1.000 in every tier in distribution,
  0.983 to 0.988 held out; swapped followed 1.000 / 0.967 to 0.972, `mr_ll` under 0.003; prompt
  copy as at 300M. **The tier 1000 leak is written within 50M of the switch, as on the micro
  warm-up arm**: closed-book entity `delta` in distribution 0.008 / 0.003 / 0.001 / 0.043 by tier
  (tier 1000 z 3.9, raw rank 0.407 at z 5.8 from chance), held out 0.005 / 0.006 / 0.007 / 0.012
  (z at most 1.9); pooled over tiers 0.0065 (z 2.9) / 0.0062 (z 3.1), carried by tier 1000 in
  distribution and by small same-sign cells held out. Tier 100 is flat on both forms. The micro
  arm had tier 1000 at 0.081 fifty million tokens after its switch at a third of the parameters
  and a third of the tokens per exposure; here 0.043 at the same point on the run.
- **426M read, 125M after the switch** (rank reads, 20 minutes; stopped by the watcher right after
  the save, resumed from the 426M stop save). The tier 1000 leak grows and reaches the held-out
  form: entity `delta` in distribution 0.009 / -0.003 / 0.006 / 0.050 by tier (tier 1000 z 5.0),
  held out 0.004 / 0.007 / 0.004 / 0.029 (z 3.3); tier 100 flat on both (z 1.4 / 1.1). Copying in
  distribution unchanged (gold top-1 1.000, swapped followed 1.000); held out it slips: gold top-1
  0.919 pooled, 0.890 at tier 1000 (0.985 at 351M), swapped followed 0.928 to 0.934, 0.906 at tier
  1000 (0.967 to 0.972 at 351M), `mr_ll` up to 0.006. The micro warm-up arm ended with held-out
  copying at 0.66 to 0.69 followed; the pilot is on the same slope. Watch item from the read
  table ("copying regressed, read with A3"): the final A3 read pairs this against the prompt copy.
- **530M read, 229M after the switch** (2026-10-11; the full set, 24 minutes: rank 300 / 276 /
  272 / 265 s, chains 199 + 101 s; the run was paused by the user at the 429M stop save overnight
  and resumed from it; the watcher stopped it right after the regular 530M save, so the read and
  the resume are on the 530M stop save, `loss5.8884`). **The 1-hop gate holds at its 500M bar and
  is rising**: `chains_eval` 1-hop with any site kept 0.520 to 0.533 against chance 0.115 (z +40
  to +41) at depths 1 to 3, from 0.32 at 300M; gold NLL per answer 2.7 against ln K 2.2 (4.0 at
  300M). **L1 reads flat**: 2-hop 0.18 to 0.19 (chance 0.185), 3-hop 0.19 to 0.21 (0.206),
  `chains_heldout_tmpl` 2-hop 0.18 to 0.20 (0.191), `chains_hop4` 0.18 to 0.21 (0.210) in every
  cell at every depth, depth 4 included; paired Delta kept 3 minus kept 1 on the held-out 2-hop
  -0.004 +- 0.005, seen -0.008 +- 0.008; "sub-hop cells outside 3 sigma of chance: 0". Composed
  answers cost 3.7 to 3.8 nats against ln K 1.6 to 1.7 with every site kept: the read carries the
  answer type and the one-hop pick, nothing composes yet. **The tier 1000 leak is level in
  distribution and has caught up held out**: entity `delta` in distribution 0.007 / 0.001 / 0.002 /
  0.040 by tier (tier 1000 z 4.5; 0.050 at 426M), held out 0.007 / -0.001 / 0.003 / 0.043 (z 5.9;
  0.029 at 426M); tier 100 flat on both forms (z 0.4 / 0.9); pooled over tiers 0.0055 (z 2.6) /
  0.0054 (z 2.8). By attribute at tier 1000 the birth city carries it (0.048, z 3.5 / 0.062,
  z 4.7), then the university (0.039 / 0.040), the employer under 2.2 sigma; dates at tier 1000
  0.022 / 0.025 (z under 1.2), nouns 0.034 (z 1.8 / 2.6). The micro warm-up arm ended at 0.085 /
  0.047. **Held-out copying recovered**: gold top-1 0.973 for entities (0.969 / 0.968 / 0.981 /
  0.977 by tier), 0.993 dates, 0.952 nouns (0.919 pooled at 426M); swapped followed 0.966 to 0.974
  held out (0.928 to 0.934 at 426M), `mr_ll` under 0.005; in distribution gold top-1 1.000 and
  followed 1.000 in every cell; prompt copy 1.000 / 0.981. So the 426M slip was a dip, not the
  micro arm's slope, on one read each; the final A3 read decides. Trainer reads over the stretch:
  gold gain on `evidence_fixed` +3.83 at 455M and +3.74 at 480M (content gain +3.79 / +3.69,
  grounded AUROC 0.973 / 0.971), `[eval]` CE 3.453 at 480M, `loop_scale` [1.17, 0.45, 0.19] at
  483M (the third pass up from 0.13 at 300M), 42k to 44k tokens per second. Outputs
  `ckpts/inject/pilot_p_530M_*`. Next reads at about 630M, 730M, 830M and the final save.
- **631M read, 330M after the switch** (the full set, 25 minutes; the watcher stopped the run right
  after the regular 631M save, the read and the resume are on the 631M stop save, `loss6.3982`).
  **1-hop keeps rising**: `chains_eval` 1-hop 0.585 to 0.594 with a site kept (chance 0.115,
  z +47; 0.53 at 530M), gold NLL per answer 2.25 against ln K 2.20, so the 1-hop pick now costs
  what a uniform pick among the type-matched candidates costs. **L1 still flat**: 2-hop 0.18 to
  0.19 (chance 0.185), 3-hop 0.18 to 0.19 (0.206, z down to -2.1), held-out 2-hop 0.175 to 0.182
  (0.191), 4-hop 0.20 to 0.21 (0.210) at depths 3 and 4; paired Delta kept 3 minus kept 1 held out
  -0.006 +- 0.005, seen +0.004 +- 0.007; "instrument valid: yes". Composed answers 3.44 to 3.60
  nats against ln K 1.6 to 1.7. **The leak: tier 1000 level, the lower tiers creeping in
  distribution.** Entity `delta` in distribution 0.011 / 0.002 / 0.009 / 0.041 by tier (tier 1000
  z 4.3; tier 1 z 2.8, tier 100 z 2.0, from 0.007 / 0.001 / 0.002 at 530M), pooled over tiers
  0.0095 (z 4.3, from 0.0055 at z 2.6); held out 0.007 / 0.000 / 0.006 / 0.028 (tier 1000 z 3.5,
  down from 0.043; tier 100 z 1.6), pooled 0.0057 (z 2.8). No single lower-tier cell is at 3 sigma
  on either form, but the in-distribution pool is, with the same sign in three of four tiers: the
  A1(b) tier 100 criterion is not breached, and it is the cell to watch at 731M. At tier 1000 the
  employer leads now (0.050, z 3.1 / 0.036, z 2.3), the birth city 0.029 (z 2.6) / 0.023; dates
  0.034 / 0.007 (z under 1.3), nouns 0.042 (z 2.4) / 0.020. **Held-out copying slipped again**:
  gold top-1 0.921 for entities (0.930 at tier 1000), 0.986 dates, 0.971 nouns; swapped followed
  0.932 to 0.952 (`mr_ll` under 0.008); prompt copy 0.950 held out; in distribution gold top-1
  1.000 and followed 1.000 in every cell. Over the four reads held-out entity gold top-1 ran 0.985,
  0.919, 0.973, 0.921: an oscillation of about 0.05 between reads taken on alternating saves,
  not a monotone slope, around 0.95; the final A3 read is the one that counts. Trainer reads over
  the stretch: gold gain on `evidence_fixed` +3.71 / +3.92 / +4.09 at 555 / 580 / 605M (content
  gain +3.67 / +3.88 / +3.88; the 605M gold gain rides a higher `none` CE), grounded AUROC 0.971
  to 0.979, `[eval]` CE 3.351 at 605M, `loop_scale` [1.22, 0.50, 0.23] at 625M, 40k to 47k
  tokens per second (69k on one light stretch). Outputs `ckpts/inject/pilot_p_631M_*`.
- **731M read, 430M after the switch** (the full set, 24 minutes; read and resumed on the 731M
  stop save, `loss5.9930`). **1-hop lookup now selects**: `chains_eval` 1-hop 0.802 to 0.803
  with a site kept (chance 0.115, z +68; 0.59 at 631M), gold NLL per answer 1.09 against ln K
  2.20, half the uniform-pick cost, the first read where the one-hop answer is chosen rather than
  type-matched. **L1 still flat**: 2-hop 0.19 to 0.20 (chance 0.185, z at most 1.2), 3-hop 0.18
  (0.206, z -1.8 to -2.3), held-out 2-hop 0.197 to 0.200 (0.191, z at most 1.0), 4-hop 0.20
  (0.210) at depths 3 and 4; paired Delta kept 3 minus kept 1 held out +0.002 +- 0.005, seen
  +0.008 +- 0.007; composed answers 2.97 to 3.07 nats against ln K 1.6 to 1.7 (3.4 to 3.6 at
  631M), so the composed answer is getting cheaper without the pick getting right. **The leak
  grows again and the A1(b) tier 100 cell crosses 3 sigma in distribution**: entity `delta` in
  distribution 0.008 / 0.006 / 0.013 / 0.064 by tier (tier 100 z 3.2, tier 1000 z 7.0, from
  0.009 / 0.041 at 631M), pooled 0.0123 (z 5.5); held out 0.008 / -0.003 / 0.008 / 0.045 (tier 100
  z 2.4, tier 1000 z 5.7), pooled 0.0069 (z 3.4). Nouns in distribution 0.024 (z 3.0) at tier 100
  and 0.057 (z 3.2) at tier 1000; dates inside 2 sigma everywhere. At tier 1000 the employer
  0.081 (z 4.6) / 0.051 (z 3.2), the university 0.064 (z 3.7) / 0.039, the birth city 0.048
  (z 3.8) / 0.046 (z 4.0). The A1(b) criterion ("arm P within 3 sigma through tier 100 on both
  forms") is breached on this read on the in-distribution form, at z 3.2 against a bar of 3.0,
  and the held-out form sits at 2.4; the verdict is taken at the final save with arm C alongside,
  and the direction since 530M (0.002, 0.009, 0.013 at tier 100; 0.040, 0.041, 0.064 at tier
  1000, in distribution) is up through the cosine's tail (LR 1.3e-4 at 710M, floor 3e-5). The
  micro warm-up arm's final was 0.015 (z 2.6) at tier 100 and 0.085 at tier 1000. **Held-out
  copying at its low**: gold top-1 0.935 for entities, 0.974 dates, 0.833 nouns (0.971 at 631M);
  swapped followed 0.916 to 0.952 (`mr_ll` under 0.008); prompt copy 0.961; in distribution gold
  top-1 1.000 and followed 1.000 in every cell. Trainer reads over the stretch: gold gain on
  `evidence_fixed` +3.99 / +3.98 / +3.96 / +4.06 at 655 / 680 / 705 / 730M (content gain +3.94 /
  +3.90 / +3.86 / +4.01), grounded AUROC 0.975 to 0.989, `[eval]` CE 3.277 at 730M, `loop_scale`
  [1.25, 0.53, 0.24] at 710M, 41k to 47k tokens per second. Outputs `ckpts/inject/pilot_p_731M_*`.
- **832M read, 531M after the switch** (the full set, 24 minutes; read and resumed on the 832M
  stop save, `loss5.4559`; the relaunch runs to the final save without another stop). **The tier
  100 breach moves to the other form**: entity `delta` in distribution 0.006 / 0.006 / 0.008 /
  0.068 by tier (tier 100 z 1.8, back inside; tier 1000 z 7.2), pooled 0.0106 (z 4.7); held out
  0.004 / 0.000 / 0.013 / 0.046 (tier 100 z 3.4, the breach now on this form; tier 1000 z 5.6),
  pooled 0.0081 (z 3.8). Over 731M and 832M the tier 100 cell reads 0.013 (z 3.2) then 0.008
  (z 1.8) in distribution and 0.008 (z 2.4) then 0.013 (z 3.4) held out: one form at 3 sigma on
  each read, neither on both, and the two forms pooled 0.011 then 0.011 at z 4 on each (the cells
  are independent, 1,500 items a form). Tier 1000 keeps rising in distribution (0.064 to 0.068;
  birth city 0.060, z 5.1, employer 0.071, z 4.1, university 0.074, z 4.0) and holds held out
  (0.045 to 0.046; employer 0.060, z 3.9); nouns 0.063 (z 3.6) / 0.043 (z 2.8) at tier 1000,
  0.016 (z 2.2) at tier 100 in distribution; dates inside 2 sigma everywhere. **1-hop 0.85**
  (z +73) with the gold answer at 0.86 nats against ln K 2.20; **L1 flat**: 2-hop 0.18 to 0.19,
  3-hop 0.19 to 0.20, held-out 2-hop 0.196 to 0.201 (chance 0.191, z at most 1.1), 4-hop 0.19 to
  0.21 at depths 3 and 4, paired Delta kept 3 minus kept 1 held out +0.004 +- 0.005, seen +0.005
  +- 0.007; composed answers 2.81 to 2.90 nats against ln K 1.6 to 1.7, down from 3.0 at 731M.
  **Held-out copying back up**: gold top-1 0.956 entities (0.963 at tier 1000), 0.989 dates, 0.940
  nouns; swapped followed 0.956 to 0.970 (`mr_ll` under 0.003); prompt copy 0.967; in
  distribution 1.000 everywhere. Trainer reads: gold gain on `evidence_fixed` +4.04 / +4.02 /
  +4.15 at 755 / 805 / 830M (content gain +3.97 / +3.95 / +4.05), grounded AUROC 0.982 to 0.991,
  `[eval]` CE 3.237 at 830M, `loop_scale` [1.28, 0.56, 0.25] at 824M, 43k to 48k tokens per
  second. Outputs `ckpts/inject/pilot_p_832M_*`. The final save decides A1(b): the cell has read
  at 3 sigma on one form at each of the last two reads, so the final read is paired against arm C
  before any verdict is written.
- **Arm P complete: 860.61M tokens, one epoch, 2026-10-11** (`checkpoint_evidence_final.pt`,
  exit 0; the trainer stops at the epoch's end, not the token target, so the cosine fitted to
  910M ended at LR about 6e-5 against its 3e-5 floor; one epoch is 860M on the trainer's counter
  because the packer skips over-length conversations and loses fill against the merge's 910M
  prompt count). Final filler CE 3.2321 on `inject_val`. Trainer's final fixed eval: gold gain
  +4.00 nats (124% of the 3.23 ceiling measured on other rows), mixed +3.74, distractors +0.03,
  content gain +3.97; per source SQuAD gold +3.80 with distractors -0.28, HotpotQA gold +4.19 with
  distractors +0.33; chunk AUROC per token 0.78 at loop 3 (0.66 / 0.68 / 0.78 by loop), gold
  share 0.51; grounded AUROC 0.994 all rows, 0.988 evidence rows. **A2 and A5 clear on the
  trainer's numbers** (bars: half the ceiling, distractors at least 0 pooled, chunk AUROC 0.59,
  grounded 0.674); the formal A5 still wants `--ceiling-json` on this checkpoint.
- **Final read set (`ckpts/inject/pilot_p_final_*`, 25 minutes).** Closed book, entity `delta` by
  tier 1 / 10 / 100 / 1000: in distribution 0.008 / 0.005 / 0.010 (z 2.2) / 0.064 (z 7.2), pooled
  0.0111 (z 4.9); held out 0.007 / -0.002 / 0.0125 (z 3.2) / 0.046 (z 5.5), pooled 0.0082
  (z 3.8). Nouns at tier 1000 0.066 (z 3.8) / 0.056 (z 3.4), at tier 100 0.018 (z 2.4) / -0.005;
  dates inside 2 sigma everywhere. By attribute at tier 1000 the employer 0.072 (z 4.6) / 0.059
  (z 3.5), the university 0.069 (z 3.9) / 0.040 (z 2.9), the birth city 0.051 (z 4.2) / 0.040
  (z 3.6); at tier 100 every attribute under 2.5 sigma on both forms. **The arm P half of A1(b)
  ("within 3 sigma through tier 100 on both forms") is not met at the final save: tier 100 reads
  z 3.2 on the held-out form** (z 2.2 in distribution; the forms swapped sides at 731M and 832M,
  and the pooled-over-forms cell has sat at z 4 to 5 since 731M). The verdict waits for arm C's
  final `none` read and the paired `compare`, as the A1(b) detail requires (arm C above its prior
  first). The micro warm-up arm's final was 0.015 (z 2.6) / 0.007 (z 2.0) at tier 100 and 0.085 /
  0.047 at tier 1000; the pilot's tier 1000 is lower held out and level in distribution, its
  tier 100 the same size at a smaller sigma. Copying: in distribution gold top-1 1.000 and
  swapped followed 1.000 in every cell, `mr_ll` 0.000; held out gold top-1 0.953 entities (0.942
  / 0.951 / 0.962 / 0.967 by tier), 0.996 dates, 0.899 nouns, swapped followed 0.943 to 0.968
  (`mr_ll` under 0.005), prompt copy 0.970 entities. Held-out entity gold top-1 across the reads:
  0.985, 0.919, 0.973, 0.921, 0.935, 0.956, 0.953. **Chains: 1-hop 0.844 to 0.849 with a site
  kept (chance 0.115, z +72), gold NLL per answer 0.82 against ln K 2.20; the 1-hop gate holds.
  L1: `chains_heldout_tmpl` 2-hop at depth 3 0.193 / 0.194 / 0.195 at kept 1 / 2 / 3 (chance
  0.191), paired Delta kept 3 minus kept 1 +0.002 +- 0.005 held out, +0.001 +- 0.008 seen;
  Delta + 2 sigma = 0.012, under the 0.05 bar with the gate passed: the spec's "falsified" cell.**
  2-hop seen 0.19 (chance 0.185), 3-hop 0.18 (0.206, z -2.1), 4-hop 0.18 to 0.19 (0.210, z down
  to -2.1 at depths 3 and 4); every composition cell at chance or slightly under; composed
  answers 2.81 to 2.90 nats against ln K 1.6 to 1.7 with every site kept, 45 without one; "sub-hop
  cells outside 3 sigma of chance: 0, instrument valid". Pending the user's decision: the decision
  table's first matching row is "A1(b) fails on either form" (the recipe fails at shape, no
  blueprint work until the leak is understood), with the caveat that the failing cell is at
  z 3.2 on one form and z 2.2 on the other and arm C is not read yet; the L1 row ("falsified at
  this shape; looping stays a requirement; the goal's reasoning clause is reworded before the
  real-run spec") applies on its own terms once the gate row is settled.
- **Arm C (the control) launched 2026-10-11 21:04** (`launch_pilot_c.sh`, `ckpts/evidence_pilot_c/`,
  log `ckpts/inject/pilot_c.log`; the control split through `SFTDataset`, no evidence, the port's
  cross-reader tensors at their init; 51k to 59k tokens per second, 18.1 GB). Paused by the user at
  the 74M save on the night of 2026-10-11 and resumed on 2026-10-12, so its save grid sits at
  124M, 174M, ... (24M off arm P's). Its port stays neutral on the fixed eval (gain 0.000, grounded
  AUROC 0.500 at every read). `[eval]` CE on `inject_val` 5.08 / 4.25 / 3.98 / 3.84 / 3.73 / 3.65
  at 50 / 100 / 150 / 200 / 250 / 300M (arm P 3.47 at 455M, 3.23 at the end).
- **Arm C 325M closed-book read** (`ckpts/inject/pilot_c_325M_rank_none.*`, the stop save
  `loss6.5000`; the chain reads ran too and sit at chance in every cell, as a model that never read
  evidence must). **The control stores the biographies as the micro arm (a) did**: entity `delta`
  in distribution 0.006 / 0.044 (z 6.2) / 0.299 (z 43) / 0.390 (z 25; raw rank 0.000, top-1
  1.000) by tier, pooled 0.130 (z 28); held out 0.001 / 0.008 / 0.029 (z 7.7) / 0.105 (z 10.3),
  pooled 0.019 (z 9.0). Dates 0.232 / 0.217 at tiers 100 / 1000 in distribution, 0.052 / 0.175
  held out; nouns 0.405 / 0.457 in distribution, held out flat but 0.044 (z 4.0) at tier 1000.
  Every tier 100 and tier 1000 attribute is above 14 sigma in distribution and above 3.9 held out.
  The micro arm (a) at 300M read 0.378 / 0.467 in distribution and 0.023 / 0.104 held out. **So
  arm C is far above its prior at tier 100 and 1000 on both forms, and A1(b) is readable at the
  final.** Against arm P at 351M (0.001 / 0.043 at tiers 100 / 1000 in distribution) the control
  stores about 300x more at tier 100 and 9x more at tier 1000, on the same tokens of the same
  document order; the paired `compare` at the final saves is the read that carries the verdict.
- **Arm C complete: 909.72M tokens, one epoch, 2026-10-14** (`checkpoint_evidence_final.pt`,
  exit 0; paused by the user at 854M on 2026-10-12 and resumed 2026-10-14; the control's epoch
  runs 49M longer than arm P's because the control split has no evidence and so no over-length
  skips). Final `inject_val` CE 3.2195 (arm P 3.2321, a 0.013 gap, the micro gap was 0.026).
  Trainer's final fixed eval on `evidence_fixed`: answer CE 4.35 under every condition, gain
  0.000, chunk AUROC 0.52, grounded AUROC 0.50: the port is neutral, as a run without evidence
  must leave it. Note the `none` answer CE: 4.35 on the control against 5.86 on arm P, so arm P
  answers worse than the control when no evidence is attached and 2.5 nats better with gold
  (1.85).
- **Arm C final closed-book read** (`ckpts/inject/pilot_c_final_rank_none.*`): entity `delta` in
  distribution 0.016 / 0.201 (z 21) / 0.467 (z 62; top-1 0.959) / 0.468 (z 28; top-1 1.000) by
  tier, pooled 0.243 (z 43); held out -0.006 / 0.005 / 0.049 (z 12.3) / 0.118 (z 10.8), pooled
  0.022 (z 10.7). Dates 0.339 / 0.295 in distribution and 0.057 / 0.105 held out at tiers 100 /
  1000; nouns 0.482 / 0.471 and 0.016 / 0.040. The micro arm (a) final was 0.378 / 0.467 and
  0.023 / 0.104: the control at pilot scale stores tier 100 to the same level as tier 1000, and
  now stores tier 10 (0.20) as well. **The paired `compare`** (`compare_pilot_c_vs_p.log`, arm C
  minus arm P on the raw rank of the real name, all classes, negative means the control ranks it
  better): in distribution tier 10 -0.226 (z -30), tier 100 -0.480 (z -81), tier 1000 -0.372
  (z -29); held out tier 100 -0.033 (z -6.4), tier 1000 -0.050 (z -3.7); tier 1 flat on both forms.
  **A1(b) as the read table states it: arm C is far above its prior at tiers 100 and 1000 on both
  forms (the first half holds), and arm P is within 3 sigma through tier 100 in distribution but
  not held out (0.0125, z 3.2), so the second half fails on one form.** In ratio the pathway
  stores 47x less than the control at tier 100 in distribution (0.010 against 0.467), 4x less
  held out (0.0125 against 0.049), 7x less at tier 1000 in distribution (0.064 against 0.468),
  2.6x less held out (0.046 against 0.118). The decision table's first matching row is "A1(b)
  fails on either form"; the cell that fails is 4x under the control on the same form and 47x
  under it on the other. The user decides the row.
- **The ladder's remaining reads, 2026-10-14** (`ckpts/inject/ladder_pilot.sh`, log
  `ladder_pilot.log`, outputs `ckpts/inject/pilot_{p,c}_ladder_*.{json,log}`; 45 minutes for the
  twelve reads).
  - **A5, holds with margin.** `evidence_ceiling_probe.py --fixed-split evidence_fixed` on arm
    P's final: the in-context ceiling (the passage in the prompt, no port) is 2.56 nats pooled
    (HotpotQA 2.56, SQuAD 2.56; in-context gold CE 3.49 against `none` 6.05), and the trainer's
    port read on the same rows was +4.00 (gold CE 1.85), so the port gain is 1.56x its own
    in-context ceiling, against a bar of half. The model never trained with a passage in the
    prompt, so the in-context number is a floor for this model rather than a ceiling.
  - **Exit reads, both arms** (`eval_exit.py` on `inject_val`, 40 batches): arm P CE 3.490 /
    3.417 / 3.409 at exits 1 / 2 / 3 (+0.073 then +0.0075), arm C 3.470 / 3.389 / 3.384 (+0.080
    then +0.0050); the gain 1 to 3 is +0.08 to +0.11 in every confidence decile but the top one
    (+0.05); a confidence exit rule is no better than a fixed-depth mix at the same passes on
    either arm (tau 0.5: 3.428 against 3.415 on arm P); the beta 0.1 optimum is near uniform. The
    micro read's shape, on both arms: depth pays evenly, the third pass is faint.
  - **Depth reads** (`closed_book_rank.py --n-loops 1 | 2`, with the depth 3 final read): arm C
    entity `delta` at tier 100 in distribution 0.348 / 0.464 / 0.467 by depth (the real name
    first on 27% / 94% / 96% of items), tier 1000 saturated at every depth, held out 0.042 /
    0.048 / 0.049 at tier 100: the control recalls weakly stored facts in pass 2, as the micro
    arm (a) did (0.269 / 0.369 / 0.378). Arm P tier 100 0.005 / 0.010 / 0.010 and tier 1000
    0.050 / 0.059 / 0.064 in distribution, held out 0.0125 / 0.012 / 0.0125 and 0.040 / 0.044 /
    0.046: a small growth with depth at tier 1000 in distribution, under the unpaired noise.
  - **A4, fails.** `eval_store.py recall` on `data/index/openqa` (8,715 questions, 274,212
    chunks): the model's query at loop 1 recall@5 0.004 against bge 0.829 (z -203), loops 2 and 3
    0.008 / 0.007, every source alike. The IR expert's query was trained by the selection loss
    over the handful of chunks visible to a document (chunk AUROC 0.78 there) and has never
    competed against a store; as a retriever it is at chance.
  - **A7, unreadable.** `pathway`: generated EM 0.000 with the model's query (abstains on 97%),
    0.026 with the oracle buffer (abstains on 88%), 0.022 with bge's buffer; the gate fails
    (z -12.5) against an oracle ceiling of 2.6% exact match, so the read is about a 99M model
    generating open-domain answers, not the pathway. `edit`: 0 of 2,000 answered correctly
    before the edit, so the flip rate has no denominator.
  - **A3, likelihood holds, behaviour unreadable.** `eval_abstention.py --evidence-port` on
    SQuAD v2 dev (2,000 questions): answer-span CE with the real target 2.13 under gold, 6.11
    under distractors, 5.87 with nothing attached (gold against none +3.75 nats). Generated:
    the model abstains on 95.6% of answerable questions under gold (EM 0.010), 99.5% under
    distractors and 100% under none; abstention precision 0.51 under gold. Counterfactual: 218
    eligible items, follow 0.018, stuck 0.000, "other" 0.982 (it abstains), so the follow gate
    fails for want of answers, not for memory; in likelihood the swapped answer wins (`mr_ll`
    0.114 pooled, 0.016 to 0.040 in the 0 to 99 strata, 0.208 / 0.217 in the 100+ strata).
    External memory mass AUROC for answerability 0.46 (chance). The SFT pattern on record:
    calibration passes, generated behaviour is all abstention.
  - **A6, within noise, holds.** `eval_benchmarks.py` on both arms (13 tasks, 6.7 minutes
    each): arm P / arm C hellaswag 0.265 / 0.263, arc_easy 0.370 / 0.359, arc_challenge 0.213 /
    0.216, piqa 0.575 / 0.565, winogrande 0.505 / 0.508, openbookqa 0.240 / 0.264, sciq 0.599 /
    0.566, boolq 0.410 / 0.488 (acc_norm 0.619 / 0.622), lambada 0.115 / 0.110, mmlu 0.244 /
    0.230, triviaqa and nq_open 0.000 / 0.000, gsm8k 0.014 / 0.017. Both arms sit at chance on
    most tasks (99M parameters, 0.9B tokens, mostly synthetic); arm P is within two sigma of
    arm C everywhere but boolq's raw accuracy (both under chance, the normalized score equal)
    and mmlu (+0.014, z 2.7), so the pathway costs no language at this shape.
  - **The ladder at the final saves, in one line each**: copy holds; A1(b) control half holds,
    arm P half fails on the held-out form at tier 100 by z 3.2 at a 4x to 47x reduction; 1-hop
    gate holds (0.85); L1 "falsified" (Delta +0.002 +- 0.005, nothing composes); A2 and A5
    hold; A3 holds in likelihood and is unreadable in generation; A4 fails; A7 unreadable; A6
    holds. The decision table's rows are the user's.
  - **The chain continuation, 2026-10-17** ([chain_continuation_pilot.md](../measurements/chain_continuation_pilot.md),
    `ckpts/evidence_pilot_chains/`, reads `ckpts/inject/pilot_ch_{TAG}_chains_*`): arm P's
    final save for one epoch of `chains_train` alone (102.66M tokens, lr 3e-4, batch 4 x 2 after
    batch 8 spilled, 307.9 minutes). Held-out answer CE 0.315 to 0.201 in a smooth decay, top-1
    per token 0.906 to 0.913; loop scales end [1.23, 0.49, 0.27] (no collapse). The final
    save's chain read: 1-hop 0.99 (z +87, the gold answer 0.04 nats), 2-hop 0.185 to 0.198
    (chance 0.185), 3-hop 0.17 to 0.21 (0.206), 4-hop 0.20 to 0.22 at depths 1 to 4 (0.210),
    held-out 2-hop 0.19 to 0.21 (0.191), paired kept 3 minus kept 1 +0.004 +- 0.008 (Delta + 2
    sigma 0.020 against 0.05), composed answers 1.93 nats against ln K 1.72. **L1 failed at
    this shape; the "falsified" row applies** (the user's rule of 2026-10-17). Still to read
    for the record: the 10M to 90M saves by the chain instrument.

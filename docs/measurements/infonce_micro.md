# The retriever's in-batch InfoNCE at micro scale (2026-10-17): the A4 fix against its twin

Built 2026-10-17 (`evidence_contrastive_loss` in `modules/model/information_retrieval.py`,
`LoopMixtureOfExperts.evidence_contrastive_term`, the trainer gate in `pretrain.train_step`,
`evidence_contrastive_weight` and `evidence_contrastive_temperature` in `TrainingConfig`,
`tests/test_evidence_contrastive.py`), on Co-LMLM's recipe from
[review_2026-10-16.md](../review_2026-10-16.md): the selector's unit query at every answer-start
position against every chunk in the batch (the other documents' chunks as negatives), the
position's own document's gold chunks as positives (joint mass when several, rows without one
excluded), both directions averaged, fixed temperature 0.07. The pilot's A4 read motivated it: arm
P's query at recall@5 0.004 against bge 0.829 over the 274k-chunk `openqa` store, the selector
having only ever ranked a document's few visible chunks.

## The arm

- `ckpts/inject/launch_infonce.sh`: `config_micro_infonce.yaml` (weight 0.25) against
  `config_micro_infonce0.yaml` (weight 0), identical otherwise (`config_micro.yaml` with the QA
  slice as data, evals every 20M); `sft.py --evidence --reader-kv` from `seed_micro.pt` on
  `data/prepared/evidence_nomany_train` (SQuAD v2, HotpotQA, web, replay; 301,653 conversations),
  `evidence_nomany_val` held out; batch 32 x 1024, lr 1e-3 cosine to 1e-4, seed 1234 on both.
- At seq 1024 the dataset skips the conversations over the row or the 3072 evidence cap (the
  long web rows), so one epoch is 34.82M prompt tokens, 1,508 optimizer steps, 15.5 minutes
  at 36k to 45k tokens per second and 17.2 GB; the trainer's plan (48,240 packed rows) matched.
- Each final read by `eval_store.py recall --store data/index/openqa --buffer 4 --loop 1`
  (8,715 questions over 274,212 chunks; outputs `ckpts/inject/infonce_{w25,w0}_store_recall.*`).

## One epoch (35M tokens)

Held-out `evidence_nomany_val` at 0 / 20M / 35M: the arm's `contrastive` 5.49 / 2.62 / 2.18
(ln of a batch's chunk count is about 5.5, so the query separates documents in-batch);
`selection` 0.735 / 0.717 / 0.712 on the arm against 0.735 / 0.716 / 0.714 on the twin (equal:
the term does not reach the within-document ranking the selection loss measures); answer CE 5.38
/ 5.15 against 5.36 / 5.12 (the term costs 0.025 nats of language at most); the port not yet
read on either arm (gold 5.38 against none 5.17 at the end, the fresh-port phase the R0b micro
arms sat in before 50M). On the training rows the arm puts about 1.7x the twin's read mass on
the store's chunks (external mass 0.042 against 0.025 at step 1000).

Store recall, loop 1 (bge 0.662 / 0.757 / 0.829 / 0.868 / 0.897 at k 1 / 2 / 5 / 10 / 20):

| arm | @1 | @2 | @5 | @10 | @20 | HotpotQA @20 | NQ @20 | TriviaQA @20 |
|---|---|---|---|---|---|---|---|---|
| weight 0.25 | 0.002 | 0.005 | 0.012 | 0.018 | 0.035 | 0.070 | 0.002 | 0.030 |
| weight 0 (twin) | 0.000 | 0.000 | 0.000 | 0.001 | 0.001 | 0.003 | 0.000 | 0.001 |
| arm P final (pilot) | | | 0.004 | | | | | |

Loops 2 and 3 read lower on the arm (0.008 and 0.006 at k 20). The gate (recall@5 at loop 1
against bge, paired): the arm finds 2 questions bge misses against 7,126 the other way, the
twin 0 against 7,227; both FAIL by the gate's bar.

**Read.** The term moves store recall from zero to small: about 105 questions at k 5 and 305
at k 20 on the arm against 0 and 9 on the twin, with the in-batch loss at 2.2 nats. In-batch
separation over about 250 chunks transfers weakly to 274k at this dose (35M tokens, a 35M
model, the port unread). The direction is the one A4 needs; the magnitude says nothing yet
about the pilot shape. The three-epoch round below reads the same pair at the dose where the
micro arms start reading the port.

## Three epochs (about 105M tokens)

`config_micro_infonce{,0}_e3.yaml` (`evidence.num_epochs: 3`, the cosine over the three
passes), arms `w25e3` and `w0e3`, same seeds and data; 104.46M tokens in 40.7 minutes per arm
(the first launch of the pair was stopped at 2M on 2026-10-17 and relaunched clean from the seed
on 2026-10-18).

The arm's held-out series at 20 / 40 / 60 / 80 / 100 / 104M: `contrastive` 2.66 / 1.49 / 1.17
/ 1.03 / 0.98 / 0.98, answer CE 5.37 / 4.95 / 4.68 / 4.55 / 4.47 / 4.45, `selection` 0.717 /
0.715 / 0.704 / 0.698 / 0.696 / 0.695. The twin at the same reads: answer CE 5.32 / 4.89 / 4.61
/ 4.48 / 4.39 / 4.38 (the arm 0.05 to 0.08 nats behind at every read, the gap widening from 0.05
at 20M to 0.08 at 100M and 0.074 at the end), `selection` 0.717 / 0.711 / 0.710 / 0.707 / 0.707
/ 0.706 (the arm 0.01 lower from 60M on, the term now reaching the within-document ranking
slightly). Gold still costs more than none at the end on both (arm 4.71 against 4.41, twin 4.61
against 4.34), so the port is not read on held-out rows at this dose on either. On the training
rows at the last logged step the external mass under `gold` is 0.123 on the arm against 0.076 on
the twin, under `distractors` 0.036 against 0.043 (one epoch: 0.042 against 0.025).

Store recall, loop 1 (bge 0.662 / 0.757 / 0.829 / 0.868 / 0.897 at k 1 / 2 / 5 / 10 / 20):

| arm | @1 | @2 | @5 | @10 | @20 | HotpotQA @20 | NQ @20 | TriviaQA @20 |
|---|---|---|---|---|---|---|---|---|
| weight 0.25, 3 epochs | 0.016 | 0.029 | 0.059 | 0.089 | 0.133 | 0.237 | 0.019 | 0.133 |
| weight 0, 3 epochs (twin) | 0.000 | 0.000 | 0.000 | 0.001 | 0.002 | 0.003 | 0.001 | 0.001 |
| weight 0.25, 1 epoch | 0.002 | 0.005 | 0.012 | 0.018 | 0.035 | 0.070 | 0.002 | 0.030 |

The twin was run twice: the first launch was stopped on the user's order at about 85M on
2026-10-18 (`ckpts/evidence_infonce_w0e3_stopped/`, `infonce_w0e3_stopped.log`, no final) and
the pair's number comes from a clean relaunch from the seed the same afternoon (104.46M tokens
in 39.8 minutes, `ckpts/evidence_infonce_w0e3/`, `infonce_w0e3_store_recall.*`).

Loops 2 and 3 stay at 0.005 and 0.004 at k 20 on the arm (0.001 on the twin): the read
extracts the query from a forward with no evidence attached, so the later loops' state is one
the term never saw. The gate (recall@5 at loop 1 against bge): the arm finds 14 questions bge
misses against 6,726 the other way, the twin 1 against 7,225; both FAIL by the bar. Three times
the tokens gave five times the recall at k 5 and four times at k 20 on the arm and nothing on the
twin; HotpotQA (whose distractor-free chunks the training rows resemble) leads NQ by 12x.

**Read.** At the dose where the micro arms start reading the port, the term is the whole of
the model's store recall: 0.059 against 0.000 at k 5 (about 515 questions against 2) and 0.133
against 0.002 at k 20 (1,160 against 18), one sixth of bge at k 20 and growing with dose. It
costs 0.07 nats of held-out language on a 35M model at weight 0.25, a gap that widens through
the run (0.05 at 20M, 0.08 at 100M), and buys 0.01 on the within-document selection loss and
about 1.6x the twin's read mass on the store's chunks. The port is read on neither arm here, so
whether the retrieved mass turns into answer gain is unread at micro scale; the pilot shape
reads it. The options put to the user: the term enters the pilot-shape recipe at 0.25 (the
language cost at that shape then read against the control, as every other term's is), at a
lower weight (no micro sweep was run; the 0.07 nats are the only cost number), or it stays
held. Recommended then: enter at 0.25, since it is the only lever that has moved A4 at all and
the cost is small against the pilot's between-arm gap of 0.013 nats only at the pilot's own
scale. The decision and its pilot-shape read follow.

## Pilot shape, arm PI (2026-10-21, stopped at 87M)

- Arm P's recipe (`config_pilot.yaml`: the pilot shape at 99.1M parameters, `pilot_warm_train`,
  from `seed_pilot_grounded.pt`, AdamW) with `evidence_contrastive_weight: 0.25` and
  `moe_balancing: loss_free` (`ckpts/inject/config_pilot_infonce.yaml`,
  `ckpts/inject/launch_pilot_pi.sh`, run directory `ckpts/evidence_pilot_pi`, log
  `ckpts/inject/pilot_pi.log`). 45k to 52k tokens per second, peak 25.3 GB.
- Stopped by STOP after the step 2660 eval (about 87M training tokens, trainer exit 10; saves at
  50M and 75M in the run directory) because the held-out language gap against arm P grew at
  every eval. The bar was held-out CE within the pilot's own noise: arm P and arm C differed by
  0.013 nats at their finals.
- The load fields stayed even (MLP load max over mean 1.03 to 1.09 per pass, dense share 0.01
  to 0.08), so the gap is the term's, not the balancing's. On the same training batch at step
  1770 the CE is 4.906 on arm PI against 4.750 on arm P. The contrastive term fell 5.58 (step
  10) / 2.59 (890) / 1.98 (1770), about 1.8 by step 2660.

`[eval]` CE on `inject_val` (arm P from `ckpts/inject/pilot_p.log` at the same steps; about
32.7k tokens per step):

| step | tokens | arm PI | arm P | gap |
|---|---|---|---|---|
| 890 | about 29M | 5.7867 | 5.7549 | +0.03 |
| 1770 | about 58M | 5.1581 | 5.0448 | +0.11 |
| 2660 | about 87M | 4.7552 | 4.4574 | +0.30 |

The port read (`[eval fixed]` on `evidence_fixed`, gold gain / content gain):

| step | arm PI | arm P |
|---|---|---|
| 890 | +0.46 / +0.025 | +0.30 / +0.034 |
| 1770 | +0.60 / +0.13 | +0.53 / +0.21 |
| 2660 | +1.04 / +0.43 | +2.02 / +1.76 |

At step 2660 arm P's gold gain is 62.6% of the 3.23 nat ceiling (answer CE gold 4.53 against
none 6.55), arm PI's 32.1% (gold 5.88 against none 6.91): the port read has taken off on arm P
and lags on arm PI.

Store recall on the 75M save (`eval_store.py recall --loop 1`,
`ckpts/inject/pilot_pi_75M_recall.{log,json}`; bge 0.662 / 0.757 / 0.829 / 0.868 / 0.897 at
k 1 / 2 / 5 / 10 / 20):

| arm | @1 | @2 | @5 | @10 | @20 | HotpotQA @5 / @20 | NQ @5 / @20 | TriviaQA @5 / @20 |
|---|---|---|---|---|---|---|---|---|
| arm PI, 75M, loop 1 | 0.001 | 0.003 | 0.006 | 0.011 | 0.018 | 0.012 / 0.029 | 0.000 / 0.001 | 0.006 / 0.024 |
| arm P final (pilot) | | | 0.004 | | | | | |

Loops 2 and 3 read 0.008 and 0.005 at k 5. The gate (recall@5 at loop 1 against bge, paired):
model 0.006 against 0.829, z -201 (model only 0, bge only 7,173 of 8,715): FAIL.

**Read.** At pilot shape and this dose the term, with its gradient into the trunk, costs 0.30
nats of held-out language and growing, while the model's own search has not moved (recall@5
0.006 against arm P's 0.004) and the port read lags arm P's (content gain +0.43 against +1.76
at step 2660). The in-batch term is much harder at pilot shape: it starts at the same ln of the
batch's chunk count as at micro (5.58 against 5.49) but is still about 1.8 nats at 87M, where
the micro arms read 2.2 at 35M (one epoch) and 1.0 at 80M (three epochs, both held out), so its gradient into the
trunk stays large for longer. Consequence: the term as built does not enter the real-run spec.
The live directions are the head-only term (trunk gradient stopped, loop 1 only) and a separate
search tower, per [review_2026-10-21_retriever.md](../review_2026-10-21_retriever.md); an
offline probe of the hidden state and one head-only micro arm are being built the same day.

## Head-only arm (2026-10-21): the term on loop 1 with no gradient into the trunk

- `ckpts/inject/launch_infonce_head.sh`, `config_micro_infonce_head_e3.yaml`: the `w25e3`
  recipe (weight 0.25, three epochs, same seed and data) with `evidence_contrastive_loops: [0]`
  (the term on loop 1 only, the pass the store read uses) and
  `evidence_contrastive_trunk_grad: 0.0` (the term's gradient stops at the IR expert's input, so
  it trains the search head and nothing upstream). Run directory
  `ckpts/evidence_infonce_head_e3`, log `ckpts/inject/infonce_head_e3.log`; 104.46M tokens over
  3 epochs in 39.4 minutes. Recall read by `eval_store.py recall --loop 1` into
  `ckpts/inject/infonce_head_e3_store_recall.{log,json}`.
- Bar, set at launch in [query_head_probe.md](query_head_probe.md): held-out language within
  noise of `w0e3` and recall above `w0e3`; the probe predicted about 0.02 to 0.03 at k5.

Held-out `[eval]` CE on `evidence_nomany_val` at the same steps on the three arms (the steps
restart per epoch, 1,508 optimizer steps each, so the reads are keyed by epoch and step; tokens
approximate):

| eval (epoch, step) | tokens | head-only | `w0e3` | `w25e3` | head-only minus `w0e3` | `w25e3` minus `w0e3` |
|---|---|---|---|---|---|---|
| 0, 870 | about 20M | 5.3151 | 5.3164 | 5.3707 | -0.001 | +0.054 |
| 1, 230 | about 40M | 4.8799 | 4.8905 | 4.9452 | -0.011 | +0.055 |
| 1, 1100 | about 60M | 4.6212 | 4.6129 | 4.6776 | +0.008 | +0.065 |
| 2, 460 | about 80M | 4.4942 | 4.4756 | 4.5454 | +0.019 | +0.070 |
| 2, 1330 | about 100M | 4.3933 | 4.3902 | 4.4711 | +0.003 | +0.081 |
| 2, 1511 (final) | 104.46M | 4.3904 | 4.3809 | 4.4546 | +0.010 | +0.074 |

The language cost is zero within noise: the head-only arm sits within 0.02 of `w0e3` at every
eval, ahead at two, 0.010 behind at the final (the seed spread at micro is about 0.03), while
`w25e3` is 0.05 to 0.08 behind at every eval.

The held-out `contrastive` at the same evals: head-only 5.53 at step 0, then 4.61 / 4.40 /
4.22 / 4.11 / 4.03 / 4.02; `w25e3` 5.49, then 2.66 / 1.49 / 1.17 / 1.03 / 0.98 / 0.98. Against
ln of a batch's chunk count (about 5.5) the head alone takes off 1.5 nats of the in-batch task
where the term with the trunk takes off 4.5: the head barely learns it. `selection` ends at
0.716 against 0.706 on `w0e3` and 0.695 on `w25e3` (the term on a frozen trunk pulls the same
query head the selection loss trains, most likely why the within-document ranking is 0.01
worse). Gold still costs more than none at the end (4.63 against 4.34), the port unread as on
both twins.

Store recall, loop 1, 8,715 questions over 274,212 chunks:

| system | @1 | @2 | @5 | @10 | @20 | HotpotQA @5 / @20 | NQ @5 / @20 | TriviaQA @5 / @20 |
|---|---|---|---|---|---|---|---|---|
| head-only | 0.000 | 0.001 | 0.003 | 0.004 | 0.007 | 0.007 / 0.017 | 0.000 / 0.000 | 0.000 / 0.002 |
| `w25e3` (term, trunk gradient, every loop) | 0.016 | 0.029 | 0.059 | 0.089 | 0.133 | 0.111 / 0.237 | 0.008 / 0.019 | 0.055 / 0.133 |
| `w0e3` (twin, term off) | 0.000 | 0.000 | 0.000 | 0.001 | 0.002 | 0.001 / 0.003 | 0.000 / 0.001 | 0.001 / 0.001 |
| bge | 0.662 | 0.757 | 0.829 | 0.868 | 0.897 | 0.958 / 0.985 | 0.723 / 0.813 | 0.798 / 0.884 |

Loops 2 and 3 read zero at every k (TriviaQA 0.001 at k 20). The gate (recall@5 at loop 1
against bge, paired): model 0.003 against 0.829, z -204 (model only 0, bge only 7,205 of
8,715): FAIL.

**Read.** With no gradient into the trunk the search head gets nothing: 0.003 at k5, at the
probe's ridge row on the last token (0.004 to 0.006 on `w0e3`), under the probe's pooled MLP
(0.025) because the real head is a linear map on the last token, and under the probe's
prediction of 0.02 to 0.03. The probe's heads were fitted on the store's own questions against
bge's vector, this head on in-batch negatives from the training slice, so part of the shortfall
may be the objective rather than the head's form; neither reaches a usable retriever. The whole
of `w25e3`'s 0.059 therefore needed the trunk to change, and that change cost 0.07 nats at
micro and 0.30 at pilot shape (arm PI). The "train only the search head" direction is closed at
this size. What remains: a separate search tower (a small bge-style encoder over the prompt
text beside the model, exactly zero language cost, a retriever beside the model rather than the
trunk searching) or holding (bge for serving, A4 untested, revisit at the real run's size). The
user's call.

## Decision

- Decided 2026-10-21 (the user): the term enters the pilot-shape recipe at 0.25 and is read at
  pilot shape before the real-run spec commits to it.
- The pilot-shape read (arm PI) failed the language bar at 87M: 0.30 nats behind arm P and
  growing, against a bar of the pilot's own noise (0.013), with store recall unmoved.
- The term as built (gradient into the trunk, every loop) stays out of the real-run spec.
- Next: the offline probe of the hidden state, then the head-only micro arm.
- The head-only arm, read 2026-10-21: zero language cost (4.3904 against `w0e3`'s 4.3809)
  and zero search (recall@5 0.003 against `w25e3`'s 0.059), so training only the search head
  is closed at this size; a separate search tower or holding is the user's call.

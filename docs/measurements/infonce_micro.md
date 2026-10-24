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
offline probe of the hidden state and one head-only micro arm are being built the same day
(both read the same day; the tower held, the Decision below).

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
trunk searching) or holding. Decided 2026-10-21 (the user): the tower is held, bge keys the
store and embeds the question at serving, A4 stays untested, and the model's own retrieval is
revisited at the real run's size (the two keys stay as the knob).

## The first pass only arm and the fresh twin (2026-10-24)

Two of the three micro retriever arms the repository review proposed
([review_2026-10-24.md](../review_2026-10-24.md)), on the code in the tree after the review's
fixes (micro shape, key/value reader, three passes over `evidence_nomany_train` from
`ckpts/inject/seed_micro.pt`, the same seed and data as the pairs above; launcher
`ckpts/inject/launch_review_arms.sh`):

- `w0e3b`, the fresh twin: the `w0e3` recipe (no contrastive term) rerun on the reviewed code,
  `ckpts/inject/config_micro_infonce0_e3.yaml`, run directory `ckpts/evidence_infonce_w0e3b`, log
  `ckpts/inject/infonce_w0e3b.log`, read `ckpts/inject/infonce_w0e3b_store_recall.{json,log}`;
  104.56M tokens, exit 0. The code differences against the old `w0e3`: the log-space selection
  loss, the fp32 CE accounting, the dense-slot GEMM order.
- `l0e3`: the `w25e3` recipe (weight 0.25) with `evidence_contrastive_loops: [0]` (the term on the
  first pass only) and `evidence_contrastive_trunk_grad: 1.0` (the gradient into the trunk on),
  `ckpts/inject/config_micro_infonce_l0_e3.yaml`, run directory `ckpts/evidence_infonce_l0e3`, log
  `ckpts/inject/infonce_l0e3.log`, read `ckpts/inject/infonce_l0e3_store_recall.{json,log}`;
  launched 21:02 fresh, 104.46M tokens in 39.3 minutes at about 45k tokens per second, exit 0.
- The third arm, `seg_e3` (the `w0e3` recipe with the selection loss weighted per query segment),
  runs 2026-10-25; its yaml is parked as `ckpts/inject/config_micro_segsel_e3.yaml.hold` (rename it
  back first).

The twin was interrupted once and resumed twice. The watch that owned the launcher expired at
20:58 and killed launcher and trainer at 74M (last save 50M); the first detached resume ran
under the SFT resume epoch bug (accelerate's prepared loader set the dataset epoch from its own
pass counter, so a resume at epoch 1 read epoch 0's permutation; fixed that evening by
`sft.select_epoch`, test 8 of `tests/test_sft_resume_state.py`; no recorded run ever resumed past
epoch 0) and was killed before any save, and the resume at 21:45 under the fix from the 50M save
reproduced the original run's 60M eval within 0.0001 (4.6262 against 4.6261).

Held-out `[eval]` CE on `evidence_nomany_val` (tokens approximate; `w0e3` and `w25e3` from the
head-only table above):

| eval | `w0e3b` | `w0e3` | `l0e3` | `w25e3` | `w0e3b` minus `w0e3` | `l0e3` minus `w0e3b` | `w25e3` minus `w0e3` |
|---|---|---|---|---|---|---|---|
| about 20M | 5.3049 | 5.3164 | 5.3164 | 5.3707 | -0.012 | +0.012 | +0.054 |
| about 40M | 4.8834 | 4.8905 | 4.9015 | 4.9452 | -0.007 | +0.018 | +0.055 |
| about 60M | 4.6261 (resumed 4.6262) | 4.6129 | 4.6511 | 4.6776 | +0.013 | +0.025 | +0.065 |
| about 80M | 4.4903 | 4.4756 | 4.5251 | 4.5454 | +0.015 | +0.035 | +0.070 |
| about 100M | 4.4073 | 4.3902 | 4.4477 | 4.4711 | +0.017 | +0.040 | +0.081 |
| final | 4.3996 | 4.3809 | 4.4404 | 4.4546 | +0.019 | +0.041 | +0.074 |

The fresh twin ends 0.019 above the old one, inside the micro seed spread of about 0.03. The
held-out `contrastive` on `l0e3`: 5.53 at step 0, then 3.79 / 2.89 / 2.23 / 1.96 / 1.84 / 1.84
at the reads above (`w25e3` 2.66 / 1.49 / 1.17 / 1.03 / 0.98 / 0.98, the head-only arm 4.02 at
the end).

Store recall (`eval_store.py recall`, store `data/index/openqa`, 8,715 questions over 274,212
chunks; bge 0.829 / 0.897 at k 5 / 20):

| arm | term loops | trunk grad | final CE | @5, loop 1 | @20, loop 1 | @5, loops 2 / 3 |
|---|---|---|---|---|---|---|
| `w0e3b` (fresh twin) | none | | 4.3996 | 0.000 | 0.002 | |
| `w0e3` (old twin) | none | | 4.3809 | 0.000 | 0.002 | |
| `l0e3` | first pass | 1.0 | 4.4404 | 0.049 | 0.122 | 0.010 / 0.005 |
| `w25e3` | every loop | 1.0 | 4.4546 | 0.059 | 0.133 | |
| head-only | first pass | 0.0 | 4.3904 | 0.003 | 0.007 | |

`l0e3` in full at loop 1: @1 0.015, @2 0.025, @5 0.049, @10 0.079, @20 0.122; per source at k 5 /
k 20 HotpotQA 0.095 / 0.232, NQ 0.005 / 0.016, TriviaQA 0.044 / 0.108 (`w25e3` 0.111 / 0.237,
0.008 / 0.019, 0.055 / 0.133). Loops 2 and 3 read 0.010 and 0.005 at k 5, 0.030 and 0.015 at
k 20. The gate (recall@5 at loop 1 against bge, paired): `l0e3` z -174.3, `w0e3b` z -205.6;
both FAIL.

**Read.** The term on the first pass alone keeps about five sixths of the every-loop term's small
recall gain (0.049 against 0.059 at k 5, 0.122 against 0.133 at k 20) at a language cost of 0.041
nats against the fresh twin (4.4404 against 4.3996), where the recorded every-loop pair cost
0.074. So the language cost sits in the trunk's change and not in the later passes, and the later
passes, where the term no longer acts, find almost nothing (0.010 and 0.005 at k 5). The gate
stays failed; the decision of 2026-10-21 (no retriever term in the real-run spec, the model's own
retrieval revisited at the real run's size) is unchanged. Next: `seg_e3` on 2026-10-25, then the
two leak arms and the clean-slice CE re-read.

## Decision

- Decided 2026-10-21 (the user): the term enters the pilot-shape recipe at 0.25 and is read at
  pilot shape before the real-run spec commits to it.
- The pilot-shape read (arm PI) failed the language bar at 87M: 0.30 nats behind arm P and
  growing, against a bar of the pilot's own noise (0.013), with store recall unmoved.
- The term as built (gradient into the trunk, every loop) stays out of the real-run spec.
- Next: the offline probe of the hidden state, then the head-only micro arm.
- The head-only arm, read 2026-10-21: zero language cost (4.3904 against `w0e3`'s 4.3809)
  and zero search (recall@5 0.003 against `w25e3`'s 0.059), so training only the search head
  is closed at this size.
- Decided 2026-10-21 (the user): the separate search tower (a small bge-style encoder over the
  prompt text beside the model) is held, not built; bge keys the store and embeds the question at
  serving, A4 stays untested, and the model's own retrieval is revisited at the real run's size.
  `evidence_contrastive_loops` and `evidence_contrastive_trunk_grad` stay in the code as the knob
  for that revisit; the real-run spec carries no retriever term.
- The first pass only arm, read 2026-10-24 against a fresh twin on the reviewed code: recall@5
  0.049 against `w25e3`'s 0.059 at 0.041 nats of held-out language against the twin (the
  every-loop pair 0.074); the cost sits in the trunk's change, the gate stays failed, and the
  decision of 2026-10-21 is unchanged.

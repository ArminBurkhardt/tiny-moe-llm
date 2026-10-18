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
reads it. Decision pending (the user): the term enters the pilot-shape recipe at 0.25 (the
language cost at that shape is then read against the control, as every other term's is), at a
lower weight (no micro sweep was run; the 0.07 nats are the only cost number), or it stays
held. Recommended: enter at 0.25, since it is the only lever that has moved A4 at all and the
cost is small against the pilot's between-arm gap of 0.013 nats only at the pilot's own scale.

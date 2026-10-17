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
passes), arms `w25e3` and `w0e3`, same seeds and data; the result goes here.

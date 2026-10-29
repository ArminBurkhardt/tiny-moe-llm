# Seed-side instruments: PopQA with a prior control, and the counterfactual likelihood read

Date 2026-10-06 (PopQA run 2026-10-02, counterfactual likelihood run 2026-10-06). Checkpoints:

- the seed `ckpts/repair/checkpoint_repair_final_irrandom_evidence_grounded.pt` (16.76B training
  tokens, the evidence port migrated in and neutral);
- arm A, the grafted reader as built, 10M evidence tokens
  ([evidence_arm_a.md](evidence_arm_a.md)): PopQA on the kill save
  `ckpts/evidence/checkpoint_evidence_tok10M_loss4.5772.pt`, the counterfactual on the cadence save
  `ckpts/evidence/checkpoint_evidence_tok10M_loss3.7712.pt`;
- arm B, the same without reader rotary ([evidence_arm_b.md](evidence_arm_b.md)): PopQA on
  `ckpts/evidence_norope/checkpoint_evidence_tok10M_loss4.6326.pt`, the counterfactual on
  `ckpts/evidence_norope/checkpoint_evidence_tok10M_loss3.7359.pt`.

The cadence and kill saves of each arm are one optimizer step apart (scheduler step 242 against
243, 188 of 400 tensors differ, max abs difference 0.0078 in both arms), so they are the same model
for this purpose.

Commands:

```
python scripts/eval_benchmarks.py -c CKPT --tasks popqa --rank-candidates 20 --json-out ckpts/instruments/popqa_<name>.json
python scripts/eval_abstention.py -c CKPT --evidence-port --evidence-condition counterfactual --counterfactual-likelihood-only --json-out ckpts/instruments/cfll_<name>.json
```

The generation-mode counterfactual runs of 2026-10-02 (`cf_{seed,armA,armB}.log`) printed an
empty correct-with-gold table: both arms abstain on almost every answerable row, so nothing was
eligible for the follow rate. The likelihood-only read exists for that reason.

**Verdict.** PopQA with its prior control resolves the seed's weak stored knowledge, and 10M tokens
of evidence finetuning moved neither the PopQA `delta` nor counterfactual following. The raw rank
alone would have read a 7 sigma effect of the finetune at the tail; it is the answer prior moving.

## PopQA

The gold object is ranked among 20 same-relation objects by summed log-probability; `norm_rank` 0
is first, chance 0.5. The prior control scores the same candidates with the subject replaced by
`X`; `delta = prior - real`, positive means the subject helps. Tiers are thirds of subject page
views (tail at most 375, mid at most 2,879, head above). 14,233 items (11,197 entity, 3,036
attribute), relation `color` excluded (5 distinct objects), 3 items without a prior; the item set
is identical across the three runs. The headline `1 - norm_rank` (entity) is 0.5716 for the seed,
0.5667 for arm A and 0.5688 for arm B.

| checkpoint | class | tier | n | norm_rank | prior | delta | z | top1 |
|---|---|---|---|---|---|---|---|---|
| seed | entity | tail | 3,454 | 0.4583 | 0.4869 | 0.0287 | 11.7 | 0.097 |
| seed | entity | mid | 3,843 | 0.4514 | 0.4777 | 0.0263 | 12.8 | 0.084 |
| seed | entity | head | 3,900 | 0.3793 | 0.4150 | 0.0356 | 19.4 | 0.157 |
| seed | entity | all | 11,197 | 0.4284 | 0.4587 | 0.0303 | 24.9 | 0.114 |
| seed | attribute | tail | 1,306 | 0.3230 | 0.3372 | 0.0143 | 4.6 | 0.156 |
| seed | attribute | mid | 907 | 0.3117 | 0.3350 | 0.0233 | 6.1 | 0.159 |
| seed | attribute | head | 823 | 0.3018 | 0.3260 | 0.0241 | 6.5 | 0.156 |
| arm A | entity | tail | 3,454 | 0.4673 | 0.4971 | 0.0299 | 12.9 | 0.093 |
| arm A | entity | mid | 3,843 | 0.4568 | 0.4832 | 0.0264 | 13.7 | 0.083 |
| arm A | entity | head | 3,900 | 0.3800 | 0.4140 | 0.0339 | 19.9 | 0.156 |
| arm A | entity | all | 11,197 | 0.4333 | 0.4634 | 0.0301 | 26.1 | 0.112 |
| arm B | entity | tail | 3,454 | 0.4641 | 0.4948 | 0.0309 | 13.2 | 0.093 |
| arm B | entity | mid | 3,843 | 0.4550 | 0.4821 | 0.0270 | 14.4 | 0.084 |
| arm B | entity | head | 3,900 | 0.3786 | 0.4132 | 0.0345 | 20.1 | 0.155 |
| arm B | entity | all | 11,197 | 0.4312 | 0.4620 | 0.0308 | 26.8 | 0.112 |

Paired on the same items, arm minus seed (bootstrap sigma of the paired difference):

| class, tier | arm A real | arm A prior | arm A delta | arm B real | arm B prior | arm B delta |
|---|---|---|---|---|---|---|
| entity, tail | +0.0090 (z 7.4) | +0.0102 (z 8.2) | +0.0012 (z 1.0) | +0.0057 (z 5.1) | +0.0079 (z 6.7) | +0.0022 (z 1.9) |
| entity, mid | +0.0055 (z 5.3) | +0.0055 (z 5.2) | +0.0001 (z 0.1) | +0.0036 (z 3.6) | +0.0044 (z 4.2) | +0.0007 (z 0.8) |
| entity, head | +0.0007 (z 0.7) | -0.0010 (z -1.0) | -0.0017 (z -1.7) | -0.0007 (z -0.7) | -0.0018 (z -1.8) | -0.0011 (z -1.2) |
| entity, all | +0.0049 (z 7.9) | +0.0047 (z 7.3) | -0.0002 (z -0.4) | +0.0028 (z 4.7) | +0.0033 (z 5.2) | +0.0005 (z 0.9) |
| attribute, all | +0.0087 (z 5.9) | +0.0050 (z 3.7) | -0.0037 (z -2.9) | +0.0079 (z 5.4) | +0.0053 (z 3.7) | -0.0026 (z -2.1) |

A positive `real` or `prior` difference is a worse rank in the arm.

The cue relations. Four relations carry most of the entity `delta`: father, mother, capital and
capital of (1,764 items).

| subset | tail | mid | head | all |
|---|---|---|---|---|
| cue relations, seed delta | 0.261 (z 13.1, n 272) | 0.174 (z 14.1, n 434) | 0.113 (z 21.3, n 1,058) | 0.151 (z 27.9) |
| other entity relations, seed delta | 0.0088 (z 5.1, n 3,181) | 0.0075 (z 5.7, n 3,409) | 0.0070 (z 5.4, n 2,840) | 0.0078 (z 9.2, n 9,430) |
| cue relations, arm A minus seed | -0.010 (z -2.6) | -0.005 (z -1.6) | -0.009 (z -4.2) | -0.008 (z -5.0) |
| other entity relations, arm A minus seed | +0.0022 (z 1.8) | +0.0008 (z 0.7) | +0.0009 (z 0.9) | +0.0013 (z 2.1) |

Per relation (seed): father 0.162 (z 17.4), capital 0.174 (z 16.6), capital of 0.129 (z 13.3),
mother 0.080 (z 7.2); the next largest are country 0.027 and place of birth 0.026.

- **The prior control is required.** Paired at the entity tail, arm A ranks the gold object worse
  than the seed by 0.0090 (z 7.4), but the prior moved by 0.0102 (z 8.2), so `delta` moved +0.0012
  (z 1.0). Over all entity items `delta` moved -0.0002 (z -0.4) for arm A and +0.0005 (z 0.9) for
  arm B. 10M tokens of evidence finetuning shifted the answer prior and stored nothing; a raw-rank
  reading would have reported a 7 sigma effect.
- **The seed's stored signal is small but resolved.** Entity `delta` 0.029 / 0.026 / 0.036 by tier,
  z 11.7 to 19.4; the paired sigma at the tail is about 0.0012, so the instrument resolves changes
  of a few thousandths.
- **Most of it sits in four relations, and there it runs against popularity.** The cue relations
  read 0.261 at the tail against 0.113 at the head, the reverse of what stored knowledge should do.
  The likely reason is a cue in the subject string (a parent sharing the subject's surname, a
  capital named in or like the country), a hypothesis not separated here. The other 9,430 entity
  items read 0.0088 / 0.0075 / 0.0070 by tier (z 5.1 / 5.7 / 5.4), flat across popularity. Arm A
  lost a little on the cue relations (-0.008, z -5.0) and nothing elsewhere.

## Counterfactual likelihood

SQuAD v2 dev, 11,873 questions; 1,409 eligible (skipped: unanswerable 5,945, answer type not
swappable 3,432, no substitute 1,084, answer not in the passage 3). The gold chunk is given with the
answer entity swapped for a same-type substitute; `mr_ll` is the mean of
`sigmoid(ll_orig - ll_swap)` under that chunk (1 keeps memory, 0 follows the chunk); strata count
the original answer's token sequence in `data/prepared/ir.bin`. The item set and the substitutes
are identical across the three runs.

| stratum | n | seed mr_ll | seed prefers_swap | arm A mr_ll | arm A prefers_swap | arm B mr_ll | arm B prefers_swap |
|---|---|---|---|---|---|---|---|
| 0 | 293 | 0.295 | 0.706 | 0.312 | 0.700 | 0.310 | 0.693 |
| 1-9 | 213 | 0.436 | 0.573 | 0.499 | 0.507 | 0.494 | 0.507 |
| 10-99 | 210 | 0.578 | 0.424 | 0.601 | 0.400 | 0.591 | 0.419 |
| 100-999 | 275 | 0.606 | 0.396 | 0.611 | 0.404 | 0.612 | 0.411 |
| 1000+ | 418 | 0.766 | 0.208 | 0.770 | 0.196 | 0.767 | 0.191 |
| all | 1,409 | 0.559 | 0.436 | 0.577 | 0.419 | 0.574 | 0.420 |

Paired, arm minus seed, in nats summed over the answer (gap = ll_orig - ll_swap):

| stratum | A ll_orig | A ll_swap | A gap | A mr_ll | B ll_orig | B ll_swap | B gap | B mr_ll |
|---|---|---|---|---|---|---|---|---|
| 0 | +3.90 | +2.13 | +1.77 (z 6.4) | +0.017 (z 1.5) | +3.83 | +2.01 | +1.82 (z 6.2) | +0.015 (z 1.4) |
| 1-9 | +2.79 | +2.23 | +0.57 (z 2.0) | +0.063 (z 3.7) | +2.69 | +2.25 | +0.45 (z 1.5) | +0.058 (z 3.2) |
| 10-99 | +1.68 | +1.70 | -0.02 (z -0.1) | +0.023 (z 1.7) | +1.47 | +1.67 | -0.20 (z -0.8) | +0.013 (z 1.0) |
| 100-999 | +0.94 | +1.66 | -0.72 (z -3.6) | +0.005 (z 0.4) | +0.65 | +1.46 | -0.82 (z -3.9) | +0.006 (z 0.5) |
| 1000+ | +0.37 | +1.39 | -1.02 (z -7.6) | +0.003 (z 0.4) | +0.01 | +1.09 | -1.08 (z -7.7) | +0.001 (z 0.1) |
| all | +1.78 | +1.77 | +0.01 (z 0.1) | +0.018 (z 3.3) | +1.55 | +1.62 | -0.07 (z -0.6) | +0.015 (z 2.8) |

Seed `mr_ll` binned by the log10 count ratio of original over substitute:

| log10 ratio | under -1 | -1 to -0.3 | -0.3 to 0.3 | 0.3 to 1 | 1 and over |
|---|---|---|---|---|---|
| n | 311 | 228 | 255 | 217 | 398 |
| seed mr_ll | 0.050 | 0.329 | 0.564 | 0.794 | 0.957 |

Median substitute count per stratum 5 / 5 / 19 / 103 / 705 against median original counts 0 / 2 /
34 / 229 / 4,976.

- **The seed's numbers are closed-book.** Its port is neutral (gold minus none answer-span CE gap
  +0.0000 in `cf_seed.log`), so the 0.295 to 0.766 gradient over strata is a frequency prior, not a
  reading of the chunk. The log10 count ratio correlates 0.636 with `ll_orig - ll_swap`, and the
  ratio bins run from 0.050 (substitute over ten times more frequent) to 0.957 (original over ten
  times more frequent). Strata keyed on the original's count alone are confounded by the ratio,
  because the substitute's count rises with the original's.
- **The arms do not follow.** Both answers gain about 1.6 to 1.8 nats under a buffer (arm A
  ll_orig +1.78, ll_swap +1.77; arm B +1.55 / +1.62), the buffer-presence effect seen in the 10M
  arms' fixed-target reading. The paired gap does not move pooled (A +0.01, z 0.1; B -0.07, z
  -0.6); by stratum it shrinks toward zero from both sides (stratum 0 +1.77 toward the original at
  z 6.4, 100-999 -0.72 at z -3.6, 1000+ -1.02 at z -7.6), which is the prior flattening, not
  following. Pooled `mr_ll` rises slightly (A +0.018, z 3.3), away from following. Neither arm
  follows the swapped card in any stratum.

## What it decides

- **A1(a)**: the PopQA instrument has the resolution (paired sigma about 0.0012 at the tail), but
  the seed's stored signal outside the cue relations is tiny (0.0088 at the tail), so a pilot
  control trained on fewer tokens may not clear 3 sigma at the tail. A1(a) reports the cue
  relations separately and requires the no-retrieval control to sit above its prior at the tail
  before the pathway arm is read against it. Never read the raw rank.
- **A3**: read the memorization bar in high-frequency strata paired against a neutral-port
  baseline, or binned by the frequency ratio of original to substitute, never by the original's
  count alone, or a reader that copies will look stratum-dependent when it is not.
- **Relation to R0b**: the micro key/value reader follows a swapped card on every in-distribution
  item (`mr_ll` 0.000); the grafted cross reader at 10M sits at the seed's prior. At full shape,
  expect `mr_ll` to stay on the seed's frequency curve until copying emerges.
- **Established**: PopQA with the prior control resolves the seed's weak stored knowledge; 10M
  tokens of evidence finetuning moved neither the PopQA `delta` nor counterfactual following.
- **Not established**: whether the cue-relation `delta` is stored knowledge or a surface cue; any
  A3 reading proper (no arm answers, so there is no follow rate on answered items).

Raw output (gitignored): `ckpts/instruments/popqa_{seed,armA,armB}.{log,json}`,
`ckpts/instruments/cfll_{seed,armA,armB}.{log,json}`, `ckpts/instruments/cf_{seed,armA,armB}.log`
(the generation-mode runs).

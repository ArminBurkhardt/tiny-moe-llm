# Held-out CE on a clean validation slice (2026-10-25): arm P, arm C and arm E4

The question, from the repository review of 2026-10-24 ([review_2026-10-24.md](../review_2026-10-24.md),
finding 3): `inject_val` is partly a training read. Its windows come from the same `ir.bin`
documents the evidence QA builder cuts its web rows from, excluded per window, not per document;
about 7.4% of its tokens appear verbatim as supervised text in rows both pilot arms train on and
another 6.7% in the 4094-token rows only the control (arm C) trains on (the evidence reader drops
them, review finding 2). Every held-out CE on `inject_val` therefore leans toward the arm that saw
more of it, and the record's reading "arm C final CE 3.2195 against arm P's 3.2321" (the control
0.013 better, [PILOT.md](../plans/PILOT.md)) is the comparison most exposed. Re-read here: the
three pilot finals on a slice cut from documents the evidence builder never used, beside the same
read on `inject_val`.

## Setup

- The slice: `data/prepared_pilot/inject_val_clean`, built 2026-10-24 by
  `scripts/prepare_clean_val.py` (`tests/test_prepare_clean_val.py`): 974 windows, 650,125 tokens,
  `inject_val`'s windows minus those from the first 80,672 `ir.bin` documents the evidence QA
  builder used (phase1 and phase2 windows kept). Verbatim 64-token overlap with the QA training
  rows' supervised tokens 0.99% against 13.3% for `inject_val` under the same check (the review's
  7.4% is a narrower definition).
- The read: `bash ckpts/inject/read_clean_val.sh`, run 2026-10-25 16:16 to 16:25 as the last step
  of `ckpts/inject/queue_2026_10_25b.sh`, every read exit 0. `scripts/eval_exit.py`, 40 batches of
  4, no evidence attached, per exit, on each slice, for arm P
  (`ckpts/evidence_pilot_p/checkpoint_evidence_final.pt`, `config_pilot.yaml`), arm C
  (`ckpts/evidence_pilot_c/checkpoint_evidence_final.pt`, `config_pilot.yaml`) and arm E4
  (`ckpts/evidence_pilot_e4/checkpoint_evidence_final.pt`, `ckpts/inject/config_pilot_e4.yaml`).
- 588,155 supervised tokens on the clean slice (889 clusters), 598,164 on `inject_val` (704
  clusters); the same tokens for every arm on a slice. These 40 batches are not the trainer's
  `[eval]` batches (whose finals were 3.2321, 3.2195 and 3.2396 for P, C and E4), so the numbers
  compare within this read only.

## CE per exit

| arm | slice | exit 1 | exit 2 | exit 3 | gain 1 to 3 |
|---|---|---|---|---|---|
| P | clean | 3.3478 | 3.2739 | 3.2669 | +0.0809 |
| C | clean | 3.3503 | 3.2696 | 3.2650 | +0.0854 |
| E4 | clean | 3.3474 | 3.2806 | 3.2740 | +0.0735 |
| P | `inject_val` | 3.4898 | 3.4166 | 3.4092 | +0.0806 |
| C | `inject_val` | 3.4697 | 3.3894 | 3.3844 | +0.0852 |
| E4 | `inject_val` | 3.4932 | 3.4274 | 3.4201 | +0.0731 |

The paired gain 1 to 3 has a cluster standard error of 0.0008 to 0.0011 on every row (z 74 to
100).

Differences against arm P on the same tokens (negative means better than arm P):

| slice | C minus P, exits 1 / 2 / 3 | E4 minus P, exits 1 / 2 / 3 |
|---|---|---|
| clean | +0.0025 / -0.0043 / -0.0019 | -0.0004 / +0.0067 / +0.0071 |
| `inject_val` | -0.0201 / -0.0272 / -0.0248 | +0.0034 / +0.0108 / +0.0109 |

## Verdict (the main thread's)

- **The control's lower held-out CE was contamination, not better language.** At exit 3 the
  control leads arm P by 0.025 on `inject_val` and by 0.002 on the clean slice: the lead was the
  validation documents' presence in rows only the control trains on (the review's 6.7%). On clean
  text arm P and the control are equal (C is 0.0025 behind P at exit 1 and 0.004 ahead at exit 2).
- **E4 is 0.007 behind arm P on clean text** (0.011 on `inject_val`), under the pilot's noise of
  0.013, as the record read it on the trainer's evals.
- **The depth gains repeat** on every arm and slice: exit 1 to 3 about 0.08 (0.073 to 0.085),
  almost all in pass 2, as in every per-exit read since 2026-10-05.
- **This amends the reading "arm C final CE 3.2195 against arm P's 3.2321"** in PILOT.md and the
  record: that 0.013 gap is contamination; on clean text arm P and its matched control are equal
  within this read (0.002 at exit 3, no paired sigma).

## Open

- No paired per-token sigma between arms in this read: each arm was read on its own. `eval_exit.py
  --pair TWIN` on the clean slice (no evidence, so the pair mode accepts it) would give one for
  P against C and P against E4 if a number at that precision is ever needed.
- The future language bar for pilot-shape arms reads on `inject_val_clean`, not `inject_val`, at
  the user's call (the trainer's `[eval]` still reads the split its config names).
- Earlier reads on `inject_val` that compared arms with different exposure to the contaminated
  rows (arm PI's language stop against arm P, the E4 read) compared arms that drop the same rows,
  so they are matched in exposure; only the comparisons against arm C carry the bias.

## Raw files

`ckpts/inject/pilot_{p,c,e4}_exit_{inject_val_clean,inject_val}.{json,log}`,
`ckpts/inject/read_clean_val.log`, `read_clean_val_run.log`, the script
`ckpts/inject/read_clean_val.sh`.

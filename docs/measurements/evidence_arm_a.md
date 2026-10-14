# Evidence finetune, arm A (reader as built, gate frozen)

Date 2026-10-01. Seed `ckpts/repair/checkpoint_repair_final_irrandom_evidence_grounded.pt`,
`python scripts/sft.py --evidence -c <seed>`, HEAD d3a0fde, log `ckpts/evidence_armA.log`
(gitignored), checkpoints `ckpts/evidence/checkpoint_evidence_tok10M_loss3.7712.pt` (the cadence
save) and `checkpoint_evidence_tok10M_loss4.5772.pt` (the kill save, `kill_checked` set).

**Verdict: killed by the automatic rule at 10.08M evidence tokens.** Gold gain +0.5448 nats cleared
the 0.1 bar; content gain (gold minus distractors) +0.0577 did not. Against the pass bar of 1.6 nats
the gold gain is 17% of the SQuAD in-context ceiling, and almost all of it is condition-independent:
any attached buffer lowers the real-answer CE by about 0.49 nats, the gold chunk adds 0.06 on top.
The reader learned "evidence present, answer" (the none rows are trained to abstain), not the
content of the chunk. The selector did learn: per-token chunk AUROC climbed from 0.515 to 0.654 at
loop 1 and cleared the 0.59 bar at every loop by the 7.5M reading.

## Fixed-target readings (`evidence_fixed`, 46,424 answer tokens over 100 batches, final loop)

| tokens | step | CE gold | CE mixed | CE distractors | CE none | gain gold | gain distractors | content gain | grounded AUROC all / evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 4.7348 | 4.7332 | 4.7339 | 4.7329 | -0.0020 | -0.0010 | -0.0009 | 0.5000 / 0.5000 |
| 2.5M | 490 | 4.3326 | 4.3408 | 4.3496 | 4.5751 | +0.2426 | +0.2256 | +0.0170 | 0.7550 / 0.5131 |
| 5.0M | 980 | 4.2275 | 4.2409 | 4.2624 | 4.6637 | +0.4361 | +0.4012 | +0.0349 | 0.7162 / 0.5319 |
| 7.5M | 1450 | 4.1767 | 4.1933 | 4.2144 | 4.6803 | +0.5036 | +0.4659 | +0.0377 | 0.7281 / 0.5452 |
| 10.1M | 1950 | 4.1518 | 4.1767 | 4.2095 | 4.6967 | +0.5448 | +0.4872 | +0.0577 | 0.6529 / 0.5508 |

The content gain is still rising (0.017, 0.035, 0.038, 0.058) and not saturated; at this slope it
would need on the order of 100M tokens to reach the 1.6 nat bar if it stayed linear, which nothing
suggests.

## Per loop at 10.1M tokens

| loop | gain gold | gain distractors | content gain | chunk AUROC per token | mass/chunk AUROC | gold share | external mass gold / distractors |
|---|---|---|---|---|---|---|---|
| 1 | +0.4368 | +0.3913 | +0.0455 | 0.6536 | 0.5662 | 0.3419 | 0.047 / 0.084 |
| 2 | +0.5230 | +0.4682 | +0.0548 | 0.6466 | 0.5726 | 0.3417 | 0.050 / 0.088 |
| 3 | +0.5448 | +0.4872 | +0.0577 | 0.6450 | 0.5737 | 0.3414 | 0.052 / 0.089 |

Chunk AUROC by loop over the run: 0.515 / 0.484 / 0.490 (0), 0.538 / 0.522 / 0.521 (2.5M),
0.589 / 0.580 / 0.579 (5M), 0.625 / 0.614 / 0.614 (7.5M), 0.654 / 0.647 / 0.645 (10M). Loop 1 leads
at every reading. Mass/chunk AUROC stayed under the 0.674 bar (0.57 at 10M); the external mass
itself hardly moved (about 5% of the softmax on gold rows), so the selector ranks gold above
distractors inside a buffer but does not route mass to the store.

## Other readings

- Per-condition `[eval]` on `evidence_dev` (each row's own target): gold 2.8947 to 2.5234, none
  0.5603 to 0.3097, selection loss flat at 0.426 to 0.424. Grounded AUROC on evidence rows 0.50 to
  0.71 there, which is the head learning the base rate plus the "evidence present" signal.
- `|shared_evidence.o_proj|rms` 0 to 2.89e-3, monotone. With the flat content gain this is the
  reader bias the plan warned about.
- Gradient norms (pre-clip) at optimizer steps 10 / 100: `loop_query_bias` 8.6e-4 / 7.4e-4,
  `key_adapter` 1.36e-3 / 1.26e-3, `value_adapter` 0 / 0 (the value path is dead, as predicted),
  `down_proj` 3.05e-3 / 2.89e-3.
- Throughput 6.6k to 18.3k tok/s, about 11k typical; peak 26.42 GB; packing fill 61 to 67% with 84
  to 88% of the evidence cap used and about 70% of rows closed by the evidence budget.
- Warmup reached 1e-5 at step 800 (4.1M tokens); the readings at 2.5M and 5M are inside warmup.

## What it decides

The grafted reader can carry a buffer-presence signal within 10M tokens and the selector can rank
gold; the reader cannot carry the chunk's content at this budget. Per the plan, a fail under 50% of
the ceiling moves the first pilot reader arm to a reader per loop or reads in the dense prelude,
before more data. Arm B (reader without rotary, `--run-name norope`) runs next on the same seed and
decides whether the rotary offset between prompt and chunk positions is the block.

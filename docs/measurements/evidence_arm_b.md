# Evidence finetune, arm B (reader without rotary, gate frozen)

Date 2026-10-01. Same seed, HEAD and flags as arm A ([evidence_arm_a.md](evidence_arm_a.md)) plus
`--reader-no-rotary --run-name norope`. Log `ckpts/evidence_armB.log` (gitignored), checkpoints
`ckpts/evidence_norope/checkpoint_evidence_tok10M_loss3.7359.pt` (cadence save) and the kill save.
The step-0 readings reproduce arm A's to four decimals (the marker buffer changes nothing on a zero
reader).

**Verdict: killed by the automatic rule at 10.08M evidence tokens.** Gold gain +0.5711 nats,
content gain +0.0814 nats (bar 0.1 to survive, 1.6 to pass). The rotary offset between prompt and
chunk positions is not the block: arm B's content gain runs about 40% above arm A's at every
reading past warmup, which is a real but small effect, and the buffer-presence signal is identical.

## Fixed-target readings (final loop), arm B against arm A

| tokens | gain gold B / A | gain distractors B / A | content gain B / A | chunk AUROC loop 1 B / A | mass/chunk AUROC loop 1 B / A |
|---|---|---|---|---|---|
| 0 | -0.0020 / -0.0020 | -0.0010 / -0.0010 | -0.0009 / -0.0009 | 0.5147 / 0.5147 | 0.4806 / 0.4806 |
| 2.5M | +0.2825 / +0.2426 | +0.2662 / +0.2256 | +0.0163 / +0.0170 | 0.5385 / 0.5380 | 0.4918 / 0.4912 |
| 5.0M | +0.4399 / +0.4361 | +0.4043 / +0.4012 | +0.0356 / +0.0349 | 0.5871 / 0.5891 | 0.5215 / 0.5219 |
| 7.5M | +0.5265 / +0.5036 | +0.4759 / +0.4659 | +0.0506 / +0.0377 | 0.6258 / 0.6247 | 0.5480 / 0.5488 |
| 10.1M | +0.5711 / +0.5448 | +0.4897 / +0.4872 | +0.0814 / +0.0577 | 0.6501 / 0.6536 | 0.5665 / 0.5662 |

CE at 10.1M: gold 4.1518, mixed 4.1866, distractors 4.2331, none 4.7228. Grounded AUROC all rows
0.6315, evidence rows 0.5534. Per loop at 10.1M: content gain +0.0700 / +0.0781 / +0.0814, chunk
AUROC 0.6501 / 0.6431 / 0.6442, gold share 0.342.

## Reading

- The selector is unaffected by the reader's rotary (it does not use it), and the two selector
  curves are identical, as they should be.
- The content gain in B is accelerating (0.016, 0.036, 0.051, 0.081) where A's was flatter (0.017,
  0.035, 0.038, 0.058). Both are an order of magnitude under the bar at the budget the plan set, so
  neither changes the design; B is the better reader variant for any further graft arm.
- Both arms put about 0.49 nats on "a buffer is attached", condition-independent. That is the
  abstain-versus-answer prior of the training mix, not retrieval, and it is why the content gain is
  the kill number.

## What it decides

The grafted reader, with or without rotary, cannot carry chunk content at 10M tokens. Per the plan
the first pilot reader arm is a reader per loop or reads in the dense prelude; the no-rotary reader
is the default for it. Nothing here bears on where facts live.

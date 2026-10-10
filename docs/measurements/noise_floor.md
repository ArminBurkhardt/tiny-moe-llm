# Noise floor — eval sampling spread on the abstention metrics (2026-08-23)

Gates need honest thresholds more than they need ambition. From here on "must not regress" means
"within the documented noise" and "must improve" means "by ≥3σ", so the noise has to be a measured
number rather than an intuition.

This record covers the **eval sampling** half: the same checkpoint, the same flags, two disjoint
2,000-question slices of the SQuAD v2 validation split. The training seed half (re-running the
finetune under a different `sft.seed`) is not measured here and is still open.

## What was run

`scripts/eval_abstention.py` gained `--example-offset`, which skips that many *usable* questions
before collecting any. Both slices therefore come out of the same seeded shuffle of the same split
at the same flags, and the only thing that differs between them is which questions landed in them:

```bash
CKPT=ckpts/repair/checkpoint_repair_final.pt      # the 0.55 repair finetune

python scripts/eval_abstention.py -c $CKPT --max-examples 2000 --batch-size 16 --skip-forced
python scripts/eval_abstention.py -c $CKPT --max-examples 2000 --batch-size 16 --skip-forced \
  --example-offset 2000
```

The offset counts questions that survived rendering and the 1,024-token prompt cap, not raw rows —
an offset over raw rows would land somewhere else in the sequence the un-offset run collects, and
the two slices would then not be disjoint in the way the number claims.

**The standard slice reproduced [the recorded 0.55 numbers](abstention_repair.md#the-055-rebuild-2026-08-22)
exactly**, to all four printed decimals on every metric. That is worth stating separately: it makes
the spread below attributable to the questions and nothing else, and it confirms the `--example-offset`
refactor left the default path bit-for-bit unchanged.

## The two slices

| | questions 0–2,000 | questions 2,000–4,000 | spread (σ) |
|---|---|---|---|
| unanswerable / answerable | 1,013 / 987 | 999 / 1,001 | — |
| **false abstention rate (answerable)** | **0.1611** | **0.1568** | **0.0043 (0.0030)** |
| abstention precision | 0.5782 | 0.5757 | 0.0025 (0.0018) |
| abstention recall | 0.2152 | 0.2132 | 0.0020 (0.0014) |
| abstention F1 | 0.3137 | 0.3112 | 0.0025 (0.0018) |
| overall abstention rate | 0.1885 | 0.1850 | 0.0035 (0.0025) |
| exact match (answerable half) | 0.1641 | 0.1489 | 0.0152 (0.0107) |
| token F1 (answerable half) | 0.2189 | 0.2121 | 0.0068 (0.0048) |
| overall correctness | 0.1900 | 0.1810 | 0.0090 (0.0064) |
| mean `p_max` | 0.6894 | 0.6887 | 0.0007 (0.0005) |
| ECE(`p_max`) | 0.4994 | 0.5077 | 0.0083 (0.0059) |
| AUROC(`p_max` → answer correct) | 0.6595 | 0.6437 | 0.0158 (0.0112) |
| AUROC(`1 - p_max` → unanswerable) | 0.4775 | 0.5062 | 0.0287 (0.0203) |
| abstentions made / correct | 377 / 218 | 370 / 213 | — |
| distinct completions | 1,263 | 1,279 | — |
| most common completion | `"The passage doesn't say."`, 274 | `"The passage doesn't say."`, 273 | — |

"spread" is the absolute difference between the two readings; the figure in brackets is that
difference read as a standard deviation (`|a - b| / sqrt(2)`, the unbiased σ estimate from a pair).
**Two samples is a weak estimator of σ** — the 95% interval on a σ from n=2 spans roughly 0.45x to
16x the point estimate — so these are order-of-magnitude floors, and the sensible way to use them is
the operating-point column below rather than the raw number.

## What it says

**The decision metrics are tight; the answer-quality and discrimination metrics are not.**

- **False abstention, precision, recall, F1 and abstention rate all move by ≤0.004.** They are
  proportions over ~1,000 questions each, and the binomial standard error alone predicts that:
  0.578 over 377 abstentions is ±0.025 binomial, and the measured slice-to-slice spread of 0.0025
  sits an order of magnitude *inside* it, which is the expected outcome for two samples out of one
  population. The practical floor to gate against is therefore the **binomial** error on the count,
  not this spread — for precision on ~370 abstentions that is **±0.026**, and a "3σ improvement"
  means clearing 0.578 by about **0.08**.
- **EM on the answerable half moves 0.0152** — 1.5 points, the widest of the behavioural metrics.
  It is a mean over ~1,000 questions of a quantity whose per-question variance is far higher than a
  0/1 abstention decision (a near-miss span scores 0), so a 1-point EM change between two runs is
  not a result.
- **AUROC(`1 - p_max` → unanswerable) moves 0.0287, from 0.4775 to 0.5062** — across chance. That
  is the most important line here. The "below chance on three checkpoints running" reading in
  [abstention_repair.md](abstention_repair.md) survives as "not above chance", but the *sign* of
  that signal is not stable across a 2,000-question slice, so no argument should rest on it being
  specifically below 0.5. It carries no usable discrimination either way, which is the conclusion
  that was already drawn from it.
- **Mean `p_max` moves 0.0007.** Aggregate confidence is essentially a constant of the checkpoint;
  what the slices move is what that confidence is scored *against*.

## Operating points for the later gates

| quantity | floor to treat as noise | source |
|---|---|---|
| abstention precision | ±0.026 | binomial on ~370 abstentions, which dominates the 0.0018 slice spread |
| abstention recall | ±0.013 | binomial on ~1,000 unanswerable questions |
| false abstention rate | ±0.012 | binomial on ~1,000 answerable questions |
| EM (answerable half) | ±0.015 | measured slice spread, which exceeds the binomial here |
| AUROC (either) | ±0.030 | measured slice spread |
| mean `p_max` | ±0.002 | measured slice spread |

Where the binomial error and the measured spread disagree, the table takes the larger — the spread
is estimated from two samples and can only be trusted as a lower bound, while the binomial error is
exact for the count it describes.

**Still open: the training-seed σ.** Re-running the repair finetune at a different `sft.seed` on the
identical corpus and re-measuring is a ~22 minute run and has not been done. Every number above is
one checkpoint measured twice; it bounds how much the *questions* move a metric and says nothing
about how much a *rerun of the training* would.

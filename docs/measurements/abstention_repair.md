# Gate P2 — abstention repair, before and after (2026-08-20)

`scripts/eval_abstention.py` on the SQuAD v2 **validation** split, which no SFT or repair corpus has
ever consumed. Identical flags on both checkpoints, which is what makes them comparable at all
(the MoE's grouped GEMM tiles by the batch's per-expert row counts, so `--batch-size` is part of the
measurement — see [runbook.md](../runbook.md) §10.5):

```
--max-examples 2000 --batch-size 16 --skip-forced   # greedy, <= 32 new tokens
```

2,000 questions: 1,013 unanswerable, 987 answerable. Per-question records were written next to this
file as `abstention_before_repair.json` / `abstention_after_repair.json` — untracked, since
`.gitignore` swallows `*.json`, so regenerate them with the command above if you need them.

| | before (`checkpoint_sft_final_phase0`) | after (`checkpoint_repair_final`) |
|---|---|---|
| **false abstention rate (answerable)** | **0.7832** | **0.1358** |
| abstention precision (base rate 0.5065) | 0.5154 ± 0.0125 | 0.5759 ± 0.0278 |
| abstention recall | 0.8115 | 0.1797 |
| abstention F1 | 0.6304 | 0.2739 |
| overall abstention rate | 0.7975 | 0.1580 |
| exact match (answerable half) | 0.0648 | 0.1611 |
| token F1 (answerable half) | 0.0806 | 0.2162 |
| overall correctness | 0.4430 | 0.1705 |
| mean `p_max` | 0.7941 | 0.6846 |
| ECE(`p_max`) | 0.3511 | 0.5141 |
| AUROC(`p_max` → answer correct) | 0.6187 | 0.6602 |
| AUROC(`1 - p_max` → unanswerable) | 0.4618 | 0.4711 |
| most common completion | `"The passage doesn't say."`, 1,398 / 2,000 | `"The passage doesn't say."`, 259 / 2,000 |
| distinct completions | 320 | 1,293 |

The ± on precision is the binomial standard error over the abstentions actually made (1,595 before,
316 after), stated because the "clearly above the base rate" half of Gate P2 rests on a 316-sample
proportion: 0.5759 clears the 0.5065 base rate by ~2.5 standard errors.

## What moved

**The collapse is gone.** One literal string was 70% of all completions before and is 13% now;
distinct completions went 320 → 1,293. False abstention on the answerable half fell 5.8x, and the
answerable half's EM and token-F1 both roughly tripled off it — the model is reading the passage
instead of refusing it.

**Recall collapsed the other way**, 0.81 → 0.18, and that is the honest cost. The model now
under-abstains: it answers 84% of unanswerable questions instead of refusing them. NEXT.md assigns
that half to Phase 4's no-evidence condition ("grounded in retrieval, not memorized as a string"),
and this is the first measurement that separates the two — but it is a real regression on the
unanswerable half, not a rounding error.

**Overall correctness fell 0.443 → 0.171, and it should be read carefully.** On a validation split
that is 50.65% unanswerable, "refuse everything" scores 0.5065 on this metric. The *pre-repair*
checkpoint scored 0.443 — i.e. below the trivial refusal baseline, which is what a near-degenerate
policy looks like when the metric rewards refusal. The repaired checkpoint is further below it,
because the metric's unanswerable half is exactly what recall lost. The metric is dominated by the
abstention decision on this split; the answerable-half EM/F1 rows above are the ones that describe
answer quality.

**`p_max` still says nothing about answerability.** AUROC of `1 - p_max` for detecting an
unanswerable question is 0.471 after and 0.462 before — below chance both times. Confidence
predicts whether *this answer* is right (AUROC 0.660, up from 0.619) and not whether the question
was answerable. That is the same conclusion Phase 0 reached about `p_correct` and it survives the
repair unchanged.

ECE(`p_max`) degrading 0.351 → 0.514 is mostly the recall collapse re-entering through the accuracy
term: mean confidence fell (0.794 → 0.685) while accuracy fell further, because "correct" on the
unanswerable half means abstaining. AUROC — which is calibration-free — improved.

## The corpus behind it

`scripts/prepare_sft_data.py --profile repair`, 49.5M tokens, 146,289 conversations, 40.3%
supervised. Realized per-source conversation counts (the unit that matters, since the run weights
its loss per conversation):

| source | tokens | conversations | share | abstentions |
|---|---|---|---|---|
| `squad_v2` (unanswerable rows kept: 40%) | 12.5M | 61,038 | 41.7% | 10,139 |
| `squad` (SQuAD 1.1) | 8.5M | 41,603 | 28.4% | 0 |
| `hotpot_qa` (distractor) | 4.0M | 2,970 | 2.0% | 0 |
| `smoltalk2` | 13.9M | 21,703 | 14.8% | 0 |
| `ultrachat_200k` | 7.0M | 6,014 | 4.1% | 1 |
| `no_robots` | 4.0M | 14,411 | 9.9% | 4 |

**Unanswerable share of QA conversations: 9.6%** (10,139 / 105,611), against NEXT.md's 10–15%
target — slightly under the band, and the most likely lever on the recall collapse. Raising
`--squad-unanswerable-fraction` from 0.40 toward 0.55–0.60 is the direct retune.

QA / general chat splits 72 / 28 by conversation against 50 / 50 by token, which is the whole reason
the mix weights were retuned after a 2M-token trial: a SQuAD row is ~206 tokens and a HotpotQA row
~1,340, so token-budget shares are not gradient shares once the loss is per conversation.

## The run

`python scripts/sft.py --repair -c ckpts/trained/checkpoint_sft_final_phase0.pt`, locally on the
5090 in BF16 (no `USE_FP8`). 49.34M tokens, 1 epoch, ~830 optimizer steps at 4 × 4096 × 4, 31.5
minutes at 22–37k tokens/sec, 21.36 GB peak.

`repair_val` CE fell 2.088 → 1.910 monotonically across all nine eval points, with `p_max` (0.567 →
0.566) and top-1 (0.543 → 0.543) flat — the trunk did not move, the abstention decision did.

`loop_scale` drifted `[0.5000, 0.2412, 0.1152]` → `[0.5039, 0.2383, 0.1123]`, i.e. the fp32 masters
are moving it as intended and it is not being decayed toward zero.

**One measurement paid for itself before training started:** at `sft:`'s batch size of 8 the run
peaks at 29.55 GB allocated, which puts total driver usage at ~32.0 of 32.6 GB and silently spills
into shared system memory — 9.4k tokens/sec. At 4 × 4096 with accumulation 4 (identical tokens per
optimizer step) it is 21.36 GB, fully resident, and 22–37k tokens/sec. Same objective, ~3x the
throughput.

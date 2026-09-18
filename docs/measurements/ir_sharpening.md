# IR sharpening finetune — arm A vs. arm B (2026-08-28)

The companion to [ir_reshape.md](ir_reshape.md). That file measured what the `65536 x 256` reshape
costs and confirmed the migrated checkpoint is behaviourally its own read-zeroed ablation. This one
records what happened when the reshaped table was actually trained: two ~208M-token finetunes off
the same seed, differing only in how `z_keys` was initialized, with the retrieval temperature
annealed 1.0 → 0.05 across the run.

**Gate G2 asks four things**: post-anneal retrieval entropy well below uniform; held-out CE and the
benchmark suite within noise of the seed; the read-zeroed ablation well off the 0.0004-nat floor;
and a winning arm chosen on those numbers rather than on the prior.

> A defect found after arm A finished invalidated its first two measurements and is worth recording
> because it is invisible from a training log. `temperature_scale`, the externally driven anneal
> multiplier, was a plain Python float, so it never entered the state dict — every saved checkpoint
> reloaded at scale 1.0 and read ~20x flatter than it trained. The training log said entropy 0.9065
> of `ln 32`; the eval said 0.9998. It is a `register_buffer` now, and arm A's checkpoint was
> patched to the 0.05 it finished at. Every number below is post-fix.

## 1. The corpus

`scripts/prepare_data.py --phases ir --manifest-key ir_prep` built **210,000,429 tokens over
146,131 documents in 796s**, split at document 143,676 into 208,001,381 train / 1,999,048 val.
Realized against target per source: fineweb 26.05M, finepdfs 17.29M, code_edu 37.42M,
nemotron_math 52.00M, wikipedia 13.92M, smoltalk2 63.31M — between +0.5% and −1.6%. The
`--manifest-key` is what kept the 16B run's `smoltalk2_holdout_hashes` intact; the build writes its
own key rather than overwriting the one SFT's exclusion reads.

## 2. Arm A (random keys)

208M tokens in ~70 minutes at ~48k tokens/sec, 22.54 GB peak, micro batch `4 x 4096` in BF16.

**The finetune's own objective improved.** `ir_val` CE went 2.9226 → a 2.9378 peak at 25M tokens →
**2.9109** at the end (−0.0117 nats from the first eval, −0.0269 from the peak). The peak is the
fresh tensors moving at `fresh_lr` before the trunk absorbs them; the recovery is monotone after
50M tokens.

**The table sharpened, but only on the first loop.** Training-time `E / ln 32`, and the same
quantity read back by `scripts/eval_stage0.py`:

| | loop 1 | loop 2 | loop 3 |
|---|---|---|---|
| end of training (`IR E/ln32`) | 0.9054 | 0.9821 | 0.9823 |
| `eval_stage0.py`, frac of max | 0.9109 | 0.9862 | 0.9866 |
| max weight (uniform = 0.03125) | 0.1451 | 0.0532 | 0.0531 |
| read dispersion | 1.1182 | 0.4346 | 0.3932 |

Loops 2 and 3 read essentially uniformly over their 32 candidates — the table stores nothing they
use. This is the same split `eval_stage0.py`'s query drift matrix shows: `cos(q1, q2) = 0.3710`
while `cos(q2, q3) = 0.9803`, so loop 1 asks a different question and loops 2–3 ask each other's.

**The cluster structure held.** Ten refreshes at 20M-token intervals:

| refresh | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| temperature scale | 0.661 | 0.437 | 0.289 | 0.191 | 0.126 | 0.083 | 0.055 | 0.050 | 0.050 | 0.050 |
| candidate recall@32 | 0.995 | 0.999 | 0.995 | 0.955 | 0.988 | 0.977 | 0.973 | 0.985 | 0.936 | **0.917** |
| dead fraction | 0.006 | 0.005 | 0.012 | 0.028 | 0.031 | 0.054 | 0.060 | 0.071 | 0.062 | **0.086** |
| recycled | 393 | 302 | 797 | 1310 | 1310 | 1310 | 1310 | 1310 | 1310 | 1310 |

Recall stayed above the 0.9 bar throughout, which is what the 4 → 8 probe-cluster decision bought:
the shakedown at 4 probes read 0.86–0.95 on the same schedule. The dead fraction climbs to 8.6% and
the recycler is capped at 1310 (the 2% `dead_quantile`) from the fourth refresh on, so the table is
shedding entries faster than the cap replaces them — not fatal over 208M tokens, but it would be
the first thing to raise on a longer run.

**Gate G2 fails on substance.** `scripts/eval_calibration.py` and `scripts/eval_stage0.py`, same
held-out slice and identical flags as the seed's (`--start-doc-idx 0 --max-batches 40
--batch-size 4`):

| | seed (`..._phase0`) | arm A | Δ |
|---|---|---|---|
| per-loop CE | 3.5152 / 3.3922 / **3.3843** | 3.5515 / 3.4234 / **3.4109** | **+0.0266** |
| top-1 accuracy | 0.3963 | 0.3950 | −0.0013 |
| ECE(`p_max`) | 0.0089 | **0.0341** | +0.0252 |
| AUROC(`p_max` → correct) | 0.8376 | 0.8358 | −0.0018 |
| dCE, read zeroed | 0.0002–0.0004 | **0.0001** | — |
| dCE, read set to its mean | — | −0.0001 | — |
| IR routed weight (per loop) | — | 0.0199 / 0.0537 / 0.0413 | — |

The read's per-token *content* — `dCE(zero) − dCE(mean)` — is worth **0.0002 nats**, which is the
floor the whole phase set out to clear, and the same 0.0002 the untrained migrated seed measured.
The table learned to discriminate among its candidates on loop 1 and the model still does not use
what it retrieves. Held-out CE is 0.027 nats *worse* than the seed, consistent with a finetune on a
different mixture rather than with damage.

The ECE regression (0.0089 → 0.0341) is the one number here that is not explainable as mixture
shift, and it is not caused by the retrieval: the per-loop CE is identical to four decimals whether
the checkpoint is read at temperature scale 1.0 or the trained 0.05, which is only possible if the
read is contributing nothing to the logits either way. `p_max` moved because the trunk moved.

## 3. Arm B (bge-small warm start)

208M tokens in 71.8 minutes at ~52k tokens/sec, 22.54 GB peak, identical settings. `z_keys` was
initialized from bge-small embeddings of 65,536 corpus chunks, PCA'd 384 → 256; everything else,
including `y_values`' zero-init, matches arm A.

**It converged to the same place, to four decimals.** The two runs are not merely close, they are
indistinguishable on every number the gate reads:

| | arm A (random keys) | arm B (warm keys) |
|---|---|---|
| `ir_val` CE, final | 2.9109 | **2.9101** |
| per-loop CE (held out) | 3.5515 / 3.4234 / **3.4109** | 3.5516 / 3.4235 / **3.4109** |
| top-1 accuracy | 0.3950 | 0.3949 |
| ECE(`p_max`) | 0.0341 | 0.0346 |
| AUROC(`p_max` → correct) | 0.8358 | 0.8360 |
| entropy `E / ln 32`, loops 1–3 | 0.9109 / 0.9862 / 0.9866 | 0.9232 / 0.9839 / 0.9848 |
| max weight, loop 1 | 0.1451 | 0.1385 |
| dCE, read zeroed (final loop) | 0.0001 | 0.0002 |
| dCE, read set to its mean | −0.0001 | 0.0000 |
| IR routed weight (per loop) | 0.0199 / 0.0537 / 0.0413 | 0.0199 / 0.0540 / 0.0405 |
| `cos(q1, q2)` | 0.3710 | 0.3176 |

Arm B's read content is the same **0.0002 nats**. Its loop-1 read is a hair flatter than arm A's
(0.9232 vs 0.9109) and its loop-1 dispersion lower (0.9258 vs 1.1182), so if anything the warm start
sharpened *less*, but the difference is well inside what two seeds of the same profile disagree by.

**The clustering fit the warm keys worse.** Arm B's ten refreshes:

| refresh | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| temperature scale | 0.661 | 0.437 | 0.289 | 0.191 | 0.126 | 0.083 | 0.055 | 0.050 | 0.050 | 0.050 |
| candidate recall@32 | 0.998 | 1.000 | 0.996 | **0.836** | 0.967 | 0.955 | 0.939 | 0.944 | **0.841** | **0.876** |
| dead fraction | 0.006 | 0.007 | 0.027 | 0.038 | 0.061 | 0.047 | 0.054 | 0.039 | 0.025 | 0.030 |
| recycled | 421 | 453 | 1310 | 1310 | 1310 | 1310 | 1310 | 1310 | 1310 | 1310 |

Three readings below the 0.9 bar against arm A's zero, and it ends there (0.876 vs. arm A's 0.917).
That is the one place the initialization is visibly doing something: bge embeddings arrive with
their own cluster structure, and 256 balanced spherical clusters over 65,536 of them is a worse fit
than over isotropic noise, which has no structure to fight. The dead fraction is the mirror image —
arm B's peaks at 6.1% and comes back down to 3.0%, where arm A's climbs monotonically to 8.6%. It
does not change the conclusion: the read is worth 0.0002 nats either way, so a candidate set that
misses a sixth of the exact top-32 is missing entries the model was not using.

## 4. The benchmark suite

Both arms against the seed's column in [benchmark_snapshot.md](benchmark_snapshot.md), same frozen
flags (`--batch-size 32 --max-context 1024 --gen-limit 1000 --seed 1234`):

| task | metric | seed | arm A | arm B | Δ A | Δ B |
|---|---|---|---|---|---|---|
| hellaswag | acc_norm | 0.2826 | 0.2761 | 0.2765 | −0.0065 | −0.0061 |
| arc_easy | acc_norm | 0.3809 | 0.3826 | 0.3864 | +0.0017 | +0.0055 |
| arc_challenge | acc_norm | 0.2261 | 0.2287 | 0.2312 | +0.0026 | +0.0051 |
| piqa | acc_norm | 0.5789 | 0.5816 | 0.5838 | +0.0027 | +0.0049 |
| winogrande | acc | 0.5185 | 0.5028 | 0.5162 | −0.0158 | −0.0024 |
| openbookqa | acc_norm | 0.2720 | 0.2760 | 0.2760 | +0.0040 | +0.0040 |
| sciq | acc | 0.6050 | 0.5930 | 0.5920 | −0.0120 | −0.0130 |
| boolq | acc | 0.4599 | 0.4141 | 0.4098 | **−0.0459** | **−0.0502** |
| lambada_openai | acc | 0.1644 | 0.1828 | 0.1816 | +0.0184 | +0.0173 |
| mmlu | acc | 0.2430 | 0.2334 | 0.2344 | −0.0095 | −0.0086 |
| triviaqa | EM | 0.0000 | 0.0000 | 0.0000 | — | — |
| nq_open | EM | 0.0000 | 0.0000 | 0.0000 | — | — |
| gsm8k | EM | 0.0180 | 0.0190 | 0.0160 | +0.0010 | −0.0020 |
| **mean MC headroom** | | **+0.088** | **+0.072** | **+0.076** | −0.016 | −0.012 |

Twelve of thirteen tasks move by less than two points, in both directions, which is the "within
noise of the seed" half of Gate G2 passing. BoolQ is the exception at −4.6 / −5.0, and it is the
same task that fell 4.1 points across the SFT and repair finetunes for reasons that had nothing to
do with retrieval either — a near-chance binary task on a model this size is the suite's noisiest
column. The 0.012–0.016 drop in mean headroom is comparable to the 0.007 spread the three snapshot
checkpoints already show across 758M tokens of finetuning. Nothing here is retrieval; it is what a
208M-token finetune on a different mixture costs.

## 5. Reading

**Gate G2 fails, on both arms, on its central question.** Its first and second conditions pass —
the table sharpens, and held-out CE and the benchmark suite stay within a finetune's worth of the
seed. The third does not. Loop 1 reaches `E / ln 32 ≈ 0.91` against a uniform 1.0, with a max weight
4.5x uniform, and the model still does not use what it retrieves. Zeroing the read costs 0.0002 nats on the final loop against a 0.02 bar,
the same 0.0002 the *untrained* migrated seed measured in
[ir_reshape.md](ir_reshape.md). 208M tokens of training at `fresh_lr` moved the read's contribution
by nothing measurable.

Three things this pins down that the plan left open:

1. **The initialization is not the lever.** The A/B was run rather than guessed precisely because
   the prior favoured the warm start, and the two arms land on the same CE to four decimals, the
   same 0.0002-nat content, and the same routed weight per loop. Neither arm "wins"; the honest
   read is that the choice does not matter at this scale, and a third init would not either.
2. **Only loop 1 learned to discriminate.** Loops 2–3 read at 0.984–0.987 of uniform over their 32
   candidates on both arms. That is the same split the query drift matrix shows —
   `cos(q1, q2) ≈ 0.32–0.37` while `cos(q2, q3) ≈ 0.98` — so the recurrence asks one question at
   the first loop and then repeats itself, and the table can only be sharp for a query that varies.
3. **The bottleneck is downstream of retrieval.** The read is sharp, the routed weight is nonzero
   (2–5%), and the CE is unmoved. Per-loop CE is identical whether the checkpoint reads at
   temperature scale 1.0 or the trained 0.05, which is only possible if the retrieved value reaches
   the logits with ~zero weight. `y_values` starts at zero for neutrality and, on this evidence, the
   gradient never gave it a reason to leave — `g_proj` and the residual's own scale are where the
   signal is being discarded, not the table.

Held-out CE is 0.027 nats worse than the seed on both arms, and the ECE regression (0.0089 → 0.0341
/ 0.0346) tracks it. Neither is retrieval damage: the finetune trained on a different mixture than
the one the held-out slice is drawn from, the read demonstrably contributes nothing to the logits,
and the two arms — which differ only in the table — regress identically.

# Loss-free MoE balancing at micro scale (2026-10-19): the balancing rung's arm

Loss-free balancing in place of the aux loss on the arm (a) recipe, against arm (a) itself.
Held-out CE ends 0.0018 above the twin (paired 0.0055 at exit 3, z -1.8), inside the seed
spread at every save. Storage is unchanged cell for cell, within the two AdamW seeds. The MLP
load is even by the arm's own field (max over mean 1.03 to 1.05 per pass on average). The
comparison of evenness against arm (a) is not readable from the logs: the logged aux value
saturates in bf16 and reads the batch's token count, not the load (the load section). An arm
(a) rerun under the fixed aux term (fp32 counts, 2026-10-20) reads the other side: the working
aux term ends 0.0105 above arm (a) (paired -0.0076 at exit 3, z -2.3), puts 0.35 to 0.54 of the
picks on the dense slots with the MLP load at 1.08 to 1.26, and stores the biographies more in
distribution (tier 100 z 14.5 against arm (a), far outside the seed spread). The rung passes for
loss-free on all three bars.

## The build

In [NEXT.md](../plans/NEXT.md) (the R8 ladder row and the 2026-10-19 changelog), after arXiv
2408.15664. `training: moe_balancing: aux | loss_free` with `moe_balance_bias_rate` (0.001
here). Under `loss_free` the loss has no aux term. A persistent buffer `moe.balance_bias`
`[n_loops, num_experts]` is added to the routing probabilities for the top-k selection only;
the weights are gathered from the unbiased probabilities, and a zero bias is exactly the old
router. After each optimizer step `LoopMixtureOfExperts.update_balance_bias` moves the bias on
an fp32 shadow by the rate times the sign of each MLP slot's load error against the pass's mean
MLP load, with the signs centred over the MLP slots (zero sum), so the update moves load between
MLP slots and leaves the MLP against dense split to the router. Counts are fp32, accumulated on
device across micro steps; a pass a sampled depth skipped is left alone. The step line carries
`load max/mean` (the most loaded MLP expert's share over the mean MLP share, per pass, from an
EMA) and `dense share` (the fraction of top-k picks on the attention and IR slots, per pass).

The first launch used the raw signs. A detached review the same day found that under a skewed
load more MLP slots sit under the mean than over it, so the raw signs push every MLP bias up
together against the dense slots, whose bias is pinned at zero. That launch was stopped at 3M
and the arm below is the clean relaunch on the centred update.

## The arm and its twin

`ckpts/inject/launch_lossfree.sh` (`config_micro_lossfree.yaml`: `moe_balancing: loss_free`,
`moe_balance_bias_rate: 0.001`, every other number as arm (a)), from `seed_micro.pt` on
`inject_full_train`, full CE, no evidence, the same data order and seed as the twin. Run
directory `ckpts/evidence_lossfree_full/`, log `ckpts/inject/lossfree_full.log`, reads
`ckpts/inject/lossfree_full_{50M,100M,final}_rank_none.*` and
`lossfree_full_{TAG}_exit.*` at every save (paired per token against the twin's save at the
same tokens, `eval_exit.py --pair`). 299.44M tokens, one epoch, in 80.8 minutes (arm (a) 89.9).

The twin is arm (a) (`ckpts/evidence_inject_full/`, aux loss weight 0.01, `sft.seed` 1234,
log `ckpts/inject/inject_full.log`). The spread is the AdamW reseed of the Muon rung
(`ckpts/evidence_adamw_full_s4321/`, log `ckpts/inject/adamw_full_s4321.log`,
[muon_micro.md](muon_micro.md)).

The bar, set before the run: held-out CE not worse than the twin's beyond the paired sigma and
the seed spread (0.03 at the end); load at least as even as under the aux loss.

## Held-out CE

`inject_val` CE by the trainer's eval (every 1,000 steps, 25M tokens each; final step 11918 for
loss-free and arm (a), 11925 for seed 4321):

| step | loss-free | arm (a), seed 1234 | seed 4321 | loss-free minus arm (a) |
|---|---|---|---|---|
| 1k | 5.6715 | 5.6683 | 5.7073 | +0.0032 |
| 2k | 5.1399 | 5.1370 | 5.1788 | +0.0029 |
| 3k | 4.7485 | 4.7441 | 4.8000 | +0.0044 |
| 4k | 4.4886 | 4.4973 | 4.5314 | -0.0087 |
| 5k | 4.3234 | 4.3352 | 4.3716 | -0.0118 |
| 6k | 4.2182 | 4.2306 | 4.2460 | -0.0124 |
| 7k | 4.1306 | 4.1395 | 4.1511 | -0.0089 |
| 8k | 4.0585 | 4.0674 | 4.0824 | -0.0089 |
| 9k | 4.0005 | 4.0015 | 4.0158 | -0.0010 |
| 10k | 3.9558 | 3.9543 | 3.9782 | +0.0015 |
| 11k | 3.9247 | 3.9250 | 3.9486 | -0.0003 |
| final | 3.9055 | 3.9037 | 3.9324 | +0.0018 |

The gap to arm (a) stays within 0.013 nats either way, lower mid-run, level at the end; seed
4321 sits 0.012 to 0.056 above arm (a) throughout (0.029 at the end), and loss-free is 0.027
below it at the end.

## Paired exits per save

Twin CE minus loss-free CE per token on 148,450 supervised tokens of `inject_val` (162
clusters); positive means the twin is worse. The last column is the reseed's spread at exit 3 on
the same read ([muon_micro.md](muon_micro.md)).

| save | exit 3 diff | se | z | seed spread at exit 3 |
|---|---|---|---|---|
| 50M | -0.0097 | 0.0044 | -2.2 | 0.049 |
| 100M | +0.0075 | 0.0043 | +1.7 | 0.029 |
| 150M | +0.0148 | 0.0050 | +3.0 | 0.005 |
| 200M | +0.0031 | 0.0035 | +0.9 | 0.020 |
| 250M | -0.0021 | 0.0031 | -0.7 | 0.030 |
| final | -0.0055 | 0.0031 | -1.8 | 0.032 |

At the final, per exit: exit 1 4.3938 against 4.3874 (-0.0064, se 0.0032, z -2.0), exit 2
4.2971 against 4.2942 (-0.0030, se 0.0031, z -1.0), exit 3 4.2940 against 4.2886 (-0.0055, se
0.0031, z -1.8). Depth pays the same way: loss-free 1 to 2 +0.0966 (z 42), 2 to 3 +0.0031
(z 9.8); the twin +0.0932, +0.0056. `loop_scale` at the end [1.578, 0.613, 0.148] against
[1.625, 0.637, 0.211]. The sign of the paired difference changes across saves and its size
stays under the seed spread except at 150M (where loss-free is the better one, z 3.0).

## Storage

Closed-book entity `delta` (`closed_book_rank.py bios --evidence none`) by tier 10 / 100 /
1000 (tier 1 within noise everywhere: loss-free 0.004 to 0.005 at the final, z under 1.5).
Arm (a) and seed 4321 from [muon_micro.md](muon_micro.md) (the reseed's tier 10 was read at the
final only).

| save | form | loss-free | arm (a), seed 1234 | seed 4321 |
|---|---|---|---|---|
| 50M | in distribution | 0.003 / 0.009 / 0.083 | 0.003 / 0.010 / 0.087 | - / 0.011 / 0.094 |
| 50M | held out | 0.003 / 0.006 / 0.045 | 0.005 / 0.003 / 0.053 | - / 0.002 / 0.027 |
| 100M | in distribution | 0.007 / 0.076 / 0.325 | 0.012 / 0.099 / 0.332 | - / 0.091 / 0.332 |
| 100M | held out | 0.004 / 0.017 / 0.112 | 0.003 / 0.018 / 0.099 | - / 0.025 / 0.113 |
| final | in distribution | 0.072 / 0.365 / 0.474 | 0.063 / 0.378 / 0.467 | 0.080 / 0.382 / 0.492 |
| final | held out | 0.004 / 0.020 / 0.094 | 0.009 / 0.023 / 0.104 | 0.001 / 0.031 / 0.075 |

At the final, sigma on `delta` 0.008 at tier 10 and 100, 0.017 at tier 1000 in distribution,
0.004 and 0.009 held out. Every cell sits inside or at the edge of the two seeds' range; the one
cell outside it is tier 100 in distribution at 100M (0.076 against 0.091 and 0.099), slower by
0.015, and the final tier 100 cells sit 0.003 to 0.017 under both seeds. Gold top-1 at tier 100
in distribution at the final 0.331 (arm (a) 0.415). Balancing does not move storage.

## Load

The arm's own field, `load max/mean` per pass (1.0 is even) and `dense share` per pass, the step
line nearest each thousand:

| step | load max/mean | dense share |
|---|---|---|
| 10 | 1.04 / 1.11 / 1.24 | 0.181 / 0.167 / 0.148 |
| 1000 | 1.02 / 1.07 / 1.07 | 0.052 / 0.095 / 0.114 |
| 2000 | 1.02 / 1.05 / 1.05 | 0.065 / 0.095 / 0.099 |
| 3000 | 1.03 / 1.03 / 1.05 | 0.066 / 0.085 / 0.087 |
| 4000 | 1.04 / 1.05 / 1.03 | 0.073 / 0.086 / 0.093 |
| 5000 | 1.02 / 1.04 / 1.03 | 0.075 / 0.091 / 0.093 |
| 6000 | 1.03 / 1.02 / 1.05 | 0.073 / 0.087 / 0.092 |
| 7000 | 1.02 / 1.04 / 1.03 | 0.071 / 0.088 / 0.096 |
| 8000 | 1.03 / 1.04 / 1.05 | 0.075 / 0.091 / 0.101 |
| 9000 | 1.03 / 1.04 / 1.04 | 0.073 / 0.094 / 0.099 |
| 10000 | 1.02 / 1.03 / 1.03 | 0.075 / 0.092 / 0.098 |
| 11000 | 1.02 / 1.02 / 1.03 | 0.074 / 0.090 / 0.096 |
| 11910 | 1.04 / 1.03 / 1.03 | 0.077 / 0.093 / 0.100 |

From step 1000 on, over every step line: load max/mean 1.01 to 1.09 in pass 1 (mean 1.026),
1.01 to 1.14 in passes 2 and 3 (mean 1.046 each); dense share 0.047 to 0.084, 0.078 to 0.104,
0.080 to 0.114 by pass. The MLP slots are within 5% of even on average; the router gives the
three dense slots 7 to 10% of the picks, about a third of an even 3 in 11, and the centred
update leaves that split alone as designed.

The `Aux:` field (the router's balancing value, computed on all three runs, trained on only by
the two AdamW arms), mean over the 100 step lines around each thousand:

| window | loss-free | arm (a), seed 1234 | seed 4321 |
|---|---|---|---|
| 1k | 0.0540 | 0.0548 | 0.0549 |
| 2k | 0.0545 | 0.0556 | 0.0558 |
| 3k | 0.0548 | 0.0558 | 0.0560 |
| 4k | 0.0547 | 0.0555 | 0.0555 |
| 5k | 0.0547 | 0.0556 | 0.0556 |
| 6k | 0.0553 | 0.0561 | 0.0561 |
| 7k | 0.0551 | 0.0558 | 0.0557 |
| 8k | 0.0551 | 0.0558 | 0.0561 |
| 9k | 0.0553 | 0.0559 | 0.0559 |
| 10k | 0.0554 | 0.0561 | 0.0560 |
| 11k | 0.0553 | 0.0558 | 0.0558 |
| whole run | 0.0549 | 0.0556 | 0.0557 |

At the last step line (11910): 0.0566, 0.0574, 0.0593 (seed 4321's 11920: 0.0520). By the
letter loss-free sits at or below arm (a) in every window and on 1162 of 1192 matched step
lines. The field does not read load, for three reasons in the logs and the code:

- The value is impossible for an intact balancing loss. `compute_aux_loss` returns
  `num_experts * dot(f, P)` with `f` the hard pick fractions and `P` the mean router
  probabilities, which is about 1 or more whenever the picks follow the probabilities; all three
  runs log 0.05 to 0.06.
- It follows the batch, not the router. Loss-free and arm (a) see the same batches in the same
  order and their per-step values correlate at 0.990 (ratio 0.987, sd 0.007), although the two
  routers are balanced by different mechanisms; seed 4321 sees another order and correlates
  with arm (a) at 0.057.
- The mechanism the arithmetic points to: with the token mask (both trainers pass it) the hard
  counts are accumulated by `scatter_add_` in the router probabilities' dtype, bf16 here, and a
  bf16 count stops growing at 256. With every one of the 11 slots past 256 picks, the value is
  11 x 256 over the batch's pick count, 11 x 256 / (2 x 25.1k) = 0.056 at this recipe's 25.1k
  real tokens per step, which is what all three logs show. Loss-free's 1.3% lower value then
  means some slot falls under 256 picks more often, most likely a dense slot, the opposite of
  more even.

So arm (a)'s evenness is unmeasured (its log has no load field), and its aux term, with every
saturated slot carrying the same `f`, has a gradient only toward a slot under 256 picks per
batch and pass: a floor against collapse, not a balancer. The rung compared loss-free
balancing against that floor. The fix landed 2026-10-20: the masked branch of
`compute_aux_loss` now counts, masks and reduces in fp32 (`test_evidence_selector.py`
`test_aux_loss_mask_bf16_counts`); the arm (a) rerun under the fixed term is the next section.

## The aux side under the fixed term (2026-10-20)

The rerun reads the half of the load bar the section above left open, and gives the first arm
(a) whose aux term balances anything. Every earlier aux run counted picks in bf16, so its term
was the floor described above. The fix counts, masks and reduces in fp32, and the step line now
carries `load max/mean` and `dense share` in every balancing mode, not only under `loss_free`.
With it the logged `Aux:` reads 1.07 to 1.22 from step 1000 (mean 1.131 over the step lines)
instead of the constant 0.056.

`ckpts/inject/launch_auxfix.sh` (`config_micro.yaml`: `moe_balancing: aux`, `aux_loss_weight`
0.01, every number as arm (a), the same seed and data order), from `seed_micro.pt` on
`inject_full_train`. Run directory `ckpts/evidence_auxfix_full/`, log
`ckpts/inject/auxfix_full.log`, reads `ckpts/inject/auxfix_full_{50M,100M,final}_rank_none.*`
and `auxfix_full_{TAG}_exit.*` at every save (paired against arm (a)'s save at the same tokens).
299.44M tokens, one epoch, in 79.7 minutes. A first launch before load tracking covered every
mode was stopped at step 1080 (27.1M tokens; `ckpts/evidence_auxfix_full_stopped/`, log
`ckpts/inject/auxfix_full_stopped.log`, no save) and is not read. The only difference from arm
(a) is the working aux term, so the comparison reads that term alone.

### Held-out CE

`inject_val` CE by the trainer's eval (final step 11918; 11925 for seed 4321):

| step | fixed aux | arm (a), seed 1234 | loss-free | seed 4321 | fixed aux minus arm (a) |
|---|---|---|---|---|---|
| 1k | 5.6558 | 5.6683 | 5.6715 | 5.7073 | -0.0125 |
| 2k | 5.1564 | 5.1370 | 5.1399 | 5.1788 | +0.0194 |
| 3k | 4.7674 | 4.7441 | 4.7485 | 4.8000 | +0.0233 |
| 4k | 4.5144 | 4.4973 | 4.4886 | 4.5314 | +0.0171 |
| 5k | 4.3469 | 4.3352 | 4.3234 | 4.3716 | +0.0117 |
| 6k | 4.2419 | 4.2306 | 4.2182 | 4.2460 | +0.0113 |
| 7k | 4.1480 | 4.1395 | 4.1306 | 4.1511 | +0.0085 |
| 8k | 4.0702 | 4.0674 | 4.0585 | 4.0824 | +0.0028 |
| 9k | 4.0127 | 4.0015 | 4.0005 | 4.0158 | +0.0112 |
| 10k | 3.9597 | 3.9543 | 3.9558 | 3.9782 | +0.0054 |
| 11k | 3.9325 | 3.9250 | 3.9247 | 3.9486 | +0.0075 |
| final | 3.9142 | 3.9037 | 3.9055 | 3.9324 | +0.0105 |

From 2k on the fixed aux arm sits 0.003 to 0.023 above arm (a) and 0.004 to 0.026 above
loss-free, and below seed 4321 at every eval (0.018 below at the end).

### Paired exits per save

Arm (a) CE minus fixed aux CE per token on the same 148,450 tokens; positive means arm (a) is
worse. Loss-free's exit 3 on the same read beside it, and the reseed's spread at exit 3.

| save | fixed aux exit 3 diff | se | z | loss-free exit 3 diff (z) | seed spread at exit 3 |
|---|---|---|---|---|---|
| 50M | -0.0201 | 0.0058 | -3.5 | -0.0097 (-2.2) | 0.049 |
| 100M | -0.0165 | 0.0044 | -3.8 | +0.0075 (+1.7) | 0.029 |
| 150M | -0.0107 | 0.0039 | -2.7 | +0.0148 (+3.0) | 0.005 |
| 200M | +0.0027 | 0.0037 | +0.7 | +0.0031 (+0.9) | 0.020 |
| 250M | -0.0036 | 0.0032 | -1.1 | -0.0021 (-0.7) | 0.030 |
| final | -0.0076 | 0.0033 | -2.3 | -0.0055 (-1.8) | 0.032 |

At the final, per exit: exit 1 4.4057 against 4.3874 (-0.0183, se 0.0033, z -5.5), exit 2
4.3021 against 4.2942 (-0.0079, z -2.4), exit 3 4.2962 against 4.2886 (-0.0076, z -2.3);
loss-free on the same read -0.0064 (z -2.0), -0.0030 (z -1.0), -0.0055 (z -1.8). Depth pays
1 to 2 +0.1036 (z 46.4), 2 to 3 +0.0059 (z 13.5). `loop_scale` at the end [1.4766, 0.7148,
0.2217] (arm (a) [1.625, 0.637, 0.211], loss-free [1.578, 0.613, 0.148]). The fixed aux arm is
worse than arm (a) at five of six saves, beyond 3 sigma at 50M and 100M (inside the seed spread
there) and outside the seed spread at 150M; at the final it is inside the seed spread at every
exit, and worse than loss-free at every exit.

### Load

Step line nearest each thousand, `load max/mean` and `dense share` per pass. Loss-free's `Aux:`
is the saturated value above and is not repeated.

| step | fixed aux `Aux:` | fixed aux load max/mean | fixed aux dense share | loss-free load max/mean | loss-free dense share |
|---|---|---|---|---|---|
| 10 | 1.0088 | 1.13 / 1.22 / 1.32 | 0.181 / 0.167 / 0.147 | 1.04 / 1.11 / 1.24 | 0.181 / 0.167 / 0.148 |
| 1000 | 1.1621 | 1.10 / 1.14 / 1.23 | 0.300 / 0.429 / 0.452 | 1.02 / 1.07 / 1.07 | 0.052 / 0.095 / 0.114 |
| 2000 | 1.1672 | 1.18 / 1.20 / 1.34 | 0.321 / 0.460 / 0.430 | 1.02 / 1.05 / 1.05 | 0.065 / 0.095 / 0.099 |
| 3000 | 1.1106 | 1.08 / 1.14 / 1.19 | 0.340 / 0.476 / 0.386 | 1.03 / 1.03 / 1.05 | 0.066 / 0.085 / 0.087 |
| 4000 | 1.1261 | 1.11 / 1.17 / 1.08 | 0.356 / 0.513 / 0.356 | 1.04 / 1.05 / 1.03 | 0.073 / 0.086 / 0.093 |
| 5000 | 1.1209 | 1.11 / 1.32 / 1.06 | 0.380 / 0.523 / 0.348 | 1.02 / 1.04 / 1.03 | 0.075 / 0.091 / 0.093 |
| 6000 | 1.0999 | 1.09 / 1.34 / 1.08 | 0.384 / 0.513 / 0.337 | 1.03 / 1.02 / 1.05 | 0.073 / 0.087 / 0.092 |
| 7000 | 1.0968 | 1.11 / 1.24 / 1.10 | 0.383 / 0.535 / 0.351 | 1.02 / 1.04 / 1.03 | 0.071 / 0.088 / 0.096 |
| 8000 | 1.1293 | 1.08 / 1.30 / 1.09 | 0.396 / 0.529 / 0.344 | 1.03 / 1.04 / 1.05 | 0.075 / 0.091 / 0.101 |
| 9000 | 1.1066 | 1.07 / 1.38 / 1.08 | 0.407 / 0.536 / 0.345 | 1.03 / 1.04 / 1.04 | 0.073 / 0.094 / 0.099 |
| 10000 | 1.1386 | 1.10 / 1.35 / 1.07 | 0.404 / 0.544 / 0.353 | 1.02 / 1.03 / 1.03 | 0.075 / 0.092 / 0.098 |
| 11000 | 1.1616 | 1.10 / 1.32 / 1.11 | 0.405 / 0.544 / 0.356 | 1.02 / 1.02 / 1.03 | 0.074 / 0.090 / 0.096 |
| 11910 | 1.1333 | 1.08 / 1.26 / 1.08 | 0.410 / 0.535 / 0.350 | 1.04 / 1.03 / 1.03 | 0.077 / 0.093 / 0.100 |

From step 1000 on, over every step line of the fixed aux arm: load max/mean 1.04 to 1.25 in
pass 1 (mean 1.110), 1.07 to 1.53 in pass 2 (mean 1.274), 1.03 to 1.48 in pass 3 (mean 1.128),
against loss-free's means 1.026 / 1.046 / 1.046; dense share 0.280 to 0.421, 0.415 to 0.553,
0.334 to 0.453 by pass (means 0.374 / 0.514 / 0.364), against loss-free's 0.07 to 0.11. An
even split over the 11 slots (3 dense, 8 MLP) puts 3/11 = 0.27 of the picks on the dense slots.
The aux term balances all eleven slots, so with a real gradient it moves a third to a half of
the picks onto the attention and IR slots, which run unconditionally anyway, and its MLP slots
end less even than loss-free's in every pass (most in pass 2). Loss-free is the more even
balancer on the MLP slots, and it leaves the dense split to the router.

### Storage

Closed-book entity `delta` by tier 10 / 100 / 1000 (tier 1 within noise: fixed aux -0.005 and
-0.002 at the final, z under 1), the table above with the fixed aux column added:

| save | form | fixed aux | loss-free | arm (a), seed 1234 | seed 4321 |
|---|---|---|---|---|---|
| 50M | in distribution | 0.001 / 0.013 / 0.148 | 0.003 / 0.009 / 0.083 | 0.003 / 0.010 / 0.087 | - / 0.011 / 0.094 |
| 50M | held out | 0.005 / 0.005 / 0.053 | 0.003 / 0.006 / 0.045 | 0.005 / 0.003 / 0.053 | - / 0.002 / 0.027 |
| 100M | in distribution | 0.007 / 0.116 / 0.339 | 0.007 / 0.076 / 0.325 | 0.012 / 0.099 / 0.332 | - / 0.091 / 0.332 |
| 100M | held out | 0.007 / 0.024 / 0.101 | 0.004 / 0.017 / 0.112 | 0.003 / 0.018 / 0.099 | - / 0.025 / 0.113 |
| final | in distribution | 0.107 / 0.426 / 0.486 | 0.072 / 0.365 / 0.474 | 0.063 / 0.378 / 0.467 | 0.080 / 0.382 / 0.492 |
| final | held out | 0.008 / 0.027 / 0.106 | 0.004 / 0.020 / 0.094 | 0.009 / 0.023 / 0.104 | 0.001 / 0.031 / 0.075 |

The fixed aux arm stores earlier (tier 1000 in distribution at 50M 0.148, sigma 0.008, against
0.083 to 0.094; tier 100 at 100M 0.116 against 0.076 to 0.099) and more at the final in
distribution at tiers 10 and 100 (gold top-1 at tier 100 0.621, arm (a) 0.415, loss-free
0.331). Tier 1000 in distribution is saturated at the final on every arm (rank 0), so its
`delta` reads the prior there.

Paired on the real name's rank per tier and form at the final (`compare`, all classes pooled;
`ckpts/inject/compare_full_vs_auxfix.log`, `compare_lossfree_vs_auxfix.log`,
`compare_s4321_vs_auxfix.log`; the diff is the other arm's rank minus the fixed aux arm's,
positive means the fixed aux arm stores more):

| tier, form | vs arm (a) | vs loss-free | vs seed 4321 |
|---|---|---|---|
| 10, in distribution | +0.0285 (z 4.2) | +0.0330 (z 4.9) | +0.0246 (z 3.5) |
| 100, in distribution | +0.0295 (z 14.5) | +0.0481 (z 19.1) | +0.0312 (z 14.4) |
| 1000, in distribution | 0.0000 (saturated) | 0.0000 (saturated) | 0.0000 (saturated) |
| 10, held out | -0.0019 (z -0.4) | +0.0018 (z 0.3) | +0.0058 (z 1.2) |
| 100, held out | +0.0141 (z 2.8) | +0.0105 (z 1.9) | +0.0124 (z 2.3) |
| 1000, held out | -0.0014 (z -0.1) | +0.0175 (z 1.4) | +0.0438 (z 3.5) |
| all | +0.0107 (z 5.2) | +0.0155 (z 7.1) | +0.0156 (z 7.4) |

The two AdamW seeds differ by z -0.7 at tier 100 in distribution and z -3.6 at tier 1000 held
out ([muon_micro.md](muon_micro.md), the reseed section), so the fixed aux effect at tier 100 in
distribution (z 14.5 against arm (a), z 14.4 against the reseed) is far outside the seed spread,
while the z 3.5 at tier 1000 held out against the reseed is the seed pair's own size. Muon on
the same cell was +0.044 (z 22.1); the working aux term is about two thirds of that, in the same
direction, the wrong one for the goal. Held out the fixed aux arm is within z 2.8 of arm (a).

## Verdict (the user's; the record's reading)

All three bars are read against the working aux term (the fixed aux arm), with arm (a) and the
reseed as the seed spread.

- **Loss: both pass; loss-free is the better of the two.** Against arm (a) both end inside the
  seed spread (0.029 at the end): loss-free 3.9055, fixed aux 3.9142, arm (a) 3.9037; paired at
  exit 3 loss-free -0.0055 (z -1.8), fixed aux -0.0076 (z -2.3). The fixed aux arm is worse
  than loss-free at every exit at the final and at every eval from 2k on.
- **Load: passes for loss-free.** Loss-free is more even on the MLP slots (max over mean 1.03
  to 1.05 at the final, means 1.026 / 1.046 / 1.046 from step 1000, against 1.08 / 1.26 / 1.08
  and 1.110 / 1.274 / 1.128) and leaves the dense split to the router (dense share 0.07 to 0.11
  against 0.35 to 0.54). The aux term balances all eleven slots, so with a real gradient it
  pushes a third to a half of the picks onto the attention and IR slots, which run
  unconditionally anyway, above the even 0.27, and leaves the MLP slots less even.
- **Storage: passes for loss-free; the working aux term stores more.** Loss-free sits inside the
  two AdamW seeds cell for cell. The fixed aux arm stores the biographies more in distribution
  (tier 100 +0.030, z 14.5 against arm (a), z 14.4 against the reseed, z 19.1 against
  loss-free; tier 10 z 4), far outside the seed spread (z -0.7 on the tier 100 cell), the wrong
  direction for the goal, about two thirds of Muon's effect on the same cell (+0.044, z 22.1);
  it also stores earlier (tier 1000 at 50M 0.148 against 0.083 to 0.094). Held out it is within
  z 2.8 of arm (a).

Rung R8 passes for loss-free on all three bars. Every earlier "aux" arm (the R0b arms, both
pilot arms, the Muon arms) ran with an aux term that balanced nothing, so its routing was the
router's alone; the fixed aux arm is the first balanced aux run, and it is worse than loss-free
on every bar and than the unbalanced arm (a) on loss and storage (arm (a)'s load is not in its
log).

Decision pending (the user): loss-free balancing into the real-run spec (the record's
recommendation), or aux stays. No pilot-shape arm is needed for the loss bar; the one open
question for the real run is the bias rate at 8 experts and seq 4096, which a pilot-shape arm P
under loss-free would read if the user wants it.

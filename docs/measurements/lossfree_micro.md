# Loss-free MoE balancing at micro scale (2026-10-19): the balancing rung's arm

Loss-free balancing in place of the aux loss on the arm (a) recipe, against arm (a) itself.
Held-out CE ends 0.0018 above the twin (paired 0.0055 at exit 3, z -1.8), inside the seed
spread at every save. Storage is unchanged cell for cell, within the two AdamW seeds. The MLP
load is even by the arm's own field (max over mean 1.03 to 1.05 per pass on average). The
comparison of evenness against arm (a) is not readable from the logs: the logged aux value
saturates in bf16 and reads the batch's token count, not the load (the load section). The rung
passes on loss and storage.

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
balancing against that floor.

## Verdict (the user's; the record's reading)

- **Loss: passes.** Held-out CE ends 0.0018 above arm (a) and 0.027 below seed 4321; paired at
  exit 3 the twin is better by 0.0055 (z -1.8), within 2 sigma and about one fifth of the 0.029
  seed spread at the final (one sixth of the paired 0.032); every save's paired difference is
  under the seed spread at that save except 150M, where loss-free is the better one.
- **Storage: unchanged.** Every cell sits inside or at the edge of the two AdamW seeds' range
  at the final, in distribution and held out; balancing is not a storage lever.
- **Load: even by its own field, the comparison unread.** Loss-free's MLP load is within 5% of
  even on average from step 1000 (max over mean 1.03 to 1.05 per pass, at most 1.14); the
  `Aux:` series sits at or below arm (a)'s at every matched window, but the field saturates in
  bf16 and reads the batch, so it cannot carry the "at least as even" half. Arm (a)'s load is
  not in its log.

The rung passes on loss and storage; the load bar's comparison half is unread, and the aux side
of the comparison is a floor rather than a balancer. Decision pending (the user): loss-free
balancing into the real-run spec, or aux stays. A pilot-shape check would be one pilot-shape
arm against arm P, as for Muon, only if the user wants the load evenness confirmed at 8 experts
and seq 4096; the cheaper read first is the load of arm (a)'s saves on `inject_val`, which needs
fp32 hard counts in the balancing value.

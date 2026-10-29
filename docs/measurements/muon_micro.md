# Muon on the hidden matrices at micro scale (2026-10-18): the optimizer rung's three arms

The build is in [NEXT.md](../plans/NEXT.md) (the R7 ladder row and the 2026-10-18 changelog):
`modules/model/optim.py` steps the hidden 2D matrices with Muon (Nesterov 0.95, five
Newton-Schulz steps in bf16, the update scaled to AdamW's RMS so the twin's lr and weight decay
carry over, fused `gate_up` weights orthogonalized as two halves) and everything else (the
embedding table, the heads, the router, the IR table, every zero-init tensor, scalars and
vectors) with AdamW's own arithmetic, chosen by `training: optimizer`. At the micro shape the
partition puts 25.7M parameters in 60 tensors on AdamW and 9.8M in 78 tensors on Muon (the
embedding table is most of this model), so every difference below comes from the hidden
matrices alone.

## The arms

`ckpts/inject/launch_muon.sh` (`config_micro_muon.yaml`), each from `seed_micro.pt` on the R0b splits, same data order
and seed as its AdamW twin:

- `full`: Muon on arm (a) (`inject_full_train`, full CE, no evidence) against
  `ckpts/evidence_inject_full/`. Run directory `ckpts/evidence_muon_full/`, log
  `ckpts/inject/muon_full.log`, reads `ckpts/inject/muon_full_{50M,100M,final}_rank_none.*`
  and `muon_full_{TAG}_exit.*` (the exit read paired per token against the twin's save at
  the same tokens, `eval_exit.py --pair`).
- `cf`: Muon on the copy-first warm-up arm of (c) (`--reader-kv`, 100M on
  `inject_retrieval_s100a50_train`, then `inject_retrieval_train`) against
  `ckpts/evidence_inject_retrieval_kv_cf/`. Reads `none`, `gold`, `swapped`, `prompt` at 50M,
  100M and the final, the exit read at every save.
- `reseed`: AdamW arm (a) at `sft.seed` 4321 against the original arm (a), the run-to-run
  spread the paired sigma does not carry.

The bar (NEXT.md, decided on the leak, not the loss): filler CE below the twin's at matched
tokens beyond the paired sigma and beyond the reseed spread; copying no later (the warm-up
switch criterion met by 100M); the warm-up arm's closed-book `delta` not above its twin's by 3
sigma paired at tier 100 or tier 1000 on either form. A loss gain with a worse leak fails.

## Arm (a): Muon against AdamW on full CE

299.44M tokens in 84.2 minutes (53k to 61k tokens per second between evals, 10.5 GB; the AdamW
twin ran 48k to 56k), so the per-tensor Newton-Schulz costs nothing at this shape.

Held-out `inject_val` CE by the trainer's eval (every 1,000 steps, 25M tokens each):

| step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Muon | 5.593 | 4.983 | 4.594 | 4.372 | 4.221 | 4.121 | 4.036 | 3.971 | 3.915 | 3.875 | 3.847 | 3.834 |
| AdamW | 5.668 | 5.137 | 4.744 | 4.497 | 4.335 | 4.231 | 4.140 | 4.067 | 4.002 | 3.954 | 3.925 | 3.904 |
| gap | 0.075 | 0.154 | 0.150 | 0.125 | 0.114 | 0.110 | 0.104 | 0.096 | 0.087 | 0.079 | 0.078 | 0.070 |

The gap peaks at 50M and halves through the cosine. Paired per token on the saves at the same
tokens (`muon_full_{TAG}_exit.log`, 148k supervised tokens of `inject_val`, 162 clusters), AdamW
CE minus Muon CE at exit 3: +0.136 (se 0.007, z 20) at 50M, +0.114 at 100M, +0.108 at 150M,
+0.095 at 200M, +0.073 at 250M, +0.068 (se 0.004, z 16) at the final (4.2208 against 4.2886);
at the final +0.048 at exit 1 and +0.067 at exit 2. Depth pays
the same way on both (Muon 1 to 2 +0.112, 2 to 3 +0.006; AdamW +0.094, +0.006); the gain
concentrates in the least confident deciles on both (ratio 4.9 lowest to highest).

Closed-book entity `delta` (`closed_book_rank.py bios --evidence none`), in distribution /
held out, by tier 10 / 100 / 1000 (tier 1 within noise on both):

| save | Muon in distribution | AdamW in distribution | Muon held out | AdamW held out |
|---|---|---|---|---|
| 50M | 0.014 / 0.072 / 0.284 | 0.003 / 0.010 / 0.087 | 0.003 / 0.020 / 0.151 | 0.005 / 0.003 / 0.053 |
| 100M | 0.034 / 0.211 / 0.421 | 0.012 / 0.099 / 0.332 | 0.003 / 0.035 / 0.179 | 0.003 / 0.018 / 0.099 |
| final | 0.145 / 0.451 / 0.491 | 0.063 / 0.378 / 0.467 | 0.002 / 0.040 / 0.136 | 0.009 / 0.023 / 0.104 |

Gold top-1 in distribution at tier 100: Muon 0.026 / 0.110 / 0.821 at 50M / 100M / final,
AdamW 0.015 / 0.046 / 0.415. Paired (`compare`, AdamW rank minus Muon rank, classes and forms
pooled, positive means Muon ranks the real value better): at 50M +0.052 (z 15.9) at tier 100
and +0.129 (z 19.6) at tier 1000; at 100M +0.079 (z 20.5) and +0.035 (z 6.1); at the final
+0.036 (z 11.8) and +0.019 (z 2.9), tier 10 +0.034 (z 7.8).

**Read.** On the control recipe Muon stores the biographies about twice as early (its 50M read
sits where AdamW's 100M does) and more at the end at every tier above 1, in distribution and
held out, while its language CE is 0.07 nats lower at the end and 0.15 lower mid-run. This is
the descriptive storage read the row asked for, with no bar attached; for the goal it is the
wrong direction, and it raises the prior that the warm-up arm's leak is worse than its twin's,
since that leak is written by real-value supervision and an optimizer that writes facts faster
writes more of it unless copying also comes earlier. The warm-up arm decides.

## The copy-first warm-up arm: Muon against AdamW on the retrieval recipe

`ckpts/evidence_muon_kv_cf/`, log `ckpts/inject/muon_kv_cf.log`: the warm-up split to the
100M save (the launcher drops `STOP` when it appears; the stop save a few steps later is the
one the main split resumed from and the reads use), then `inject_retrieval_train` to 299.19M,
60.8 minutes for the main split at 52k to 60k tokens per second, 13.8 GB. The AdamW twin is
the warm-up arm of 2026-10-05 (`ckpts/evidence_inject_retrieval_kv_cf/`, filler CE 3.9106).

Held-out `inject_val` CE at 1k / 2k / 3k / 4k / 5k / 6k / 7k / 8k / 9k / 10k / 11k / final
steps: Muon 5.589 / 4.992 / 4.596 / 4.372 / 4.232 / 4.126 / 4.042 / 3.972 / 3.917 / 3.876 /
3.849 / 3.833, AdamW 5.667 / 5.135 / 4.741 / 4.487 / 4.338 / 4.224 / 4.134 / 4.062 / 4.004 /
3.956 / 3.932 / 3.911: the same 0.08 to 0.14 nat lead as on arm (a), 0.078 at the end.
Paired per token at exit 3 (`muon_cf_{TAG}_exit.log`), AdamW minus Muon: +0.124 (z 15) at
50M, +0.107 at 100M, +0.090 at 150M, +0.091 at 200M, +0.082 at 250M, +0.078 (se 0.005, z 16.5)
at the final (4.2196 against 4.2978), the same at exits 1 and 2; depth pays the same way on
both (1 to 2 +0.109, 2 to 3 +0.005).

The switch criterion at 100M: gold top-1 in distribution 0.981 / 0.987 / 0.985 / 0.980 by
tier (AdamW 0.973 / 0.973 / 0.959 / 0.953), held out 0.55 to 0.59 (AdamW 0.63 to 0.65);
closed-book `delta` within 2.1 sigma of the prior in every entity cell on both forms (AdamW
within 2.0), and the same at 50M (within 1.2 sigma on both arms). At 50M neither arm copies
yet (gold rank 0.44 to 0.45 at top-1 0.02 on Muon, 0.45 to 0.48 at 0.01 to 0.02 on AdamW),
so copying emerges between 50M and 100M under both optimizers and no later under Muon; the
step 2000 reader mass of 0.36 under the mixed condition was attention, not yet copying.

Copying at the final (`gold`, `swapped`, `prompt`; AdamW twin in brackets): gold top-1 in
distribution 1.000 at every tier (0.999 to 1.000), held out 0.90 at every tier (0.60 / 0.63 /
0.61 / 0.74 by tier); swapped card followed on every item in distribution on both, held out
0.89 to 0.92 with `mr_ll` 0.004 to 0.010 (0.66 to 0.69, `mr_ll` 0.027 to 0.042); prompt copy
in distribution 0.999 on both, held out rank 0.004 with top-1 0.90 (0.008, 0.84). Muon copies
the held-out template far better than AdamW at the same tokens.

The leak, closed-book entity `delta` at the final (`none`), Muon against the AdamW twin:

| form | tier 10 | tier 100 | tier 1000 |
|---|---|---|---|
| in distribution | 0.023 (z 3.5) vs 0.000 | 0.035 (z 5.6) vs 0.015 (z 2.6) | 0.136 (z 8.8) vs 0.085 (z 6.4) |
| held out | 0.004 vs -0.001 | 0.012 (z 2.6) vs 0.007 (z 2.0) | 0.037 (z 3.7) vs 0.047 (z 5.1) |

Dates at tier 1000 in distribution 0.114 (z 3.1) against 0.106 (z 3.6); nouns 0.110 (z 3.9).
Paired on the real name's rank (`compare --by tier,form`, AdamW minus Muon, positive means
Muon ranks the real value better, i.e. leaks more): tier 1000 in distribution +0.034 (z 2.6),
tier 100 in distribution +0.006 (z 0.9), tier 10 in distribution +0.008 (z 1.3); held out
+0.007 (z 0.6), +0.008 (z 1.6), +0.008 (z 1.6); pooled over forms +0.020 (z 2.4) at tier 1000
and +0.007 (z 1.7) at tier 100. On the prior-corrected `delta`, each in-distribution cell at
tiers 10, 100 and 1000 is worse by z 2.4 to 2.6 (the two arms' sigmas combined), the held-out
cells equal (tier 1000 held out -0.010, z -0.8).

**Read.** By the letter of the bar (not above the twin's by 3 sigma paired at tier 100 or
tier 1000 on either form) Muon passes, with the largest cell at z 2.6. By the sign it leaks
more in distribution at every tier above 1, about 1.6x at tier 1000 and 2.4x at tier 100, and
the three cells pooled are beyond 3 sigma; the arm (a) storage read predicted this. Held out
the two arms are equal, and Muon copies better held out, so the extra leak is on the surface
form, the same place the AdamW leak sits. No seed-spread control exists for the leak: the
reseed arm below measures arm (a)'s spread, not this recipe's.

## The AdamW reseed: the run-to-run spread

`ckpts/evidence_adamw_full_s4321/` (`config_micro_seed4321.yaml`: `sft.seed` 4321, every
other number as arm (a), the same `seed_micro.pt`), 299.44M tokens in 82.4 minutes. Held-out
CE at 1k / 2k / 3k / 4k / 5k / 6k / 7k / 8k / 9k / 10k / 11k / final steps: 5.707 / 5.179 /
4.800 / 4.531 / 4.372 / 4.246 / 4.151 / 4.082 / 4.016 / 3.978 / 3.949 / 3.932 against the
original seed's 5.668 / 5.137 / 4.744 / 4.497 / 4.335 / 4.231 / 4.140 / 4.067 / 4.002 / 3.954
/ 3.925 / 3.904: the seed spread is 0.012 to 0.056 nats, 0.029 at the end, the reseed worse
throughout. Muon's lead over the original seed is 2.4x the spread at the end (0.070) and 3 to
4x mid-run (0.154 at 50M against 0.042).

Storage mid-run, entity `delta` at tier 100 / 1000: at 50M in distribution 0.011 / 0.094
(original seed 0.010 / 0.087; Muon 0.072 / 0.284), held out 0.002 / 0.027 (0.003 / 0.053;
Muon 0.020 / 0.151); at 100M in distribution 0.091 / 0.332 (0.099 / 0.332; Muon 0.211 /
0.421), held out 0.025 / 0.113 (0.018 / 0.099; Muon 0.035 / 0.179). The two seeds agree to
within 0.01 in distribution mid-run, so Muon's early storage is the optimizer, not the seed.

Storage at the final, entity `delta` by tier 10 / 100 / 1000: in distribution 0.080 / 0.382 /
0.492 (original seed 0.063 / 0.378 / 0.467; Muon 0.145 / 0.451 / 0.491), held out 0.001 /
0.031 / 0.075 (0.009 / 0.023 / 0.104; Muon 0.002 / 0.040 / 0.136). Paired on the real name's
rank per tier and form (`compare --by tier,form`), the two AdamW seeds differ by z -0.7 at tier
100 in distribution, z 0.4 at tier 100 held out, z 0.6 at tier 10 in distribution and z -3.6
at tier 1000 held out (-0.045, the reseed storing less); Muon against the original seed on the
same read: +0.075 (z 11.6) at tier 10 in distribution, +0.044 (z 22.1) at tier 100 in
distribution, +0.027 (z 4.9) at tier 100 held out, +0.037 (z 2.9) at tier 1000 held out. So
the Muon storage effect at tiers 10 and 100 in distribution is far outside the seed spread;
at tier 1000 held out it is inside it (one seed pair alone moves that cell by z 3.6). The
seed spread of this instrument at tier 1000 is of the same size as the warm-up arm's leak
difference (0.034 at z 2.6), which bears on the verdict below.

Paired per token at the finals (`muon_reseed_final_exit.log`, the reseed against the original
seed): the reseed is worse by 0.032 at exit 3 (se 0.0055, z -5.8), 0.036 at exits 1 and 2;
at the earlier saves the spread at exit 3 is 0.049 (50M, se 0.020), 0.029 (100M), 0.005
(150M), 0.020 (200M) and 0.030 (250M) against Muon's 0.136, 0.114, 0.108, 0.095 and 0.073 at
the same saves, so Muon's gain exceeds the spread at every save by 2x or more.
So two AdamW seeds differ at z 6 on the paired per-token read, which makes "beyond the paired
sigma" a weak bar on its own; the bar that counts is the spread itself, and Muon's gain at the
final is 2.1x it (0.068 against 0.032), 3 to 4x mid-run.

## Verdict (the user's; the record's reading)

The three bars of the R7 row, in order:

- **Loss: passes.** Muon's held-out CE is below its twin's at every read on both recipes,
  0.07 to 0.15 nats, z 14.9 to 24.9 paired per token, 2 to 4x the one seed pair's spread.
- **Copying no later: passes.** Both warm-up arms copy between 50M and 100M; the switch
  criterion is met at 100M with gold top-1 0.98 to 0.99 against 0.95 to 0.97, and at the final
  Muon copies the held-out template at 0.90 against 0.60 to 0.74 and follows a swapped
  held-out card at 0.90 against 0.67.
- **The leak: passes by the letter, worse by the sign.** No cell is above the twin's by 3
  sigma paired at tier 100 or 1000 on either form (the largest z 2.6, tier 1000 in
  distribution). In distribution Muon leaks more at every tier above 1 (1.6x at tier 1000,
  2.4x at tier 100, each cell z 2.4 to 2.6 on `delta`), held out it is equal, and the arm (a)
  read says why: Muon writes facts into the hidden matrices about twice as fast, far outside
  the seed spread at tiers 10 and 100 in distribution. Against that, the storage instrument's
  own seed spread at tier 1000 (z 3.6 between two AdamW seeds on arm (a)) is as large as the
  leak difference, so one micro pair cannot separate a 1.6x leak at tier 1000 from the seed;
  the pilot shape can (arm C against arm P: z 43 is arm C's `delta` against its own prior; the paired `compare` reads z -81 in distribution and -6.4 held out).
- The attention-logit bound is unread (the sampled read is not built).

**Recommended at the time (dropped 2026-10-21, the user: Muon is out of the real-run spec and the pilot-shape arm was not run).** The rung's micro step passes as written and the row's next step is one
Muon arm P at the pilot shape against arm P, about one night, which also times the per-tensor
Newton-Schulz at that shape. The reason to run it rather than stop here is power: the micro
leak read is one seed pair at z 2.6 against a seed spread of z 3.6, and the pilot's paired
read against the control is the instrument that can tell a 1.6x leak from noise. The reason to
stop here is the goal: an optimizer that stores facts in the weights twice as fast is the
wrong direction for "facts in the store, not the weights", the in-distribution sign is
consistent across tiers, and a loss gain with a worse leak fails. If the pilot-shape arm runs,
its leak read is the decision and the loss is not; if it does not, Muon stays out of the
real-run spec and the record keeps the storage result (arXiv 2509.26030's direction
confirmed at this scale: Muon stores the tail more, and earlier).

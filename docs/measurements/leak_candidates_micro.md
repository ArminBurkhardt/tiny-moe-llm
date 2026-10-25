# The two leak candidates at micro scale (2026-10-25): the first name mention masked, support from the gold card only

The question, from the repository review of 2026-10-24 ([review_2026-10-24.md](../review_2026-10-24.md),
"Leak candidates"): what writes the tier 1000 level that the copy-first warm-up arm stores after
copying is in place (entity `delta` 0.085, z 6.4 in distribution, 0.047, z 5.1 held out at 300M;
[r0b_micro_pilot.md](r0b_micro_pilot.md), "The warm-up arm in full"). The placeholder name arm
already showed that the level does not follow the number of supervised spans pairing the real name
with the real value (2026-10-08). The review named two data-side routes outside the card path:

- **m1, value-first templates.** Templates such as "{v} is the hometown of {s}." put the name
  after the value; the name mask is always 1, so when such a sentence opens a biography the name
  is predicted from the value before it is in context, a value-to-name direction the closed-book
  instrument cannot see. Lever: mask the first run of name (or placeholder) tokens per document
  (`prepare_injection_data.py --mask-first-name-mention`, 1,474,885 tokens of 300M masked).
- **g1, distractor-collision spans.** Gold-absent documents count a value as supported when it
  collides with a distractor card, about 5.6% of gold-absent spans and 0.56% of all spans, each one
  supervising the real name with the real value with no gold card present. Lever: count a value as
  supported by the gold card only (`--support-gold-only`).

Both flags change `.mask` only (44 of 44 companion files byte identical to the flagless builds,
built 2026-10-24, `ckpts/inject/build_leak_arms.sh`). Criterion, as for the placeholder arm: the
candidate writes the level if the paired prior-controlled `delta` (warm-up arm minus the leak arm)
at tier 1000 reaches z at or beyond -3 in the leak arm's favour, i.e. the leak arm binds names
clearly less; with the sign convention of `compare` (positive means the first file, the warm-up
arm, binds more) that is a paired value at z +3 or above.

## Run

- Recipe: the copy-first warm-up arm's, micro shape (`config_micro.yaml`, 35.3M parameters),
  key/value reader, `seed_micro.pt`, on the code in the tree (the reviewed code of 2026-10-24,
  including the resume epoch fix). 100M tokens on the swap rate 1.0 split, STOP dropped on the
  100M save, `closed_book_rank.py bios --evidence none|gold` on that save, then the same run
  relaunched with `--switch-split` on the swap rate 0.15 split to the epoch's end at 299.19M; at
  the final `none`, `gold` and `swapped`, both forms, batch 1024, then `compare` against the
  recorded warm-up arm's final (`ckpts/evidence_inject_retrieval_kv_cf/checkpoint_evidence_final.pt`,
  its `rank_retrieval_kv_cffinal_none.json`) with `--class entity --by form,tier`.
- Launcher `bash ckpts/inject/launch_leak_arm.sh m1|g1`, run in sequence on 2026-10-25 by
  `ckpts/inject/queue_2026_10_25b.sh` (after the paired `seg_e3` exit read, before the clean-slice
  read), detached from any watch.

| arm | splits (warm-up, then main) | warm-up | main | reads done |
|---|---|---|---|---|
| m1 | `inject_retrieval_s100a50m1_train`, `inject_retrieval_s15a50m1_train` | 12:33 to 13:04, exit 10 | 13:11 to 14:13 (60.5 min), exit 0 | 14:24 |
| g1 | `inject_retrieval_s100a50g1_train`, `inject_retrieval_s15a50g1_train` | 14:24 to 14:56, exit 10 | 15:03 to 16:05 (60.6 min), exit 0 | 16:16 |
| cfb (the same-code twin, below) | `inject_retrieval_s100a50_train`, `inject_retrieval_train` | 16:49 to 17:19, exit 10 | 17:26 to 18:25 (57.0 min), exit 0 | 18:36 |

- About 54k tokens per second in the warm-up, peak 12.2 GB there and 12.5 GB at the most (the
  recorded warm-up arm 13.3 to 13.8 GB on the code of that week).
- The stop fell within one log interval of the 100M save (10 optimizer steps), so the trainer
  exited 10 on that save without a second one (the rule landed 2026-10-24). 100M saves
  `ckpts/evidence_leak_m1/checkpoint_evidence_tok100M_loss7.4254.pt` and
  `ckpts/evidence_leak_g1/checkpoint_evidence_tok100M_loss7.3624.pt`; finals
  `ckpts/evidence_leak_{m1,g1}/checkpoint_evidence_final.pt`.
- Final `loop_scale` m1 [1.51, 0.58, 0.19], g1 [1.48, 0.58, 0.18] (the recorded warm-up arm
  [1.58, 0.63, 0.22]).

## Filler CE on `inject_val` (40 batches, 1,104,743 supervised tokens)

The two new arms run on the fixed save and eval grid (steps 4980 to 11915 after the switch); the
recorded arms ran on the old grid (steps 5000 to 11916). The new arms log CE accumulated in fp32,
the recorded ones in bf16 (about 0.002, review finding 4); the step 0 eval reads 11.1109 against
11.0975.

| step (new grid / old grid) | m1 | g1 | cfb | recorded warm-up arm | original arm (c) |
|---|---|---|---|---|---|
| 1000 | 5.658 | 5.655 | 5.661 | 5.667 | 5.662 |
| 2000 | 5.156 | 5.141 | 5.150 | 5.135 | 5.139 |
| 3000 | 4.793 | 4.773 | 4.767 | 4.741 | 4.734 |
| 4000 | 4.513 | 4.502 | 4.497 | 4.487 | 4.481 |
| 4980 / 5000 | 4.344 | 4.338 | 4.338 | 4.338 | 4.326 |
| 5980 / 6000 | 4.227 | 4.225 | 4.220 | 4.224 | 4.214 |
| 6980 / 7000 | 4.121 | 4.123 | 4.120 | 4.134 | 4.111 |
| 7980 / 8000 | 4.042 | 4.043 | 4.043 | 4.062 | 4.033 |
| 8980 / 9000 | 3.979 | 3.982 | 3.982 | 4.004 | 3.975 |
| 9980 / 10000 | 3.935 | 3.938 | 3.937 | 3.956 | 3.933 |
| 10980 / 11000 | 3.903 | 3.906 | 3.906 | 3.932 | 3.898 |
| final (11915 / 11916) | 3.886 | 3.889 | 3.8885 | 3.9106 | 3.8842 |

- The two leak arms agree within 0.003 from the switch on, so the flags cost nothing in language.
- In the warm-up phase both sit within 0.021 of the recorded warm-up arm at steps 1000 and 2000
  and 0.015 to 0.052 above it at steps 3000 and 4000; after the switch they track the original arm (c) within 0.002 to 0.018 and end 0.002 and
  0.005 above it, where the recorded warm-up arm drifted to 0.034 above it at step 11000 and ended
  0.026 above. The recorded arm's gap, called unexplained in the record, was therefore not the
  warm-up: the same recipe on the reviewed code closes it. Which change of the week closes it is
  not identified, and each arm is one seed.

## The switch read at 100M

Entity class, closed-book `delta` (z) by tier 1 / 10 / 100 / 1000:

| arm, 100M | form | tier 1 | tier 10 | tier 100 | tier 1000 |
|---|---|---|---|---|---|
| m1 | indist | -0.006 (-1.2) | -0.004 (-0.8) | 0.006 (1.1) | -0.007 (-0.6) |
| m1 | heldout | -0.004 (-1.0) | 0.003 (0.7) | 0.001 (0.2) | 0.005 (0.5) |
| g1 | indist | -0.002 (-0.4) | 0.002 (0.4) | 0.005 (1.0) | 0.003 (0.3) |
| g1 | heldout | 0.001 (0.2) | 0.001 (0.2) | 0.005 (1.3) | 0.011 (1.2) |
| cfb | indist | -0.008 (-1.5) | -0.001 (-0.2) | -0.001 (-0.1) | 0.010 (1.0) |
| cfb | heldout | -0.002 (-0.6) | -0.002 (-0.4) | 0.006 (1.3) | -0.002 (-0.2) |
| recorded warm-up arm | indist | | | -0.006 (-1.1) | -0.001 (-0.1) |
| recorded warm-up arm | heldout | | | -0.002 (-0.5) | -0.003 (-0.3) |

Inside 3 sigma in every cell of every class on both forms (largest |z| 1.6 on m1, 2.0 on g1, 1.8
on cfb).

Gold card top-1, entity, by tier:

| arm, 100M | indist | heldout |
|---|---|---|
| m1 | .887 .904 .882 .890 | .453 .465 .471 .477 (rank 0.027) |
| g1 | .898 .899 .883 .877 | .678 .702 .687 .687 (rank 0.016) |
| cfb | .964 .961 .949 .950 | .597 .615 .599 .560 (rank 0.016 to 0.021) |
| recorded warm-up arm | .973 .973 .959 .953 | 0.63 to 0.65 (rank 0.018 to 0.021) |

- Both arms switched at the launcher's fixed 100M with copying at 0.88 to 0.90 in distribution,
  under the recorded arm's 0.95 to 0.97 and just under the pilot spec's switch bar (0.9 at every
  tier, [PILOT.md](../plans/PILOT.md)); the micro launcher does not check that bar. The closed-book
  half of the criterion holds. By the final save copying is complete on both (below), so the
  early switch did not leave copying short; whether it let more real-value supervision land before
  copying was complete is not separable from the reads.
- m1 copies worse held out at 100M (0.45 to 0.48) and catches up by the final; g1 copies better
  held out from 100M on.

## The final read

Entity class, closed-book `delta` (z) from each save's own log:

| arm, final | form | tier 1 | tier 10 | tier 100 | tier 1000 |
|---|---|---|---|---|---|
| m1 | indist | -0.001 (-0.1) | 0.005 (0.8) | 0.024 (3.6) | 0.073 (5.2) |
| g1 | indist | -0.010 (-1.4) | -0.001 (-0.2) | 0.020 (3.0) | 0.083 (5.6) |
| cfb | indist | -0.005 (-0.8) | 0.000 (0.1) | 0.018 (2.9) | 0.070 (5.0) |
| recorded warm-up arm | indist | 0.004 (0.6) | 0.000 (0.0) | 0.015 (2.6) | 0.085 (6.4) |
| m1 | heldout | -0.003 (-0.9) | 0.002 (0.5) | 0.011 (3.1) | 0.032 (3.6) |
| g1 | heldout | 0.003 (0.7) | 0.008 (1.7) | 0.015 (3.3) | 0.035 (3.4) |
| cfb | heldout | 0.002 (0.3) | 0.006 (1.3) | 0.016 (3.7) | 0.038 (3.5) |
| recorded warm-up arm | heldout | 0.001 (0.3) | -0.001 (-0.3) | 0.007 (2.0) | 0.047 (5.1) |

Other classes at the final: m1 dates at tier 1000 -0.015 (z -0.5) in distribution, nouns 0.054
(z 2.5); g1 dates 0.071 (z 2.6) in distribution and 0.049 (z 2.5) held out, nouns 0.030 (z 3.1)
at tier 100 and 0.086 (z 3.8) at tier 1000 in distribution (the recorded arm's nouns 0.020, z 1.9
and 0.024, z 1.1).

**Paired against the recorded warm-up arm** (`compare_cf_vs_leak_{m1,g1}.log`, same items, entity,
prior-controlled `delta`, warm-up arm minus leak arm; positive means the warm-up arm binds names
more; z item-level / clustered by person):

| cell | m1 | g1 |
|---|---|---|
| indist, tier 100 | -0.0094 (z -1.2 / -1.1) | -0.0050 (z -0.6 / -0.6) |
| indist, tier 1000 | +0.0115 (z 0.7 / 0.7) | +0.0017 (z 0.1 / 0.1) |
| heldout, tier 100 | -0.0039 (z -0.9 / -0.9) | -0.0078 (z -1.5 / -1.4) |
| heldout, tier 1000 | +0.0147 (z 1.3 / 1.4) | +0.0118 (z 0.9 / 1.0) |

On the raw rank the warm-up arm ranks the real name better at tier 1000 in distribution (m1
-0.032, z -2.2; g1 -0.015, z -1.0), and the prior moves with it, so `delta` does not.

## Copying at the final

| arm, final | gold top-1 indist | gold top-1 heldout | swapped followed indist | swapped followed heldout (`mr_ll`) |
|---|---|---|---|---|
| m1 | 1.000 at every tier | .660 .653 .715 .773 (rank 0.012 to 0.020) | 1.000 (`mr_ll` 0.000) | .673 .668 .667 .646 (0.020 to 0.037) |
| g1 | 1.000 at every tier | .789 .811 .819 .860 (rank 0.004 to 0.009) | 1.000 (`mr_ll` 0.000) | .782 .804 .784 .786 (0.010 to 0.012) |
| cfb | 1.000 at every tier | .581 .624 .622 .677 (rank 0.010 to 0.019) | 1.000 (`mr_ll` 0.000) | .646 .651 .640 .636 (0.021 to 0.037) |
| recorded warm-up arm | 0.999 to 1.000 | 0.60 to 0.74 (rank 0.010 to 0.022) | 1.000 (`mr_ll` 0.000) | 0.663 to 0.690 (0.027 to 0.042) |

Gold top-1 is the entity class; the swapped follow rate is all classes, as in the record.

## Verdict (the main thread's)

- **Both candidates are falsified.** No tier 1000 cell is near the bar: the paired value is
  +0.0115 (z 0.7) and +0.0147 (z 1.3) for m1 and +0.0017 (z 0.1) and +0.0118 (z 0.9) for g1, all
  far from z +3, and the tier 100 cells point the other way inside noise. The tier 1000 residual is
  neither the name's first mention (value-to-name supervision in value-first openings) nor the
  gold-absent documents' collision spans. The candidates the record lists as open stay open: the
  prior drifting under real-name exposure and the input path.
- **Side finding: `--support-gold-only` raises held-out swapped-card following** from about 0.67
  (the recorded warm-up arm and m1) to about 0.79, with held-out gold top-1 0.79 to 0.86 against
  0.60 to 0.74 and `mr_ll` 0.010 to 0.012 against 0.027 to 0.042, at no cost in language (filler
  CE within 0.003 of m1) or in the leak. Removing the spans that pair a value with a name when only
  a distractor card carries it removes a small conflicting signal against reading the card, and
  the held-out templates gain most. One seed. A candidate for the real-run data spec.
- **The warm-up recipe on the reviewed code reaches filler CE 3.886 and 3.889**, against the
  recorded warm-up arm's 3.9106 and the original arm (c)'s 3.8842, so the 0.026 gap the record
  calls unexplained was the code of that week, not the warm-up.

## Open

- The R0b rule's letter at tier 100: both arms sit at or beyond 3 sigma off their prior at tier
  100 on at least one form (m1 3.6 in distribution and 3.1 held out; g1 3.0 and 3.3), where the
  recorded warm-up arm read 2.6 and 2.0 and the placeholder arm 3.0 and 3.7. Paired against the
  warm-up arm every tier 100 cell is inside noise (z -0.6 to -1.5), so this reads as the recipe's
  tier 100 margin being thin on one seed, not as a change the flags made. `compare full masked`
  was not run for these arms. The same-code twin `cfb` (next section) reads 2.9 in distribution
  and 3.7 held out, so on today's code the thin margin belongs to the recipe itself.
- What writes the tier 1000 level: the prior drifting and the input path, untested by a lever.
- The value-to-name closed-book probe (the Open items row) is still not built; m1 masks the
  route without measuring the direction itself.
- `--support-gold-only` as a real-run data lever: decided 2026-10-25, carried into the real corpus builder behind a flag (RUN2.md row 3).

## The same-code twin `cfb` (2026-10-25, evening)

The two leak arms were paired above against the recorded warm-up arm, which ran on the code of
2026-10-05, so a flag effect and a code effect were not separable in that compare, nor in the
copying at the switch or in the filler CE gap. `cfb` reruns the recorded warm-up arm's exact
recipe on the flagless splits (`inject_retrieval_s100a50_train`, then `inject_retrieval_train`)
on the code in the tree, as the twin both leak arms are read against. The cosine's planned steps rule the estimator out as the cause of the 0.022 filler CE gap: about 11,907 for the recorded warm-up arm and about 11,911 for `cfb`, `m1` and `g1`.

- Launcher `bash ckpts/inject/launch_leak_arm.sh cfb` (the `cfb` case added the same day),
  launched 16:49 detached. Warm-up to the 100M save, STOP dropped on it, exit 10 at 17:19 on that
  save (the last checkpoint 10 optimizer steps back, within one log interval, so no second save),
  `none` and `gold` read at 100M; main split with `--switch-split` 17:26 to 18:25, 299.19M tokens
  in 57.0 minutes, exit 0; `none`, `gold` and `swapped` at the final, then the launcher's
  `compare` against the recorded warm-up arm (done 18:36). The two same-code compares were run by
  hand at 18:29: `closed_book_rank.py compare rank_leak_cfb_final_none.json
  rank_leak_{m1,g1}_final_none.json --class entity --by form,tier`.
- About 53k to 60k tokens per second, peak 12.79 GB in the warm-up and 12.52 GB in the main phase;
  final `loop_scale` [1.49, 0.59, 0.19]. Saves: 100M
  `ckpts/evidence_leak_cfb/checkpoint_evidence_tok100M_loss7.3953.pt`, final
  `ckpts/evidence_leak_cfb/checkpoint_evidence_final.pt`.
- Its rows are in the tables above (the run table, the filler CE column, the 100M and final
  closed-book rows, gold top-1 at 100M, copying at the final). Other classes at the final: dates
  below the prior at every tier but 10 in distribution (tier 1000 -0.029, z -1.0), nouns 0.022
  (z 2.2) at tier 100 and 0.020 (z 0.9) at tier 1000 in distribution; held out every date and noun
  cell within z 1.5. Clustered by person, the entity tier 100 cells read z 2.9 in distribution and
  3.4 held out.

**Same-code pairing** (`compare_cfb_vs_leak_{m1,g1}.log`, entity, prior-controlled `delta`, cfb
minus the leak arm; positive means cfb binds names more, i.e. the lever worked; z item-level /
clustered by person):

| cell | cfb minus m1 | cfb minus g1 |
|---|---|---|
| indist, tier 100 | -0.0057 (z -0.7 / -0.7) | -0.0012 (z -0.1 / -0.1) |
| indist, tier 1000 | -0.0030 (z -0.2 / -0.2) | -0.0129 (z -0.7 / -0.7) |
| heldout, tier 100 | +0.0053 (z 1.1 / 1.0) | +0.0014 (z 0.3 / 0.2) |
| heldout, tier 1000 | +0.0054 (z 0.4 / 0.4) | +0.0024 (z 0.2 / 0.2) |

Tiers 1 and 10 are inside z 0.9 on both pairs; on the raw rank the largest paired cell is z 1.1
against m1 and 1.4 against g1 (tier 1000 in distribution, where g1 ranks the real name better and
its prior moves with it).

**The recorded warm-up arm against cfb** (`compare_cf_vs_leak_cfb.log`, the same recipe on the
two codes, recorded arm minus cfb): `delta` in distribution -0.0037 (z -0.5) at tier 100 and
+0.0146 (z 0.8) at tier 1000, held out -0.0093 (z -1.8) and +0.0094 (z 0.7); tier 10 held out
-0.0074 (z -1.4); on the raw rank the recorded arm ranks the real name better at tier 1000 in
distribution (-0.037, z -2.5) and its prior moves with it. No cell beyond z 1.8 on `delta`.

**Filler CE.** cfb sits 0.010 to 0.026 above the recorded warm-up arm at warm-up steps 2000 to
4000, level with it at the switch (4.338), then 0.014 to 0.026 below it from step 6980 on and
0.0221 below at the final (3.8885 against 3.9106); it ends within 0.003 of m1 (3.886) and g1
(3.889). In the warm-up m1 sits 0.026 and 0.016 above cfb at steps 3000 and 4000 and g1 within
0.006; the difference is gone after the switch.

### Verdict on the twin (the main thread's)

1. **The same-code pairing confirms both candidates falsified:** every cell of both pairs is within
   z 1.1, with no sign of either lever binding names less at tier 1000.
2. **Copying at the switch:** cfb copies at 0.95 to 0.96 in distribution at 100M (the recorded arm
   0.95 to 0.97), so the 0.88 to 0.90 of m1 and g1 was the flags slowing copying, not the code.
3. **The filler CE gap:** the 0.022 gap between today's arms and the recorded warm-up arm (3.889
   against 3.911) is the code since 2026-10-05, not the flags; the specific change is not
   identified (the fp32 CE accounting explains about 0.002 of it).
4. **The tier 100 caveat:** on today's code and this seed the warm-up recipe itself reads tier 100
   at z 2.9 in distribution and 3.7 held out (the recorded arm 2.6 and 2.0), so the recipe's tier
   100 margin is thin for every arm of the matched set. The held-out tier 100 cell is over 3 sigma
   on all three arms today (m1 3.1, g1 3.3, cfb 3.7), which by the letter of the R0b rule is a
   fail at tier 100 held out on one seed. It is stated as a caveat for the user, not a verdict
   change: paired against the recorded arm the same cell moves -0.009 (z -1.8), inside noise.
   Tier 1000 (0.070, z 5.0 in distribution; 0.038, z 3.5 held out) is the known residual.

Decided 2026-10-25 (the user, RUN2.md row 9): the thin tier 100 margin is accepted as the recipe's known state at micro on one seed; no second seed of `cfb`.

## Raw files

`ckpts/inject/leak_{m1,g1}.log` (training and launcher lines), `launch_leak_{m1,g1}.log`,
`queue_2026_10_25.log`, `rank_leak_{m1,g1}_100M_{none,gold}.{json,log}`,
`rank_leak_{m1,g1}_final_{none,gold,swapped}.{json,log}`, `compare_cf_vs_leak_{m1,g1}.log`;
the twin's `leak_cfb.log` (and its copy `launch_leak_cfb.log`),
`rank_leak_cfb_100M_{none,gold}.{json,log}`, `rank_leak_cfb_final_{none,gold,swapped}.{json,log}`,
`compare_cf_vs_leak_cfb.log`, `compare_cfb_vs_leak_{m1,g1}.log`;
builds `build_leak_arms.sh`, `build_leak_arms.log`; the recorded warm-up arm's
`inject_retrieval_kv_cf_warm.log`, `inject_retrieval_kv_cf_main.log`,
`rank_retrieval_kv_cffinal_{none,gold,swapped}`, the original arm (c)'s `inject_retrieval_kv.log`,
`inject_retrieval_kv_resume.log`.

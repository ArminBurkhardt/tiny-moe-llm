# The expert count at pilot shape (2026-10-24): arm E4, 4 routed experts against arm P's 8

The expert half of rung R4b ([NEXT.md](../plans/NEXT.md), the R4b ladder row; run order decided
2026-10-23 by the user, R4b's expert arm first because it needs no build). Arm E4 is arm P's
recipe with the routed MLP experts halved from 8 to 4 at top 2, so 10M fewer parameters at the
same active compute. Read 2026-10-24 against arm P's final
([PILOT.md](../plans/PILOT.md), the "Run" section) on held-out language, copying, the leak, the
1-hop gate and L1, the port read and the load.

Halving the expert count costs nothing measurable at this shape and dose. Held-out CE ends 0.0075
above arm P and sits 0.001 to 0.013 above it from 50M on (0.019 at 25M), under the pilot's noise
of 0.013 at every read but two early ones; copying and the port read are equal; the 1-hop gate is
as high or higher (0.875 against 0.848); every composition cell is at chance on both; the leak is
equal at tier 100 on both forms and smaller at tier 1000 (paired raw rank z -3.8, about one sigma
on the prior-controlled `delta`).

## Run

- `bash ckpts/inject/launch_pilot_e4.sh`, config `ckpts/inject/config_pilot_e4.yaml`
  (`config_pilot.yaml` with `num_mlp_experts: 4`, top 2 as before, and `moe_balancing:
  loss_free`), seed `ckpts/inject/seed_pilot_e4_grounded.pt` (`init_scratch_seed.py` seed 0, then
  `migrate_groundedness_head.py`): 89.6M total, 84.9M active, against arm P's 99.1M / 84.9M, so
  the same active compute and FLOP per token.
- Arm P's recipe: the copy-first warm-up on `pilot_warm_train` to the 300M save, the switch read,
  then `SPLIT=pilot_main_train bash ckpts/inject/launch_pilot_e4.sh` to the epoch's end. Launched
  2026-10-23 17:09; stopped after the 300M save by `stop_after_save.sh e4 '30[0-4]M'` for the
  switch read; relaunched on the main split from the 301M save. Killed once at about 680M on the
  user's request (the evening of 2026-10-23) and resumed on 2026-10-24 from the 652M save, which
  replays the same documents from the saved position: the eval at about 677M reproduced the first
  pass's 678.69M eval within 0.001 (3.3142 against 3.3149).
- Complete 2026-10-24 at 858.40M tokens, one epoch, exit 0 (arm P 860.61M; the trainer stops at
  the epoch's end, so both cosines stop near LR 6e-5). 44k to 48k tokens per second, peak 26.4 GB
  for most of the run and 26.9 GB at the most (arm P 42k to 48k).
- Run directory `ckpts/evidence_pilot_e4/`, final save
  `ckpts/evidence_pilot_e4/checkpoint_evidence_final.pt`, log `ckpts/inject/pilot_e4.log`, reads by
  `bash ckpts/inject/read_pilot_save.sh CKPT e4 TAG` (which loads the e4 yaml), read log
  `ckpts/inject/pilot_e4_reads.log`.

## Held-out CE (`inject_val`, 40 batches, 1,190,582 supervised tokens)

The trainer's eval every 25M tokens. Both logs hold the same step numbers and token counts
through step 14130 (401M), so those rows match by step. After that arm P's stop, read and
relaunch cycles shift its schedule (each relaunch resumes from the last save), so the rows from
426M on are matched by the nearest token count, arm P 0 to 4M later than E4; late in the cosine
the CE falls about 0.009 per 25M, so a 4M lead favours arm P by about 0.0015. E4's log holds two
launches and one resume; the resumed rows are used from 677M on. No eval ran at 300M on either arm
(the switch stop).

| E4 tokens | E4 step | E4 CE | arm P tokens | arm P step | arm P CE | E4 minus arm P |
|---|---|---|---|---|---|---|
| 25.27M | 890 | 5.7735 | 25.27M | 890 | 5.7549 | +0.0186 |
| 50.28M | 1770 | 5.0526 | 50.28M | 1770 | 5.0448 | +0.0078 |
| 75.50M | 2660 | 4.4687 | 75.50M | 2660 | 4.4574 | +0.0113 |
| 100.77M | 3550 | 4.2210 | 100.77M | 3550 | 4.2115 | +0.0095 |
| 126.05M | 4440 | 4.0766 | 126.05M | 4440 | 4.0668 | +0.0098 |
| 151.09M | 5320 | 3.9644 | 151.09M | 5320 | 3.9621 | +0.0023 |
| 176.10M | 6200 | 3.8980 | 176.10M | 6200 | 3.8867 | +0.0113 |
| 201.30M | 7090 | 3.8207 | 201.30M | 7090 | 3.8196 | +0.0011 |
| 226.36M | 7970 | 3.7726 | 226.36M | 7970 | 3.7655 | +0.0071 |
| 251.37M | 8850 | 3.7350 | 251.37M | 8850 | 3.7233 | +0.0117 |
| 276.37M | 9730 | 3.6946 | 276.37M | 9730 | 3.6812 | +0.0134 |
| 326.11M | 11480 | 3.6177 | 326.11M | 11480 | 3.6086 | +0.0091 |
| 351.35M | 12370 | 3.5890 | 351.35M | 12370 | 3.5812 | +0.0078 |
| 376.36M | 13250 | 3.5539 | 376.36M | 13250 | 3.5480 | +0.0059 |
| 401.39M | 14130 | 3.5273 | 401.41M | 14130 | 3.5233 | +0.0040 |
| 426.67M | 15020 | 3.5048 | 426.41M | 15010 | 3.4965 | +0.0083 |
| 451.84M | 15910 | 3.4798 | 455.04M | 16020 | 3.4731 | +0.0067 |
| 477.12M | 16800 | 3.4613 | 480.05M | 16900 | 3.4531 | +0.0082 |
| 502.14M | 17680 | 3.4370 | 505.08M | 17780 | 3.4246 | +0.0124 |
| 527.39M | 18570 | 3.4152 | 530.31M | 18670 | 3.4101 | +0.0051 |
| 552.63M | 19460 | 3.3959 | 555.86M | 19570 | 3.3864 | +0.0095 |
| 577.89M | 20350 | 3.3794 | 581.14M | 20460 | 3.3672 | +0.0122 |
| 603.15M | 21240 | 3.3595 | 606.39M | 21350 | 3.3512 | +0.0083 |
| 628.40M | 22130 | 3.3423 | - | - | - | - |
| 653.67M | 23020 | 3.3286 | 656.37M | 23110 | 3.3176 | +0.0110 |
| 677.31M | 23850 | 3.3142 | 681.37M | 23990 | 3.3051 | +0.0091 |
| 702.56M | 24740 | 3.3013 | 706.63M | 24880 | 3.2925 | +0.0088 |
| 727.83M | 25630 | 3.2899 | 731.91M | 25770 | 3.2773 | +0.0126 |
| 752.84M | 26510 | 3.2739 | 756.96M | 26650 | 3.2649 | +0.0090 |
| 778.08M | 27400 | 3.2665 | 782.17M | 27540 | 3.2555 | +0.0110 |
| 803.32M | 28290 | 3.2559 | 807.40M | 28430 | 3.2469 | +0.0090 |
| 828.53M | 29180 | 3.2473 | 832.64M | 29320 | 3.2371 | +0.0102 |
| 853.80M | 30070 | 3.2404 | 857.66M | 30200 | 3.2313 | +0.0091 |
| 858.30M (final) | 30234 | 3.2396 | 860.48M (final) | 30307 | 3.2321 | +0.0075 |

E4 is above arm P at every read, by 0.001 to 0.019: over 0.013 at 25M (0.019) and at 276M
(0.0134, at the edge), under it at every other read, about 0.009 through the cosine's tail
(0.007 to 0.008 after the token lead is taken out), 0.0075 at the final. One sign throughout, so
four experts are probably a little worse on language, by about half the pilot's noise and with
no growth over the epoch. Arm P's eval at about 628M is missing from its log (a stop window).

## The switch read at 300M (`pilot_e4_300M_*`)

- Copying: gold card top-1 1.000 at every tier in distribution; held out 0.966 / 0.961 / 0.967 /
  0.943 by tier 1 / 10 / 100 / 1000 (arm P at its 300M save 0.983 / 0.980 / 0.977 / 0.953).
- Closed-book entity `delta`, in distribution 0.0076 (z 1.8) / 0.0021 (0.5) / -0.0019 (-0.5) /
  -0.0035 (-0.4), held out -0.0007 / 0.0023 / -0.0019 / 0.0142 (z 2.0): within 3 sigma in every
  cell, so the switch criterion is met, as on arm P.
- `chains_eval` depth 3, 1-hop 0.311 with three sites kept (arm P 0.32 at 300M), composition at
  chance.

## The final reads (`pilot_e4_final_*`)

### Copying

| read | E4 in distribution | E4 held out (tier 1 / 10 / 100 / 1000) | arm P final |
|---|---|---|---|
| gold card top-1 | 1.000 at every tier | 0.959 / 0.963 / 0.967 / 0.973 | 1.000 / 0.95 held out |
| swapped card followed | 1.000 at every tier | 0.914 / 0.916 / 0.914 / 0.934 | 1.000 / 0.94 to 0.97 held out |
| `mr_ll` | 0.000 | under 0.01 | 0.000 / under 0.01 |
| prompt copy, entity top-1 | 0.999 to 1.000 | 0.940 / 0.933 / 0.933 / 0.957 | |

Equal in distribution; held out the gold card is copied as well or slightly better and the swapped
card followed slightly less (0.91 to 0.93 against 0.94 to 0.97), one seed each.

### The leak (closed-book entity `delta`, `--evidence none`)

| tier | E4 in distribution | arm P in distribution | E4 held out | arm P held out |
|---|---|---|---|---|
| 1 | 0.0076 (z 1.8) | 0.0077 (z 2.0) | -0.0007 (z -0.2) | 0.0068 (z 1.7) |
| 10 | 0.0017 (z 0.4) | 0.0054 (z 1.3) | 0.0023 (z 0.7) | -0.0022 (z -0.6) |
| 100 | 0.0170 (z 3.9) | 0.0096 (z 2.2) | 0.0120 (z 3.2) | 0.0125 (z 3.2) |
| 1000 | 0.0534 (z 5.3) | 0.0639 (z 7.2) | 0.0413 (z 5.0) | 0.0458 (z 5.5) |

Paired (`closed_book_rank.py compare`, `ckpts/inject/compare_pilot_e4_vs_p.log`): arm P minus E4
on `norm_rank`, every class and both forms pooled, negative means arm P ranks the real value
better, that is stores more:

| tier | n | diff | sigma | z |
|---|---|---|---|---|
| 1 | 5,000 | -0.0076 | 0.0038 | -2.0 |
| 10 | 5,000 | -0.0001 | 0.0038 | 0.0 |
| 100 | 5,000 | -0.0047 | 0.0039 | -1.2 |
| 1000 | 1,000 | -0.0320 | 0.0084 | -3.8 |
| all | 16,000 | -0.0059 | 0.0021 | -2.8 |

At tier 100 the leak is equal: the in-distribution cell is higher on E4 by 0.007 (inside 2 sigma
of the difference), the held-out cell is the same, and the paired rank is z -1.2. At tier 1000 E4
ranks the real value worse by 0.032 (z -3.8 paired). The paired read is on the raw rank; the
compare's pooled tables show E4's prior (the rank without the name) also worse at tier 1000
(entities 0.471 against 0.454), so on the prior-controlled `delta` the reduction is 0.005 to
0.011 per form (0.0534 against 0.0639, 0.0413 against 0.0458), about one sigma each. Read: the
arm with fewer parameters stores the frequent facts no more and probably less; most of the
3.8 sigma is the frequent names' candidates ranking worse with and without the name.

### The chain read (2,000 questions per split, every read site)

| split, hops (chance) | kept 0 | kept 1 | kept 2 | kept 3 | arm P kept 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|
| `chains_eval` depth 3, 1-hop (0.115) | 0.111 | 0.871 | 0.876 | 0.875 | 0.116 / 0.844 / 0.849 / 0.848 |
| `chains_eval` depth 3, 2-hop (0.185) | 0.193 | 0.188 | 0.196 | 0.200 | at chance |
| `chains_eval` depth 3, 3-hop (0.206) | 0.214 | 0.214 | 0.205 | 0.213 | at chance |
| `chains_heldout_tmpl` depth 3, 2-hop (0.191) | 0.184 | 0.184 | 0.195 | 0.192 | at chance |
| `chains_hop4` depth 4, 4-hop (0.210) | 0.208 | 0.194 | 0.197 | 0.197 | at chance |

The 1-hop gate holds (0.875, 3 points over arm P's 0.848, about 2.5 sigma unpaired at n 2,000, one
seed each) and
every composition cell is at chance, so L1 fails on this arm as on arm P.

### The port read (the final `[eval fixed]`)

E4 gold gain +4.19, content gain +3.82, grounded AUROC 0.972 on all rows (0.944 on evidence rows);
arm P +4.00, +3.97, 0.994 (0.988). By source on E4: SQuAD v2 gold +4.02, content +3.95; HotpotQA
gold +4.35, content +3.71. Chunk AUROC per token 0.75 to 0.76 by pass (arm P 0.78). Equal within
what one seed resolves: E4 a little higher on gold, a little lower on content and groundedness.

### Load and loop scales

Under loss-free balancing at the end of the run: MLP load max over mean 1.01 to 1.06 per pass,
dense share 0.33 / 0.60 / 0.47 by pass (the share of top-k picks on the attention and IR slots;
an even split over 3 dense and 4 MLP slots would be 3/7 = 0.43). Arm P's log has no load fields
(they were added 2026-10-20). `loop_scale` at the end 1.219 / 0.488 / 0.200 (arm P 1.289 /
0.566 / 0.258).

## Verdict

**Halving the routed expert count from 8 to 4 at top 2 (10M fewer parameters, the same active
compute) costs nothing measurable at this shape and dose.** Language is within noise (E4 0.0075
above at the final, one sign throughout, about half the pilot's noise); copying and the port read
are equal; the 1-hop gate is as high or higher; composition is at chance on both; the leak is
equal at tier 100 and smaller at tier 1000 (the arm with fewer parameters stores fewer frequent
facts, by 3.8 sigma on the paired raw rank and about one sigma on the prior-controlled `delta`).

Caveat on record: E4 ran under loss-free balancing and arm P under the aux term that bf16
saturation had made a constant, so the comparison carries two changes. At micro scale loss-free
alone moved CE and storage inside the seed spread ([lossfree_micro.md](lossfree_micro.md)), so
the difference is read as the expert count with that caveat. One seed each; the pilot has no
training-seed spread of its own.

What it decides for shape L: the expert count is not where this design's capacity binds at 1B
tokens and 99M parameters, so the real run's expert count is a cost choice, not a quality one,
until a read at the real run's size says otherwise. The sublayer half of R4b stays open (the
three-sublayer core is unbuilt, its build the user's choice).

## Raw files

- Log `ckpts/inject/pilot_e4.log` (two launches and one resume), read log
  `ckpts/inject/pilot_e4_reads.log`; arm P's log `ckpts/inject/pilot_p.log`.
- Switch reads `ckpts/inject/pilot_e4_300M_{rank_none,rank_gold,rank_swapped,rank_prompt,chains_d123,chains_hop4}.{json,log}`.
- Final reads `ckpts/inject/pilot_e4_final_{rank_none,rank_gold,rank_swapped,rank_prompt,chains_d123,chains_hop4}.{json,log}`.
- Paired compare `ckpts/inject/compare_pilot_e4_vs_p.log` (a = arm P final, b = E4 final).
- Saves `ckpts/evidence_pilot_e4/` (50M steps to 852M kept, plus the final).

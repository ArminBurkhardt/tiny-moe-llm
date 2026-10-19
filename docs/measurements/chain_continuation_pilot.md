# The chain continuation of arm P (2026-10-17): one pass over the chain split, read for the two-hop transition

Decided 2026-10-17 (the user, PILOT.md "Status"): L1 is recorded as failed at the pilot's shape
unless this continuation shows composition. It is the chain-overtraining direction of
[review_2026-10-16.md](../review_2026-10-16.md) at its cheapest: arm P's final save
(`ckpts/evidence_pilot_p/checkpoint_evidence_final.pt`, 860.61M tokens, 1-hop 0.85, every
composition cell at chance) trained one more epoch on `data/prepared/chains_train` alone (2M
fictional k-hop questions, 104.66M prompt tokens, 439M evidence tokens, the split the micro chain
arm used), the only slice of the merged corpus that asks for composition, at ten times its share
in the pilot's mix. The published in-context two-hop transition (the slow random-guess phase, then
an abrupt flip) is what the per-10M saves were read for.

## Run

- `ckpts/inject/launch_pilot_chains.sh`, config `ckpts/inject/config_pilot_chains.yaml` (from
  `config_pilot.yaml`: lr and fresh lr 3e-4 with a 2% warm-up and the cosine fitted to the
  split, saves and evals every 10M, `fixed_split` off), `sft.py --evidence --reader-kv -c
  <arm P final> --train-split chains_train --val-split chains_val`, run directory
  `ckpts/evidence_pilot_chains/`, log `ckpts/inject/pilot_chains.log`.
- Batch 8 x 4096 spilled into shared memory in minute one (32.0 GB, 1.1k tokens per second: every
  chain row closes on the 4608 evidence cap, so the evidence axis is full on every row); relaunched
  at batch 4 with accumulation 2 (the same tokens per update): 16.1 GB peak, flat for the whole
  pass, 4.3k to 9k prompt tokens per second by window (74 to 80 seconds per 100 steps, the token
  rate moving with the rows' prompt length, not the compute).
- Complete at 102.66M tokens (the epoch's end; the micro chain arm ended at 102.66M on the same
  split) in 307.9 minutes, exit 0; saves at 10M to 100M plus the final, 24,439 steps.

## The trainer's series (held-out `chains_val`, 25 batches, 14,647 supervised tokens)

| save | step | answer CE | top-1 per token | selection |
|---|---|---|---|---|
| arm P final (step 0) | 0 | 0.315 | 0.906 | 0.372 |
| 10M | 2,380 | 0.336 | 0.901 | 0.392 |
| 20M | 4,760 | 0.290 | 0.905 | 0.374 |
| 30M | 7,150 | 0.271 | 0.906 | 0.359 |
| 40M | 9,540 | 0.244 | 0.910 | 0.348 |
| 50M | 11,930 | 0.232 | 0.911 | 0.338 |
| 60M | 14,320 | 0.220 | 0.912 | 0.331 |
| 70M | 16,700 | 0.215 | 0.912 | 0.323 |
| 80M | 19,080 | 0.208 | 0.912 | 0.318 |
| 90M | 21,470 | 0.205 | 0.913 | 0.314 |
| 100M | 23,860 | 0.202 | 0.912 | 0.311 |
| final | 24,439 | 0.201 | 0.913 | 0.311 |

A smooth decay with no break: the CE falls 0.135 nats over the pass while top-1 per token moves
0.906 to 0.913, which is the model growing more confident on the tokens it already had right. On
the training rows the per-loop CE is flat across passes in most windows (pass 3 led in a run of
five windows around steps 10k to 13k and fell back), the loop scales go [1.289, 0.566, 0.258] to
[1.227, 0.486, 0.272] (the third pass dips to 0.183 at step 4k and recovers, the opposite of the
micro arm's collapse under 0.01), the per-loop evidence gain splits to [1.85, 1.91, 1.49] nats with
pass 3 reading the port least, and the selector's external mass climbs 0.22 to 0.39 while its
held-out selection loss flattens at 0.31: the model reads the chunks more, not better.

## The chain read (`eval_chains.py`, 2,000 questions per split, batch 256, every read site)

Final save (`ckpts/inject/pilot_ch_final_chains_{d123,hop4}.*`), against arm P's final in
brackets:

- **1-hop lookup is saturated**: `chains_eval` 1-hop 0.991 to 0.993 at every depth with one
  site kept (chance 0.115, z +87; arm P final 0.85), the gold answer at 0.04 to 0.07 nats per
  answer against ln K 2.20 (arm P 0.82). The instrument reads valid (no sub-hop cell outside
  3 sigma of chance).
- **Nothing composes.** `chains_eval` 2-hop 0.185 to 0.198 at every depth and site count
  (chance 0.185, largest z +1.1); 3-hop 0.171 to 0.212 (chance 0.206, z -2.7 to +0.5);
  `chains_hop4` 0.195 to 0.223 at depths 1 to 4 (chance 0.210, |z| at most 1.1; depth 4 with
  every site 0.222, answer CE 0.255 at depths 3 and 4 alike); the held-out 2-hop composition
  0.190 to 0.210 (chance 0.191, z at most +2.2 at depth 1 kept 1, a single cell), paired Delta
  (kept 3 minus kept 1, depth 3) held out +0.004 +- 0.008 (n 2,000; arm P +0.002 +- 0.005), seen
  -0.011 +- 0.012. Delta + 2 sigma 0.020 against the 0.05 bar: the spec's "falsified" cell, now
  with lookup at 0.99.
- **The composed answers are still an answer-type pick.** Gold NLL per answer at 2 hops 1.93
  (depth 3, every site) against ln K 1.72, at 3 hops 1.85 against 1.62, held-out 2-hop 1.90
  against 1.68: above a uniform pick among the type-matched candidates in every composition cell
  (arm P final 3.16 at 2 hops, so the pass bought about 1.2 nats and stopped at uniform). Depth
  lowers it by 0.19 nats from depth 1 to 3 (2.12 to 1.93) without the pick getting right.
  Without a read site an answer costs 59 to 63 nats (arm P 40), so the read carries the content
  and the model copies an entity of the right type and does not select which.

The other saves, depth 3 with every site kept (`chains_eval` 1-hop / 2-hop / 3-hop, the held-out
2-hop, `chains_hop4` 4-hop, then the paired held-out Delta), the decisive ones first and the
rest after the retriever micro arm, which has the GPU next:

| save | 1-hop | 2-hop (0.185) | 3-hop (0.206) | held out (0.191) | 4-hop (0.210) | Delta |
|---|---|---|---|---|---|---|
| 10M | 0.819 | 0.202 | 0.199 | 0.180 | 0.202 | -0.007 +- 0.006 |
| 20M | 0.911 | 0.190 | 0.195 | 0.204 | 0.211 | +0.002 +- 0.006 |
| 30M | 0.932 | 0.185 | 0.194 | 0.196 | 0.214 | +0.005 +- 0.007 |
| 40M | 0.951 | 0.199 | 0.193 | 0.205 | 0.207 | +0.003 +- 0.007 |
| 50M | 0.966 | 0.180 | 0.217 | 0.211 (z 2.3) | 0.201 | +0.002 +- 0.007 |
| 60M | 0.977 | 0.192 | 0.192 | 0.206 | 0.203 | +0.002 +- 0.008 |
| 70M | 0.982 | 0.183 | 0.199 | 0.195 | 0.184 | +0.002 +- 0.008 |
| 80M | 0.985 | 0.200 | 0.202 | 0.202 | 0.189 | -0.004 +- 0.008 |
| 90M | 0.989 | 0.205 | 0.192 | 0.189 | 0.210 | -0.002 +- 0.009 |
| 100M | 0.992 | 0.191 | 0.181 | 0.216 (z 2.7) | 0.225 | +0.003 +- 0.009 |
| final | 0.991 | 0.185 | 0.180 | 0.201 | 0.215 | +0.004 +- 0.008 |

Every save is read (the remaining eight on 2026-10-18 and 2026-10-19, about 5 minutes each).
The 1-hop column saturates smoothly; the four composition columns stay at chance at all eleven
reads (the two held-out cells at z 2.3 and 2.7 are what 44 chance cells produce, and the
paired Delta next to them is within one sigma), so there is no transition anywhere inside the
epoch, not only at its end.

## Verdict

**L1 is failed at this shape** (the user's 2026-10-17 rule: failed unless the continuation shows
composition). One more epoch of 2M chain questions on top of the pilot, with lookup saturated and
the loop scales healthy, leaves every composition cell at chance and the composed answer at a
uniform pick among candidates. The published transition did not come at this dose: the slow phase
extends through 2.1 epochs of the chain split at 99M parameters. The decision table's
"falsified" row applies: the loop clause is recorded as falsified at this shape, looping stays a
requirement, the goal's reasoning clause is reworded before the real-run spec, and the blueprint
items return as arms. What this read does not say: whether a larger shape, the identity-bridge
(zero-hop) examples, the sublayer read sites or a written bridge (R9) compose; those are the open
directions of the research round, and R9 is the next one queued on this question.

## What was not read

Closed-book rank on these saves (the chain split carries no biographies, so the leak can only
decay here; `read_pilot_save.sh CKPT p TAG 0` reads it if wanted), the fixed-split gains (off in
the config), and the saves 20M to 40M and 60M to 100M by the chain instrument until the GPU is
free (the series above shows no break to look for).

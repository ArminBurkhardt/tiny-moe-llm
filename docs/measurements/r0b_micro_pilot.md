# Fact injection micro-pilot: arms (a), (b) and (c), the arm (c) readers, the loop reading, the per-exit read (2026-10-02 to 10-05)

The first from-scratch runs. Model: `config_micro.yaml`, 35.5M parameters (12.6M outside the
embeddings), random init from `init_scratch_seed.py --seed 0`. Corpus: `data/prepared_inject`, 300M
tokens per arm, 3,600 fictional people at 1 / 10 / 100 / 1000 exposures (4.6% of tokens), filler
from the local `ir`, `phase1`, `phase2` bins repeated 1.06x. Reader: `closed_book_rank.py bios`,
100 same-type candidates, every item paired with a fresh-name prior control; `delta = prior -
real`, positive means the name helps. Raw output under `ckpts/inject/` (gitignored).

## Throughput

`config_micro.yaml` now runs `batch_size: 32`, `grad_accumulation_steps: 1` (was 16 x 2; the same
~32k tokens per optimizer step). 45k to about 57k tok/s, 10.5 GB peak without evidence, 13.2 GB with
the arm (c) card buffer. The run is GPU bound (90% utilization, 427 W): at hidden width this small
the kernels are inefficient, not the host. Arm (a) took 90.1 minutes for 299.4M tokens.

`closed_book_rank.py` needs `--batch-size 1024` on micro checkpoints: the rows are 20 to 40 tokens,
so the default 64 is launch bound (over an hour per checkpoint against a few minutes).

## Arm (a), full CE, no retrieval: the instrument is readable

Entity class, `delta` with its paired z:

| save | form | tier 1 | tier 10 | tier 100 | tier 1000 |
|---|---|---|---|---|---|
| 50M | indist | -0.004 (-2.1) | 0.003 (1.5) | 0.010 (4.4) | 0.087 (13.7) |
| 50M | heldout | -0.003 (-1.3) | 0.005 (2.5) | 0.003 (1.6) | 0.053 (10.7) |
| final | indist | 0.009 (1.0) | 0.063 (7.6) | 0.378 (51.4) | 0.467 (27.4), rank 0, top-1 1.000 |
| final | heldout | 0.001 (0.3) | 0.009 (2.5) | 0.023 (6.4) | 0.104 (10.2) |

- The pass criterion needs (a) 3 sigma above zero through tier 100 while (b) and (c) stay within 3
  sigma of zero. (a) clears it in distribution from tier 10 up and on held-out templates at tier 100
  and 1000, so the instrument reads.
- Memorization is mostly surface form: tier 1000 is perfect on the five training templates and
  moves only 0.10 on a held-out template; held-out nouns barely move (0.036 at tier 1000, 2.6
  sigma). Read (b) and (c) on both forms; the in-distribution form is the sharper leak detector.
- The prior control ranks 0.24 to 0.40 at tier 1000, not 0.5: corpus value frequency follows
  exposure. Only `delta` is a measurement.

Loop scale went from equal (0.578) to [1.04, 0.64, 0.16] by 37M tokens, with loop 3's CE equal to
loop 2's, then recovered to [1.61, 0.63, 0.21] by 275M with loop 3 about 0.05 nats under loop 2.

## Arm (c) smoke, 20M tokens: the from-scratch reader does not read

`inject_retrieval`, stopped by `STOP` at 20.1M tokens (2.0M evidence tokens),
`checkpoint_evidence_tok20M_loss10.0385.pt`. Entity class, `norm_rank`:

| form | evidence | tier 1 | tier 10 | tier 100 | tier 1000 |
|---|---|---|---|---|---|
| indist | gold card | 0.492 | 0.502 | 0.465 | 0.265 |
| indist | none | 0.494 | 0.499 | 0.483 | 0.393 |
| indist | none, prior | 0.494 | 0.500 | 0.482 | 0.397 |
| heldout | gold card | 0.491 | 0.506 | 0.466 | 0.285 |
| heldout | none | 0.497 | 0.503 | 0.484 | 0.406 |

- With the gold card attached, tiers 1 to 100 rank at chance. A reader that copies does not care how
  often a person was seen, so this is the "near 0.5" outcome: the reader design is what failed at
  this budget.
- Tier 1000 moves (0.39 to 0.27 with the card) while closed-book `delta` stays at zero (z 1.6). The
  card helps only for people seen a thousand times: an association learned through the card
  pathway, not a copy.
- Closed-book `delta` is within 2 sigma of zero at every tier and class. At 20M that says little
  (arm (a) needed tens of millions of tokens to move tier 1000), but nothing leaked yet.
- Training signals agreed: `selection` swung 0.43 to 0.58 over the last 400 steps and the external
  mass with gold present (0.008 to 0.013) never separated from distractors only (0.006 to 0.012).
  `|shared_evidence.o_proj|rms` grew steadily to 2.25e-2, so the port trains; it does not select.

## The copy control, and the key/value reader at 100M (2026-10-03)

The 20M verdict above was read before the model could copy at all, so it does not judge the reader.
Two instruments were added and two runs made, matched in tokens:

- `closed_book_rank.py bios --evidence prompt`: the same card text as plain prompt text before the
  probe, no port. The in-context copy control.
- `sft.py --evidence --reader-kv`: the key/value reader. No separate cross attention: the evidence
  states become leading keys and values of `shared_attn` in every loop, through its own norm,
  `k_proj` and `v_proj`, under one softmax with the segment's causal keys, rotated as if the
  evidence were the text right before the document. `evidence_loop_scale` scales the evidence
  values per loop; the chunk gate is the same (frozen at 0 in the micro profile). Seeded from the
  same `seed_micro.pt` with the cross reader's 11 tensors dropped, so the trunk init is identical.
- Arm (c) with the cross reader was resumed from the 20M save to 100M as the matched control
  (`checkpoint_evidence_tok100M_loss8.4518.pt`); the kv run stopped at 100.4M
  (`ckpts/evidence_inject_retrieval_kv/checkpoint_evidence_tok100M_loss7.7670.pt`). Same data, flags
  and LR schedule (both stopped a third of the way down the 300M cosine).

Entity class, `norm_rank` by tier 1 / 10 / 100 / 1000 (lower is better; chance 0.5, the prior
control ranks 0.33 to 0.44 at tier 1000):

| save | form | none | gold card (port) | card in prompt | swapped card: follow |
|---|---|---|---|---|---|
| cross, 20M | indist | .494 .499 .483 .393 | .492 .502 .465 .265 | .491 .496 .480 .394 | .009 .010 .012 .014 |
| cross, 20M | heldout | .497 .503 .484 .406 | .491 .506 .466 .285 | .494 .503 .481 .404 | .008 .008 .007 .016 |
| cross, 100M | indist | .493 .491 .438 .193 | .472 .463 .350 .019 | .131 .118 .103 .052 | .017 .020 .020 .006 |
| cross, 100M | heldout | .496 .497 .461 .279 | .486 .476 .407 .107 | .101 .088 .080 .046 | .012 .010 .012 .014 |
| **kv, 100M** | indist | .489 .499 .454 .257 | **.003 .003 .002 .000** | .003 .002 .001 .000 | **.925 .920 .913 .880** |
| **kv, 100M** | heldout | .492 .499 .481 .360 | **.023 .020 .017 .008** | .028 .023 .023 .012 | .702 .699 .689 .668 |
| arm (a) final | indist | | | .287 .264 .122 .003 | |
| arm (a) final | heldout | | | .197 .192 .191 .201 | |

`mr_ll` under the swapped card: cross 100M 0.48 / 0.48 / 0.58 / 0.94 (it keeps its memory), kv 100M
0.005 / 0.004 / 0.006 / 0.028 in distribution and 0.02 to 0.05 held out (it takes the card).

- **At 20M nothing copies, in context or through the port.** The prompt read equals the closed-book
  read for entities at every tier; only dates move (0.38 against 0.49). The 20M smoke was read before
  any copy circuit existed and says nothing about either reader.
- **At 100M the key/value reader copies from the card as well as the model copies from its own
  prompt**: top-1 0.92 in distribution at every tier, 0.6 held out, flat in exposure, and the
  swapped card is followed 92% of the time with a memorization ratio under 0.03. That is the A3
  shape (follow at least 0.9, ratio at most 0.05) on the in-distribution probe already.
- **The cross reader does not copy at 100M**, although the same model copies from its prompt
  (0.10 to 0.13): with the card in the port it ranks tier 1 at 0.47 and follows a swapped card 2%
  of the time; its tier-1000 gain (0.019) is memory cued by the card, since the swapped card still
  ranks the original value at 0.023. The reader design, not the budget, is what failed.
- **Training the key/value reader also trained in-context copying** (prompt 0.003 against the cross
  model's 0.13 at the same tokens): the read and the prompt share the circuit, as intended.
- The selector separated in both runs by 100M without the reader's help (the gate is frozen):
  external mass with gold present 0.39 to 0.46 against 0.02 to 0.12 with distractors only,
  `selection` 0.14 to 0.17. Filler validation CE is the same (kv 4.481 at 100M, cross 4.509 at
  96M), so the read costs the language model nothing.

### Arm (c) leaks at 100M, with either reader

Closed-book `delta` (prior minus real, entity class, z in brackets) at matched 100M tokens, arm (a)
being `checkpoint_evidence_tok100M_loss7.7188.pt`:

| arm | form | tier 1 | tier 10 | tier 100 | tier 1000 |
|---|---|---|---|---|---|
| (a) full CE | indist | 0.007 (1.3) | 0.012 (2.1) | 0.099 (17.3) | 0.332 (24.0) |
| (c) cross | indist | -0.006 (-1.7) | 0.003 (0.9) | 0.037 (10.5) | 0.203 (20.3) |
| (c) kv | indist | 0.001 (0.1) | -0.000 (-0.0) | 0.026 (5.2) | 0.149 (12.3) |
| (a) full CE | heldout | -0.001 (-0.3) | 0.003 (1.0) | 0.018 (5.0) | 0.099 (10.4) |
| (c) cross | heldout | -0.002 (-0.6) | 0.005 (1.6) | 0.025 (8.5) | 0.139 (18.0) |
| (c) kv | heldout | -0.010 (-2.4) | 0.002 (0.6) | 0.006 (1.6) | 0.077 (8.1) |

- Arm (c) is 3 sigma off its prior at tier 100 in distribution under both readers, so at 100M it
  already fails the pass rule; the leak is 25 to 40% of arm (a)'s at tier 100 and 45 to 60% at
  tier 1000 in distribution. The reader that copies leaks least: the kv arm sits under the cross
  arm at tiers 100 and 1000 on both forms, the cue the plan predicts (a working read takes away the pressure to
  memorize), but not under the bar.
- Held out, the cross arm leaks more than arm (a). The card form ("Name. Born ... in City.") is
  closer to the held-out question than the biography templates are; the card trains the binding
  in a form the held-out probe reads.
- These are mid-run readings (a third of the cosine). Arm (c) in full and arm (b) decide the
  verdict; the plan's lever for "only (c) climbs" is the swap rate. Corrected 2026-10-05: arm (c)
  in full does not climb after 100M; it fails on a leak that is already there at 50M, before the
  reader copies, and stays flat after 100M (below). The warm-up arm (below) reads it as both: part
  written before copying, part a level that training maintains while real values are supervised.
  The swap rate sweep was set aside and then reopened as a candidate on the same day.

Raw output: `ckpts/inject/rank_retrieval20M_{prompt,swapped}`, `rank_full_prompt`,
`rank_retrieval100M_{gold,none,prompt,swapped}`, `rank_retrieval_kv100M_{gold,none,prompt,swapped}`,
`rank_full_100M` (`.json` and `.log`); training logs `inject_retrieval_kv.log`,
`inject_retrieval_resume.log`.

## The loop reading on arm (a): a weakly stored fact is recalled better after two passes (2026-10-04)

The storage half of the loop axiom (NEXT.md Decisions). `closed_book_rank.py bios --n-loops N` runs
the block N times and reads the last loop run; loop k never depends on the total depth, so this is
the readout a training step at sampled depth N supervised (`loop_count_sampling` 0.3,
`loop_ce_weights` 0.2 / 0.3 / 1.0). `--n-loops 3` reproduces the default read item for item on an
800-item smoke. Arm
(a) final, closed book, `delta` at depth 1 / 2 / 3 with the paired difference between depths (same
items, bootstrap sigma of the difference of deltas, z in brackets):

| class, form | tier | depth 1 | depth 2 | depth 3 | 1 to 2 | 2 to 3 |
|---|---|---|---|---|---|---|
| entity, indist | 1 | 0.006 | 0.008 | 0.009 | +0.002 (0.3) | +0.001 (0.8) |
| entity, indist | 10 | 0.047 | 0.063 | 0.063 | +0.016 (3.1) | -0.001 (-0.4) |
| entity, indist | 100 | 0.269 | 0.369 | 0.378 | +0.100 (19.7) | +0.008 (6.5) |
| entity, indist | 1000 | 0.393 | 0.453 | 0.467 | +0.059 (6.4) | +0.014 (5.0) |
| date, indist | 100 | 0.221 | 0.312 | 0.318 | +0.090 (11.8) | +0.006 (2.9) |
| date, indist | 1000 | 0.249 | 0.295 | 0.306 | +0.046 (3.1) | +0.011 (2.9) |
| noun, indist | 100 | 0.362 | 0.447 | 0.450 | +0.085 (9.6) | +0.003 (1.2) |
| noun, indist | 1000 | 0.477 | 0.521 | 0.525 | +0.045 (2.3) | +0.004 (0.8) |
| entity, heldout | 100 | 0.029 | 0.024 | 0.023 | -0.004 (-2.8) | -0.001 (-1.8) |
| entity, heldout | 1000 | 0.107 | 0.105 | 0.104 | -0.002 (-0.6) | -0.001 (-0.8) |

Entity class in distribution, tier 100: top-1 0.141 / 0.388 / 0.415; summed gold log-probability
-4.47 / -3.26 / -3.15 nats for the real name against -6.71 / -6.57 / -6.64 for the fresh-name prior.
Tier 1000: top-1 0.980 / 1.000 / 1.000, gold log-probability -0.92 / -0.20 / -0.16, real-name
`norm_rank` 0.0008 / 0.0000 / 0.0000 against a prior of 0.394 / 0.453 / 0.467.

- **The falsifier's condition is met at tier 100, in every class.** The `delta` grows with depth
  beyond the paired sigma in distribution: +0.100 (z 19.7) for entities, +0.090 (z 11.8) for dates,
  +0.085 (z 9.6) for nouns from the first pass to the second. The third pass adds 2% of the total
  (+0.008, z 6.5, entities). The prior does not move at tier 100 (0.435 / 0.434 / 0.436), so this
  is the real name ranking better.
- **The tier 1000 rows are not recall.** The real name is saturated after one pass (rank 0.0008,
  top-1 0.98), and the growth in `delta` (+0.059, +0.014) is the fresh-name prior drifting toward
  chance (0.394 / 0.453 / 0.467); the same holds for dates and nouns, whose real rank is 0 at every
  depth. Tier 1000 says a fact seen a thousand times is fully readable after one pass.
- **At tier 100 the gain is specific to the stored fact.** From depth 1 to 2 the real name gains
  1.21 nats on the answer (summed over about 5 tokens) while the fresh-name prior, same items and
  same units, gains 0.14. For scale, the general per-loop CE gap on the training stream is 0.09
  nats per token (3.96 / 3.87 / 3.86 over the last 200 log lines), so about 0.5 nats over an answer
  of that length, and 0.007 per token from loop 2 to 3.
- **Depth 1 reaches 71% of the depth 3 `delta` at tier 100.**
- **Held out, nothing grows with depth** (tier 100: 0.029 / 0.024 / 0.023, falling slightly, z -3.2
  from depth 1 to 3). The part of the memory that survives a change of template is fully readable
  after one pass; what the second pass adds is the surface-form recall.
- **What it does and does not say.** The block's weights are the same in every pass, so a second
  pass adds no capacity; it adds a second step of computation over the same weights. The reading is
  that recalling a weakly stored fact (100 exposures) is a two-step computation in this model, and
  that the one-pass exit is a worse reader of the same store. It does meet the falsifier's
  condition (the `delta` grows with depth beyond its paired sigma), so closed-book recall is not
  independent of depth here and a leak test has to be read at full depth, which is what every other
  read in this file does. Whether that makes the axiom "wrong for this design" or shows that the
  falsifier measured recall depth instead of storage is a plan decision, not settled by this read.
  Decided 2026-10-04: the axiom is reworded to "passes add computation, not capacity; recalling a
  weakly stored fact can take more than one pass" (NEXT.md Decisions).
- Confound that remains: the depth 1 and 2 exits are trained less (weights 0.2 and 0.3 on a quarter
  of the positions, plus 15% of steps each as the last pass). The prior control rules out a generic
  readout gain; whether the early exits recall stored facts worse because they are trained less
  needs an arm with equal CE weight on every exit and no subsampling (a last-loop-only arm would
  not do: its early exits are untrained and read at chance).

Raw output: `ckpts/inject/rank_full_loops{1,2}` and `rank_full` (depth 3).

## Arm (b), masked fact spans, no retrieval: nothing stored at 3 sigma (2026-10-04)

`inject_masked`, the same documents and order as arm (a) with the loss mask at 0 on every token of
an attribute value (1.56% of tokens against 0.19% in arm (a)); the facts are still in the context.
299.44M tokens in 86.8 minutes, final filler validation CE 3.877 (arm (a) 3.904), `loop_scale`
[1.63, 0.59, 0.18]. Saves in `ckpts/evidence_inject_masked/` (50M steps to final).

Closed-book `delta` with its paired z, final save:

| class | form | tier 1 | tier 10 | tier 100 | tier 1000 |
|---|---|---|---|---|---|
| entity | indist | 0.001 (0.5) | 0.003 (1.3) | 0.002 (0.6) | 0.011 (1.7) |
| entity | heldout | -0.001 (-0.4) | 0.001 (0.5) | -0.000 (-0.1) | 0.013 (2.4) |
| date | indist | -0.004 (-0.5) | 0.002 (0.3) | -0.009 (-1.1) | -0.010 (-0.6) |
| date | heldout | -0.007 (-1.4) | -0.004 (-0.9) | 0.001 (0.1) | -0.015 (-1.2) |
| noun | indist | 0.000 (0.1) | 0.004 (1.1) | 0.002 (0.4) | 0.002 (0.3) |
| noun | heldout | -0.003 (-0.7) | 0.000 (0.0) | 0.003 (0.7) | -0.020 (-2.4) |

At matched 100M tokens (`checkpoint_evidence_tok100M_loss7.7500.pt`), entity class: in distribution
-0.000 (-0.1) / 0.001 (0.6) / -0.003 (-1.1) / 0.001 (0.2), held out 0.001 (0.5) / 0.002 (0.6) /
0.001 (0.4) / 0.003 (0.5).

- **Arm (b) passes its half of the rule.** Every cell is within 3 sigma of the fresh-name prior, on
  both forms, at 100M and at 300M, where arm (a) reads 0.378 (z 51) at tier 100 and 0.467 at tier
  1000. Paired against arm (a) on the same items (`compare`, all classes and forms): norm rank
  difference -0.242 (z -47.4) at tier 100 and -0.311 (z -27.9) at tier 1000. A fact that is read
  but never predicted is not stored at any size this read resolves: the input side does not leak
  at the 3 sigma bar.
- **The tier 1000 entity cells are the ones to watch.** They are the largest and both positive
  (0.011 and 0.013, z 1.7 and 2.4), 0.012 at z 2.8 pooled over the two forms, and they rose from
  100M (0.001 and 0.003). The two forms share the same 100 people, so they are not independent
  evidence. Under the bar, against arm (a)'s 0.467 in distribution, and the cell to reread on any
  longer run.
- **The masked arm copies from its prompt** (`--evidence prompt`, entity `norm_rank` by tier): 0.153 /
  0.144 / 0.154 / 0.220 in distribution (top-1 0.29) and 0.051 / 0.055 / 0.053 / 0.094 held out
  (top-1 0.56): flat over tiers 1 to 100, worse at tier 1000 by about 4 sigma. Arm (a) reads 0.287 /
  0.264 / 0.122 / 0.003 and 0.19 held out: it answers the frequent people from memory and copies
  worse for the rest. The masked arm copies better than arm (a) held out and at tiers 1 and 10.
- **`compare` on the three arms at matched 100M** (full, masked, key/value arm (c)): the held-out
  form HOLDS, the in-distribution form FAILS on arm (c) at tier 100 (0.026, z 5.2), with tier 1000 at
  0.149 (z 12.3) read alongside. Corrected 2026-10-05: the held-out hold did not last. At 100M the
  held-out tier 100 cell was 0.006 (z 1.6), with tier 1000 already at 0.077 (z 8.1); from 150M on
  tier 100 sits 3.3 to 3.9 sigma off its prior, 0.013 (z 3.6) at the final save, so arm (c) in
  full fails on both forms. Since arm (b) is flat at the same tokens, the arm (c) leak is not
  the in-context input path. Most likely it comes from the spans arm (c) supervises, 81% of the
  fact tokens, with the real name in the prompt and the unswapped value as the target 85% of the
  time; arm (b) does not test the card path, which arm (c) attends in every loop. Sharpened
  2026-10-05 by the copy-first warm-up arm (below): with the same cards and no consistent
  real-value target nothing is stored in 100M tokens, so the store is written by the supervised
  real-value spans, not by the input or the card path, and not only before the reader copies: the
  warm-up arm stored tier 1000 facts again after copying was in place.
- **What the corpus levers act on** (`inject_build.json`): anonymization only applies to the 20% of
  biography documents without the gold card (16,294 of 162,000 documents, 10%, carry a placeholder
  name). In those documents a span is supervised only when its value also occurs in a distractor
  card (5.6% of their spans), so anonymization reaches at most about 0.7% of the supervised spans
  that see the real name. The other 99% are in documents with the gold card, which always show the
  real name. The anonymization rate therefore cannot materially reduce the arm (c) leak; the swap
  rate acts on the supervised spans and can. Note 2026-10-05: once the reader copies, a real and a
  swapped span both cost about zero in the loss, but the warm-up arm stored tier 1000 facts after
  copying was in place (below), so whether the swap rate still acts then is open.

Raw output: `ckpts/inject/rank_masked`, `rank_masked_prompt`, `rank_masked_100M`, `compare_100M.log`,
`compare_full_masked.log`; training log `inject_masked.log`.

## Arm (c) in full with the key/value reader: it copies, and fails on a leak present before copying (2026-10-05)

`inject_retrieval_kv`, resumed from its 100M save with the same flags (`--reader-kv` included):
299.22M tokens, the 200M of the resume in 59.2 minutes, final filler validation CE 3.8842 (arm (a)
3.9037, arm (b) 3.8769), `loop_scale` [1.60, 0.61, 0.21]. Saves in
`ckpts/evidence_inject_retrieval_kv/` at 50M, 100M, 150M, 200M, 250M and final. Every save from
100M on got the four reads (`gold`, `swapped`, `none`, `prompt`) at `--batch-size 1024`; the 50M
save got `none` and `gold`. Every save scores the same 16,000 items (1,600 people) against the same
candidates, so changes between saves are paired.

### The final save

Entity class, `norm_rank` by tier 1 / 10 / 100 / 1000 (top-1 in brackets where it carries the
reading; the follow rate is over all classes, as the script prints it):

| form | none | gold card (port) | card in prompt | swapped card: follow |
|---|---|---|---|---|
| indist | .482 .499 .453 .294 | .000 .000 .000 .000 (top-1 1.000 .999 1.000 1.000) | .000 .000 .000 .000 (top-1 .996 .997 .999 1.000) | 1.000 1.000 1.000 1.000 |
| heldout | .491 .498 .478 .402 | .024 .021 .019 .010 (top-1 .581 .595 .609 .743) | .018 .017 .016 .017 (top-1 .757 .752 .757 .777) | .623 .627 .614 .638 |

`mr_ll` under the swapped card: 0.000 at every tier in distribution, 0.021 / 0.024 / 0.026 / 0.045
held out. Dates and nouns with the gold card: rank 0.0000, top-1 1.000 at every tier in
distribution; held out dates 0.0002 (top-1 0.98), nouns 0.022 (top-1 0.39).

Closed-book `delta` with its paired z, final save, each cell from that save's own log:

| class | form | tier 1 | tier 10 | tier 100 | tier 1000 |
|---|---|---|---|---|---|
| entity | indist | 0.002 (0.3) | 0.003 (0.5) | 0.027 (4.1) | 0.120 (8.8) |
| entity | heldout | 0.000 (0.1) | -0.001 (-0.2) | 0.013 (3.6) | 0.071 (8.2) |
| date | indist | -0.010 (-0.7) | 0.005 (0.4) | 0.008 (0.6) | 0.101 (3.3) |
| date | heldout | 0.008 (0.7) | 0.003 (0.3) | 0.006 (0.5) | 0.068 (3.2) |
| noun | indist | -0.006 (-0.6) | 0.018 (2.0) | 0.039 (4.0) | 0.092 (4.1) |
| noun | heldout | -0.003 (-0.4) | 0.008 (1.4) | 0.020 (3.1) | 0.016 (1.1) |

- **The reader copies, and the card wins every conflict in distribution.** The gold card ranks the
  value at 0 with top-1 1.000 at every tier in every class (0.999 for entities at tier 10), and the
  swapped card is followed on every item with `mr_ll` 0.000: the A3 shape on the in-distribution
  probe with room to spare. Held out, the gold card ranks entities at 0.010 to 0.024 (top-1 0.58 to
  0.74) and the swapped card is followed 0.61 to 0.64 of the time with `mr_ll` 0.02 to 0.05, under
  the 0.9 follow bar on a template the card never trained. The model copies from its own prompt as
  well as through the port (in distribution 0.0002 or better, top-1 0.996 to 1.000; held out
  0.017, top-1 0.76).
- **The card overrides the memory, it does not erase it.** Under the swapped card the real value
  ranks .494 / .505 / .461 / .318 in distribution, about where it ranks closed book.
- **The rule fails on both forms.** `compare full masked retrieval_kv` at the finals: FAILS on both
  forms, "retrieval off its prior at tiers ['100'] while full climbs". Entities at tier 100 read
  0.027 (z 4.1) in distribution and 0.013 (z 3.6) held out, and tier 1000 is a leak by its own rule
  on both forms (0.120, z 8.8; 0.071, z 8.2). Nouns (the major) are 3 sigma off at tier 100 on
  both forms; dates stay within 3 sigma through tier 100. At the same save arm (a) reads 0.378 /
  0.467 and arm (b) 0.002 / 0.011 in distribution.

### The trajectory: an offset that does not grow

Entity class, closed-book `delta` (z) from each save's own log, with the copy reads:

| save | indist, tier 100 | indist, tier 1000 | heldout, tier 100 | heldout, tier 1000 | gold card top-1, indist | swapped followed, indist |
|---|---|---|---|---|---|---|
| 50M | 0.010 (4.0) | 0.105 (14.4) | 0.006 (2.7) | 0.052 (9.0) | .007 .013 .013 .127 | not read |
| 100M | 0.026 (5.2) | 0.149 (12.3) | 0.006 (1.6) | 0.077 (8.1) | .911 .919 .922 .997 | .925 .920 .913 .880 |
| 150M | 0.029 (5.3) | 0.140 (11.4) | 0.013 (3.3) | 0.070 (7.6) | .997 .996 .999 1.000 | .999 .998 .998 .998 |
| 200M | 0.023 (3.9) | 0.115 (9.4) | 0.015 (3.9) | 0.066 (7.1) | .999 .999 .999 1.000 | 1.000 1.000 1.000 .998 |
| 250M | 0.024 (3.8) | 0.112 (8.8) | 0.013 (3.3) | 0.075 (8.7) | .999 .999 1.000 1.000 | 1.000 1.000 1.000 1.000 |
| final | 0.027 (4.1) | 0.120 (8.8) | 0.013 (3.6) | 0.071 (8.2) | 1.000 .999 1.000 1.000 | 1.000 1.000 1.000 1.000 |

- **No growth after 100M.** Paired from 100M to final (same items, bootstrap sigma of the
  difference), entity in distribution: tier 100 +0.001 (z 0.2), tier 1000 -0.029 (z -2.0); tier
  1000 over all classes -0.038 (z -3.1). Held out: tier 100 +0.007 (z 1.5), tier 1000 -0.006
  (z -0.6). Arm (a) over the same tokens gains +0.279 (z 34.1) at tier 100 and +0.134 (z 8.0) at
  tier 1000. Over the last 200M, with copying complete from 150M on, the arm (c) leak is flat at
  tier 100 and decays slowly at tier 1000.
- The held-out tier 100 cell crosses the bar between 100M and 150M (0.006 to 0.013), but its paired
  change from 100M to final is +0.007 (z 1.5): an offset sitting near 3 sigma, not growth at the
  bar.
- **Not concentrated in people.** Across a person's five attributes the deltas are uncorrelated
  (mean correlation -0.02 to +0.03 at 100M and final, permutation p 0.17 or more). By attribute at
  tier 100, final, in distribution: university 0.041 (z 3.3), major 0.039 (4.0), employer 0.024
  (2.2), city 0.017 (1.6), date 0.008 (0.6).
- **Share of arm (a)'s stored signal** (entity `delta`, arm (c) over arm (a) at the same tokens):
  in distribution 26% / 45% at 100M and 7% / 26% at final (tier 100 / 1000); held out 35% / 78%
  at 100M and 57% / 69% at final. Arm (a) keeps storing; arm (c) does not.
- **In likelihood** (the real name's summed answer log-probability over the fresh-name prior's,
  entity, in distribution): 0.07 nats at tier 100 and 2.84 at tier 1000 at the final save (0.23 /
  3.04 at 100M), against arm (a)'s 3.49 / 6.55. At tier 100 the leak is a rank offset with almost
  no likelihood behind it; at tier 1000 it is a stored association.

### The 50M read: the leak comes before copying

The 50M save was read to time the leak against copying. **No copying yet**: with the gold card the
entity value ranks 0.482 / 0.483 / 0.435 / 0.136 in distribution, top-1 0.007 / 0.013 / 0.013 /
0.127 (chance 0.01; held out top-1 0.009 to 0.027). Paired against the closed-book read on the same
items the card already moves the real value by 0.018 / 0.013 / 0.029 / 0.082 (z 4.4 / 3.4 / 7.2 /
10.3): a cue, not a copy. **The leak is already there**: closed-book entity `delta` in distribution
-0.001 (-0.5) / 0.005 (2.1) / 0.010 (4.0) / 0.105 (14.4), held out 0.002 (1.1) / -0.001 (-0.5) /
0.006 (2.7) / 0.052 (9.0).

Paired against arm (a) at 50M (`rank_full_50M`, same items), entity in distribution: tier 100
+0.000 (z 0.1), tier 1000 +0.018 (z 2.1). The tier 1000 excess is the prior ranking worse (+0.088)
more than the real name does (+0.070), not the real name ranking better: the swaps draw substitutes
uniformly, which flattens the value marginal the prior reads. Growth from 50M to 100M at tier 100 /
1000: arm (c) +0.015 (z 2.9) / +0.044 (z 3.6), arm (a) +0.089 / +0.245.

The reading as first written. The leak is acquired before copying exists, while a supervised fact
span can only be predicted from a memory of the name, and at about arm (a)'s rate (at 50M the two
arms are equal at tier 100). Once the reader copies, between 50M and 100M (gold top-1 0.01 at 50M,
0.91 at 100M, 1.00 from 150M), the span loss goes to about zero for real and swapped spans alike,
and the leak stops growing: from 50M to 100M it grows at a sixth of arm (a)'s rate at tier 100,
after 100M not at all, and tier 1000 decays slowly. Weakened the same day by the warm-up arm
(below): this reading holds in part, next to a level that training maintains after copying.

- **Established**: the leak is present at 50M without copying; it does not grow after 100M.
- **Not established**: the mechanism. The card path before copying is not isolated by this arm
  alone (the card already cues the value at 50M). The warm-up arm (below) since showed that the
  card and input path store nothing detectable without real-value targets, but also that storage
  of tier 1000 facts resumes after copying once real values are supervised.
- **A dose prediction failed.** The arm (c) corpus supervises the real value next to the real name
  in about 0.68 of biography spans (`inject_build.json`: gold card present in 79.9% of documents,
  97,395 of its 647,555 spans swapped; value collisions in the gold-less documents that keep the
  real name add about 0.56%). So the leak at 50M should sit under arm (a)'s. On `delta` it does not:
  it equals arm (a).

### What it means for the levers

- **The swap rate, as first read.** After copying, a real and a swapped span both cost about zero,
  so on the first reading the swap no longer changes what the weights are asked to predict; before
  it, 0.30 in place of 0.15 cuts the real-value share from 0.68 to 0.56, an expected 10 to 20% less
  leak, and tier 100 most likely still fails. On that basis the sweep arm at swap rate 0.30
  (`s30a50`) was set aside on 2026-10-05. The warm-up arm's 150M read removes the support for
  "the swap acts only before copying": if the leak is a level that training maintains, swapped
  exposures act against memory at every point. `s30a50` is reopened as a candidate by the 150M
  read; it runs since 2026-10-05 as a branch of the warm-up arm (below).
- **The copy-first warm-up arm, run 2026-10-05**, run name
  `inject_retrieval_kv_cf` (`ckpts/evidence_inject_retrieval_kv_cf/`). It trains on a second
  retrieval split at swap rate 1.0, `inject_retrieval_s100a50_train`, built with
  `prepare_injection_data.py ... --swap-rate 1.0 --anon-rate 0.5 --gold-drop-rate 0.2 --arms
  retrieval --suffix s100a50` (same seed, same documents and order, same card selection). Every
  span in a gold-present document carries a substitute that the card carries too (647,555 swapped
  spans against 97,395), so there is no consistent name-to-value target to store; the only real
  values left next to the real name are the value collisions in gold-less documents (about 0.6% of
  spans against 68%). The arm trains from `seed_micro.pt` with `--reader-kv` on that split until
  copying holds (planned stop at the 100M save), then the same run is relaunched with
  `--train-split inject_retrieval_train` to 300M (the data overrides are not stored in the
  checkpoint, and the resume position indexes the same permutation). No code change.
- **Criteria, fixed before the read.** At the switch: gold top-1 in distribution at least 0.9, and
  closed-book `delta` within 3 sigma of its prior at every tier; above 3 sigma there falsifies
  "nothing to store" (the input or encoder path stores before copying). At 300M: the R0b rule on
  both forms (tier 100 within 3 sigma), tier 1000 read alongside with an expectation of at most
  about 0.03; a `delta` that climbs after the switch (paired z above 3) falsifies the copy-first
  reading. If it passes, the pilot inherits a requirement: fact spans are not supervised with real
  values before the reader copies (a warm-up or a copy criterion). Read in full below: the rule
  holds through tier 100, tier 1000 leaks, the 0.03 expectation failed and the falsifier triggered
  at tier 1000. Decided 2026-10-05: the requirement enters the pilot (below).
- **Remaining levers against tier 1000**, cheapest first: a second phase at swap rate 0.30
  branched from the warm-up arm's 100M save (read 2026-10-06: no large effect, below); a placeholder name in gold-present documents, with the
  card carrying the same placeholder; a span weight by a copy criterion in the trainer. The
  goldfish loss is expected to do no better than a dose cut, since the renders are paraphrased.

### The warm-up arm in full: the rule holds through tier 100, tier 1000 leaks

First phase: `inject_retrieval_kv_cf` on `inject_retrieval_s100a50_train`, stopped by its `STOP` at
the 100M save (trainer exit 10, 100.4M tokens, `checkpoint_evidence_tok100M_loss7.7318.pt`); filler
validation CE 4.4870 at step 4000, against 4.4807 for the original arm (c) at the same step. The
50M and 100M saves were read with `gold`, `none` and `swapped`. Second phase, 2026-10-05: the same
run relaunched from that save with every flag and `--train-split inject_retrieval_train`
(`ckpts/inject/inject_retrieval_kv_cf_main.log`), finished at 299.19M tokens, the second phase in
62.6 minutes, trainer exit 0. Final filler validation CE 3.9106, against 3.8842 for the original
arm (c) and 3.9037 for arm (a); `loop_scale` [1.58, 0.63, 0.22]. The 0.026 gap to the original
arm is not explained by the logs: the two runs are within 0.008 of each other through step 4000,
the gap opens after the switch (0.012 at step 5000, 0.029 at 8000 and 9000, 0.034 at 11000), and the
warm-up arm ends 0.007 above arm (a), which never had evidence. Saves in
`ckpts/evidence_inject_retrieval_kv_cf/` (50M steps to final).

Reads in hand when these tables were built: `none` at 50M, 100M, 150M, 200M, 250M and final (250M
landed last and was added to the trajectory only; the paired tests below do not use it); `gold`
and `swapped` at 50M, 100M and final; `prompt` at final. Not used here (they landed after the
tables were built and are on disk, not tabulated): `gold`, `swapped` and `prompt` at 150M, 200M
and 250M.

Entity class, closed-book `delta` (z) from each save's own log, next to the original arm (c) on the
same items; gold top-1 and the swapped follow rate (all classes) in distribution:

| save | arm | indist, tier 100 | indist, tier 1000 | heldout, tier 100 | heldout, tier 1000 | gold top-1 indist | swapped followed indist |
|---|---|---|---|---|---|---|---|
| 50M | warm-up | 0.002 (0.7) | -0.002 (-0.2) | 0.001 (0.6) | 0.004 (0.8) | .010 .012 .019 .013 | .099 .096 .096 .096 |
| 50M | original (c) | 0.010 (4.0) | 0.105 (14.4) | 0.006 (2.7) | 0.052 (9.0) | .007 .013 .013 .127 | not read |
| 100M | warm-up | -0.006 (-1.1) | -0.001 (-0.1) | -0.002 (-0.5) | -0.003 (-0.3) | .973 .973 .959 .953 | .980 .980 .976 .978 |
| 100M | original (c) | 0.026 (5.2) | 0.149 (12.3) | 0.006 (1.6) | 0.077 (8.1) | .911 .919 .922 .997 | .925 .920 .913 .880 |
| 150M | warm-up, 50M on real values | 0.011 (1.9) | 0.081 (6.0) | 0.002 (0.5) | 0.021 (2.3) | not read | not read |
| 150M | original (c) | 0.029 (5.3) | 0.140 (11.4) | 0.013 (3.3) | 0.070 (7.6) | .997 .996 .999 1.000 | .999 .998 .998 .998 |
| 200M | warm-up | 0.013 (2.1) | 0.095 (6.6) | 0.007 (1.8) | 0.033 (3.5) | not read | not read |
| 200M | original (c) | 0.023 (3.9) | 0.115 (9.4) | 0.015 (3.9) | 0.066 (7.1) | .999 .999 .999 1.000 | 1.000 1.000 1.000 .998 |
| 250M | warm-up | 0.013 (2.3) | 0.087 (6.0) | 0.011 (2.8) | 0.036 (3.9) | not read | not read |
| 250M | original (c) | 0.024 (3.8) | 0.112 (8.8) | 0.013 (3.3) | 0.075 (8.7) | .999 .999 1.000 1.000 | 1.000 1.000 1.000 1.000 |
| final | warm-up | 0.015 (2.6) | 0.085 (6.4) | 0.007 (2.0) | 0.047 (5.1) | .999 1.000 1.000 1.000 | 1.000 1.000 1.000 1.000 |
| final | original (c) | 0.027 (4.1) | 0.120 (8.8) | 0.013 (3.6) | 0.071 (8.2) | 1.000 .999 1.000 1.000 | 1.000 1.000 1.000 1.000 |

Final save, closed-book `delta` with its paired z for every class:

| class | form | tier 1 | tier 10 | tier 100 | tier 1000 |
|---|---|---|---|---|---|
| entity | indist | 0.004 (0.6) | 0.000 (0.0) | 0.015 (2.6) | 0.085 (6.4) |
| entity | heldout | 0.001 (0.3) | -0.001 (-0.3) | 0.007 (2.0) | 0.047 (5.1) |
| date | indist | 0.000 (0.0) | 0.008 (0.6) | 0.021 (1.4) | 0.106 (3.6) |
| date | heldout | -0.009 (-0.9) | 0.000 (0.0) | -0.005 (-0.4) | 0.060 (2.4) |
| noun | indist | -0.003 (-0.3) | -0.004 (-0.4) | 0.020 (1.9) | 0.024 (1.1) |
| noun | heldout | 0.002 (0.3) | 0.013 (2.0) | 0.008 (1.2) | 0.010 (0.7) |

Final copy reads, entity class: gold card top-1 0.999 / 1.000 / 1.000 / 1.000 in distribution
(rank 0.0000), held out rank 0.010 to 0.022 with top-1 0.60 to 0.74; swapped card followed 1.000
in distribution with `mr_ll` 0.000, held out 0.663 to 0.690 (`mr_ll` 0.027 to 0.042, all
classes); prompt copy rank 0.0000 in distribution (top-1 0.997 to 1.000), held out 0.008 with top-1
0.84, against the original arm's 0.017 and 0.76.

- **Both switch criteria are met.** At 100M gold top-1 in distribution is 0.953 to 0.973 (rank
  0.001 or better; held out rank 0.018 to 0.021, top-1 0.63 to 0.65), and closed-book `delta` is
  within 3 sigma of its prior at every tier, class and form at 50M and at 100M (the largest |z| is
  2.3 at 50M, noun held out tier 1, and 2.0 at 100M, entity held out tier 1). The swapped card is
  followed 0.976 to 0.980 of the time in distribution (`mr_ll` 0.001 to 0.002) and 0.70 to 0.73
  held out. Dates copy first: at 50M the gold card already ranks dates at 0.03 (top-1 0.39 to 0.46)
  while entities sit at 0.45 to 0.48.
- **What this establishes.** With the same documents, cards and schedule and no consistent
  real-value target, the input side and the card path store nothing detectable in 100M tokens,
  where the original arm (c) at 50M already read 0.010 (z 4.0) and 0.105 (z 14.4). The arm (c)
  store is therefore written by the supervised real-value spans. "Nothing to store" is not
  falsified.
- **Copying is learned at least as well on fully swapped spans.** Paired against the original arm
  at 100M (same items), entity in distribution: gold top-1 +0.061 / +0.053 / +0.037 at tiers 1 /
  10 / 100 (z 8.0 / 7.3 / 5.0) and -0.043 (z -3.7) at tier 1000, where the original arm's memory
  helps; swapped follow +0.057 to +0.113 (z 5.6 to 8.2), `mr_ll` lower at every tier. Held out the
  follow rate is about the same (0.70 to 0.73 against 0.67 to 0.70, all classes).
- **The falsifier triggered at tier 1000.** At 150M, after 50M tokens of real-value supervision
  with copying already in place, entity `delta` in distribution is 0.001 (0.2) / -0.005 (-0.9) /
  0.011 (1.9) / 0.081 (6.0), held out -0.001 (-0.4) / -0.002 (-0.5) / 0.002 (0.5) / 0.021 (2.3).
  Paired from 100M to 150M within the arm (same items): tier 100 +0.017 (z 2.8), tier 1000 +0.082
  (z 5.9), the real name gaining 0.109 in rank against 0.027 for the prior; over all classes in
  distribution tier 100 moves +0.018 (z 3.6) and tier 1000 +0.077 (z 6.3); held out entity tier
  1000 +0.023 (z 2.5). The pre-registered rule ("a `delta` that climbs after the switch, paired z
  above 3") is met at tier 1000: copying first does not stop storage of frequent facts. Tier 100
  for entities stays within 3 sigma of its prior so far.
- **Then flat.** Within the warm-up arm, paired 200M to final, entity in distribution: tier 100
  +0.002 (z 0.5), tier 1000 -0.011 (z -1.2); held out -0.000 (z -0.1) and +0.014 (z 2.2). Paired 100M
  to final: tier 100 +0.020 (z 3.3), tier 1000 +0.086 (z 5.5). So the climb after the switch is
  beyond 3 sigma at tier 1000 and, over the whole second phase, at tier 100 too, while tier 100
  itself stays within 3 sigma of its prior (0.015, z 2.6).
- **The rule holds through tier 100; tier 1000 leaks.** `compare full masked retrieval_kv_cf` at the
  finals: HOLDS on both forms, "masked and retrieval are within 3 sigma of their fresh-name prior
  through tier 100 while full climbs". The margin is small in distribution (tier 100 z 2.6). Tier
  1000 is a leak on both forms by the rule's own reading of tier 1000 (0.085, z 6.4; 0.047, z 5.1),
  and the date class leaks at tier 1000 in distribution (0.106, z 3.6). No tier 100 cell in any
  class or form is beyond 3 sigma. The gate verdict was the user's; decided 2026-10-05 (below).
- **Lower than the original arm, not at 3 sigma per form.** Paired against the original arm (c) at
  the final save (same items), entity: in distribution tier 100 -0.012 (z -1.5), tier 1000 -0.035
  (z -2.2); held out -0.006 (z -1.4) and -0.024 (z -2.2). At 150M the same comparison read -0.018
  (z -2.5) / -0.059 (z -3.7) in distribution and -0.010 (z -2.0) / -0.049 (z -4.2) held out; at
  200M -0.010 (z -1.3) / -0.019 (z -1.2) and -0.008 (z -1.5) / -0.033 (z -3.1). In
  `compare_kv_vs_cf.log` (original arm in the first slot, warm-up arm in the third, all classes
  pooled over forms) the real name ranks better in the original arm at tiers 100 and 1000: the same
  direction, quoted as direction only. As ratios of the final entity `delta`, the warm-up arm holds
  0.54 / 0.71 of the original arm's leak at tier 100 / 1000 in distribution and 0.52 / 0.66 held
  out; of arm (a)'s stored signal it holds 4% / 18% in distribution (original 7% / 26%) and 30% /
  45% held out (original 57% / 69%). The tier 1000 likelihood gap (real name over prior, summed
  answer log probability) is 1.81 nats at the final save against 2.84 for the original arm (1.61 at
  150M, 2.12 at 200M; -0.56 at the switch).
- **Expectations against the read.** The pre-registered expectation "tier 1000 at most about 0.03"
  failed (0.085). The falsifier "a `delta` that climbs after the switch, paired z above 3" triggered
  at tier 1000 (from 150M on) and, over 100M to final, at tier 100.
- **Established.** (1) With no consistent real-value target nothing detectable is stored, at 50M and
  at 100M. (2) Real-value supervision after the reader copies still writes frequent facts into the
  weights: tier 1000 goes from about 0 to 0.08 to 0.10 within 50M to 100M tokens of it, then stays
  flat. (3) With the warm-up, tier 100 is within 3 sigma of its prior on both forms at the final
  save (z 2.6 and 2.0), where the original arm was not (z 4.1 and 3.6).
- **Not established at 3 sigma per form**: that the warm-up lowers the final leak against the
  original arm. Every per-form cell at tiers 100 and 1000 points that way (z -1.4 to -2.2 at the
  final save, -3.7 and -4.2 at tier 1000 at 150M), and so does the pooled comparison.
- **The two earlier readings both hold in part.** Some of the original arm's leak was most likely
  written before copying (the warm-up arm, which never had a real-value target before copying,
  ends lower in every cell); the rest is a level that training maintains while real values are
  supervised next to the real name (the warm-up arm reaches 0.08 to 0.10 at tier 1000 after the
  switch and holds it, as the original arm held 0.11 to 0.15).
- **Open: what sets the maintained level.** Candidates: the swap rate (swapped exposures penalize
  answering from memory at every point of training; untested), the gold-drop documents (20% of
  biography documents carry no gold card; their spans are supervised only where a distractor card
  happens to carry the same value, 1.1% of all spans), and residual span loss after copying.

**Decided 2026-10-05 (by the user, both).** (1) The copy-first warm-up is accepted as the basis for
R0b: the rule holds through tier 100 on both forms with it (marginally, in distribution z 2.6), the
tier 1000 leak is recorded as known, and "no real-value fact supervision before the reader copies"
goes into the pilot as a requirement, a new element of the pilot's schedule. The warm-up is the
better of two arms on one seed, not an optimum; its improvement over the original arm is a
direction, not established per form. (2) The swap rate 0.30 branch below ran before the pilot spec
is frozen, because it sets a pilot parameter (the swap rate) either way.

### The swap rate 0.30 branch (started 2026-10-05, read 2026-10-06)

Run name `inject_retrieval_kv_cf30` (`ckpts/evidence_inject_retrieval_kv_cf30/`, log
`ckpts/inject/inject_retrieval_kv_cf30.log`, launcher `ckpts/inject/launch_kv_cf30.sh`). The split
`inject_retrieval_s30a50_train` is built with `--swap-rate 0.30 --anon-rate 0.5 --gold-drop-rate
0.2 --arms retrieval --suffix s30a50`; the warm-up arm's 100M save
(`checkpoint_evidence_tok100M_loss7.7318.pt`) and a `run_state.json` at 100,419,895 tokens are
copied into the new run directory, and the run goes to 300M with `--reader-kv` and `--train-split
inject_retrieval_s30a50_train`. No code change. Every save past 100M gets `none`, `gold`,
`swapped` and `prompt` reads into `ckpts/inject/rank_retrieval_kv_cf30<tag>_<mode>`. The two arms
share everything up to 100M, so later differences come from the swap rate (0.30 against 0.15) and
the draw of which spans are swapped.

Criteria, fixed before the read:

- **The question**: does the level that training maintains follow the swap rate. Read closed-book
  entity `delta` at tiers 1000 and 100, paired against the warm-up arm at the same saves (150M,
  200M, 250M, final), per form. It follows the swap rate if tier 1000 is lower than the warm-up
  arm's at paired z at or beyond -3 per form at the final save, or on both forms pooled.
- **Power, stated before the read.** The paired sigma at tier 1000 is about 0.016 in distribution
  and 0.011 held out (from the warm-up against original comparison at the final save: -0.035 at
  z -2.2 and -0.024 at z -2.2). Only a drop of about 0.05 in distribution (the 0.085 falling to
  about 0.035 or below) reads at 3 sigma per form. A pure dose effect (real-value share 0.68 to
  0.56) predicts about 0.07, which this arm cannot separate from no effect. A null therefore means
  "no large effect of the swap rate", not "no effect".
- **Alongside**: copying must not degrade (gold top-1 in distribution, swapped follow, held-out
  follow, against the warm-up arm), tier 100 must stay within 3 sigma, and the filler CE gap is
  read again (the warm-up arm ended 0.026 above the original arm after its switch).
- **After it**: if the level follows the swap rate, the pilot's swap rate is set from it; if not,
  the placeholder name in gold-present documents with the card carrying the same placeholder
  (about 25 lines in the builder and `biographies.render_store_chunk`) is the next candidate
  against tier 1000, and the pilot can start with the warm-up alone (the user's call then).

**Read 2026-10-06.** Finished at 299.19M tokens, the second phase in 63.8 minutes, filler
validation CE 3.9131 (warm-up arm 3.9106, original arm (c) 3.8842). Reads taken: `none` at 150M
and final, `gold` and `swapped` at final. Not taken: the 200M and 250M reads and `gold`, `swapped`,
`prompt` at 150M. The final `prompt` read landed after the tables were built: in distribution rank
0.0001 (top-1 0.993 to 0.997), held out 0.0084 (top-1 0.834), the same as the warm-up arm's 0.0081
(0.839).

Entity class, closed-book `delta` (z) from each save's own log:

| save | arm | indist, tier 100 | indist, tier 1000 | heldout, tier 100 | heldout, tier 1000 |
|---|---|---|---|---|---|
| 150M | swap 0.30 branch | 0.003 (0.6) | 0.072 (5.3) | 0.006 (1.4) | 0.027 (3.0) |
| 150M | warm-up arm (swap 0.15) | 0.011 (1.9) | 0.081 (6.0) | 0.002 (0.5) | 0.021 (2.3) |
| final | swap 0.30 branch | 0.010 (1.7) | 0.083 (6.2) | 0.003 (0.9) | 0.039 (4.6) |
| final | warm-up arm (swap 0.15) | 0.015 (2.6) | 0.085 (6.4) | 0.007 (2.0) | 0.047 (5.1) |

Final branch, tiers 1 and 10: 0.005 (0.8) / 0.004 (0.7) in distribution, -0.002 (-0.5) / -0.001
(-0.2) held out. No tier 100 cell in any class or form is beyond 3 sigma (largest noun in
distribution 0.026, z 2.5); dates at tier 1000 in distribution 0.099 (z 3.0). `compare full masked
retrieval_kv_cf30`: HOLDS on both forms.

- **Paired branch minus warm-up arm at the final save** (same items, entity): tier 1000 -0.002
  (z -0.3) in distribution, -0.008 (z -1.4) held out, -0.005 (z -1.0) pooled over forms; tier 100
  -0.004 (z -1.0) and -0.004 (z -1.4). All classes pooled over forms: tier 1000 -0.009 (z -2.1),
  with the real name ranking worse by +0.014 (z 3.8) and the prior worse by +0.005 (z 1.2). That
  real-rank change is what `compare_cf_vs_cf30.log` shows as tier 1000 -0.014 at z -4.0 (raw rank,
  warm-up minus branch); the `delta` change is smaller and under 3 sigma.
- **Criterion: not met.** Tier 1000 is not lower at paired z at or beyond -3, per form or pooled.
  The tier 1000 level does not follow the swap rate in any large way (0.083 against 0.085 in
  distribution): doubling the swap rate after the switch leaves the frequent-fact leak where it
  was, so the swap rate is not a strong dial for the level that training maintains. Within the
  stated power this is "no large effect", not "no effect": a dose-sized effect (about 0.07) could
  not be separated from none.
- **Tier 100**: lower on both forms as a direction (paired z -1.0 and -1.4), and the rule holds
  with more margin (z 1.7 and 0.9 against 2.6 and 2.0).
- **Copying does not degrade.** Gold top-1 in distribution 0.999 to 1.000 in both arms, held out
  entity 0.62 to 0.73 against 0.60 to 0.74; swapped card followed on every item in distribution,
  held out 0.654 to 0.673 against 0.663 to 0.690 (all classes; tier 1000 -0.036, z -2.1); `mr_ll`
  0.000 in distribution and within 0.005 held out. Prompt copy: not read.
- **Plan state.** Open, the user's: the pilot's swap rate, and whether the placeholder name arm
  runs before the pilot. Recommendation: choose the swap rate on copy quality and tier 100 margin,
  not on tier 1000. On those reads 0.30 against 0.15 copies the same in distribution, follows a
  swapped card about 0.01 to 0.04 less often held out (under 3 sigma), and gives tier 100 more
  margin, a direction only. The open lever against tier 1000 is the placeholder name in
  gold-present documents with the card carrying the same placeholder (about 25 lines in the
  builder and `biographies.render_store_chunk`).

Raw output: `ckpts/inject/rank_retrieval_kv{50M,100M,150M,200M,250M,final}_{gold,swapped,none,prompt}`
(`.json` and `.log`; 50M has `none` and `gold` only), `rank_full_50M`, `compare_full_masked_kv.log`;
training logs `inject_retrieval_kv.log`, `inject_retrieval_kv_resume.log`; warm-up
`build_s100.log`, `inject_retrieval_kv_cf_warm.log`, `inject_retrieval_kv_cf_main.log`,
`rank_retrieval_kv_cf{50M,100M}_{gold,none,swapped}`, `rank_retrieval_kv_cf{150M,200M}_none`,
`rank_retrieval_kv_cffinal_{none,gold,swapped,prompt}`, `compare_full_masked_kv_cf.log`,
`compare_kv_vs_cf.log`.

## The per-exit read: depth pays evenly on this corpus (2026-10-05)

`scripts/eval_exit.py` (eval only; `tests/test_eval_exit.py` is GPU-free) reads a checkpoint at
every loop exit on one split with no evidence attached: per-exit CE, paired gains with a standard
error clustered by document, two bounds that choose the exit with the true label (the per-token
oracle and an entropy-regularized optimum), a confidence exit rule against a mix of fixed depths at
the same mean passes, and deciles by confidence at exit 1. The exit is readout-only: every token
still runs all passes. Both finals on `inject_val`, 1,104,743 supervised tokens in 1,492 documents:

| arm, final | CE exit 1 / 2 / 3 | gain 1 to 2 | gain 2 to 3 | oracle CE | beta 0.1 optimum, share by exit | confidence rule, tau 0.5 | fixed depths, same passes |
|---|---|---|---|---|---|---|---|
| (a) full CE | 4.0072 / 3.9132 / 3.9071 | +0.0940 (z 134) | +0.0061 (z 36) | 3.7804 | 0.343 / 0.286 / 0.371 | 3.9237 at 2.43 passes | 3.9106 |
| (c) key/value | 3.9902 / 3.8919 / 3.8859 | +0.0983 (z 139) | +0.0061 (z 38) | 3.7577 | 0.342 / 0.283 / 0.375 | 3.9024 at 2.43 passes | 3.8893 |

- Exit 3 matches the trainer's `[eval]` CE to 0.003 (3.9037 and 3.8842; the head runs in fp32
  here, on bf16 logits in the trainer).
- The second pass is worth 0.09 to 0.10 nats per token, the third 0.006. **The second pass's gain
  is spread evenly over tokens**: by `p_max` decile at exit 1, arm (a)'s gain from exit 1 to 3 is
  +0.107 to +0.122 over the seven least confident deciles, +0.098 in the eighth, +0.072 in the
  ninth and +0.024 in the most confident.
- **A confidence exit rule buys nothing over fixed depths.** It is worse than a mix of fixed depths
  at the same mean passes at tau 0.5, 0.7 and 0.9 (by 0.0131, 0.0057 and 0.0010 on arm (a)) and
  equal at 0.99 (0.0002 better at 2.96 passes).
- **The entropy-regularized optimum is near uniform even as a bound**: 0.343 / 0.286 / 0.371 at
  beta 0.1, 0.358 / 0.257 / 0.385 at beta 0.05.
- Arm (c) reads the same as arm (a): training through the reader changed nothing about where depth
  pays on filler.

**For the learned depth allocation arm.** A design was drawn up on 2026-10-05 after Ouro (arXiv
2510.25741, checked against the paper): a per-token gate `lambda_t = sigmoid(linear(h_t))`, an exit
distribution `p_t = lambda_t * prod_{j<t} (1 - lambda_j)` with the remainder on the last pass, and
the objective `sum_t p_t L_t - beta H(p)` with beta 0.1. The gate only reweights the per-pass
losses and never touches the forward, which is how it differs from the removed halt head. It is not
built. On this corpus the gate has little to learn: the second pass's gain is spread evenly over
tokens, the third pass adds 0.006 nats, and the gate's own optimum is near uniform, which would cut
the last pass's share of the CE weight from 0.67 (`loop_ce_weights` 0.2 / 0.3 / 1.0) to about 0.37
to 0.39, against the requirement that a weak later loop is fixed, not cut. Recommendation recorded
with it: do not train the gate arm on the biography corpus; the depth question needs a corpus where
depth pays (the chain splits, of which only the eval splits are built). Three decisions on it are
open (NEXT.md, ladder step 8).

Raw output: `ckpts/inject/exit_full.{json,log}` (arm (a) final), `exit_retrieval_kv.{json,log}`
(arm (c) final).

# Fact injection micro-pilot: arms (a) and (b), the arm (c) readers at 100M, the loop reading (2026-10-02 to 10-04)

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
  verdict; the plan's lever for "only (c) climbs" is the swap rate.

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
  0.149 (z 12.3) read alongside. Since arm (b) is flat at the same tokens, the arm (c) leak is not
  the in-context input path. Most likely it comes from the spans arm (c) supervises, 81% of the
  fact tokens, with the real name in the prompt and the unswapped value as the target 85% of the
  time; arm (b) does not test the card path, which arm (c) attends in every loop.
- **What the corpus levers act on** (`inject_build.json`): anonymization only applies to the 20% of
  biography documents without the gold card (16,294 of 162,000 documents, 10%, carry a placeholder
  name). In those documents a span is supervised only when its value also occurs in a distractor
  card (5.6% of their spans), so anonymization reaches at most about 0.7% of the supervised spans
  that see the real name. The other 99% are in documents with the gold card, which always show the
  real name. The anonymization rate therefore cannot materially reduce the arm (c) leak; the swap
  rate acts on the supervised spans and can.

Raw output: `ckpts/inject/rank_masked`, `rank_masked_prompt`, `rank_masked_100M`, `compare_100M.log`,
`compare_full_masked.log`; training log `inject_masked.log`.

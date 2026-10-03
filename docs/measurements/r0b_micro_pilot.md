# Fact injection micro-pilot: arm (a) and the arm (c) reader smoke (2026-10-02)

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

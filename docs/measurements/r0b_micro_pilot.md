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

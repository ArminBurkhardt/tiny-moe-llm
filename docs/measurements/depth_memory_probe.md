# Depth memory probe (2026-10-21): is the depth schedule trainable at pilot shape

Rung R1b asks whether one training step at depth 5 fits the 5090 at the pilot's shape, with and
without the detached prefix the blueprint's depth schedule assumes, against depth 3 (the pilot's
own depth). Answer: it fits. Depth 5 with full gradient does not fit at the pilot's batch 8 x
4096 (about 31 GB allocated by extrapolation, the card holds 32 GB physical) but fits at batch 4
x 4096 (17.3 GB), so batch 4 with 2 accumulation steps keeps the tokens per update the same, at
about 1.27x the step time per token of depth 3 (config `config_pilot.yaml`). The detached prefix fits easily (21.5 GB at batch 8,
under depth 3) but as built it cuts the prelude decoder off from the language loss.

## Build

- `detach_loops: int = 0` on `LoopMixtureOfExperts.forward` and `TinyMoETransformer.forward`: the
  first `detach_loops` passes run under `no_grad` as plain steps before the checkpoint branch,
  their states kept in `hidden_states_all`, their aux terms constants; asserted
  `0 <= detach_loops < n_loops`; 0 is bit-identical to before. `tests/test_loop_machinery.py`
  checks bit-identity at 0, every per-loop state unchanged at 2, the `loop_scale` gradient
  reaching only the live pass, and the bounds. A detached review of the forward change ran the
  same day; its findings were folded in.
- `scripts/probe_depth_memory.py`: builds the pilot-shape model from `seed_pilot_grounded.pt` as
  `sft.py --evidence --reader-kv` does (fp32 masters, the trainer's AdamW), takes one real batch
  from `pilot_main_train` through `EvidenceDataset` and `evidence_from_batch` (the heaviest in
  evidence of the first 16 batches), and runs one full step per setting: forward at the setting's
  depth and prefix, the trainer's loss terms, backward, clip, master sync, optimizer step at zero
  learning rate. The depth is the forward's `n_loops`; passes past the seed's 3 reuse the last
  `loop_scale` and `evidence_loop_scale` entry. Each setting runs in its own child process, so a
  WSL OOM (`CUDA driver error: device not ready`) reads as an `oom` row and the next setting runs.
  Two steps per setting: peak is the maximum, time the minimum. Flags `--settings
  depth:detach_loops`, `--batch-size`, `--repeats`, `--json-out`.

## Results

Card 34.2 GB visible, 32 GB physical. Outputs `ckpts/inject/probe_depth_memory.{log,json}` and
`ckpts/inject/probe_depth_memory_b4.{log,json}`.

Batch 8 x 4096 (the pilot's own): 26,424 prompt tokens, 26,454 evidence tokens (ratio 1.00).

| setting | depth | detached passes | peak allocated GB | peak reserved GB | seconds per step |
|---|---|---|---|---|---|
| 3:0 | 3 | 0 | 25.20 | 27.09 | 0.974 |
| 5:0 | 5 | 0 | oom | oom | |
| 5:2 | 5 | 2 | 21.51 | 23.55 | 1.054 |

Batch 4 x 4096: 13,803 prompt tokens, 13,166 evidence tokens (ratio 0.95).

| setting | depth | detached passes | peak allocated GB | peak reserved GB | seconds per step |
|---|---|---|---|---|---|
| 3:0 | 3 | 0 | 14.43 | 14.78 | 0.471 |
| 5:0 | 5 | 0 | 17.25 | 17.85 | 0.599 |
| 5:2 | 5 | 2 | 12.24 | 13.42 | 0.471 |

The depth-3 row at batch 8 matches the trainer's own 25.3 GB on arm PI. Two extra live passes
cost 2.8 GB at batch 4; doubled onto the batch 8 depth-3 row that puts depth 5 with full gradient
at about 31 GB allocated, over the card, as the OOM row says. Depth 5 at batch 4 costs 0.599
against 0.471 s per step, 1.27x per token.

## Reading

1. Depth 5 is trainable at pilot shape on the 5090 with full gradient at batch 4 x 2
   accumulation (the same tokens per update as batch 8 x 1), at about 1.27x the step time per
   token of depth 3.
2. The detached prefix as built fits easily (21.5 GB at batch 8, under depth 3) but is not a
   usable training recipe on its own. With the first two passes detached the prelude decoder
   gets no gradient from the language loss: the state entering the first live pass carries no
   graph and the MTP reads the final loop. On the test fixture, decoder layer parameters with a
   nonzero gradient are 26 of 26 at detach 0 and 0 of 26 at detach 2, so the prelude would train
   only through the evidence encoding and the per-layer table. The blueprint's note that
   `loop_inject` carries "gradient flow through the detached depth prefix" is the missing piece;
   the cheapest route is a straight-through residual at the first live pass
   (`h = decoder_out + (prefix_out - decoder_out).detach()`), one line, unbuilt.

## Verdict

Fits. The depth schedule is trainable at pilot shape without the prefix at batch 4 x 2, and with
the prefix only once the gradient route to the prelude exists. The pilot ran at depth 3 and the
loop did not compose there, so depth 5 is not on the path today; nothing changes in the real-run
spec. No decision is needed now; the straight-through route is a one-line build if depth 5 with
a prefix is ever wanted.

# Current Status and Next Steps

## Current status (verified from source)

- The repository contains a working prototype for a recurrent Mixture-of-Experts architecture around Gemma hidden states.
- The core pipeline exists in `FinalTransformer`:
  1. Encode tokens with `Gemma3Encoder`
  2. Transform latent states with `MixtureOfExperts`
  3. Decode with an invertible-style `Decoder`
- Dynamic MoE behavior is implemented:
  - repeated normal expert routing
  - periodic expert addition (`solve_from_batch`-style fitting)
  - explicit OUTPUT expert routing
  - least-used expert pruning
- Unit-like tests exist for many components under `modules/model` and `modules/data`.

## Verification notes

- Pre-change test command `python test.py` was executed.
- In this environment, execution failed early because `torch` is not installed (`ModuleNotFoundError: No module named 'torch'`).
- Because of this environment limitation, runtime claims in these docs are restricted to what can be verified statically from source.

## Next steps to test training efficiency vs regular LLM architectures

1. **Define baseline comparisons**
   - Compare against a standard dense transformer fine-tuning setup with the same dataset slices and batch sizes.

2. **Measure efficiency metrics**
   - Wall-clock time per training step
   - Tokens/sec throughput
   - Peak/average memory usage
   - FLOPs estimate (or profiler-derived operator time)

3. **Measure quality metrics**
   - Training/validation loss curves
   - Perplexity or task-specific accuracy
   - Stability metrics (NaNs, divergence frequency)

4. **Ablation studies**
   - With vs without expert pruning
   - Different `steps_per_expert`
   - Different output skew schedules
   - Solvable expert variants (e.g., `SolvableLinear` vs `ExpertModule`)

5. **Reproducible experiment harness**
   - Scripted benchmark entrypoints with fixed seeds
   - Logged configs + metrics to compare runs directly
   - Repeat runs to report variance, not just single-run numbers

6. **Scalability checks**
   - Increase sequence lengths and batch sizes
   - Track whether recurrent routing keeps gains at larger scales

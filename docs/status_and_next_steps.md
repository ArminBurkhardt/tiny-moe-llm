# Current Status and Next Steps

## Current status (verified from source)

- `modules/model/transformer.py` implements `FinalTransformer`, which composes:
  1. `Gemma3Encoder` (`modules/model/encoder.py`)
  2. `MixtureOfExperts` (`modules/model/moe.py`)
  3. `Decoder` (`modules/model/decoder.py`)
- Dynamic MoE lifecycle is implemented in `modules/model/moe.py`:
  - repeated routing over existing experts (`cycle_pos < steps_per_expert`)
  - expert addition and solving (`cycle_pos == steps_per_expert`)
  - OUTPUT-only routing step (`cycle_pos == steps_per_expert + 1`)
  - pruning helper (`prune_least_used`)
- Tests are present for core modules in:
  - `modules/model/test_moe.py`
  - `modules/model/test_final_transformer.py`
  - `modules/data/test_vectorized_dataset.py`

## Next steps to test training efficiency vs regular LLM architectures

1. **Baseline setup**
   - Use one dense baseline with the same tokenizer, sequence length, dataset slice, and batch size.

2. **Efficiency metrics (per run)**
   - Step time (ms/step)
   - Throughput (tokens/s)
   - Peak memory (GPU/CPU)
   - Router/expert overhead share (profiler time by module)

3. **Quality metrics (same checkpoints/intervals)**
   - Train/validation loss
   - Perplexity or task metric
   - Instability events (NaN/Inf/divergence)

4. **Ablations (one variable at a time)**
   - `steps_per_expert`
   - pruning on/off
   - output skew schedule
   - expert template choice (`SolvableLinear` vs `ExpertModule`)

5. **Reproducibility**
   - Fixed random seeds
   - Logged config + metrics for each run
   - At least 3 repeated runs per key setting

6. **Scaling checks**
   - Repeat the benchmark at larger sequence lengths and batch sizes to check whether the efficiency trend holds.

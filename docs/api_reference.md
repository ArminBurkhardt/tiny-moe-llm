# API Reference

This document summarizes the current public/usable APIs with concise usage notes and tensor shape guidance.

## `utils.py`

### Constants and paths
- `DIR`
  - Path constants used across training/data scripts:
    - `BASE_DIR`
    - `GEMMA_EMBEDDING_DIR`
    - `GEMMA_3_1B_DIR`
    - `GEMMA_3_270M_DIR`
    - `GEMMA_2_T5_270M_DIR`
    - `GEMMA_3_DIR` (alias of `GEMMA_3_1B_DIR`)
    - `DATA_DIR`
    - `UFW_V1_4_DIR`
    - `KIMI_DIR`
    - `REASNONING_DIR`
- `PATH = DIR`
- `FP64`, `FP32`: dtype aliases (`torch.float64`, `torch.float32`)

### Base interface
- `class InvertibleModule`
  - `inverse(y: torch.Tensor, **kwargs) -> torch.Tensor`
  - `auto_inverse(y: torch.Tensor, **kwargs) -> torch.Tensor`
  - Meaning: contract for modules that support explicit inverse (or best-effort inverse).

---

## `modules/model`

### Exported API (`modules/model/__init__.py`)
- `ParameterizedSigmoid`
- `InvertibleActivation`
- `LinearAttention`
- `MLP`
- `Gemma3Encoder`
- `Decoder`
- `LatentRouter`
- `SolvableLinear`
- `InvertibleLinear`
- `InvertibleLinearAttention`
- `MatrixInvertabilityLoss`
- `MixtureOfExperts`

### Activations (`modules/model/activations.py`)
- `class ParameterizedSigmoid`
  - `f(a, b) -> Callable[[Tensor], Tensor]`
  - `f_inv(a, b) -> Callable[[Tensor], Tensor]`
  - Element-wise, shape-preserving transform; `f_inv` expects values in `(-b, a)`.
- `class InvertibleActivation(nn.Module, InvertibleModule)`
  - `forward(x)` / `inverse(y)` / `auto_inverse(y)`
  - Input/output: same shape `[..., D]`.
- `class InvertibleLeakyReLUActivation(nn.Module, InvertibleModule)`
  - `forward(x)` / `inverse(y)` / `auto_inverse(y)`
  - Input/output: same shape `[..., D]`.
- `class ShiftActivation(nn.Module, InvertibleModule)`
  - `forward(x)` / `inverse(y)` / `auto_inverse(y)`
  - Adds/removes constant shift; shape-preserving `[..., D]`.

### Core modules (`modules/model/modules.py`)
- `class LinearAttention(nn.Module)`
  - `forward(x, other=None)`
  - Expected shapes:
    - `x`: `[B, T_q, D_in]`
    - `other` (optional): `[B, T_kv, D_in]` (defaults to `x`)
    - output: `[B, T_q, D_out]`
  - Meaning: single-head linear projections (`q/k/v`) + softmax attention over sequence axis.
- `class MLP(nn.Module)`
  - `forward(x)`
  - Shape: `[..., input_size] -> [..., output_size]`.

### Linear/invertible layers (`modules/model/linear.py`)
- `class InvertibleLinear(nn.Module, InvertibleModule)`
  - `forward(x)`
  - `inverse(y)` (exact inverse; square + full-rank only)
  - `approx_linear_inverse(y)` (pseudo-inverse)
  - `auto_inverse(y)` (exact if square else pseudo-inverse)
  - `is_square`
  - Shapes:
    - `forward`: `[..., input_size] -> [..., output_size]`
    - inverse variants: `[..., output_size] -> [..., input_size]`
- `class SolvableLinear(InvertibleLinear)`
  - `enable_grad(enabled=True)` / `disable_grad()`
  - `forward(x)`
  - `solve_from_batch(x, y, l2=1e-4)`
  - `auto_solve(x, y, l2=1e-4)`
  - Solve contract:
    - `x`: `[N, input_size]`
    - `y`: `[N, output_size]`
    - returns `(weight, bias)` with shapes `[output_size, input_size]`, `[output_size]`.

### Router (`modules/model/router.py`)
- `class LatentRouter(nn.Module)`
  - `output_index`: index of special OUTPUT expert (`== num_experts`)
  - `add_experts(k)`: extends expert logits while preserving OUTPUT head row
  - `forward(z, is_final=None, output_skew=0.0)`
    - `z`: latent `[..., input_size]`
    - output: probabilities `[..., num_experts + 1]` (last index is OUTPUT expert)
  - Training behavior:
    - requires `is_final`
    - `is_final=False`: OUTPUT masked out
    - `is_final=True`: only OUTPUT allowed
  - Eval behavior: all experts (including OUTPUT) are eligible.
- `Router = LatentRouter` compatibility alias

### Experts/MoE (`modules/model/expert.py`, `modules/model/moe.py`)
- `class ExpertModule(nn.Module)`
  - `forward(x)`: `[..., input_size] -> [..., output_size]`
  - `solve_from_batch(x, y, l2=1e-5)`:
    - expects 2D solve inputs (`[N, input_size]`, `[N, output_size]`)
    - inverts activation then solves linear weights
  - `consolidate(force=False, disable_grad=True, dtype=torch.float32)`: replaces solvable linear with plain `nn.Linear`
  - `enable_grad(enabled=False)` / `disable_grad()`
- `class ExpertModuleWithSkip(ExpertModule)`
  - Pre-norm + dropout + residual expert: `x + dropout(activation(linear(norm(x))))`
  - Requires `input_size == output_size`
- `class MixtureOfExperts(nn.Module)`
  - `prune_least_used()`: removes least-used expert and corresponding router head row
  - `forward(x, target=None, output_skew=0.0)`
    - Common input: `x` latent `[B, T, D]` (or `[B, D]`)
    - Training returns depend on curriculum phase (`steps_per_expert + 2` cycle):
      - normal phase: returns `output` (`[... , D]`)
      - add-expert phase: returns `(output, probs, target_idx)`
      - output-only phase: returns `(output, probs, target_idx)`
    - Inference (`eval`) returns `(output, probs)` where:
      - `output`: `[..., D]`
      - `probs`: `[..., num_experts + 1]`
  - `reset_step()`: resets internal curriculum step counter.

### Encoder/decoder (`modules/model/encoder.py`, `modules/model/decoder.py`)
- `@dataclass EncoderOutput`
  - `last_hidden_state`: `[B, T, H]`
  - `hidden_states`: optional tuple of per-layer tensors (`len = num_layers + 1` including embeddings)
- `class Gemma3Encoder(nn.Module)`
  - `forward(input_ids, attention_mask=None, position_ids=None, return_all_hidden_states=False) -> EncoderOutput`
  - Inputs:
    - `input_ids`: `[B, T]`
    - `attention_mask` (optional): `[B, T]`
    - `position_ids` (optional): `[B, T]`
  - Output:
    - `EncoderOutput.last_hidden_state`: `[B, T, hidden_size]`
    - `hidden_states` populated only when requested
  - `hidden_size` property: model hidden width from HF config.
- `class Decoder(nn.Module)`
  - `forward(x, context)`
    - `x`: latent `[B, T, hidden_size]`
    - `context`: encoder context `[B, T_ctx, hidden_size]` (typically same `T`)
    - output: `[B, T, output_size]`
  - `inverse(output, context)`
    - `output`: `[B, T, output_size]`
    - returns approximate/exact latent `[B, T, hidden_size]` depending on invertibility conditions.

### Invertible attention (`modules/model/invertible_modules.py`)
- `class InvertibleLinearAttention(nn.Module, InvertibleModule)`
  - `forward(x, other=None)`
    - `x`: `[B, T_q, D_in]`
    - `other`: `[B, T_kv, D_in]` (defaults to `x`)
    - output: `[B, T_q, D_out]`
  - `inverse(output, other)` / `auto_inverse(output, other)`
    - requires known `other`
    - inverse quality depends on activation invertibility and linear algebra conditions
  - `is_square`: `input_size == output_size`

### Losses (`modules/model/losses.py`)
- `class MatrixInvertabilityLoss(nn.Module)`
  - `forward(matrices)`
    - determinant/pinverse methods expect square `[..., N, N]`
    - `non_square_pinverse_method` supports `[..., M, N]`
    - returns scalar mean loss
  - `determinant_method(matrices)`
  - `pinverse_method(matrices)`
  - `non_square_pinverse_method(matrices)`

### Integrated model (`modules/model/transformer.py`)
- `class FinalTransformer(nn.Module)`
  - `forward(input_ids, target_vectors=None, attention_mask=None)`
    - Inputs:
      - `input_ids`: `[B, T]`
      - `target_vectors` (training only): `[B, T, vocab_size]`
      - `attention_mask` (optional): `[B, T]`
    - Outputs:
      - training mode: `(logits, router_loss)` where `logits` is `[B, T, vocab_size]`
      - eval mode: `logits` `[B, T, vocab_size]`
  - `sft_forward(input_ids, attention_mask=None)`
    - SFT path that uses inference-style routing loop while keeping gradients enabled
    - output: logits `[B, T, vocab_size]`.

---

## `modules/data`

### File/batch loading (`modules/data/dataloader.py`)
- `class FileLoader`
  - `__iter__()` / `__next__()`
  - `reset()`
  - `_get_next_file()`
  - `load_file(parquet_file_path)`
  - Usage: iterates parquet files under `root/subdir/file.parquet` and yields `pd.DataFrame`.
- `class DataLoader`
  - `load_next_file()`
  - `get_next_file()`
  - `get_next_batch(batch_size)`
  - `__iter__()` / `__next__()`
  - Usage: sequential dataframe batching with optional score filter and single-column projection.

### Embedding/vector datasets (`modules/data/vector_dataset.py`)
- `class _EmbeddingGemmaModel`
  - `encode(texts, convert_to_tensor=True, batch_size=64, show_progress_bar=False, precision='float32', normalize_embeddings=True)`
    - input: `list[str]`
    - output: embedding tensor `[N, embedding_dim]`
  - `similarity_speedy(text0, others)`
    - computes dot-similarity against many vectors.
- `class GemmaVectorDataset`
  - `from_texts(texts, model)`
  - `save(path)` / `load(path, model)`
  - `compute_embeddings(batch_size=64, show_progress_bar=True)`
  - `__len__()` / `__getitem__(idx)`
  - `get_similar_batch(batch_size, delta, text_only=False)`
    - returns either `list[str]` or `list[{"text", "embedding"}]`
- `class _EmbeddingLFM2ColBERTModel`
  - `encode_documents(...)`, `encode_queries(...)`, `build_index(...)`, `load_index()`, `retrieve(...)`
- `class LFM2ColBERTVectorDataset`
  - `from_texts(texts, model)`
  - `save(path)` / `load(path, model)`
  - `compute_embeddings(batch_size=32, show_progress_bar=True)`
  - `__len__()`
  - `get_similar_batch(batch_size, text_only=False, delta=0.5, return_embeddings=False)`
    - returns texts or list of dicts (`id`, `text`, optional `embedding`)
- `VectorDataset = LFM2ColBERTVectorDataset` (current alias used by imports)

### Tokenized iterable dataset (`modules/data/dataset.py`)
- `class Dataset(torch.utils.data.IterableDataset)`
  - `__iter__()` / `_batch_iterator()`
  - `_advance_to_next_file()`
  - `_load_vectorized_dataset(texts)`
  - `_extract_texts_and_embeddings(batch)`
  - `_maybe_to_device(device)`
  - Yields dict batches with:
    - `input_ids`: `[B, T]`
    - `attention_mask`: `[B, T]` (if tokenizer returns it)
    - `labels`: `[B, T]` with pad positions masked to `-100`
    - optional `embeddings`: `[B, E]`
    - optional `texts`: `list[str]`.

### Chat Template (`modules/data/chat_template.py`)
- `class Chat`
  - `__init__(tokenizer)`
  - `format_chat(messages)`
  - Input: `messages: list[{"role": str, "content": str}]`
  - Output: single prompt string formatted via tokenizer chat template.

---

## `modules/util`

### Linear probe experiment (`modules/util/linear_classifier.py`)
- `generate_synthetic_reasoning_data(num_samples=None)`
  - returns `(train_texts, train_labels, test_texts, test_labels)`
- `extract_hidden_states(model, tokenizer, texts, device)`
  - returns per-layer features list; each layer entry is `[N, H]` after pooling strategy
- `run_linear_probe(model_name, dataset_size=200)`
  - utility experiment: trains logistic regression probes over layers.

These are experimental helpers and not required by the core pretraining/SFT pipeline.

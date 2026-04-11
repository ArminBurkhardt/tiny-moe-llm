# API Reference

This document describes the currently usable classes, methods, and functions in this repository.

## `utils.py`

### Constants and paths
- `DIR`
  - `BASE_DIR`
  - `GEMMA_EMBEDDING_DIR`
  - `GEMMA_3_1B_DIR`
  - `GEMMA_3_270M_DIR`
  - `GEMMA_2_T5_270M_DIR`
  - `GEMMA_3_DIR`
  - `DATA_DIR`
  - `UFW_V1_4_DIR`
- `PATH = DIR`
- `FP64`, `FP32`

### Base interface
- `class InvertibleModule`
  - `inverse(y: torch.Tensor, **kwargs) -> torch.Tensor`
  - `auto_inverse(y: torch.Tensor, **kwargs) -> torch.Tensor`

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
  - `f(a: float, b: float) -> Callable[[torch.Tensor], torch.Tensor]`
  - `f_inv(a: float, b: float) -> Callable[[torch.Tensor], torch.Tensor]`
- `class InvertibleActivation(nn.Module, InvertibleModule)`
  - `forward(x)`
  - `inverse(y)`
  - `auto_inverse(y)`
- `class InvertibleLeakyReLUActivation(nn.Module, InvertibleModule)`
  - `forward(x)`
  - `inverse(y)`
  - `auto_inverse(y)`
- `class ShiftActivation(nn.Module, InvertibleModule)`
  - `forward(x)`
  - `inverse(y)`
  - `auto_inverse(y)`

### Core modules (`modules/model/modules.py`)
- `class LinearAttention(nn.Module)`
  - `forward(x, other=None)`
- `class MLP(nn.Module)`
  - `forward(x)`

### Linear/invertible layers (`modules/model/linear.py`)
- `class InvertibleLinear(nn.Module, InvertibleModule)`
  - `forward(x)`
  - `inverse(y)`
  - `approx_linear_inverse(y)`
  - `auto_inverse(y)`
  - `is_square`
- `class SolvableLinear(InvertibleLinear)`
  - `enable_grad(enabled=True)`
  - `disable_grad()`
  - `forward(x)`
  - `solve_from_batch(x, y, l2=1e-4)`
  - `auto_solve(x, y, l2=1e-4)`

### Router (`modules/model/router.py`)
- `class LatentRouter(nn.Module)`
  - `output_index`
  - `add_experts(k)`
  - `forward(z, is_final=None, output_skew=0.0)`
- `Router = LatentRouter` (compatibility alias)

### Experts/MoE (`modules/model/expert.py`, `modules/model/moe.py`)
- `class ExpertModule(nn.Module)`
  - `forward(x)`
  - `solve_from_batch(x, y, l2=1e-5)`
- `class MixtureOfExperts(nn.Module)`
  - `prune_least_used()`
  - `forward(x, target=None, output_skew=0.0)`
  - `reset_step()`

### Encoder/decoder (`modules/model/encoder.py`, `modules/model/decoder.py`)
- `class Gemma3Encoder(nn.Module)`
  - `forward(input_ids, attention_mask=None, position_ids=None, return_all_hidden_states=False)`
  - `hidden_size`
- `class Decoder(nn.Module)`
  - `forward(x, context)`
  - `inverse(output, context)`

### Invertible attention (`modules/model/invertible_modules.py`)
- `class InvertibleLinearAttention(nn.Module, InvertibleModule)`
  - `forward(x, other=None)`
  - `inverse(output, other)`
  - `auto_inverse(output, other)`
  - `is_square`

### Losses (`modules/model/losses.py`)
- `class MatrixInvertabilityLoss(nn.Module)`
  - `forward(matrices)`
  - `determinant_method(matrices)`
  - `pinverse_method(matrices)`
  - `non_square_pinverse_method(matrices)`

### Multi-token prediction (`modules/model/mtp.py`)
- `compute_mtp_loss(logits, labels, mtp_steps=1, mtp_lambda=1.0, ignore_index=-100)`

### Integrated model (`modules/model/transformer.py`)
- `class FinalTransformer(nn.Module)`
  - `forward(input_ids, target_vectors=None)`

---

## `modules/data`

### File/batch loading (`modules/data/dataloader.py`)
- `class FileLoader`
  - `__iter__()`
  - `__next__()`
  - `reset()`
  - `_get_next_file()`
  - `load_file(parquet_file_path)`
- `class DataLoader`
  - `load_next_file()`
  - `get_next_file()`
  - `get_next_batch(batch_size)`
  - `__iter__()`
  - `__next__()`

### Embedding dataset (`modules/data/vectorized_dataset.py`)
- `class _EmbeddingGemmaModel`
  - `encode(texts, convert_to_tensor=True, batch_size=64, show_progress_bar=False, precision='float32', normalize_embeddings=True)`
  - `similarity_speedy(text0, others)`
- `class VectorizedDataset`
  - `from_texts(texts, model)`
  - `save(path)`
  - `load(path, model)`
  - `compute_embeddings(batch_size=64, show_progress_bar=True)`
  - `__len__()`
  - `__getitem__(idx)`
  - `get_similar_batch(batch_size, delta, text_only=False)`

### Tokenized iterable dataset (`modules/data/dataset.py`)
- `class Dataset(torch.utils.data.IterableDataset)`
  - `__iter__()`
  - `_batch_iterator()`
  - `_advance_to_next_file()`
  - `_load_vectorized_dataset(texts)`
  - `_extract_texts_and_embeddings(batch)`
  - `_maybe_to_device(device)`

### Chat Template (`modules/data/chat_template.py`)
- `class Chat`
  - `__init__(self, tokenizer)`
  - `format_chat(messages)`

---

## `modules/util`

### Linear probe experiment (`modules/util/linear_classifier.py`)
- `generate_synthetic_reasoning_data(num_samples=None)`
- `extract_hidden_states(model, tokenizer, texts, device)`
- `run_linear_probe(model_name, dataset_size=200)`

These are utility experiment helpers, not part of the main training pipeline.

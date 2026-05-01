# API Reference

This document summarizes the current public/usable APIs and key script entry points.

## `utils.py`

### Constants and paths
- `BASE_DIR`
- `class DIR`
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
- `FP64`, `FP32`
- `logger`

### Interfaces
- `class InvertibleModule`
  - `inverse(y, **kwargs) -> Tensor`
  - `auto_inverse(y, **kwargs) -> Tensor`
- `class SolvableModule`
  - `solve_from_batch(x, y, **kwargs) -> Tensor`

---

## `modules/model` (`modules/model/__init__.py`)

### Exported API
- `ParameterizedSigmoid`
- `InvertibleActivation`
- `LinearAttention`
- `MLP`
- `Gemma3Encoder`
- `Gemma4Encoder`
- `Decoder`
- `LatentRouter`
- `SolvableLinear`
- `InvertibleLinear`
- `InvertibleLinearAttention`
- `MatrixInvertabilityLoss`
- `MixtureOfExperts`
- `PerLayerEmbedding`
- `RoPE`
- `RotaryPositionEmbeddingsForAttention`
- `ExpertModuleWithSkip`
- `ExpertModuleWithSkipAndEmbedding`
- `GroupedQueryAttention`

### Activations (`modules/model/activations.py`)
- `class ParameterizedSigmoid`
  - `f(a, b)`, `f_inv(a, b)`
- `class InvertibleActivation(nn.Module, InvertibleModule)`
  - `forward(x)`, `inverse(y)`, `auto_inverse(y)`
- `class InvertibleLeakyReLUActivation(nn.Module, InvertibleModule)`
  - `forward(x)`, `inverse(y)`, `auto_inverse(y)`
- `class ShiftActivation(nn.Module, InvertibleModule)`
  - `forward(x)`, `inverse(y)`, `auto_inverse(y)`

### Attention blocks
- `modules/model/modules.py`
  - `class LinearAttention`
    - `forward(x, other=None) -> [..., T_q, D_out]`
  - `class MLP`
    - `forward(x) -> [..., output_size]`
  - `class MultiHeadAttention`
    - `forward(hidden_states, other=None, attention_mask=None) -> [B, T, H]`
- `modules/model/attention.py`
  - `class GroupedQueryAttention`
    - `repeat_kv(hidden_states, n_rep)`
    - `forward(hidden_states, attention_mask=None, use_causal_mask=True)`

### Embeddings / position encoding (`modules/model/embeddings.py`)
- `class PerLayerEmbedding`
  - `forward(input_ids)` -> `[B, S, D]`
- `class RoPE`
  - `forward(x)` -> rotated `[B, S, D]`
- `class RotaryPositionEmbeddingsForAttention`
  - `forward(x, seq_len=None)` -> `(cos, sin)` caches
- helpers:
  - `rotate_half(x)`
  - `apply_rotary_pos_emb(q, k, cos, sin)`

### Linear/invertible layers (`modules/model/linear.py`)
- `class InvertibleLinear(nn.Module, InvertibleModule)`
  - `forward(x)`
  - `inverse(y)` (square/full-rank only)
  - `approx_linear_inverse(y)` (pseudo-inverse)
  - `auto_inverse(y)`
  - `is_square`
- `class SolvableLinear(InvertibleLinear, SolvableModule)`
  - `enable_grad(enabled=True)` / `disable_grad()`
  - `solve_from_batch(x, y, l2=1e-4)`
  - `auto_solve(x, y, l2=1e-4)`

### Router (`modules/model/router.py`)
- `class LatentRouter`
  - `output_index`
  - `add_experts(k)`
  - `forward(z, is_final=None, output_skew=0.0)` -> probabilities over `[num_experts + 1]`
- `Router = LatentRouter` compatibility alias

### Experts and MoE (`modules/model/expert.py`, `modules/model/moe.py`)
- `class ExpertModule`
  - `forward(x)`
  - `solve_from_batch(x, y, l2=1e-5)`
  - `consolidate(force=False, disable_grad=True, dtype=torch.float32)`
  - `enable_grad(enabled=False)` / `disable_grad()`
- `class ExpertModuleWithSkip(ExpertModule)`
  - residual pre-norm expert:
    - `x + dropout(activation(linear(norm(x))))`
  - `solve_from_batch` with pre-norm and residual target transform
- `class ExpertModuleWithSkipAndEmbedding(ExpertModuleWithSkip)`
  - token-conditioned expert:
    - `forward(x, input_ids)`
    - `solve_from_batch(x, y, input_ids, l2=1e-5)`
- `class SelfAttentionExpert`
  - `forward(x, **kwargs)`
- `class CrossAttentionExpert`
  - `forward(x, context)`
- `class MixtureOfExperts`
  - constructor supports:
    - `router`, optional `experts`, optional `special_experts`, optional `expert_template`
    - `steps_per_expert`, `hidden_size` (for post-norm)
  - `forward(x, target=None, output_skew=0.0, *args, **kwargs)`
    - training: normal routing / add-expert / output-only phases
    - eval: weighted combination including OUTPUT identity contribution
  - `prune_least_used()`
  - `reset_step()`

### Information retrieval expert (`modules/model/information_retrieval.py`)
- `class InformationRetrievalLayer`
  - `forward(query)`
- `class InformationRetrievalModule`
  - `reset_keys()`
  - `forward(x, return_weights=False, **kwargs)`

### Encoder/decoder (`modules/model/encoder.py`, `modules/model/decoder.py`)
- `@dataclass EncoderOutput`
  - `last_hidden_state`
  - `hidden_states`
- `class Gemma3Encoder`
  - `forward(input_ids, attention_mask=None, position_ids=None, return_all_hidden_states=False) -> EncoderOutput`
  - `hidden_size` property
- `class Gemma4Encoder(Gemma3Encoder)`
  - same forward contract as Gemma3 wrapper
- `class Decoder`
  - `forward(x, context)`
  - `inverse(output, context)`

### Invertible attention (`modules/model/invertible_modules.py`)
- `class InvertibleLinearAttention(nn.Module, InvertibleModule)`
  - `forward(x, other=None)`
  - `inverse(output, other)`
  - `auto_inverse(output, other)`
  - `is_square`

### Losses (`modules/model/losses.py`)
- `class MatrixInvertabilityLoss`
  - `forward(matrices)`
  - `determinant_method(matrices)`
  - `pinverse_method(matrices)`
  - `non_square_pinverse_method(matrices)`

### Integrated model (`modules/model/transformer.py`)
- `class FinalTransformer`
  - `forward(input_ids, target_vectors=None, attention_mask=None)`
    - train mode: returns `(logits, router_loss)`
    - eval mode: returns `logits`
  - `sft_forward(input_ids, attention_mask=None)`
    - inference-style routing loop with gradients enabled

---

## `modules/data`

### File/batch loading (`modules/data/dataloader.py`)
- `class FileLoader`
  - `__iter__()`, `__next__()`, `reset()`, `_get_next_file()`, `load_file(path)`
- `class DataLoader`
  - `load_next_file()`
  - `get_next_file()`
  - `get_next_batch(batch_size)`
  - `__iter__()`, `__next__()`

### Embedding/vector datasets (`modules/data/vector_dataset.py`)
- `class _EmbeddingGemmaModel`
  - `encode(...)`
  - `similarity_speedy(text0, others)`
- `class GemmaVectorDataset`
  - `from_texts(texts, model)`
  - `save(path)`, `load(path, model)`
  - `compute_embeddings(batch_size=64, show_progress_bar=True)`
  - `__len__()`, `__getitem__(idx)`
  - `get_similar_batch(batch_size, delta, text_only=False)`
- `class _EmbeddingLFM2ColBERTModel`
  - `encode_documents(...)`, `encode_queries(...)`
  - `build_index(document_ids, document_embeddings)`, `load_index()`
  - `retrieve(queries_embeddings, k=10)`
- `class LFM2ColBERTVectorDataset`
  - `from_texts(...)`
  - `save(path)`, `load(path, model)`
  - `compute_embeddings(...)`
  - `__len__()`
  - `get_similar_batch(batch_size, text_only=False, delta=0.5, return_embeddings=False)`
- `VectorDataset = GemmaVectorDataset`

### Iterable tokenized dataset (`modules/data/dataset.py`)
- `class Dataset(torch.utils.data.IterableDataset)`
  - `__iter__()`, `_batch_iterator()`
  - `_advance_to_next_file()`
  - `_load_vectorized_dataset(texts)`
  - `_extract_texts_and_embeddings(batch)`
  - `_maybe_to_device(device)`
  - yields dicts with:
    - `input_ids`: `[B, T]`
    - `attention_mask`: `[B, T]` (optional)
    - `labels`: `[B, T]` (`-100` on masked tokens)
    - optional `embeddings`, optional `texts`

### Chat template helper (`modules/data/chat_template.py`)
- `class Chat`
  - `format_chat(messages)`

---

## `modules/util`

### Linear probe utility (`modules/util/linear_classifier.py`)
- `generate_synthetic_reasoning_data(num_samples=None)`
- `extract_hidden_states(model, tokenizer, texts, device)`
- `run_linear_probe(model_name, dataset_size=200)`

---

## Top-level training/config scripts

### `training_config.py`
- `EXPERT_TEMPLATES`
- `@dataclass PretrainConfig`
  - `as_dict()`
  - `copy()`

### `pretrain.py`
- CLI / setup:
  - `parse_args()`
  - `build_model(args, vocab_size, latent_dim)`
  - `build_optimizer(model, args)`
- lifecycle helpers:
  - `_enable_expert_grad(expert)`
  - `_add_expert_to_optimizer(optimizer, expert, lr)`
- training:
  - `train_step(model, batch, optimizer, args, vocab_size, device)`
  - `save_checkpoint(path, model, optimizer, step, args, config)`
  - `main()`

### `posttrain.py`
- CLI / setup:
  - `parse_args()`
  - `build_model(args, vocab_size, latent_dim)`
  - `build_datasets(args)`
- dataset:
  - `class SFTDataset`
    - `__iter__()`, `_batch_iterator()`, `_sample_iterator()`, `_tokenize_conversation(messages)`
- training:
  - `sft_step(model, batch, optimizer, args, vocab_size, device)`
  - `save_checkpoint(path, model, optimizer, step, epoch, args)`
  - `main()`

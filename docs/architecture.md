# Architecture

`TinyMoETransformer` ([modules/model/transformer.py](../modules/model/transformer.py)) is the
top level model. A forward pass runs four stages:

```
input_ids
   │
   ▼
┌─────────────────────────┐
│ Gemma4TextModel         │  dense decoder: embed -> N × (GQA + MLP + PLE) -> RMSNorm
└─────────────────────────┘
   │  last_hidden_state
   ▼
┌─────────────────────────┐
│ LoopMixtureOfExperts    │  n_loops × (route -> weighted experts -> norm)
└─────────────────────────┘
   │
   ▼  RMSNorm
   ├──────────────► SmallLMHead ──► logits            [B, S, vocab]
   └──────────────► MTPHead     ──► extra-token hidden [B, S, k_mtp, H/2]
```

## 1. Dense decoder - `Gemma4TextModel`

[modules/model/gemma4.py](../modules/model/gemma4.py). A Gemma4-style dense transformer:

- **Embedding** scaled by `sqrt(hidden_size)`. No `padding_idx` - with this tokenizer `pad == eos`
  (id 1) must remain trainable
- **Decoder layer** (pre-norm): `x + Attn(RMSNorm(x))`, then `x + MLP(RMSNorm(x))`, each gated by a
  learned `layer_scalar`.
- **Attention**: grouped-query attention (`num_key_value_heads = num_heads // 4`) with RoPE, run
  through the document-packed varlen kernel (see §5)
- **MLP**: SwiGLU (`down(SiLU(gate) * up)`)
- **Per-layer embeddings (PLE)**: a separate embedding table produces a small vector per token per
  layer, each layer gates it in via `x + sigmoid(gate(x)) * proj(ple)`. Adds token-specific,
  depth-specific signal

Output is an `EncoderOutput` ([utils.py](../modules/model/utils.py)) carrying `last_hidden_state`
and the per-layer hidden states

## 2. Looped MoE - `LoopMixtureOfExperts`

Applied to the decoder output, optionally with a projected **MoE per-layer
embedding** (`moe_embeddings`) supplied as the cross-attention `other` stream. Runs `n_loops`
routing iterations over a shared expert pool. Fully described in [moe.md](moe.md)

## 3. LM head - `SmallLMHead`

[modules/model/modules.py](../modules/model/modules.py). The output projection to a 129k-token
vocabulary is expensive, so its **factored**: one shared `hidden -> hidden` projection, then `factor`
independent `(hidden/factor)->(vocab/factor)` heads whose outputs are concatenated. This cuts the
heads parameter/compute cost by roughly `factor` versus a dense `hidden -> vocab` matrix

## 4. Multi-token prediction - `MTPHead`

[modules/model/mtp.py](../modules/model/mtp.py). Besides the next token, the model predicts
`mtp_num_extra_tokens` further out tokens from the same hidden state. The head expands the hidden
state, splits it into one slice per extra token, and (in the default delayed mode) returns
`[B, S, k_mtp, H/2]` hidden states rather than full logits. The LM head is applied later, during
loss computation, to keep the VRAM footprint down. See [training.md](training.md) for the loss

## 5. Document-packed attention

[modules/model/attention.py](../modules/model/attention.py). During pretraining many documents are
packed into each `max_length` sequence. A token must attend only within its own document, causally.
Instead of a dense `[B,1,S,S]` mask (which disables the flash backend), the packing is expressed as
`cu_seqlens` - cumulative segment boundaries over the flattened `B*S` axis - and passed to
`flash_attn_varlen_func`. Cost scales with `sum(segment_len^2)` instead of `S^2` (quadratic per segment vs entire sequence), and no mask is
materialized. Without flash-attn installed, a slower SDPA fallback rebuilds the block
mask.

`cu_seqlens` is derived from a batch-aligned `[B, S]` `document_ids` tensor via
`cu_seqlens_from_doc_ids`. `document_ids` (not `cu_seqlens`) travels through the dataloader so
`accelerate`s batch splitting handles it like `input_ids`

## Parameter count

The default config yields ~243M parameters. The bulk sit in the 36 MLP experts and the embedding /
LM-head tables, the attention/IR experts and the dense decoder are comparatively small.

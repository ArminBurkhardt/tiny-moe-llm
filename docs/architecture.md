# Architecture

`TinyMoETransformer` ([modules/model/transformer.py](../modules/model/transformer.py)) is the
top level model. A forward pass runs four stages, plus an optional fifth on the side when retrieved
evidence is attached:

```
input_ids                                    evidence_ids (optional)
   │                                              │
   ▼                                              ▼
┌─────────────────────────┐              ┌─────────────────────────┐
│ Gemma4TextModel         │  dense       │ the SAME decoder, once  │  chunk-causal, per-chunk
│ embed -> 8 × (GQA+MLP+  │  decoder     │ over the evidence tokens│  positions, no LM head
│ PLE) -> RMSNorm         │              └─────────────────────────┘
└─────────────────────────┘                       │  evidence states, cached across loops
   │  last_hidden_state                           │
   ▼                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ LoopMixtureOfExperts    n_loops × (route -> experts -> norm)     │
│   always-on: shared_mlp + shared_attn (+ evidence reader)        │
│   routed:    self-attn | cross-attn | IR (selector) | 32 × MLP   │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼  RMSNorm at EVERY loop  -> [loops_run, B, S, H]
   ├──────────────► SmallLMHead ──► per-loop logits / CE        [B, S, vocab]
   └──────────────► MTPHead     ──► extra-token hidden (final loop) [B, S, k_mtp, H/2]
```

## 1. Dense decoder - `Gemma4TextModel`

[modules/model/gemma4.py](../modules/model/gemma4.py). A Gemma4-style dense transformer:

- **Embedding** scaled by `sqrt(hidden_size)`. No `padding_idx`: with this tokenizer `pad == eos`
  and id 0 is BOS, and a `padding_idx` froze BOS at zero.
- **Decoder layer** (pre-norm): `x = x + Attn(RMSNorm(x))`, then `x = x + MLP(RMSNorm(x))`, then
  the PLE add below, and finally the whole layer output is multiplied once by a learned scalar
  `layer_scalar` (init 1, excluded from weight decay). The scalar scales the layer's output,
  residual included, not each sublayer's update.
- **Attention**: grouped-query attention (`num_key_value_heads = num_heads // 4`) with RoPE, run
  through the document-packed varlen kernel (see §5).
- **MLP**: SwiGLU (`down(SiLU(gate) * up)`).
- **Per-layer embeddings (PLE)**: a separate embedding table produces a small vector per token per
  layer, and each layer adds it after the MLP as `x + sigmoid(gate_proj(post_feedforward_layernorm(x))) * ple_proj(ple)`:
  the gate reads the normed state, the add goes onto the un-normed residual.
  `per_layer_embeddings_size: 0` builds no PLE tensors in the decoder.

Output is an `EncoderOutput` ([modules/model/utils.py](../modules/model/utils.py)) carrying
`last_hidden_state` and the per-layer hidden states.

## 2. Looped MoE - `LoopMixtureOfExperts`

Applied to the decoder output. Runs `n_loops` routing iterations over a shared expert pool; each
loop's output is a **residual update** scaled by a per-loop `loop_scale`, and the router is
conditioned on the loop index through a sinusoidal encoding so consecutive loops route differently.
The cross-attention expert reads `other`, a projected **MoE per-layer embedding**
(`moe_embeddings`) of the same tokens. Fully described in [moe.md](moe.md).

## 3. The evidence port

Two halves that arrive together with `scripts/migrate_evidence_port.py` and are absent from a
checkpoint that predates it (`utils.model_params_for_state_dict` infers `evidence_port` from the
state dict):

- **The encoder** (`TinyMoETransformer._encode_evidence`) embeds the retrieved chunks with the
  model's own `embed_tokens` and runs them through the dense decoder once per forward: causal
  *within each chunk* (its own `cu_seqlens` from the chunk ids), positions restarting at 0 per chunk,
  no LM head, and the result cached across the loops. No new parameters. `evidence_encoder: true`
  in `config.yaml`; an int runs only the first N layers; `false` falls back to the raw per-token
  `moe_embeddings` (kept for the encoded-vs-raw comparison only: a reader over uncontextualized
  token identities cannot find "the token after *born in*"). Cost is published separately as
  ~118M FLOP per evidence token, deliberately not folded into `flops_per_token_fwd`.
- **The reader** reads the evidence states at every loop. Two kinds exist, chosen when the port
  is built and carried by the checkpoint:
  - **The key/value reader** (`evidence_reader="kv"`, `sft.py --reader-kv`, marker buffer
    `moe.evidence_reader_kv`): the measured reader since 2026-10-03 and the pilot's. It has no
    module of its own. The evidence states go through `shared_attn`'s own norm, the per-chunk gate
    (applied after the norm, which would otherwise divide it out) and `shared_attn`'s own
    `k_proj`/`v_proj`, and become leading keys of each query segment under one softmax with that
    segment's own causal keys (`attention.prefix_varlen_attention`: the two key sets interleaved
    `[prefix_s, own_s]`, one flash call). Evidence token `i` of segment `s` is rotated at
    `start_s - E_s + i` (`evidence.prefix_position_ids`, negative positions allowed), so it reads as
    the text right before the document. `evidence_loop_scale[loop]` (one-init) scales the evidence
    values. Because the read shares the softmax, it is **not neutral with evidence attached**; with
    no evidence attached it is bit-identical to the model without the port. No KV cache with
    evidence (asserted). `last_reader_output` is `shared_attn`'s output in this mode.
  - **The cross reader** (`moe.shared_evidence`, the legacy mode, still loadable): an always-on
    cross-attention over the evidence states, seeded into the loop's accumulator beside
    `shared_mlp`/`shared_attn` at every loop with its own per-loop query bias
    (`evidence_query_bias`) and per-loop gain (`evidence_loop_scale`). Its `o_proj` is zero-init,
    so a fresh port is neutral even with evidence attached.

  Either way evidence is bidirectional and has its own segments (one per query document, paired by
  position with the query side's `cu_seqlens`), so a document only ever sees its own chunks; a
  document that retrieved nothing has a zero-length segment.
- **The selector** is the IR expert: the external chunk vectors (bge-small, 384-d) enter its softmax
  through square orthogonal adapters and a learned per-source scale, so the fraction of read mass
  that went external is a groundedness signal computed for every token. The selector's per-chunk
  mass gates the reader's evidence states through a zero-init scalar, which is what ties the two
  halves together.

**With no evidence attached the forward is bit-identical to the model without the port**, asserted
in `tests/test_evidence_port.py`. That property is what lets one checkpoint serve both modes and
makes the replay fraction of a finetune protect the trunk rather than train the port.

## 4. LM head - `SmallLMHead`

[modules/model/modules.py](../modules/model/modules.py). The output projection to a 65536-token
vocabulary is **factored**: one shared `hidden -> hidden` projection, then `factor` independent
`(hidden/factor) -> (vocab/factor)` blocks whose outputs are concatenated. This cuts the head's
parameter/compute cost by roughly `factor` at the price of rank (192 per vocab quarter). Whether
that price is worth paying is [NEXT.md](plans/NEXT.md) 7d's question; it applies at every loop for
per-loop CE.

## 5. Multi-token prediction - `MTPHead`

[modules/model/mtp.py](../modules/model/mtp.py). Besides the next token, the model predicts
`mtp_num_extra_tokens` further-out tokens from the final loop's hidden state. The head expands the
hidden state, splits it into one slice per extra token, and (in the default delayed mode) returns
`[B, S, k_mtp, H/2]` hidden states rather than logits. The LM head is applied later, inside the
chunked, checkpointed loss, to keep the VRAM footprint down. `skip_mtp=True` skips the head
entirely where its output would be discarded (every eval, non-drafting inference); the logits are
bit-identical either way. See [training.md](training.md) for the loss.

## 6. Document-packed attention

[modules/model/attention.py](../modules/model/attention.py). During training many documents are
packed into each `max_length` sequence. A token must attend only within its own document, causally.
Instead of a dense `[B,1,S,S]` mask (which disables the flash backend), the packing is expressed as
`cu_seqlens` (cumulative segment boundaries over the flattened `B*S` axis) and passed to
`flash_attn_varlen_func`. Cost scales with `sum(segment_len^2)` instead of `S^2`, and no mask is
materialized. Without flash-attn installed, a slower SDPA fallback rebuilds the block mask.
`prefix_varlen_attention` is the key/value reader's variant: each query segment also attends, in
full, to its own evidence segment placed in front of its keys, under the same softmax.

`cu_seqlens` is derived from a batch-aligned `[B, S]` `document_ids` tensor via
`cu_seqlens_from_doc_ids`. `document_ids` (not `cu_seqlens`) travels through the dataloader so
`accelerate`'s batch splitting handles it like `input_ids`. The cross reader passes a separate
`cu_seqlens_k` for its key side and `causal=False`; the two lists must carry the same number of
segments, because flash pairs them by position and never checks.

## Parameter count

The default config yields **383.5M parameters, 224.2M active per token** (155.0M excluding the
three vocab tables). The bulk sits in the 32 MLP experts (only 2 of which run per token) and in the
50.3M-parameter IR table, which is a lookup rather than a matmul; the port itself is 1.8M. Forward
compute is ~502M FLOP/token at a fully packed 4096-token sequence (body 276M, heads 100M, attention
126M), plus ~118M FLOP per evidence token. The 16B-token run was trained at 332M / 173M with an
`8192 × 128` exact-read table and no port; `TinyMoETransformer.__init__` prints the live numbers.

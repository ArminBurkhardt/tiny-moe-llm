# Model Architecture

This document describes the current `FinalTransformer` implementation in `modules/model/transformer.py`.

## High-level flow

```
input_ids [B,S]
  └─ Gemma3Encoder(target_layer=12) -> context [B,S,D]
      └─ LayerNorm + Dropout (forward train/eval path)
          └─ z = context.clone()
              └─ MixtureOfExperts recurrent routing
                  ├─ LatentRouter: probs over (all experts + OUTPUT)
                  ├─ weighted expert combination
                  └─ post LayerNorm in normal/eval routing paths
                      └─ Decoder(z, context) -> logits [B,S,V]
```

`B` = batch size, `S` = sequence length, `D` = latent dimension, `V` = vocabulary size.

## Expert composition in the current model

By default, `FinalTransformer` builds:

- **Standard experts**: deep copies of `ExpertModuleWithSkipAndEmbedding`
  - Uses token-conditioned `PerLayerEmbedding(input_ids)` before pre-norm/linear/activation.
  - Residual form: `x + dropout(activation(linear(norm(x + embed))))`.
- **Special experts**:
  - `num_attention_experts` instances of `SelfAttentionExpert`
  - one `InformationRetrievalModule` (memory-style retrieval expert)
- **OUTPUT expert**: implicit identity branch handled by the router/MoE logic (not stored as a module in `self.experts`).

## Router and MoE behavior

### Router (`LatentRouter`)

- Backbone:
  - `LayerNorm -> Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> LinearAttention`
- Head projects to `num_experts + 1` logits (`+1` is OUTPUT expert).
- `output_skew` is added to the OUTPUT logit before softmax.
- Training masking:
  - `is_final=False`: OUTPUT masked out.
  - `is_final=True`: only OUTPUT allowed.

### MoE cycle (`MixtureOfExperts`, training mode)

Cycle length is `steps_per_expert + 2`:

1. **Normal routing phase** (`cycle_pos < steps_per_expert`)
   - Route to existing experts (excluding OUTPUT via mask), weighted sum, then post-norm.
2. **Add-expert phase** (`cycle_pos == steps_per_expert`)
   - New expert is solved from `(x, target)` and appended.
   - Router is trained to prefer this new expert.
3. **Output-only phase** (`cycle_pos == steps_per_expert + 1`)
   - Router is trained to pick OUTPUT (identity path).

Additional lifecycle behavior:

- `prune_least_used()` removes the least-used non-special expert and shrinks router head accordingly.

## Decoder

`Decoder` layer sequence:

1. `InvertibleLinear(D,D)`
2. `InvertibleLeakyReLUActivation`
3. `InvertibleLinear(D,D)`
4. `InvertibleActivation`
5. `InvertibleLinearAttention(D,D, activation=ShiftActivation(InvertibleActivation(a=0.9,b=0.1), shift=0.1))`
6. `InvertibleActivation`
7. `InvertibleLinear(D,V)`

Training uses `decoder.inverse(target_vectors, context)` to compute the solve target for expert addition.

## Forward modes

### `forward(..., target_vectors=...)` (pretraining mode)

- Uses encoder output **with** `encoder_norm` and `encoder_dropout`.
- Runs MoE training cycle.
- Returns `(logits, router_loss)`.

### `forward(...)` in eval mode (inference)

- Uses encoder output **with** `encoder_norm` and dropout disabled by eval mode.
- Recurrent loop with increasing `output_skew = loop_count * skew_factor`.
- Early exit when mean OUTPUT probability `> 0.5`.
- Returns `logits`.

### `sft_forward(...)` (SFT path)

- Uses inference-style MoE routing loop with gradients enabled.
- Temporarily sets `self.moe.training = False` and `self.moe.router.training = False` to avoid training-time routing masks/expert lifecycle.
- Returns `logits`.

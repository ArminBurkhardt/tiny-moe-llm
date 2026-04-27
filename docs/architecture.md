# Model Architecture

This document describes the data flow of the `FinalTransformer` model, including all
normalization and dropout placements.

---

## High-Level Overview

```
Input token IDs  [B, S]
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Gemma3Encoder  (fine-tuned, target_layer=12)                     │
│  Output: context  [B, S, D]                                       │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
  LayerNorm(D)          ← encoder output normalisation
        │
        ▼
  Dropout(p)            ← encoder output regularisation
        │
        ▼  context = z (cloned for the recurrent loop)
┌───────────────────────────────────────────────────────────────────┐
│  MixtureOfExperts recurrent loop  (training or inference)         │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  LatentRouter                                               │  │
│  │    LayerNorm(D) → Linear(D,H) → GELU → Dropout              │  │
│  │    → Linear(H,H) → GELU → Dropout → LinearAttention(H)      │  │
│  │    → Linear head  →  softmax(logits + output_skew)          │  │
│  │  Output: probability distribution over experts + OUTPUT     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                │                                                  │
│       ┌────────┴──────────────────────────────────────────────┐   │
│       │  ExpertModuleWithSkip  (one per active expert)        │   │
│       │    LayerNorm(D)        ← per-expert pre-norm          │   │
│       │    → SolvableLinear(D, D)                             │   │
│       │    → InvertibleActivation (parameterised sigmoid)     │   │
│       │    → Dropout(p)        ← contribution regularisation  │   │
│       │    + x                 ← residual / skip connection   │   │
│       └───────────────────────────────────────────────────────┘   │
│                │                                                  │
│       Weighted sum over all active experts (training: old only;   │
│       inference: old + OUTPUT identity)                           │
│                │                                                  │
│       LayerNorm(D)   ← post-MoE normalisation (MixtureOfExperts)  │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼  (final z after recurrent loop)
┌───────────────────────────────────────────────────────────────────┐
│  Decoder  (fully invertible; trained by back-prop)                │
│    InvertibleLinear(D, D)                                         │
│    → InvertibleLeakyReLUActivation                                │
│    → InvertibleLinear(D, D)                                       │
│    → InvertibleActivation                                         │
│    → InvertibleLinearAttention(D, D)  (cross-attention: z × ctx)  │
│    → InvertibleActivation                                         │
│    → InvertibleLinear(D, O)           (final projection)          │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
Output logits / embeddings  [B, S, O]
```

**Notation:**  `B` = batch, `S` = sequence length, `D` = `hidden_size`,
`H` = router hidden size, `O` = `output_dim`, `p` = dropout probability.

---

## Component Descriptions

### 1. Gemma3Encoder
- Wraps a pretrained Gemma 3 transformer.
- Hidden states are extracted at a fixed target layer (default: layer 12).
- The encoder is fine-tuned end-to-end (all parameters have `requires_grad=True`).
- Output shape: `[B, S, D]`.

### 2. Encoder-output normalisation and dropout
```
context = encoder(input_ids).last_hidden_state  # [B, S, D]
context = LayerNorm(D)(context)
context = Dropout(p)(context)
z       = context.clone()
```
- **Why here:** The encoder's internal representation can have arbitrary scale.
  A `LayerNorm` stabilises the latent distribution before it enters the MoE loop
  and the decoder cross-attention.
- **Dropout** prevents the MoE and decoder from over-fitting to specific encoder
  features, acting as input regularisation for the recurrent processing block.

### 3. LatentRouter
- A two-layer MLP with a `LinearAttention` head that maps `z` to a probability
  distribution over the `N` active experts plus a special **OUTPUT** expert.
- **Normalisation and dropout** are placed inside the router backbone:
  ```
  LayerNorm(D) → Linear(D,H) → GELU → Dropout
               → Linear(H,H) → GELU → Dropout → LinearAttention(H)
  → head: Linear(H, N+1)
  ```
- The router's internal `LayerNorm` ensures stable gradient flow through the
  routing decision, regardless of the scale of the incoming latent.
- Training masks:
  - `is_final=False` — OUTPUT expert logit is set to `-inf`.
  - `is_final=True` — all non-OUTPUT logits are set to `-inf`.
- Inference: skew factor linearly increases the OUTPUT logit with each recurrent
  step, biasing early termination.

### 4. ExpertModuleWithSkip (chosen expert for the final model)
```
output = x + Dropout(p)(InvertibleActivation(SolvableLinear(LayerNorm(D)(x))))
```
Step-by-step:
1. **Pre-LayerNorm:** `x_norm = LayerNorm(D)(x)`
   - Normalises the input before the linear projection, following the pre-norm
     style of modern transformers (e.g., GPT-2, PaLM).  Prevents covariate
     shift across recurrent expert calls.
2. **SolvableLinear:** `x_linear = SolvableLinear(D, D)(x_norm)`
   - A closed-form solvable linear layer.  During training, when a new expert is
     added, its weights are solved directly without back-prop:
     `linear(norm(x)) = activation⁻¹(y − x)`.
   - The pre-norm input `norm(x)` is used as the design matrix so that the solve
     is consistent with the forward pass.
3. **InvertibleActivation:** `x_act = σ(x_linear)`
   - Parameterised sigmoid mapping `ℝ → (−b, a)`.  Default `a=b=1` so the
     activation output is bounded in `(−1, 1)`.
4. **Dropout:** `x_act = Dropout(p)(x_act)`
   - Applied to the activated expert contribution *before* the residual addition.
     This regularises the magnitude of the expert's influence on `z` without
     disturbing the skip path.
5. **Residual addition:** `output = x + x_act`
   - Skip connection ensures gradient flows directly through the identity path
     and allows the expert to learn only the incremental transformation.

**Solve consistency:**  `solve_from_batch(x, y)` applies `norm(x)` to obtain the
linear-layer input that matches the forward pass.  Dropout is excluded from the
solve (it is a closed-form regression, not a stochastic optimisation step).

### 5. MixtureOfExperts — post-norm
```
output = sum_i( prob_i * expert_i(z) )
output = LayerNorm(D)(output)           ← post-MoE normalisation
```
- After the weighted combination of expert outputs, a single `LayerNorm` is
  applied before the result is fed back into the recurrent loop or passed to the
  decoder.
- **Why here:** Expert outputs are accumulated with probability weights that can
  vary widely during training.  The post-norm prevents the latent vector from
  growing unbounded across recurrent steps and stabilises the gradient signal
  back through the router.

### 6. Decoder
- Fully invertible stack used to project the final latent `z` to output space.
- The invertibility is essential for computing the closed-form solve target:
  `z_target = Decoder⁻¹(y_target, context)`.
- **No additional normalisation or dropout** is added here because:
  - The invertibility constraint requires precise numerical operations;
    normalisation layers inside the decoder would break exact inversion.
  - The decoder receives a well-normalised `z` (after the post-MoE LayerNorm)
    and `context` (after the encoder LayerNorm), so additional normalisation
    would be redundant.
- The cross-attention in `InvertibleLinearAttention` uses `context` as keys/
  values and `z` as queries, grounding each output token in the encoder's
  representation.

---

## Normalisation Placement Summary

| Location | Type | Purpose |
|---|---|---|
| After encoder output | `LayerNorm(D)` | Stabilise latent scale before MoE |
| After encoder output | `Dropout(p)` | Regularise encoder feature usage |
| Router backbone (start) | `LayerNorm(D)` | Stable routing decision gradients |
| Router backbone (×2) | `Dropout` | Regularise routing MLP |
| Expert (input) | `LayerNorm(D)` | Pre-norm before linear projection |
| Expert (output) | `Dropout(p)` | Regularise expert contribution magnitude |
| After MoE combination | `LayerNorm(D)` | Bound latent growth across recurrent steps |

---

## Training vs Inference Data Flow

### Training
```
input_ids → Encoder → LayerNorm → Dropout → z
  ┌─ Recurrent loop ─────────────────────────────────────────────┐
  │  while not done:                                             │
  │    if cycle_pos < steps_per_expert:   # normal routing       │
  │        probs = Router(z, is_final=False)                     │
  │        z     = sum_i(prob_i * expert_i(z))                   │
  │        z     = LayerNorm(z)                                  │
  │    elif cycle_pos == steps_per_expert:  # add expert         │
  │        new_expert.solve_from_batch(z, z_target)              │
  │        z     = new_expert(z)           (no post-norm here)   │
  │        probs = Router(z, is_final=False)                     │
  │        loss += CrossEntropy(probs, new_expert_idx)           │
  │        break                                                 │
  │    else:  # output phase                                     │
  │        z     = z  (identity OUTPUT expert)                   │
  │        probs = Router(z, is_final=True)                      │
  │        loss += CrossEntropy(probs, output_idx)               │
  │        break                                                 │
  └──────────────────────────────────────────────────────────────┘
z → Decoder(z, context) → output  [B, S, O]
return output, router_loss
```

### Inference
```
input_ids → Encoder → LayerNorm → Dropout(disabled) → z
  ┌─ Recurrent loop ─────────────────────────────────────────────┐
  │  for step in range(max_recurrence):                          │
  │    skew  = step * skew_factor                                │
  │    z, probs = MoE(z, output_skew=skew)   # includes post-norm│
  │    if probs[OUTPUT].mean() > 0.5: break                      │
  └──────────────────────────────────────────────────────────────┘
z → Decoder(z, context) → output  [B, S, O]
return output
```


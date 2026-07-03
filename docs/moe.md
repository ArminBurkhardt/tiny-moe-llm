# Looped Mixture-of-Experts

`LoopMixtureOfExperts` ([modules/model/moe.py](../modules/model/moe.py)) routes each token to a
mixture of heterogeneous experts, and repeats this `n_loops` times over a shared pool. A recurrent
(LoopLM-style) refinement of the representation rather than a single MoE pass

## Expert pool

The router indexes a single flat expert list, ordered:

```
[ self-attn × A | cross-attn × A | IR × I | identity | MLP × M ]
                                            ▲
                                     identity_expert_index
```

With the default config: `A=1`, `I=1`, `M=36` -> **40 experts** (`num_attn_experts` counts self *and*
cross, so it contributes `2A`). Types:

- **Self-attention** ([experts.py](../modules/model/experts.py)) - GQA over the sequence, its own
  head count (16 heads / 4 KV heads) and RoPE cache
- **Cross-attention** - same, but keys/values come from the `other` stream (the projected MoE
  per-layer embedding), letting tokens attend to a side channel
- **Information-retrieval (IR)** ([information_retrieval.py](../modules/model/information_retrieval.py)) -
  down-projects the token, does a cosine-similarity lookup over a learned key/value table
  (`num_ir_entries` x `ir_dim`), up-projects, and feeds the result as cross-attention values. A
  differentiable key-value memory realized with cross attention
- **Identity** - `nn.Identity` passing through it leaves the token unchanged. See Identity skew
- **MLP** - SwiGLU FFNs, run sparsely as one grouped GEMM (see Sparse MLP dispatch).

Attention/IR/identity experts run over the **full sequence regardless of routing**, so each is
computed **once per loop** and cached across the `top_k` slots (recomputing per slot would waste
compute). Only the MLP experts are dispatched sparsely (as thats more "easily" (not really) implementable)

## Routing

`route()` per loop:

1. Router ([router.py](../modules/model/router.py)) = `RMSNorm -> Linear` produces logits. During
   training it adds **annealed exploration noise** `noise_factor * softplus(noise_proj(x)) * ε`.
2. **Load-balancing aux loss** is computed on the pre-skew softmax: `num_experts * Σ f_i * P_i`
   (hard token fraction `f_i` * mean soft prob `P_i`), minimized at a uniform distribution. This
   prevents routing collapse. Returned and weighted by `aux_loss_weight` in the trainer
3. **Identity skew** (below) optionally biases the identity logit
4. Softmax -> `top_k` selection -> renormalize the selected weights to sum to 1.

Experts other than MLP are applied by masked accumulation; MLP slots are remapped to expert-local
indices and dispatched to the sparse layer. All expert outputs are summed, then `RMSNorm` +
dropout. The mean aux loss over loops is returned.

## Identity skew

The identity expert doubles as a **learned early exit / skip**. Landing on it could mean "this
representation is already good enough - dont modify it again"

`identity_skew` (config `model.identity_skew`, default 0.8) adds an increasing bias to the identity
logit as the loop deepens:

```
id_skew = (1 + exp(-|identity_scalar| / identity_skew)) ** (on_loop / n_loops)
logits[..., identity] += id_skew - 1.0
```

so early loops are unbiased and later loops are nudged toward stopping. `identity_skew <= 0`
disables it. Higher values shorten the internal "reasoning path" and may lower output quality,
`identity_scalar` is a learned parameter. Note `on_loop = 0` always disables the bias.

## Sparse MLP dispatch - `ParallelSparseMoELayer`

A naive MoE gathers a dense `[num_experts, tokens, ...]` tensor and runs every expert over every
token, wasting `num_experts / top_k` in matmul FLOPs. Instead:

1. Flatten `(token, slot)` assignments, **sort by expert id** (stable, for deterministic checkpoint
   recompute).
2. `bincount` gives per-expert group sizes (the only host sync)
3. One variable-sized grouped GEMM per expert via TE `GroupedLinear` (fused gate+up, then down).
4. Scale each output by its routing weight (null/identity slots carry weight 0), `index_add_` back.

MLP experts run in **BF16 even under low-precision autocast**: NVFP4 requires each groups row count
divisible by 16, which cant be guaranteed for dynamic per-expert group sizes without padding every
group, so sparsity and NVFP4 are mutually exclusive here.

## Expert-selection tracking

`_ExpertTracking` maintains per-token EMA statistics, like selection fraction, mean routed weight, mean
post-skew probability and is plotted every `sliding_window_size` (256) steps during training. A
recompute guard (`begin_forward` / `_expected_updates`) prevents double counting when
`route()` reruns on the gradient-checkpointing backward pass, under checkpointing the guard covers
the sub-checkpointed case, though stats can still be double-counted in some configurations

# Configuration

All hyperparameters live in [config.yaml](../config.yaml) and are surfaced as `ModelConfig` /
`TrainingConfig` in [config.py](../config.py).

## `model`

| Key | Default | Meaning |
|-----|---------|---------|
| `vocab_size` | 129280 | Tokenizer vocabulary size |
| `max_seq_length` | 4096 | Max context / RoPE cache length |
| `hidden_size` | 512 | Model dimension |
| `intermediate_size` | 2048 | FFN inner dimension (decoder + MLP experts) |
| `num_layers` | 5 | Dense decoder layers |
| `num_attention_heads` | 8 | Decoder attention heads (KV heads = `//4`, GQA) |
| `head_dim` | 64 | Per-head dimension |
| `dropout` | 0.0 | Dropout (pretraining uses 0) |
| `per_layer_embeddings_size` | 32 | PLE vector size per layer. `0` disables PLE |
| `num_mlp_experts` | 36 | Sparse MLP experts - **all expert counts summed must be divisible by 8** for FP8 |
| `num_attn_experts` | 1 | Attention experts, counts self and cross (thus contributes `2x`) |
| `num_ir_experts` | 1 | Information-retrieval (IR) experts |
| `num_ir_entries` | 16384 | Entries in each IR key/value memory |
| `ir_dim` | 128 | IR latent dimension |
| `top_k` | 2 | Experts selected per token per loop |
| `n_loops` | 4 | MoE routing iterations |
| `identity_skew` | 0.8 | Bias toward the identity expert in later loops, higher = exit sooner, `<=0`: disable |
| `mtp_num_extra_tokens` | 2 | Extra future tokens predicted (`0` disables MTP) |
| `lm_head_factor` | 4 | Factorization factor of the LM head (higher = cheaper, lower rank) |

`identity_skew` is passed at forward time via `ModelConfig.Forward`, not baked into the module.

## `training`

| Key | Default | Meaning |
|-----|---------|---------|
| `batch_size` | 3 | Sequences per batch |
| `seq_length` | 4096 | Training sequence length |
| `lr` | 4e-4 | Peak learning rate (cosine floor = `0.1x`) |
| `weight_decay` | 0.02 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient-norm clip (applied on real update steps only) |
| `num_epochs` | 10 | Passes over the data |
| `lambda_mtp` | 0.1 | Weight on each auxiliary MTP loss |
| `aux_loss_weight` | 0.01 | Weight on the MoE load-balancing loss |
| `target_tokens` | 5e9 | Target token budget, drives total steps + cosine length |
| `warmup_steps` | 1000 | Linear LR warmup before cosine decay |
| `noise_anneal_tokens` | 1e9 | Tokens over which router exploration noise decays 1 -> 0 |
| `grad_accumulation_steps` | 8 | Mini batches accumulated per optim step |

`total_steps` is derived as `target_tokens // (batch_size * seq_length * grad_accumulation_steps)`

## Derived / hardcoded values

These are not in `config.yaml` but affect behavior:

- **Decoder KV heads** = `num_attention_heads // 4` (GQA)
- **Expert attention heads** = 16 heads / 4 KV heads, fixed in `LoopMixtureOfExperts`
- **`rope_theta`** = 100000 for both the decoder and the expert RoPE caches
- **Data path**: `data_config.json` (see [training.md](training.md)) and the tokenizer path are set
  in `pretrain()` not in `config.yaml`
- **Low precision** is toggled by `USE_LOW_PRECISION` / `chosen_recipe` in
  [scripts/pretrain.py](../scripts/pretrain.py), not the config (though currently disabled anyway)

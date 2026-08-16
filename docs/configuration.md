# Configuration

All hyperparameters live in [config.yaml](../config.yaml) and are surfaced as `ModelConfig` /
`TrainingConfig` in [config.py](../config.py).

## `model`

| Key | Default | Meaning |
|-----|---------|---------|
| `vocab_size` | 65536 | Tokenizer vocabulary size (`<= 65536`, asserted -- Step 8's `train.bin` is uint16) |
| `max_seq_length` | 4096 | Max context / RoPE cache length |
| `hidden_size` | 768 | Model dimension |
| `intermediate_size` | 2304 | Dense decoder FFN inner dimension only |
| `moe_intermediate_size` | 2304 | Routed + shared MoE expert FFN size; defaults to `intermediate_size` if omitted |
| `num_layers` | 8 | Dense decoder layers |
| `num_attention_heads` | 12 | Decoder attention heads (KV heads = `//4`, GQA) |
| `head_dim` | 64 | Per-head dimension |
| `dropout` | 0.0 | Dropout (pretraining uses 0) |
| `per_layer_embeddings_size` | 32 | PLE vector size per layer. `0` disables PLE |
| `num_mlp_experts` | 32 | Sparse MLP experts |
| `num_attn_experts` | 1 | Attention experts, counts self and cross (thus contributes `2x`) |
| `num_ir_experts` | 1 | Information-retrieval (IR) experts |
| `num_ir_entries` | 8192 | Entries in each IR key/value memory |
| `ir_dim` | 128 | IR latent dimension |
| `top_k` | 2 | Experts selected per token per loop |
| `n_loops` | 3 | MoE routing iterations |
| `mtp_num_extra_tokens` | 2 | Extra future tokens predicted (`0` disables MTP) |
| `lm_head_factor` | 4 | Factorization factor of the LM head (higher = cheaper, lower rank) |

There is no identity expert / `identity_skew` anymore (removed, PLAN.md Step 3) — a halt head
(`p_halt`, see [moe.md](moe.md)) replaces it as the early-exit signal. `ModelConfig.Forward` is
currently empty.

**Construction-time assertions** (`TinyMoETransformer.__init__`, PLAN.md Step 5): `vocab_size` and
`hidden_size` must each be divisible by `lm_head_factor`; if MTP is enabled, `vocab_size` and
`hidden_size // 2` must each also be divisible by `lm_head_factor * 2` (the MTP head's own
`SmallLMHead`); `vocab_size <= 65536`. These raise immediately on a bad config instead of silently
truncating inside `SmallLMHead`'s chunking. The model also prints total/active param counts and a
forward FLOP/token estimate at construction — see [moe.md](moe.md) for the methodology.

## `training`

| Key | Default | Meaning |
|-----|---------|---------|
| `batch_size` | 8 | Sequences per batch |
| `seq_length` | 4096 | Training sequence length |
| `lr` | 4e-4 | Peak learning rate (cosine floor = `0.1x`) |
| `weight_decay` | 0.02 | AdamW weight decay. Applied only to tensors with `ndim >= 2` — norms/biases/gates are excluded because their zero is a degenerate state, not a regularization preference |
| `grad_clip` | 1.0 | Gradient-norm clip (applied on real update steps only) |
| `num_epochs` | 1 | Safety net on the outer loop only. The real stop condition is the phase's token target |
| `lambda_mtp` | 0.1 | Weight on each auxiliary MTP loss |
| `aux_loss_weight` | 0.01 | Weight on the MoE load-balancing loss |
| `target_tokens` | 29.9e9 | **Combined** phase1+phase2 budget. Drives `total_steps` and the cosine length |
| `warmup_steps` | 1000 | Linear LR warmup before cosine decay |
| `noise_anneal_tokens` | 1e9 | Tokens over which router exploration noise decays 1 -> 0 |
| `lambda_ponder` | 0.15 | Target weight on the ponder loss (`(1 - p_halt)` over real tokens) |
| `ponder_warmup_tokens` | 1e9 | Tokens before `lambda_ponder` starts ramping up from 0 |
| `ponder_ramp_tokens` | 1e9 | Tokens over which `lambda_ponder` ramps from 0 to its target after warmup |
| `loop_ce_weights` | `[0.2, 0.3, 1.0]`, required | Per-loop CE weight, ascending. Length must equal `model.n_loops` (asserted at config-load time) |
| `loop_ce_subsample` | 0.25 | Fraction of token positions supervised on the **non-final** loops. The final loop is always supervised in full. `1.0` disables it |
| `loop_count_sampling` | 0.3 | Probability a step runs a random reduced depth in `1..n_loops-1`. `loop_ce_weights` is truncated and rescaled so the deepest loop run always carries weight 1.0. Log steps are pinned to full depth. `0.0` disables it |
| `lambda_conf` | 0.05 | Weight on the correctness head's BCE loss (`correct_proj`, final loop only). Not warmup-ramped |
| `grad_accumulation_steps` | 16 | Mini batches accumulated per optim step |
| `seed` | 42 | Seeds the loop-depth RNG (kept separate from the model's init/dropout/router-noise stream) |
| `data_dir` | `data/prepared` | Directory holding `{phase}.bin` / `{phase}.idx` from `scripts/prepare_data.py` |
| `phase` | `phase1` | Which corpus to train on. Overridden by `pretrain.py --phase` |

`total_steps` is derived as `target_tokens // (batch_size * seq_length * grad_accumulation_steps)`
— from the **combined** target, so phase 2 continues the cosine rather than restarting it.

### Unattended-run keys

| Key | Default | Meaning |
|-----|---------|---------|
| `checkpoint_every_tokens` | 4e8 | Checkpoint cadence in **tokens**, not steps — invariant to batch size and grad accumulation. ~30 min at 200K tok/s |
| `keep_local_checkpoints` | 2 | Rolling checkpoints kept on disk. A checkpoint is deleted only once it is **both** outside this window **and** confirmed uploaded |
| `phase1_fraction` | 0.85 | Phase 1 stops at this fraction of `target_tokens`; phase 2 runs to the full figure |
| `hf_upload_repo` | `ikeafisch4/temp-train` | Upload destination. `""` disables uploads; **deleting the key entirely** falls back to `utils.HF_UPLOAD_REPO` |

The `""`-vs-absent distinction is deliberate: an earlier version read
`hf_upload_repo or HF_UPLOAD_REPO`, so setting it to `""` uploaded anyway. See
`TrainingConfig.upload_repo`.

## Derived / hardcoded values

These are not in `config.yaml` but affect behavior:

- **Decoder KV heads** = `num_attention_heads // 4` (GQA)
- **Expert attention heads** = 16 heads / 4 KV heads, fixed in `LoopMixtureOfExperts`
- **`rope_theta`** = 100000 for both the decoder and the expert RoPE caches
- **Tokenizer path** = `utils.TOKENIZER_DIR`, overridable with `$TINY_LLM_TOKENIZER`. One constant
  shared by every entry point, not a per-script hardcode
- **`NUM_DATA_WORKERS`** = 4 and **`LOG_INTERVAL`** = 20, in [pretrain.py](../scripts/pretrain.py)
- **Low precision** is toggled by the `USE_FP8` environment variable (`USE_FP8=1` selects TE's
  `DelayedScaling` / `Format.HYBRID`), not the config

# Configuration

All hyperparameters live in [config.yaml](../config.yaml) and are surfaced by
[config.py](../config.py) as `ModelConfig`, `TrainingConfig`, and the four finetune profiles
`SFTConfig`, `RepairConfig`, `IRConfig`, `EvidenceConfig`. The yaml carries the reasoning next to
every non-obvious number; this file is the map.

## `model`

`ModelConfig.Params` is splatted straight into `TinyMoETransformer(**...)`.

| Key | Default | Meaning |
|-----|---------|---------|
| `vocab_size` | 65536 | Tokenizer vocabulary size (`<= 65536`, asserted — the corpus is uint16) |
| `max_seq_length` | 4096 | Max context / RoPE cache length |
| `hidden_size` | 768 | Model dimension |
| `intermediate_size` | 2304 | Dense decoder FFN inner dimension only |
| `moe_intermediate_size` | 2304 | Routed + shared MoE expert FFN size; defaults to `intermediate_size` if omitted. The only knob that moves total params without moving active (dense-decoder) params |
| `num_layers` | 8 | Dense decoder layers |
| `num_attention_heads` | 12 | Decoder attention heads (KV heads = `//4`, GQA) |
| `head_dim` | 64 | Per-head dimension |
| `dropout` | 0.0 | Dropout (pretraining uses 0; the finetune profiles override it) |
| `per_layer_embeddings_size` | 32 | PLE vector size per layer. `0` disables PLE |
| `num_mlp_experts` | 32 | Sparse MLP experts |
| `num_attn_experts` | 1 | Attention experts, counts self and cross (thus contributes `2x`) |
| `num_ir_experts` | 1 | Information-retrieval (IR) experts |
| `num_ir_entries` | 65536 | Entries in each IR key/value table. Must be divisible by `ir_num_clusters` |
| `ir_dim` | 384 | IR latent dimension — bge-small's native width, so the evidence adapters are rotations, not compressions |
| `ir_num_clusters` | 256 | Centroids for the two stage read. `0` = the exact full-table softmax every pre-reshape checkpoint was trained under. Capacity (`entries / clusters`) must stay `<= ir_dim` for the orthonormal value init |
| `ir_probe_clusters` | 8 | Clusters opened for exact scoring per token (4 measured recall@32 straddling the 0.9 bar) |
| `ir_read_top_k` | 32 | Entries the read softmax spans |
| `ir_direct_read` | true | IR output stage: each token's own read through a zero-init `direct_gate` (true), or the original inner attention that prefix-averages reads (false, kept for the A/B). **Inferred from the checkpoint at load time**, not from here |
| `evidence_encoder` | true | How retrieved evidence is embedded: the full dense decoder (`true`), the first N layers (an int), or the raw per-token MoE embedding (`false`, comparison only). No parameters, so it applies to any checkpoint |
| `top_k` | 2 | Experts selected per token per loop |
| `n_loops` | 3 | MoE routing iterations |
| `mtp_num_extra_tokens` | 2 | Extra future tokens predicted (`0` disables MTP) |
| `lm_head_factor` | 4 | Factorization factor of the LM head (higher = cheaper, lower rank) |

**Shape-bearing choices come from the checkpoint, not the yaml.** `utils.model_params_for_state_dict`
reads `num_ir_entries` / `ir_dim` from `z_keys`'s shape, forces `ir_num_clusters=0` when the state
dict has no centroids, and sets `loop_inject`, `evidence_port` and `ir_direct_read` from whether the
corresponding tensors exist. Every eval and finetune seed goes through it, so a pre-reshape
baseline keeps scoring after the yaml moves on.

There is no identity expert, no halt head and no config key for the depth policy — `converge_tol`
/ `min_loops` are inference-time arguments to `TinyMoETransformer.forward` (see [moe.md](moe.md)).
`ModelConfig.Forward` is empty.

**Construction-time assertions** (`TinyMoETransformer.__init__`): `vocab_size` and `hidden_size`
must each be divisible by `lm_head_factor`; if MTP is enabled, `vocab_size` and `hidden_size // 2`
must each also be divisible by `lm_head_factor * 2`; `vocab_size <= 65536`. The model prints
total/active param counts and the forward FLOP/token estimate at construction, split into body,
heads and attention (and a separate per-evidence-token figure for the encoder) because the three
scale differently.

## `training`

| Key | Default | Meaning |
|-----|---------|---------|
| `batch_size` | 8 | Sequences per micro batch |
| `seq_length` | 4096 | Training sequence length |
| `lr` | 4e-4 | Peak learning rate (cosine floor = `0.1x`) |
| `weight_decay` | 0.02 | AdamW weight decay. Applied only to tensors with `ndim >= 2` — norms/biases/gates are excluded because their zero is a degenerate state, not a regularization preference |
| `grad_clip` | 1.0 | Gradient-norm clip (applied on real update steps only) |
| `num_epochs` | 1 | Safety net on the outer loop only. The real stop condition is the phase's token target |
| `lambda_mtp` | 0.1 | Weight on each auxiliary MTP loss |
| `aux_loss_weight` | 0.01 | Weight on the MoE load-balancing loss |
| `groundedness_weight` | 0.1 | Weight on the groundedness readout's BCE. Needs a checkpoint carrying the head (`migrate_groundedness_head.py`) **and** a corpus with `.evgold`/`.ans`; 0 by construction otherwise. `0.0` disables it |
| `evidence_selection_weight` | 0.1 | Weight on the evidence selector's supervised ranking loss. Only a batch carrying evidence **and** the corpus's per-chunk gold flag has this term at all, so it is 0 by construction outside `--evidence`. `0.0` disables it |
| `target_tokens` | 16e9 | **Combined** phase1+phase2 budget. Drives `total_steps` and the cosine length |
| `warmup_steps` | 1000 | Linear LR warmup before cosine decay |
| `noise_anneal_tokens` | 1e9 | Tokens over which router exploration noise decays 1 → 0 |
| `loop_ce_weights` | `[0.2, 0.3, 1.0]`, required | Per-loop CE weight, ascending. Length must equal `model.n_loops` (asserted at config-load time) |
| `loop_ce_subsample` | 0.25 | Fraction of token positions supervised on the **non-final** loops. `1.0` disables it |
| `loop_count_sampling` | 0.3 | Probability a step runs a random reduced depth in `1..n_loops-1`. Log steps are pinned to full depth. `0.0` disables it |
| `grad_accumulation_steps` | 16 | Micro batches accumulated per optimizer step |
| `seed` | 42 | Seeds the loop-depth RNG |
| `data_dir` | `data/prepared` | Directory holding `{phase}.bin` / `{phase}.idx` from `scripts/prepare_data.py` |
| `phase` | `phase1` | Which corpus to train on. Overridden by `pretrain.py --phase` |

`total_steps` is derived as `target_tokens // (batch_size * seq_length * grad_accumulation_steps)`
from the **combined** target, so phase 2 continues the cosine rather than restarting it. **Every loss
weight lives here and only here**: the finetune profiles reuse `pretrain.train_step`, which reads
them from `TrainingConfig`.

### Unattended-run keys

| Key | Default | Meaning |
|-----|---------|---------|
| `checkpoint_every_tokens` | 4e8 | Checkpoint cadence in **tokens**, not steps — invariant to batch size and grad accumulation |
| `keep_local_checkpoints` | 2 | Rolling checkpoints kept on disk. A checkpoint is deleted only once it is **both** outside this window **and** confirmed uploaded |
| `phase1_fraction` | 0.85 | Phase 1 stops at this fraction of `target_tokens`; phase 2 runs to the full figure |
| `hf_upload_repo` | `ikeafisch4/temp-train` | Upload destination. `""` disables uploads; **deleting the key entirely** falls back to `utils.HF_UPLOAD_REPO` |

The `""`-vs-absent distinction is deliberate: an earlier version read
`hf_upload_repo or HF_UPLOAD_REPO`, so setting it to `""` uploaded anyway. See
`TrainingConfig.upload_repo`.

## The finetune profiles: `sft`, `repair`, `ir`, `evidence`

One trainer (`scripts/sft.py`), four `config.yaml` blocks, four classes. `RepairConfig`, `IRConfig`
and `EvidenceConfig` **subclass `SFTConfig`**: only the keys a block names are read from it, and
everything else (batch size, sequence length, dropout, warmup, `model_params()`) inherits by
ordinary attribute lookup. That inheritance is the invariant — each later profile is meant to be
the SFT run with different data and a few different numbers, and a second copy of the shared
numbers is a second thing that can drift. All four default `hf_upload_repo` to `""` (local runs).

| Key | `sft` | `repair` | `ir` | `evidence` |
|-----|-------|----------|------|------------|
| splits | `sft_train`/`sft_val` | `repair_train`/`repair_val` | `ir_train`/`ir_val` | `evidence_train`/`evidence_dev`, plus `fixed_split: evidence_fixed`. Both held-out splits come from `prepare_evidence_data.py --heldout` (SQuAD v2 dev, HotpotQA dev). `evidence_val` is not held out for QA (every QA question in it is also in train) |
| `lr` | 3e-5 | 1e-5 | 1e-5 | 1e-5 |
| `fresh_lr` | — | — | 3e-4 (the rebuilt `ir_module` subtree + `down_proj`/`up_proj`) | 3e-4 (the port's own zero-init tensors only; the table trains at `lr`) |
| `num_epochs` | 2 | 1 | 1 | 1 |
| `batch_size` × `grad_accumulation_steps` | 8 × 4 (spills into shared memory on the 5090 — use 4 × 8) | 4 × 4 | 4 × 4 | **2 × 8** — same tokens per optimizer step, half the evidence axis per micro step. Peak memory here is ~3.5 GiB + 0.61 MiB per evidence token, so the evidence axis sets it and 4 rows overflow the card |
| `dropout` | 0.05 | inherited | inherited | inherited |
| `conversation_loss_weighting` | false | **true** | false | **true**, floored by `loss_weight_floor_tokens: 64` |
| `checkpoint_every_tokens` / `eval_every_tokens` | 100M / 25M | 10M / 5M | 50M / 10M | 25M / 2.5M, plus an eval before step 1 on a fresh start (every profile) |
| fixed-target pass | - | - | - | `fixed_eval_max_batches` 100 over `evidence_fixed`, token-level (unweighted) answer CE per condition, gain = CE(none) - CE(cond), per-loop selector readings. `kill_tokens` 10M / `kill_min_gain` 0.1: the first eval past 10M with a gold gain under 0.1 nats saves and exits 10 |
| `freeze_evidence_gate` | - | - | - | true: `evidence_gate_scale` gets `requires_grad=False` and is zeroed if loaded nonzero (the gate reads a document-mean, so it sees future tokens) |
| anneal | — | — | `temperature_start` 1.0 → `temperature_end` 0.05 over `temperature_anneal_fraction` 0.7, geometric | — |
| `cluster_refresh_tokens` / `dead_quantile` | — | — | 20M / 0.02 | 20M / 0.02 (the table still trains, so its centroids still track it) |
| `max_evidence_tokens` | — | — | — | 14336, one row's evidence cap. It must be read against the corpus's evidence-to-prompt ratio, not set once: a row needs `ratio × seq_length` of evidence to fill its token budget, and under that the evidence budget closes the row early and the unused prompt slots become padding the body still pays for. The old 12288 was 3× `seq_length` against a 2.93 ratio; the four-pass corpus reads 3.75, where 12288 closed 206 of the first 256 rows for 55% fill. 14336 is 3.5× — the ratio's 15.4k minus a safety margin, since the cap is also the dominant term in peak memory — and measures 64% fill, short of what the ratio predicts because packing is greedy and a row drawing QA conversations closes on evidence early |

Other keys: `warmup_fraction` 0.03, `lr_min_factor` 0.05 (cosine floor), `seed` 1234 (the per-epoch
document permutation, checkpointed; changing it mid-run is a hard error), `eval_max_batches` 40,
`keep_local_checkpoints` 3. `IRConfig.temperature_scale(progress)` interpolates geometrically:
temperature is a divisor, so equal ratios are equal steps in sharpness.

## Derived / hardcoded values

Not in `config.yaml` but affecting behavior:

- **Decoder KV heads** = `num_attention_heads // 4` (GQA)
- **Expert attention heads** = 16 heads / 4 KV heads, and **`rope_theta`** = 100000, fixed in
  `LoopMixtureOfExperts`
- **`ROUTER_NOISE_SCALE`** = 0.3 ([router.py](../modules/model/router.py)); **`CE_CHUNK_SIZE`** =
  8192 ([mtp.py](../modules/model/mtp.py)); **`MAX_CHUNKS_PER_SEGMENT`** = 256
  ([evidence.py](../modules/model/evidence.py))
- **Tokenizer path** = `utils.TOKENIZER_DIR`, overridable with `$TINY_LLM_TOKENIZER`
- **`NUM_DATA_WORKERS`** = 4; **`LOG_INTERVAL`** = 20 in `pretrain.py`, 10 in `sft.py`
- **Low precision** is toggled by the `USE_FP8` environment variable, not the config. **Every local
  finetune runs in BF16 — do not set it**

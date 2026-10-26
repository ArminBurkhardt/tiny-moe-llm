# Configuration

All hyperparameters live in a yaml file surfaced by [config.py](../config.py) as `ModelConfig`,
`TrainingConfig`, and the four finetune profiles `SFTConfig`, `RepairConfig`, `IRConfig`,
`EvidenceConfig`. The file is [config.yaml](../config.yaml) unless the environment variable
`TINY_LLM_CONFIG` names another one: `config_micro.yaml` (the from-scratch micro arms) and
`config_pilot.yaml` (the 99.1M pilot shape) sit beside it in the repo root, and every script reads
whichever file the variable names (a relative path resolves against the working directory, so run
from the repo root). The checkpoint still decides shape and mode on load (below). The yaml carries
the reasoning next to every non-obvious number; this file is the map.

**Which value the tables show.** The `config.py` column is the fallback `config.py` uses when the
key is absent from the yaml ("required" when it has none and a missing key is a `KeyError`); the
`config.yaml` column is the value in the repo's default `config.yaml`. Where the two agree the
`config.yaml` cell says "same".

## `model`

`ModelConfig.Params` is splatted straight into `TinyMoETransformer(**...)`.

| Key | `config.py` | `config.yaml` | Meaning |
|-----|-------------|---------------|---------|
| `vocab_size` | required | 65536 | Tokenizer vocabulary size (`<= 65536`, asserted: the corpus is uint16) |
| `max_seq_length` | required | 4096 | Max context / RoPE cache length (passed as `max_seq_len`) |
| `hidden_size` | required | 768 | Model dimension |
| `intermediate_size` | required | 2304 | Dense decoder FFN inner dimension only |
| `moe_intermediate_size` | `intermediate_size` | 2304 | Routed + shared MoE expert FFN size. The only knob that moves total params without moving active (dense decoder) params |
| `num_layers` | required | 8 | Dense decoder layers |
| `num_attention_heads` | required | 12 | Decoder attention heads (KV heads = `//4`, GQA) |
| `head_dim` | required | 64 | Per-head dimension |
| `dropout` | required | 0.0 | Dropout (pretraining uses 0; the finetune profiles override it) |
| `per_layer_embeddings_size` | required | 32 | PLE vector size per layer (passed as `ple_embeddings_size`). `0` turns PLE off in the dense decoder only: the MoE path checks for `None`, not 0, so it still builds a zero-width `moe_embeddings` table and its projection for the cross-attention expert's `other` stream |
| `num_mlp_experts` | required | 32 | Sparse MLP experts |
| `num_attn_experts` | required | 1 | Attention experts, counts self and cross (thus contributes `2x`) |
| `num_ir_experts` | required | 1 | Information-retrieval (IR) experts |
| `num_ir_entries` | required | 65536 | Entries in each IR key/value table. Must be divisible by `ir_num_clusters` |
| `ir_dim` | required | 384 | IR latent dimension: bge-small's native width, so the evidence adapters are rotations, not compressions |
| `ir_num_clusters` | 0 | 256 | Centroids for the two stage read. `0` is the exact full-table softmax every pre-reshape checkpoint was trained under. Capacity (`entries / clusters`) must stay `<= ir_dim` for the orthonormal value init |
| `ir_probe_clusters` | 4 | 8 | Clusters opened for exact scoring per token (4 measured recall@32 straddling the 0.9 bar) |
| `ir_read_top_k` | 32 | same | Entries the read softmax spans |
| `ir_direct_read` | true | same | IR output stage: each token's own read through a zero-init `direct_gate` (true), or the original inner attention that prefix-averages reads (false, kept for the A/B; since 2026-10-26 the inner attention is built only when this is off, and an old checkpoint's IR inner attention keys are dropped on load by `utils.drop_ir_inner_attn`). **Inferred from the checkpoint at load time** (from whether a `direct_gate` tensor exists), not from here |
| `evidence_encoder` | true | same | How retrieved evidence is embedded: the full dense decoder (`true`), the first N layers (an int), or the raw per-token MoE embedding (`false`, comparison only). No parameters, so it applies to any checkpoint, and it always comes from the yaml |
| `top_k` | required | 2 | Experts selected per token per loop |
| `n_loops` | required | 3 | MoE routing iterations |
| `mtp_num_extra_tokens` | required | 2 | Extra future tokens predicted (`0` disables MTP) |
| `lm_head_factor` | required | 4 | Factorization factor of the LM head (higher = cheaper, lower rank) |

**Shape-bearing choices come from the checkpoint, not the yaml.** `utils.model_params_for_state_dict`
reads `num_ir_entries` / `ir_dim` from `z_keys`'s shape, forces `ir_num_clusters=0` when the state
dict has no centroids, and sets `loop_inject`, `evidence_port`, `ir_direct_read`, the groundedness
head, the reader's rotary (off iff the marker buffer `moe.evidence_reader_rotary_off` exists) and
the reader kind (`evidence_reader="kv"` iff the marker buffer `moe.evidence_reader_kv` exists,
which also implies the port) from whether the corresponding tensors exist. Every eval and finetune
seed goes through it, so a pre-reshape baseline keeps scoring after the yaml moves on.

There is no identity expert, no halt head and no config key for the depth policy: `converge_tol`
/ `min_loops` are inference-time arguments to `TinyMoETransformer.forward` (see [moe.md](moe.md)).
`ModelConfig.Forward` is empty.

**Construction-time assertions** (`TinyMoETransformer.__init__`): `vocab_size` and `hidden_size`
must each be divisible by `lm_head_factor`; if MTP is enabled, `vocab_size` and `hidden_size // 2`
must each also be divisible by `lm_head_factor * 2`; `vocab_size <= 65536`. The model prints
total/active param counts and the forward FLOP/token estimate at construction, split into body,
heads and attention (and a separate per-evidence-token figure for the encoder) because the three
scale differently.

## `training`

| Key | `config.py` | `config.yaml` | Meaning |
|-----|-------------|---------------|---------|
| `batch_size` | required | 8 | Sequences per micro batch |
| `seq_length` | required | 4096 | Training sequence length |
| `lr` | required | 4e-4 | Peak learning rate (cosine floor = `0.1x`). `config.py` also reads it into `TrainingConfig.learning_rate`, an unread duplicate of `lr` that nothing in the repo uses |
| `weight_decay` | required | 0.02 | AdamW weight decay. Applied only to tensors with `ndim >= 2`: norms, biases and gates are excluded because their zero is a degenerate state, not a regularization preference |
| `grad_clip` | required | 1.0 | Gradient-norm clip (applied on real update steps only) |
| `optimizer` | `adamw` | same | `adamw` or `muon` (`modules/model/optim.py`). `muon` steps the hidden 2D matrices (attention, MLPs, experts, the selector's projections, the store adapters, the PLE projections) with Muon: Nesterov momentum 0.95, five Newton-Schulz steps in bf16, the update scaled by `0.2 sqrt(max(m, n))` to AdamW's RMS so `lr` and `weight_decay` carry over; tables, heads, the router, scalars, vectors and the zero-init matrices stay on AdamW. Read by both trainers. A checkpoint resumes only under the optimizer it was saved with (the group layout differs) |
| `num_epochs` | required | 1 | Safety net on the outer loop only. The real stop condition is the phase's token target |
| `lambda_mtp` | required | 0.1 | Weight on each auxiliary MTP loss |
| `aux_loss_weight` | required | 0.01 | Weight on the MoE load-balancing loss. Added to the objective only under `moe_balancing: aux` |
| `groundedness_weight` | 0.0 | 0.1 | Weight on the groundedness readout's BCE. Needs a checkpoint carrying the head (`migrate_groundedness_head.py`) **and** a corpus with `.evgold`/`.ans`; 0 by construction otherwise. `0.0` disables it |
| `evidence_selection_weight` | 0.0 | 0.1 | Weight on the evidence selector's supervised ranking loss. Only a batch carrying evidence **and** the corpus's per-chunk gold flag has this term at all, so it is 0 by construction outside `--evidence`. `0.0` disables it |
| `evidence_selection_weighting` | `token` | same (`segment` decided 2026-10-25 for the real run) | How the selection loss averages: `token` (default) is the mean over supervised predicting positions; `segment` averages each query segment first (from `cu_seqlens`), then the segments. Must be `token` or `segment`. The recorded arms all ran under `token`; the real run's default is `segment` (decided 2026-10-25, RUN2.md row 2) |
| `evidence_contrastive_weight` | 0.0 | same | Weight on the retriever's in-batch InfoNCE (`evidence_contrastive_loss`): the selector's unit query at each answer-start position against every chunk in the batch, the other documents' chunks as negatives. Needs evidence with the per-chunk gold flag and gradient checkpointing off. `0.0` (every run before 2026-10-17) disables it |
| `evidence_contrastive_temperature` | 0.07 | same | The fixed softmax temperature on the cosine for that term, apart from the selector's learned read temperature |
| `evidence_contrastive_loops` | `[0]` (since 2026-10-24; null before) | same | Zero-based loop indices the term averages over; `null` is still accepted and means every loop that read the store (the objective before 2026-10-21; the recorded arms `w0e3` and `w25e3` ran with it). `[0]` trains only the first loop's query, the one `eval_store.py recall --loop 1` reads, before any evidence has been read through the reader. Each index must be below `n_loops`; a listed loop a sampled depth skipped is left out of that step, and loop 0 is the only pass every depth runs, so a list without it drops the term on some steps |
| `evidence_contrastive_trunk_grad` | 1.0 | same | Share of the term's gradient that reaches the trunk through the query's input state, in [0, 1]. Below 1 the term rebuilds its query at the positions from the stashed IR expert input under a scaled backward (`InformationRetrievalExpert.contrastive_query`; the trainers turn the input stash on through `set_contrastive_query_stash` only when the weight is above 0 and this is below 1); `0.0` trains the query head alone (the expert's `norm` and `down_proj`, `loop_query_bias`, `key_adapter`). The IR read's own gradient is untouched. `1.0` is bit-identical to the objective before 2026-10-21 |
| `target_tokens` | required | 16e9 | **Combined** phase1+phase2 budget. Drives `total_steps` and the cosine length |
| `warmup_steps` | 0 | 1000 | Linear LR warmup before cosine decay |
| `noise_anneal_tokens` | 0 | 1e9 | Tokens over which router exploration noise decays 1 to 0. At 0 `pretrain.py` skips the anneal and the router keeps its full noise factor of 1.0 for the whole run; `sft.py` sets the noise to 0 regardless |
| `loop_ce_weights` | required | `[0.2, 0.3, 1.0]` | Per-loop CE weight, ascending. Length must equal `model.n_loops` (asserted at config-load time) |
| `loop_ce_subsample` | 1.0 | 0.25 | Fraction of token positions supervised on the **non-final** loops. `1.0` disables it |
| `loop_count_sampling` | 0.0 | 0.3 | Probability a step runs a random reduced depth in `1..n_loops-1`. Log steps are pinned to full depth. `0.0` disables it |
| `moe_balancing` | `aux` | same | `aux` adds `aux_loss_weight` times the router's balancing loss to the objective (every run before 2026-10-19). `loss_free` adds no such term and instead keeps a per-pass, per-expert bias `moe.balance_bias` (`[n_loops, num_experts]`, a persistent buffer) that is added to the routing probabilities for the top-k selection only, the routing weights staying the unbiased probabilities; after each optimizer step the MLP slots' bias moves by `moe_balance_bias_rate` times the sign of each expert's load error against the pass's mean MLP load, the signs centred over the MLP slots so the update is zero sum (arXiv 2408.15664). The attention and IR slots keep a zero bias and the split between them and the MLP slots stays the router's. A zero bias is the unbiased router, so every earlier checkpoint loads as itself; a checkpoint carrying a bias routes with it in any mode. Read by both trainers. The balancing loss is logged under both modes, and since 2026-10-20 `load max/mean` and `dense share` per pass are logged in every balancing mode (the hard counts are always on; under `aux` the trainers pass the bias update a rate of zero, so the bias never moves there) |
| `moe_balance_bias_rate` | 0.001 | same | The fixed step on the balance bias per optimizer step under `loss_free` (must be above 0) |
| `grad_accumulation_steps` | 1 | 16 | Micro batches accumulated per optimizer step |
| `seed` | 42 | same | Seeds the loop-depth RNG in `pretrain.py` only; `sft.py` uses the profile's `cfg.seed` (1234) |
| `data_dir` | `data/prepared` | same | Directory holding `{phase}.bin` / `{phase}.idx` from `scripts/prepare_data.py` |
| `phase` | `phase1` | same | Which corpus to train on. Overridden by `pretrain.py --phase` |

`total_steps` is derived as `target_tokens // (batch_size * seq_length * grad_accumulation_steps)`
from the **combined** target, so phase 2 continues the cosine rather than restarting it. **Every loss
weight lives here and only here**: the finetune profiles reuse `pretrain.train_step`, which reads
them from `TrainingConfig`.

### Unattended-run keys

| Key | `config.py` | `config.yaml` | Meaning |
|-----|-------------|---------------|---------|
| `checkpoint_every_tokens` | 4e8 | same | Checkpoint cadence in **tokens**, not steps: invariant to batch size and grad accumulation |
| `keep_local_checkpoints` | 2 | same | Rolling checkpoints kept on disk. A checkpoint is deleted only once it is **both** outside this window **and** confirmed uploaded |
| `phase1_fraction` | 0.85 | 0.85 in `config.yaml`; absent from the micro and pilot yamls, where it falls back to 0.85 | Phase 1 stops at this fraction of `target_tokens`; phase 2 runs to the full figure |
| `hf_upload_repo` | absent: `utils.HF_UPLOAD_REPO` | `ikeafisch4/temp-train` | Upload destination. `""` disables uploads; **deleting the key entirely** falls back to `utils.HF_UPLOAD_REPO` |

The `""`-vs-absent distinction is deliberate: an earlier version read
`hf_upload_repo or HF_UPLOAD_REPO`, so setting it to `""` uploaded anyway. See
`TrainingConfig.upload_repo`.

## The finetune profiles: `sft`, `repair`, `ir`, `evidence`

One trainer (`scripts/sft.py`), four yaml blocks, four classes. `RepairConfig`, `IRConfig`
and `EvidenceConfig` **subclass `SFTConfig`**: only the keys a block names are read from it, and
everything else (sequence length, `eval_max_batches`, `model_params()`) inherits by ordinary
attribute lookup. That inheritance is the invariant: each later profile is meant to be the SFT run
with different data and a few different numbers, and a second copy of the shared numbers is a
second thing that can drift. Note that several subclass keys have their own fallback in
`config.py` rather than inheriting `SFTConfig`'s (the table names them). All four default
`hf_upload_repo` to `""` (local runs).

Each cell reads "yaml value (config.py fallback)" where the two differ, and the bare value where
they agree.

| Key | `sft` | `repair` | `ir` | `evidence` |
|-----|-------|----------|------|------------|
| splits | `sft_train`/`sft_val` | `repair_train`/`repair_val` | `ir_train`/`ir_val` | `evidence_train`/`evidence_dev`, plus `fixed_split: evidence_fixed`. Both held-out splits come from `prepare_evidence_data.py --heldout` (SQuAD v2 dev, HotpotQA dev). `evidence_val` is not held out for QA (every QA question in it is also in train) |
| `lr` | 3e-5 | 1e-5 | 1e-5 | 1e-5 |
| `fresh_lr` | - | - | 3e-4, for `is_rebuilt_ir_param or is_fresh_loop_param`: the whole `ir_module` subtree, the IR expert's `down_proj`/`up_proj`, and every tensor the evidence profile's set names | 3e-4, for `moe.is_fresh_loop_param` alone: `moe.inject`, the reader `shared_evidence.*`, the selector's `key_adapter`/`value_adapter`/`log_memory_scale`, `loop_query_bias`, `evidence_query_bias`, `evidence_loop_scale`, `direct_gate`, `evidence_gate_scale` (frozen and in no group under `freeze_evidence_gate`) and `groundedness_head.*`; the IR table trains at `lr` |
| `num_epochs` | 2 | 1 | 1 | 1 |
| `batch_size` x `grad_accumulation_steps` | 8 x 4 (4 x 8); 8 spills into shared memory on the 5090, so use 4 x 8 | 4 x 4 (inherits `sft`'s resolved values when absent) | 4 x 4 | **2 x 8** (4 x 4): same tokens per optimizer step, half the evidence axis per micro step. Peak memory here is ~3.5 GiB + 0.61 MiB per evidence token, so the evidence axis sets it and 4 rows overflow the card |
| `dropout` | 0.05 | inherited | inherited | inherited |
| `conversation_loss_weighting` | false | **true** | false | **true**, floored by `loss_weight_floor_tokens: 64` |
| `checkpoint_every_tokens` / `eval_every_tokens` | 100M (25M) / 25M | 10M / 5M | 50M / 10M | 10M / 2.5M (the save at the 10M decision), plus an eval before step 1 on a fresh start (every profile) |
| fixed-target pass | - | - | - | `fixed_eval_max_batches` 100 over `evidence_fixed`, token-level (unweighted) answer CE per condition, gain = CE(none) - CE(cond), per-loop selector readings. `kill_tokens` 10M / `kill_min_gain` 0.1: the first eval past 10M with a gold gain or a content gain under 0.1 nats saves and exits 10 (`kill_tokens: 0`, as in `config_micro.yaml` and `config_pilot.yaml`, disables it) |
| `keep_every_tokens` | 0 (off) | 0 | 0 | 0 (every block inherits `SFTConfig`'s value; `config_run2.yaml` sets 10_000_000_000 with `keep_local_checkpoints` 2). Since 2026-10-26 rank 0 deletes rolling saves outside the newest `keep_local_checkpoints` unless the save is the first at or past a multiple of this many tokens or a `*_final.pt` |
| `freeze_evidence_gate` | - | - | - | true (config.py:466, every config; the gate is also skipped in the forward since 2026-10-24): `evidence_gate_scale` gets `requires_grad=False` and is zeroed if loaded nonzero (the gate reads a document-mean, so it sees future tokens) |
| anneal | - | - | `temperature_start` 1.0 to `temperature_end` 0.05 over `temperature_anneal_fraction` 0.7, geometric | - |
| `cluster_refresh_tokens` / `dead_quantile` | - | - | 20M / 0.02 | 20M / 0.02 (the table still trains, so its centroids still track it) |
| `max_evidence_tokens` | - | - | - | 14336 (12288), one row's evidence cap. It must be read against the corpus's evidence-to-prompt ratio, not set once: a row needs `ratio x seq_length` of evidence to fill its token budget, and under that the evidence budget closes the row early and the unused prompt slots become padding the body still pays for. The old 12288 was 3x `seq_length` against a 2.93 ratio; the four-pass corpus reads 3.75, where 12288 closed 206 of the first 256 rows for 55% fill. 14336 is 3.5x (the ratio's 15.4k minus a safety margin, since the cap is also the dominant term in peak memory) and measures 64% fill, short of what the ratio predicts because packing is greedy and a row drawing QA conversations closes on evidence early |

Other keys: `weight_decay` (the `sft` block's fallback 0.01; each profile reads its own key and falls back to `SFTConfig.weight_decay`), `seq_length` and `data_dir` (read only from the `sft:` block, 4096 and `data/prepared`; the other profiles inherit them, so no profile can set them), `warmup_fraction` 0.03, `lr_min_factor` 0.05 (cosine floor), `seed` 1234 (the per-epoch
document permutation, checkpointed; changing it mid-run is a hard error), `eval_max_batches` 40, `keep_every_tokens` 0 (see the table),
`keep_local_checkpoints` 3. Under several processes (2026-10-26) `batch_size` is per GPU and the global batch is `batch_size` x processes x `grad_accumulation_steps`; `config_run2.yaml` is written for 8 GPUs (524,288 tokens per update). `IRConfig.temperature_scale(progress)` interpolates geometrically:
temperature is a divisor, so equal ratios are equal steps in sharpness.

## Derived / hardcoded values

Not in the yaml but affecting behavior:

- **Decoder KV heads** = `num_attention_heads // 4` (GQA)
- **Expert attention heads** = 16 heads / 4 KV heads, and **`rope_theta`** = 100000, fixed in
  `LoopMixtureOfExperts`
- **`ROUTER_NOISE_SCALE`** = 0.3 ([router.py](../modules/model/router.py)); **`CE_CHUNK_SIZE`** =
  8192 ([mtp.py](../modules/model/mtp.py)); **`MAX_CHUNKS_PER_SEGMENT`** = 256
  ([evidence.py](../modules/model/evidence.py))
- **Tokenizer path** = `utils.TOKENIZER_DIR`, overridable with `$TINY_LLM_TOKENIZER`
- **Config file** = `config.yaml`, overridable with `$TINY_LLM_CONFIG`
- **`NUM_DATA_WORKERS`** = 4; **`LOG_INTERVAL`** = 20 in `pretrain.py`, 10 in `sft.py`
- **Low precision** is toggled by the `USE_FP8` environment variable, not the config. **Every local
  finetune runs in BF16; do not set it**

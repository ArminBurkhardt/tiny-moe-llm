# CLAUDE.md

Working notes for this repo. Prose docs live in [docs/](docs/) — this file is the operational
map: what lives where, what the non-obvious invariants are, and how to run things.

## What this is

`tiny-moe-llm`: an experimental ~243M-param LM. A dense Gemma4-style decoder feeds a **single
MoE block applied `n_loops` times** (LoopLM-style recurrence), with a heterogeneous expert pool
(self-attn / cross-attn / information-retrieval / MLP) behind one router plus always-on shared
experts, a per-loop halt head, and multi-token prediction heads. Trained on document-packed
streams with flash-attn varlen attention, optionally in FP8/NVFP4 via NVIDIA Transformer Engine.

Research code, not a library: no packaging, no test framework, no CI. Entry points are the two
scripts under [scripts/](scripts/).

## Layout

```
config.py / config.yaml     all hyperparameters (yaml -> ModelConfig / TrainingConfig)
utils.py                    logger, BASE_DIR, dtype aliases, save/load_checkpoint
data_config.json            data source roots + column names per mode (gitignored)
memory-benchmark.py         standalone peak-VRAM probes (not used by training)
env_init                    WSL/CUDA env + venv activation (gitignored, `source env_init`)
scripts/
  pretrain.py               THE training loop
  inference.py              greedy/top-k sampling CLI
modules/model/
  transformer.py            TinyMoETransformer (top level) + TokenTracker
  gemma4.py                 dense decoder: GQA, RoPE, RMSNorm(te), per-layer embeddings
  moe.py                    LoopMixtureOfExperts, ParallelSparseMoELayer, _ExpertTracking
  router.py                 Router (+ annealed exploration noise), compute_aux_loss
  experts.py                SelfAttention / CrossAttention / InformationRetrievalExpert
  information_retrieval.py  learned key/value table with softmax retrieval
  mtp.py                    MTPHead, chunked LM-head CE, compute_mtp_loss
  attention.py              varlen_attention, cu_seqlens_from_doc_ids, SDPA fallback
  modules.py                SmallLMHead (factored vocab projection)
  embeddings.py             RoPE cache + apply_rotary_pos_emb
  utils.py                  EncoderOutput dataclass
modules/data/dataset.py     streaming IterableDataset with document packing + resume
tests/                      tracked sanity scripts (see "Tests")
ckpts/                      gitignored: pretrained/<tokenizer dirs>, training/<*.pt, *.png>
data/datasets/              gitignored parquet/jsonl shards
```

`modules/util/` is an empty leftover. `modules/*/__init__.py` are empty — imports are always
fully qualified (`from modules.model.moe import ...`).

## Environment

- Dev box is Windows, but **training runs under WSL**: `tests/run_tests.sh` hardcodes
  `cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm` and `source env_init` (CUDA 12.9 paths,
  `venv/`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- **Transformer Engine is a hard import dependency** — `import transformer_engine.pytorch as te`
  at module scope in gemma4/moe/experts/mtp/modules/router/transformer. There is no CPU-only or
  TE-less path; nothing in `modules/model/` imports on a machine without it. flash-attn *is*
  optional ([attention.py:4-10](modules/model/attention.py#L4-L10) falls back to a slow SDPA
  block-mask path).
- `config.py` opens `"config.yaml"` with a **relative** path, so every script must be launched
  from the repo root (`python scripts/pretrain.py`, not `cd scripts && python pretrain.py`).

## Commands

```bash
source env_init                      # WSL: CUDA paths + venv
python scripts/pretrain.py           # resumes from the newest ckpts/training/*.pt if any
python scripts/inference.py          # interactive; -c CKPT -p PROMPT -n 200 --temperature 0.8
bash tests/run_env_check.sh          # torch/flash/TE/tokenizer smoke check
bash tests/run_tests.sh tests/test_attention_equiv.py tests/test_overfit.py
```

Tests are plain scripts (no pytest): each `sys.path.insert`s the repo root, asserts, and prints.
Most require a GPU. `tests/` is tracked (the `.gitignore` line for it is commented out) — edits
land in normal commits like any other source file.

## Config

`config.yaml` -> `config.py` exposes three surfaces:

- `ModelConfig.Params` — kwargs splatted straight into `TinyMoETransformer(**...)`.
- `ModelConfig.Forward` — kwargs splatted into `model(...)`; currently empty (`identity_skew` was
  its only key, deleted in PLAN.md Step 3).
- `TrainingConfig` — class attributes; `total_steps` is *derived* as
  `target_tokens // (batch_size * seq_length * grad_accumulation_steps)`. Also holds the ponder
  loss knobs (`lambda_ponder`, `ponder_warmup_tokens`, `ponder_ramp_tokens`) even though they read
  from `config.yaml`'s `training:` block rather than `model:` — they're consumed directly in
  `scripts/pretrain.py`'s loss calc, not passed into the model.

Constraints worth remembering:
- `moe_intermediate_size` (optional) sizes the routed MLP experts and the always-on shared
  MLP/attn (see "Model invariants" below); defaults to `intermediate_size` if omitted from
  `config.yaml`. `Gemma4TextModel`'s dense decoder always uses plain `intermediate_size`, so this
  is the only knob that moves total params without moving active (dense-decoder) params.
- `mtp_num_extra_tokens` must be <= the dataset's `num_mtp_tokens` separator budget, otherwise
  MTP gets supervised across document boundaries.
- `vocab_size` and `hidden_size` must both be divisible by `lm_head_factor` (SmallLMHead chunks
  both dims).
- Things *not* in the yaml but hardcoded: `NUM_DATA_WORKERS=4`, `LOG_INTERVAL=10`, checkpoint
  every 5000 steps ([pretrain.py](scripts/pretrain.py)), expert head counts `n_heads=16 /
  n_kv_heads=4` and `rope_theta` ([moe.py](modules/model/moe.py)), `CE_CHUNK_SIZE=2048`
  ([mtp.py](modules/model/mtp.py)).

## Model invariants

**Expert index layout** (one router over the whole pool, order matters everywhere):

```
[ SelfAttention x A | CrossAttention x A | IR x I | MLP x M ]
                                            ^ first_mlp_index = 2A + I
```

No identity expert (removed in PLAN.md Step 3c — see "Halt head" below for its replacement).
`LoopMixtureOfExperts._num_attn_experts` is `num_attn_experts * 2` (self + cross). Indices
`>= first_mlp_index` are remapped into `ParallelSparseMoELayer`'s local expert space in
`forward_step`; non-MLP slots become `(index 0, weight 0)` so they contribute nothing.

- **Non-MLP experts run unconditionally**, once per `forward_step`, and are cached across the
  top-k slots — attention has to see the whole sequence regardless of routing. Only the MLP
  experts are genuinely sparse (grouped GEMM over sorted assignments).
- **`shared_mlp` + `shared_attn` seed `forward_step`'s output accumulator unconditionally, every
  loop** (PLAN.md Step 2) — a dense SwiGLU MLP and a `SelfAttention` reused for its RoPE/varlen
  path, neither in the router pool (not in `Router`'s output dim, not in `compute_aux_loss`).
  Sized by `moe_intermediate_size`, not the dense decoder's `intermediate_size`. Static row count
  (`B*S`), so unlike `ParallelSparseMoELayer` they run inside the outer `te.autocast` — don't wrap
  them in `te.autocast(enabled=False)`.
- **`forward_step` returns an updated `hidden_states`, not a replacement** (PLAN.md Step 1):
  `hidden_states = hidden_states + (1 - p_halt) * loop_scale * dropout(post_norm(output))`, giving
  a gradient path across loop boundaries independent of routing. `loop_scale` (`nn.Parameter`,
  init `0.1`, not `0` — see the comment at its definition in [moe.py](modules/model/moe.py)) is a
  LayerScale/ReZero-style per-loop gate, distinct from `layer_scalar` in the dense decoder
  (init-1 whole-layer gain).
- **Halt head** (PLAN.md Step 3a, replaces the deleted identity expert): `self.halt_proj` is a
  `Linear(hidden_size, 1)`, zero-init weight / bias `-2.0` (`p_halt ~ 0.12` at init), applied to
  the *incoming* hidden state each loop before the update above. `p_halt -> 1` means "don't modify
  me further" — a compute-allocation signal, not a correctness score (that's Step 4b's separate
  `correct_proj`, not yet implemented). It's **greedy per-loop, not cumulative ACT**: recomputed
  fresh each loop, so a token can halt at loop 1 and un-halt at loop 2. `LoopMixtureOfExperts.forward`
  stacks per-loop `p_halt` into `[n_loops, B, S]` and returns it alongside `hidden_states` — never
  reduced with `.item()` inside the model.
- **Ponder loss requires its warmup to actually be wired up** (`TrainingConfig.ponder_warmup_tokens`
  / `ponder_ramp_tokens`, applied in `scripts/pretrain.py`'s `train_step`) — this is a correctness
  requirement, not tuning. At `loop_scale ~ 0.1`, CE loss has near-zero gradient wrt `p_halt`, so an
  un-warmed ponder term is briefly the halt head's only (constant-sign) signal; AdamW climbs the
  halt bias regardless of `lambda_ponder`'s magnitude, `p_halt` saturates before `loop_scale` grows,
  and the loop goes silently dead while the dense decoder keeps the loss descending. See
  `tests/test_ponder_deadlock.py` for a reproduction of both the failure mode and the fix.
- Selection is applied with a **mask multiply**, not `mask.sum()`/boolean indexing, deliberately:
  boolean indexing forces a device sync per expert per step.
- `ParallelSparseMoELayer.forward` runs its GEMMs under `te.autocast(enabled=False)` — NVFP4
  needs each group's row count divisible by 16, which dynamic routing can't guarantee. Sparsity
  wins over precision here. `m_splits` via `.tolist()` is a known, accepted host sync.
- `torch.argsort(..., stable=True)` in the same function is required for determinism between the
  checkpoint forward and the recompute pass.
- The aux (load-balancing) loss is computed directly on the router's softmax probabilities — no
  skew/bias step anymore (that was the deleted identity mechanism).
- `_ExpertTracking` guards against activation-checkpoint recompute double counting
  (`begin_forward(expected_updates)`) and only samples every 8th forward. Its stats are per-token
  EMAs in [0, 1], plotted to `ckpts/training/expert_selection_*.png`.

**Gradient checkpointing**: use `from transformer_engine.pytorch import checkpoint`, never
`torch.utils.checkpoint` — the latter breaks FP8/NVFP4 quantized layers. Two levels:
`set_checkpointing(stage_level, sub_level)` for the decoder/MoE stages and the per-loop /
per-MTP substages. Training currently runs with both **off**.

**Training-mode forward returns hidden states, not logits** (`return_hidden=True` plus
`delayed_mtp_loss(True)`): `compute_mtp_loss` applies the LM head inside a chunked, checkpointed
cross-entropy so `[T, vocab]` logits are never fully materialized. If you add a call site,
pass `main_lm_head=` or you'll silently double the activation peak.

**Document packing**: the dataset emits batch-aligned `document_ids [B, S]`; the trainer converts
them to `cu_seqlens` **in-thread** via `cu_seqlens_from_doc_ids`. Never put `cu_seqlens` in the
batch dict — it's ragged (`dim0 = num_segments + 1`) and accelerate's `split_batches` truncates
dim 0 to the batch size, silently corrupting segmentation. `max_seqlen` is passed as `S` (a valid
upper bound) rather than the true max, to avoid an `.item()` sync every step.

**Token counting**: `TokenTracker` accumulates non-pad counts in an on-device scalar and only
drains to host on `sync()`, called at log/checkpoint cadence. Read `.get_count()` / `token_count`
for a sync-free (slightly stale) value; assign `.num_tokens` to restore on resume. Don't add a
`.item()` in the step path.

**Tokenizer quirk**: with the DeepSeek tokenizer `pad_token_id == eos_token_id`, and id 0 is BOS.
`Gemma4TextModel.embed_tokens` therefore has **no `padding_idx`** — setting one froze BOS at zero.

## Training loop notes ([scripts/pretrain.py](scripts/pretrain.py))

- Order: tokenizer -> `Dataset` -> `DataLoader(batch_size=None, num_workers=4)` -> model ->
  optimizer/scheduler -> **checkpoint resume** -> `dry_run` -> `Accelerator.prepare`. The resume
  happens on the *unwrapped* model, before `prepare`.
- `dry_run` exercises the packed path with synthetic docs and asserts a finite loss, then restores
  the token counter so it doesn't pollute a resumed count.
- LR schedule: linear warmup -> cosine to `0.1 * lr`. On resume it is **re-anchored by tokens**
  (`resume_token_count // tokens_per_step`) rather than by saved step, because `total_steps` moves
  with batch size / grad accumulation.
- Router exploration noise anneals 1 -> 0 over `noise_anneal_tokens`, driven from the live token
  count each step.
- Ponder loss (`TrainingConfig.lambda_ponder`, applied to `(1 - p_halt)` on real tokens) ramps
  0 -> `lambda_ponder` over `ponder_warmup_tokens` -> `ponder_warmup_tokens + ponder_ramp_tokens`,
  also driven from the live token count — see the ponder-deadlock invariant above.
- Everything that needs the host (loss `.item()`, token sync, tokens/sec, peak mem) is throttled
  to `LOG_INTERVAL`. Keep it that way — the model is small enough that per-step syncs dominate.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, or `moe` must go through
  `accelerator.unwrap_model(model)`; the DDP wrapper has none of those attributes.
- `KeyboardInterrupt` prompts on stdin before saving an `*_interrupted.pt`.

## Checkpoints & resume

`ckpts/training/checkpoint_epoch{E}_idx{STEP}_loss{L}[_interrupted].pt`; "latest" means newest
mtime, not highest step. Payload (see [utils.py](utils.py)): model/optimizer/scheduler states,
`token_count`, `losses`, `file_order`, a **global** `(file_idx, record_idx, shard_token_count)`,
and **per-worker** positions keyed by worker id plus the `num_data_workers` they were produced
with.

- Per-worker resume is only valid when `NUM_DATA_WORKERS` is unchanged; otherwise the loader falls
  back to the conservative global minimum (and re-reads some data).
- Restoring `file_order` is what stops consumed shards from reappearing — `file_idx` indexes into
  that order, not into the sorted file list. `build_legacy_order` handles pre-`file_order`
  checkpoints.
- All `load_checkpoint` extras use `.get(..., default)` so old checkpoints still load. Keep that
  when adding fields, and add them to `save_checkpoint`'s signature with a default too.

## Dataset ([modules/data/dataset.py](modules/data/dataset.py))

`IterableDataset` yielding **fully assembled batches** (hence `batch_size=None` on the DataLoader).
Files are discovered from `data_config.json` roots (parquet/json/jsonl, optional `glob`), sorted
for determinism, then shuffled per epoch with `seed + epoch`. Workers shard the *same* global
order by `idx::num_workers`, so `file_idx` is globally meaningful.

Packing: documents are concatenated into `max_length` sequences, split across sequence boundaries
when they don't fit, each followed by `EOS + (num_mtp_tokens - 1)` pads. Trailing padding becomes
length-1 attention segments. Labels are `-100` everywhere except the interior of each document
block plus the terminating EOS. `max_tokens_per_shard` caps how much is drawn from one file before
moving on.

Batches carry `file_idx / record_idx / shard_token_count / worker_id` as `[B]`-shaped tensors
purely so accelerate's batch splitting treats them like `input_ids`.

## Conventions

- Comments are lowercase, explanatory, and justify *why* (especially around sync avoidance,
  checkpoint recompute, and accelerate's batch handling). Match that density; don't strip them.
- Google-style docstrings with an `Args:` block on the public modules.
- Config values flow yaml -> `config.py` -> kwargs. Don't read `config.yaml` from a module under
  `modules/`.
- `utils.logger` (yellow-formatted) is the logging channel; scripts use `print` only in the
  inference CLI.

## Git

Current branch `train-build`; PRs target `main`. Commit style is
`feat:` / `docs:` / `chore:` / `merge:`. Note the `.gitignore` swallows `*.json` (so
`data_config.json` is untracked), `*.cmd`, `ckpts/`, `venv/`, and `env_init`. `tests/` is tracked.

## Known rough edges

- `flash-attn` and `transformer-engine` in `requirements.txt` need CUDA builds matched to the GPU;
  a plain `pip install -r requirements.txt` will usually fail on them.
- `huggingface.key` sits in the repo root (gitignored via `*.key`).
- `inference.py` runs the model with no `cu_seqlens` (plain causal) and no KV cache — it re-runs
  the full prefix per token, which is fine for smoke-testing checkpoints and slow for anything else.
- README notes token counts can be inflated by tens of tokens per batch.

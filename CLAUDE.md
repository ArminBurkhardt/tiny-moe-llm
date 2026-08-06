# CLAUDE.md

Working notes for this repo. Prose docs live in [docs/](docs/) — this file is the operational
map: what lives where, what the non-obvious invariants are, and how to run things.

## What this is

`tiny-moe-llm`: an experimental ~243M-param LM. A dense Gemma4-style decoder feeds a **single
MoE block applied `n_loops` times** (LoopLM-style recurrence), with a heterogeneous expert pool
(self-attn / cross-attn / information-retrieval / MLP) behind one router plus always-on shared
experts, a per-loop halt head, and multi-token prediction heads. Trained on document-packed
streams with flash-attn varlen attention, optionally in FP8/NVFP4 via NVIDIA Transformer Engine.

Research code, not a library: no packaging, no test framework, no CI. Entry points are the
scripts under [scripts/](scripts/).

## Layout

```
config.py / config.yaml     all hyperparameters (yaml -> ModelConfig / TrainingConfig)
utils.py                    logger, BASE_DIR, dtype aliases, TOKENIZER_REPO/TOKENIZER_DIR/
                             HF_UPLOAD_REPO, get_hf_token, save/load_checkpoint
data_config.json            local parquet source roots + column names per mode (gitignored) --
                             no longer read by any script since the Step 9 mmap dataset rewrite
                             (`prune_vocab.py` hardcodes its own mix instead); kept around as a
                             record of the pre-Step-9 local shard layout
memory-benchmark.py         standalone peak-VRAM probes (not used by training)
env_init                    WSL/CUDA env + venv activation (gitignored, `source env_init`).
                             The rented box has no counterpart -- `scripts/setup.sh` installs into
                             the NGC image's system python, and the test runners take
                             `TINY_LLM_ENV_INIT=/dev/null` there (`vast_init` is deleted)
scripts/
  run_training.py           THE unattended entry point: supervises phase1 -> phase2, relaunches
                             through preemptions, honours the exit-code contract
  pretrain.py               THE training loop; `pretrain(phase=None) -> exit code`, `--phase`
  setup.sh                  box setup: deps, huggingface.key, tokenizer, upload/gated preflight
  onstart.sh                vast.ai onstart hook: clone, setup.sh, nohup run_training.py
  fetch_tokenizer.py        pulls TOKENIZER_REPO into TOKENIZER_DIR (ckpts/ is gitignored)
  inference.py              greedy/top-k sampling CLI
  prune_vocab.py            one-shot 129280 -> 65536 vocab prune (Step 8), not part of training
  prepare_data.py           builds phase1/phase2.bin/.idx from the Hub source mix (Step 11), runs
                             on the rented box, not locally -- see "Data prep" below
  prepare_sft_data.py       builds sft_train/sft_val .bin/.idx/.mask from the Step 12 source mix,
                             runs LOCALLY -- see "SFT" below
  sft.py                    THE post-training entry point (Step 12): local, single GPU, reuses
                             pretrain.train_step verbatim
  eval_calibration.py       Gate 5: ECE / abstention AUROC for the correctness head
  __init__.py               empty; exists only so tests can `from scripts.run_training import ...`
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
modules/data/dataset.py     mmap flat-file IterableDataset (bin/idx) with document packing (Step 9)
modules/data/sft_dataset.py mmap SFT dataset (bin/idx/mask): explicit loss mask, per-epoch shuffle,
                             never splits a conversation across rows (Step 12)
modules/data/chat.py        chat template + token-level loss masking (Step 12)
modules/data/abstention.py  the fixed abstention/hedge phrasings (Steps 12 and 13)
modules/runtime/            unattended-run machinery. MUST NOT import torch.nn, transformer_engine
                             or anything under modules/model/ -- every test here runs GPU-free
  checkpoints.py            naming, stale cleanup, latest-VALID resume, retention, phase scoping,
                             run_state sidecar, verify_resume, resume_phase_index (which phase the
                             supervisor should (re)start at)
  hf_sync.py                HFSync: background upload thread, retries, droppable jobs, drain,
                             remote delete + throttled history squash (mirrors local retention)
  ponder.py                 PonderController: runtime lambda_ponder auto-adjust, checkpointed
  control.py                RunControl: STOP sentinel + SIGTERM/SIGUSR1, EXIT_* contract
  status.py                 write_status (atomic), eta_seconds, format_duration
tests/                      tracked sanity scripts (see "Tests")
ckpts/                      gitignored: pretrained/<tokenizer dirs>, training/<*.pt, *.png,
                             run_state.json, status.json, STOP>
data/datasets/              gitignored parquet/jsonl shards
docs/runbook.md             what to run on the box, how to stop it, what is normal, what is not
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
- **FP8 is gated by an env var, off by default** (PLAN.md Step 10): `scripts/pretrain.py`'s
  `USE_LOW_PRECISION = os.environ.get("USE_FP8", "0") == "1"` picks `chosen_recipe = fp8_recipe`
  (TE `DelayedScaling`, `Format.HYBRID`) when set, else `None`/BF16 — same code path on the local
  5090 (`USE_FP8` unset) and a rented H100 (`USE_FP8=1`). `te.autocast(enabled=False)` still wraps
  `ParallelSparseMoELayer`'s GEMMs regardless (NVFP4/row-divisibility reasons, see moe.py).

## Commands

```bash
source env_init                      # WSL: CUDA paths + venv (local box only)
bash scripts/setup.sh --hf-token X   # rented box only: deps, token, tokenizer, preflight
python scripts/run_training.py       # the real run: phase1 -> phase2, restarts on preemption
python scripts/pretrain.py --phase phase1   # one phase; resumes from the newest LOADABLE ckpt
python scripts/prepare_sft_data.py   # Step 12 corpus; needs manifest.json's holdout hashes first
python scripts/sft.py --from-hub     # Step 12 SFT: pull the pretrained ckpt + manifest, then train
python scripts/inference.py          # interactive; -c CKPT -p PROMPT -n 200 --temperature 0.8
bash tests/run_env_check.sh          # torch/flash/TE/tokenizer smoke check
bash tests/run_tests.sh tests/test_attention_equiv.py tests/test_overfit.py
touch ckpts/training/STOP            # stop the run cleanly (exit 10, supervisor does not restart)
kill -USR1 <pid>                     # checkpoint now, keep training
```

Tests are plain scripts (no pytest): each `sys.path.insert`s the repo root, asserts, and prints.
Most require a GPU — the exceptions are the `modules/runtime/` tests (`test_checkpoint_lifecycle`,
`test_hf_sync`, `test_control`, `test_supervisor`, `test_phase_targets`, `test_hf_token`,
`test_checkpoint_atomic`, `test_ponder_autoadjust`), `test_prepare_data` and `test_sft_dataset`,
which run anywhere.
`tests/` is tracked (the
`.gitignore` line for it is commented out) — edits land in normal commits like any other source
file.

Operational detail (what to run on the box, how to stop it, what is normal, what to do when it is
not) lives in [docs/runbook.md](docs/runbook.md), not here.

## Config

`config.yaml` -> `config.py` exposes four surfaces:

- `ModelConfig.Params` — kwargs splatted straight into `TinyMoETransformer(**...)`.
- `ModelConfig.Forward` — kwargs splatted into `model(...)`; currently empty (`identity_skew` was
  its only key, deleted in PLAN.md Step 3).
- `TrainingConfig` — class attributes; `total_steps` is *derived* as
  `target_tokens // (batch_size * seq_length * grad_accumulation_steps)`. Also holds the ponder
  loss knobs (`lambda_ponder`, `ponder_warmup_tokens`, `ponder_ramp_tokens`), `loop_ce_weights` +
  `loop_ce_subsample` (PLAN.md Step 4a), `loop_count_sampling`, and `lambda_conf` (PLAN.md Step 4b)
  even though they read from
  `config.yaml`'s `training:` block rather than `model:` — they're consumed directly in
  `scripts/pretrain.py`'s / `compute_mtp_loss`'s loss calc, not passed into the model.
  `loop_ce_weights`' length is asserted against `n_loops` at config-load time (import-time
  `assert` in `config.py`, not construction time). `data_dir` (default `data/prepared`) and
  `phase` (default `phase1`) pick which `{phase}.bin`/`{phase}.idx` pair the mmap `Dataset`
  reads (PLAN.md Step 9) -- both are gitignored artifacts produced by `scripts/prepare_data.py`
  (Step 11, see "Data prep" below); a real `pretrain()` run has nothing to read until that's
  been run once, normally on the rented box.
- `SFTConfig` — the `sft:` block (PLAN.md Step 12). Only what SFT genuinely does differently: its
  own lr/epochs/batch/seq, the `{train,val}_split` stems, the per-epoch shuffle `seed`, the eval and
  checkpoint cadences, and a `dropout` override applied via `SFTConfig.model_params()` (dropout is
  not a parameter, so the pretrained state dict still loads unchanged). **Every loss weight is
  absent on purpose** — `scripts/sft.py` reuses `pretrain.train_step`, which reads them from
  `TrainingConfig`. Same `None`-vs-`""` `hf_upload_repo` distinction as `TrainingConfig`, but it
  defaults to `""` (uploads off) because SFT is a local run.

Constraints worth remembering:
- `moe_intermediate_size` (optional) sizes the routed MLP experts and the always-on shared
  MLP/attn (see "Model invariants" below); defaults to `intermediate_size` if omitted from
  `config.yaml`. `Gemma4TextModel`'s dense decoder always uses plain `intermediate_size`, so this
  is the only knob that moves total params without moving active (dense-decoder) params.
- `mtp_num_extra_tokens` must be <= the dataset's `num_mtp_tokens` separator budget, otherwise
  MTP gets supervised across document boundaries. Currently trivially true: `scripts/pretrain.py`
  passes `Dataset(..., num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"])` -- the same
  value on both sides -- so there's nothing to assert yet. The Step 9 mmap dataset still adds
  the EOS/pad separator dynamically at pack time (not baked into `train.bin`), so this stays
  trivially true for now; revisit if a later step bakes the separator budget into the bin file
  itself, independent of the model config.
- `vocab_size` and `hidden_size` must both be divisible by `lm_head_factor` (SmallLMHead chunks
  both dims), and (if MTP is enabled) by `lm_head_factor * 2` for the MTP head's own `SmallLMHead`
  (which runs on `hidden_size // 2`). `vocab_size` must also be `<= 65536` (Step 8's `train.bin` is
  uint16). **Asserted at model construction** (`TinyMoETransformer.__init__`, PLAN.md Step 5) --
  distinct from `loop_ce_weights`' config-load-time assert above.
- Things *not* in the yaml but hardcoded: `NUM_DATA_WORKERS=4`, `LOG_INTERVAL=10`, checkpoint
  every 1500 steps (was 5000, PLAN.md Step 6 -- interruptible-instance granularity)
  ([pretrain.py](scripts/pretrain.py)), expert head counts `n_heads=16 /
  n_kv_heads=4` and `rope_theta` ([moe.py](modules/model/moe.py)), `CE_CHUNK_SIZE=2048`
  ([mtp.py](modules/model/mtp.py)).
- `TinyMoETransformer.__init__` prints total/active param counts and a forward FLOP/token estimate
  (PLAN.md Step 5) -- PLAN.md's Step 11 budget math is keyed to this number and goes stale
  silently if it's not recomputed after a config change. "Active" excludes the routed MLP experts'
  unused capacity (`num_mlp_experts` weights exist but only `top_k` run per token); "excl. emb"
  further drops `embed_tokens` / the decoder's PLE table / this model's own PLE table (lookups, not
  matmuls).
- **FLOPs are exposed as three separately-scaled components, not one per-token constant** --
  folding them together is what made the pre-fix estimate understate real compute by ~2x:
  - `body_flops_per_token` -- dense decoder + MoE matmuls. The MoE portion multiplies by `n_loops`
    (one shared module reused every loop: param count appears once, compute happens `n_loops`
    times); the decoder runs once.
  - `lm_head_flops_per_token` (**per application** -- `lm_head` runs once *per loop* for per-loop
    CE, not once) and `mtp_flops_per_token`. Both are chunk-checkpointed inside `compute_mtp_loss`,
    so they cost fwd + recompute + bwd (**4x**) while the body costs 3x (activation checkpointing
    is off).
  - `attn_flops_per_seqsq` -- attention scales with `sum(segment_len^2)`, not token count, so it
    can't be a per-token number under document packing at all. `scripts/pretrain.py` accumulates
    the real `sum(seg^2)` on-device from `cu_seqlens` and drains it at `LOG_INTERVAL`.
  `flops_per_token_fwd` is still published as a single log-line figure, now assuming a fully-packed
  `max_seq_len` (worst case for the attention term).

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
  `hidden_states = hidden_states + (1 - p_halt) * loop_scale[loop] * dropout(post_norm(output))`,
  giving a gradient path across loop boundaries independent of routing. `loop_scale` is an
  `nn.Parameter` of shape `[n_loops]` — **one gate per loop**, init `1/sqrt(n_loops)`, not `0` and
  no longer the old shared `0.1` scalar (see the comment at its definition in
  [moe.py](modules/model/moe.py): `post_norm` makes every delta unit-RMS, so at `0.1` the whole
  MoE block was a ~1.5% perturbation of a unit-RMS decoder output, and a lone scalar at `lr=4e-4`
  can't climb out of that inside a short run). Distinct from `layer_scalar` in the dense decoder
  (init-1 whole-layer gain). Loop indices past `n_loops - 1` reuse the last entry, so an
  inference-time loop-count override doesn't fall off the table. **Excluded from weight decay** by
  `build_param_groups` — decaying it is decaying the loop toward "off".
- **Routing is conditioned on the loop index** so consecutive loops don't all pick the same
  experts: `route()` adds `loop_router_bias(loop_enc[loop])` to the router logits. The encoding is
  *sinusoidal in the absolute loop index* (a `[max_enc_loops, loop_enc_dim]` non-persistent buffer),
  deliberately not a learned `[n_loops, num_experts]` table — the loop count stays a runtime choice,
  so `LoopMixtureOfExperts.forward(..., n_loops=N)` / `TinyMoETransformer.forward(..., n_loops=N)`
  can run a trained checkpoint at any depth without reshaping a weight. `loop_router_bias` is
  zero-init, so routing at step 0 is identical to the unconditioned router.
- **Halt head** (PLAN.md Step 3a, replaces the deleted identity expert): `self.halt_proj` is a
  `Linear(hidden_size, 1)`, zero-init weight / bias `-2.0` (`p_halt ~ 0.12` at init), applied to
  the *incoming* hidden state each loop before the update above. `p_halt -> 1` means "don't modify
  me further" — a compute-allocation signal, not a correctness score (that's Step 4b's separate
  `correct_proj`, not yet implemented). It's **greedy per-loop, not cumulative ACT**: recomputed
  fresh each loop, so a token can halt at loop 1 and un-halt at loop 2. `LoopMixtureOfExperts.forward`
  stacks per-loop `p_halt` into `[n_loops, B, S]` and returns it alongside `hidden_states` — never
  reduced with `.item()` inside the model.
- **Anything reducing `p_halt` must divide by the loop axis too.** `p_halt` is `[n_loops, B, S]`
  while `pad_mask`/`valid_mask` are `[B, S]`, so `sum() / valid_mask.sum()` is `n_loops` times too
  large. Both the ponder term and the logged `p_halt` had this bug; `train_step` now normalizes by
  `valid_mask.sum() * p_halt.size(0)`. Symptom if it regresses: logged `p_halt` can never read
  below `n_loops * sigmoid(halt_bias)` (~0.36 at init instead of ~0.12), and `lambda_ponder` is
  silently `n_loops` × its configured value.
- **Ponder loss requires its warmup to actually be wired up** (`TrainingConfig.ponder_warmup_tokens`
  / `ponder_ramp_tokens`, applied in `scripts/pretrain.py`'s `train_step`) — this is a correctness
  requirement, not tuning. At small `loop_scale`, CE loss has near-zero gradient wrt `p_halt`, so an
  un-warmed ponder term is briefly the halt head's only (constant-sign) signal; AdamW climbs the
  halt bias regardless of `lambda_ponder`'s magnitude, `p_halt` saturates before `loop_scale` moves,
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
  skew/bias step anymore (that was the deleted identity mechanism). One `torch.topk` feeds both it
  and the selection (score renormalization doesn't change indices).
- The non-MLP experts' routing weights are folded into a single `[B, S, first_mlp_index]` gate via
  `scatter_add_` before being applied, rather than a `top_k × first_mlp_index` nested loop of
  `[B, S, H]` masked multiplies — same mask-multiply semantics (still no `mask.sum()`/boolean
  indexing, which would sync per expert), a fraction of the retained activations.
- **Router exploration noise is scaled by `router.noise_scale`** (`ROUTER_NOISE_SCALE = 0.3` in
  [router.py](modules/model/router.py)). The learned `softplus(noise_proj(h))` lands near ~0.7 at
  init while the clean logits' std is ~0.33 — un-scaled, early routing/expert weights/aux loss all
  measure noise rather than the router. This is a ceiling on the *initial* level; `noise_factor`
  still anneals it to 0 over `noise_anneal_tokens`.
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

**Per-loop CE supervision** (PLAN.md Step 4a): with `return_hidden=True` the returned "hidden
states" are actually `self.norm` applied at *every* loop, stacked to `[n_loops, B, S, H]` (index
`[-1]` is the final loop, what MTP reads and what `return_hidden=False` projects to logits) —
`LoopMixtureOfExperts.forward` already returns this stack as a 4th value alongside
`hidden_states`/`aux_loss`/`p_halt`. Without it, intermediate loops are optimized only as inputs
to the next loop, never as something `lm_head` can read, which is what an early-exit policy on
`p_halt` would actually need. `compute_mtp_loss` requires `loop_ce_weights` (one entry per loop,
`TrainingConfig.loop_ce_weights`, length-checked against `n_loops` at config-load time) whenever
`main_lm_head` is set, and applies the chunked CE per loop — never materializing more than one
loop's one chunk of `[chunk, vocab]` logits at a time. `loss_ce` (the value returned for logging)
is still just the *final* loop's raw, unweighted CE. Ascending weights don't guarantee a strictly
descending per-loop CE once training is deep into an overfit regime: earlier loops backprop-receive
gradient from every later loop's CE too (that's just backprop through the recurrence), not only
their own `loop_ce_weights` entry, so their remaining headroom can let them read out a *lower* CE
than a later loop despite the smaller weight. `tests/test_per_loop_ce.py` samples early in training
for exactly this reason — see its comment before changing its step count.

**Correctness head** (PLAN.md Step 4b): `self.correct_proj` (`TinyMoETransformer`, zero-init
weight/bias) is a second, independent head from `p_halt` — `p_halt` asks "is more compute useful
here", `correct_proj` asks "is this specific prediction right", and they come apart on confident
hallucinations (a stable-under-refinement wrong answer halts early *and* reads as certain). Applied
externally, like `lm_head`/`mtp_head`, only inside `compute_mtp_loss` on the **final loop's** hidden
states — never per loop, never inside `forward()`. Its BCE target `is_correct` is free (derived
from the same chunk's CE logits' argmax vs. labels, no extra forward pass) but **must** be computed
under `torch.no_grad()`: without that, the "correct" label would itself carry gradient back through
the LM logits, on top of `correct_proj`'s own gradient — the exact leak `tests/test_correctness_head.py`
checks for by comparing `lm_head`'s gradient with the conf term on vs. off. `lambda_conf` (`TrainingConfig.lambda_conf`,
default `0.05`) is not warmup-gated like `lambda_ponder` — no deadlock precondition applies here,
`correct_proj` doesn't gate anything else's gradient. This head is provisional: Gate 5
(`scripts/eval_calibration.py`) compares its ECE/abstention-AUROC against the
free `p_max = softmax(logits).max()` baseline, and Step 4b reverts (head, loss term, `lambda_conf`)
if `p_correct` doesn't beat `p_max` on both. Gate 5 has been *run* once, on the 45M-token Gate 4
checkpoint: `p_max` won on both metrics, but the revert was **deferred** rather than taken -- only
one real cloud run remains and `correct_proj` is proven gradient-isolated and compute-free, so the
call gets re-made against the real final checkpoint.

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
Both invariants (and the byte-level round-trip guarantee) carry over unchanged into the pruned
65536-vocab tokenizer below — they hold on the *old* id numbering trivially, and the prune's id
remap sorts kept old ids ascending before renumbering, so id 0 (the global minimum, always kept)
lands on new id 0 again and the single pad/eos id keeps whatever new id it's remapped to.

**Vocab prune** (`scripts/prune_vocab.py`, PLAN.md Step 8): `ckpts/pretrained/DeepSeek-V4-Pro-tokenizer`
(129280 tokens) is pruned to exactly 65536 so `scripts/prepare_data.py`'s `train.bin` (Step 11) can
be uint16 instead of uint32 — a disk constraint, not a param-count one. **Every entry point now
reads one constant**, `utils.TOKENIZER_DIR` (default `ckpts/pretrained/DeepSeek-V4-Pro-tokenizer-65536`,
overridable with `$TINY_LLM_TOKENIZER`), instead of six independent hardcoded paths — `ckpts/` is
gitignored, so a fresh clone on the box used to fail in six places with six different messages.
`scripts/fetch_tokenizer.py` downloads it from `utils.TOKENIZER_REPO`
(`ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536`, public, no token needed).
`prune_vocab.py`'s own `SRC_TOKENIZER_DIR` is deliberately *not* that constant — it points at the
unpruned 129280-token source, which is the prune's input, not the trained model's tokenizer.
The prune script:
- Samples ~2GB of text from a **local stand-in** for Step 11's phase-1 mix (`data_config.json`'s
  `pretrain` sources: `fineweb`=web -- absorbing the DCLM/FinePDFs weight that has no local shard,
  `nemotron-pre-specialized-v1.1`=code, `nemotron-pre-math-v1/4plus_MIND`=math, `wikipedia/en`=wiki;
  `nemotron-pre-specialized-v1`/InfiniByte-Reasoning is deliberately excluded, it isn't one of
  Step 11's phase-1 rows), counts how often the **current** tokenizer emits each of its 128000 base
  BPE ids over that sample, and keeps the most frequent ones.
- Every special/added token (1283 ids) and the 256-entry byte-level alphabet are unconditionally
  required regardless of frequency — with `byte_fallback` disabled on this tokenizer, that
  alphabet is the only thing guaranteeing arbitrary text stays encodable after merges are dropped.
- Kept tokens are closed under BPE merge dependency (`select_kept_ids`/`closure` in the script): a
  multi-piece token is only kept if both the pieces it was merged from are kept too, recursively
  down to the byte alphabet — "a kept merge never depends on a dropped one."
- The id remap sorts all kept *old* ids ascending and renumbers `0..65535`, which is why BOS
  staying at id 0 and pad==eos need no special-casing (see above). Written alongside the new
  tokenizer as `id_remap.json`.
- Gated on **fertility** (tokens/byte), not round-trip identity — byte-level fallback makes
  round-trips pass almost regardless of prune damage; the real cost of a bad prune is common text
  costing more tokens. Measured on a held-out ~200MB sample (disjoint files from the frequency
  sample, same source mix) via `manifest.json`'s `vocab_prune` key; last measured overall
  regression 0.15% (well under the 3% acceptance bound), worst single source (wiki) 0.37%.

## Data prep (`scripts/prepare_data.py`, PLAN.md Step 11)

Builds `phase1.bin`/`.idx` and `phase2.bin`/`.idx` from the seven-source Hub mix, meant to run
**on the rented, interruptible, unattended box**, not locally -- interruption safety is a design
requirement here, not an afterthought.

- **One shard file in flight per source at a time**: download -> tokenize -> append raw content
  ids to `{phase}.bin` + cumulative offsets to `{phase}.idx` -> delete the local shard. Peak disk
  is bounded by final bin size plus a handful of in-flight files, never the source corpora
  (terabytes for fineweb-edu/finepdfs-edu alone).
- **Sources are interleaved document-by-document via smooth weighted round-robin** (`run_phase`'s
  `swrr` dict), not written source-by-source then concatenated -- `modules/data/dataset.py` reads
  sequentially with no shuffling, so the mix ratio has to already be baked into on-disk order.
- **`PLAN.md`'s phase-1 mix row sums to 90%, not 100%** (55+10+7+12+3+3, FineWeb/DCLM/FinePDFs/
  Stack-Edu-code/Nemotron-Math/Wikipedia) -- almost certainly a spec gap rather than an intentional
  10% shortfall. `run_phase`'s caller renormalizes each phase's active weights to sum to 1.0,
  preserving relative ratios rather than leaving part of the token budget unwritten. Revisit if
  PLAN.md is ever corrected with an explicit 8th phase-1 row.
- **Checkpointed every `--checkpoint-docs` documents** (default 2000): `bin`/`idx` are `fsync`ed
  and a `_prepare_state_{phase}.json` sidecar records each source's `(file_idx, row_idx, tokens,
  done)`. On restart, `truncate_to_state` trims `bin`/`idx` back down to exactly what the sidecar
  last confirmed *before* reopening them for append -- a crash between checkpoints can only redo
  up to one checkpoint interval, never desyncs the `bin`/`idx` pairing.
- **State is only advanced on a document actually committed, never at pick time** -- `run_phase`
  buffers a `tokenize_batch`-sized group of `(source, file_idx, row_idx, text)` picks, tokenizes
  them together (batched, avoids per-doc tokenizer call overhead), and only then commits each to
  `bin`/`idx` and advances that source's `file_idx`/`row_idx`. Advancing at pick time was an actual
  bug caught by `tests/test_prepare_data.py`: the commit loop can break mid-batch (target token
  count reached), and any already-"picked" documents past that point would never be written yet
  would be marked consumed, silently losing them forever on the next resume.
- **Two independent `run_phase` calls (fresh process, e.g. across a real interruption) do not
  reproduce the same interleave order** as one uninterrupted call -- the SWRR fractional counters
  live only in memory and reset on every call. This is fine: each source's own document sequence
  is still exactly gap-free and repeat-free across the boundary (what `test_prepare_data.py`
  actually checks), and the acceptance bar is realized token counts within 2% of target, not exact
  byte-for-byte reproducibility.
- **Gated sources**: only `nvidia/Nemotron-CC-Math-v1` (`SourceSpec.gated=True`) needs `HF_TOKEN`
  set *and* the dataset's access request accepted on huggingface.co first; `main()` and the
  per-file downloader both fail with that hint on a 401/403 instead of hanging. The code source
  (`common-pile/stackv2_edu_filtered`, key `code_edu`) replaced the originally-planned
  `nvidia/Nemotron-CC-Code-v1` -- NVIDIA's gate on that one requires a recognized-org account and
  was rejecting independent-developer access requests outright, not just delaying them. Common
  Pile's release is fully public: it's Stack-Edu's educational-quality code selection (same
  classifier-based curation, filtered to openly-licensed repos only) with the actual `text`
  re-materialized as gzipped JSONL, sidestepping the Software Heritage-ID reconstruction step that
  the *original* `HuggingFaceTB/stack-edu` would have required (that's why PLAN.md's Step 11 table
  explicitly ruled plain `stack-edu` out -- the Common Pile derivative didn't exist yet when that
  was written). Format is `"jsonl.gz"`, read via Python's stdlib `gzip` (no new dependency, unlike
  `dclm`'s `zstandard`-based `"jsonl.zst"`).
- **Text column is auto-detected at runtime**, not hardcoded, via `SourceSpec.text_columns`
  candidate tuples (`pick_text_column`) -- several sources here (the gated Nemotron-Math one
  especially) had schemas that couldn't be verified offline; failing loudly with the actual
  observed columns beats silently reading the wrong field for 25B tokens unattended.
- `smoltalk2`'s `messages` field is rendered as plain `"role: content"` turns (not a real chat
  template -- that's Step 12 SFT's job), and only its non-reasoning (`_no_think`) SFT splits are
  used. Every document actually consumed from it gets a `sha1(text)[:16]` recorded into
  `manifest.json`'s `data_prep.smoltalk2_holdout_hashes`, so Step 12 can exclude anything already
  seen in phase-2 pretraining.
- Tokenization relies on the fast tokenizer's own Rust-side thread parallelism (all cores) rather
  than wrapping it in a `ProcessPoolExecutor` -- multiprocessing plus a loaded fast tokenizer is a
  known fork-deadlock risk, a bad trade for an unattended box with no one watching to restart it.
- `tests/test_prepare_data.py` exercises `run_phase`/`truncate_to_state` against synthetic
  in-memory sources (no HF Hub calls, no GPU/TE) -- mix-ratio tracking, bin/idx consistency,
  interrupt-then-resume gap/repeat-freedom, and holdout-hash bookkeeping.

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
- **`TrainingConfig.lambda_ponder` is only the ramp's target at a cold start.** `pretrain()`
  constructs a `modules.runtime.ponder.PonderController` seeded from it, and `train_step` reads
  the controller's live `.lambda_ponder` (a `lambda_ponder=` param, not `TrainingConfig` directly)
  every step. After the warmup+ramp finishes, `train_step` feeds the controller each log
  interval's `p_halt_mean`; the controller keeps an EMA and, once every `ponder_adjust_cooldown_
  tokens`, nudges its value by `ponder_adjust_factor` if the EMA sits outside
  `ponder_target_p_halt +/- ponder_p_halt_band` (up if `p_halt` reads too low, down if too high —
  see the controller's docstring for why that's the correct direction), clamped to `[ponder_
  lambda_min, ponder_lambda_max]`. `PonderController.state_dict()` is checkpointed
  (`save_checkpoint`'s `ponder_state`) and restored on resume via `load_state_dict` — **without
  this the adjustment would evaporate on every preemption restart**, since `config.yaml` is
  re-read unmoved at every relaunch and only the controller's own persisted state remembers that
  it moved away from that starting value. `ponder_auto_adjust: false` disables the whole
  mechanism and pins `lambda_ponder` at the config value, matching the pre-auto-adjust behaviour.
- Main CE is now a weighted sum over `n_loops` per-loop CE terms (`TrainingConfig.loop_ce_weights`,
  PLAN.md Step 4a) rather than just the final loop's — see the per-loop CE invariant above. The
  **non-final** loops are token-subsampled by `TrainingConfig.loop_ce_subsample` (default `0.25`);
  the final loop is always supervised in full. A CE mean over a uniform subsample is an unbiased
  estimate of the full mean, so `loop_ce_weights` semantics are unchanged — only the variance on
  the low-weight intermediate readouts goes up, in exchange for not running the model's largest
  GEMM `n_loops` times at full width. `1.0` disables it.
- **Stochastic loop depth** (`TrainingConfig.loop_count_sampling`, default `0.3`): that fraction of
  steps runs a uniformly random depth in `1..n_loops-1` via `sample_n_loops`, the rest run full
  depth. `loop_ce_weights` is truncated and rescaled by `loop_ce_weights_for(n)` so the deepest loop
  actually run always carries weight `1.0` (and holds the correctness head) — truncating alone
  would shrink the whole CE term on shallow steps, i.e. a per-step LR change. Purpose: make every
  depth a real operating point so an inference-time `n_loops` override lands on something the model
  trained at; it also cuts mean body compute (~85% of always-full at `p=0.3`, `n_loops=3`).
  **Log steps are pinned to full depth** so `losses`/per-loop CE/`p_halt` are always read at the
  same operating point. This replaces the ponder loss as the "bounded refinement" mechanism —
  `p_halt` gates the loop's *output* while every expert still runs, so penalizing it buys back no
  compute and only pushes the loop toward a no-op.
- **Optimizer uses two param groups** (`build_param_groups`): weight decay applies only to tensors
  with `ndim >= 2`. Norms/biases/gates are excluded because their zero is a *degenerate state*, not
  just a regularization preference — `moe.loop_scale` decayed toward 0 is the loop decayed toward
  "off", and `layer_scalar` is a gain on the whole residual stream so its decay compounds across
  depth (~0.5x over 8 layers at `lr=4e-4`/`wd=0.02` over a 5B-token run, before any gradient).
- **A parameter stepped through an fp32 master is NOT in any optimizer param group, so
  `optimizer.zero_grad()` never clears its `.grad` — `train_step` has to, by hand, gated on
  `accelerator.sync_gradients` exactly like the step itself** (clearing every micro step would
  throw away grad accumulation for those tensors). That loop is load-bearing, not tidiness:
  without it the bf16 `.grad` accumulates across every
  optimizer step for the entire run, and (a) `accelerator.clip_grad_norm_(model.parameters(), ...)`
  reads those tensors, so the total norm grows without bound and the clip coefficient throttles
  *every other* parameter's gradient — a silent run-wide LR collapse behind a loss curve that still
  descends, just far too slowly — and (b) `sync_master_grads_` copies a run-length sum instead of
  the step's gradient, giving the shadowed params a permanent non-decaying momentum. This bites
  proportionally harder in `scripts/sft.py`, where *every* parameter is shadowed.
- **A checkpoint that exists but fails to load is never a fresh start.** The resume path
  distinguishes "no file in `ckpts/training`" (warn, start from scratch) from "files exist but
  none loads" (raise). Collapsing both into one warning is how a preempted box silently restarts
  from token 0 with a plausible-looking loss curve. `find_resume_checkpoint` softens only the
  middle case -- one corrupt newest file falls back to the next oldest -- which is why
  `verify_resume` exists to bound how far back that fallback may silently go.
- **`collect_metrics` is gated on the log cadence.** `train_step(..., collect_metrics=step %
  LOG_INTERVAL == 0)`; `metrics` is `None` otherwise. The `p_max`/`p_correct` reductions run over
  every CE chunk's live logits *and* again on the checkpoint recompute, so gathering them on the
  other 9 steps in 10 is pure waste. `p_max` itself is computed as
  `1 / sum_j exp(l_j - l_max)` rather than `logits.float().softmax(-1).max(-1)` — mathematically
  identical, but avoids two ~537MB fp32 transients per chunk at `chunk=2048`/`vocab=65536`.
- The correctness head's BCE loss (`TrainingConfig.lambda_conf`, PLAN.md Step 4b) is added
  unconditionally, no warmup ramp — see the correctness-head invariant above.
- Everything that needs the host (loss `.item()`, token sync, tokens/sec, peak mem) is throttled
  to `LOG_INTERVAL`. Keep it that way — the model is small enough that per-step syncs dominate.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, or `moe` must go through
  `accelerator.unwrap_model(model)`; the DDP wrapper has none of those attributes.
- **`KeyboardInterrupt`'s `input()` prompt is gated on `sys.stdin.isatty()`.** On a box with no tty
  it raises `EOFError`, which the old bare `except Exception` swallowed -- so the interrupt path
  saved nothing. Worse, vast preemption sends SIGTERM, which never raised `KeyboardInterrupt` at
  all, so that path never ran on the one machine it was written for.
- **Training stops at the PHASE's token target, not `target_tokens` and not an epoch count**
  (PLAN.md Step 6): `num_epochs: 1` in `config.yaml` now just bounds the outer loop as a safety
  net. The real stop condition is `n_tokens >= phase_target`
  (`TrainingConfig.phase_target_tokens(phase)`), checked inside the existing `LOG_INTERVAL` block
  (the sync-free counter is already being drained there for logging, so this adds no extra sync).
  `target_tokens` stays the **combined** budget and still anchors `total_steps` and the cosine, so
  phase 2 continues the decay instead of restarting it. Without a stop, a dataset bigger than the
  target would train past the point the schedule was anchored to, wasting the anneal.
- **`compute_mtp_loss(..., return_metrics=True)`** (PLAN.md Step 7) returns a 3rd value, a dict of
  still-on-device tensors: `per_loop_ce` (list, one per loop), `conf_loss`, `p_correct`, `p_max`,
  `top1_acc` (the last three `None` if `correct_proj` wasn't passed). `train_step` adds `p_halt_mean`,
  `ponder`, and `lambda_ponder_now` (already host-side, see the ponder ramp above) to the same dict
  and returns it as a 4th value. Default `return_metrics=False` (2-tuple return) is unchanged, so
  existing test call sites don't need updating. The metrics are cheap reductions over each chunk's
  *already-materialized* logits/`correct_logit` inside `_chunked_linear_ce`'s existing chunked/
  checkpointed loop — not a second forward pass — so requesting them doesn't add a real sync or
  meaningfully more compute; only `.item()`-ing the dict's values (done once, inside the
  `LOG_INTERVAL` block) is the actual host sync.

## Checkpoints & resume

`ckpts/training/checkpoint_{phase}_tok{N}M_loss{L}.pt` for rolling saves, plus exactly one
`checkpoint_{phase}_final.pt` per phase (`modules/runtime/checkpoints.rolling_name`/`final_name`).
Epoch/step are deliberately gone from the name: `num_epochs` is a safety net now and `dataset_idx`
stopped being the resume key when `global_offset` landed, so the token count is the only figure
that says where a checkpoint sits in the run.

**"Latest" means newest mtime that actually LOADS**, not simply newest mtime --
`find_resume_checkpoint` walks the candidates newest-first, logs and skips one that raises, and
only raises itself when *every* candidate fails. Payload (see [utils.py](utils.py)):
model/optimizer/scheduler states, `token_count`, `losses`, `phase`, and a single
**`global_offset`** (PLAN.md Step 9) -- a doc index into the
flat, unshuffled document stream. There is no per-file or per-worker state anymore: doc sharding
across `DataLoader` workers is pure `doc_idx % num_workers` arithmetic (see "Dataset" below), so
one conservative (min-across-workers) scalar is enough to resume from without skipping any
worker's unconsumed documents. Workers further ahead than the minimum at checkpoint time just
redo a few already-seen documents on resume -- harmless, and the same kind of slop the pre-Step-9
design already accepted when the worker count changed.

- `pretrain.py`'s `snapshot_global_offset()` computes this: each worker records the last `doc_idx`
  it reached (`batch["doc_idx"]`, host-synced only at checkpoint time), and the checkpoint stores
  `min(seen) + NUM_DATA_WORKERS` (the smallest "next document any worker still wants").
- Resume is document-granular, not sub-document: if a worker was frozen mid-way through packing a
  document that spans a batch boundary, that document's still-unflushed remainder is not
  separately tracked and is simply redone. `tests/test_dataset_resume.py` checks the guarantee
  the design actually makes (no full document skipped or duplicated at the granularity it
  checkpoints), using corpora where every document fills exactly one row to keep the boundary
  unambiguous.
- Legacy (pre-Step-9) checkpoints have no `global_offset` and restart the doc stream from 0 --
  there's no sound mapping from the old per-file position into the new flat corpus.
- All `load_checkpoint` extras use `.get(..., default)` so old checkpoints still load. Keep that
  when adding fields, and add them to `save_checkpoint`'s signature with a default too.
  `load_checkpoint` returns a **7-tuple** now (`phase`, then `ponder_state` last, both `None` for
  legacy files -- `ponder_state` is `PonderController.state_dict()`, see "Ponder auto-adjust"
  below).
- **Writes are atomic**: `save_checkpoint` writes `path + ".tmp"`, `fsync`s, then `os.replace`s.
  Without this a preemption mid-write leaves a truncated `.pt` that is *also* the newest by mtime,
  i.e. exactly the file the resume would pick. Leftover `.pt.tmp` files are swept at startup by
  `cleanup_stale_files`.
- **Retention deletes a checkpoint only when it is BOTH outside `keep_local_checkpoints` AND
  confirmed uploaded** (`prune_checkpoints`, `HFSync.is_uploaded`). Never relax the second
  condition: a locally-deleted, never-uploaded checkpoint is gone for good, whereas one held past
  the window only costs disk. Sustained upload failure is therefore meant to fill the disk.
  `checkpoint_{phase}_final.pt` is exempt entirely.
- **Cadence is in TOKENS** (`TrainingConfig.checkpoint_every_tokens`), checked inside the existing
  `LOG_INTERVAL` block so it adds no host sync. The old `step % 1500` counted *micro* steps, so at
  `grad_accum 16` it fired every ~49M tokens -- ~608 checkpoints, ~1.2TB, against a 120GB disk.

## Dataset ([modules/data/dataset.py](modules/data/dataset.py))

`IterableDataset` yielding **fully assembled batches** (hence `batch_size=None` on the DataLoader),
reading from a pre-tokenized flat-file corpus (PLAN.md Step 9): `{data_dir}/{phase}.bin` (a flat
uint16 token stream) and `{data_dir}/{phase}.idx` (uint64 document-start offsets, one entry per
document plus a trailing entry == `len(bin)`). Both files are produced by `scripts/prepare_data.py`
(PLAN.md Step 11, see "Data prep" below) -- `phase1`/`phase2` are the two mix ratios described there.

Both files are opened via `np.memmap` **inside** `_batch_iterator` (once per worker/epoch, not
held on the `Dataset` object across its lifetime) -- a long-lived memmap handed across DataLoader
worker restarts is a known leak vector.

Documents are read **once, in on-disk order, with no shuffling**: Step 11 already interleaves
sources at the target mix ratios while writing the bin file, so a straight sequential read
reproduces that mix -- reshuffling here would undo it. Workers shard the doc stream by pure
`doc_idx % num_workers == worker_id` arithmetic (no file lists, no `file_order`, no
`build_legacy_order` -- all dead code from the pre-Step-9 parquet design and removed). `start_doc_idx`
(from the checkpoint's `global_offset`) is the one resume input; each worker derives its own first
owned index from it via `first = start_doc_idx + ((worker_id - start_doc_idx) % num_workers)`.

Packing is otherwise unchanged from the pre-Step-9 design: documents are concatenated into
`max_length` sequences, split across sequence boundaries when they don't fit, each followed by
`EOS + (num_mtp_tokens - 1)` pads. Trailing padding becomes length-1 attention segments. Labels are
`-100` everywhere except the interior of each document block plus the terminating EOS. BOS is
prepended if the document's first stored token isn't already BOS -- idempotent regardless, since
`train.bin` stores raw (BOS-less) content ids only; `prepare_data.py` doesn't bake one in either.

Batches carry `doc_idx / worker_id` as `[B]`-shaped tensors purely so accelerate's batch splitting
treats them like `input_ids`; `doc_idx` is the last global document index this worker had reached
when the batch was assembled, read by the trainer for checkpointing (see "Checkpoints & resume").

## SFT / post-training (PLAN.md Step 12)

`scripts/sft.py` + `scripts/prepare_sft_data.py` + `modules/data/{chat,abstention,sft_dataset}.py`.
**Written for a local run** — the pretrained checkpoint and `manifest.json` come down from the Hub
once pretraining finishes (`sft.py --from-hub`, `prepare_sft_data.py --pull-manifest`; the manifest
is gitignored so a fresh clone never has the holdout hashes). It does run unattended on a rented
box: same `modules/runtime/control` contract as pretraining (SIGTERM → save + exit 20, STOP → exit
10, SIGUSR1 → save and continue) and it returns those codes, so a `while ! python scripts/sft.py;
do :; done`-style wrapper covers an interruptible instance. There is deliberately **no phase
supervisor** — there are no phases here, only epochs, and epoch position is in the checkpoint.
What to actually run on a rented box, and what to change from the local defaults, lives in
[docs/runbook.md](docs/runbook.md) §10, not here.

- **`sft.py` reuses `pretrain.train_step` verbatim**, deliberately. PLAN.md Step 12 wants
  `p_halt`/`p_correct` supervision to stay active during SFT, and the only way to guarantee it stays
  *identical* (per-loop CE weights, aux loss, ponder ramp, correctness BCE, loop-count sampling) is
  to have one copy. Prompt masking needs nothing there: the dataset emits `-100` labels and every
  loss term — including the MTP heads, which read the same `labels` tensor — already honours
  `ignore_index=-100`. Consequence: every loss weight lives in `TrainingConfig`, not `SFTConfig`.
- **The model's global token counter is CONTINUED, not reset.** The ponder ramp and the router
  noise anneal are both driven from it and both have long finished at ~30B tokens; resetting to 0
  would restart the ponder warmup, i.e. silently switch the ponder loss off for all of SFT. SFT
  progress is `token_count - start_token_count`, and `start_token_count` is why `sft.py` writes its
  own checkpoint payload (a strict *superset* of `utils.save_checkpoint`'s, so `inference.py` /
  `eval_calibration.py` read an SFT checkpoint unchanged) instead of extending `utils.py` —
  changing `load_checkpoint`'s tuple arity would break the *live* pretraining run on its next
  preemption relaunch, since `onstart.sh` does `git pull --ff-only` on `train-build`.
- **Every parameter gets an fp32 master, not just the undecayed ones** (`build_sft_param_groups`).
  Pretraining shadows only `ndim <= 1` on the argument that 2D weights' values and steps both scale
  with their init std. That argument does not survive SFT's LR: at `lr=3e-5` a weight near its init
  std ~0.02-0.03 has a bf16 ulp of ~1e-4, three times *larger* than the ~`lr`-sized AdamW step, so
  `param -= lr * update` rounds to exactly the old value forever. At `4e-4` the same step is ~4x
  above the ulp, which is why the narrower fix was right there and wrong here. See also the
  `sync_master_values_` grad-clearing invariant under "Training loop notes".
- **`SFTDataset` differs from the pretraining `Dataset` in exactly three ways**, each forced:
  a third file `{split}.mask` (uint8/token, 1 = supervised) because prompt and completion interleave
  inside a multi-turn conversation and labels can't be derived from position; **conversations are
  never split across rows** (a split tail is supervised with its prompt missing, and loses the
  supervised EOS that teaches the model to stop — over-long ones are dropped, never truncated, at
  ~5-10% trailing-padding cost); and **documents are shuffled per epoch** via a
  `(seed, epoch)`-seeded permutation every worker regenerates independently. `global_offset` is
  therefore a *position in that permutation*, not a raw doc id — which is why `sft.seed` is
  checkpointed and a mid-run change of it is a hard error rather than a silent reshuffle.
- **Separator slots after each conversation are all pads**, unlike the pretraining dataset's
  `EOS + pads`: an SFT document already ends with its own *supervised* EOS.
- **The chat template's control tokens are resolved from the tokenizer and asserted**
  (`ChatTemplate._control_id`). They are DeepSeek's `<｜User｜>` / `<｜Assistant｜>` /
  `<｜begin▁sys｜>` / `<｜end▁sys｜>`, which survived the Step 8 prune because `prune_vocab.py` keeps
  every special/added token unconditionally. They are spelled with explicit `｜`/`▁`
  escapes in `chat.py` — those are FULLWIDTH VERTICAL LINE and LOWER ONE EIGHTH BLOCK, visually
  identical to `|` and `_`, and a mistyped one resolves to a different or missing id. Only the
  assistant's own text and its terminating EOS are ever supervised; markers belong to the prompt
  (inference appends `<｜Assistant｜>` itself, so the model never has to predict it).
- **Conversations with roles outside {system, user, assistant} are dropped whole**, not mangled
  into user turns — smoltalk2's `hermes_function_calling`/`xlam_traces` splits carry `tool` turns,
  which are noise for a calibrated-abstention target and would teach tool syntax the model can
  never complete.
- **`prepare_sft_data.py` honours the smoltalk2 holdout** from `manifest.json`'s
  `data_prep.smoltalk2_holdout_hashes` (phase-2 pretraining already trained on those). Those hashes
  are of `prepare_data.render_pretrain_chat`'s output, so the exclusion **imports that exact
  function** rather than reimplementing the rendering, and hashes the *raw* parquet rows — a
  reimplementation that drifted by one character would silently exclude nothing. It refuses to run
  with an empty holdout list unless `--ignore-holdout` is passed.
- **Only train splits are consumed.** `squad_v2`'s validation split and `gsm8k`'s test split are
  the acceptance-metric eval sets (abstention precision/recall on the unanswerable half); pulling
  them into SFT would make that number meaningless.
- **Abstention phrasings are a small fixed closed set** (`modules/data/abstention.py`), shared with
  Step 13. SQuAD v2's unanswerable rows have a literally empty reference answer, so a phrasing has
  to be supplied; keeping the set closed is what makes `is_abstention` an exact check rather than a
  classification problem, and means Step 13 measures a shift in *when* the model abstains rather
  than in how it words it.
- `estimate_packed_rows` replays the packing rule over `{split}.idx` to anchor the LR schedule.
  "corpus tokens / (batch * seq)" is not a usable estimate here — no-split packing plus separator
  slots easily costs 10%, which would end the cosine well before the data does.

## Run lifecycle (`modules/runtime/`, `scripts/run_training.py`)

Everything needed to leave a 40 hour run unattended on a preemptible box. Operational instructions
are in [docs/runbook.md](docs/runbook.md); these are the invariants.

- **`modules/runtime/` must stay GPU-free.** No `torch.nn`, no `transformer_engine`, nothing under
  `modules/model/`. It imports `utils` (which imports torch) only for `logger`. Every one of its
  tests runs on a machine with no GPU, which is the whole reason the phase-reset logic lives in
  `resolve_resume_scope` rather than inline in `pretrain.py`.
- **The exit-code contract is shared, not local**: `0` phase complete, `10` user stop, `20`
  preempted, `30` resume verification failed (`modules/runtime/control.py`). `pretrain()` returns
  one; `run_training.py` restarts on anything except `TERMINAL_CODES = (10, 30)`. Changing a number
  means changing both sides plus the runbook's table.
- **Signal handlers set a flag and nothing else.** No I/O, no allocation, no logging (the logging
  lock can deadlock inside a signal context). `RunControl.poll()` reads the flag and stats the
  `STOP` sentinel at the `LOG_INTERVAL` cadence -- ~3s granularity, inside vast's SIGTERM grace,
  and no GPU sync.
- **A stale `STOP` sentinel is cleared at startup** (`clear_sentinel()` before `install()`).
  Otherwise the previous run's stop file kills every relaunch before it trains a step.
- **Upload failures never propagate into the training loop.** `HFSync` retries 3x with tripling
  backoff, then logs and gives up on that file. Crashing a 40 hour run over a transient 503 is the
  worse outcome. The failed file stays *unmarked*, which is what makes retention refuse to delete
  it -- see the retention invariant under "Checkpoints & resume". `HFSync.delete()` (queued by
  `pretrain.py`'s `save_and_sync` for whatever `prune_checkpoints` just removed locally) follows
  the identical swallow-and-log policy -- a failed remote delete just leaves stale clutter on the
  Hub, never takes the run down.
- **A Hub file delete alone does not free storage; `HFSync` also squashes history.** `delete_file`
  only removes the blob from the repo's current tree -- git history still references it, so a 2GB
  checkpoint keeps costing storage until the history itself is rewritten. Every successful delete
  calls `super_squash_history` too, throttled to at most once per `squash_min_interval` (default
  1800s) so a burst of same-cycle deletes (one `prune_checkpoints` call can remove several files)
  costs one squash, not one per file. Acceptable only because `temp-train` is a disposable scratch
  mirror -- don't reuse this pattern against a repo anyone reads commit-by-commit.
- **`HFSync.drain()` waits for the in-flight job too, not just an empty queue** (hence `_busy`).
  It is called in `pretrain()`'s `finally`, so a stop or a phase transition cannot race the
  uploader.
- **`run_training.py`'s `main()` does not blindly start at `PHASE_ORDER[0]`.** It calls
  `checkpoints.resume_phase_index(checkpoint_dir)` first and starts there. This exists because a
  vast.ai reclaim kills the whole supervisor, not just the `pretrain.py` child -- `onstart.sh`
  then launches a brand new `run_training.py` process, and the disk (which survives a reclaim)
  can already hold a phase-2 checkpoint. The old unconditional loop re-entered phase 1 regardless,
  which resumed the phase-2 checkpoint under `--phase phase1`, tripped `resolve_resume_scope`'s
  cross-phase reset, immediately hit phase 1's (already-exceeded) token target, and overwrote
  `checkpoint_phase1_final.pt` with phase-2 weights -- destroying it -- before phase 2 restarted
  its own document offset from 0 on top. `resume_phase_index` treats a phase as already complete
  once its final checkpoint is on disk, or once any checkpoint for a *later* phase is found (the
  latter covers a recovery that only restored the phase-2 rolling file, per the runbook).
- **`scripts/setup.sh` fails hard, not just a warning, when no HF token is found AND
  `config.yaml`'s resolved upload repo is non-empty.** `set -euo pipefail` in `onstart.sh` means
  this stops the box before training ever launches, rather than 40 hours of every checkpoint
  upload 401ing silently while retention (correctly) refuses to delete the un-uploaded files and
  the disk fills. `hf_upload_repo: ""` is still a legitimate way to opt out and run local-only.
- **Crossing a phase boundary resets `global_offset`/epoch/step to 0 but PRESERVES `token_count`**
  (`resolve_resume_scope`). Both halves matter: phase 1's ~23M-document offset in phase 2's ~4M
  document corpus makes every worker's `range()` empty, so the dataloader yields zero batches and
  the run exits *looking successful having trained nothing*; and the token count must carry over
  because the cosine is anchored to the combined budget.
- **`verify_resume` is the backstop against silent restart-from-zero.** `run_state.json`
  (`{phase, token_count, checkpoint}`, rewritten atomically at every save) records where the last
  process got to; resuming more than `2 * checkpoint_every_tokens` behind that aborts with exit 30
  rather than burning 40 GPU-hours retraining covered ground. The slack exists to permit
  `find_resume_checkpoint`'s one-file fallback. A different recorded phase means no comparison.
- **`config.yaml`'s `hf_upload_repo: ""` really does disable uploads.** `None` (key absent) is a
  distinct state meaning "use `utils.HF_UPLOAD_REPO`", resolved by `TrainingConfig.upload_repo()`.
  The obvious-looking `hf_upload_repo or HF_UPLOAD_REPO` collapses the two and uploads anyway --
  it was caught by a local smoke run pushing a 2GB checkpoint to the real repo, not by a test.
- **Graphs are fixed filenames, overwritten** (`loss_graph.png`, `expert_selection.png`), written
  inside `save_and_sync` at checkpoint cadence. The old per-step `*_epoch{E}_step{S}.png` naming
  produced them by the thousand; `cleanup_stale_files` sweeps any left over.

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
`data_config.json` is untracked), `*.cmd`, `ckpts/`, `venv/`, `env_init`, and `data/prepared`
(the Step 9/11 `{phase}.bin`/`.idx` corpus). `tests/` is tracked.

## Known rough edges

- `flash-attn` and `transformer-engine` in `requirements.txt` need CUDA builds matched to the GPU;
  a plain `pip install -r requirements.txt` will usually fail on them.
- `huggingface.key` sits in the repo root (gitignored via `*.key`).
- `inference.py` runs the model with no `cu_seqlens` (plain causal) and no KV cache — it re-runs
  the full prefix per token, which is fine for smoke-testing checkpoints and slow for anything else.
- README notes token counts can be inflated by tens of tokens per batch.

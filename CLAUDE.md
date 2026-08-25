# CLAUDE.md

Working notes for this repo. Prose docs live in [docs/](docs/) — this file is the operational
map: what lives where, what the non-obvious invariants are, and how to run things.

## Running anything: WSL + `env_init`

**Every command runs under WSL, from the repo root, after `source env_init`.** The dev box is
Windows; CUDA 12.9, flash-attn and Transformer Engine all live in the WSL Ubuntu install and none
of them exist on the Windows side. `env_init` is gitignored — it exports `CUDA_HOME`, the
include/library paths, `LD_LIBRARY_PATH` (including `/usr/lib/wsl/lib`),
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and activates `venv/`.

From a Windows shell:

```powershell
wsl bash -lc "cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init && python scripts/inference.py --help"
```

Two ways this bites: without `source env_init` nothing under `modules/model/` imports at all
(`transformer_engine` is a hard module-scope dependency), and from any other working directory
`config.py` fails immediately because it opens `config.yaml` by relative path.

`tests/run_tests.sh` already encodes this (`cd` + `source`), and takes `TINY_LLM_ROOT` /
`TINY_LLM_ENV_INIT` overrides so it runs unchanged on the rented box — where `scripts/setup.sh`
has installed into the NGC image's system python and there is nothing to source
(`TINY_LLM_ENV_INIT=/dev/null`).

## What this is

`tiny-moe-llm`: an experimental 332M-param LM. A dense Gemma4-style decoder feeds a **single
MoE block applied `n_loops` times** (LoopLM-style recurrence), with a heterogeneous expert pool
(self-attn / cross-attn / information-retrieval / MLP) behind one router plus always-on shared
experts, and multi-token prediction heads. Trained on document-packed streams with flash-attn
varlen attention, optionally in FP8/NVFP4 via NVIDIA Transformer Engine.

Research code, not a library: no packaging, no test framework, no CI. Entry points are the
scripts under [scripts/](scripts/).

One real run exists: 16B tokens of pretraining on a rented H100 plus a local 2-epoch SFT pass, then
a local 49M-token abstention repair pass on top of it (NEXT.md Phase 2).
[docs/CONCLUSION.md](docs/CONCLUSION.md) is the write-up, including the two failures that Phase 0
below acted on. The plan from here is [docs/plans/NEXT.md](docs/plans/NEXT.md); Phases 0, 1 and 2
are done, and their measurements are in [docs/measurements/](docs/measurements/).

**All future finetuning runs locally on the 5090, in BF16 — do not set `USE_FP8`.** Size the micro
batch to stay resident: 4 x 4096 peaks at 21.4GB, 8 x 4096 peaks at 29.6GB and spills into shared
system memory at a ~3x throughput cost. A cluster with ~5k train-hours is expected for the real
post-POC run.

## Phase 0: what was just removed

Two learned heads were deleted. Both failed structurally, not by mistuning, and the details matter
because half this file used to describe them.

- **Correctness head (`correct_proj`).** Asked "is this prediction right", supervised by BCE
  against `lm_head`'s own argmax on the same hidden state — so "reproduce `p_max`" was the
  reachable optimum by construction, and that is what it learned. `p_max` beat it on ECE and AUROC
  on every checkpoint tested. **`p_max` is now the confidence signal everywhere.**
- **Halt head (`halt_proj`) and the whole ponder subsystem.** Gated the loop update as
  `(1 - p_halt) * loop_scale * delta`. `p_halt` saturated (~0.78 mean, 0.92 on the final loop) and
  the runtime controller cut its λ 11 times with no effect — a saturated sigmoid has no gradient.
  `loop_scale` had grown to `[1.73, 1.81, 1.32]` to compensate.

**Deleting the gate naively would multiply every loop's delta by ~1/(1 − p_halt).**
`scripts/migrate_phase0.py` folds the gate's *measured* per-loop mean into `loop_scale` instead:
`loop_scale_new[k] = loop_scale_old[k] * mean_k(1 - p_halt)`. Measured, not assumed — the training
log only ever recorded `p_halt` averaged over all loops, hiding a strong decreasing trend:

| | `sft_final` | `phase2_final` |
|---|---|---|
| mean `(1 - p_halt)` | `[0.290, 0.134, 0.084]` | `[0.367, 0.179, 0.074]` |
| folded `loop_scale` | `[0.501, 0.242, 0.115]` | `[0.637, 0.326, 0.098]` |

So **a migrated checkpoint's `loop_scale` is far below a fresh `1/sqrt(n_loops)` init, and that is
correct.** The gate is stable across corpora within a checkpoint (~5%) but differs a lot between
checkpoints, hence per-checkpoint measurement. Its per-token std is small (0.03–0.06), which is
what makes folding to a mean legitimate.

Gate P0 passed on both (same held-out slice, identical settings): final-loop CE 3.7564 → 3.7604
and top-1 0.3644 → 0.3628 on `sft_final`; 3.3720 → 3.3843 and 0.3979 → 0.3963 on `phase2_final`.

A migrated checkpoint **drops its optimizer/scheduler state** — its Adam moments are indexed by
param-group position and two tensors left the model. It is a finetune *seed*, not a resume point,
and `utils.load_checkpoint` says so by name if you try.

## Layout

```
config.py / config.yaml     all hyperparameters (yaml -> ModelConfig / TrainingConfig / SFTConfig)
utils.py                    logger, BASE_DIR, dtype aliases, TOKENIZER_REPO/TOKENIZER_DIR/
                             HF_UPLOAD_REPO, get_hf_token, save/load_checkpoint
env_init                    WSL/CUDA env + venv activation (gitignored) -- see the top of this file
scripts/
  run_training.py           THE unattended entry point: supervises phase1 -> phase2, relaunches
                             through preemptions, honours the exit-code contract
  pretrain.py               THE training loop; `pretrain(phase=None) -> exit code`, `--phase`
  setup.sh                  box setup: deps, huggingface.key, tokenizer, upload/gated preflight
  onstart.sh                vast.ai onstart hook: clone, setup.sh, nohup run_training.py
  run_sft_after_pretrain.sh unattended pretrain -> SFT -> abstention eval chain
  fetch_tokenizer.py        pulls TOKENIZER_REPO into TOKENIZER_DIR (ckpts/ is gitignored)
  inference.py              greedy/top-k sampling CLI: KV cache, MTP drafting, convergence exit
  gradio_app.py             browser UI over inference.stream_generate (same path, no duplicate)
  prune_vocab.py            one-shot 129280 -> 65536 vocab prune, not part of training
  migrate_phase0.py         folds the deleted halt gate into loop_scale, strips both old heads
  prepare_data.py           builds phase1/phase2.bin/.idx from the Hub source mix, runs
                             on the rented box, not locally -- see "Data prep" below
  prepare_sft_data.py       builds sft_train/sft_val .bin/.idx/.mask, runs LOCALLY.
                             `--profile repair` builds Phase 2's repair_train/repair_val instead
  archive_corpus.py         pack/list/restore a prepared split as one .tar.gz + sha256 sidecar,
                             so replacing a corpus never costs a re-download -- both prepare
                             scripts delete each source shard as soon as they have appended it
  sft.py                    THE post-training entry point: local, single GPU, reuses
                             pretrain.train_step verbatim. `--repair` runs Phase 2's repair
                             finetune through the same function (RepairConfig, ckpts/repair)
  eval_calibration.py       p_max ECE/AUROC, early-exit curve, loop-convergence statistics.
                             Also the Gate P0 harness for any head removal
  eval_abstention.py        the acceptance metric: SQuAD v2 abstention precision/recall + ECE,
                             LOCAL, needs an SFT checkpoint
  eval_benchmarks.py        the fixed benchmark suite: log-likelihood MC + closed-book generation,
                             ONE scoring path for this model and for the HF peers. Downloads into
                             data/benchmarks (refused inside data/prepared*, data/datasets,
                             data/archives) and peer weights into ckpts/peers
  eval_probe.py             linear answerability probe: fits logistic regression on the final
                             loop's last-position hidden state over squad_v2/TRAIN, reads AUROC on
                             the same standard validation slice eval_abstention.py uses. Answers
                             "is the signal in the representation or only missing from the policy"
  eval_stage0.py            NEXT.md's Phase 1 diagnostics: IR retrieval entropy + ablation (Gate
                             G1), per-loop query drift, residual/readout loop dynamics, oracle
                             minimum sufficient depth. Read-only -- probes via forward hooks so a
                             diagnostic can never change what it measures
  __init__.py               empty; exists only so tests can `from scripts.run_training import ...`
modules/model/
  transformer.py            TinyMoETransformer + TokenTracker + the convergence exit
  gemma4.py                 dense decoder: GQA, RoPE, RMSNorm(te), per-layer embeddings
  moe.py                    LoopMixtureOfExperts, ParallelSparseMoELayer, _ExpertTracking
  router.py                 Router (+ annealed exploration noise), compute_aux_loss
  experts.py                SelfAttention / CrossAttention / InformationRetrievalExpert
  information_retrieval.py  learned key/value table with softmax retrieval
  mtp.py                    MTPHead, chunked LM-head CE, compute_mtp_loss
  attention.py              varlen_attention, cu_seqlens_from_doc_ids, SDPA fallback
  kv_cache.py               LayerKVCache / KVCache: one slot per decoder layer and per
                             (loop, non-MLP expert) pair. Additive; never touches training
  modules.py                SmallLMHead (factored vocab projection)
  embeddings.py             RoPE cache + apply_rotary_pos_emb
  utils.py                  EncoderOutput dataclass
modules/data/dataset.py     mmap flat-file IterableDataset (bin/idx) with document packing
modules/data/sft_dataset.py mmap SFT dataset (bin/idx/mask): explicit loss mask, per-epoch shuffle,
                             never splits a conversation across rows
modules/data/chat.py        chat template + token-level loss masking
modules/data/abstention.py  the fixed abstention/hedge phrasings
modules/runtime/            unattended-run machinery. MUST NOT import torch.nn, transformer_engine
                             or anything under modules/model/ -- every test here runs GPU-free
  checkpoints.py            naming, stale cleanup, latest-VALID resume, retention, phase scoping,
                             run_state sidecar, verify_resume, resume_phase_index
  hf_sync.py                HFSync: background upload thread, retries, droppable jobs, drain,
                             remote delete + throttled history squash
  control.py                RunControl: STOP sentinel + SIGTERM/SIGUSR1, EXIT_* contract
  status.py                 write_status (atomic), eta_seconds, format_duration
tests/                      tracked sanity scripts (see "Tests")
ckpts/                      gitignored: pretrained/<tokenizer dirs>, trained/<the 16B run>,
                             training/<*.pt, *.png, run_state.json, status.json, STOP>
data/datasets/              gitignored parquet/jsonl shards
```

`modules/*/__init__.py` are empty — imports are always fully qualified
(`from modules.model.moe import ...`).

## Commands

```bash
source env_init                      # WSL: CUDA paths + venv. Required. See the top of this file
bash scripts/setup.sh --hf-token X   # rented box only: deps, token, tokenizer, preflight
python scripts/run_training.py       # the real run: phase1 -> phase2, restarts on preemption
python scripts/pretrain.py --phase phase1   # one phase; resumes from the newest LOADABLE ckpt
python scripts/prepare_sft_data.py   # SFT corpus; needs manifest.json's holdout hashes first
python scripts/prepare_sft_data.py --profile repair          # Phase 2's ~50M-token repair corpus
python scripts/archive_corpus.py pack --all      # save data/prepared before overwriting it
python scripts/archive_corpus.py list            # measured counts vs. what the builder claimed
python scripts/sft.py --from-hub     # SFT: pull the pretrained ckpt + manifest, then train
python scripts/sft.py --repair -c ckpts/trained/checkpoint_sft_final_phase0.pt   # Phase 2
python scripts/migrate_phase0.py -c CKPT     # fold+strip a pre-Phase-0 checkpoint
python scripts/eval_calibration.py -c CKPT --start-doc-idx 0 --max-batches 40 --batch-size 4
python scripts/eval_abstention.py    # acceptance; -c CKPT --baseline-checkpoint PRETRAINED
python scripts/eval_abstention.py -c CKPT --max-examples 2000 --batch-size 16 --skip-forced \
  --example-offset 2000              # the disjoint slice, i.e. the eval sampling noise measurement
python scripts/eval_probe.py -c CKPT --json-out docs/measurements/probe_CKPT.json   # answerability
python scripts/eval_stage0.py -c CKPT --start-doc-idx 0 --max-batches 40 --batch-size 4 --max-loops 6
python scripts/eval_benchmarks.py --peer pythia-410m --validate \
  --json-out docs/measurements/benchmarks/pythia-410m.json   # a peer, measured once and frozen
python scripts/eval_benchmarks.py -c CKPT --compare docs/measurements/benchmarks/*.json
python scripts/inference.py          # interactive; -c CKPT -p PROMPT -n 200 --temperature 0.8
python scripts/gradio_app.py         # same generation path, browser UI
bash tests/run_env_check.sh          # torch/flash/TE/tokenizer smoke check
bash tests/run_tests.sh tests/test_attention_equiv.py tests/test_overfit.py
touch ckpts/training/STOP            # stop the run cleanly (exit 10, supervisor does not restart)
kill -USR1 <pid>                     # checkpoint now, keep training
```

Tests are plain scripts (no pytest): each `sys.path.insert`s the repo root, asserts, and prints.
Most require a GPU — the exceptions are the `modules/runtime/` tests (`test_checkpoint_lifecycle`,
`test_hf_sync`, `test_control`, `test_supervisor`, `test_phase_targets`, `test_hf_token`,
`test_checkpoint_atomic`), plus `test_prepare_data`, `test_sft_dataset`, `test_dataset_packing`
and `test_token_tracker`, which run anywhere. `tests/` is tracked — edits land in normal commits.

Operational detail (what to run on the box, how to stop it, what is normal, what to do when it is
not) lives in [docs/runbook.md](docs/runbook.md), not here.

## Config

`config.yaml` -> `config.py` exposes five surfaces:

- `ModelConfig.Params` — kwargs splatted straight into `TinyMoETransformer(**...)`.
- `ModelConfig.Forward` — kwargs splatted into `model(...)`; currently empty.
- `TrainingConfig` — class attributes; `total_steps` is *derived* as
  `target_tokens // (batch_size * seq_length * grad_accumulation_steps)`. Also holds
  `loop_ce_weights` + `loop_ce_subsample` and `loop_count_sampling`, even though they read from
  `config.yaml`'s `training:` block rather than `model:` — they're consumed directly in
  `scripts/pretrain.py`'s / `compute_mtp_loss`'s loss calc, not passed into the model.
  `loop_ce_weights`' length is asserted against `n_loops` at config-load time (import-time
  `assert` in `config.py`, not construction time). `data_dir` (default `data/prepared`) and
  `phase` (default `phase1`) pick which `{phase}.bin`/`{phase}.idx` pair the mmap `Dataset`
  reads — both gitignored artifacts produced by `scripts/prepare_data.py`.
- `SFTConfig` — the `sft:` block. Only what SFT genuinely does differently: its own
  lr/epochs/batch/seq, the `{train,val}_split` stems, the per-epoch shuffle `seed`, the eval and
  checkpoint cadences, `conversation_loss_weighting`, and a `dropout` override applied via
  `SFTConfig.model_params()` (dropout is not a parameter, so the pretrained state dict still loads
  unchanged). **Every loss weight is absent on purpose** — `scripts/sft.py` reuses
  `pretrain.train_step`, which reads them from `TrainingConfig`. Same `None`-vs-`""`
  `hf_upload_repo` distinction as `TrainingConfig`, but it defaults to `""` (uploads off) because
  SFT is a local run.
- `RepairConfig` — the `repair:` block, and a **subclass of `SFTConfig`**: only the keys it names
  are readable from that block, and everything else (batch size, seq length, dropout, warmup,
  `model_params()`) inherits by ordinary attribute lookup. That inheritance is the invariant — the
  repair pass is meant to be the SFT run with different data, so a second copy of those numbers is
  a second thing that can drift. It overrides `lr` (1e-5), `num_epochs` (1), the splits, the
  cadences, `grad_accumulation_steps` (2, purely to buy optimizer steps on a 14x smaller corpus)
  and `conversation_loss_weighting` (**true** — this is off in `SFTConfig` so that class still
  describes the run that produced the existing SFT checkpoint).

There is **no config key for the depth policy**. `converge_tol` / `min_loops` are inference-time
arguments to `TinyMoETransformer.forward`, not trained quantities — that is the point of them.

Constraints worth remembering:
- `moe_intermediate_size` (optional) sizes the routed MLP experts and the always-on shared
  MLP/attn; defaults to `intermediate_size` if omitted. `Gemma4TextModel`'s dense decoder always
  uses plain `intermediate_size`, so this is the only knob that moves total params without moving
  active (dense-decoder) params.
- `mtp_num_extra_tokens` must be <= the dataset's `num_mtp_tokens` separator budget, otherwise
  MTP gets supervised across document boundaries. Currently trivially true: `scripts/pretrain.py`
  passes the same value on both sides, and the separator is added dynamically at pack time rather
  than baked into the bin file. Revisit if that ever changes.
- `vocab_size` and `hidden_size` must both be divisible by `lm_head_factor` (SmallLMHead chunks
  both dims), and (if MTP is enabled) by `lm_head_factor * 2` for the MTP head's own `SmallLMHead`
  (which runs on `hidden_size // 2`). `vocab_size` must also be `<= 65536` (the corpus is uint16).
  **Asserted at model construction** — distinct from `loop_ce_weights`' config-load-time assert.
- Things *not* in the yaml but hardcoded: `NUM_DATA_WORKERS=4`, `LOG_INTERVAL=20` in `pretrain.py`
  and `10` in `sft.py`, expert head counts `n_heads=16 / n_kv_heads=4` and `rope_theta`
  ([moe.py](modules/model/moe.py)), `CE_CHUNK_SIZE=8192` ([mtp.py](modules/model/mtp.py)),
  `ROUTER_NOISE_SCALE=0.3` ([router.py](modules/model/router.py)).
- `TinyMoETransformer.__init__` prints total/active param counts and a forward FLOP/token estimate.
  Any budget math is keyed to that number and goes stale silently if it's not recomputed after a
  config change. "Active" excludes the routed MLP experts' unused capacity (`num_mlp_experts`
  weights exist but only `top_k` run per token); "excl. emb" further drops `embed_tokens` / the
  decoder's PLE table / this model's own PLE table (lookups, not matmuls).
- **FLOPs are exposed as three separately-scaled components, not one per-token constant** —
  folding them together is what made the pre-fix estimate understate real compute by ~2x:
  - `body_flops_per_token` — dense decoder + MoE matmuls. The MoE portion multiplies by `n_loops`
    (one shared module reused every loop: param count appears once, compute happens `n_loops`
    times); the decoder runs once.
  - `lm_head_flops_per_token` (**per application** — `lm_head` runs once *per loop* for per-loop
    CE, not once) and `mtp_flops_per_token`. Both are chunk-checkpointed inside `compute_mtp_loss`,
    so they cost fwd + recompute + bwd (**4x**) while the body costs 3x (activation checkpointing
    is off).
  - `attn_flops_per_seqsq` — attention scales with `sum(segment_len^2)`, not token count, so it
    can't be a per-token number under document packing at all. `scripts/pretrain.py` accumulates
    the real `sum(seg^2)` on-device from `cu_seqlens` and drains it at `LOG_INTERVAL`.
  `flops_per_token_fwd` is still published as a single log-line figure, assuming a fully-packed
  `max_seq_len` (worst case for the attention term).

## Model invariants

**Expert index layout** (one router over the whole pool, order matters everywhere):

```
[ SelfAttention x A | CrossAttention x A | IR x I | MLP x M ]
                                            ^ first_mlp_index = 2A + I
```

No identity expert. `LoopMixtureOfExperts._num_attn_experts` is `num_attn_experts * 2` (self +
cross). Indices `>= first_mlp_index` are remapped into `ParallelSparseMoELayer`'s local expert
space in `forward_step`; non-MLP slots become `(index 0, weight 0)` so they contribute nothing.

- **Non-MLP experts run unconditionally**, once per `forward_step`, and are cached across the
  top-k slots — attention has to see the whole sequence regardless of routing. Only the MLP
  experts are genuinely sparse (grouped GEMM over sorted assignments).
- **`shared_mlp` + `shared_attn` seed `forward_step`'s output accumulator unconditionally, every
  loop** — a dense SwiGLU MLP and a `SelfAttention` reused for its RoPE/varlen path, neither in the
  router pool (not in `Router`'s output dim, not in `compute_aux_loss`). Sized by
  `moe_intermediate_size`, not the dense decoder's `intermediate_size`. Static row count (`B*S`),
  so unlike `ParallelSparseMoELayer` they run inside the outer `te.autocast` — don't wrap them in
  `te.autocast(enabled=False)`.
- **`forward_step` returns an updated `hidden_states`, not a replacement**:
  `hidden_states = hidden_states + loop_scale[loop] * dropout(post_norm(output))`, giving a
  gradient path across loop boundaries independent of routing. `loop_scale` is an `nn.Parameter` of
  shape `[n_loops]` — **one gate per loop**, init `1/sqrt(n_loops)` (`post_norm` makes every delta
  unit-RMS, so at the old `0.1` the whole MoE block was a ~1.5% perturbation of a unit-RMS decoder
  output, and a lone scalar at `lr=4e-4` can't climb out of that inside a short run). Distinct from
  `layer_scalar` in the dense decoder (init-1 whole-layer gain). Loop indices past `n_loops - 1`
  reuse the last entry, so an inference-time loop-count override doesn't fall off the table.
  **Excluded from weight decay** by `build_param_groups` — decaying it is decaying the loop toward
  "off", and on a migrated checkpoint it is already small (see Phase 0 above).
- **Routing is conditioned on the loop index** so consecutive loops don't all pick the same
  experts: `route()` adds `loop_router_bias(loop_enc[loop])` to the router logits. The encoding is
  *sinusoidal in the absolute loop index* (a `[max_enc_loops, loop_enc_dim]` non-persistent buffer),
  deliberately not a learned `[n_loops, num_experts]` table — the loop count stays a runtime choice,
  so `forward(..., n_loops=N)` can run a trained checkpoint at any depth without reshaping a weight.
  `loop_router_bias` is zero-init, so routing at step 0 is identical to the unconditioned router.
- **Depth policy is the parameter-free convergence exit** (`TinyMoETransformer._convergence_exit`,
  reached via `forward(..., converge_tol=..., min_loops=...)`, plumbed into
  `LoopMixtureOfExperts.forward` as an `exit_check` callable). After each loop it reads out the
  **last position only** and stops when the top-1 token is unchanged *and* its log-probability
  moved less than `converge_tol`. Three things are load-bearing:
  - It reads the **readout**, not `‖Δh‖`. `loop_scale` still injects a sizeable hidden delta on the
    last loop while the prediction is already stationary, so a hidden-state criterion never fires.
  - **Asserted inference-only.** A short `hidden_states_all` would silently break per-loop CE
    (`loop_ce_weights` is length-checked against `n_loops`).
  - **Asserted mutually exclusive with `kv_cache`.** An exited loop appends no K/V for that token,
    so a later full-depth step would attend over a cache with a hole in it. `inference.py` resolves
    this by turning the cache off when `--converge-tol` is set. Filling those caches cheaply (K/V
    projections only, skipping the skipped loops' experts) is unimplemented plumbing through every
    attention expert — the honest option if this ever needs to be fast.
  `eval_calibration.py` prints per-transition top-1 agreement and mean `|Δ log p_top|`, which is how
  the threshold gets picked. Measured on the 16B checkpoints: loop 1→2 agreement ~0.81 / gap ~0.23,
  loop 2→3 ~0.93 / ~0.08.
- Selection is applied with a **mask multiply**, not `mask.sum()`/boolean indexing, deliberately:
  boolean indexing forces a device sync per expert per step.
- `ParallelSparseMoELayer.forward` runs its GEMMs under `te.autocast(enabled=False)` — NVFP4
  needs each group's row count divisible by 16, which dynamic routing can't guarantee. Sparsity
  wins over precision here. `m_splits` via `.tolist()` is a known, accepted host sync.
- `torch.argsort(..., stable=True)` in the same function is required for determinism between the
  checkpoint forward and the recompute pass.
- The aux (load-balancing) loss is computed directly on the router's softmax probabilities. One
  `torch.topk` feeds both it and the selection (score renormalization doesn't change indices). Note
  it is normalized by `loops_run`, not `n_loops`, so an early exit doesn't inflate it.
- The non-MLP experts' routing weights are folded into a single `[B, S, first_mlp_index]` gate via
  `scatter_add_` before being applied, rather than a `top_k × first_mlp_index` nested loop of
  `[B, S, H]` masked multiplies — same mask-multiply semantics, a fraction of the retained
  activations.
- **Router exploration noise is scaled by `router.noise_scale`** (`ROUTER_NOISE_SCALE = 0.3`). The
  learned `softplus(noise_proj(h))` lands near ~0.7 at init while the clean logits' std is ~0.33 —
  un-scaled, early routing/expert weights/aux loss all measure noise rather than the router. This
  is a ceiling on the *initial* level; `noise_factor` still anneals it to 0 over
  `noise_anneal_tokens`.
- `_ExpertTracking` guards against activation-checkpoint recompute double counting
  (`begin_forward(expected_updates)`) and only samples every 8th forward. Its stats are per-token
  EMAs in [0, 1], plotted to `ckpts/training/expert_selection.png`.
- **IR retrieval entropy is instrumented during training too**, by `RetrievalEntropyTracking`
  ([information_retrieval.py](modules/model/information_retrieval.py)) — same two guards as
  `_ExpertTracking` and for the same reasons, one tracker (`moe.ir_tracker`, `None` when
  `num_ir_experts == 0`) shared by every IR expert, one EMA slot per loop. It reads the softmax
  weights that already exist inside `InformationRetrievalModule.forward` under `no_grad`, upcast to
  fp32 in 1024-row chunks (67MB transient; a whole `[B*S, num_ir_entries]` fp32 copy would be
  ~0.5GB of peak) via `torch.special.entr`, which is `-x log x` in one kernel — the written-out
  `-(w * w.clamp_min(1e-12).log())` form measured 3x slower at `[16384, 8192]`. Cost is **1.5ms per
  (loop, IR expert) on every 8th forward**, ~0.5ms amortized against a ~400ms step, and
  **`update()` performs no host sync** (asserted with `torch.cuda.set_sync_debug_mode("error")`;
  `get_stats()` is the only sync and runs at log cadence). **Reported
  as `E / ln(num_ir_entries)`** — the same units `scripts/eval_stage0.py` prints, so a log line and
  a Stage 0 report compare directly; 1.0 means the read is uniform, i.e. the table stores nothing.
  `pretrain.py` and `sft.py` log it per loop at `LOG_INTERVAL` as `IR E/lnN: [...]` (`get_stats()`
  is the only host sync). `loop_idx` is plumbed into the IR expert *only* to bucket this — the
  retrieval itself is loop-independent, which is exactly what Stage 0's query-drift number found.

**Gradient checkpointing**: use `from transformer_engine.pytorch import checkpoint`, never
`torch.utils.checkpoint` — the latter breaks FP8/NVFP4 quantized layers. Two levels:
`set_checkpointing(stage_level, sub_level)` for the decoder/MoE stages and the per-loop /
per-MTP substages. Training currently runs with both **off**.

**Training-mode forward returns hidden states, not logits** (`return_hidden=True` plus
`delayed_mtp_loss(True)`): `compute_mtp_loss` applies the LM head inside a chunked, checkpointed
cross-entropy so `[T, vocab]` logits are never fully materialized. If you add a call site,
pass `main_lm_head=` or you'll silently double the activation peak.

**Forward return arity.** `return_aux_loss=True` yields `(x, aux_loss)`, plus
`extra_token_outputs` as a third element when MTP is on. It used to carry `p_halt` between the two
— any call site still unpacking four values is pre-Phase-0. **`skip_mtp=True` drops that third
element** *and* skips running the head: it is a pure function of the final loop's normed hidden
state, so not running it cannot move the logits (`tests/test_mtp_skip.py` asserts bit-identity, not
a tolerance), but leaving it on costs the whole prefix on every generated token. Every eval script
and the non-drafting inference path pass it; the drafting path and training do not.

**Per-loop CE supervision**: with `return_hidden=True` the returned "hidden states" are actually
`self.norm` applied at *every* loop, stacked to `[loops_run, B, S, H]` (index `[-1]` is the final
loop, what MTP reads and what `return_hidden=False` projects to logits). Without it, intermediate
loops are optimized only as inputs to the next loop, never as something `lm_head` can read, which
is exactly what an early-exit policy needs. `compute_mtp_loss` requires `loop_ce_weights` (one
entry per loop, length-checked against `n_loops` at config-load time) whenever `main_lm_head` is
set, and applies the chunked CE per loop — never materializing more than one loop's one chunk of
`[chunk, vocab]` logits at a time. `loss_ce` (returned for logging) is the *final* loop's raw,
unweighted CE. Ascending weights don't guarantee a strictly descending per-loop CE once training is
deep into an overfit regime: earlier loops receive gradient from every later loop's CE too (just
backprop through the recurrence), so their remaining headroom can let them read out a *lower* CE
than a later loop despite the smaller weight. `tests/test_per_loop_ce.py` samples early in training
for exactly this reason — see its comment before changing its step count.

**`p_max` is the confidence signal**, computed as `1 / sum_j exp(l_j - l_max)` rather than
`logits.float().softmax(-1).max(-1)` — mathematically identical, but avoids two ~2GB fp32
transients per chunk at `chunk=8192`/`vocab=65536`, allocated every step *and* again on the
checkpoint recompute. Anything that ever replaces it has to *add* information over it rather than
reproduce it (see Phase 0 above for why the last attempt didn't).

**Document packing**: the dataset emits batch-aligned `document_ids [B, S]`; the trainer converts
them to `cu_seqlens` **in-thread** via `cu_seqlens_from_doc_ids`. Never put `cu_seqlens` in the
batch dict — it's ragged (`dim0 = num_segments + 1`) and accelerate's `split_batches` truncates
dim 0 to the batch size, silently corrupting segmentation. `max_seqlen` is passed as `S` (a valid
upper bound) rather than the true max, to avoid an `.item()` sync every step.

**Token counting**: `TokenTracker` accumulates non-pad counts in an on-device scalar and only
drains to host on `sync()`, called at log/checkpoint cadence. Read `.get_count()` / `token_count`
for a sync-free (slightly stale) value; assign `.num_tokens` to restore on resume. Don't add a
`.item()` in the step path.

**KV cache** ([modules/model/kv_cache.py](modules/model/kv_cache.py)): one `LayerKVCache` per dense
decoder layer, plus one per `(loop, non-MLP expert)` pair *and* per `(loop, shared_attn)` — the MoE
block is the same weights re-applied `n_loops` times over an evolving hidden state, not `n_loops`
independent layers, so each loop needs its own cache. Every `forward()` in the package defaults
`kv_cache=None`, reproducing the exact packed/varlen training path; this module is additive and
never touches training. Single-sequence only (`cu_seqlens` must be None), which is why
`eval_abstention.py`'s batched left-padded decode does not use it.

**Tokenizer quirk**: with the DeepSeek tokenizer `pad_token_id == eos_token_id`, and id 0 is BOS.
`Gemma4TextModel.embed_tokens` therefore has **no `padding_idx`** — setting one froze BOS at zero.
Both invariants (and the byte-level round-trip guarantee) carry over unchanged into the pruned
65536-vocab tokenizer — they hold on the *old* id numbering trivially, and the prune's id remap
sorts kept old ids ascending before renumbering, so id 0 (the global minimum, always kept) lands on
new id 0 again and the single pad/eos id keeps whatever new id it's remapped to.

**Vocab prune** (`scripts/prune_vocab.py`): `ckpts/pretrained/DeepSeek-V4-Pro-tokenizer` (129280
tokens) is pruned to exactly 65536 so the corpus can be uint16 instead of uint32 — a disk
constraint, not a param-count one. **Every entry point reads one constant**, `utils.TOKENIZER_DIR`
(default `ckpts/pretrained/DeepSeek-V4-Pro-tokenizer-65536`, overridable with
`$TINY_LLM_TOKENIZER`), instead of six independent hardcoded paths — `ckpts/` is gitignored, so a
fresh clone on the box used to fail in six places with six different messages.
`scripts/fetch_tokenizer.py` downloads it from `utils.TOKENIZER_REPO`
(`ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536`, public, no token needed). `prune_vocab.py`'s own
`SRC_TOKENIZER_DIR` is deliberately *not* that constant — it points at the unpruned 129280-token
source, which is the prune's input, not the trained model's tokenizer. The prune script:
- Samples ~2GB of text from a **local stand-in** for the phase-1 mix, counts how often the
  **current** tokenizer emits each of its 128000 base BPE ids over that sample, and keeps the most
  frequent ones.
- Every special/added token (1283 ids) and the 256-entry byte-level alphabet are unconditionally
  required regardless of frequency — with `byte_fallback` disabled on this tokenizer, that
  alphabet is the only thing guaranteeing arbitrary text stays encodable after merges are dropped.
- Kept tokens are closed under BPE merge dependency (`select_kept_ids`/`closure`): a multi-piece
  token is only kept if both the pieces it was merged from are kept too, recursively down to the
  byte alphabet — "a kept merge never depends on a dropped one."
- The id remap sorts all kept *old* ids ascending and renumbers `0..65535`, which is why BOS
  staying at id 0 and pad==eos need no special-casing. Written alongside the new tokenizer as
  `id_remap.json`.
- Gated on **fertility** (tokens/byte), not round-trip identity — byte-level fallback makes
  round-trips pass almost regardless of prune damage; the real cost of a bad prune is common text
  costing more tokens. Measured on a held-out ~200MB sample via `manifest.json`'s `vocab_prune`
  key; last measured overall regression 0.15% (bound was 3%), worst single source (wiki) 0.37%.

## Data prep (`scripts/prepare_data.py`)

Builds `phase1.bin`/`.idx` and `phase2.bin`/`.idx` from the seven-source Hub mix, meant to run
**on the rented, interruptible, unattended box**, not locally — interruption safety is a design
requirement here, not an afterthought.

> **The `data/prepared/` corpus on the dev box is an outdated local stand-in**, not what the 16B
> run used. Absolute numbers measured against it are not comparable to
> [docs/CONCLUSION.md](docs/CONCLUSION.md); before/after deltas on the same slice are, which is all
> Gate P0 needs.

- **One shard file in flight per source at a time**: download -> tokenize -> append raw content
  ids to `{phase}.bin` + cumulative offsets to `{phase}.idx` -> delete the local shard. Peak disk
  is bounded by final bin size plus a handful of in-flight files, never the source corpora
  (terabytes for fineweb-edu/finepdfs-edu alone).
- **Sources are interleaved document-by-document via smooth weighted round-robin** (`run_phase`'s
  `swrr` dict), not written source-by-source then concatenated — `modules/data/dataset.py` reads
  sequentially with no shuffling, so the mix ratio has to already be baked into on-disk order.
  `run_phase`'s caller renormalizes each phase's active weights to sum to 1.0.
- **Checkpointed every `--checkpoint-docs` documents** (default 2000): `bin`/`idx` are `fsync`ed
  and a `_prepare_state_{phase}.json` sidecar records each source's `(file_idx, row_idx, tokens,
  done)`. On restart, `truncate_to_state` trims `bin`/`idx` back down to exactly what the sidecar
  last confirmed *before* reopening them for append — a crash between checkpoints can only redo
  up to one checkpoint interval, never desyncs the `bin`/`idx` pairing.
- **State is only advanced on a document actually committed, never at pick time** — `run_phase`
  buffers a `tokenize_batch`-sized group of picks, tokenizes them together, and only then commits
  each to `bin`/`idx` and advances that source's position. Advancing at pick time was an actual bug
  caught by `tests/test_prepare_data.py`: the commit loop can break mid-batch (target token count
  reached), and any already-"picked" documents past that point would never be written yet would be
  marked consumed, silently losing them forever on the next resume.
- **Two independent `run_phase` calls (fresh process, e.g. across a real interruption) do not
  reproduce the same interleave order** as one uninterrupted call — the SWRR fractional counters
  live only in memory. This is fine: each source's own document sequence is still exactly gap-free
  and repeat-free across the boundary (what `test_prepare_data.py` checks), and the acceptance bar
  is realized token counts within 2% of target, not byte-for-byte reproducibility.
- **Gated sources**: only `nvidia/Nemotron-CC-Math-v1` (`SourceSpec.gated=True`) needs `HF_TOKEN`
  set *and* the dataset's access request accepted first; `main()` and the per-file downloader both
  fail with that hint on a 401/403 instead of hanging. The code source
  (`common-pile/stackv2_edu_filtered`) replaced the originally-planned `nvidia/Nemotron-CC-Code-v1`
  — NVIDIA's gate on that one requires a recognized-org account and was rejecting
  independent-developer access requests outright. Common Pile's release is fully public: Stack-Edu's
  educational-quality code selection with the actual `text` re-materialized as gzipped JSONL,
  sidestepping the Software Heritage-ID reconstruction step plain `stack-edu` would require.
- **Text column is auto-detected at runtime**, not hardcoded, via `SourceSpec.text_columns`
  candidate tuples (`pick_text_column`) — several sources had schemas that couldn't be verified
  offline; failing loudly with the actual observed columns beats silently reading the wrong field
  for 25B tokens unattended.
- `smoltalk2`'s `messages` field is rendered as plain `"role: content"` turns (not a real chat
  template — that's SFT's job), and only its non-reasoning (`_no_think`) SFT splits are used. Every
  document actually consumed gets a `sha1(text)[:16]` recorded into `manifest.json`'s
  `data_prep.smoltalk2_holdout_hashes`, so SFT can exclude anything already seen in phase-2.
- Tokenization relies on the fast tokenizer's own Rust-side thread parallelism rather than a
  `ProcessPoolExecutor` — multiprocessing plus a loaded fast tokenizer is a known fork-deadlock
  risk, a bad trade for an unattended box with no one watching to restart it.

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
- Main CE is a weighted sum over `n_loops` per-loop CE terms (`TrainingConfig.loop_ce_weights`)
  rather than just the final loop's. The **non-final** loops are token-subsampled by
  `TrainingConfig.loop_ce_subsample` (default `0.25`); the final loop is always supervised in full.
  A CE mean over a uniform subsample is an unbiased estimate of the full mean, so `loop_ce_weights`
  semantics are unchanged — only the variance on the low-weight intermediate readouts goes up, in
  exchange for not running the model's largest GEMM `n_loops` times at full width.
- **Stochastic loop depth** (`TrainingConfig.loop_count_sampling`, default `0.3`): that fraction of
  steps runs a uniformly random depth in `1..n_loops-1` via `sample_n_loops`, the rest run full
  depth. `loop_ce_weights` is truncated and rescaled by `loop_ce_weights_for(n)` so the deepest loop
  actually run always carries weight `1.0` — truncating alone would shrink the whole CE term on
  shallow steps, i.e. a per-step LR change. Purpose: make every depth a real operating point so an
  inference-time `n_loops` override (or the convergence exit) lands on something the model trained
  at; it also cuts mean body compute (~85% of always-full at `p=0.3`, `n_loops=3`). **Log steps are
  pinned to full depth** so `losses` and per-loop CE are always read at the same operating point.
  This is the entire depth mechanism during training now that the halt head is gone.
- **Optimizer uses two param groups** (`build_param_groups`): weight decay applies only to tensors
  with `ndim >= 2`. Norms/biases/gates are excluded because their zero is a *degenerate state*, not
  just a regularization preference — `moe.loop_scale` decayed toward 0 is the loop decayed toward
  "off", and `layer_scalar` is a gain on the whole residual stream so its decay compounds across
  depth (~0.5x over 8 layers at `lr=4e-4`/`wd=0.02`, before any gradient).
- **A parameter stepped through an fp32 master is NOT in any optimizer param group, so
  `optimizer.zero_grad()` never clears its `.grad` — `train_step` has to, by hand, gated on
  `accelerator.sync_gradients` exactly like the step itself** (clearing every micro step would
  throw away grad accumulation for those tensors). That loop is load-bearing, not tidiness:
  without it the bf16 `.grad` accumulates across every optimizer step for the entire run, and (a)
  `accelerator.clip_grad_norm_(model.parameters(), ...)` reads those tensors, so the total norm
  grows without bound and the clip coefficient throttles *every other* parameter's gradient — a
  silent run-wide LR collapse behind a loss curve that still descends, just far too slowly — and
  (b) `sync_master_grads_` copies a run-length sum instead of the step's gradient, giving the
  shadowed params a permanent non-decaying momentum. This bites proportionally harder in
  `scripts/sft.py`, where *every* parameter is shadowed.
- **A checkpoint that exists but fails to load is never a fresh start.** The resume path
  distinguishes "no file in `ckpts/training`" (warn, start from scratch) from "files exist but
  none loads" (raise). Collapsing both into one warning is how a preempted box silently restarts
  from token 0 with a plausible-looking loss curve. `find_resume_checkpoint` softens only the
  middle case — one corrupt newest file falls back to the next oldest — which is why
  `verify_resume` exists to bound how far back that fallback may silently go.
- **`collect_metrics` is gated on the log cadence.** `train_step(..., collect_metrics=step %
  LOG_INTERVAL == 0)`; `metrics` is `None` otherwise. The `p_max`/top-1 reductions run over every
  CE chunk's live logits *and* again on the checkpoint recompute, so gathering them on the other 19
  steps in 20 is pure waste.
- Everything that needs the host (loss `.item()`, token sync, tokens/sec, peak mem) is throttled
  to `LOG_INTERVAL`. Keep it that way — the model is small enough that per-step syncs dominate.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, or `moe` must go through
  `accelerator.unwrap_model(model)`; the DDP wrapper has none of those attributes.
- **`KeyboardInterrupt`'s `input()` prompt is gated on `sys.stdin.isatty()`.** On a box with no tty
  it raises `EOFError`, which the old bare `except Exception` swallowed — so the interrupt path
  saved nothing. Worse, vast preemption sends SIGTERM, which never raised `KeyboardInterrupt` at
  all, so that path never ran on the one machine it was written for.
- **Training stops at the PHASE's token target**, not `target_tokens` and not an epoch count:
  `num_epochs: 1` just bounds the outer loop as a safety net. The real stop condition is
  `n_tokens >= phase_target`, checked inside the existing `LOG_INTERVAL` block (the counter is
  already being drained there, so this adds no extra sync). `target_tokens` stays the **combined**
  budget and still anchors `total_steps` and the cosine, so phase 2 continues the decay instead of
  restarting it.
- **`compute_mtp_loss(..., return_metrics=True)`** returns a 3rd value, a dict of still-on-device
  tensors: `per_loop_ce` (list, one per loop), `p_max`, `top1_acc`. Default `return_metrics=False`
  (2-tuple return) is unchanged. The metrics are cheap reductions over each chunk's
  *already-materialized* logits inside `_chunked_linear_ce`'s existing chunked/checkpointed loop —
  not a second forward pass — so requesting them doesn't add a real sync; only `.item()`-ing the
  dict's values (done once, inside the `LOG_INTERVAL` block) is the actual host sync.

## Checkpoints & resume

`ckpts/training/checkpoint_{phase}_tok{N}M_loss{L}.pt` for rolling saves, plus exactly one
`checkpoint_{phase}_final.pt` per phase (`modules/runtime/checkpoints.rolling_name`/`final_name`).
Epoch/step are deliberately gone from the name: `num_epochs` is a safety net now and `dataset_idx`
stopped being the resume key when `global_offset` landed, so the token count is the only figure
that says where a checkpoint sits in the run.

**"Latest" means newest mtime that actually LOADS**, not simply newest mtime —
`find_resume_checkpoint` walks the candidates newest-first, logs and skips one that raises, and
only raises itself when *every* candidate fails. Payload (see [utils.py](utils.py)):
model/optimizer/scheduler states, `token_count`, `losses`, `phase`, and a single
**`global_offset`** — a doc index into the flat, unshuffled document stream. There is no per-file
or per-worker state: doc sharding across `DataLoader` workers is pure `doc_idx % num_workers`
arithmetic, so one conservative (min-across-workers) scalar is enough to resume from without
skipping any worker's unconsumed documents. Workers further ahead than the minimum at checkpoint
time just redo a few already-seen documents on resume.

- `pretrain.py`'s `snapshot_global_offset()` computes this: each worker records the last `doc_idx`
  it reached (`batch["doc_idx"]`, host-synced only at checkpoint time), and the checkpoint stores
  `min(seen) + NUM_DATA_WORKERS` (the smallest "next document any worker still wants").
- Resume is document-granular, not sub-document: a document a worker was mid-way through packing
  is simply redone. `tests/test_dataset_resume.py` checks the guarantee the design actually makes.
- All `load_checkpoint` extras use `.get(..., default)` so old checkpoints still load. Keep that
  when adding fields, and add them to `save_checkpoint`'s signature with a default too.
  `load_checkpoint` returns a **6-tuple** (it was 7 before `ponder_state` was deleted), and raises
  a named error on a `migrate_phase0.py` output, which carries no optimizer state on purpose.
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
  `grad_accum 16` it fired every ~49M tokens — ~608 checkpoints, ~1.2TB, against a 120GB disk.

## Dataset ([modules/data/dataset.py](modules/data/dataset.py))

`IterableDataset` yielding **fully assembled batches** (hence `batch_size=None` on the DataLoader),
reading from a pre-tokenized flat-file corpus: `{data_dir}/{phase}.bin` (a flat uint16 token
stream) and `{data_dir}/{phase}.idx` (uint64 document-start offsets, one entry per document plus a
trailing entry == `len(bin)`).

Both files are opened via `np.memmap` **inside** `_batch_iterator` (once per worker/epoch, not
held on the `Dataset` object across its lifetime) — a long-lived memmap handed across DataLoader
worker restarts is a known leak vector.

Documents are read **once, in on-disk order, with no shuffling**: `prepare_data.py` already
interleaves sources at the target mix ratios while writing the bin file, so a straight sequential
read reproduces that mix — reshuffling here would undo it. Workers shard by pure
`doc_idx % num_workers == worker_id` arithmetic. `start_doc_idx` (from the checkpoint's
`global_offset`) is the one resume input.

Packing: documents are concatenated into `max_length` sequences, split across sequence boundaries
when they don't fit, each followed by `EOS + (num_mtp_tokens - 1)` pads. Trailing padding becomes
length-1 attention segments. Labels are `-100` everywhere except the interior of each document
block plus the terminating EOS. BOS is prepended if the document's first stored token isn't already
BOS — idempotent regardless, since the bin stores raw (BOS-less) content ids only.

Batches carry `doc_idx / worker_id` as `[B]`-shaped tensors purely so accelerate's batch splitting
treats them like `input_ids`.

## SFT / post-training

`scripts/sft.py` + `scripts/prepare_sft_data.py` + `modules/data/{chat,abstention,sft_dataset}.py`.
**Written for a local run** — the pretrained checkpoint and `manifest.json` come down from the Hub
(`sft.py --from-hub`, `prepare_sft_data.py --pull-manifest`; the manifest is gitignored so a fresh
clone never has the holdout hashes). It does run unattended on a rented box: same
`modules/runtime/control` contract as pretraining (SIGTERM → save + exit 20, STOP → exit 10,
SIGUSR1 → save and continue). There is deliberately **no phase supervisor** — there are no phases
here, only epochs, and epoch position is in the checkpoint. What to change from the local defaults
on a rented box lives in [docs/runbook.md](docs/runbook.md) §10.

- **`sft.py` reuses `pretrain.train_step` verbatim**, deliberately: the only way to guarantee the
  objective stays *identical* across the two runs (per-loop CE weights, aux loss, loop-count
  sampling) is to have one copy. Prompt masking needs nothing there: the dataset emits `-100`
  labels and every loss term — including the MTP heads, which read the same `labels` tensor —
  already honours `ignore_index=-100`. Consequence: every loss weight lives in `TrainingConfig`,
  not `SFTConfig`.
- **`sft(args)` runs two profiles, selected by `--repair`** (Phase 2 of
  [NEXT.md](docs/plans/NEXT.md)). It swaps three things and nothing else: the config class
  (`SFTConfig` / `RepairConfig`), the phase label (`"sft"` / `"repair"`) and the checkpoint
  directory (`ckpts/sft` / `ckpts/repair`). Distinct phase labels are load-bearing in both
  directions — `load_sft_checkpoint` takes the expected phase and refuses anything else, so neither
  run can adopt the other's AdamW moments or LR schedule, and the SFT checkpoint is passed to the
  repair run with `-c` (an *initializer*, via `load_pretrained_weights`, which drops optimizer
  state on purpose). The seed must be a **migrated** checkpoint (`*_phase0.pt`); a pre-Phase-0 one
  still has `halt_proj`/`correct_proj` in its state dict and won't load.
- **Per-conversation loss weighting** (`SFTConfig.conversation_loss_weighting`, off; `RepairConfig`,
  on). `SFTDataset` always emits `loss_weights [B, S]` = `1/(supervised tokens in this
  conversation)` on supervised positions and 0 elsewhere; the trainer decides whether to pass them,
  so one corpus serves both objectives and not passing them is bit-for-bit the old per-token mean.
  They flow `train_step -> compute_mtp_loss -> _chunked_linear_ce`, which turns every CE term into
  `sum(w*ce)/sum(w)`. Three details:
  - **Aligned with `targets`, not with the shifted labels.** Each term shifts them exactly as it
    shifts its labels (`[:, 1:]` for the main CE, `[:, shift:]` for MTP head *i*), and
    `_chunked_linear_ce` masks the denominator by `labels != -100` itself, so a stray weight on an
    ignored position can't inflate it.
  - **`p_max`/`top1_acc` stay unweighted**, always. They are compared across pretraining, SFT and
    `eval_calibration.py`; weighting them would redefine a reported number without renaming it.
  - **It changes what a source's mix weight buys.** `prepare_sft_data.py`'s weights are token-budget
    shares, but under conversation weighting a source's influence is its share of *conversations* —
    and at ~206 tokens for a SQuAD row against ~1,340 for a HotpotQA one those differ by 6.5x. The
    repair mix's 50/50 QA/chat token split is what produces its 72/28 conversation split; retune
    against the realized counts prep prints, never against the weights.
- **The model's global token counter is CONTINUED, not reset.** The router noise anneal is driven
  from it and has long finished at ~16B tokens. SFT progress is `token_count - start_token_count`,
  and `start_token_count` is why `sft.py` writes its own checkpoint payload (a strict *superset* of
  `utils.save_checkpoint`'s, so `inference.py` / `eval_calibration.py` read an SFT checkpoint
  unchanged) instead of extending `utils.py`.
- **Every parameter gets an fp32 master, not just the undecayed ones** (`build_sft_param_groups`).
  Pretraining shadows only `ndim <= 1` on the argument that 2D weights' values and steps both scale
  with their init std. That argument does not survive SFT's LR: at `lr=3e-5` a weight near its init
  std ~0.02-0.03 has a bf16 ulp of ~1e-4, three times *larger* than the ~`lr`-sized AdamW step, so
  `param -= lr * update` rounds to exactly the old value forever. At `4e-4` the same step is ~4x
  above the ulp, which is why the narrower fix was right there and wrong here.
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
  `<｜begin▁sys｜>` / `<｜end▁sys｜>`, which survived the vocab prune because `prune_vocab.py` keeps
  every special/added token unconditionally. They are spelled with explicit `｜`/`▁` escapes in
  `chat.py` — those are FULLWIDTH VERTICAL LINE and LOWER ONE EIGHTH BLOCK, visually identical to
  `|` and `_`, and a mistyped one resolves to a different or missing id. Only the assistant's own
  text and its terminating EOS are ever supervised.
- **Conversations with roles outside {system, user, assistant} are dropped whole**, not mangled
  into user turns — smoltalk2's `hermes_function_calling`/`xlam_traces` splits carry `tool` turns,
  which would teach tool syntax the model can never complete.
- **`prepare_sft_data.py` honours the smoltalk2 holdout** from `manifest.json`'s
  `data_prep.smoltalk2_holdout_hashes`. Those hashes are of `prepare_data.render_pretrain_chat`'s
  output, so the exclusion **imports that exact function** rather than reimplementing the
  rendering — a reimplementation that drifted by one character would silently exclude nothing. It
  refuses to run with an empty holdout list unless `--ignore-holdout` is passed.
- **Only train splits are consumed.** `squad_v2`'s validation split and `gsm8k`'s test split are
  the acceptance-metric eval sets; pulling them into SFT would make that number meaningless.
- **`scripts/eval_abstention.py` is the acceptance metric**, and it *generates* — `sft.py`'s
  `sft_val` pass reports `p_max`/top-1 at checkpoint cadence, which is an early warning, not the
  number. It decodes an answer for every `squad_v2` validation question and classifies it with
  `abstention.is_abstention`, reusing `SQUAD_INSTRUCTION` and `ChatTemplate.encode_prompt` by
  **import** rather than restating them: the instruction is what licenses abstention at all, and a
  prompt that drifted by a word would score the model on something it never saw. Also:
  - **Batched decode is left-padded, with the pad run given its own `document_ids` segment.** Left
    padding puts every row's last real token at the same index so one append extends all rows; the
    separate segment is what stops real tokens attending to pads. RoPE positions shift by the pad
    length, harmless because the score depends only on the relative offset inside a segment. No KV
    cache here (`kv_cache.py` is single-sequence), so cost is quadratic in answer length — hence
    `--max-new-tokens 32` and length-sorted batches.
  - **The isolation is exact through attention and only through attention.** Change what is in the
    pad region and the dense decoder's output for the real tokens is *bit-identical* —
    `tests/test_pad_isolation.py` asserts exactly that (with an unsegmented control, so it can't
    pass vacuously). Deliberately not asserted on the full model: `ParallelSparseMoELayer` tiles its
    grouped GEMM by `m_splits`, the per-expert row counts over *every* token in the batch, pads
    included, so batch composition changes the bf16 accumulation order for the real tokens' rows
    too (~0.5–1%). Same input twice is bit-identical — this is reduction order, not RNG — but eval
    numbers are only comparable at a fixed `--batch-size`.
  - **Two calibration numbers, deliberately.** Answer-level (`p_max` over the generated tokens vs.
    whether the answer was right) is the user-facing claim; token-level teacher-forced is the only
    one that can also be read off the *pretrained* checkpoint (`--baseline-checkpoint`). The
    pretrained model is out of distribution on the chat control tokens, so that baseline is
    conservative. **The token-level number passed while the behavioural one failed catastrophically
    on the real run** — read the generated-answers block first.
  - `expected_calibration_error`/`roc_auc` are imported from `scripts/eval_calibration.py` so both
    evals compute them with the same code. Don't fork them.
  - **`scripts/eval_probe.py` reuses this script's loader, renderer and slice by import**
    (`load_squad_split`, `build_records`, `_final_hidden`), which is what makes its AUROC comparable
    to the numbers above rather than a second measurement of a slightly different thing. Its finding
    is a fact about the model, not about a checkpoint: a linear probe of the final loop's
    last-position hidden state reads **0.584** for unanswerable detection on the pretrained trunk,
    the SFT checkpoint and the repair checkpoint alike (within 0.005), while every confidence scalar
    sits at chance. The finetunes re-read a fixed representation; they did not change it.
- **Abstention phrasings are a closed set** (`modules/data/abstention.py`). SQuAD v2's unanswerable
  rows have a literally empty reference answer, so a phrasing has to be supplied; keeping the set
  closed is what makes `is_abstention` an exact check rather than a classification problem. **The
  SFT run collapsed onto them** — 7,786 of 11,873 completions were literally `"The passage doesn't
  say."`, including on 78.4% of *answerable* questions; assume any abstention number off the
  pre-repair SFT checkpoint is measuring that collapse. **Phase 2 fixed that half**: `--repair`
  brings false abstention to 16.1% and answerable-half EM from 0.065 to 0.164, at the cost of
  recall falling 0.81 → 0.22 (the model now under-abstains). Retuning
  `--squad-unanswerable-fraction` 0.40 → 0.55 (now the default) moved recall 0.18 → 0.22 at flat
  precision ~0.578, which is the finding: the corpus ratio picks where on the curve the model sits
  and does not improve the curve, so the rest belongs to Phase 4. Numbers and caveats in
  [docs/measurements/abstention_repair.md](docs/measurements/abstention_repair.md). Phase 2 widened the set the corpus draws
  from to 15 phrasings (`ABSTENTIONS_PASSAGE_TRAIN`) while leaving `ABSTENTIONS_PASSAGE` (the
  original 5) as what `eval_abstention.py` *forces* as a reference target, so its teacher-forced CE
  stays comparable. **`is_abstention` matches the union**, and must: a detector that knew only the
  original five would miss abstentions worded the new way and report a lower false-abstention rate
  than the model earns — flattering the exact number Gate P2 checks.
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
  `STOP` sentinel at the `LOG_INTERVAL` cadence — ~3s granularity, inside vast's SIGTERM grace,
  and no GPU sync.
- **A stale `STOP` sentinel is cleared at startup** (`clear_sentinel()` before `install()`).
  Otherwise the previous run's stop file kills every relaunch before it trains a step.
- **Upload failures never propagate into the training loop.** `HFSync` retries 3x with tripling
  backoff, then logs and gives up on that file. Crashing a 40 hour run over a transient 503 is the
  worse outcome. The failed file stays *unmarked*, which is what makes retention refuse to delete
  it. `HFSync.delete()` follows the identical swallow-and-log policy.
- **A Hub file delete alone does not free storage; `HFSync` also squashes history.** `delete_file`
  only removes the blob from the current tree — git history still references it, so a 2GB
  checkpoint keeps costing storage until history is rewritten. Every successful delete calls
  `super_squash_history` too, throttled to at most once per `squash_min_interval` (default 1800s).
  Acceptable only because `temp-train` is a disposable scratch mirror — don't reuse this pattern
  against a repo anyone reads commit-by-commit.
- **`HFSync.drain()` waits for the in-flight job too, not just an empty queue** (hence `_busy`).
  It is called in `pretrain()`'s `finally`, so a stop or a phase transition cannot race the
  uploader.
- **`run_training.py`'s `main()` does not blindly start at `PHASE_ORDER[0]`.** It calls
  `checkpoints.resume_phase_index(checkpoint_dir)` first and starts there, because a vast.ai
  reclaim kills the whole supervisor and `onstart.sh` then launches a brand new `run_training.py`
  against a disk that can already hold a phase-2 checkpoint. The old unconditional loop re-entered
  phase 1, resumed the phase-2 checkpoint under `--phase phase1`, tripped the cross-phase reset,
  immediately hit phase 1's already-exceeded token target, and overwrote
  `checkpoint_phase1_final.pt` with phase-2 weights.
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
  The obvious-looking `hf_upload_repo or HF_UPLOAD_REPO` collapses the two and uploads anyway —
  caught by a local smoke run pushing a 2GB checkpoint to the real repo, not by a test.
- **Graphs are fixed filenames, overwritten** (`loss_graph.png`, `expert_selection.png`), written
  inside `save_and_sync` at checkpoint cadence.

## Inference

`scripts/inference.py` is the reference path; `scripts/gradio_app.py` imports `stream_generate`
from it rather than reimplementing, so the CLI and the UI cannot drift.

- **KV-cached by default** (`--no-kv-cache` for the slow full-prefix reference path). Exactness is
  structural, not approximate: every attention call in the model is causal, so a past token's
  output at a given depth never changes when later tokens are appended.
- **`--num-mtp-tokens` drafts self-speculatively** off the same step's final hidden state, greedily
  accepted with **no rejection sampling** against the main path. It trades quality for forward
  passes; 0 (off) is the default for a reason.
- **`--converge-tol` turns on the convergence exit and forces the KV cache off** (see the depth
  policy invariant). Pick the threshold from `eval_calibration.py`'s per-transition table.
- Streaming re-decodes the full generated id sequence each step and yields only the new suffix — a
  lone step's tokens can decode differently out of context (subword/space merges).

## Conventions

- Comments are lowercase, explanatory, and justify *why* (especially around sync avoidance,
  checkpoint recompute, and accelerate's batch handling). Match that density; don't strip them.
- Google-style docstrings with an `Args:` block on the public modules.
- Config values flow yaml -> `config.py` -> kwargs. Don't read `config.yaml` from a module under
  `modules/`.
- `utils.logger` (yellow-formatted) is the logging channel; scripts use `print` only in the
  inference CLI and the eval scripts' report blocks.

## Git

Current branch `ir-train-build`; PRs target `prototype`.

**Commit messages are a single line. No body, no bullets, no `Co-Authored-By` trailer** — the
trailer counts as a body and must be omitted even though the harness's default instructions ask for
it. Style is `feat:` / `docs:` / `chore:` / `merge:` plus a short description of the change itself.
No config/version labels, and prefer plain unhyphenated phrasing over compound modifiers —
"construction time assertions", not "construction-time assertions".

**Never name a plan document, phase, gate or step in a commit message or a code comment.** Not
"Phase 1b", not "Gate G1", not "NEXT.md", not "PLAN.md Step 5". Plans get rewritten, renumbered and
superseded; a comment that says *why the code is the way it is* keeps working afterwards and a
comment that says *which step asked for it* becomes a dangling reference to a document that no
longer says that. Write the reason instead — "the corpus builders delete shards from here", not
"1b.1 requires isolation". Docs under [docs/](docs/) are the exception: prose about the plan belongs
in `docs/plans/` and `docs/measurements/`, which is where the numbering is maintained. Some older
comments still carry `PLAN.md Step N` references; leave them alone unless you are editing that line
anyway, and drop the reference when you do.

Note the `.gitignore` swallows `*.json` (so `data_config.json` and `manifest.json` are untracked,
and so are the eval scripts' per-question and per-benchmark result files), `*.log` (eval run logs;
the numbers belong in `docs/measurements/`, not in a captured stdout), `*.cmd`, `*.key`, `ckpts/`,
`venv/`, `env_init`, `data/prepared*` and `data/benchmarks`. `tests/` is tracked.

## Known rough edges

- `flash-attn` and `transformer-engine` in `requirements.txt` need CUDA builds matched to the GPU;
  a plain `pip install -r requirements.txt` will usually fail on them.
- `huggingface.key` sits in the repo root (gitignored via `*.key`).
- The convergence exit cannot be combined with the KV cache (see the depth policy invariant) — the
  fix is real plumbing, not a flag.
- `eval_abstention.py` has no KV cache at all, so it is quadratic in answer length; use
  `--max-examples` for a first look.
- The depth policy's second criterion from [NEXT.md](docs/plans/NEXT.md) ("evidence still
  arriving") is **unimplemented** — the append-only evidence buffer it reads does not exist until
  Phase 4.
- README notes token counts can be inflated by tens of tokens per batch.

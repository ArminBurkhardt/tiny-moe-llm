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
utils.py                    logger, BASE_DIR, dtype aliases, save/load_checkpoint
data_config.json            local parquet source roots + column names per mode (gitignored) --
                             no longer read by any script since the Step 9 mmap dataset rewrite
                             (`prune_vocab.py` hardcodes its own mix instead); kept around as a
                             record of the pre-Step-9 local shard layout
memory-benchmark.py         standalone peak-VRAM probes (not used by training)
env_init                    WSL/CUDA env + venv activation (gitignored, `source env_init`)
scripts/
  pretrain.py               THE training loop
  inference.py              greedy/top-k sampling CLI
  prune_vocab.py            one-shot 129280 -> 65536 vocab prune (Step 8), not part of training
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
  loss knobs (`lambda_ponder`, `ponder_warmup_tokens`, `ponder_ramp_tokens`), `loop_ce_weights`
  (PLAN.md Step 4a), and `lambda_conf` (PLAN.md Step 4b) even though they read from
  `config.yaml`'s `training:` block rather than `model:` — they're consumed directly in
  `scripts/pretrain.py`'s / `compute_mtp_loss`'s loss calc, not passed into the model.
  `loop_ce_weights`' length is asserted against `n_loops` at config-load time (import-time
  `assert` in `config.py`, not construction time). `data_dir` (default `data/prepared`) and
  `phase` (default `phase1`) pick which `{phase}.bin`/`{phase}.idx` pair the mmap `Dataset`
  reads (PLAN.md Step 9) -- both are gitignored artifacts `scripts/prepare_data.py` (Step 11)
  hasn't been written yet to produce, so a real `pretrain()` run has nothing to read until then.

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
  matmuls). The FLOP estimate multiplies only the MoE block's active params by `n_loops` -- its
  weights are one shared module reused every loop, so the param count appears once but the compute
  happens `n_loops` times; the dense decoder and the heads (`lm_head`/`mtp_head`/`correct_proj`)
  run once regardless and aren't multiplied.

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
(`scripts/eval_calibration.py`, not written yet) compares its ECE/abstention-AUROC against the
free `p_max = softmax(logits).max()` baseline, and Step 4b reverts (head, loss term, `lambda_conf`)
if `p_correct` doesn't beat `p_max` on both.

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
be uint16 instead of uint32 — a disk constraint, not a param-count one. `scripts/pretrain.py` and
`scripts/inference.py` both point at the pruned `ckpts/pretrained/DeepSeek-V4-Pro-tokenizer-65536`
by default. The prune script:
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
- Main CE is now a weighted sum over `n_loops` per-loop CE terms (`TrainingConfig.loop_ce_weights`,
  PLAN.md Step 4a) rather than just the final loop's — see the per-loop CE invariant above.
- The correctness head's BCE loss (`TrainingConfig.lambda_conf`, PLAN.md Step 4b) is added
  unconditionally, no warmup ramp — see the correctness-head invariant above.
- Everything that needs the host (loss `.item()`, token sync, tokens/sec, peak mem) is throttled
  to `LOG_INTERVAL`. Keep it that way — the model is small enough that per-step syncs dominate.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, or `moe` must go through
  `accelerator.unwrap_model(model)`; the DDP wrapper has none of those attributes.
- `KeyboardInterrupt` prompts on stdin before saving an `*_interrupted.pt`.
- **Training stops at `target_tokens`, not a fixed epoch count** (PLAN.md Step 6): `num_epochs: 1`
  in `config.yaml` now just bounds the outer loop as a safety net; the real stop condition is
  `n_tokens >= TrainingConfig.target_tokens`, checked inside the existing `LOG_INTERVAL` block (the
  sync-free counter is already being drained there for logging, so this adds no extra sync). On
  hitting it, a final checkpoint is saved (same `save_checkpoint` call as the periodic one) before
  breaking both loops via a `stop_training` flag. Without this, a dataset bigger than `target_tokens`
  would keep training past the point the cosine LR schedule was anchored to, wasting the phase-2
  anneal (see PLAN.md's "Budget decision").
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

`ckpts/training/checkpoint_epoch{E}_idx{STEP}_loss{L}[_interrupted].pt`; "latest" means newest
mtime, not highest step. Payload (see [utils.py](utils.py)): model/optimizer/scheduler states,
`token_count`, `losses`, and a single **`global_offset`** (PLAN.md Step 9) -- a doc index into the
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

## Dataset ([modules/data/dataset.py](modules/data/dataset.py))

`IterableDataset` yielding **fully assembled batches** (hence `batch_size=None` on the DataLoader),
reading from a pre-tokenized flat-file corpus (PLAN.md Step 9): `{data_dir}/{phase}.bin` (a flat
uint16 token stream) and `{data_dir}/{phase}.idx` (uint64 document-start offsets, one entry per
document plus a trailing entry == `len(bin)`). Both files are produced by `scripts/prepare_data.py`
(PLAN.md Step 11, not written yet) -- `phase1`/`phase2` are the two mix ratios described there.

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
prepended if the document's first stored token isn't already BOS -- idempotent whether or not a
future `prepare_data.py` bakes one in, since `train.bin` stores raw (BOS-less, in the current
implementation) content ids only.

Batches carry `doc_idx / worker_id` as `[B]`-shaped tensors purely so accelerate's batch splitting
treats them like `input_ids`; `doc_idx` is the last global document index this worker had reached
when the batch was assembled, read by the trainer for checkpointing (see "Checkpoints & resume").

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

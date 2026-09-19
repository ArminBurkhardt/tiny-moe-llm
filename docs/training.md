# Training

Two entry points share one `train_step`:

- **Pretraining**: [scripts/pretrain.py](../scripts/pretrain.py), normally launched through the
  supervisor [scripts/run_training.py](../scripts/run_training.py). Rented box, two phases,
  preemption-safe.
- **Post-training**: [scripts/sft.py](../scripts/sft.py), local, single GPU, BF16, four profiles
  (`sft`, `--repair`, `--ir`, `--evidence`) selected by flag. It reuses `pretrain.train_step`
  **verbatim**, so every loss term is identical across all five runs by construction.

Both use HuggingFace `accelerate` for device placement and gradient accumulation. For the
operational side of a real run — starting, stopping, monitoring, recovering — see
[runbook.md](runbook.md). This document covers what the code does.

## Pretraining pipeline

1. Load the tokenizer from `utils.TOKENIZER_DIR` and build the mmap `Dataset` for the selected phase
2. Build `TinyMoETransformer` from `ModelConfig.Params`, cast to BF16, `.train()`
3. Optimizer `AdamW` over **two param groups**: weight decay applies only to tensors with
   `ndim >= 2` — norms, biases and gates are excluded because their zero is a degenerate state
   (`loop_scale` decayed to 0 is the loop switched off). The undecayed group is stepped through
   fp32 masters. LR schedule = **linear warmup → cosine decay** to `0.1 * lr`
4. Clean up stale files, then resume from the newest checkpoint that actually **loads**
5. Verify the resume against `run_state.json`, aborting with exit 30 if this process came back
   materially behind where the last one got to
6. **Dry run** — one synthetic packed batch forward+backward to fail fast on shape/precision
   issues. Its tokens are excluded from the counter and a non-finite loss aborts startup
7. Train loop: for each batch, build `cu_seqlens` from `document_ids`, anneal router noise, sample
   this step's loop depth, run `train_step`; at the log cadence, log throughput, write
   `status.json`, poll for stop requests, and checkpoint on the token cadence

Everything that needs the host — loss `.item()`, the token sync, tokens/sec, peak memory, the
metrics dict — is throttled to `LOG_INTERVAL`. The model is small enough that per-step syncs
dominate, so nothing in the step path may add one.

## Data & document packing

`Dataset` ([modules/data/dataset.py](../modules/data/dataset.py)) is an `IterableDataset` reading a
pre-tokenized flat-file corpus built by [scripts/prepare_data.py](../scripts/prepare_data.py):

- `{data_dir}/{phase}.bin` — a flat `uint16` token stream (hence `vocab_size <= 65536`)
- `{data_dir}/{phase}.idx` — `uint64` document-start offsets, one per document plus a trailing
  entry equal to `len(bin)`

Documents are read **once, in on-disk order, with no shuffling**. `prepare_data.py` already
interleaves the seven sources at the target mix ratios while writing, so a sequential read
reproduces that mix — reshuffling here would undo it. Both files are `np.memmap`ed inside the
worker iterator rather than held on the `Dataset` across worker restarts.

Workers shard the stream by pure `doc_idx % num_workers == worker_id` arithmetic, which is why a
single `global_offset` scalar is enough to resume: each worker derives its own first owned index
from it.

- **Framing**: BOS is prepended if the document's first stored token is not already BOS. Each
  document ends with a supervised EOS so the model learns to terminate, followed by
  `num_mtp_tokens - 1` unsupervised pad separators. `num_mtp_tokens` must be **>= the model's MTP
  head count** so MTP is never supervised across a document boundary
- Each batch emits `input_ids`, a `[B, S]` `document_ids` segment map, `labels` (non-predicted
  positions set to `-100`), and `doc_idx`/`worker_id` as `[B]`-shaped tensors so accelerate's batch
  splitting treats them like `input_ids`
- The dataset yields **fully assembled batches**, hence `batch_size=None` on the `DataLoader`
- `cu_seqlens` is built in-thread by the trainer and never carried in the batch dict — it is ragged
  (`dim0 = num_segments + 1`) and accelerate's `split_batches` would truncate it, silently
  corrupting the attention segmentation

### The finetune datasets

`SFTDataset` ([modules/data/sft_dataset.py](../modules/data/sft_dataset.py)) differs in exactly
three forced ways: a third file `{split}.mask` (1 = supervised) because prompt and completion
interleave inside a conversation; conversations are **never split across rows** (over-long ones are
dropped); and documents are shuffled per epoch by a `(seed, epoch)` permutation, so
`global_offset` is a position in that permutation. It always emits `loss_weights [B, S]` =
`1 / (supervised tokens in the conversation)`; the trainer decides whether to pass them.

`EvidenceDataset` ([modules/data/evidence_dataset.py](../modules/data/evidence_dataset.py)) is
`SFTDataset` plus a second token stream per row, from the seven extra files
`prepare_evidence_data.py` writes (`.ev`, `.evidx`, `.evchunk`, `.evkey`, `.evkeyidx`, `.evgold`,
`.cond`). Each batch additionally carries `evidence_ids` / `evidence_chunk_ids` /
`evidence_doc_slot` `[B, S_ev]`, `chunk_keys [B, C, 384]`, `chunk_slot` / `chunk_gold [B, C]`, and
`condition_ids [B, S]`; the trainer flattens them in-thread (`evidence_from_batch`) into an
`EvidenceBatch` once it has the query side's segmentation. One evidence segment per query segment,
paired by position; a document that retrieved nothing contributes a zero-length one. Rows pack to
`max_length - 1` so the evidence padding always has a trailing pad segment to belong to. A row's
evidence is capped by `max_evidence_tokens`, which must be set against the corpus's
evidence-to-prompt ratio — below it, evidence closes every row early and the run pays a full-width
forward for mostly padding, visible only as throughput. The dataset logs realized row fill and how
many rows the evidence budget closed. The per-conversation weight is floored at
`loss_weight_floor_tokens`.

## Phases

Pretraining runs in two phases over two different corpora and mix ratios:

| | phase 1 | phase 2 |
|---|---|---|
| corpus | `phase1.{bin,idx}`, ~25.5B tokens | `phase2.{bin,idx}`, ~4.5B tokens |
| stops at | `phase1_fraction * target_tokens` | `target_tokens` |
| role | bulk pretraining | anneal on a higher-quality mix |

`target_tokens` is the **combined** budget and `token_count` carries across the boundary, so phase
2 continues the same cosine decay rather than restarting it. Crossing phases resets the document
offset, epoch and step to zero (`resolve_resume_scope`) — a phase-1 offset of ~23M documents fed
into phase 2's ~4M-document corpus makes every worker's range empty, and the run would exit looking
successful having trained nothing.

Either "reached the token target" or "the corpus ran out" ends a phase; both write
`checkpoint_{phase}_final.pt` and exit 0.

## Loss - `compute_mtp_loss`

[modules/model/mtp.py](../modules/model/mtp.py). Total loss:

```
loss = Σ_loops w_loop · CE_loop(next token)          # loop_ce_weights, non-final loops subsampled
     + lambda_mtp · Σ_i CE(token at offset i+2)      # MTP heads, final loop only
     + aux_loss_weight · load_balance_loss           # normalized by loops actually run
```

- **Main CE**: the trainer receives hidden states for every loop (`return_hidden=True`) and applies
  the LM head inside a chunked, checkpointed cross-entropy (`CE_CHUNK_SIZE = 8192`) so
  `[T, vocab]` logits are never fully materialized. `loss_ce` (logged) is the final loop's raw CE.
  With `loss_weights` given, every CE term becomes `sum(w · ce) / sum(w)`; `p_max` and top-1 stay
  unweighted so they remain comparable across every run.
- **MTP CE**: each extra head `i` predicts the token at offset `i + 2`; pad positions are masked to
  `-100`. `lambda_mtp: 0.0` does **not** skip the compute — pass `mtp_outputs=None` for that.
- **Aux loss**: the MoE load-balancing term (see [moe.md](moe.md)).
- **Stochastic depth**: `loop_count_sampling` runs 30% of steps at a random reduced depth with
  `loop_ce_weights` truncated and rescaled so the deepest loop run carries weight 1.0; log steps are
  pinned to full depth so every logged number is read at one operating point.
- `compute_mtp_loss(..., return_metrics=True)` also returns still-on-device `per_loop_ce`, `p_max`,
  `top1_acc`, computed on the chunks already materialized; only `.item()`-ing them at log cadence
  is a host sync.

Two more losses exist for the evidence profile and are **not yet wired in**:
`information_retrieval.evidence_selection_loss` and `evidence.groundedness_loss`. See
[NEXT.md](plans/NEXT.md) Phase 4.

## The finetune profiles

`sft(args)` swaps three things per profile and nothing else: the config class, the phase label
(`sft` / `repair` / `ir` / `evidence`, which `load_sft_checkpoint` checks so no run can adopt
another's optimizer state) and the checkpoint directory (`ckpts/sft`, `ckpts/repair`, `ckpts/ir`,
`ckpts/evidence`; `--run-name` suffixes it so two seeds of one profile do not resume each other). A
seed checkpoint passed with `-c` is an *initializer* (`load_pretrained_weights`, optimizer state
dropped); shape-bearing model params are read off the seed's state dict.

What is genuinely different from pretraining:

- **fp32 master weights for every parameter** (`build_sft_param_groups`), not only the undecayed
  ones. At `lr = 3e-5` an AdamW step on a bf16 2D weight near its init std is below that weight's
  own ulp, so `param -= lr · update` rounds to the old value forever. A parameter stepped through
  a master is in no optimizer group, so `train_step` clears its `.grad` by hand on sync steps —
  otherwise the bf16 grads accumulate for the whole run, the clip norm grows without bound and
  every other parameter's LR silently collapses.
- **A fresh-parameter LR group** (`fresh_lr`) for the `--ir` and `--evidence` profiles, with a
  per-profile predicate: `--ir` matches the rebuilt `ir_module` subtree plus the port tensors;
  `--evidence` matches `moe.is_fresh_loop_param` alone, because its table carries a full sharpening
  run and would be wrecked at the from-scratch rate. The group's membership is logged by name.
- **The IR temperature anneal** (`--ir` only) on micro-step progress, and **the cluster refresh** in
  any profile whose config sets `cluster_refresh_tokens` (`--ir` and `--evidence`), logged as its
  own line with the measured candidate recall.
- **A validation pass** at `eval_every_tokens` reporting CE, `p_max` and top-1 on the val split at
  full depth with subsampling off; with evidence attached when the split carries it, and **per
  condition** when the corpus carries `.cond` — the gold-vs-none CE gap is the early-kill number.
- **Per-loss-and-gate instrumentation** at every log step: `|g_proj|rms` and `|y|row` (`--ir`),
  `|inject|rms` (when the injection exists), `|shared_evidence.o_proj|rms` and the external mass per
  condition (`--evidence`), the evidence token count as its own field, and `IR E/ln32` per loop.
- The model's global token counter is **continued**, not reset (the router noise anneal reads it
  and finished at ~1B tokens); profile progress is `token_count - start_token_count`, and the
  checkpoint payload is a strict superset of pretraining's.
- `estimate_packed_rows` replays the packing rule over the index (and the evidence index, when
  present) to anchor the cosine; "corpus tokens / (batch × seq)" is not a usable estimate under
  no-split packing.

## Precision (Transformer Engine)

Modules are built on TE `Linear` / `RMSNorm` / `GroupedLinear`, enabling FP8 / MXFP8 / NVFP4 via
`te.autocast`. Recipes are defined at the top of `pretrain.py` and selected by the `USE_FP8`
environment variable; **default is off** (BF16 everywhere), and every local finetune stays off.
NVFP4 cannot be used for the sparse MLP experts (row counts not divisible by 16). **Gradient
checkpointing uses TE's `checkpoint`, not `torch.utils.checkpoint`** — required for correct
FP8/NVFP4 recompute. Toggle with `model.set_checkpointing(stage, sub_stage)`; training runs with
both off.

## Token counting

`TokenTracker` counts real (non-padding) tokens once `pad_token_id` is set, accumulating into an
on-device scalar that is only drained to the host at the log/checkpoint cadence. `target_tokens`
drives the total-step count and the cosine schedule length; router-noise annealing decays over
`noise_anneal_tokens`. Because `pad_token_id == eos_token_id`, each packed document counts its
content plus the prepended BOS but **not** its terminating EOS or pad separators.

## Checkpointing & resume

Checkpoints save model/optimizer/scheduler state plus `epoch`, `token_count`, `global_offset`,
`phase`, and the loss history ([utils.py](../utils.py)); the SFT payload adds `start_token_count`
and the shuffle seed.

- **Cadence is in tokens** (`checkpoint_every_tokens`), checked inside the existing log block so it
  costs no extra host sync. `SIGUSR1` forces one immediately.
- **Writes are atomic**: write to `.pt.tmp`, `fsync`, then `os.replace`. A preemption mid-write
  cannot leave a truncated file that is also the newest by mtime.
- **Naming is token-keyed**: `checkpoint_{phase}_tok{N}M_loss{L}.pt`, plus one
  `checkpoint_{phase}_final.pt` per phase that is never pruned.
- **Retention** keeps the newest `keep_local_checkpoints` and deletes the rest **only** once their
  upload is confirmed. Sustained upload failure therefore fills the disk rather than discarding
  history — the loud, recoverable failure.
- **Resume picks the newest checkpoint that loads**, not simply the newest file. A corrupt newest
  file is logged and skipped; only if *every* candidate fails does startup raise. "A checkpoint
  exists but will not load" must never degrade into "start from token 0".
- **The LR schedule is re-anchored by token count**, not by saved step, so resuming after a batch
  size or grad-accumulation change still lands on the right point of the cosine.
- **`run_state.json`** records `{phase, token_count, checkpoint}` at every save. On startup the
  resumed token count is compared against it, and a gap larger than `2 * checkpoint_every_tokens`
  aborts with exit 30 rather than silently retraining ground already covered.
- **Migrated checkpoints are seeds, not resume points.** Every `migrate_*.py` output drops its
  optimizer state (Adam moments are indexed by param-group position and the migration adds or
  removes tensors); `utils.load_checkpoint` names that case when you try. Loading goes through
  `utils.load_model_state`, which is strict except for a named set of neutrally-initialized tensors
  (`IR_TEMPERATURE_KEYS`, `NEUTRAL_LOOP_KEYS`) whose absence means "the old value", never "state was
  lost" — the two resume paths stay fully strict.
- The fp32 masters are optimizer-only shadows built before the resume and reseeded from the
  just-loaded BF16 weights; sub-ulp progress since the last save is lost, bounded by one interval.

## Stopping

`modules/runtime/control.py` maps stop requests to an exit-code contract the supervisor reads:
`0` phase complete, `10` user stop, `20` preempted, `30` resume verification failed. Signal
handlers only set flags; the flags and the `STOP` sentinel file are read at the log cadence.
`input()` on `KeyboardInterrupt` is gated on `sys.stdin.isatty()` — on a box with no tty it would
raise `EOFError` and save nothing. `sft.py` honours the same contract with no phase supervisor.
See [runbook.md](runbook.md) §4.

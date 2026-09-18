# Training

Entry point: [scripts/pretrain.py](../scripts/pretrain.py), normally launched through
[scripts/run_training.py](../scripts/run_training.py). Uses HuggingFace `accelerate` for device
placement, gradient accumulation, and (optional) distributed training.

For the operational side of a real run — starting, stopping, monitoring, recovering — see
[runbook.md](runbook.md). This document covers what the code does.

## Pipeline

1. Load the tokenizer from `utils.TOKENIZER_DIR` and build the mmap `Dataset` for the selected phase
2. Build `TinyMoETransformer` from `ModelConfig.Params`, cast to BF16, `.train()`
3. Optimizer `AdamW` over **two param groups**: weight decay applies only to tensors with
   `ndim >= 2`. LR schedule = **linear warmup -> cosine decay** to `0.1 * lr`
4. Clean up stale files, then resume from the newest checkpoint that actually **loads**
5. Verify the resume against `run_state.json`, aborting with exit 30 if this process came back
   materially behind where the last one got to
6. **Dry run** — one synthetic packed batch forward+backward to fail fast on shape/precision
   issues. Its tokens are excluded from the counter and a non-finite loss aborts startup
7. Train loop: for each batch, build `cu_seqlens` from `document_ids`, anneal router noise, sample
   this step's loop depth, run `train_step`; at the log cadence, log throughput, write
   `status.json`, poll for stop requests, and checkpoint on the token cadence

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

- **Framing**: BOS is prepended if the document's first stored token is not already BOS (the bin
  file stores raw content ids, no BOS baked in). Each document ends with a supervised EOS so the
  model learns to terminate, followed by `num_mtp_tokens - 1` unsupervised pad separators.
  `num_mtp_tokens` must be **>= the model's MTP head count** so MTP is never supervised across a
  document boundary
- Each batch emits `input_ids`, a `[B, S]` `document_ids` segment map, `labels` (non-predicted
  positions set to `-100`), and `doc_idx`/`worker_id` as `[B]`-shaped tensors so accelerate's batch
  splitting treats them like `input_ids`
- The dataset yields **fully assembled batches**, hence `batch_size=None` on the `DataLoader`
- `cu_seqlens` is built in-thread by the trainer and never carried in the batch dict — it is ragged
  (`dim0 = num_segments + 1`) and accelerate's `split_batches` would truncate it, silently
  corrupting the attention segmentation

## Phases

Training runs in two phases over two different corpora and mix ratios:

| | phase 1 | phase 2 |
|---|---|---|
| corpus | `phase1.{bin,idx}`, ~25.5B tokens | `phase2.{bin,idx}`, ~4.5B tokens |
| stops at | `phase1_fraction * target_tokens` = 25.415B | `target_tokens` = 29.9B |
| role | bulk pretraining | anneal on a higher-quality mix |

`target_tokens` is the **combined** budget and `token_count` carries across the boundary, so phase
2 continues the same cosine decay rather than restarting it. Crossing phases resets the document
offset, epoch and step to zero (`resolve_resume_scope`) — a phase-1 offset of ~23M documents fed
into phase 2's ~4M-document corpus makes every worker's range empty, and the run would exit looking
successful having trained nothing.

Either "reached the token target" or "the corpus ran out" ends a phase; both write
`checkpoint_{phase}_final.pt` and exit 0. Phase 1's corpus is only ~0.3% larger than its token
target, so which one fires first is genuinely not knowable in advance.

## Loss - `compute_mtp_loss`

[modules/model/mtp.py](../modules/model/mtp.py). Total loss:

```
loss = CE(next_token) + lambda_mtp * sum_i(CE(token at offset i+2)) + aux_loss_weight * load_balance_loss
```

- **Main CE**: standard next-token prediction. To save memory the trainer returns hidden states
  (`return_hidden=True`) and applies the LM head inside the loss on a flattened, low-fp padded
  tensor
- **MTP CE**: each extra head `i` predicts the token at offset `i + 2`, its LM head is applied to
  the corresponding hidden slice. Pad positions are masked to `-100`. `_safe_cross_entropy` returns
  a graph-connected zero (not NaN) when a slice is all `-100`
- **Aux loss**: the MoE load-balancing term (see [moe.md](moe.md))

## Precision (Transformer Engine)

Modules are built on TE `Linear` / `RMSNorm` / `GroupedLinear`, enabling FP8 / MXFP8 / NVFP4 via
`te.autocast`. Recipes are defined at the top of `pretrain.py`:

- `USE_LOW_PRECISION` gates the autocast. **Default is off** (BF16 everywhere)
- NVFP4 is finicky on consumer Blackwell (needs stochastic rounding + RHT disabled) and cannot be
  used for the router/MTP heads (divisibility constraints on the backward pass) or the sparse MLP
  experts (see [moe.md](moe.md)). So thats a nono for me :(
- **Gradient checkpointing uses TEs `checkpoint`, not `torch.utils.checkpoint`**. Thats required for
  correct FP8/NVFP4 recompute. Toggle with `model.set_checkpointing(stage, sub_stage)`

## Token counting

`TokenTracker` counts real (non-padding) tokens once `pad_token_id` is set, accumulating into an
on-device scalar that is only drained to the host at the log/checkpoint cadence — reading it every
forward would serialize CPU/GPU. `target_tokens` drives the total-step count and the cosine
schedule length; router-noise annealing decays over `noise_anneal_tokens`.

Because `pad_token_id == eos_token_id` for this tokenizer, each packed document counts its content
plus the prepended BOS but **not** its terminating EOS or pad separators.

## Checkpointing & resume

Checkpoints save model/optimizer/scheduler state plus `epoch`, `dataset_idx`, `token_count`,
`global_offset`, `phase`, and the loss history ([utils.py](../utils.py)).

- **Cadence is in tokens** (`checkpoint_every_tokens`, default 400M), checked inside the existing
  log block so it costs no extra host sync. `SIGUSR1` forces one immediately.
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
- The fp32 master weights for the no-decay group are optimizer-only shadows built before the
  resume, so they are reseeded from the just-loaded BF16 weights.

## Stopping

`modules/runtime/control.py` maps stop requests to an exit-code contract the supervisor reads:
`0` phase complete, `10` user stop, `20` preempted, `30` resume verification failed. Signal
handlers only set flags; the flags and the `STOP` sentinel file are read at the log cadence.
`input()` on `KeyboardInterrupt` is gated on `sys.stdin.isatty()` — on a box with no tty it would
raise `EOFError` and save nothing. See [runbook.md](runbook.md) §4.

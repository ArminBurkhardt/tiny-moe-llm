# Training

Entry point: [scripts/pretrain.py](../scripts/pretrain.py). Uses HuggingFace `accelerate` for
device placement, gradient accumulation, and (optional) distributed training.

## Pipeline

1. Load tokenizer from `ckpts/pretrained/<tokenizer>` and build the streaming `Dataset`
2. Build `TinyMoETransformer` from `ModelConfig.Params`, cast to BF16, `.train()`
3. Optimizer `AdamW`: LR schedule = **linear warmup -> cosine decay** to `0.1 * lr`
4. Resume from the newest `ckpts/training/checkpoint_*.pt` if present
5. **Dry run** - one synthetic packed batch forward+backward to fail fast on shape/precision
   issues. Its tokens are excluded from the counter and a non-finite loss aborts startup
6. Train loop: for each batch, build `cu_seqlens` from `document_ids`, anneal router noise, run
   `train_step`, log throughput, periodically save graphs and checkpoints

## Data & document packing

`Dataset` ([modules/data/dataset.py](../modules/data/dataset.py)) is an `IterableDataset` that
streams parquet/jsonl/json files and packs tokenized documents densely into `max_length`
sequences (a document that overflows continues in the next sequence as its own segment).

Data sources come from a `data_config.json` (path passed in `pretrain()`), keyed by mode:

```json
{
  "pretrain": [
    { "root": "data/ultrafineweb", "column": "text", "glob": "*.parquet" },
    { "root": "data/wikipedia",    "column": "text" }
  ],
  "sft": [
    { "root": "data/reasoning", "column": "messages" }
  ]
}
```

- `column: "messages"` is rendered through the tokenizer chat template.
- **Framing**: BOS is prepended if the tokenizer omits it, each document ends with a supervised EOS
  so the model learns to terminate, followed by `num_mtp_tokens - 1` unsupervised pad separators.
  `num_mtp_tokens` must be **=> the models MTP head count** so MTP is never supervised across a
  document boundary
- Each sample emits `input_ids`, a `[B, S]` `document_ids` segment map, `labels` (non predicted
  positions set to `-100`), and a `file_idx` used for resume bookkeeping
- File order is deterministic (sorted), files are sharded across DataLoader workers, and
  `start_file_idx` fast-forwards past already-consumed files on resume

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

`TokenTracker` counts real (non padding) tokens once `pad_token_id` is set. `target_tokens` (config)
drives the total-step count and the cosine schedule length, router-noise annealing decays over
`noise_anneal_tokens`

## Checkpointing & resume

Checkpoints save model/optimizer/scheduler state plus `epoch`, `dataset_idx`, `token_count`,
`file_idx`, and the loss history ([utils.py](../utils.py)). Saved every 5000 steps and on
`KeyboardInterrupt` (interactive confirm). On restart the newest checkpoint is auto loaded and the
dataset fast-forwards to `file_idx` (a small `RESUME_SAFETY_WINDOW` accounts for multi-worker
lookahead)

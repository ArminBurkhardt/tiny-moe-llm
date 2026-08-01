# PLAN.md

The plan for the first (possibly third) real pretraining run of `tiny-moe-llm`: what changes, why, in what order,
and how to validate each step before spending money.

**Read [CLAUDE.md](CLAUDE.md) first** — it has the invariants this plan assumes and is not
repeated here.

## Rules

- Steps are ordered by dependency. Do them **one at a time**, in order. Each has an Acceptance
  block — run it before starting the next step, don't batch.
- Commit after each passing step (`feat:` / `chore:`, branch `train-build`).
- Match existing conventions: lowercase explanatory comments justifying *why*, Google-style
  docstrings with `Args:` on public modules. Don't strip existing comments.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, or `moe` in the training
  loop goes through `accelerator.unwrap_model(model)`.
- **Never add `.item()`, `.tolist()`, `.cpu()`, or boolean mask indexing to the per-step path.**
  Host syncs dominate at this model size. Accepted exceptions: `m_splits` via `.tolist()`,
  `TokenTracker.sync()` at log/checkpoint cadence.

---

## Target config: A' (768x8)

Chosen over a 492M/1024x10 variant: ~25% cheaper per token, and at a fixed euro budget cheaper
wins (see Step 11's budget math — the bigger variant would buy ~28 tokens/param instead of ~76).

```yaml
model:
  hidden_size: 768               # was 512 -- 512 kept every GEMM skinny and MFU low
  intermediate_size: 2304        # dense decoder MLP only
  moe_intermediate_size: 2304    # NEW key -- routed + shared experts, see Step 2
  num_layers: 8                  # was 5
  num_attention_heads: 12        # was 8
  head_dim: 64                   # unchanged
  num_mlp_experts: 32            # was 36/21, see below
  num_attn_experts: 1            # unchanged, see below
  num_ir_entries: 8192           # was 16384, see below
  n_loops: 3                     # was 4 -- non-MLP experts run densely every loop
  top_k: 2                       # unchanged
  vocab_size: 65536              # was 129280, see Step 8
  lm_head_factor: 4              # unchanged
  loop_ce_weights: [0.2, 0.3, 1.0]   # Step 4, len == n_loops
  lambda_ponder: 3e-3                # Step 3
  lambda_conf: 0.05                  # Step 4
  # identity_skew: DELETED, Step 3
training:
  ponder_warmup_tokens: 1000000000
  ponder_ramp_tokens: 1000000000
```

**Param budget** (counted from the module tree):

| config | total | active (incl. emb) | active (excl. emb) | fwd FLOP/token |
|---|---|---|---|---|
| current `config.yaml` (512x5, V=129280) | 254M | 147M | — | — |
| **A' (768x8, M=32, V=65536)** | **332M** | **174M** | **104M** | **357M** |

Training FLOPs ≈ 3x forward ≈ **1071 MFLOP/token**. Recompute if the config moves, especially
`num_attn_experts`, `n_loops`, or `moe_intermediate_size`.

**Sizing decisions, briefly:**
- `n_loops: 4 -> 3` — non-MLP experts run densely every loop, so 4 loops meant 12 dense attention
  passes per forward.
- `num_ir_entries: 16384 -> 8192` — the IR expert was ~11% of forward FLOPs (two 16384-wide
  matmuls + a 16384-way softmax, dense every loop) holding ~1GB activation per loop. Halving frees
  throughput and ~2GB peak memory.
- `num_mlp_experts: 32` — routed experts add total params at **zero active compute** (the limit
  is per-expert training data, not FLOPs). The "pool divisible by 8 for FP8" rule some earlier
  notes assumed **is not real**: nothing enforces it, and `ParallelSparseMoELayer.forward` runs
  under `te.autocast(enabled=False)` because dynamic routing can't guarantee NVFP4's 16-row
  alignment — routed MLPs never hit FP8 regardless of pool size. At M=32/pool=35/top_k=2/~25B
  tokens, each expert sees ~1.4B tokens; M=52 would drop that to ~0.9B (undertrained). M=32 is the
  end of the free lunch, not a step toward a bigger model.
- `num_attn_experts` stays at **1** — attention experts are cheap in params (+3M each) but run
  densely every loop and get masked by routing afterward, so most of the compute is discarded.
  Each additional one costs ~5% throughput for ~1% params. Revisit only alongside the shared
  attention expert (Step 2), which makes that compute unconditional instead of discarded.
- `moe_intermediate_size` (new key, Step 2) is the only clean total-vs-active knob going forward.
  Hit future param targets with it or with `num_mlp_experts` (total only) — not with
  `intermediate_size`, which moves the dense decoder (fully active) too.

---

## Timeline

### Step 1 — Loop residual + `loop_scale`

**Problem:** `forward_step` currently *replaces* `hidden_states` rather than updating it — no
gradient path across a loop boundary except through a stochastically-selected identity expert.

**File:** `modules/model/moe.py`, `LoopMixtureOfExperts`.

```python
# LayerScale/ReZero-style gate: loop starts as a near-no-op, learns how much refinement to apply.
# init 0.1, NOT 0: at loop_scale == 0 the CE loss has exactly zero gradient wrt p_halt (Step 3),
# so the ponder term becomes the halt head's only signal, saturates p_halt at 1, and freezes
# loop_scale in turn -- the whole loop silently goes dead while the loss curve still descends.
self.loop_scale = nn.Parameter(torch.full((1,), 0.1))
```

Not the same mechanism as `layer_scalar` in the dense decoder (init-1 gain on a whole layer
output) — that precedent doesn't transfer to this init value.

Replace the `forward_step` tail:
```python
delta = self.dropout(self.post_norm(output))
hidden_states = hidden_states + self.loop_scale * delta
return hidden_states, load_balancing_loss
```
Update the `n_loops` caller for the new return signature.

**Acceptance**
- `test_overfit.py` reaches a **lower** loss in the same steps than before. If not, stop —
  nothing downstream is worth doing yet.
- `test_attention_equiv.py` still passes.

---

### Step 2 — Shared (always-on) experts + `moe_intermediate_size`

**Problem:** every token goes through `top_k=2` routed slots and nothing else — no guaranteed
dense transform, no guaranteed full-sequence attention.

**File:** `modules/model/moe.py`, `LoopMixtureOfExperts`.

Add `self.shared_mlp` (same hidden size/activation as a routed MLP expert) and `self.shared_attn`
(reuse `SelfAttention` for the RoPE cache/varlen path). Neither is in the router pool — not in
`Router`'s output dim, not in `compute_aux_loss`. Seed the accumulator with them:

```python
# every token gets a dense transform + full-sequence attention regardless of routing.
# stabilises early training, lets routed experts specialise instead of learning the same generic fn.
output = self.shared_mlp(hidden_states) + self.shared_attn(
    hidden_states, cu_seqlens, max_seqlen, position_embeddings
)
# ... existing weighted accumulation of routed expert outputs into output ...
```

Their row counts are static (`B*S`), unlike the routed grouped GEMM, so they **may** run inside
`te.autocast` — don't move `ParallelSparseMoELayer`'s `te.autocast(enabled=False)`.

**`moe_intermediate_size` — same commit.** `LoopMixtureOfExperts` currently reuses the dense
decoder's `intermediate_size`, so total and active params can't move independently. Add it:
`config.yaml` under `model:` -> `ModelConfig.Params` -> new `TinyMoETransformer.__init__` kwarg
(default to `intermediate_size` if absent, so old checkpoints/tests still construct) -> threaded
into `LoopMixtureOfExperts` -> `ParallelSparseMoELayer` and `self.shared_mlp`. `Gemma4TextModel`
keeps using plain `intermediate_size`.

**Acceptance**
- Param count increases by `3 * hidden_size * moe_intermediate_size` (shared MLP) plus
  `~2.5 * hidden_size^2` (shared attention) — +5.3M / +1.5M at the Step-final config.
- `test_overfit.py` passes; router output dim unchanged from Step 1.
- Omitting `moe_intermediate_size` from `config.yaml` still constructs; setting it to a different
  value changes only MoE expert sizes (verify per-module param counts, not just the total).
- Watch the expert selection plots over the following steps: if routed weights collapse toward
  zero, the shared path has swallowed the block.

NOTE: There should be an actual resdiual stream here, with actual skip connections to keep gradients stable.

---

### Step 3 — Halt head; delete identity expert and `identity_skew`

The invasive step. Don't start until Steps 1-2 are green and committed.

**3a. Halt head.** File: `modules/model/moe.py`.
```python
self.halt_proj = nn.Linear(hidden_size, 1, bias=True)
nn.init.zeros_(self.halt_proj.weight)
nn.init.constant_(self.halt_proj.bias, -2.0)   # p_halt ~ 0.12 at init
```
In `forward_step`:
```python
# soft, differentiable halting. p_halt -> 1 means "don't modify me further".
# compute-allocation signal, not a confidence score -- see Step 4b.
p_halt = torch.sigmoid(self.halt_proj(hidden_states))   # [B, S, 1]
hidden_states = hidden_states + (1.0 - p_halt) * self.loop_scale * delta
return hidden_states, load_balancing_loss, p_halt
```
Collect `p_halt` across loops (stack/running mean, never `.item()`) and return it from
`LoopMixtureOfExperts.forward`.

`p_halt` is a **greedy per-loop gate, not cumulative ACT** — recomputed each loop from the
current hidden state, so a token can halt at loop 1 and un-halt at loop 2. Deliberate: an
ACT-style accumulator adds a remainder term and a hyperparameter whose benefit is unmeasurable
until Gate 5 (Step "Local validation") shows the signal is informative at all. Inference exit
rule: **stop once `p_halt > tau` at the current loop.** Revisit only if Gate 5's early-exit
degradation curve shows exiting at loop 1 is catastrophic rather than merely worse.

**3b. Ponder loss.** File: `scripts/pretrain.py` (near where `load_balancing_loss` is added).
```python
# mask to real tokens: unmasked, trailing document padding dominates the term
ponder = ((1.0 - p_halt_all) * valid_mask).sum() / valid_mask.sum().clamp(min=1)
loss = loss + lambda_ponder_now * ponder
```
`lambda_ponder = 3e-3` target.

**`lambda_ponder` must be warmed from zero — this is a correctness requirement, not tuning.**
With `loop_scale` small, CE loss has near-zero gradient wrt `p_halt`, so the ponder term is
briefly the halt head's *only* signal, and it's constant-sign: AdamW normalizes away magnitude,
so the halt bias climbs at ~`lr`/step regardless of how small `lambda_ponder` is. If `p_halt`
saturates before `loop_scale` has grown, `loop_scale` stops receiving gradient and the loop is
permanently, silently disabled (the dense decoder keeps the loss descending, so nothing looks
wrong).
```python
# hold at 0 while loop_scale grows, then ramp -- driven from the live token count like the
# router noise anneal above it in the step loop, no host sync.
tokens = unwrapped_model.token_count
warm, ramp = TrainingConfig.ponder_warmup_tokens, TrainingConfig.ponder_ramp_tokens  # 1e9 each
lambda_ponder_now = TrainingConfig.lambda_ponder * min(1.0, max(0.0, (tokens - warm) / ramp))
```

**3c. Remove the identity expert.** Grep `identity_expert_index` / `identity_skew`
(`moe.py`, `router.py`, `transformer.py`, `config.py`/`.yaml`, docs).

```
[ SelfAttn x A | CrossAttn x A | IR x I | identity | MLP x M ]     ->
[ SelfAttn x A | CrossAttn x A | IR x I | MLP x M ]
                                          ^ first_mlp_index = 2A + I  (renamed from identity_expert_index)
```
- Router pool size drops by 1; update `num_experts`.
- Remap in `forward_step`: `idx > identity_expert_index` -> local `idx - (2A+I) - 1` becomes
  `idx >= first_mlp_index` -> local `idx - (2A+I)`.
- Non-MLP slots still collapse to `(index 0, weight 0)` via **mask multiply** — never
  `mask.sum()` / boolean indexing (per-expert device sync).
- `_ExpertTracking` drops the identity entry from counts/plot labels.
- `compute_aux_loss` needs no identity exclusion — confirm it iterates only real experts.
- Delete `identity_skew` entirely (the `on_loop / n_loops` scaling, pre/post-skew aux-loss
  split — aux loss is now computed on router probabilities directly).
- `ModelConfig.Forward` currently holds only `identity_skew`; verify `model(**ModelConfig.Forward)`
  works with `{}`, or move `lambda_ponder` into it.

**Acceptance**
- `grep -rn "identity_skew\|identity_expert_index" .` empty outside docs history.
- Model constructs; router output dim == `2A + I + M`; `test_overfit.py` passes.
- 50-step run: mean `p_halt` in ~0.05-0.4. Pinned at 0 or 1 is a bug.
- **Deadlock check**: force `ponder_warmup_tokens=0` for 2000 steps and confirm `p_halt` climbs
  to 1 / `loop_scale` stalls (reproduces the failure mode). Restore warmup, confirm `loop_scale`
  moves off 0.1 while `p_halt` stays in range. If the failure mode does *not* reproduce, the halt
  head isn't wired into the loss — check that before concluding warmup is unnecessary.

---

### Step 4 — Per-loop CE supervision and a correctness head

Two independent problems, do as separate commits. Rationale applies once you try to *use*
`p_halt` for early exit.

**4a. Per-loop CE.** With CE only at the final loop, intermediate hidden states are optimized
only as inputs to the next loop, not as things `lm_head` can read — thresholding `p_halt` and
exiting at loop 1 reads from an interface never trained. (Controlled evidence: 44M/129M looped
models, close to this scale.)

Files: `modules/model/transformer.py`, `modules/model/mtp.py`. Apply `main_lm_head` at every
loop with ascending weights (`loop_ce_weights: [0.2, 0.3, 1.0]`, asserted `len == n_loops`).

- **Apply the final `RMSNorm` at every loop, not just the last.** `lm_head` reads `self.norm(x)`,
  never the raw residual stream. Skipping this makes per-loop losses meaningless and looks
  identical to "hidden states not threaded through" in the monitoring table — misdiagnosis risk.
- Reuse the existing chunked, checkpointed CE path (`CE_CHUNK_SIZE=2048`) — do not materialize
  `[T, vocab]` per loop (3x activation blowup).
- Training forward still returns hidden states (`return_hidden=True`, `delayed_mtp_loss(True)`),
  now a **list/stack of per-loop hidden states**; LM head applied inside the loss. Any new call
  site must pass `main_lm_head=` or the activation peak silently doubles.
- MTP heads apply to the **final loop only**, never per loop.

Expect ~3x LM-head compute and a measurable tokens/sec drop — intended cost.

**4b. Correctness head.** `p_halt` is trained by LM loss + ponder cost, so it learns *"does more
compute change my prediction?"*, not *"is my prediction right?"* A confidently hallucinated fact
is stable under refinement — it halts at loop 0 and reads as maximum certainty. Utility
calibration ≠ correctness calibration. Don't fix this by reweighting `p_halt`; add a second head.

File: `modules/model/transformer.py`.
```python
# separate from p_halt on purpose: p_halt asks "is more compute useful", this asks
# "is this prediction correct" -- they come apart on confident hallucinations.
self.correct_proj = nn.Linear(hidden_size, 1, bias=True)
nn.init.zeros_(self.correct_proj.weight)
nn.init.constant_(self.correct_proj.bias, 0.0)
```
Target is free (no labels, no extra forward pass), inside the existing chunked CE:
```python
with torch.no_grad():
    is_correct = (logits_chunk.argmax(-1) == labels_chunk).float()
conf_loss = F.binary_cross_entropy_with_logits(correct_logit_chunk, is_correct, reduction="none")
# mask label == -100 positions, then mean
```
`loss = loss + lambda_conf * conf_loss`, `lambda_conf = 0.05`. Computed on the **final loop's**
hidden states only. `is_correct` must be `no_grad` (a gradient leak here quietly degrades LM
loss). Respect the `-100` mask.

| signal | question | used for |
|---|---|---|
| `p_halt` | is more latent compute useful here? | inference early exit |
| `p_correct` | is this prediction right? | abstention, ECE, truthfulness |

Report ECE on `p_correct`, never on `p_halt`.

**This head is provisional.** `max(softmax(logits))` (`p_max`) is already a strong, free
per-token correctness predictor. Log both from the start:
```python
with torch.no_grad():
    p_max = logits_chunk.softmax(-1).max(-1).values
```
**Gate 5** (below) measures ECE + abstention AUROC for both on the same held-out slice. If
`p_correct` doesn't beat `p_max` on both, **revert Step 4b** (drop the head, loss term,
`lambda_conf`) and use `p_max` as the abstention signal for all of the post-training steps —
the abstention thesis needs *a* calibrated signal, not specifically a learned one. Cheaper to
decide now (one commit) than after renting (a rerun).

**Acceptance (4a)**
- `len(loop_ce_weights) == n_loops` asserted at construction.
- Per-loop loss strictly decreasing (loop 0 > loop 1 > loop 2) after a few hundred steps. Equal
  losses mean per-loop hidden states aren't threaded through.
- Peak memory +<20% (a bigger jump means logits are being materialized). `test_overfit.py` passes.

**Acceptance (4b)**
- Mean `p_correct` within ~0.1 of batch top-1 accuracy after a short run (smoke test only, a
  bias-only constant predictor passes it — real check is Gate 5).
- Gradient check: LM loss at `lambda_conf=0` matches `lambda_conf=0.05` within noise at step 1
  (no gradient leak through `is_correct`).
- Tokens/sec regression from 4b alone under 3%. `p_max` logged alongside `p_correct`.

---

### Step 5 — Write config A', assertions, and delete the FP8 divisibility myth

**File:** `config.yaml` — diff against the existing `model:` block (don't drop keys `config.py`
reads: `num_ir_experts`, `ir_dim`, `dropout`, `per_layer_embeddings_size`, `max_seq_length`,
`mtp_num_extra_tokens` — that's a `KeyError`). Apply the yaml block from "Target config" above.

**Delete the FP8 pool-divisibility constraint** — verified not real: `grep -rn "divisible" .`
only finds a comment on `config.yaml:13` and one in `ParallelSparseMoELayer.forward`; the routed
MLP GEMMs never run in FP8 (`te.autocast(enabled=False)` wraps them, precisely because dynamic
routing can't guarantee NVFP4's 16-row group alignment — a constraint on row count, not pool
size). Remove the `# due to fp8 requirement...` comment from `config.yaml` and the "divisible by
8" line from `CLAUDE.md`'s Config section (CLAUDE.md already documents the real constraint
elsewhere under Model invariants).

Assert at construction:
- `vocab_size % lm_head_factor == 0` (65536/4=16384 OK), `vocab_size % (lm_head_factor*2) == 0`
  for MTP (65536/8=8192 OK).
- `hidden_size % lm_head_factor == 0` (768/4=192 OK), `(hidden_size//2) % (lm_head_factor*2) == 0`
  for MTP (384/8=48 OK).
- `len(loop_ce_weights) == n_loops`.
- `vocab_size <= 65536` (uint16 fit, Step 8's `train.bin` dtype).
- `mtp_num_extra_tokens <= num_mtp_tokens` (dataset separator budget).

Print total and active param counts (separately) and the per-token training FLOP estimate at
construction — the budget math in Step 11 is keyed to it and goes stale silently otherwise.

**Acceptance**
- Model constructs, assertions pass, printed counts ~332M total / ~174M active (~104M excl.
  embeddings). Printed FLOP estimate ~357 MFLOP/token forward.
- `test_overfit.py` passes at the new size. `grep -rn "divisible by 8" .` empty.

---

### Step 6 — Token-count break and schedule fixes

**File:** `scripts/pretrain.py`. Currently `total_steps` anchors the cosine schedule to
`target_tokens` but the loop runs `num_epochs` epochs regardless, so a larger dataset trains past
the schedule at `eta_min`.

```python
# stop at the token budget the LR schedule is anchored to. read from the sync-free counter at
# LOG_INTERVAL cadence -- no .item() in the step path.
if unwrapped.token_count >= TrainingConfig.target_tokens:
    break
```
Place inside the existing `LOG_INTERVAL` throttle block, not per step. Also: `num_epochs: 1`;
checkpoint interval 5000 -> **1500** steps (hardcoded, makes the run interruptible);
`LOG_INTERVAL` stays 10.

**Acceptance**
- Short run with `target_tokens` set very low exits at the right token count.
- Interrupt/resume works; LR re-anchors by tokens (`resume_token_count // tokens_per_step`), not
  by saved step.

---

### Step 7 — Instrumentation

**File:** `scripts/pretrain.py`, inside the existing `LOG_INTERVAL` throttle. Add to the log
line: `loop_scale`, mean `p_halt`, mean `p_correct`, mean `p_max`, batch top-1 accuracy, per-loop
CE (all `n_loops` values), aux loss, ponder loss, conf loss, current `lambda_ponder` (constant in
the log means the warmup isn't wired to the token counter).

```python
unwrapped = accelerator.unwrap_model(model)
loop_scale = unwrapped.moe.loop_scale.item()
```
All reads through `unwrap_model`. Log `p_correct` and top-1 accuracy **as a pair** — their
divergence is the calibration signal to watch during the run.

**Acceptance:** 100-step run prints all of the above without a per-step sync (tokens/sec should
not regress beyond the Step 4a cost).

---

### Step 8 — Vocab prune to 65536

Do before tokenizing anything. **Target 65536, not smaller** — the load-bearing reason is disk,
not params: 30B tokens as uint32 is 120GB on a 120GB instance with no room for source shards.
Any `vocab_size <= 65536` fits uint16 and halves the dataset to 60GB — the entire disk argument.
Going lower (e.g. 49152) buys ~21M params for a much more aggressive merge-table surgery and more
fertility damage.

New script: `scripts/prune_vocab.py`
1. Sample ~2GB of text from the **final training mix** (web+code+math+wiki at training ratios —
   web-only counts badly underweight brackets/braces/indentation/LaTeX).
2. Count token frequencies under the current DeepSeek tokenizer.
3. Keep top-N plus all special/byte-fallback tokens to reach exactly 65536; keep merges closed
   under prefix (a kept merge never depends on a dropped one).
4. Remap ids, write new tokenizer to `ckpts/pretrained/<name>-65536/`, emit an
   `old_id -> new_id` table.

**Preserve:** `pad_token_id == eos_token_id`; BOS pinned to id 0 (`embed_tokens` deliberately has
no `padding_idx` — see CLAUDE.md).

**Gate on fertility, not round-trip identity.** Byte fallback means decode round-trips almost
regardless of prune damage — the actual cost is tokens-per-byte going up (text that cost 1 token
now costs 2-3), which shrinks the effective token budget and eats the FLOP saving the prune was
partly justified by.

**Acceptance**
- `len(tokenizer)==65536`, `pad_token_id==eos_token_id`, `bos_token_id==0`, max id `<65536`.
- **Fertility regression under 3%**, measured tokens/byte before/after on a held-out ~200MB
  sample at the phase-1 mix ratios, reported per source (code/math regress worst). Above 3%,
  keep more tokens.
- Record measured fertility in `manifest.json` and **scale Step 11's token target by it** (a 3%
  regression = 3% fewer documents for the same token budget).
- Encode/decode round-trip over 5000 held-out docs is byte-identical (catches a broken remap,
  not a bad prune — expected to pass trivially).

---

### Step 9 — mmap dataset

**File:** `modules/data/dataset.py`. Replace streaming-parquet + on-the-fly tokenization with a
flat-file reader over `train.bin` (flat uint16 stream) + `train.idx` (document start offsets,
uint64 — without stored boundaries `document_ids` can't be reconstructed and varlen attention
silently corrupts). Two pairs from Step 11: `phase1.{bin,idx}`, `phase2.{bin,idx}`, selected by
config.

**Must survive the rewrite:** still `IterableDataset` + `DataLoader(batch_size=None)`; still
emits `document_ids [B,S]` -> `cu_seqlens` built **in-thread**; `cu_seqlens` never in the batch
dict (ragged dim0, accelerate's `split_batches` would truncate it); `max_seqlen` still passed as
`S` (no `.item()` sync); packing shape unchanged (docs concatenated to `max_length`, split at
boundaries, `EOS + (num_mtp_tokens-1)` pad, trailing pad = length-1 segments); labels `-100`
outside document interior + terminating EOS; `[B]`-shaped bookkeeping tensors for accelerate
splitting; `NUM_DATA_WORKERS=4`, worker `w` takes `[w::num_workers]`.

**Simplifies:** resume state collapses from `(file_idx, record_idx, shard_token_count)` + per-
worker positions to a single **global sequence offset**. `file_order`, `build_legacy_order`,
`max_tokens_per_shard` become dead — remove them and their checkpoint fields but keep
`load_checkpoint`'s `.get(..., default)` pattern for old checkpoints; add `global_offset` to
`save_checkpoint` with a default.

**Acceptance**
- `test_attention_equiv.py` passes (catches broken `document_ids`).
- Synthetic `train.bin` with known boundaries produces matching `cu_seqlens`.
- Label mask spot-check: `-100` on pads, real ids on interiors + terminating EOS.
- Interrupt at step N, resume: next sequence consumed is `N*batch_size`, no gap or repeat.

---

### Step 10 — FP8 wiring (do not enable locally)

**File:** `scripts/pretrain.py`. `fp8_recipe = DelayedScaling(...)` exists unused — gate it so
the same code runs on the 5090 (off) and H100 (on):
```python
USE_LOW_PRECISION = os.environ.get("USE_FP8", "0") == "1"
chosen_recipe = fp8_recipe if USE_LOW_PRECISION else None
```
Leave `te.autocast(enabled=False)` around `ParallelSparseMoELayer`'s GEMMs. Gradient checkpointing
stays off; if ever enabled, use `transformer_engine.pytorch.checkpoint`, never
`torch.utils.checkpoint` (breaks quantized layers).

**Acceptance:** `USE_FP8=0` reproduces current BF16 behavior bit-for-bit. (H100 only, later)
`USE_FP8=1` runs with no TE fallback warnings.

---

### Step 11 — Data prep script + budget decision

New script: `scripts/prepare_data.py`, **runs on the rented box, not locally**. For each source:
stream shard -> tokenize -> append to target `.bin` -> delete shard (peak disk = bin size + a few
GB working set). Use `huggingface_hub.hf_hub_download` on individual files (`list_repo_files()`
first, filter, pull N) — never `load_dataset` for bulk sources (no control over disk footprint).
Tokenize with `num_proc=os.cpu_count()`, batched.

**Sources** (selected for quality at <500M scale, actual text not metadata-requiring-
reconstruction, sliceable by file):

| dataset | slice | target tokens |
|---|---|---|
| `HuggingFaceFW/fineweb-edu` | `data/CC-MAIN-2025-26/`, partial | 15B |
| `mlfoundations/dclm-baseline-1.0` | handful of global shards | 2.5B |
| `HuggingFaceFW/finepdfs-edu` | `eng_Latn`, few files | 2.2B |
| `nvidia/Nemotron-CC-Code-v1` | few shards | 4B |
| `nvidia/Nemotron-CC-Math-v1` | config `4plus`, few shards | 2B |
| `wikimedia/wikipedia` | `20231101.en`, subset | 1.1B |
| `HuggingFaceTB/smoltalk2` | non-reasoning splits, whole | 0.7B |

Do **not** substitute `HuggingFaceTB/stack-edu`, `nvidia/Nemotron-Pretraining-Code-v1/v2`, or
`HuggingFaceTB/smollm-corpus` — all need reconstruction from an external store or are superseded.

Write two bin/idx pairs, sources **interleaved at mix ratios during writing** (straight
sequential read at train time, no online sampling):

| | Phase 1 — 25.5B tok (85% of budget) | Phase 2 anneal — 4.5B tok (15% of budget) |
|---|---|---|
| FineWeb-Edu | 55% | 15% |
| DCLM-baseline | 10% | — |
| FinePDFs-Edu | 7% | 10% |
| Nemotron-CC-Code | 12% | 22% |
| Nemotron-CC-Math `4plus` | 3% | 30% |
| Wikipedia | 3% | 8% |
| SmolTalk2 | — | 15% |
| LR | warmup -> cosine to 10% | -> ~0 |

The instruct row in phase 2 is deliberate (Nemotron 3 / Olmo 3 fold SFT-style data into late
pretraining) — it speeds SFT convergence and stops the base model being pure completion-mode.
Expect math benchmarks near zero regardless (math data is for representation quality, not GSM8K).

**Hold out** the smoltalk2 rows used here — record ids in the manifest so SFT (Step 13) doesn't
retrain on data already seen. Shuffle at document granularity within each phase.

**Acceptance**
- Realized per-source token counts within 2% of target.
- `train.idx` monotonically increasing, last entry == `len(train.bin)`.
- `phase1.bin` dtype uint16, max value < 65536.
- Peak disk during the run under ~70GB.
- `manifest.json` records exact repo ids/revisions/filenames, per-source token counts, smoltalk2
  holdout ids, and the measured fertility from Step 8.

---

## Local validation gates (5090) — do not rent until all five pass

**Gate 1** — `bash tests/run_env_check.sh`

**Gate 2** — `bash tests/run_tests.sh tests/test_attention_equiv.py`

**Gate 3** — `bash tests/run_tests.sh tests/test_overfit.py` drives a small batch to near-zero
loss at config A'.

**Gate 4** — ~500M-token run on a local slice at the new hidden size. Record:

| metric | pass condition |
|---|---|
| tokens/sec, MFU | recorded — input to the budget decision below |
| `loop_scale` | moves off its 0.1 init (stuck = ponder deadlock, Step 3b) |
| mean `p_halt` | ~0.05-0.4, not pinned; climbing to 1 = deadlock |
| `lambda_ponder` | visibly ramping, not constant |
| per-loop CE | strictly decreasing across loops |
| expert selection plots | no collapse to a few MLP slots; not swallowed by shared experts |
| loss curve | monotone decreasing, no spikes |

Then **interrupt and restart** to prove resume works against the mmap dataset before it costs
money.

**Gate 5 — calibration sanity.** New script `scripts/eval_calibration.py`, held-out slice.
- **ECE of `p_correct` < 0.15** — the load-bearing number for the whole abstention thesis. If
  uninformative at 500M tokens it won't become informative at 25B; debug Step 4b before renting.
- **`p_correct` vs `p_max`**: ECE and abstention AUROC for both, same slice, side by side. If
  `p_correct` doesn't beat `p_max` on both, **revert Step 4b** and use `p_max` throughout
  post-training.
- **Early-exit degradation curve**: force exit at loop 1/2/3, record perplexity. Expect monotone
  improvement; catastrophic loop-1 exit means Step 4a isn't working, and is also the only
  evidence that would justify revisiting the greedy `p_halt` decision (Step 3a).
- **`p_halt`/`p_correct` correlation**: expect weak (that's the point of two heads); above ~0.8
  means one head collapsed into the other.

### Budget decision (before renting)

**Budget is fixed at EUR 100** — token count is the output, not an input. At ~EUR 2/hr that's 50
instance-hours; reserve ~3h for prep/smoke-test/extraction, leaving **~47 training hours**. Scale
measured 5090 tokens/sec by the expected H100 ratio for an MFU estimate (assumes ~1071
MFLOP/token for A', ~990 TFLOPS H100 SXM BF16 peak — use Step 5's printed FLOP number if it's
drifted):

| MFU vs BF16 peak | tokens in 47h | tokens/total param |
|---|---|---|
| 10% | 16B | 47 |
| 12.5% | 20B | 59 |
| 15% | 23B | 71 |
| 17.5% | 27B | 82 |
| 20% | 31B | 94 |
| 25% | 39B | 118 |

Prepared data caps the run at 30B — above ~18% MFU the run is data-limited, not budget-limited
(surplus should go to `moe_intermediate_size`, not a longer run). Measure MFU **after** Step 4a
lands, not before (it costs real throughput).

1. Round the table result **down**, scale by Step 8's measured fertility regression.
2. Set `target_tokens` to that number **now**, before renting — the cosine schedule is anchored
   to it, and stopping early against a longer schedule wastes the phase-2 anneal (the single most
   expensive avoidable mistake in this plan).
3. `target_tokens` describes the **combined** phase 1 + phase 2 run, not either phase alone.

If the number comes out below ~16B: cheaper instance or smaller `moe_intermediate_size`, **not**
a longer run.

---

## Repo prep before renting

`.gitignore` swallows `*.json`, `*.cmd`, `ckpts/`, `venv/`, `tests/`, `env_init` — so
`data_config.json`, the test suite, and the env script won't exist on the rented box after a
clone. Before renting:

1. Un-ignore `tests/` (recommended — the gates above need to run on the box too), or write a
   `deploy.sh` that scp's `tests/` + `data_config.json` post-clone.
2. Write `vast_init` (committed, unlike `env_init`): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
   **no** WSL CUDA 12.9 paths, **no** `/mnt/d/AI/...` hardcode, **no** venv activation (NGC
   container has the environment already).
3. Fix `tests/run_tests.sh` — it hardcodes `cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm` and
   `source env_init`. Make repo root and init script overridable by env var.
4. Commit `manifest.json` (Step 11) and `scripts/prepare_data.py`.

---

## Vast.ai runbook

**Total spend capped at EUR 100** (~47 training hours after prep/smoke-test/extraction at
EUR 2/hr — steps 3-4 below are billed too).

1. **Rent**: 1x H100 SXM/NVL 80GB, **interruptible** (~half price; the per-step checkpoint/resume
   path exists to exploit this), **120GB disk** (fixed at creation, can't grow later), image
   `nvcr.io/nvidia/pytorch:25.xx-py3`. Confirm hourly rate; above ~EUR 2.2/hr re-read the budget
   table and cut `target_tokens` first. Do **not** `pip install -r requirements.txt` wholesale —
   its TE/flash-attn wheels target sm120, H100 is sm90; the NGC image ships both prebuilt, layer
   only non-CUDA deps.
2. **Verify environment** before downloading 47GB:
   ```bash
   python -c "import torch, transformer_engine.pytorch, flash_attn; print(torch.cuda.get_device_capability())"
   bash tests/run_env_check.sh
   ```
   Expect `(9, 0)`. Catch a broken TE at minute two, not minute ninety.
3. **Data prep**: `python scripts/prepare_data.py`, in the instance's on-start script (unattended,
   40min-2h, likely download-bound). Verify against Step 11's acceptance criteria.
4. **Smoke test**: `USE_FP8=1 python scripts/pretrain.py`, kill after ~200 steps. Confirm
   `dry_run` asserts finite loss, FP8 actually active (TE warns on silent fallback), tokens/sec
   in range of the Gate-4 extrapolation, peak memory has headroom. **If tokens/sec differs from
   the extrapolation by more than ~20%, redo the budget math before phase 1** — this is the last
   cheap moment to correct `target_tokens`. Then delete the checkpoint and restart clean.
5. **Phase 1**: 85% of budget on `phase1.{bin,idx}`. LR warmup -> cosine to `0.1*lr`; router noise
   anneals over `noise_anneal_tokens` from the live token count.
6. **Phase 2 anneal**: 15% of budget on `phase2.{bin,idx}`, LR -> ~0, resume from the phase-1
   checkpoint. Resume re-anchors the schedule by token count, so `target_tokens` must describe
   the combined run.
7. **Extraction**: scp checkpoints down as written (or at minimum after phase 1) — a reclaimed
   instance takes its disk with it. "Latest" resolves by newest mtime, not highest step; scp can
   rewrite mtimes, verify resume picks the right file after any round-trip. Before releasing the
   instance, re-run `eval_calibration.py` (Gate 5) on the final checkpoint and record the numbers
   — re-renting later to compute them is wasteful.

---

## Monitoring reference

**Every `LOG_INTERVAL`:** loss, tokens/sec, peak mem, `loop_scale`, current `lambda_ponder`, mean
`p_halt`, mean `p_correct`, mean `p_max`, batch top-1 accuracy, per-loop CE (all `n_loops`), aux
loss, ponder loss, conf loss.

**Every checkpoint:** `ckpts/training/expert_selection_*.png` from `_ExpertTracking`.

| symptom | likely cause |
|---|---|
| `loop_scale` stuck at 0.1 **and** `p_halt` climbing | ponder deadlock — warmup not wired to token counter (Step 3b) |
| `loop_scale` stuck at 0.1, `p_halt` in range | residual wiring wrong (Step 1) |
| `p_halt` saturated at 1 | `lambda_ponder` too high, or ramping too early |
| `p_halt` pinned at 0 | halt head not receiving gradient |
| `lambda_ponder` constant in the log | warmup reading a stale/absent token count |
| per-loop CE flat across loops | hidden states not threaded through, **or** final `RMSNorm` not applied per loop (Step 4a) |
| per-loop CE huge at loops 0-1, normal at last | final `RMSNorm` not applied to intermediate states |
| `p_correct` far from top-1 accuracy | label mask or target wrong (Step 4b) |
| `p_correct` collapses to a constant | `lambda_conf` too low, or gradient leaking through `is_correct` |
| `p_correct` tracks `p_max` exactly | head learned nothing beyond baseline — flag for the Gate 5 revert |
| routed expert weights near zero, loss still falling | shared MLP/attention (Step 2) swallowed the block |
| expert selection collapses to a few MLP slots | aux loss weight too low |
| tokens/sec drops after a code change | a host sync entered the step path |
| loss spikes on resume | schedule re-anchoring or `global_offset` wrong |

`_ExpertTracking` guards against activation-checkpoint recompute double counting via
`begin_forward(expected_updates)`, samples every 8th forward. If expert counts look wrong after
any loop-structure change, check `expected_updates` matches the new `n_loops`.

---

## Post-training

**Target: calibrated abstention, not chain-of-thought.** Calibrated "knows when it doesn't know"
is a shallower, directly measurable (ECE, abstention precision/recall) capability that the
halt/correctness machinery is actually positioned to deliver — multi-step reasoning is not
learnable at 332M total/174M active. Do not evaluate primarily on GSM8K/MATH; math data is in
the mix for representation quality, not benchmark score (SmolLM2-1.7B scored 3.21 on math after
6T tokens).

### Step 12 — SFT

New script `scripts/sft.py`. Reuses the model/packing path; swaps data source, adds loss masking
over prompt tokens.

| dataset | role |
|---|---|
| `HuggingFaceTB/smoltalk2` (no-think splits) | general instruction following |
| `rajpurkar/squad_v2` | **primary abstention supervision** — unanswerable questions are the point |
| `allenai/tulu-3-sft-personas-math` | short worked solutions |
| `openai/gsm8k` (socratic) | short numbered steps |
| `HuggingFaceH4/no_robots` | human-written; tone and refusal style |

- **Exclude the smoltalk2 holdout ids** from `manifest.json` (Step 11).
- Mask loss over prompt/system tokens; train on completions only.
- **Do not** use long-CoT trace datasets (KIMI-K2.5/Claude/Fable 5 sets) — Small Model
  Learnability Gap: at this scale they teach fluent filler before a wrong answer, and typically
  carry provider terms restricting training of competing models.
- Keep `p_correct`/`p_halt` supervision active during SFT (still free). If Gate 5 reverted Step
  4b, substitute `p_max` everywhere below; nothing else changes.

**Acceptance:** SQuAD v2 abstention precision/recall on the unanswerable split both reported; ECE
of the abstention signal doesn't degrade relative to the pretrained checkpoint.

### Step 13 — Self-labelled calibration set

The one dataset worth building rather than downloading — requires this model. New script
`scripts/build_calibration_set.py`.

1. Sample the Step 12 checkpoint N=16x at temperature 0.8 on short-answer QA (`trivia_qa`,
   `nq_open`, `squad_v2`).
2. Label each question by empirical pass rate (normalized exact/alias match against reference).
3. Rewrite targets by pass rate: `>0.8` -> the answer; `<0.2` -> an abstention; in between -> a
   hedge, from a small fixed set of phrasings (not free text).
4. Hold out 10% before rewriting (the calibration eval set).
5. Second SFT pass on the rewritten data.

**Acceptance:** abstention rate on the held-out low-pass-rate bucket >60%; on the high-pass-rate
bucket <10% (catches the degenerate "refuse everything" solution); ECE of the abstention signal
improves relative to Step 12.

### Step 14 — RL: deferred, gated

**Do not start.** Evidence at this scale is consistently negative: a 135M single-GPU RLVR study
on GSM8K went from SFT base 24/1319 (1.82%) to 21/1319 at 192-token completions and 16/1319 at
320 — GRPO made it *worse*. On Qwen2.5-0.5B base, even a format reward stayed below 0.1 after 300
steps with no upward trend. Mechanism: under a 0/1 reward, a base model that can't sample correct
solutions produces no gradient signal, and RLVR only amplifies what's already in the base
distribution — at 174M active/~25B tokens there's little to amplify.

**Gate:** pass@8 on the target task with the Step 13 checkpoint. **Below ~15%, do not proceed** —
the budget would be spent confirming the null result.

If the gate ever passes: vanilla GRPO is architecturally mismatched to a looped model (credits
output tokens while the computation is latent). Read LoopRPT (arXiv 2603.19714) and RLTT first —
they assign reward to per-loop latent states and report improved gate calibration (more early
exits, final-step dominance maintained) as a side effect, directly relevant to the `p_halt`
machinery here.

# EXECUTION_PLAN.md

Implementation instructions for the changes decided in [TRAINING_PLAN.md](TRAINING_PLAN.md).

**Read [CLAUDE.md](CLAUDE.md) first.** It contains the invariants this plan assumes.

## Rules for whoever implements this

- Tasks are ordered by dependency. Do them **one at a time**, in order.
- Each task has an **Acceptance** block. Run it before starting the next task. Do not batch.
- Commit after each passing task. `feat:` / `chore:` prefix, branch `train-build`.
- Match existing conventions: lowercase explanatory comments justifying *why* (especially around
  sync avoidance, checkpoint recompute, accelerate batch handling), Google-style docstrings with
  `Args:` on public modules. Do not strip existing comments.
- All scripts run from repo root (`config.py` opens `config.yaml` with a relative path).
- Do not read `config.yaml` from anything under `modules/`. Config flows yaml -> `config.py` ->
  kwargs.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, or `moe` in the training
  loop must go through `accelerator.unwrap_model(model)`.
- **Never add a `.item()`, `.tolist()`, `.cpu()`, or boolean mask indexing to the per-step path.**
  Host syncs dominate at this model size. Existing accepted syncs: `m_splits` via `.tolist()`,
  and `TokenTracker.sync()` at log/checkpoint cadence only.
- Line references in this document come from a prior review and may have drifted. Locate by
  symbol name, not by line number.

## What changed in this revision (post code-review)

If you read an earlier copy of this plan, these are the substantive corrections. Everything was
verified against the code, not inferred.

| # | change | task |
|---|---|---|
| 1 | `loop_scale` inits to **0.1, not 0**, and `lambda_ponder` is **warmed from zero**. As originally written the two could deadlock and permanently disable the MoE loop, silently. | 1, 3b |
| 2 | `moe_intermediate_size` **does not exist** and must be added. The MoE reused the dense decoder's `intermediate_size`, so total and active params could not be moved independently. | 2 |
| 3 | The FP8 "pool divisible by 8" constraint **is not real** — nothing enforces it and the routed GEMMs never run in FP8. `num_mlp_experts` freed 21 -> **32** (more total params at zero active compute). | 5 |
| 4 | Per-loop CE must **re-apply the final `RMSNorm`**; `lm_head` never reads the raw residual stream. | 4a |
| 5 | `p_correct` must **beat a free `max(softmax(logits))` baseline** at Gate 5 or Task 4b is reverted. | 4b, Gate 5 |
| 6 | Vocab prune target **65536, not 49152**, gated on **fertility** rather than round-trip identity. | 8 |
| 7 | Budget is **fixed at EUR 100**; `target_tokens` is derived from measured MFU (~16-31B) rather than fixed at 30B. | PART 2 |
| 8 | `p_halt` stickiness resolved explicitly: greedy per-loop gate, **not** cumulative ACT. | 3a |
| 9 | Parameter figures recomputed from the module tree. Config A' is **~332M total / ~174M active**; the previous "333M / 184M active" was wrong in both numbers. | 5 |

The proposal to move attention experts out of the router was dropped — the shared always-on
attention expert in Task 2 covers the same ground at lower risk.

---

# PART 1 - MODEL AND TRAINING CODE

## Task 1 - Loop residual + `loop_scale`

**File:** `modules/model/moe.py`, class `LoopMixtureOfExperts`

Currently `forward_step` replaces `hidden_states` rather than updating it, so there is no
gradient path across a loop boundary that does not pass through a stochastically-selected
expert.

In `__init__`:

```python
# loop starts as a near-no-op and learns how much refinement it wants (LayerScale/ReZero).
# NOT zero-init: at loop_scale == 0 the CE loss has exactly zero gradient wrt p_halt (task 3),
# leaving the ponder term as the only signal reaching the halt head -> it saturates at 1, which
# zeroes the gradient to loop_scale in turn and freezes the whole loop off. 0.1 keeps the CE
# path alive from step 1 while still starting close to a no-op.
self.loop_scale = nn.Parameter(torch.full((1,), 0.1))
```

Note `layer_scalar` in the dense decoder is a *different* mechanism (an init-1 gain on the whole
layer output), so do not treat it as precedent for the init value here.

In `forward_step`, replace the tail (currently `output = self.post_norm(output)` /
`return output`):

```python
delta = self.dropout(self.post_norm(output))
hidden_states = hidden_states + self.loop_scale * delta
return hidden_states, load_balancing_loss
```

Update the caller in the loop over `n_loops` to consume the new return signature.

**Acceptance**
- `bash tests/run_tests.sh tests/test_overfit.py` reaches a **lower** loss in the same step
  count than before the change. If it does not, stop and investigate — nothing downstream is
  worth doing.
- `bash tests/run_tests.sh tests/test_attention_equiv.py` still passes.

---

## Task 2 - Shared (always-on) expert

**File:** `modules/model/moe.py`, class `LoopMixtureOfExperts`

Add one always-on MLP and Self-Attention expert with a skip connection applied to every token, outside the router.

In `__init__`, build a single MLP with the same `hidden_size` and activation as a routed MLP
expert. Name it `self.shared_mlp`. Same for the residual attention expert (`self.shared_attn`),
reusing the existing `SelfAttention` class so it gets the same RoPE cache and varlen path.

In `forward_step`, seed the accumulator with them instead of zeros. Something like this:

```python
# every token gets a dense transform and a full-sequence attention pass regardless of routing.
# stabilises early training and lets routed experts specialise instead of all learning the same
# generic function.
output = self.shared_mlp(hidden_states) + self.shared_attn(
    hidden_states, cu_seqlens, max_seqlen, position_embeddings
)
# ... existing weighted accumulation of routed expert outputs into `output` ...
```

### `moe_intermediate_size` — do this in the same commit

`LoopMixtureOfExperts` currently receives the dense decoder's `intermediate_size` and uses it
for the routed experts too ([transformer.py](modules/model/transformer.py), the
`LoopMixtureOfExperts(...)` construction). There is no separate `moe_intermediate_size` anywhere
in the codebase, despite later tasks referring to one.

That means total and active parameters cannot be moved independently: raising
`intermediate_size` inflates the dense decoder MLP (8 layers, fully active) *and* all routed
experts at once. Every sizing decision from Task 5 onwards needs them separated.

- `config.yaml`: add `moe_intermediate_size` under `model:`.
- `config.py`: add it to `ModelConfig.Params`.
- `TinyMoETransformer.__init__`: new kwarg, defaulting to `intermediate_size` if absent so
  existing checkpoints and tests keep constructing.
- Pass it to `LoopMixtureOfExperts` -> `ParallelSparseMoELayer` and to `self.shared_mlp`.
  `Gemma4TextModel` keeps using `intermediate_size`.

Constraints:
- Neither shared expert is registered in the router pool. They do not appear in `Router`'s
  output dimension and do not appear in `compute_aux_loss`. (The divisibility constraint they
  were previously described as exempt from does not exist — see Task 5.)
- Their row counts are static (`B*S`), unlike the routed grouped GEMM, so they **may** run
  inside `te.autocast`. Do not move `ParallelSparseMoELayer`'s `te.autocast(enabled=False)`.
- These two experts should not dominate every loop. The expert selection plots are the detector:
  if routed weights collapse toward zero, the shared path has swallowed the block.

**Acceptance**
- Parameter count increases by `3 * hidden_size * moe_intermediate_size` (shared MLP, gated)
  plus `~2.5 * hidden_size^2` (shared attention). At the Task 5 config that is +5.3M and +1.5M.
- `test_overfit.py` passes.
- Router output dimension is unchanged from Task 1.
- `moe_intermediate_size` absent from `config.yaml` still constructs (falls back to
  `intermediate_size`), and setting it to a different value changes only the MoE expert sizes —
  verify by diffing per-module parameter counts, not just the total.

---

## Task 3 - Halt head; delete identity expert and `identity_skew`

The invasive task. Do not start until Tasks 1-2 are green and committed.

### 3a. Add the halt head

**File:** `modules/model/moe.py`, class `LoopMixtureOfExperts`

```python
self.halt_proj = nn.Linear(hidden_size, 1, bias=True)
nn.init.zeros_(self.halt_proj.weight)
nn.init.constant_(self.halt_proj.bias, -2.0)   # p_halt ~ 0.12 at init
```

In `forward_step`:

```python
# soft, differentiable halting. p_halt -> 1 means "don't modify me further".
# NOTE: this is a compute-allocation signal, not a confidence score. see Task 4b.
p_halt = torch.sigmoid(self.halt_proj(hidden_states))   # [B, S, 1]
hidden_states = hidden_states + (1.0 - p_halt) * self.loop_scale * delta
return hidden_states, load_balancing_loss, p_halt
```

Collect `p_halt` across all loops (stack or accumulate a running mean — do **not** `.item()`)
and return it from `LoopMixtureOfExperts.forward` alongside the aux loss.

`p_halt` is a **greedy per-loop gate, not a cumulative ACT halting distribution**. It is
recomputed each loop from the current hidden state, so a token can halt at loop 1 and un-halt at
loop 2. This is a deliberate choice (TRAINING_PLAN §2.3) — do not add an accumulator or a
remainder term. The inference exit rule is "stop once `p_halt > tau` at the current loop", and
Gate 5's early-exit degradation curve is what would justify revisiting it.

### 3b. Ponder loss

**File:** `scripts/pretrain.py` (or where `load_balancing_loss` is currently added: `modules/model/mtp.py`)

```python
# mask to real tokens: unmasked, trailing document padding dominates the term
ponder = ((1.0 - p_halt_all) * valid_mask).sum() / valid_mask.sum().clamp(min=1)
loss = loss + lambda_ponder_now * ponder
```

`lambda_ponder = 3e-3` target value. Add to `config.yaml` under model or training and thread it
through `config.py` the same way the aux loss weight is.

**`lambda_ponder` must be warmed from zero.** This is not tuning, it is a correctness
requirement — see the deadlock analysis in TRAINING_PLAN §2.3. With `loop_scale` small, the CE
loss has near-zero gradient wrt `p_halt`, so the ponder term is briefly the *only* signal the
halt head receives, and it is constant-sign. AdamW normalises away its magnitude, so the halt
bias climbs at ~`lr`/step regardless of how small `lambda_ponder` is; if `p_halt` saturates
first, `loop_scale` stops receiving gradient and the loop is permanently disabled — silently,
since the dense decoder keeps the loss curve descending.

```python
# hold at 0 while loop_scale grows, then ramp. driven from the live token count, exactly like
# the router noise anneal above it in the step loop -- no host sync.
tokens = unwrapped_model.token_count
warm = TrainingConfig.ponder_warmup_tokens          # 1_000_000_000
ramp = TrainingConfig.ponder_ramp_tokens            # 1_000_000_000
lambda_ponder_now = TrainingConfig.lambda_ponder * min(1.0, max(0.0, (tokens - warm) / ramp))
```

Add both new keys to `config.yaml` / `config.py`. Place the computation next to the existing
`noise_factor` anneal in the step loop ([pretrain.py](scripts/pretrain.py), the
`TrainingConfig.noise_anneal_tokens > 0` block) — it reads the same sync-free counter.

### 3c. Remove the identity expert

Grep for `identity_expert_index` and `identity_skew`. Expect hits in `moe.py`, `router.py`,
`transformer.py`, `config.py`, `config.yaml`, and possibly `README.md` / `docs/`.

Expert index layout changes from:

```
[ SelfAttn x A | CrossAttn x A | IR x I | identity | MLP x M ]
                                          ^ identity_expert_index = 2A + I
```

to:

```
[ SelfAttn x A | CrossAttn x A | IR x I | MLP x M ]
                                          ^ first_mlp_index = 2A + I
```

Required edits:
- Rename `identity_expert_index` -> `first_mlp_index`. Value is unchanged (`2A + I`), meaning
  changes from "the identity slot" to "the first MLP slot".
- Router pool size drops by 1. Update wherever `num_experts` is computed.
- The remap into `ParallelSparseMoELayer`'s local expert space in `forward_step` changes from
  `idx > identity_expert_index` -> `local = idx - (2A + I) - 1`
  to
  `idx >= first_mlp_index` -> `local = idx - (2A + I)`.
- Non-MLP slots still collapse to `(index 0, weight 0)` so they contribute nothing. Keep the
  **mask multiply** — do not switch to `mask.sum()` or boolean indexing; that forces a device
  sync per expert per step.
- `_ExpertTracking` expert-count and plot labels drop the identity entry.
- `compute_aux_loss` in `router.py` needs no identity exclusion, because identity is gone from
  the pool. Confirm it now iterates only over real experts.
- Delete `identity_skew` entirely, including the `on_loop / n_loops` scaling and the pre-skew /
  post-skew distinction in the aux loss (there is no longer a skew, so the aux loss is computed
  on the router probabilities directly).
- `ModelConfig.Forward` currently contains only `identity_skew`. Removing it leaves an empty
  dict. Either verify `model(**ModelConfig.Forward)` still works with `{}`, or move
  `lambda_ponder` into it.

**Acceptance**
- `grep -rn "identity_skew\|identity_expert_index" .` returns nothing outside docs history.
- Model constructs; router output dim == `2A + I + M`.
- `test_overfit.py` passes.
- A 50-step run logs mean `p_halt` in a sane range (roughly 0.05-0.4). Pinned at 0 or 1 is a bug.
- **Deadlock check.** Run 2000 steps with `ponder_warmup_tokens` forced to 0 and confirm the
  failure mode is reproducible: `p_halt` climbs toward 1 and `loop_scale` stalls. Then restore
  the warmup and confirm `loop_scale` moves off 0.1 while `p_halt` stays in range. If the
  failure mode does *not* reproduce, the halt head is not wired into the loss — check that
  before concluding the warmup is unnecessary.

---

## Task 4 - Per-loop CE supervision and correctness head

Rationale in TRAINING_PLAN §2.4. Two independent problems, both required before `p_halt` or any
early-exit behaviour means anything. Do 4a and 4b as separate commits.

### 4a. Per-loop cross-entropy

**Problem:** with CE applied only at the final loop, intermediate hidden states are optimised
solely as inputs to the next loop, not as things `lm_head` can read. Thresholding `p_halt` and
exiting at loop 1 then reads out from an interface that was never trained. Controlled evidence
for this comes from 44M/129M looped models — this scale.

**Files:** `modules/model/transformer.py`, `modules/model/mtp.py`

Apply `main_lm_head` at each loop, not just the last, with ascending weights:

```yaml
# config.yaml
model:
  loop_ce_weights: [0.2, 0.3, 1.0]   # must have length == n_loops
```

Implementation constraints:
- **Apply the final `RMSNorm` at every loop, not just the last.** `lm_head` reads
  `self.norm(x)`, never the raw residual stream (`TinyMoETransformer.forward`). Feeding
  un-normalised intermediate states into the head gives meaningless per-loop losses, and the
  symptom — per-loop CE flat across loops — is identical to the "hidden states not threaded
  through" bug in the monitoring table, so it will be misdiagnosed. Either apply `self.norm`
  per loop inside the loss, or normalise each collected hidden state before returning it.
- Reuse the existing **chunked, checkpointed** CE path in `compute_mtp_loss`. Do not
  materialise `[T, vocab]` logits per loop — that is a 3x activation blowup and the reason the
  chunked path exists. `CE_CHUNK_SIZE = 2048` stays.
- Training-mode forward still returns hidden states, not logits (`return_hidden=True` plus
  `delayed_mtp_loss(True)`). Per-loop CE means returning a **list/stack of per-loop hidden
  states**, with the LM head applied inside the loss function as before. Any new call site must
  pass `main_lm_head=` or the activation peak silently doubles.
- MTP heads apply to the **final loop only**. Do not run MTP per loop.
- Assert `len(loop_ce_weights) == n_loops` at construction.

Expect ~3x LM-head compute and a measurable tokens/sec drop. That is the intended cost.

### 4b. Correctness head

**Problem:** `p_halt` is trained by the LM loss (more refinement where refinement helps) and the
ponder cost (fewer loops). It learns *"does more computation change my prediction?"* — not *"is
my prediction right?"* A confidently hallucinated fact is stable under refinement, so it halts
at loop 0 and reads as maximum certainty. Utility calibration is not correctness calibration.

Do **not** try to fix this by reweighting `p_halt`. Add a second, separate head.

**File:** `modules/model/transformer.py`

```python
# separate from p_halt on purpose. p_halt answers "is more compute useful here",
# this answers "is this prediction correct". they come apart on confident hallucinations.
self.correct_proj = nn.Linear(hidden_size, 1, bias=True)
nn.init.zeros_(self.correct_proj.weight)
nn.init.constant_(self.correct_proj.bias, 0.0)
```

Target is free at every pretraining token — no labels, no extra forward pass:

```python
# inside the chunked CE, where logits for the chunk already exist
with torch.no_grad():
    is_correct = (logits_chunk.argmax(-1) == labels_chunk).float()
conf_loss = F.binary_cross_entropy_with_logits(
    correct_logit_chunk, is_correct, reduction="none"
)
# mask out label == -100 positions, then mean
```

Add `loss = loss + lambda_conf * conf_loss`, `lambda_conf = 0.05`.

Constraints:
- Computed on the **final loop's** hidden states only.
- `is_correct` must be under `no_grad` — this is a target, not a differentiable path. Gradient
  leaking into the LM head through it will quietly degrade the LM loss.
- Respect the `-100` label mask; padding positions must not contribute.
- Reuse the existing chunk loop. Do not add a separate pass over the sequence.

**Semantics to preserve downstream — these are two different signals:**

| signal | question | used for |
|---|---|---|
| `p_halt` | is more latent compute useful here? | inference early exit |
| `p_correct` | is this prediction right? | abstention, ECE, truthfulness |

Report ECE on `p_correct`, never on `p_halt`.

### This head is provisional — it has to beat a free baseline

`max(softmax(logits))` is already a strong per-token correctness predictor and costs nothing:
no head, no loss term, no `lambda_conf` to tune, no gradient-leak failure mode. A learned
`p_correct` is only worth carrying if it is *better*.

Log both from the start so the comparison is available at Gate 5:

```python
# free baseline, same chunk, same mask. no parameters, no loss term.
with torch.no_grad():
    p_max = logits_chunk.softmax(-1).max(-1).values
```

Gate 5 measures ECE and abstention AUROC for `p_correct` and for `p_max` on the same held-out
slice. **If `p_correct` does not beat `p_max` on both, revert Task 4b** — drop the head, the
loss term, and `lambda_conf`, and use `p_max` as the abstention signal for all of PART 6. The
§6 thesis needs *a* calibrated signal, not specifically a learned one, and it survives that
substitution intact. Reverting costs one commit before renting; discovering it afterwards costs
a rerun.

**Acceptance (4a)**
- `len(loop_ce_weights) == n_loops` asserted at construction.
- Loss at loop 0 > loop 1 > loop 2 after a few hundred steps (log all three). If they are equal,
  the per-loop hidden states are not being threaded through correctly.
- Peak memory increases by less than 20%. A larger jump means logits are being materialised.
- `test_overfit.py` passes.

**Acceptance (4b)**
- Mean `p_correct` after a short run is within ~0.1 of the measured top-1 accuracy on the same
  batch. Wildly off probably means the mask or the target is wrong. Note this is a smoke test,
  not a calibration test — a bias-only constant predictor passes it. The real check is Gate 5.
- Gradient check: LM loss with `lambda_conf = 0` matches LM loss with `lambda_conf = 0.05`
  within noise at step 1 (confirms no gradient leak through `is_correct`).
- Tokens/sec regression from 4b alone is under 3%.
- `p_max` is logged alongside `p_correct` and both are available to `eval_calibration.py`.

---

## Task 5 - Config A'

**File:** `config.yaml`

This is a **diff against the existing `model:` block**, not a replacement for it. `config.py`
reads `num_ir_experts`, `ir_dim`, `dropout`, `per_layer_embeddings_size`, `max_seq_length` and
`mtp_num_extra_tokens`; dropping them is a `KeyError` at import.

```yaml
model:
  hidden_size: 768               # was 512
  intermediate_size: 2304        # was 2048 -- dense decoder MLP only
  moe_intermediate_size: 2304    # new in task 2 -- routed + shared experts
  num_layers: 8                  # was 5
  num_attention_heads: 12        # was 8
  head_dim: 64                   # unchanged
  num_mlp_experts: 32            # was 36
  num_attn_experts: 1            # unchanged
  num_ir_entries: 8192           # was 16384
  n_loops: 3                     # was 4
  top_k: 2                       # unchanged
  vocab_size: 65536              # was 129280, set by task 8
  lm_head_factor: 4              # unchanged
  loop_ce_weights: [0.2, 0.3, 1.0]
  lambda_ponder: 3e-3
  lambda_conf: 0.05
  # identity_skew: DELETED by task 3
```

```yaml
training:
  ponder_warmup_tokens: 1000000000
  ponder_ramp_tokens: 1000000000
```

### The FP8 divisibility constraint does not exist — delete it

A prior review resolved this. Verified against the code:

- Nothing enforces it. `grep -rn "divisible" .` finds only a comment on
  [config.yaml:13](config.yaml#L13) and a comment in `ParallelSparseMoELayer.forward`.
- The routed MLP GEMMs **never run in FP8**: `ParallelSparseMoELayer.forward` wraps them in
  `te.autocast(enabled=False)`. The genuine constraint — NVFP4/FP8 needing each group's row
  count divisible by 16, which dynamic routing cannot guarantee — is exactly *why* autocast is
  disabled there. A rule about the pool *size* was never implied by it.

So `num_mlp_experts` is a free integer. As part of this task:

- Delete the `# due to fp8 requirement...` comment from `config.yaml`.
- Delete the "should be divisible by 8 for FP8 GEMMs" line from the Config section of
  `CLAUDE.md`, and replace it with the real constraint (per-group rows, hence
  `te.autocast(enabled=False)`), which `CLAUDE.md` already documents correctly elsewhere under
  Model invariants. Leaving both statements in place is what made this ambiguous.

`num_mlp_experts: 32` is chosen because routed experts add total parameters at **zero** active
compute — the limit is per-expert data, not FLOPs. At M=32 / pool=35 / top_k=2 / ~25B tokens
each MLP expert sees ~1.4B tokens. Going further (M=52 -> ~0.9B) buys undertrained experts.

Constraints to assert at construction:

- `vocab_size % lm_head_factor == 0` -> `65536 / 4 = 16384`. OK.
- `vocab_size % (lm_head_factor * 2) == 0` for the MTP head -> `65536 / 8 = 8192`. OK.
- `hidden_size % lm_head_factor == 0` -> `768 / 4 = 192`. OK.
- `(hidden_size // 2) % (lm_head_factor * 2) == 0` for the MTP head -> `384 / 8 = 48`. OK.
- `len(loop_ce_weights) == n_loops` -> `3 == 3`. OK.
- `vocab_size <= 65536` so token ids fit uint16 (task 9's `train.bin` dtype).
- `mtp_num_extra_tokens <= num_mtp_tokens` (dataset separator budget), else MTP is supervised
  across document boundaries.

Add a parameter-count print and hard assertions on all of the above at model construction, so
these are startup failures rather than silent misconfigurations. Print **total and active**
separately, and print the per-token training FLOP estimate — §5's budget table is keyed to it
and goes stale silently otherwise.

Hit any future parameter target with `moe_intermediate_size` (total + active) or
`num_mlp_experts` (total only). `intermediate_size` moves the dense decoder and should be
treated as part of the body shape, not as a sizing knob.

**Acceptance**
- Model constructs, asserts pass, printed counts are **~332M total / ~174M active**
  (~104M active excluding embedding tables) at `vocab_size: 65536`.
- Printed FLOP estimate is ~357 MFLOP/token forward.
- `test_overfit.py` passes at the new size.
- `grep -rn "divisible by 8" .` returns nothing.

---

## Task 6 - Token-count break and schedule fixes

**File:** `scripts/pretrain.py`

Currently `total_steps` anchors the cosine schedule to `target_tokens` but the loop runs
`num_epochs` epochs regardless, so a dataset larger than `target_tokens` trains past the
schedule at `eta_min`.

```python
# stop at the token budget the LR schedule is anchored to.
# read from the sync-free counter at LOG_INTERVAL cadence -- do not add an .item() to the
# step path.
if unwrapped.token_count >= TrainingConfig.target_tokens:
    break
```

Place the check inside the existing `LOG_INTERVAL` throttle block, not per step.

Also:
- `config.yaml`: `num_epochs: 1`
- Checkpoint interval 5000 -> **1500** steps (hardcoded in `pretrain.py`). The run is
  interruptible.
- `LOG_INTERVAL` stays 10.

**Acceptance**
- A short run with `target_tokens` set very low exits at the right token count.
- Interrupt and resume works: the LR is re-anchored by tokens
  (`resume_token_count // tokens_per_step`), not by saved step.

---

## Task 7 - Instrumentation

**File:** `scripts/pretrain.py`, inside the existing `LOG_INTERVAL` throttle

Add to the log line: `loop_scale`, mean `p_halt`, mean `p_correct`, mean `p_max` (the free
baseline of Task 4b), batch top-1 accuracy, per-loop CE (all `n_loops` values), aux loss,
ponder loss, conf loss, and the current `lambda_ponder` (it ramps — a constant value in the log
means the warmup is not wired to the token counter).

```python
unwrapped = accelerator.unwrap_model(model)
loop_scale = unwrapped.moe.loop_scale.item()
```

All reads go through `accelerator.unwrap_model(model)` — the DDP wrapper has none of these
attributes.

`p_correct` and top-1 accuracy are logged **as a pair**. Their divergence is the calibration
signal and the thing to watch during the run.

**Acceptance**
- A 100-step run prints all of the above without a per-step sync (tokens/sec should not regress
  beyond the Task 4a cost).

---

## Task 8 - Vocab prune to 65536

Do this **before** tokenizing anything.

**Target is 65536, not 49152.** The load-bearing reason to prune is disk, not parameters: 30B
tokens as uint32 is 120 GB on a 120 GB instance with no room for source shards. *Any*
`vocab_size <= 65536` fits uint16 and halves the dataset to 60 GB, which is the entire disk
argument. Dropping to 49152 instead buys ~21M parameters in exchange for a substantially more
aggressive merge-table surgery and more fertility damage (below). Not worth it.

New script: `scripts/prune_vocab.py`

1. Sample ~2 GB of text from the **final mix** at training ratios (web + code + math + wiki).
   Web-only frequency counts badly underweight brackets, braces, indentation, and LaTeX.
2. Count token frequencies under the current DeepSeek tokenizer.
3. Keep top-N plus **all** special tokens and byte-fallback tokens, to reach exactly 65536.
   Dropping a token means dropping its merge rule; keep merges closed under prefix, so a kept
   merge never depends on a dropped one.
4. Remap ids, write the new tokenizer to `ckpts/pretrained/<name>-65536/`.
5. Emit an `old_id -> new_id` lookup table alongside it.

**Preserve these two quirks** (CLAUDE.md flags them):
- `pad_token_id == eos_token_id`
- BOS is id 0. If the remap moves BOS off zero, pin it back. `Gemma4TextModel.embed_tokens`
  deliberately has **no `padding_idx`** — setting one froze BOS at zero.

### Gate on fertility, not on round-trip identity

A round-trip test is the wrong instrument here and will pass almost regardless of the damage
done: removing merges does not corrupt decoding, because byte fallback still reconstructs the
surface string exactly. What it does is make the *same text cost more tokens* — text that used
to consume one token now consumes two or three.

That directly shrinks the effective data budget (§5 buys a token count, not a byte count) and
eats the FLOP saving the prune was partly justified by. Measure it explicitly.

**Acceptance**
- `len(tokenizer) == 65536`
- `tokenizer.pad_token_id == tokenizer.eos_token_id`
- `tokenizer.bos_token_id == 0`
- Max token id `< 65536`, i.e. ids fit uint16 exactly.
- **Fertility regression under 3%.** Measure tokens-per-byte before and after on a held-out
  ~200 MB sample drawn at the §4.2 phase-1 mix ratios, and report it per source — code and math
  will regress worse than web, and those are the sources where it matters. Above 3%, keep more
  tokens (the vocab only has to be `<= 65536`, it does not have to be a round number).
- Record the measured fertility in `manifest.json` and **scale the §5 token target by it** —
  a 3% fertility regression is 3% fewer documents seen for the same token budget.
- Encode/decode round-trip over 5000 held-out documents is byte-identical. Expected to pass
  trivially; it catches a broken remap, not a bad prune.

---

## Task 9 - mmap dataset

**File:** `modules/data/dataset.py`

Replace the streaming-parquet + on-the-fly-tokenization path with a flat-file reader.

### Files consumed

- `train.bin` — flat uint16 token stream
- `train.idx` — **document start offsets** (uint64). Without stored boundaries you cannot
  reconstruct `document_ids`, and block-diagonal varlen attention silently becomes wrong.

Two pairs are produced by Task 11: `phase1.{bin,idx}` and `phase2.{bin,idx}`. Which pair is
loaded comes from config.

### Invariants that MUST survive the rewrite

- Still an `IterableDataset` yielding **fully assembled batches**. `DataLoader(batch_size=None)`
  stays.
- Still emits `document_ids [B, S]`. The trainer still builds `cu_seqlens` **in-thread** via
  `cu_seqlens_from_doc_ids`.
- **`cu_seqlens` never goes in the batch dict.** It is ragged (`dim0 = num_segments + 1`) and
  accelerate's `split_batches` truncates dim 0 to the batch size, silently corrupting
  segmentation.
- `max_seqlen` still passed as `S` (a valid upper bound), not the true max — avoids an `.item()`
  sync every step.
- Packing unchanged in shape: documents concatenated into `max_length` sequences, split across
  sequence boundaries when they do not fit, each followed by `EOS + (num_mtp_tokens - 1)` pads.
  Trailing padding becomes length-1 attention segments.
- Labels `-100` everywhere except the interior of each document block plus the terminating EOS.
- Batches still carry `[B]`-shaped bookkeeping tensors so accelerate splits them like
  `input_ids`.
- `NUM_DATA_WORKERS = 4`. Worker `w` takes sequences `[w::num_workers]`.

### What simplifies

Resume state collapses from `(file_idx, record_idx, shard_token_count)` plus per-worker
positions to a **single global sequence offset**. `snapshot_resume_positions()` becomes trivial.

`file_order`, `build_legacy_order`, and `max_tokens_per_shard` become dead — remove them and
their checkpoint fields, but keep `load_checkpoint`'s `.get(..., default)` pattern so old
checkpoints still load. Add `global_offset` to `save_checkpoint`'s signature **with a default**.

### Note

Packing is now just slicing a contiguous token stream at `max_length` and emitting
`document_ids` from the stored boundary offsets. Tokenizer nondeterminism leaves the resume path
entirely.

**Acceptance**
- `test_attention_equiv.py` passes — this is the test that catches a broken `document_ids`.
- A synthetic `train.bin` with known document boundaries produces `cu_seqlens` matching them
  exactly.
- Label mask spot-check: `-100` on pads, real ids on document interiors + terminating EOS.
- Interrupt at step N, resume, confirm the next sequence consumed is N*batch_size (no gap, no
  repeat).

---

## Task 10 - FP8 wiring (do not enable locally)

**File:** `scripts/pretrain.py`

`fp8_recipe = DelayedScaling(...)` already exists and is unused. Gate it on an env var so the
same code runs on the 5090 (off) and the H100 (on):

```python
USE_LOW_PRECISION = os.environ.get("USE_FP8", "0") == "1"
chosen_recipe = fp8_recipe if USE_LOW_PRECISION else None
```

Leave `te.autocast(enabled=False)` around `ParallelSparseMoELayer`'s GEMMs. NVFP4/FP8 needs each
group's row count divisible by 16 and dynamic routing cannot guarantee that; sparsity wins over
precision there.

Gradient checkpointing stays **off**. If ever enabled, use
`from transformer_engine.pytorch import checkpoint`, never `torch.utils.checkpoint` — the latter
breaks quantized layers.

**Acceptance**
- `USE_FP8=0` reproduces current BF16 behaviour bit-for-bit.
- (H100 only, later) `USE_FP8=1` runs and TE emits no fallback warnings.

---

## Task 11 - Data prep script

New script: `scripts/prepare_data.py`. Runs on the rented box, not locally.

### Behaviour

For each source, stream shard -> tokenize -> append to the target `.bin` -> **delete the shard**.
Peak disk stays at bin size plus a few GB of working set, not bin + full source.

Use `huggingface_hub.hf_hub_download` on individual files. `list_repo_files()` first, filter,
then pull N files. **Do not** use `load_dataset` for the bulk sources — it gives no control over
how many GB land on disk.

Tokenize with `num_proc = os.cpu_count()`, batched.

### Sources and slices

| dataset | subset / slice | target tokens |
|---|---|---|
| `HuggingFaceFW/fineweb-edu` | `data/CC-MAIN-2025-26/`, partial | 15B |
| `mlfoundations/dclm-baseline-1.0` | handful of global shards | 2.5B |
| `HuggingFaceFW/finepdfs-edu` | `eng_Latn`, few files | 2.2B |
| `nvidia/Nemotron-CC-Code-v1` | few shards | 4B |
| `nvidia/Nemotron-CC-Math-v1` | config `4plus`, few shards | 2B |
| `wikimedia/wikipedia` | `20231101.en`, subset of files | 1.1B |
| `HuggingFaceTB/smoltalk2` | non-reasoning splits, whole | 0.7B |

Do **not** substitute `HuggingFaceTB/stack-edu` or `nvidia/Nemotron-Pretraining-Code-v1/v2` —
both ship identifiers requiring reconstruction from an external store, not text.

### Output

Two bin/idx pairs, with sources **interleaved at the mix ratios during writing** so training is
a straight sequential read with no online sampling:

`phase1.{bin,idx}` — 25.5B tokens: 55% FineWeb-Edu, 12% Nemotron-CC-Code, 10% DCLM,
7% FinePDFs-Edu, 3% Nemotron-CC-Math, 3% Wikipedia.

`phase2.{bin,idx}` — 4.5B tokens: 30% Nemotron-CC-Math, 22% Nemotron-CC-Code,
15% FineWeb-Edu, 15% SmolTalk2, 10% FinePDFs-Edu, 8% Wikipedia.

**Hold out** the smoltalk2 rows used here — record their ids in the manifest so PART 6 SFT does
not train on data already seen in pretraining.

Shuffle at document granularity within each phase. Write a `manifest.json` recording exact repo
ids, revisions, filenames, per-source token counts, and the smoltalk2 holdout ids — this is the
reproducibility artifact, in place of uploading the dataset anywhere.

**Acceptance**
- Realised per-source token counts within 2% of target.
- `train.idx` is monotonically increasing and its last entry equals `len(train.bin)`.
- `phase1.bin` dtype is uint16 and max value < 65536.
- Peak disk during the run stays under ~70 GB.
- `manifest.json` records the measured per-source fertility from Task 8.

---

# PART 2 - LOCAL VALIDATION (5090)

Do not rent anything until all five gates pass.

**Gate 1** — `bash tests/run_env_check.sh`

**Gate 2** — `bash tests/run_tests.sh tests/test_attention_equiv.py`

**Gate 3** — `bash tests/run_tests.sh tests/test_overfit.py` drives a small batch to near-zero
loss at config A'.

**Gate 4** — ~500M-token run on a local slice at the new hidden size. Measure and record:

| metric | pass condition |
|---|---|
| tokens/sec | recorded — this is the input to the budget decision |
| MFU | recorded |
| `loop_scale` | **moves off its 0.1 init.** Stuck at 0.1 means the ponder deadlock (Task 3b) |
| mean `p_halt` | roughly 0.05-0.4, not pinned. Climbing monotonically toward 1 is the deadlock |
| `lambda_ponder` | visibly ramping, not constant |
| per-loop CE | strictly decreasing across loops |
| expert selection plots | no collapse to a handful of MLP slots; routed weights not swallowed by the shared experts |
| loss curve | monotone decreasing, no spikes |

Then **interrupt and restart** to prove resume works against the mmap dataset before it costs
money.

**Gate 5 — calibration sanity.** New script: `scripts/eval_calibration.py`, run on a held-out
slice.

- **ECE of `p_correct` vs actual top-1 correctness < 0.15.** This is the load-bearing number for
  the whole abstention thesis. If `p_correct` is uninformative at 500M tokens it will not
  magically become informative at 25B — debug Task 4b before renting.
- **`p_correct` vs the `p_max` baseline.** Compute ECE *and* abstention AUROC for both
  `p_correct` and `max(softmax(logits))` on the same slice, and report them side by side.
  `p_max` is free — no head, no loss term, no hyperparameter. **If `p_correct` does not beat it
  on both metrics, revert Task 4b** and use `p_max` as the abstention signal throughout PART 6.
  This is a real decision point, not a formality: the learned head is only worth its loss term
  and its gradient-leak failure mode if it is measurably better.
- **Early-exit degradation curve.** Force exit at loop 1, 2, 3; record perplexity for each.
  Expect monotone improvement. If exiting at loop 1 is catastrophic rather than merely worse,
  Task 4a is not working. This curve is also the only evidence that would justify revisiting the
  greedy (non-cumulative) `p_halt` decision in Task 3a.
- **`p_halt` / `p_correct` correlation.** Record it. Expect it to be weak — that is the *point*
  of having two heads. A correlation above ~0.8 means one head has collapsed into the other and
  something is wired wrong.

### Budget decision (do this before renting)

**The budget is fixed at EUR 100.** The token count is what gets decided here, not the price.

At ~EUR 2/hr that is 50 instance-hours; reserve ~3 h for data prep, the smoke test, and
checkpoint extraction, leaving **~47 training hours**. Scale the measured 5090 tokens/sec by the
expected H100 ratio to get an MFU estimate, then read off:

| MFU vs BF16 peak | tokens in 47 h | tokens / total param |
|---|---|---|
| 10% | 16B | 47 |
| 12.5% | 20B | 59 |
| 15% | 23B | 71 |
| 17.5% | 27B | 82 |
| 20% | 31B | 94 |
| 25% | 39B | 118 |

Assumes ~1071 MFLOP/token to train config A' (Task 5 prints this — use the printed number if it
has drifted) and ~990 TFLOPS H100 SXM BF16 dense peak. The prepared data caps the run at 30B, so
above ~18% MFU the run is data-limited, not budget-limited.

Task 4a costs real throughput. Measure MFU **after** it lands, not before.

Then:

1. Round the table result **down**, and scale it by the measured fertility regression from
   Task 8 (a 3% regression means 3% fewer documents for the same token count).
2. Set `target_tokens` to that number **now**, before renting. The cosine schedule is anchored to
   it; stopping early against a longer schedule leaves the LR high and wastes the phase-2 anneal,
   which is the single most expensive avoidable mistake in this plan.
3. `target_tokens` describes the **combined** phase 1 + phase 2 run, not either phase alone.

If the number comes out below ~16B, the honest options are a cheaper instance or a smaller
`moe_intermediate_size` — **not** a longer run. The budget is the constraint.

---

# PART 3 - REPO PREP BEFORE RENTING

`.gitignore` swallows `*.json`, `*.cmd`, `ckpts/`, `venv/`, `tests/`, and `env_init`.

This means `data_config.json`, the entire test suite, and the environment script **will not be
on the rented box after a clone.** Fix before renting:

1. Un-ignore `tests/` (recommended — the gates in Part 2 need to run on the box too), or write a
   `deploy.sh` that scp's `tests/` and `data_config.json` after clone.
2. Write `vast_init` (committed, unlike `env_init`):
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - **no** WSL CUDA 12.9 paths, **no** `/mnt/d/AI/...` hardcode, **no** venv activation (the
     NGC container has the environment)
3. Fix `tests/run_tests.sh` — it hardcodes `cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm` and
   `source env_init`. Make the repo root and init script overridable by env var.
4. Commit `manifest.json` (Task 11) and `scripts/prepare_data.py`.

---

# PART 4 - VAST.AI RUNBOOK

**Total spend for this part is capped at EUR 100.** Track it: at EUR 2/hr the run has ~47
training hours after prep, smoke test and extraction. Steps 3-4 are billed too.

## Step 1 - Rent

- 1x **H100 SXM or NVL, 80GB**
- **Interruptible** (~half price; the per-step checkpoint/resume path exists to exploit this)
- **120 GB disk** — set at creation, **cannot be grown later**
- Image: `nvcr.io/nvidia/pytorch:25.xx-py3`
- Confirm the hourly rate before starting. Above ~EUR 2.2/hr, re-read the budget table in
  PART 2 and cut `target_tokens` accordingly *before* the run, not during it.

Do **not** `pip install -r requirements.txt` wholesale. The TE and flash-attn wheels there are
built for **sm120**; H100 is **sm90**. The NGC image ships both prebuilt. Layer only the
non-CUDA deps.

## Step 2 - Verify environment (before downloading 47 GB)

```bash
python -c "import torch, transformer_engine.pytorch, flash_attn; print(torch.cuda.get_device_capability())"
bash tests/run_env_check.sh
```

Expect `(9, 0)`. If TE is broken, discover it at minute two, not minute ninety.

## Step 3 - Data prep

```bash
python scripts/prepare_data.py
```

Put this in the instance's **on-start script** so it runs unattended. Expect 40 min - 2 hours,
likely download-bound rather than tokenizer-bound.

Verify against Task 11's acceptance criteria before proceeding.

## Step 4 - Smoke test

```bash
USE_FP8=1 python scripts/pretrain.py   # kill after ~200 steps
```

Confirm:
- `dry_run` asserts a finite loss on the packed path
- FP8 is actually active (TE warns on silent fallback)
- tokens/sec is within range of the Part 2 extrapolation
- peak memory has headroom

**If tokens/sec differs from the Part 2 extrapolation by more than ~20%, redo the budget
arithmetic before starting phase 1.** `target_tokens` anchors the cosine schedule and cannot be
changed mid-run without wasting the anneal — this is the last cheap moment to correct it.

Then **delete the checkpoint** and restart clean.

## Step 5 - Phase 1

85% of the committed budget, on `phase1.{bin,idx}`. LR: linear warmup -> cosine to `0.1 * lr`.
Router noise anneals over `noise_anneal_tokens` from the live token count.

## Step 6 - Phase 2 anneal

15% of budget, on `phase2.{bin,idx}`, LR -> ~0. Resume from the phase 1 checkpoint.

**Resume re-anchors the schedule by token count**, so `target_tokens` and the schedule config
must describe the **combined** run, not each phase separately.

## Step 7 - Extraction

`scp` checkpoints down as they are written, or at minimum after phase 1. A reclaimed and
reallocated instance takes its disk with it.

Note: "latest checkpoint" resolves by **newest mtime, not highest step**. `scp` can rewrite
mtimes — verify resume picks the right file after any round-trip.

Before releasing the instance, re-run `scripts/eval_calibration.py` (Gate 5) on the final
checkpoint and record the numbers. Re-renting to compute them later is wasteful.

---

# PART 5 - MONITORING

**Every `LOG_INTERVAL`:** loss, tokens/sec, peak mem, `loop_scale`, current `lambda_ponder`,
mean `p_halt`, mean `p_correct`, mean `p_max`, batch top-1 accuracy, per-loop CE (all
`n_loops`), aux loss, ponder loss, conf loss.

**Every checkpoint:** `ckpts/training/expert_selection_*.png` from `_ExpertTracking`.

| symptom | likely cause |
|---|---|
| `loop_scale` stuck at its 0.1 init **and** `p_halt` climbing | the ponder deadlock — `lambda_ponder` warmup not wired to the token counter (Task 3b) |
| `loop_scale` stuck at 0.1, `p_halt` in range | residual wiring wrong (Task 1) |
| `p_halt` saturated at 1 | `lambda_ponder` too high, or ramping too early |
| `p_halt` pinned at 0 | halt head not receiving gradient |
| `lambda_ponder` constant in the log | warmup reading a stale/absent token count |
| per-loop CE flat across loops | per-loop hidden states not threaded through, **or the final `RMSNorm` not applied per loop** (Task 4a) |
| per-loop CE huge at loops 0-1, normal at the last | final `RMSNorm` not applied to intermediate states (Task 4a) |
| `p_correct` far from top-1 accuracy | label mask or target wrong (Task 4b) |
| `p_correct` collapses to a constant | `lambda_conf` too low, or gradient leaking through `is_correct` |
| `p_correct` tracks `p_max` exactly | the head has learned nothing beyond the free baseline — flag for the Gate 5 revert decision |
| routed expert weights near zero, loss still falling | the shared MLP/attention experts (Task 2) have swallowed the block |
| expert selection collapses to a few MLP slots | aux loss weight too low |
| tokens/sec drops after a code change | a host sync entered the step path |
| loss spikes on resume | schedule re-anchoring or `global_offset` wrong |

`_ExpertTracking` guards against activation-checkpoint recompute double counting via
`begin_forward(expected_updates)` and samples every 8th forward. If expert counts look wrong
after any change to the loop structure, check `expected_updates` matches the new `n_loops`.

---

# PART 6 - POST-TRAINING

Rationale in TRAINING_PLAN §6. **The target is calibrated abstention, not chain-of-thought
reasoning.** Do not evaluate this model primarily on GSM8K/MATH; those will not move at 332M
total / 174M active.

## Task 12 - SFT

New script: `scripts/sft.py`. Reuses the model and packing path; swaps the data source and adds
loss masking over prompt tokens.

| dataset | role |
|---|---|
| `HuggingFaceTB/smoltalk2` (no-think splits) | general instruction following |
| `rajpurkar/squad_v2` | **primary abstention supervision** — the unanswerable questions are the point |
| `allenai/tulu-3-sft-personas-math` | short worked solutions |
| `openai/gsm8k` (socratic) | short numbered steps |
| `HuggingFaceH4/no_robots` | human-written; tone and refusal style |

Requirements:
- **Exclude the smoltalk2 holdout ids** recorded in `manifest.json` (Task 11).
- Mask loss over prompt/system tokens; train on completions only.
- **Do not** use long-CoT trace datasets (KIMI-K2.5, Claude, Fable 5 sets in README.md). Small
  Model Learnability Gap — at this scale they teach fluent filler before a wrong answer. They
  also generally carry provider terms restricting training of competing models.
- Keep `p_correct` and `p_halt` supervision active during SFT. The correctness target is still
  free. **If Gate 5 reverted Task 4b**, there is no `p_correct` head — substitute `p_max`
  everywhere below; nothing else in this task or in Task 13 changes.

**Acceptance**
- SQuAD v2 abstention: precision and recall on the unanswerable split, both reported.
- ECE of the abstention signal (`p_correct`, or `p_max` if 4b was reverted) **does not degrade**
  relative to the pretrained checkpoint.

## Task 13 - Self-labelled calibration set

The one dataset worth building rather than downloading, because it requires this model.

New script: `scripts/build_calibration_set.py`

1. Sample the Task 12 checkpoint N=16 times at temperature 0.8 on short-answer QA
   (`mandarjoshi/trivia_qa`, `google-research-datasets/nq_open`, `rajpurkar/squad_v2`).
2. Label each question by empirical pass rate against the reference answer (normalised exact
   match / alias match).
3. Rewrite targets by pass rate: `> 0.8` -> the answer; `< 0.2` -> an abstention; in between ->
   a hedged answer. Use a small fixed set of abstention and hedge phrasings, not free text.
4. Hold out 10% before rewriting — this is the calibration eval set.
5. Second SFT pass on the rewritten data.

**Acceptance**
- Abstention rate on the held-out low-pass-rate bucket > 60%.
- Abstention rate on the held-out high-pass-rate bucket < 10% (i.e. it is not just refusing
  everything — this is the metric that catches the degenerate solution).
- ECE of the abstention signal improves relative to Task 12.

## Task 14 - RL (deferred; gate before starting)

**Do not start.** Recorded here so the decision is documented rather than revisited from scratch.

Evidence at this scale is consistent and negative. A 135M single-GPU study ran RLVR on GSM8K:
SFT base 24/1319 (1.82%), GRPO at 192-token completions **fell** to 21/1319, at 320 tokens to
16/1319. On Qwen2.5-0.5B base, even with a format reward the reward stayed below 0.1 after 300
steps with no upward trend. Mechanism: under a 0/1 reward, a base model that cannot sample
correct solutions produces no gradient signal. RLVR amplifies the base distribution; at 174M
active / ~25B tokens there is little to amplify.

**Gate:** pass@8 on the target task with the Task 13 checkpoint. **Below ~15%, do not proceed** —
the budget will be spent confirming the null result.

If the gate ever passes, note that vanilla GRPO is architecturally mismatched to a looped model:
it assigns credit to output tokens while the computation is latent. Read LoopRPT
(arXiv 2603.19714) and the RLTT line of work first — they assign reward to per-loop latent
states instead, and report improved gate calibration (more early exits, final-step dominance
maintained) as a side effect. That is directly relevant to the `p_halt` machinery here.

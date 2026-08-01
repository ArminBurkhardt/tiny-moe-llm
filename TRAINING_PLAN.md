# TRAINING_PLAN.md

Decisions and rationale for the first real pretraining run of `tiny-moe-llm`.
Implementation steps live in [EXECUTION_PLAN.md](EXECUTION_PLAN.md).

Status: supersedes the previous revision. Changes: config A adopted as primary, identity
expert replaced by halt head, dataset mix rebuilt on 2025/2026 corpora, HF upload step
removed, token budget raised to 30B, per-loop CE supervision and a separate correctness head
added (§2.4), post-training target redefined from chain-of-thought reasoning to calibrated
abstention (§6).

Revision 3 (post code-review) changes:

- Parameter figures recomputed from the module tree; the previous "333M / 184M active" was
  wrong. Config A is restated as **A'** with corrected numbers (§1).
- The FP8 "pool divisible by 8" constraint **does not exist** — no code enforces it and the
  routed GEMMs never run in FP8. `num_mlp_experts` freed to 32 (§1).
- `moe_intermediate_size` added as a real config key. It did not exist; the MoE reused the
  dense decoder's `intermediate_size`, so total and active params could not be separated (§1).
- Vocab prune target moved 49152 -> **65536**, and the acceptance metric changed from
  round-trip identity to **fertility** (§1).
- Ponder loss gets a warmup and `loop_scale` a nonzero init — as originally specified the two
  could deadlock and permanently disable the MoE loop (§2.3).
- `p_halt` stickiness resolved explicitly (§2.3); per-loop CE must re-apply the final norm
  (§2.4); `p_correct` must beat a max-softmax-probability baseline or Task 4b is dropped (§2.4).
- Budget is now a **fixed EUR 100**; the token count is derived from measured MFU rather than
  fixed at 30B (§5).

---

## 1. Target config

Config **A'** (768x8). Chosen over the 492M variant: ~25% cheaper per token, and at a fixed
euro budget it wins.

```yaml
model:
  hidden_size: 768
  intermediate_size: 2304        # dense decoder MLP only
  moe_intermediate_size: 2304    # NEW KEY - routed + shared experts (see below)
  num_layers: 8
  num_attention_heads: 12
  head_dim: 64
  num_mlp_experts: 32
  num_attn_experts: 1
  num_ir_entries: 8192
  n_loops: 3
  top_k: 2
  vocab_size: 65536
  lm_head_factor: 4
```

### Parameter budget

Counted from the module tree, not estimated. The previous revision's "333M total / 184M
active" was wrong in both figures.

| config | total | active (incl. emb) | fwd FLOP/token |
|---|---|---|---|
| current `config.yaml` (512x5, V=129280) | 254M | 147M | — |
| old config A (768x8, M=21, V=49152) | 251M | 150M | 328M |
| **A' (768x8, M=32, V=65536)** | **332M** | **174M** | **357M** |

A' includes the two shared always-on experts of §2.2 (MLP + self-attention): +5.3M and +1.5M
params, and the shared attention expert runs densely every loop, so it costs ~9 MFLOP/token on
its own. "Active" includes the embedding tables (69M of pure lookup); excluding them A' is 104M.

Training FLOPs are ~3x forward, i.e. **~1071 MFLOP/token** for A'. That number is the sole
input to the budget calculation in §5 — recompute it if the config moves, especially if
`num_attn_experts`, `n_loops`, or `moe_intermediate_size` change.

### Why these numbers

- `hidden_size: 512` was the core problem. Every GEMM was skinny, arithmetic intensity was
  low, and the model never approached tensor-core peak. Widening improves quality per FLOP
  *and* MFU simultaneously.
- `n_loops: 4 -> 3`. Loops are expensive: non-MLP experts run densely every loop, so
  `n_loops=4` meant 12 dense attention passes per forward.
- `num_ir_entries: 16384 -> 8192`. The IR expert was ~11% of forward FLOPs (two 16384-wide
  matmuls plus a 16384-way softmax, run densely every loop regardless of routing) and its
  `[T, 16384]` softmax held ~1 GB of activation *per loop*. Halving frees throughput and
  ~2 GB of peak memory.
- **`moe_intermediate_size` is a new key.** The code passes one `intermediate_size` into both
  `Gemma4MLP` and `ParallelSparseMoELayer`, so the previous instruction ("hit the parameter
  target with `intermediate_size`, not with expert count") moved total and active params
  together and could not separate them. `moe_intermediate_size` is the only clean
  total-vs-active knob and must exist before any sizing decision is made.
- **`num_mlp_experts: 32`, not 21.** The "pool must be divisible by 8 for FP8 GEMMs" rule is
  **not real**: nothing in the code enforces it, and `ParallelSparseMoELayer.forward` runs its
  GEMMs under `te.autocast(enabled=False)`, so the routed MLPs are never in FP8 to begin with.
  (The genuine constraint — per-group row counts divisible by 16 — is precisely *why* autocast
  is disabled there.) `num_mlp_experts` is a free integer.
- **Where the extra experts come from.** Routed experts add total params at *zero* active
  compute: M=21 -> 32 takes total from 251M to 310M at an identical 328 MFLOP/token. The limit
  is per-expert data, not compute. At M=32 / pool=35 / top_k=2 / 25B tokens each MLP expert
  sees ~1.4B tokens; at M=52 it drops to ~0.9B and the experts are undertrained. M=32 is the
  end of the free lunch, not a step toward a bigger model.
- **`num_attn_experts` stays at 1.** Attention experts are cheap in parameters (+3M each) and
  expensive in throughput, because non-MLP experts run densely every loop and are then masked
  by routing — most of that compute is discarded. Measured at the A' body:

  | A | total | active | fwd FLOP/token | non-MLP share |
  |---|---|---|---|---|
  | 1 | 331M | 172M | 348M | 12% |
  | 2 | 334M | 175M | 365M | 17% |
  | 3 | 337M | 178M | 383M | 21% |

  Each additional attention expert costs ~5% throughput for ~1% of the parameters. Revisit
  only alongside the shared-attention-expert change (which makes the compute unconditional
  rather than discarded); until then, A=1.

### Vocab prune

`vocab_size: 129280` costs 137M params of lookup tables (99M embedding + 33M PLE + 4M MoE
PLE) — 38% of the model. But the **load-bearing** reason to prune is disk, not parameters:
30B tokens as uint32 is 120 GB on a 120 GB instance, with no room for source shards.

Any `vocab_size <= 65536` fits **uint16** and halves the dataset to 60 GB. That is the whole
disk argument, and it is satisfied at 65536. Dropping further to 49152 buys ~21M params in
exchange for a far more aggressive merge-table surgery, so:

**Prune to 65536, not 49152.**

- 331M total with an identical body
- tokens fit in uint16 (ids 0..65535 exactly)
- `65536 % lm_head_factor == 0` for both the main head (factor 4) and the MTP head (factor 8)

Count frequencies on the **final mix** (web + code + math + wiki at training ratios), not on
web alone — web-only counts badly underweight brackets, braces, indentation, and LaTeX.

**The metric that matters is fertility, not round-trip identity.** Removing BPE merges does
not break decode (byte fallback preserves surface strings), so a round-trip test will pass
almost regardless of how much damage was done. What it does is raise tokens-per-byte: text
that used to consume one token now consumes two or three. That directly shrinks the effective
data budget and eats the FLOP saving. Measure tokens/byte before and after on a held-out
sample of the final mix and fold the increase into §5's token count.

Cheaper lever, zero tokenizer risk: `per_layer_embeddings_size: 32 -> 16` saves ~9M at
V=65536. PLE costs `V * P * num_layers`, so it got more expensive when layers went 5 -> 8.
Optional; take it only if the parameter count needs trimming.

---

## 2. Architecture changes

### 2.1 Loop residual (required)

There is currently no residual anywhere in the loop. `forward_step` does:

```python
output = torch.zeros_like(hidden_states)
# ... accumulate weighted expert outputs ...
output = self.post_norm(output)
return output          # hidden_states is replaced, not updated
```

So `h_{t+1} = RMSNorm(sum_k w_k * E_k(h_t))`. The attention experts have no internal residual
either (`SelfAttention.forward` returns `dropout(attn_output)`, not `x + attn`). Consequences:

1. The only gradient path across a loop boundary that does not pass through an expert is the
   identity expert — and only when it is stochastically selected. Gradient flow across loops
   depends on a discrete routing event.
2. Even when identity is selected, the result is `w*h + (1-w)*E(h)` then RMSNorm: a
   weight-dependent partial residual, not a highway.

Fix: `hidden_states = hidden_states + loop_scale * delta`, with `loop_scale` a scalar
parameter. The loop starts as a near-no-op and learns how much refinement it wants — the
LayerScale / ReZero trick.

**Init `loop_scale` to 0.1, not 0.** A zero init is the textbook choice and works fine on its
own, but it interacts badly with the ponder loss in §2.3: at `loop_scale = 0` the LM loss has
*exactly zero* gradient with respect to `p_halt`, so the halt head sees only the ponder term.
See §2.3 for the full failure mode. 0.1 is small enough to preserve the "starts as a no-op"
property and large enough to keep the CE gradient path alive from step 1.

Note this is *not* the same mechanism as the existing `layer_scalar`, despite the resemblance:
`layer_scalar` is an init-1 gain on the whole layer output
([gemma4.py](modules/model/gemma4.py), `Gemma4TextDecoderLayer`), whereas `loop_scale` is a
small-init gate on a residual branch. The precedent is real but it is not already validated in
this codebase.

### 2.2 Shared experts (required)

There is currently no always-on path; every token goes through `top_k=2` routed slots and
nothing else. Two shared experts, both outside the router, both with a skip connection:

- **shared MLP** — guarantees every token a dense transform, which stabilises early training and
  lets routed experts specialise instead of all learning the same generic function.
- **shared self-attention** — guarantees every token sees the sequence every loop, instead of
  only when the router happens to pick an attention slot. This is also what keeps
  `num_attn_experts` at 1 defensible (§1): the routed attention experts become a specialisation
  layer on top of a path that always exists, rather than the only way to get attention at all.

Both cost active compute 1:1 — that is the point. +5.3M and +1.5M params respectively; the
shared attention expert is ~9 MFLOP/token because it runs densely on every loop.

Secondary benefit: their row counts are static (`B*S`), unlike the routed grouped GEMM, so they
can run **inside** `te.autocast` in FP8 while `ParallelSparseMoELayer` stays disabled.

Neither is in the router pool. (The pool divisibility constraint they used to be exempt from
does not exist — see §1.) Neither should dominate the loop: with `loop_scale` and the routed
accumulation both feeding the same residual, watch that the routed experts do not collapse to
near-zero weight — the expert selection plots are the detector.

### 2.3 Halt head replaces identity expert (required)

Three mechanisms currently fight each other:

- the identity expert ("signal I'm done")
- `compute_aux_loss`, computed over `self.num_experts` **including identity** — uniform balance
  penalises routing to identity above 1/40 = 2.5%
- `identity_skew`, a hack pushing the other way

Decouple "skip" from "done": the residual above is the skip; a dedicated halt head is the
signal. Then the identity expert is redundant.

```python
p_halt = torch.sigmoid(self.halt_proj(hidden_states))
hidden_states = hidden_states + (1.0 - p_halt) * self.loop_scale * delta
```

plus a ponder cost `loss += lambda_ponder * (1.0 - p_halt_all).mean()`, **masked to non-pad
positions** — unmasked, the term is dominated by trailing padding on short-document batches.

- Halting becomes soft and differentiable: `p_halt -> 1` means "don't modify me further",
  readable as a continuous confidence signal instead of inferred from a routing event.
- At inference, threshold `p_halt > tau` and stop looping for that token. Not cleanly possible
  with the identity expert because it is mixed with a second top_k slot.
- One `lambda_ponder` to tune instead of the `identity_skew` exponent hack.
- The aux-loss conflict disappears: identity is no longer in the pool.

Router exploration noise is **not** the fix and stays as-is. Noise perturbs which expert gets
picked; it does nothing about a missing gradient path.

#### The ponder / `loop_scale` deadlock

The halt head and the loop residual (§2.1) interact in a way that can permanently disable the
MoE stack, and it is silent when it happens.

At `loop_scale = 0` the update is `h + (1 - p_halt) * 0 * delta = h`, so
`dL_CE / dp_halt` is **exactly zero**. The only gradient reaching the halt head is the ponder
term, which is constant-sign and pushes `p_halt -> 1`. Gradient *magnitude* does not save this:
AdamW normalises by the running second moment, so a consistently-signed gradient moves the halt
bias at roughly `lr` per step no matter how small `lambda_ponder` is. If `p_halt` saturates
before `loop_scale` has grown, then `dL / dloop_scale = (1 - p_halt) * delta -> 0` and
`loop_scale` is frozen at zero. Both parameters are then at a stable point and the entire loop
is a no-op for the rest of the run — the loss curve still descends (the dense decoder is
untouched) so nothing obviously fails.

Two independent mitigations, apply **both**:

1. `loop_scale` init 0.1 rather than 0 (§2.1), so the CE gradient path to `p_halt` exists from
   step 1.
2. **Warm `lambda_ponder` from 0.** Hold it at zero for the first `ponder_warmup_tokens`
   (1B, matching `noise_anneal_tokens`), then ramp linearly to its target over the next 1B.
   Drive it from the live token count, exactly as the router noise anneal already does.

The `loop_scale` reading in PART 5 is the detector: if it has not moved off its init by ~1B
tokens, this is what happened.

#### `p_halt` is not sticky — decided, not deferred

`p_halt` is recomputed each loop from the *current* hidden state, so a token can halt at loop 1
and un-halt at loop 2. The exit rule sketched above ("threshold and stop looping") assumes a
monotonicity the training-time gate does not have.

**Decision: keep the per-loop formulation, and define the inference exit as
`stop once p_halt > tau at the current loop`** — i.e. accept that the exit is greedy and can
fire at a loop the training-time gate would have continued past. Rationale: an ACT-style
cumulative halting mass is the principled fix, but it adds a running accumulator, a
remainder term, and a second hyperparameter, and its benefit is unmeasurable until Gate 5 shows
the per-loop signal is informative at all. Gate 5's early-exit degradation curve is the test:
if forcing exit at loop 1 is catastrophic rather than merely worse, revisit this before
spending anything on a cumulative formulation.

Record this in the model card. `p_halt` is a greedy per-loop gate, not an ACT halting
distribution, and should not be described as one.

### 2.4 Per-loop CE supervision and a separate correctness head (required)

Two problems that only appear once you try to *use* `p_halt`.

**Problem 1: unsupervised intermediate states have no usable readout.** With CE applied only at
the final loop, intermediate states are optimised solely as inputs to the next loop, not as
things the LM head can read. Applying CE at every loop trains the prediction interfaces and
makes early exits usable; without it, thresholding `p_halt` at loop 1 gates on a readout that
does not exist. The controlled evidence for this comes from 44M- and 129M-parameter looped
models — close to this scale.

Fix: apply `main_lm_head` at each loop with ascending weights, e.g. `[0.2, 0.3, 1.0]` for
`n_loops=3`. Costs ~3x LM-head compute, which is affordable because the chunked CE already
prevents `[T, vocab]` from materialising.

**Apply the final `RMSNorm` per loop as well.** `lm_head` reads `self.norm(x)`, never the raw
residual stream (`TinyMoETransformer.forward`). Feeding un-normalised intermediate states to
the head produces meaningless per-loop losses and will look like "per-loop CE is flat" in the
monitoring table.

**Problem 2: `p_halt` measures compute utility, not correctness.** It is trained by the LM loss
(more refinement where refinement helps) and the ponder cost (fewer loops). So it learns *"does
more latent computation change my prediction?"* — not *"is my prediction right?"*

These come apart exactly where it matters. A confidently hallucinated fact is **stable under
refinement**, so it halts at loop 0 and reads as maximum certainty. The distinction is known in
the literature as utility calibration vs correctness calibration; a signal calibrated to the
value of further computation is not thereby calibrated to correctness. The Ouro authors list
principled gate calibration and gate failure-mode analysis (easy/hard misclassification) as
open problems in their own work.

Fix: do not overload `p_halt`. Add a second head:

```python
self.correct_proj = nn.Linear(hidden_size, 1, bias=True)
```

trained with BCE against **whether the model's own top-1 prediction at that position was
correct**. That target is free at every token of pretraining — no labels, no extra data, no
extra forward pass. Result: two separate scalars with clean semantics.

| signal | question it answers | use |
|---|---|---|
| `p_halt` | is more latent computation useful here? | inference-time early exit |
| `p_correct` | is this prediction right? | abstention, calibration, truthfulness |

`p_correct` is the one to report ECE on. `p_halt` is a compute-allocation knob and should not
be presented as a confidence score.

**`p_correct` must earn its place against a free baseline.** The max softmax probability of the
LM distribution, `max(softmax(logits))`, is already a strong per-token correctness predictor and
costs nothing — no head, no extra loss term, no hyperparameter. If a learned `p_correct` does
not beat max-prob on ECE and on abstention AUROC over the Gate 5 held-out slice, the head is
adding a loss term and a failure mode in exchange for nothing.

**Gate: measure both at Gate 5, on the same slice, side by side. If `p_correct` does not beat
max-prob, delete Task 4b and use max-prob for the whole PART 6 abstention story.** That path is
strictly cheaper and the §6 thesis survives it intact — abstention needs *a* calibrated signal,
not specifically a learned one. Deciding this before renting costs one eval script; deciding it
after costs a rerun.

### 2.5 Not changing

- Router annealed exploration noise — leave as-is.
- Gradient checkpointing — stays off at this size.
- `te.autocast(enabled=False)` around `ParallelSparseMoELayer` GEMMs — dynamic routing cannot
  guarantee NVFP4's 16-row alignment. Sparsity wins over precision.
- `NUM_DATA_WORKERS = 4` — with mmap the workers just slice memory.

---

## 3. Training-loop fixes

- **`num_epochs: 10` vs `target_tokens` are inconsistent.** `total_steps` anchors the cosine
  schedule to `target_tokens`, but nothing in the loop stops there — it just runs N epochs.
  Add a token-count break, set `num_epochs: 1`. For a run this size you want 1 epoch over more
  data, not 10 over less.
- Checkpoint interval 5000 -> 1500 steps. The run is interruptible.
- `m_splits = torch.bincount(...).tolist()` is a known host sync, once per loop per forward.
  Accepted, not fixed. It is the kind of thing that turns 25% MFU into 12%; revisit only if
  the measured MFU is bad.

---

## 4. Data

**Prepare 30B tokens; train on whatever the EUR 100 budget buys** (§5 — expected 20-30B).
Having 30B on disk and training on 22B is free; the reverse is not. Single epoch either way.

At 25B tokens against A's 331M total params that is ~76 tokens per param — past Chinchilla,
deliberately, because small models are trained for inference not for compute-optimality
(SmolLM2-360M saw 4T tokens).

The phase split in §4.2 is expressed as percentages of the *committed* budget, so it does not
need editing when the budget lands. Prepare the bins at 30B and stop reading when the token
break fires.

### 4.1 Sources

Selected for: quality at <500M scale, actual text (not metadata requiring reconstruction),
and sliceable by file so only what is needed gets downloaded.

| domain | dataset | why |
|---|---|---|
| web | `HuggingFaceFW/fineweb-edu` | best educational web at this scale; maintained through mid-2025 |
| web | `mlfoundations/dclm-baseline-1.0` | diversity corrective; pure edu-filtered web produces textbook-voiced models that cannot handle conversational input |
| PDF | `HuggingFaceFW/finepdfs-edu` | globally deduplicated, 350B+ tokens; only realistic source of long coherent documents |
| code | `nvidia/Nemotron-CC-Code-v1` | 427.9B tokens, Lynx + LLM pipeline, **ships actual text** |
| math | `nvidia/Nemotron-CC-Math-v1` (`4plus`) | 5.5x larger than FineMath-4+, beats it on math/code/knowledge |
| wiki | `wikimedia/wikipedia` (en) | encyclopedic coverage |
| instruct | `HuggingFaceTB/smoltalk2` | folded into phase 2, see below |

**Do not use** `HuggingFaceTB/stack-edu` or `nvidia/Nemotron-Pretraining-Code-v1/v2`: both ship
identifiers (SWHIDs / metadata) requiring reconstruction from an external store. Not worth the
pipeline for a one-shot 4B-token pull.

**Do not use** `HuggingFaceTB/smollm-corpus` as the backbone. It is the SmolLM1-era mix
(FineWeb-Edu-dedup + Cosmopedia v2 + python-edu) and has been superseded. `python-edu` also has
the blob-id problem.

**Dropped:** `nemotron-pre-specialized-v1` / `v1.1` — synthetic/specialised, inflate the
download, and the benefit is not extractable at this scale.

### 4.2 Mix

Two phases. The anneal is cheap and disproportionately effective at this scale.

Percentages are of the committed budget (§5), not of a fixed 30B. Bins are written at 30B;
the token break decides where reading stops.

| | Phase 1 (85% of budget) | Phase 2 anneal (15% of budget) |
|---|---|---|
| FineWeb-Edu | 55% | 15% |
| DCLM-baseline | 10% | — |
| FinePDFs-Edu | 7% | 10% |
| Nemotron-CC-Code | 12% | 22% |
| Nemotron-CC-Math `4plus` | 3% | 30% |
| Wikipedia | 3% | 8% |
| SmolTalk2 | — | 15% |
| LR | warmup -> cosine to 10% | -> ~0 |

The instruct row is the one people skip. Nemotron 3 and Olmo 3 both fold SFT-style data into
late pretraining (Dolma 3's second stage is 25% instruction data). It makes the eventual SFT
run converge faster and stops the base model being pure completion-mode.

Expect math benchmarks to stay near zero regardless. SmolLM2-1.7B scored 3.21 on math after
6T tokens. Math data is in the mix for representation quality, not for GSM8K.

### 4.3 Downloads

~47 GB down, ~56 GB of `.bin` out.

| source | tokens | ~download | slice |
|---|---|---|---|
| FineWeb-Edu | 15B | ~25 GB | one CC dump (`data/CC-MAIN-2025-26/`), partial files |
| DCLM-baseline | 2.5B | ~4 GB | handful of global shards |
| FinePDFs-Edu | 2.2B | ~3 GB | `eng_Latn`, few files |
| Nemotron-CC-Code-v1 | 4B | ~6 GB | few shards |
| Nemotron-CC-Math-v1 `4plus` | 2B | ~3 GB | few shards |
| wikipedia en | 1.1B | ~4 GB | subset of parquet files |
| smoltalk2 | 0.7B | ~2 GB | whole dataset, it is small |

FineWeb-Edu is sharded by crawl and averages ~13B tokens per dump across its 1.3T total, so one
recent dump plus a partial second covers the web slice.

Use `huggingface_hub.hf_hub_download` on individual files. `load_dataset` gives no control over
how many GB land on disk.

### 4.4 No upload step

30B tokens as uint16 is exactly 60 GB. The same content as compressed parquet is ~45-55 GB. The
pretokenized artifact is **larger** than its source, and a home upload link is the slowest hop
in the pipeline.

Instead: download source shards directly onto the rented box and tokenize there, once, up front.
The CPU-starvation warning applies to streaming tokenization *during training*, where it is a
permanent bottleneck — not to a single upfront pass. 30B tokens is ~120 GB of raw text; batched
HF fast tokenizers across 16 vCPUs run ~30-60 MB/s aggregate, so 40 min to 2 hours. That is the
prep time already reserved in §5's 47-hour training window. Likely download-bound anyway.

Reproducibility comes from a committed file manifest + prep script, not from a stored artifact.

### 4.5 Licensing

FineWeb-Edu / FinePDFs / DCLM: ODC-BY. SmolTalk2: Apache-2.0. Nemotron datasets: NVIDIA Data
Access Agreement for Model Training. All fine for this project; re-read the NVIDIA terms before
releasing weights.

---

## 5. GPU and budget

**Budget is fixed at EUR 100.** That is the constraint; the token count is the output.

**1x H100 SXM/NVL 80GB, interruptible, ~EUR 1.8-2.5/hr.**

- FP8 is the whole argument. `DelayedScaling` is mature on sm90 and gives a real 1.3-1.5x on the
  big GEMMs. That path is dead on the 5090 (NVFP4 on consumer Blackwell).
- Not multi-GPU. 330M params in BF16 means ~0.7 GB of gradients allreduced every step; most vast
  4090/5090 rigs are PCIe-only with no NVLink. One big GPU + `grad_accumulation_steps`.
- Not A100 — no FP8, ~312 TFLOPS BF16, often priced near H100 on vast.
- Not B200 — TE/flash-attn wheels for sm100 are finickier, 2-3x the price.

### What EUR 100 buys

At EUR 2/hr that is 50 instance-hours. Reserve ~3 h for data prep, the smoke test, and
checkpoint extraction, leaving **~47 training hours**. Config A' costs ~1071 MFLOP/token to
train (§1); H100 SXM BF16 dense peak is ~990 TFLOPS.

| MFU vs BF16 peak | tokens in 47 h | tokens / total param |
|---|---|---|
| 10% | 16B | 47 |
| 12.5% | 20B | 59 |
| 15% | 23B | 71 |
| 17.5% | 27B | 82 |
| 20% | 31B | 94 |
| 25% | 39B | 118 |

MFU here is measured against BF16 peak even though the decoder GEMMs run in FP8 — the FP8 gain
shows up as a higher apparent MFU rather than as a separate term, which is why the number is
measured rather than modelled. 30B is the ceiling of what the prepared data supports, so above
~18% MFU the run is data-limited rather than budget-limited and the surplus should go to a
larger `moe_intermediate_size` rather than more tokens.

**Decide the number at Gate 4, not now.** Scale the measured 5090 tokens/sec by the expected
H100 ratio, read the table, round down, and set `target_tokens` to that. The cosine schedule is
anchored to `target_tokens`; stopping early against a longer schedule leaves the LR high and
wastes the anneal, which is the single most expensive avoidable mistake in this plan.

**Do not spend the budget on parameters instead of tokens.** A 500M-total / 250M-active variant
(1024x10, M=40) costs 1.70x the FLOPs per token, which at a fixed EUR 100 means ~14B tokens
instead of ~24B — 28 tokens per param. The §6 thesis is calibrated abstention, a coverage
capability that wants data, and §4's own argument is for overtraining small models. The size
increase that *is* free is more routed experts (M=21 -> 32, already taken in §1); beyond that
the trade is real and it goes the wrong way.

### vast.ai gotchas

- TE and flash-attn wheels in `requirements.txt` are built for **sm120**. H100 is **sm90**. Use
  `nvcr.io/nvidia/pytorch:25.xx-py3` (ships TE + flash-attn prebuilt) and layer only non-CUDA
  deps on top.
- Disk size is fixed at instance creation and cannot be grown. Provision 120 GB.
- Interruptible instances **stop** when outbid, they do not destroy — disk persists, storage is
  billed. A reclaim costs at most one checkpoint interval.

---

## 6. Post-training: calibrated abstention, not chain-of-thought

### 6.1 The target

**Redefine the goal from "reasoning model" to "calibrated model that knows when it does not
know."** Reasons:

- Calibrated abstention is a much shallower capability than multi-step deduction and is
  plausibly learnable at 330M total / 172M active. Chain-of-thought reasoning is not.
- It is directly measurable (ECE, abstention precision/recall), unlike "reasoning quality".
- It is what the halt/correctness machinery in §2.4 is actually positioned to deliver.
- It is a more interesting result than another tiny model scoring ~3% on GSM8K.

Math and long-form reasoning benchmarks will not move at this scale regardless of post-training.
Do not treat that as a failure of the run.

### 6.2 SFT datasets

Priority order:

| dataset | role |
|---|---|
| `HuggingFaceTB/smoltalk2` (no-think splits) | general instruction following; successor to SmolTalk, built for small models |
| `rajpurkar/squad_v2` | **the single most on-target set** — unanswerable questions are genuine abstention supervision |
| `allenai/tulu-3-sft-personas-math` | short worked solutions, not long CoT |
| `openai/gsm8k` (socratic config) | short numbered steps |
| `HuggingFaceH4/no_robots` | small, human-written; tone and refusal style |

Hold out the smoltalk2 portion used in the phase-2 pretraining mix (§4.2) so SFT is not
training on data already seen.

Alternative general-purpose mixture: `allenai/tulu-3-sft-mixture` (939k, ODC-BY), tuned for
8B+ students.

**Still do not SFT on long reasoning traces** (KIMI-K2.5, Claude, Fable 5 trace datasets in
README.md). Small Model Learnability Gap: models at or below 3B do not consistently benefit
from long chain-of-thought distillation and do better on shorter chains matched to their
capacity. At this scale it teaches the model to emit thousands of tokens of confident filler
before a wrong answer. Those datasets also typically carry provider terms restricting training of
competing models.

### 6.3 Self-labelled calibration set (build this)

The one dataset worth building yourself, because it requires this model and cannot be
downloaded:

1. Sample the SFT checkpoint N=16 times on short-answer QA (TriviaQA, NQ-open, SQuAD 2.0) at
   temperature ~0.8.
2. Label each question by empirical pass rate.
3. Rewrite targets by pass rate: `> 0.8` -> the answer; `< 0.2` -> an abstention; in between ->
   a hedged answer.
4. SFT on the rewritten set.

Cheap, needs no verifier infrastructure, and produces the truthfulness behaviour directly. It
also yields a held-out set for computing ECE against `p_correct` and `p_halt`, which is how you
find out whether the halt signal means anything rather than assuming it does.

### 6.4 RL: not for this run

Do not budget for RL on the first model. The evidence at this scale is consistent and negative:

- A 135M single-GPU study ran RLVR on GSM8K: SFT base 24/1319 (1.82%), GRPO at 192-token
  completions **fell** to 21/1319, at 320 tokens to 16/1319. The authors' framing: the
  experiment is useful precisely because it fails.
- On Qwen2.5-0.5B base, even with a format reward the reward stayed below 0.1 after 300 steps
  with no upward trend. The instruct variant formatted correctly but could not sample correct
  answers, so most prompts yielded zero advantage.

Mechanism, not architecture: under a 0/1 reward, if the base model cannot sample correct
solutions there is no gradient signal. RLVR amplifies what is already in the base distribution.
At 172M active / ~25B tokens there is very little to amplify. Ouro, the LoopLM reference point,
is 1.4B-2.6B trained on 7.7T tokens — roughly 250x this run's compute.

**Gate for a future attempt:** measure pass@8 on the target task with the SFT checkpoint. Below
~15%, RL will do nothing and the budget will be spent confirming that.

If that gate ever passes, note that standard GRPO is architecturally mismatched to a looped
model — it assigns credit only to output tokens while the computation is latent. The relevant
methods reward latent trajectories or per-loop states instead (LoopRPT, arXiv 2603.19714;
RLTT), which additionally improve exit behaviour: more early-step exits with final-step
dominance maintained, i.e. better gate calibration. Read those before writing any RL code.

---

## 7. Order of work

1. Loop residual + `loop_scale` (init 0.1)
2. Shared expert; add the `moe_intermediate_size` config key
3. Halt head + ponder loss (warmed, pad-masked); delete identity expert and `identity_skew`
4. Per-loop CE supervision (re-applying the final norm) + correctness head
5. `num_ir_entries -> 8192`
6. Token-count break; `num_epochs: 1`; checkpoint interval 1500
7. Config A'; delete the fictional divisibility constraint from `CLAUDE.md` and `config.yaml`
8. Vocab prune to 65536, gated on fertility
9. mmap dataset
10. Local 500M-token sanity run at the new hidden size — **measure MFU and gate calibration,
    then commit `target_tokens`**
11. Rent H100, enable FP8, run
12. SFT: smoltalk2 no-think + SQuAD v2 + short-CoT math
13. Build the self-labelled calibration set; second SFT pass; report ECE

Steps 1-7 are an evening. Step 10 converts the fixed EUR 100 into a token count and decides
whether `p_correct` survives at all (§2.4). Steps 12-13 are where the actual thesis of the
model gets tested.
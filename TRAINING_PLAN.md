I read through the model, training loop, and configs, and ran the parameter/FLOP math for your current config plus a few alternatives. Answers in your order.

---

## 1. GPU choice on vast.ai

**Rent 1× H100 SXM/NVL 80GB (~$1.8–2.5/hr). Not an A100, not a rented 5090, not multi-GPU.**

Reasoning:

- **FP8 is the whole argument.** Your code already has `fp8_recipe = DelayedScaling(...)` sitting unused in [pretrain.py:27](scripts/pretrain.py#L27) because NVFP4 on consumer Blackwell is a mess. On sm90, `DelayedScaling` FP8 is mature and gives a real 1.3–1.5× on the big GEMMs. You flip `USE_LOW_PRECISION = True` and `chosen_recipe = fp8_recipe` and it just works. That path is dead on your 5090.
- **Don't go multi-GPU at this size.** 250–500M params in BF16 means ~0.5–1 GB of gradients allreduced every step. Most vast 4090/5090 rigs are PCIe-only with no NVLink; the allreduce will eat most of the scaling. One big GPU + `grad_accumulation_steps` is correct here.
- **Skip A100** — no FP8, ~312 TFLOPS BF16, and often priced near H100 on vast.
- **B200** only if you want NVFP4, but TE/flash-attn wheels for sm100 are finickier and it's 2–3× the price. Not worth it for a first real run.

Cost estimate for 10B tokens on 1× H100 at 25% MFU (~190 TFLOPS effective), €2/hr:

| config | total params | active/token | FLOP/token | H100-hours | € |
|---|---|---|---|---|---|
| current (243M) | 254M | 77M | 624 MFLOP | 9 | **~18** |
| A: 768×8, 16+1 experts | 333M | 184M | 1266 MFLOP | 19 | **~37** |
| B: 1024×8, 24+1 experts | **492M** | 236M | 1579 MFLOP | 23 | **~46** |

At 20B tokens, double those. **Your €200 fear is unfounded — even the 500M config at 20B tokens is ~€90 of GPU.** The 25% MFU figure is optimistic for this code as written though (see §2), so budget 2× and you're still fine.

**Two vast.ai gotchas that will actually cost you money:**

- Your TE and flash-attn wheels are built for **sm120**. H100 is **sm90**. Don't try to `pip install` on a rented box — use `nvcr.io/nvidia/pytorch:25.xx-py3`, which ships TE + flash-attn prebuilt, and layer your `requirements.txt` on top.
- Use **interruptible instances** (~half price). You already have per-worker checkpoint/resume from `018ef4f`, so you're one of the few people who can actually exploit that.

---

## 2. Architecture, and how to get to 500M

### The architecture is sound but misallocated

`hidden_size: 512` with `num_layers: 5` is the core problem, and it's exactly why you're compute-bound rather than VRAM-bound. Every GEMM is skinny, arithmetic intensity is low, and you never get near tensor-core peak — you're memory-bandwidth and kernel-launch bound. Widening improves quality per FLOP *and* MFU at the same time.

Where your FLOPs actually go right now (fwd+bwd, per token):

```
params    461 MFLOP  (74%)
IR expert 101 MFLOP  (16%)  <-- one expert
attention  63 MFLOP  (10%)
```

**The IR expert costs 16% of your training compute.** `num_ir_entries: 16384 × ir_dim: 128` is two 16384-wide matmuls plus a 16384-way softmax, run **densely every loop regardless of routing** (`expert_cache` in [moe.py:320-326](modules/model/moe.py#L320-L326) computes it whether or not it's selected). Same for the self/cross-attention experts. Dropping `num_ir_entries` to 4096 is a free ~12% speedup; I'd do that before the big run.

### Don't add more experts

More routed MLP experts add total params but **zero active compute** — and each expert then sees less data. At 36 experts, `top_k=2`, 10B tokens, each MLP expert sees roughly 500M tokens. Going to 72 halves that. You'd be paying storage for undertrained experts.

Also — **you don't currently have shared experts.** There's no always-on path; every token goes through `top_k=2` routed slots and nothing else. Adding 1 shared expert is a genuinely good idea, but for a different reason than you think: it guarantees every token a dense transform, which stabilizes early training and lets the routed experts actually specialize instead of all learning the same generic function. It costs active compute 1:1 — that's the point.

### Concrete 500M config

```yaml
model:
  hidden_size: 1024        # 512 -> 1024, the single highest-value change
  intermediate_size: 3072  # dense decoder MLP
  num_layers: 8            # 5 -> 8
  num_attention_heads: 16
  head_dim: 64
  num_mlp_experts: 24      # 36 -> 24, but each is 2.4x bigger
  # + 1 shared (always-on) expert -- needs a small code change
  num_ir_entries: 8192     # 16384 -> 8192
  n_loops: 3               # 4 -> 3, loops are expensive (12 dense attn passes at n_loops=4)
  top_k: 2
```

That lands at **492M total / 236M active**. If you'd rather spend the compute on data than params, config **A** (768×8, 16 experts + 1 shared, 333M/184M) is what I'd actually run first — it's 25% cheaper per token and will beat the 500M version at a fixed 10B-token budget.

### Optional: prune the vocab

`vocab_size: 129280` costs you 66M embedding + 21M PLE table + 17M LM head = **87M params, 34% of the model**, on a model this small. Pruning to ~49152 (count frequencies on a corpus sample, keep top-N + specials/bytes, remap ids) gives you:

- 492M → 361M params with an *identical* body, so the 500M budget goes into the parts that learn
- tokens fit in **uint16**, halving your pretokenized dataset on disk and over the wire
- ~10% fewer FLOPs/token

~100 lines of tokenizer surgery. High value, but do it before you tokenize anything.

### Two things to fix in the training loop first

- **`num_epochs: 10` vs `target_tokens: 5e9` are inconsistent.** `total_steps` = 9536 anchors the cosine schedule to 5B tokens, but nothing in [pretrain.py:384](scripts/pretrain.py#L384) stops at `target_tokens` — the loop just runs 10 epochs. If your dataset exceeds 5B tokens, epoch 0 alone blows past the schedule and you train the rest at `eta_min`. Add a token-count break. And for a run this size you want **1 epoch over more data**, not 10 over less.
- **`m_splits = torch.bincount(...).tolist()`** in [moe.py:58](modules/model/moe.py#L58) is a host sync, once per loop per forward — 8 per step with sub-checkpointing. On a rented GPU that's a direct line item. Not required to fix, but it's the kind of thing that turns 25% MFU into 12%.

---

## 3. The identity expert — you've spotted a real problem

Your instinct is right, and the situation is worse than "it kinda lacks the skip connection."

**There is no residual anywhere in the loop.** [moe.py:314-352](modules/model/moe.py#L314-L352):

```python
output = torch.zeros_like(hidden_states)
# ... accumulate weighted expert outputs ...
output = self.post_norm(output)
return output          # <- hidden_states is replaced, not updated
```

So `h_{t+1} = RMSNorm(Σ_k w_k · E_k(h_t))`. The individual attention experts don't have internal residuals either — `SelfAttention.forward` returns `dropout(attn_output)`, not `x + attn`. Which means:

1. **The only gradient path across a loop boundary that isn't through an expert is the identity expert** — and only when it's stochastically selected. Gradient flow through 4 loops depends on a discrete routing event. That's fragile.
2. Even when identity *is* selected, you get `w·h + (1-w)·E(h)` then RMSNorm — a weight-dependent, partial residual. Not a highway.
3. **Your aux loss actively fights the identity's purpose.** `compute_aux_loss` in [router.py:39-55](modules/model/router.py#L39-L55) is computed over `self.num_experts`, which *includes* identity. Uniform balance means you are explicitly penalizing the model for routing to identity more than 1/40 = 2.5% of the time. You built a "signal I'm done" mechanism and then added a loss term that caps it at 2.5%. `identity_skew` is then a second hack pushing the other way. These three mechanisms are fighting each other.

**Noise will not fix this.** Noise perturbs *which* expert gets picked; it does nothing about the missing gradient path. The fact that you were reaching for noise is the symptom — you noticed identity isn't getting picked reliably, and the root cause is (1) and (3), not insufficient exploration. You already have annealed router noise in `Router.forward`; leave it as-is.

### The fix: decouple "skip" from "done"

Give the loop a real residual (that's the skip), and give the model a dedicated halt head (that's the signal). Then the identity expert is redundant and you delete it, along with `identity_skew`.

In `LoopMixtureOfExperts.__init__`:

```python
self.loop_scale = nn.Parameter(torch.zeros(1))          # start the loop as a no-op
self.halt_proj = nn.Linear(hidden_size, 1, bias=True)
nn.init.zeros_(self.halt_proj.weight)
nn.init.constant_(self.halt_proj.bias, -2.0)            # p_halt ~ 0.12 at init
```

In `forward_step`, replacing the current return:

```python
delta   = self.dropout(self.post_norm(output))          # what the experts propose
p_halt  = torch.sigmoid(self.halt_proj(hidden_states))  # [B, S, 1]
hidden_states = hidden_states + (1.0 - p_halt) * self.loop_scale * delta
return hidden_states, load_balancing_loss, p_halt
```

Why this is strictly better on your own terms:

- **Unconditional gradient highway.** `loop_scale` at 0 means the model starts as "4 loops of nothing" and learns how much refinement it wants — the same trick as your `layer_scalar`, and it makes deep loop stacks trainable.
- **Halting is now soft and differentiable.** `p_halt → 1` means "don't modify me further," which is exactly your semantics, but it's a continuous scalar you can read out as a confidence signal instead of inferring it from "the router happened to pick expert #2."
- **At inference it's a real compute saving.** Threshold `p_halt > τ` and stop looping for that token. You can't do that cleanly with the identity expert because it's mixed with a second `top_k` slot.
- **Replaces `identity_skew` with a principled term.** Add a ponder cost to encourage early exit:
  ```python
  loss = loss + lambda_ponder * (1.0 - p_halt_all).mean()   # lambda_ponder ~ 1e-3 to 1e-2
  ```
  This is PonderNet / LoopLM's exit head. Tune one λ instead of the `identity_skew` exponent hack.
- **The aux loss conflict disappears**, because identity is no longer in the expert pool.

If you'd rather keep the identity expert for now, the minimum fix is: **add the residual, and exclude the identity index from `compute_aux_loss`** (renormalize the remaining probs, count only non-identity slot assignments). Without that second part it stays pinned at 2.5% no matter what `identity_skew` does.

---

## 4. Data

Your instinct is right but for a slightly different reason: it's not that 5B tokens is too much data — a 350M model wants **10–20B tokens** (30–50 tokens per active param; Chinchilla-optimal 20:1 undertrains small models you actually want to use). The problem is the *mix* and the *pipeline*.

### Mix

**Drop `nemotron-pre-specialized-v1` and `v1.1`.** They're synthetic/specialized, they inflate your download, and at 350M you won't extract the benefit. Save them for the anneal phase.

**Strongly consider [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) as your backbone** instead of raw Ultra-FineWeb. It's fineweb-edu-dedup + cosmopedia-v2 + python-edu, filtered specifically for small models, and SmolLM2-360M is the closest published reference point to what you're building. At <500M, education-filtered web beats raw web by a large margin.

**You have no code data.** Even 10% code measurably improves reasoning in tiny models. `python-edu` comes free with smollm-corpus.

Two-phase, which matters a lot at this scale:

| phase | tokens | mix | LR |
|---|---|---|---|
| **1 — bulk** | 85% | 78% fineweb-edu/Ultra-FineWeb · 12% code · 5% wiki · 5% math | cosine → 10% |
| **2 — anneal** | 15% | 40% math · 25% wiki/textbook · 20% code · 15% nemotron-specialized | → ~0 |

The anneal phase is cheap and disproportionately effective for small models.

### Pretokenize locally before you rent — this is the biggest cost saver

Your [dataset.py](modules/data/dataset.py) streams parquet and tokenizes on the fly with `NUM_DATA_WORKERS = 4`. On vast you frequently get 8–16 vCPUs, and HF fast-tokenizer throughput is ~1–2 MB/s/core. **You will be CPU-starved and paying €2/hr for an idle H100.**

Do this on your own machine before renting:

1. Tokenize the full mix once into a flat `train.bin` + `train.idx` (nanoGPT-style). 10B tokens = 40 GB as uint32, **20 GB as uint16 if you prune the vocab below 65536**.
2. Upload as a private HF Hub dataset repo (free, and vast pulls at 200 MB/s–1 GB/s).
3. Swap the dataset to an mmap reader. Document packing gets *simpler* — you're just slicing a contiguous token stream at `max_length` and emitting `document_ids` from stored boundary offsets.

This also removes tokenizer nondeterminism from your resume path and makes the `worker_positions` bookkeeping in `snapshot_resume_positions()` trivial (a single global offset).

---

## Suggested order

1. Add the loop residual + `loop_scale` (biggest quality win, ~15 lines, testable on your 5090 today)
2. Replace identity expert with the halt head + ponder loss; delete `identity_skew`
3. Fix the `num_epochs` / `target_tokens` inconsistency
4. `num_ir_entries` → 4096 or 8192
5. Prune vocab to 49152, pretokenize the new mix locally to uint16, upload to HF
6. Short 5090 sanity run (~500M tokens) at the new hidden size to confirm loss curves and get real tokens/sec
7. Rent the H100, enable FP8, run 10–15B tokens

Steps 1–4 are ~an evening. Step 6 is what tells you whether the €40 or the €90 config is the right call — measure your actual MFU before committing, since my cost table assumes 25% and this code may well land at half that.

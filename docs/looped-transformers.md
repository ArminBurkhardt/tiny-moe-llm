# Looped transformers, external reference

What the published looped/recurrent-depth systems do, what the evidence supports, and where this
repo's loop agrees or disagrees with it. Checked against `ir-train-build` (PR #18) and the 16B-token
run in [CONCLUSION.md](CONCLUSION.md).

Survey framing: Raschka, *GPT-6 Astra, Looped Transformers, and Hidden Reasoning* (Ahead of AI,
2026-09-09). Underlying papers in [References](#references).

Astra's use of recurrent depth is unconfirmed reporting, and OpenAI's chief scientist has said the
computation-graph depth of current frontier models is within ~2x of GPT-4. "Astra proves looping
works" is weak evidence. The load-bearing results are SMELT, Mixture-of-Recursions, and Virtual
Logic Depth.

---

## 1. The design space

| System | What is looped | Passes | Loop-count decision | Scale |
|---|---|---|---|---|
| Universal Transformer (2018) | one block | adaptive | per-token halting prob, cumulative > threshold, hard cap | small |
| Nanbeige4.2-3B (2026) | stack of 22 blocks | 2, fixed | none | 3B |
| Ouro / "LoopLM" (2025) | stack of 48 blocks | 4 | learned exit gate, cumulative-prob threshold selects output pass | 2.6B |
| Latent Reasoning (Geiping 2025) | stack of 4, between 2 prelude and 2 coda blocks | sampled in training, 8/32/64 at inference | KL between successive passes' next-token distributions | 3.5B / 800B tok |
| Mixture-of-Recursions (2025) | shared stack between a distinct first and last layer | 1 to 3 per token | learned depth router (expert-choice or token-choice) | ≤1.7B |
| SMELT (2026) | middle half of an MoE stack, applied twice | 2 | none | ≤54B non-emb |
| `tiny-moe-llm` | one MoE block | 3, `loop_count_sampling=0.3` | parameter-free convergence exit on the readout | 332M trained, 381M after the IR table widening |

Two axes get conflated in casual discussion:

1. **Weight sharing across depth**, the cheap-parameters argument. Nanbeige, Ouro, SMELT.
2. **Adaptive per-token depth**, the compute-allocation argument. UT, MoR, Geiping.

This repo now implements (1) properly and has a working inference-time version of (2). The
training-time version of (2) was tried (halt head + ponder loss) and failed. See §4.

---

## 2. What the published evidence supports

### 2.1 Looping wins at matched compute, at scale

SMELT is the closest published result to this architecture (MoE plus looping). It matches compute
per token, non-embedding parameters, **and** KV-cache size, then compares. The recipe: apply the
middle half of the blocks twice, narrow the hidden dimension to pay for the extra applications, add
experts to recover the parameter count, retune the head configuration to hold the cache constant.
Fitted curves give 6.8 to 18% less training compute for equal validation loss, up to 54B
non-embedding parameters.

### 2.2 The advantage is scale-dependent, and we are below the crossover

Mixture-of-Recursions ran four model scales by three compute budgets. **At the smallest scale (135M)
the vanilla transformer was best.** MoR caught up and won at larger sizes, most clearly at smaller
compute budgets. At the largest budget several curves converge.

Nanbeige, at 3B, kept 2 passes: more gave small quality gains while slowing training and making
optimization less stable.

**Our own run now sits exactly on top of this.** Per-loop readout CE at the end of 16B tokens:

| loop | final CE | `loop_scale` after migration |
|---|---|---|
| 1 | 3.109 | 0.63 |
| 2 | 2.969 | 0.32 |
| 3 | 2.969 | 0.11 |

Loop 3 contributes nothing measurable in the training log, and on the migrated checkpoints Stage 0
measures it at 0.021 / 0.008 nats (`sft_final` / `phase2_final`): small, not zero. The learned gain
on loop 3 is a sixth of loop 1's, but that is **not a second, independent signal** — the migrated
`loop_scale` *is* the deleted halt gate folded in, so the loss and the parameters are reporting the
same thing.

**The obvious confound is live, not ruled out.** An earlier version of this section read
`[0.290, 0.134, 0.084]` as per-loop `p_halt` and concluded loop 3 was the least gated loop. Those
numbers are mean **`(1 - p_halt)`**, the fraction of each loop's update that passed the gate
([phase0_migration.md](measurements/phase0_migration.md)). Loop 3 was the **most** suppressed loop:
through all 16B pretraining tokens it passed 8% of its update against loop 1's 29%. Whether loop 3
was gated because it was useless, or stayed useless because it was gated, cannot be decided from
this run.

Consequences:

- Looping is a spec requirement for this model, so the answer to saturation is to give later loops
  something to do, not to cut them. NEXT.md's Phase 3c (input injection), 5c (depth curriculum with
  evidence and a loop-conditioned query) and 7c (coda) are that work.
- A compute-matched `n_loops=1` arm was considered and **deliberately not adopted** (2026-09-15): it
  tests whether looping should exist, which the spec has already decided.
- Read the depth ablation on the tasks loops are supposed to help (§2.3), which is where NEXT.md's
  G5 now reads it.

### 2.3 Looping buys reasoning, not knowledge

*Beyond Parameters / Virtual Logic Depth* measures memorization and reasoning separately.
Memorization capacity is flat in block applications and grows with distinct parameter count.
Multi-step math improves with extra applications at fixed parameters.

This is the strongest published support for the EuroHPC framing, and it sharpens the gate criteria:

- loops are compute, so expect movement on GSM8K, ARC-c, the reasoning half of MMLU
- parameters and the external index are knowledge, so **do not write a gate that requires loops to
  move TriviaQA or NQ-open**
- report the depth ablation split by task category, because a flat average will hide the effect

It also reframes the IR expert's current failure. An attenuated read that costs 0.0002 nats is a
scale bug in the value path, and this result says the IR table is the right place for facts to live
regardless: loops cannot substitute for it.

### 2.4 Train the loop from scratch

Nanbeige reports that training the looped architecture from scratch beat upcycling a pretrained
dense transformer. Not an issue here, but it rules out warm-starting the body from a public
checkpoint.

---

## 3. Open gaps

### 3.1 No coda blocks after the loop

Geiping sandwiches the shared stack between 2 prelude and 2 coda blocks. MoR keeps a distinct first
and last layer outside the recursion. SMELT loops the *middle* half.

We have a prelude (the 8-layer dense decoder) and no coda: the loop output goes through one RMSNorm
straight into `lm_head` and the MTP heads. The loop therefore does iterative refinement *and*
produces a readout-ready representation. Every published design separates those.

Per-loop CE supervision makes this worse rather than better. Every loop's hidden state is trained to
be directly LM-head-readable, which is a constraint on the loop's own representation, not just on
the final one. A coda gives the loop somewhere to be un-readout-like.

Cost: 1 to 2 non-shared dense blocks at `hidden=768` is ~14 to 28M parameters. Test as an arm, not
as a default, and test it **together with** the loop-3 question: a coda is one of the few plausible
explanations for why loop 3 is currently redundant.

### 3.2 No initial-state injection, still the highest value open item

Geiping's shared stack receives, on every pass, the prelude output concatenated with the previous
loop's hidden state, through a learned linear projection. The loop always has access to the original
input representation, not just its own last output.

We pass only the previous loop state. Cost of adding it: one `[1536, 768]` projection, ~1.2M
parameters, negligible FLOPs against a 35-expert pool.

This is now more interesting than it was before the run, because it is a direct candidate
explanation for §2.2. A loop whose only input is its own output has a drift problem: by pass 3 the
state is three refinements away from the evidence, and `loop_scale` decaying to 0.11 is what that
looks like from the outside. Injection is the published fix for exactly this shape.

**The case is weaker here than in Geiping, and the reason is structural.** Geiping's recurrent core
*replaces* its state on every pass, so without injection the input really is lost. Our loop is a
residual update, `h_{k+1} = h_k + loop_scale[k] * delta_k`, starting from the decoder output, so that
output is still most of the state at every loop: `‖Δh‖/‖h‖` measures 0.87 / 0.36 / 0.10 on
`phase2_final`. CrossAttention also already re-reads a per-token embedding (`_moe_ple(input_ids)`)
on every loop. Re-injecting the decoder output hands later loops the same information they already
have, while Stage 0's reading was that nothing feeds them anything *new* — which the evidence
buffer and the loop-conditioned query address directly. Still cheap and zero-init loadable, so it is
tested rather than argued: NEXT.md Phase 3c. Note that Phase 4 plans to swap `_moe_ple(input_ids)`
for evidence tokens, removing the one per-loop input re-read the model has today.

Ordering: test injection as its own short arm before the Phase 4 spend. Looping is a spec
requirement, so a negative result moves the depth burden to the depth curriculum and a coda rather
than cutting a loop.

### 3.3 The router does not specialize

Not a looping result, but it interacts with one. The aux loss sat at ~1.0, its balanced value, from
step 0 to the end, and mean routed weight is flat across all 35 experts. Selection fractions spread
(0.03 to 0.24, tightening to 0.05 to 0.10 after SFT) but the weights do not.

SMELT's recipe is "add experts to recover the parameters you gave up to looping". That trade assumes
experts specialize. At 32 experts, `top_k=2` and 16B tokens, each MLP expert sees roughly 1B tokens
and evidently learns something close to the same function. Adding experts on SMELT's advice would
buy storage for undertrained experts.

Two things to measure before touching expert count: whether `aux_loss_weight=0.01` is overpowering
differentiation, and whether the flat routed weight is an artefact of the single softmax being
renormalized after top-k.

### 3.4 Loop-index conditioning is unpublished

`loop_router_bias` (sinusoidal loop index added to the router logits) does not appear in any of the
systems in §1. Universal Transformer adds a timestep embedding to the *hidden state*, which is the
nearest relative, not the same thing. It is a sensible mechanism and it is ours, so it needs its own
ablation arm rather than inheriting anyone's evidence.

---

## 4. Where our run contradicts or extends the literature

### 4.1 The halt head failure generalizes, and the literature does not cover it

`p_halt` collapsed to ~0.004 during the zero-λ warmup, overshot to ~0.78 when the ponder ramp
engaged, pinned there for 14B tokens, and the controller cut `lambda_ponder` 11 times with no
effect. CONCLUSION.md's root cause is the right one and it generalizes past this codebase:

> `p_halt` gates the loop's **output**, not its **compute**.

UT and PonderNet halting is trainable because halting genuinely skips work. The ponder cost trades
against a real compute saving, so the optimizer faces a two-sided pressure and the sigmoid has
somewhere to sit. Here every expert ran on every loop regardless of `p_halt`, so the ponder term had
no counterparty, the gate saturated, and λ stopped being a control knob because a saturated sigmoid
has no gradient.

This is the training-time twin of the caveat on Ouro: its released implementation computes all
configured passes and *then* selects an output, so its advertised adaptive depth yields no
wall-clock saving. We hit the same bug during training rather than at inference.

Rule to carry forward: **a learned depth mechanism must skip computation, or it is not learnable.**
Any future halt head has to break the loop for real, which means solving the KV-cache hole in §4.3
first, not after.

### 4.2 The convergence exit is a stronger version of Geiping's criterion

Geiping halts when the KL between two successive passes' next-token distributions falls below a
threshold. Our exit reads the last position only and stops when the top-1 token is unchanged *and*
`|Δ log p_top| < converge_tol`. Same family, cheaper (no full distribution comparison), and the
extra top-1-stability condition makes it stricter.

The design note in `docs/moe.md` about reading the **readout** rather than `‖Δh‖` is worth keeping
visible: `loop_scale` still injects a sizeable hidden delta on the last loop while the prediction is
already stationary, so a hidden-state criterion never fires. That observation is not in any of the
papers and it is a real trap for anyone implementing this.

Measured transitions on the 16B checkpoints, loop 1→2 top-1 agreement ~0.82 with mean `|Δ log p|`
~0.21, loop 2→3 ~0.94 / ~0.07, are consistent with §2.2: by loop 3 the prediction is already made.

### 4.3 KV cache: we agree with Nanbeige by necessity

Looping saves parameter storage, not cache. Keys and values differ per pass because the inputs
differ, so `n_loops=3` needs 3x the cache of a single pass, the same as an unrolled stack. Nanbeige
tried sharing the cache across passes, halved the memory, got a worse model, and shipped the
unshared version.

`kv_cache.py` holds one slot per decoder layer plus one per `(loop, non-MLP expert)` pair and one
per `(loop, shared_attn)`, which is the correct layout and matches the published finding. Nothing to
change.

The open item is the interaction with §4.2: an exited loop appends no K/V, so the convergence exit
is asserted mutually exclusive with the cache. Geiping has the same problem and does not solve it
either. Options, cheapest first: run the skipped loops' attention experts only (K/V without the MLP
pool, which is most of the saving lost), or carry forward the last computed loop's K/V as the
entry for the skipped loops, which is wrong but possibly harmlessly so and is one experiment to
check. Until one of them lands, adaptive depth and cached decoding are exclusive and the exit is a
research instrument rather than a speedup.

### 4.4 We have a datapoint nobody has published

None of the papers in §1 report per-loop readout CE at sub-500M scale. We do, and it is a clean
saturation curve with the confound ruled out (§2.2). Combined with the depth ablation and an
`n_loops=1` compute-matched arm, that is a publishable contribution on where recurrent depth stops
paying for small models. It is also honest about a mechanism that did not work, which is the kind of
result the EuroHPC application already commits to reporting.

---

## 5. Change list

Landed, no action:

- loop residual with per-loop learned `loop_scale`
- identity expert deleted, always-on shared MLP and attention outside the router
- per-loop CE supervision with subsampling on non-final loops
- stochastic loop depth (`loop_count_sampling=0.3`)
- convergence exit at inference
- per-`(loop, expert)` KV cache slots
- `n_loops` 4 → 3, vocab prune, mmap dataset

Open, and where each landed in NEXT.md (2026-09-15):

1. **Initial-state injection into every pass** (§3.2). Zero-init `768 x 768` additive adapter,
   ~0.59M params. **NEXT.md Phase 3c**, Arm D against Arm C at matched tokens, Gate G2c.
2. **Coda blocks after the loop** (§3.1). **NEXT.md 7c**, a from-scratch A/B rather than a graft:
   per-loop CE and the convergence exit read `lm_head` at every loop, so the coda has to sit in
   front of every readout, and a zero-init graft at a finetune LR would understate it.
3. **Measure whether `aux_loss_weight` is suppressing expert specialization** (§3.3), before any
   decision about expert count. **NEXT.md 7c.**
4. **Loop-index conditioning ablation** (§3.4). **NEXT.md Phase 3c**, read-only on Arm C's final.
5. **Resolve the exit / KV-cache exclusion** (§4.3) if adaptive depth is meant to be a speedup
   rather than an instrument. This also gates any future learned halt mechanism (§4.1). **NEXT.md
   7a** (carry-forward measured first) and Parked.
6. **Depth ablation read on reasoning and multi-hop tasks** (§2.3). **NEXT.md Gate G5.**

Not adopted: a compute-matched `n_loops=1` arm (§2.2). Looping is a spec requirement for this model,
so the plan fixes weak later loops rather than testing whether to have them.

---

## 6. What the literature still does not answer

- **Where the crossover is below 1B.** MoR says vanilla wins at 135M. Our loop saturates at 2 passes
  at 332M. Nobody has published the curve in between.
- **Looping a single MoE block.** Every published looped-MoE result (SMELT) loops a stack. One block
  with re-routing each pass over a heterogeneous pool is unpublished; the nearest analogue is
  Universal Transformer, which loops one dense block.
- **Depth routing and expert routing together.** MoR routes depth, we route experts. Separate
  routers for both is untested. One router for both is what the deleted identity expert was, and it
  did not work.
- **Recurrent depth plus retrieval.** §2.3 argues they are complementary, since loops add compute
  and not storage. Nobody has measured the combination.

---

## References

- Raschka, *GPT-6 Astra, Looped Transformers, and Hidden Reasoning*, Ahead of AI, 2026-09-09,
  <https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and>
- Dehghani et al., *Universal Transformers*, 2018, <https://arxiv.org/abs/1807.03819>
- Nanbeige4.2-3B technical report, 2026, <https://arxiv.org/abs/2607.22083>
- Ouro (cited as "LoopLM" in our README), 2025, <https://arxiv.org/abs/2510.25741>
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*,
  2025, <https://arxiv.org/abs/2502.05171>
- Bae et al., *Mixture-of-Recursions*, 2025, <https://arxiv.org/abs/2507.10524>
- Zhu et al., *Beyond Parameters: Exploring Virtual Logic Depth for Scaling Laws*, 2025,
  <https://arxiv.org/abs/2506.18233>
- *SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers*, 2026,
  <https://arxiv.org/abs/2609.01343>
- *Full-bandwidth Transformer*, 2026, <https://arxiv.org/abs/2608.08888> (recurrence across token
  positions rather than depth, adjacent, out of scope)

`README.md` links arXiv 2510.25741 as "LoopLM". That paper is Ouro, which uses "LoopLM" as the name
of its model family, so the link is not wrong; naming it "Ouro (LoopLM)" before the EuroHPC
submission references it would avoid the ambiguity.

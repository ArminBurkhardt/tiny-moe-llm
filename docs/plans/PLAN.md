# PLAN.md

Plan for `tiny-moe-llm`, from first pretraining run through everything after it. **Read
[CLAUDE.md](CLAUDE.md) first** — it documents the invariants this plan produced and is the
authoritative reference for anything already built. The step-by-step implementation instructions
this file used to carry (Steps 1-11: loop residual, shared experts, halt head/ponder loss, per-loop
CE + correctness head, config assertions, token-count stop, instrumentation, vocab prune, mmap
dataset, FP8 wiring, data prep) are done and live in the code + CLAUDE.md now — see `git log -p --
PLAN.md` for the original text if the rationale behind a specific decision is needed.

## Rules (still apply to Steps 12+ below)

- Commit per logical change (`feat:` / `chore:`, branch `train-build`).
- Match existing conventions: lowercase explanatory comments justifying *why*, Google-style
  docstrings with `Args:` on public modules. Don't strip existing comments.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, or `moe` in the training
  loop goes through `accelerator.unwrap_model(model)`.
- **Never add `.item()`, `.tolist()`, `.cpu()`, or boolean mask indexing to the per-step path.**
  Host syncs dominate at this model size. Accepted exceptions: `m_splits` via `.tolist()`,
  `TokenTracker.sync()` at log/checkpoint cadence.

---

## Status: first full run complete (2026-08-08) — repair needed before Step 13

Config A' (768x8, M=32, V=65536, n_loops=3) trained for real: **332,324,717 total params / 173.1M
active (103.9M excl. embeddings)**, 16.0B combined pretraining tokens + a 708.9M-token SFT pass, on
one rented H100 NVL. Full loss curves, per-head diagnostics, and artifact locations are in
[ckpts/trained/temp-train/CONCLUSION.md](ckpts/trained/temp-train/CONCLUSION.md), which is now the
source of truth for "what actually happened." This section only keeps what's still relevant to
planning the *next* run.

Key sizing calls, since they're not written down elsewhere:
- `n_loops: 4 -> 3` — non-MLP experts run densely every loop, so fewer loops means fewer dense
  attention passes.
- `num_ir_entries: 16384 -> 8192` — the IR expert is ~11% of forward FLOPs (dense every loop);
  halving frees throughput and ~2GB peak memory.
- `num_mlp_experts: 32` — routed experts add total params at zero active compute, but each
  expert's training-data share shrinks with pool size. At M=32/~25B tokens each sees ~1.4B tokens;
  M=52 would drop that to ~0.9B (undertrained). 32 is the end of the free lunch.
- `num_attn_experts` stays **1** — attention experts run densely every loop and get masked by
  routing afterward, so extra ones mostly discard compute (~5% throughput per +1% params).
- `moe_intermediate_size` is the only clean total-vs-active knob; hit future param targets with it
  or `num_mlp_experts`, not `intermediate_size` (moves the fully-active dense decoder too).

**Local validation gates — all 5 passed** (pre-run, on synthetic/45M-token smoke data, kept as a
historical record of what was checked before spending the rental budget):

| gate | result |
|---|---|
| 1 — env check | pass |
| 2 — `test_attention_equiv.py` | pass |
| 3 — `test_overfit.py` | pass |
| 4 — 45M-token local run, real (not synthetic) Hub-sourced slice | mean loops 2.70, 59702 tok/s, MFU 31.8%, peak mem 25.45GB, 12.60min |
| 5 — `eval_calibration.py` on the Gate 4 checkpoint | ECE(p_correct)=0.013 (passes <0.15), but `p_max` beat `p_correct` on both ECE and abstention AUROC. Literal rule says revert Step 4b; **deferred** to the real final checkpoint |

Gate 5's deferred call is now resolved: `p_max` beat `p_correct` on the real checkpoint too (same
metrics, same margin) — see Step 12b below for the fix/revert decision this triggers.

**Budget was time-capped, not token-capped.** `config.yaml`'s `target_tokens: 29900000000` is
unchanged, but the run stopped at **16.0B tokens** (13.6B phase1 + 2.4B phase2) because the EUR 100
/ ~47h spend cap in the runbook below bound first — combined pretraining wall clock was ~45.7h.
Consequence: the LR schedule is anchored to 29.9B tokens, so at 16.0B the cosine had only reached
4.0e-5 (its floor) by luck of `phase1_fraction`/`target_tokens` arithmetic, not because the anneal
actually completed against the full prepared corpus. A future full-budget run needs either a larger
rental budget or a `target_tokens` matched to what the budget can actually reach — don't reuse
29.9B unless the budget also grows to ~90h.

**Data prep exercised for real against the Hub, then run at full scale**: `scripts/prepare_data.py`
built the complete `phase{1,2}.bin/.idx` corpus during the real run (49.40GB / 24.70B tokens phase1,
9.00GB / 4.50B tokens phase2 — see `manifest.json`). Source discovery, gated access, decompression
paths, column auto-detection, and the smoltalk2 holdout-hash recording all worked unattended. A real
mid-download `SIGKILL` interruption still wasn't exercised (only clean stop/resume) — still low risk
given `tests/test_prepare_data.py`'s synthetic coverage, but worth a wary eye if a preemption lands
mid-source on a future rerun.

---

## Vast.ai runbook — executed once (2026-08-08), template for future rentals

Results are in CONCLUSION.md; this procedure is kept as the template for any future rented run,
including the shorter/cheaper ones below (the repair finetune, the IR finetune). Steps 1-4 are
proven working as written and don't need re-verification on the next box of the same image/GPU
family.

1. **Rent**: 1x H100 SXM/NVL 80GB, **interruptible** (~half price; the per-step checkpoint/resume
   path exists to exploit this), **120GB disk** (fixed at creation, can't grow later), image
   `nvcr.io/nvidia/pytorch:25.xx-py3`. Confirm hourly rate; above ~EUR 2.2/hr re-read the budget
   table (config.yaml's `target_tokens` comment) and cut `target_tokens` first. Do **not**
   `pip install -r requirements.txt` wholesale — its TE/flash-attn wheels target sm120, H100 is
   sm90; the NGC image ships both prebuilt, layer only non-CUDA deps.
2. **Verify environment** before downloading 47GB:
   ```bash
   python -c "import torch, transformer_engine.pytorch, flash_attn; print(torch.cuda.get_device_capability())"
   bash tests/run_env_check.sh
   ```
   Expect `(9, 0)`. Catch a broken TE at minute two, not minute ninety.
3. **Data prep**: `python scripts/prepare_data.py`, in the instance's on-start script (unattended).
   Verify per-source token counts within 2% of target, `phase{1,2}.idx` monotonic with last entry
   == `len(bin)`, peak disk under ~70GB. Already validated at full scale — skip re-verifying the
   Hub-access/decompression paths on a rerun unless a source spec changed.
4. **Smoke test**: `USE_FP8=1 python scripts/pretrain.py`, kill after ~200 steps. Confirm
   `dry_run` asserts finite loss, FP8 actually active (TE warns on silent fallback), tokens/sec in
   range. **Note the real run never set `USE_FP8=1`** (CONCLUSION.md: "BF16 — `USE_FP8` was never
   set") — FP8 is still unexercised end-to-end on this model; a future run that wants the H100's
   FP8 throughput needs this smoke test actually run with it on, not just available.
5. **Phase 1 / Phase 2**: unchanged from before — 85/15 split, cosine anchored to combined
   `target_tokens`, router noise anneal from live token count. Actually observed: most of the loss
   descent happens by ~3B tokens (11.13 → 3.70 nats), phase 2's drop at the corpus switch (3.359 →
   3.046 within ~200M tokens) is a distribution change, not new learning — flat over its last 1.4B.
6. **Extraction**: scp checkpoints down as written. Already done for this run
   (`ckpts/trained/temp-train/`). For a future run, still re-run `eval_calibration.py` before
   releasing the instance and record the numbers — re-renting later to compute them is wasteful.

---

## Monitoring reference

**Every `LOG_INTERVAL`:** loss, tokens/sec, peak mem, `loop_scale`, current `lambda_ponder`, mean
`p_halt`, mean `p_correct`, mean `p_max`, batch top-1 accuracy, per-loop CE (all `n_loops`), aux
loss, ponder loss, conf loss.

**Every checkpoint:** `ckpts/training/expert_selection_*.png` from `_ExpertTracking`.

| symptom | likely cause |
|---|---|
| `loop_scale` stuck at init **and** `p_halt` climbing | ponder deadlock — warmup not wired to token counter |
| `loop_scale` stuck at init, `p_halt` in range | residual wiring wrong |
| `p_halt` saturated at 1 | `lambda_ponder` too high, or ramping too early |
| `p_halt` pinned at 0 | halt head not receiving gradient |
| `lambda_ponder` constant in the log | warmup reading a stale/absent token count |
| per-loop CE flat across loops | hidden states not threaded through, **or** final `RMSNorm` not applied per loop |
| per-loop CE huge at loops 0-1, normal at last | final `RMSNorm` not applied to intermediate states |
| `p_correct` far from top-1 accuracy | label mask or target wrong |
| `p_correct` collapses to a constant | `lambda_conf` too low, or gradient leaking through `is_correct` |
| `p_correct` tracks `p_max` exactly | head learned nothing beyond baseline — **this is what actually happened**, see Step 12b |
| routed expert weights near zero, loss still falling | shared MLP/attention swallowed the block |
| expert selection collapses to a few MLP slots | aux loss weight too low |
| tokens/sec drops after a code change | a host sync entered the step path |
| loss spikes on resume | schedule re-anchoring or `global_offset` wrong |

`_ExpertTracking` guards against activation-checkpoint recompute double counting via
`begin_forward(expected_updates)`, samples every 8th forward. If expert counts look wrong after
any loop-structure change, check `expected_updates` matches the current `n_loops`.

---

## Post-training

**Target for Steps 12-13: calibrated abstention, not chain-of-thought.** Calibrated "knows when it
doesn't know" is a shallower, directly measurable (ECE, abstention precision/recall) capability
that the halt/correctness machinery is actually positioned to deliver — multi-step reasoning is
not learnable at 332M total/174M active by SFT alone. Reasoning is revisited explicitly, and
gated, in Step 17 — *after* Step 16's RL gate has real evidence to check against, which is the
whole reason it's ordered last rather than skipped.

### Step 12 — SFT — **done, passed the literal acceptance, failed behaviorally**

`scripts/sft.py` ran for real: 708.9M SFT tokens (2 epochs over a 358.8M-token corpus), 5h40m
local, reusing `pretrain.train_step` verbatim so `p_halt`/`p_correct` supervision stayed identical
to pretraining. Val CE 1.990 → 1.785 (ppl 5.96), top-1 62.1%.

| dataset | role | realized share |
|---|---|---|
| `HuggingFaceTB/smoltalk2` (no-think splits) | general instruction following | 48.9% |
| `HuggingFaceH4/ultrachat_200k` | general instruction following | 27.9% |
| `allenai/tulu-3-sft-personas-math` | short worked solutions | 14.0% |
| `rajpurkar/squad_v2` | **primary abstention supervision** | 7.5% tokens / **25.6% of conversations** |
| `HuggingFaceH4/no_robots` | human-written; tone and refusal style | 1.5% |
| `openai/gsm8k` (socratic) | short numbered steps | 0.4% |

**Literal acceptance passed** (`scripts/eval_abstention.py`: precision/recall reported, ECE change
-0.0233, i.e. didn't degrade). **Behaviorally it did not work**: the model abstains on 80.2% of all
SQuAD v2 validation questions, including 78.4% of the *answerable* half — a near-degenerate refuser,
not a calibrated one. Full breakdown, root causes, and the diagnostic plots are in CONCLUSION.md's
"Failure: the abstention mechanism" section. Step 13 as specified (<10% false abstention on the
high-pass-rate bucket) would fail hard against this checkpoint — see Step 12b.

### Step 12b — Repair the abstention mechanism (blocks Step 13)

Three independent failures stacked to produce the 80.2% refusal rate: the SFT data rewards
refusal, the correctness head learned nothing beyond the free baseline, and the halt head
saturated and stayed there. They need three independent fixes; none of the three alone explains
the collapse, so none of the three alone fixes it. **Whichever combination is attempted, the
acceptance gate is the same**: re-run `scripts/eval_abstention.py` end to end and check the
answerable-half false-abstention rate specifically, not just precision/recall in isolation
(precision at the current 0.512 is barely above the 0.501 base rate — a metric that can pass while
the behavior is still degenerate).

**Ordering matters more than usual here.** 12b-iii (the data fix) is *upstream* of both head
redesigns: a confidence probe fitted on a model that refuses 78% of answerable questions is fitting
a broken policy, and a halting criterion tuned against it inherits the same distribution. The
recommended sequence is 12b-0 (measure) → 12b-iii (repair finetune) → re-measure → then decide
whether 12b-i/ii are worth building at all. 12b-iv and 12b-v are cross-cutting notes that apply to
whichever finetune gets run.

#### 12b-0. Measure before building anything

Every route below is gated on numbers that don't exist yet and that the **existing checkpoint** can
produce in minutes. Do this first; 12b-ii and 12b-iv may close outright on the results.

1. **Is loop 3 idle or churning?** Per loop, log `‖Δh‖ / ‖h‖`, `cos(Δ_k, Δ_{k-1})`, and the **top-1
   flip rate between loop 2's and loop 3's readouts** (the per-loop hidden stack is already returned
   — `hidden_states_all`, `modules/model/moe.py`). Note the run's effective per-loop gate is
   `(1 - p_halt) * loop_scale` ≈ `[0.38, 0.40, 0.29]`, *not* `loop_scale`'s `[1.73, 1.81, 1.32]`:
   loop 3 writes a ~0.3-RMS-relative update while per-loop CE doesn't move (3.109 / 2.969 / 2.969).
   Flip rate ≈ 0 → loop 3 is a genuine no-op, ship `n_loops=2`, and 12b-ii item 1 is settled. Flip
   rate high with CE flat → loop 3 is churning between equally-good predictions, which is a
   different problem and rules *out* the noise/dropout route in 12b-iv.
2. **Oracle minimum sufficient depth.** On held-out data run `n_loops = 1, 2, 3` (the runtime
   override already exists) and per token record the smallest depth whose argmax matches the label,
   bucketing "never correct" separately. This histogram *is* the answer to "do complex tokens need
   more loops" — and given loop 3 contributes ~0 nats, expect the "3 helps where 2 doesn't" bucket
   to be small and roughly cancelled by tokens where 3 *hurts*. If so, adaptive depth has no
   headroom at this scale and 12b-ii items 3-6 close for a few GPU-minutes. If there is headroom,
   the same labels are the training signal (see 12b-ii).
3. **Does any confidence signal carry answerability information at all?** On the real checkpoint,
   AUROC of `(1 - p_max)` for flagging an unanswerable SQuAD v2 question was **0.457 — below
   chance**, and `(1 - p_correct)` was 0.457 too. Whatever replaces the correctness head has to beat
   0.5 on that task before anything is built on top of it. Measure the candidate features
   (entropy, top1-top2 margin, cross-loop KL — see 12b-i) against that bar directly.

#### 12b-i. Correctness head: revert (spec says to; redesign only if retried)

**What it does, architecturally**: `self.correct_proj` (`TinyMoETransformer`, zero-init weight/
bias) is applied to the **final loop's** hidden states inside `compute_mtp_loss`, alongside
`lm_head`. Its BCE target `is_correct` is computed under `torch.no_grad()` from that same chunk's
teacher-forced CE logits — literally `argmax(logits) == labels` on the reference continuation the
model is being shown, not on anything it generated itself.

**Why it failed**: that target is a near-deterministic function of the exact hidden state
`correct_proj` reads — the head and `lm_head` share the same input, and the label is derived from
`lm_head`'s own output on that input. The BCE optimum reachable from that setup is "reproduce
`p_max`" (softmax confidence), because that's the best predictor of teacher-forced correctness
available *from that hidden state alone*, and it's exactly what the free `p_max` baseline already
computes for nothing. There's no additional signal in the inputs the head was given, so there was
never a mechanism by which it could beat `p_max` — the gradient isolation (`no_grad` on the target)
works exactly as designed (`tests/test_correctness_head.py` still passes), the *targeting* is what's
wrong. On the real checkpoint: answer-level ECE 0.378 vs. `p_max`'s 0.371 (baseline still wins),
AUROC 0.604, and — the actual failure mode for an abstention use case — AUROC of `(1 - p_correct)`
for flagging unanswerable questions is **0.457, worse than chance**, with mean `p_correct` *higher*
on abstentions (0.835) than on real answers (0.739). The head is most confident exactly when it's
refusing.

**This is Gate 5's deferred call, now resolved on the real checkpoint** (see memory
`project-step4b-correctness-head-deferred`): `p_max` won on both ECE and AUROC, same as at Gate 4.
Per PLAN.md's own original revert criterion, **the head, its loss term, and `lambda_conf` should be
reverted** — remove `correct_proj`, the BCE term in `compute_mtp_loss`, `TrainingConfig.lambda_conf`,
and substitute `p_max` everywhere the head's output was read (`sft.py`'s val logging,
`eval_abstention.py`, `eval_calibration.py`). This is a subtraction, not a redesign, and it's
compute-free either way (the head was gradient-isolated and cheap), so there's no cost to reverting
now and revisiting later if a genuinely different design is worth trying.

**If retried instead of reverted**, the fix has to remove the shared-input problem, not just retune
`lambda_conf`:
- **Sequence-level target**: "was the whole generated answer right", not per-token teacher-forced
  correctness — the current target measures next-token prediction quality on a reference the model
  didn't generate, which is a different question from "should this specific answer be trusted."
- **Sampled, not teacher-forced**: label from the model's own greedy/sampled completion vs. ground
  truth, so the head sees the model's actual error distribution at decision time instead of an
  in-context-cheating proxy.
- **Give it features `lm_head` doesn't already imply**: margin between top-2 logits, entropy of the
  full distribution, agreement across k sampled continuations — anything that isn't a monotonic
  function of the same hidden state `p_max` is already computed from, since that's the only way to
  *add* information rather than reproduce it.

Any redesign attempt is gated behind actually beating `p_max` on both ECE and AUROC on a held-out
set before it's wired back into `eval_abstention.py`'s reported numbers — same bar as the original
Gate 5, applied honestly this time.

**Preferred replacement: a frozen-backbone probe over cross-loop disagreement.** Of the "give it
features `lm_head` doesn't imply" options above, entropy and top-2 margin are the weak ones — they
are functions of the same logits `p_max` comes from, i.e. the same family that already lost (ECE
0.371 vs. 0.378). The signal this architecture provides **for free and that `p_max` provably cannot
contain** is disagreement across *depth*: `KL(p_loop2 ‖ p_loop3)`, top-1 agreement between
consecutive loops, optionally across a few `n_loops` overrides. The recurrence is a free
depth-ensemble and its spread is a genuine epistemic-uncertainty signal, computed from hidden states
`p_max` never sees. It also shares its entire computation with 12b-ii's convergence-exit criterion —
one measurement, two consumers.

Shape of the probe (this is CONCLUSION.md's "decouple abstention from generation", made concrete):
- **Backbone frozen, fitted offline.** Minutes of compute, no training-loop change, no new loss term
  in `pretrain.train_step`, and it yields a *tunable* precision/recall operating point instead of one
  policy baked into the weights.
- **Features**: `[final hidden state, p_max, entropy, top1-top2 margin, cross-loop KL, cross-loop
  top-1 agreement]`.
- **Targets from the model's own sampled generations vs. ground truth, sequence-level.** Not
  teacher-forced argmax — that specific choice is the leak that sank Step 4b, and reproducing it
  behind a fancier classifier reproduces the failure.
- **Held-out fit and eval**, and the same Gate 5 bar (beat `p_max` on ECE *and* AUROC) before it
  enters `eval_abstention.py`'s reported numbers. Per 12b-0 item 3, its *first* job is beating 0.5
  AUROC on answerability at all.

Fit the probe **after** 12b-iii's repair finetune, not before: on the current checkpoint it would be
fitting a model whose policy is "refuse", and the operating points would not survive the data fix.

#### 12b-ii. Halt head: give it real compute authority, replace the control loop, or drop it

**What it does, architecturally**: `moe.py`'s `forward_step` computes `p_halt` from the *incoming*
hidden state every loop (`halt_proj`, zero-init weight, bias `-2.0`), then applies it as an output
gate: `hidden_states = hidden_states + (1 - p_halt) * loop_scale[loop] * dropout(post_norm(output))`.
Crucially, **every expert still runs, densely, regardless of `p_halt`** — halting suppresses the
*update*, not the *computation*. The only thing pushing `p_halt` away from a trivial constant is
the ponder loss (`lambda_ponder * mean(1 - p_halt)` on real tokens), and CE has near-zero gradient
w.r.t. `p_halt` once `loop_scale` is doing the real work of controlling output magnitude (this is
the documented ponder-deadlock precondition in CLAUDE.md, and it's exactly what happened).

**Why it failed**: `p_halt` collapsed to ~0.004 during the zero-λ warmup — pure CE pressure with no
ponder gradient pushed the halt bias down, since a near-zero gate lets the loop's residual update
through unimpeded, which is what CE alone rewards. The moment the ponder ramp engaged, `p_halt`
overshot straight past the target band (0.30 ± 0.12) to **~0.78** and pinned there for the
remaining ~14B tokens. The auto-adjust controller did exactly what it's supposed to — cut
`lambda_ponder` 11 times, 0.15 down to its 0.01 floor — with **no measurable effect on `p_halt`**.
That's the tell: a sigmoid pinned at an extreme has a gradient near zero with respect to its
pre-activation, so once `p_halt` overshot into that regime, no amount of retuning `λ`'s *magnitude*
could pull it back — only its *sign* would matter, and the controller only ever nudges magnitude.
The model found a different way to satisfy both pressures simultaneously: `loop_scale` grew from
its 0.578 init to `[1.73, 1.81, 1.32]`, i.e. the loop learned to control its own contribution
through the multiplicative gate that *does* have a live gradient, leaving `p_halt` as dead weight.
Net result: `p_halt` is a constant, useless both for early-exit and as an abstention signal, and
`lambda_ponder`'s 11 downward adjustments spent controller budget on a knob that had already stopped
doing anything.

**Root architectural issue**: `p_halt` gates output, not compute, so there is no actual FLOPs
saving on the table for CE to trade against — the only cost function that cares about `p_halt` at
all is the hand-tuned ponder term, and once that term's own gradient into `p_halt` vanishes (via
sigmoid saturation), nothing pulls it back. This is a structural gap, not a hyperparameter miss.

**Fixes, cheapest/least invasive first**:
1. **Measure the honest baseline first, before redesigning anything**: per-loop CE readouts already
   show loop 3 contributes ~0 nats (loop 1 = 3.109, loop 2 = 2.969, loop 3 = 2.969 at the end of the
   real run) — a **static `n_loops=2`** config may already recover essentially all the quality this
   halt mechanism was meant to buy adaptively, for free (no halt head needed at all). Run this
   ablation before investing in a harder fix; it sets the bar any dynamic-halting redesign has to
   clear to be worth the complexity.
2. **Drop the head entirely and exit on convergence instead** (recommended; strictly cheaper than
   items 3-6 and evaluable on the *existing* checkpoint). "Was the last loop a no-op? then stop" is
   a criterion, not a learned policy: zero parameters, zero loss terms, no `lambda_ponder`
   controller, no sigmoid that can saturate, and a single tunable threshold instead of a
   hyperparameter that has to be found by training. Three details decide whether it works:
   - **Measure convergence in the readout, not the hidden state.** `loop_scale[2] = 1.32` means the
     loop-3 hidden delta is large while the *prediction* is stationary, so a `‖Δh‖` criterion would
     never fire. Use `KL(p_k ‖ p_{k-1})` or top-1 agreement between consecutive loops' `lm_head`
     outputs.
   - **The "that's a vocab projection per loop" objection does not apply at generation time**, because
     only the *last* position needs a readout: 1 token x 65536, i.e. free. Under teacher forcing the
     per-loop readouts already exist for per-loop CE. It only becomes expensive if applied per-token
     across a full packed batch, which is not what an exit criterion needs.
   - **Trap when removing the head**: `(1 - p_halt)` is pinned at ~0.22 and `loop_scale` grew to
     absorb it. Deleting the gate multiplies every loop's delta by ~4.5x and the checkpoint stops
     working — fold the constant into `loop_scale` rather than dropping the term.
   Real compute is only saved if the remaining loops are actually skipped. For batch-1 generation a
   whole-sequence exit decision at the last position is trivial; per-token skipping inside a batch is
   item 3's problem.
3. **Make halting actually skip compute.** Currently `p_halt` can only ever be a soft signal because
   masking output post-hoc changes nothing about cost. Gathering/masking so halted tokens' experts
   genuinely don't run in a later loop turns `p_halt` into a real compute/quality trade-off with a
   live CE gradient (a token that halts and then would have benefited from another loop actually
   loses accuracy, which is signal CE can use) — this is the only fix that addresses the root cause
   rather than the symptom. It's also the most invasive: it interacts with the mask-multiply /
   grouped-GEMM machinery `moe.py` already uses for expert sparsity, and needs a per-token variable
   loop count within a single batch, not just across batches (`loop_count_sampling` already varies
   depth *per step*, this would need it *per token*). Concretely, where the difficulty actually is:
   the routed MLP path is fine (the grouped GEMM is already ragged over sorted assignments, so a
   shorter active-row list is just smaller `m_splits`), but the **non-MLP experts are the dense
   per-loop cost** and they need full K/V with queries only for still-active tokens. varlen supports
   differing q/kv lengths, but combined with document packing (`cu_seqlens` would need separate q and
   kv offset arrays) and `shared_attn` on the same path, that's the real work. It's also where the
   savings are, so it can't be skipped by doing the easy half.
4. **Replace the λ-nudge controller with a Lagrangian on an explicit compute/halt budget** (dual
   ascent: increase the multiplier when the constraint is violated, decrease when it's slack) —
   still an output-gate, but a properly constrained optimization instead of a heuristic EMA nudge
   that has no way to recover once the primal variable (`p_halt`) saturates.
5. **Switch to cumulative ACT** (Graves-style: halting probabilities that accumulate across loops
   and must sum to ≤1, with a ponder cost on the number of loops actually taken) instead of the
   current greedy-per-loop formulation (`p_halt` recomputed fresh each loop, so a token can halt at
   loop 1 and un-halt at loop 2 — CLAUDE.md flags this explicitly). ACT's cross-loop normalization
   is what gives the halting decision a coherent "when do I stop" semantics; the current design
   doesn't actually ask that question per loop, so it's unsurprising the answer collapsed to a
   constant.
6. **Supervise depth directly instead of inducing it through a penalty.** 12b-0 item 2's oracle
   histogram is already a per-token label ("smallest depth whose argmax matches the label"), so the
   halting decision can be trained as plain classification against it rather than coaxed out of a
   hand-tuned `lambda_ponder`. This removes every failure mode documented above at once: no `λ`, no
   ramp, no saturating sigmoid, no coupling to the main loss, and it fails safe — an uninformative
   classifier degrades to fixed depth rather than collapsing the loop. It answers "which tokens need
   more refinement" the way the question was actually meant, and pairs naturally with item 3 (labels
   say *when* to stop; item 3 makes stopping *cheap*). It does need a label-generation pass over
   held-out data at each depth, which is the same pass 12b-0 item 2 already runs.

Item 1 is a half-day ablation with the existing checkpoint architecture (just force `n_loops=2` at
inference and re-run eval); item 2 is the same order of effort and is the recommended landing spot
if 12b-0 shows the recurrence converges. Items 3-6 are real training-loop changes and should only be
attempted if 12b-0's oracle histogram shows a real per-token depth gap worth recovering adaptively.

#### 12b-iii. SFT data: stop rewarding refusal

Independent of both heads — this is what actually produced the 80.2% abstention rate and the
five-phrasing generation collapse (7,786/11,873 completions are literally `"The passage doesn't say."`).
**Root cause**: SQuAD v2 is 7.5% of SFT tokens but 25.6% of conversations, and its unanswerable
third is a ~6-token, extremely low-entropy target. Per-token CE makes a short memorized refusal the
cheapest available loss reduction, and nothing in the rest of the mix penalizes refusing an
answerable question — smoltalk2/UltraChat/personas-math don't contain a competing "you should have
answered this" signal for QA-shaped prompts.

**Fixes, in order of expected effect per CONCLUSION.md**:
1. Down-sample SQuAD v2's unanswerable rows to ~10-15% of the QA subset instead of its natural
   ~33-50% split — matches roughly what Step 13's target distribution should look like anyway.
2. Add answerable-only extractive QA sources (SQuAD 1.1, NQ-open, TriviaQA, HotpotQA) so the
   answerable/QA-shaped-prompt volume isn't dwarfed by SQuAD v2 alone.
3. **Weight the loss per conversation, not per token**, so a 6-token refusal stops mechanically
   out-earning a multi-sentence real answer purely by being short and low-entropy.
4. Vary the abstention *training* phrasings (a larger or generated set) while keeping the closed
   set in `modules/data/abstention.py` for eval-side `is_abstention` detection only — reduces the
   incentive to memorize one exact string as the global CE optimum.

**Cheapest path to a re-test**: a short targeted repair finetune on the existing SFT checkpoint
(~20-50M tokens, `lr=1e-5`, 1 epoch) over a rebalanced answerable/unanswerable set, rather than a
full 708.9M-token SFT rerun — the model is already chat-formatted, so this is hours on a rented
box, not days, and reuses the Vast.ai runbook's shorter-run template.

#### 12b-iv. Loop refinement: latent noise / dropout (rejected), input injection (open)

The question this answers: *should a further finetune inject a noise vector or extra dropout into
the latent between loops, to force each loop into "actual refinement"?*

**Rejected as specified — it treats a failure mode this run doesn't have.** The diagnostics say the
loop isn't idle: effective per-loop gate `(1 - p_halt) * loop_scale` ≈ `[0.38, 0.40, 0.29]`, so loop
3 writes a ~0.3-RMS-relative update into the residual stream and per-loop CE doesn't move. The
failure is "the loop does work `lm_head` is blind to", not "the loop declines to do work". Noise
injection adds a *denoising* task that exists only at train time (nothing is injected at eval), so
it spends loop capacity repairing damage that isn't there at inference — a train/test mismatch, not
a refinement pressure. The random-`h₀`-plus-randomized-depth variant that does have a track record
(latent-recurrent-depth models, where it buys path-independence / fixed-point behaviour) is a
*pretraining-time* property; it cannot be installed by a 20-50M-token finetune at `lr=1e-5`.

Related facts worth not re-deriving:
- **Dropout on the delta already exists** — `moe.py`'s `forward_step` is
  `self.dropout(self.post_norm(output))`, at `dropout: 0.00` in pretraining and `0.05` in SFT.
  Raising it is a one-line config change if the hypothesis is worth a cheap test; expect nothing.
- **Input injection already exists, but conditionally.** The cross-attention expert receives
  `other=self._moe_ple(input_ids)` (`transformer.py`), i.e. a fresh view of the input on every loop —
  structurally the same trick as "inject `e` at every recurrent step". But cross-attn is one routed
  expert out of 35 at `top_k=2`, so its contribution is gated toward 0 much of the time. **If one
  architectural change in this family is made, make the injection unconditional** (alongside
  `shared_mlp` / `shared_attn`, which already seed the accumulator every loop) rather than adding
  noise. That gives every loop a stable anchor to refine *against*, which is the mechanism the noise
  idea was reaching for.
- **The tension that can't be dodged**: `loop_ce_weights: [0.2, 0.3, 1.0]` trains loop 1 to already
  be a usable readout. Making later loops matter more means making *early* readouts worse (lower
  those weights) — which is exactly what makes 12b-ii item 2's early exit less viable. "Loop 1 reads
  out well" and "later loops do a lot" are the same knob pointed in opposite directions; the next run
  has to pick one.

Gate: 12b-0 item 1. Flip rate ≈ 0 → the recurrence has converged, ship `n_loops=2`, nothing here
applies. Flip rate high with CE flat → loop 3 is churning between equally-good predictions, which
noise would make worse, not better.

#### 12b-v. MTP: already inference-only as an output, but not free

MTP was only ever a training-time objective for richer latents, and it *is* already unused as an
output at inference — `TinyMoETransformer.forward` returns `lm_head(x)` from the final loop and
nothing reads `extra_token_outputs` outside `compute_mtp_loss`. Two things are still worth changing:

1. **`_mtp_forward` runs unconditionally** (`transformer.py`, both the checkpointed and plain
   branches) and `scripts/inference.py` discards the result. With `late_token_loss=True` that's only
   the gate/up/down MLP (no vocab projection), but `inference.py` has no KV cache and re-runs the
   full prefix per generated token, so the waste is paid over the whole prefix every step — and
   `eval_abstention.py`'s batched decode inherits it. Guard the call on `self.training` (or an
   explicit caller flag) — a small, self-contained inference win.
2. **`lambda_mtp: 0.0` does NOT skip the compute.** `compute_mtp_loss` gates on
   `mtp_outputs is not None`, so a zero weight still pays the full head plus its `num_extra_tokens`
   chunked vocab projections — which per CLAUDE.md's accounting cost 4x (fwd + recompute + bwd).
   To actually turn MTP off for a finetune, pass `mtp_outputs=None` / skip `_mtp_forward`; keep the
   weights on disk so the checkpoint stays loadable and MTP can be switched back on.

**Recommendation**: drop MTP for a *behavioral* repair finetune (12b-iii). It was a regularizer
present through all 16B pretraining tokens, so removing it does shift the objective the trunk sits
in — but at 20-50M tokens and `lr=1e-5` that drift is negligible against a real throughput win. Keep
it if any further *general* pretraining is done. Not worth touching the data pipeline over: the SFT
dataset's `num_mtp_tokens` separator slots become plain padding when MTP is off, at a cost too small
to justify a rebuild.

#### Acceptance

**For 12b overall**: re-run `scripts/eval_abstention.py`; answerable-half false-abstention
rate materially below the current 78.4% (Step 13's own bar is <10%, which is the real target to aim
for even though 12b's job is just to get off the degenerate floor); abstention precision clearly
above the ~0.50 base rate; if the correctness head is kept rather than reverted, `p_correct` must
beat `p_max` on both ECE and AUROC on this checkpoint too.

That gate belongs to 12b-iii, which is the only sub-item that can move it. The others carry their
own, narrower gates and none of them substitutes for it:

| sub-item | gate |
|---|---|
| 12b-0 | none — it's the measurement pass everything else is conditioned on |
| 12b-i (probe) | beats 0.5 AUROC on answerability, then beats `p_max` on ECE *and* AUROC, held out |
| 12b-ii (exit / depth) | a real quality gap between `n_loops=2` and `3` in 12b-0's oracle histogram |
| 12b-iii (data) | the 12b acceptance above |
| 12b-iv (injection) | 12b-0 item 1 shows the recurrence has *not* converged |
| 12b-v (MTP) | none — a correctness/throughput cleanup, measured by tokens/sec and unchanged eval |

### Step 13 — Self-labelled calibration set

**Blocked on Step 12b landing a non-degenerate baseline first.** Building this dataset from the
current checkpoint would just re-encode the collapse: with 78.4% false abstention already, most
answerable questions would sample a low empirical pass rate for reasons that have nothing to do
with genuine uncertainty, and the rewritten targets would teach the same refusal reflex with extra
steps.

The one dataset worth building rather than downloading — requires this model. New script
`scripts/build_calibration_set.py`.

1. Sample the Step 12b checkpoint N=16x at temperature 0.8 on short-answer QA (`trivia_qa`,
   `nq_open`, `squad_v2`).
2. Label each question by empirical pass rate (normalized exact/alias match against reference).
3. Rewrite targets by pass rate: `>0.8` -> the answer; `<0.2` -> an abstention; in between -> a
   hedge, from a small fixed set of phrasings (not free text).
4. Hold out 10% before rewriting (the calibration eval set).
5. Second SFT pass on the rewritten data.

**Acceptance:** abstention rate on the held-out low-pass-rate bucket >60%; on the high-pass-rate
bucket <10% (catches the degenerate "refuse everything" solution — this is the exact number Step 12
failed); ECE of the abstention signal improves relative to Step 12b.

### Step 15 — Information-retrieval expert: train toward genuine fact storage

**Current state** (`modules/model/information_retrieval.py` + `experts.py`): one IR expert,
`down_proj` 768→128, a learned table of 8192 key/value pairs at `ir_dim=128`, cosine-similarity
softmax retrieval, `up_proj` 128→768. ~2.3M params, ~5% of forward FLOPs, runs densely every token
every loop; the router only weights its contribution.

**What the real run shows the router wants it, and why it can't deliver yet**: the router
consistently gives the IR expert ~7-9% of routed weight against a 5.7% uniform share (2/35) — one
of the three most-selected slots in the pool — but `temperature=1.0` is hardcoded at construction
(`experts.py`'s `InformationRetrievalModule(..., temperature=1.0)`), and cosine similarity logits
are bounded in [-1, 1]. A softmax over 8192 entries whose logits span at most 2.0 is nearly
uniform: the retrieved vector is close to a constant `mean(y_values)`, so the expert is very likely
operating as a learned bias plus a projection today, not as retrieval of any specific fact.

**Extensions, cheapest first — all reachable by finetuning an existing checkpoint, none require
retraining from scratch:**

1. **Lower the retrieval temperature** (`temperature` ≈ 0.05-0.1 in `experts.py`'s construction
   call, or promote it to a learned `nn.Parameter` and anneal it). This is a scalar, not a shape —
   the existing checkpoint loads unchanged with the constructor value bumped. Cheapest possible
   experiment: no retraining needed to *see* the effect (swap the constant, run inference, inspect
   `return_weights=True` output for whether retrieval sharpens); a short finetune to let the rest of
   the model adapt to sharper retrieval is the real test. Do this first — everything past it is
   wasted effort if retrieval never sharpens.
2. **Sparsify the read** (top-k ≈ 32 instead of a full softmax over 8192). Turns a dense
   8192 × 128 matmul into a gather, which is what makes item 3 affordable at a larger table size.
3. **Grow the table.** `z_keys`/`y_values` are plain `[num_entries, ir_dim]` parameters — append
   rows initialized from the existing distribution, load the old checkpoint with `strict=False` on
   just those two tensors. Every projection shape is untouched. 8192 → 65,536 is ~16.8M extra
   params (~34MB bf16) and, with top-k reads from item 2, roughly flat FLOPs.
4. **Warm-start the table from real knowledge** instead of noise: `z_keys` from encoded
   entity/definition text (e.g. Wikipedia titles/first-sentences through the model's own
   embeddings), `y_values` fit to what the LM should emit for that entry. Converts a random memory
   bank into an actual index rather than hoping training discovers structure from scratch.
5. **IR-only finetune**: freeze everything except `z_keys`/`y_values`/`g_proj`, train against a
   factual corpus (Wikipedia, TriviaQA) at a higher LR for just those tensors. This is the step that
   directly targets "efficiently store uncommon facts" — the table is exactly the component that
   should absorb them, and freezing the rest avoids disturbing chat behavior fixed in Step 12b.
   **Gate**: measure exact-match on a held-out rare-entity QA slice (e.g. PopQA, which is
   specifically long-tail) before vs. after — if IR-only finetuning doesn't move that number, the
   bottleneck isn't capacity/isolation and items 1-4 need revisiting before spending more here.
6. **More IR experts** (`num_ir_experts > 1`) — the one invasive option: shifts `first_mlp_index`
   and the router's output dimension, needs a surgical remap of the router weight and expert list
   rather than a plain checkpoint load. Prefer 1-5 first; only worth it once a single sharpened,
   grown table demonstrably helps and is saturating.

**Sequencing note**: this step is independent of Step 12b/13 (different subsystem, no shared
tensors) and can run in parallel on its own short rented session using the Vast.ai runbook's
template, scaled down to a few hours.

### Step 16 — RL: deferred, gated

**Do not start.** Evidence at this scale is consistently negative: a 135M single-GPU RLVR study
on GSM8K went from SFT base 24/1319 (1.82%) to 21/1319 at 192-token completions and 16/1319 at
320 — GRPO made it *worse*. On Qwen2.5-0.5B base, even a format reward stayed below 0.1 after 300
steps with no upward trend. Mechanism: under a 0/1 reward, a base model that can't sample correct
solutions produces no gradient signal, and RLVR only amplifies what's already in the base
distribution — at 174M active/~16-30B tokens there's little to amplify.

**Gate:** pass@8 on the target task, measured on the checkpoint that comes out of Step 12b/13 (a
checkpoint that still refuses 78% of answerable questions would fail pass@8 for the wrong reason —
it never attempts, which isn't the same failure RLVR evidence is about). **Below ~15%, do not
proceed** — the budget would be spent confirming the null result.

If the gate ever passes: vanilla GRPO is architecturally mismatched to a looped model (credits
output tokens while the computation is latent). Read LoopRPT (arXiv 2603.19714) and RLTT first —
they assign reward to per-loop latent states and report improved gate calibration (more early
exits, final-step dominance maintained) as a side effect, directly relevant to whatever the halt
head became after Step 12b-ii.

### Step 17 — Reasoning training: deferred, gated on Step 16's pass@8 result

**Do not start before Step 16's gate resolves.** This is the direct answer to "should this model be
fine-tuned for reasoning": right now it isn't (SFT deliberately used only smoltalk2's `_no_think`
splits, and `modules/data/chat.py`'s template has no reasoning/answer segment distinction), and the
Small Model Learnability Gap cited in Step 12 — at this scale, long-CoT imitation tends to teach
fluent filler before a wrong answer rather than actual multi-step reasoning — applies just as much
to a dedicated reasoning pass as it did to ruling long-CoT SFT data out of Step 12's mix. Step 16's
pass@8 gate is the empirical check on whether that concern is founded for *this* checkpoint before
spending a training run on it.

**If Step 16's gate fails (pass@8 < ~15%)**: reasoning training is not worth attempting by any
method at this size/token budget — stop here, the model's base capability doesn't support the
amplification either RL or CoT-SFT depends on.

**If Step 16's gate passes**, two tracks, cheapest first:

1. **CoT-SFT distillation probe** (cheap, do first, treat as a gate rather than a commitment): add
   back smoltalk2's excluded `_think` splits (or a dedicated math/code CoT set) as a small fraction
   of a repair/follow-up SFT pass, with the chat template extended to mark a reasoning segment
   distinctly from the final answer (a real design choice, not automatic — decide whether reasoning
   tokens are loss-masked the same as answer tokens or weighted differently, mirroring how
   `loop_ce_weights` already treats intermediate vs. final loops as differently-trusted signal).
   **Gate**: does it move GSM8K/held-out math pass rate measurably above the non-CoT SFT baseline?
   If not — expected outcome per the Learnability Gap citation — this confirms distillation alone
   isn't buying real reasoning at this scale and the RL track below is the only one worth pursuing
   further.
2. **Loop-aware RLVR** (the real reasoning-training track for this architecture, contingent on
   Step 16 already being underway): this model's recurrence is structurally closer to "reasoning as
   adaptive compute" than to "reasoning as a text trace" — `n_loops`, `loop_scale`, and whatever the
   halt head became in Step 12b-ii are the actual substrate for iterative refinement, not the
   token stream. LoopRPT/RLTT-style per-loop reward assignment (already flagged as the right
   starting point in Step 16) is the mechanism that would let RL reinforce "use more loops on hard
   problems" directly, rather than optimizing a CoT text policy that has to reconstruct that signal
   indirectly through token count. This is speculative and unbudgeted — revisit the actual papers'
   results against whatever pass@8 number Step 16 measures before committing rental hours to it.

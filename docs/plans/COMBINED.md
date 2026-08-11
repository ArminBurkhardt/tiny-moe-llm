# tiny-moe-llm — combined plan

Unified plan for `tiny-moe-llm`, folding the RAG/IR analysis from `IR.md` into the master plan from `PLAN.md`. **Read [CLAUDE.md](CLAUDE.md) first** — it's the authoritative reference for anything already built. Implementation for Steps 1–11 (loop residual, shared experts, halt head / ponder loss, per-loop CE + correctness head, config assertions, token-count stop, instrumentation, vocab prune, mmap dataset, FP8 wiring, data prep) is done and lives in the code + CLAUDE.md. See `git log -p -- PLAN.md` for original rationale if needed.

## Rules (Steps 12+)

- Commit per logical change (`feat:` / `chore:`, branch `train-build`).
- Match conventions: lowercase comments justifying *why*, Google-style docstrings with `Args:` on public modules. Don't strip existing comments.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, or `moe` in the training loop goes through `accelerator.unwrap_model(model)`.
- **Never add `.item()`, `.tolist()`, `.cpu()`, or boolean mask indexing to the per-step path.** Host syncs dominate at this model size. Exceptions: `m_splits` via `.tolist()`, `TokenTracker.sync()` at log/checkpoint cadence.

---

## Status: first full run complete (2026-08-08)

Config A' (768x8, M=32, V=65536, n_loops=3): **332,324,717 total params / 173.1M active (103.9M excl. embeddings)**, 16.0B pretraining tokens + a 708.9M-token SFT pass, one rented H100 NVL. Full loss curves and diagnostics in [docs/CONCLUSION.md](docs/CONCLUSION.md).

Key sizing calls:
- `n_loops: 4 -> 3` — non-MLP experts run densely every loop; fewer loops = fewer dense passes.
- `num_ir_entries: 16384 -> 8192` — IR expert is ~11% of forward FLOPs; halving frees throughput and ~2GB peak memory.
- `num_mlp_experts: 32` — routed experts are free active-compute-wise, but per-expert data share shrinks with pool size. At M=32/~25B tokens each sees ~1.4B; M=52 drops to ~0.9B (undertrained).
- `num_attn_experts: 1` — attention experts run densely every loop and get masked by routing afterward; extras discard compute (~5% throughput per +1% params).
- `moe_intermediate_size` is the only clean total-vs-active knob. Hit future param targets with it or `num_mlp_experts`, not `intermediate_size` (moves the fully-active dense decoder too).

**Local validation gates — all 5 passed** (pre-run, on 45M-token smoke data):

| gate | result |
|---|---|
| 1 — env check | pass |
| 2 — `test_attention_equiv.py` | pass |
| 3 — `test_overfit.py` | pass |
| 4 — 45M-token real Hub slice | mean loops 2.70, 59702 tok/s, MFU 31.8%, peak mem 25.45GB, 12.60min |
| 5 — `eval_calibration.py` on Gate 4 ckpt | ECE(p_correct)=0.013 (passes), but `p_max` beat `p_correct` on both ECE and abstention AUROC. Deferred to real final checkpoint |

Gate 5 resolved on the real checkpoint too: `p_max` won again, same margin. Triggers the Step 12b-i revert.

**Budget was time-capped, not token-capped.** Run stopped at 16.0B tokens (13.6B phase1 + 2.4B phase2) at the EUR 100 / ~47h spend cap; wall clock was ~45.7h. LR schedule anchored to 29.9B, so the cosine only reached its 4.0e-5 floor by luck of `phase1_fraction` / `target_tokens` arithmetic. A future full-budget run needs either a larger budget or a `target_tokens` matched to what the budget can reach.

**Data prep exercised at full scale.** `scripts/prepare_data.py` built `phase{1,2}.bin/.idx` during the real run (49.40GB / 24.70B tokens phase1, 9.00GB / 4.50B tokens phase2 — see `manifest.json`). All paths worked unattended. Mid-download `SIGKILL` recovery still not exercised (only clean stop/resume) — low risk, but worth watching on future preemptions.

---

## Vast.ai runbook — template

Executed once on 2026-08-08; retained as the template for future rentals (repair finetune, IR finetune). Steps 1–4 proven and don't need re-verification on the same image/GPU family.

1. **Rent.** 1x H100 SXM/NVL 80GB, **interruptible** (~half price), **120GB disk** (fixed at creation), image `nvcr.io/nvidia/pytorch:25.xx-py3`. Above ~EUR 2.2/hr, cut `target_tokens` first. **Do not** `pip install -r requirements.txt` wholesale — its TE/flash-attn wheels target sm120 (H100 is sm90); the NGC image ships both prebuilt.
2. **Verify environment** before downloading 47GB:
   ```bash
   python -c "import torch, transformer_engine.pytorch, flash_attn; print(torch.cuda.get_device_capability())"
   bash tests/run_env_check.sh
   ```
   Expect `(9, 0)`.
3. **Data prep.** `python scripts/prepare_data.py`, unattended via on-start script. Verify per-source token counts within 2%, `phase{1,2}.idx` monotonic with last entry == `len(bin)`, peak disk <~70GB.
4. **Smoke test.** `USE_FP8=1 python scripts/pretrain.py`, kill after ~200 steps. Confirm `dry_run` asserts finite loss, FP8 actually active, tokens/sec in range. **The real run never set `USE_FP8=1`** — FP8 is still unexercised end-to-end.
5. **Phase 1 / Phase 2.** 85/15 split, cosine anchored to combined `target_tokens`, router noise anneal from live token count. Observed: most descent by ~3B tokens (11.13 → 3.70 nats); phase 2's drop at the corpus switch (3.359 → 3.046 within ~200M tokens) is distribution change, not new learning — flat over its last 1.4B.
6. **Extraction.** scp checkpoints down. Re-run `eval_calibration.py` before releasing the instance — re-renting later is wasteful.

---

## Monitoring reference

**Every `LOG_INTERVAL`:** loss, tokens/sec, peak mem, `loop_scale`, current `lambda_ponder`, mean `p_halt`, mean `p_correct`, mean `p_max`, batch top-1 accuracy, per-loop CE (all `n_loops`), aux loss, ponder loss, conf loss.

**Every checkpoint:** `ckpts/training/expert_selection_*.png` from `_ExpertTracking`.

| symptom | likely cause |
|---|---|
| `loop_scale` stuck **and** `p_halt` climbing | ponder deadlock — warmup not wired to token counter |
| `loop_scale` stuck, `p_halt` in range | residual wiring wrong |
| `p_halt` saturated at 1 | `lambda_ponder` too high, or ramping too early |
| `p_halt` pinned at 0 | halt head not receiving gradient |
| `lambda_ponder` constant in the log | warmup reading stale/absent token count |
| per-loop CE flat across loops | hidden states not threaded through, **or** final `RMSNorm` not applied per loop |
| per-loop CE huge at loops 0–1, normal at last | final `RMSNorm` not applied to intermediate states |
| `p_correct` far from top-1 accuracy | label mask or target wrong |
| `p_correct` collapses to a constant | `lambda_conf` too low, or gradient leaking through `is_correct` |
| `p_correct` tracks `p_max` exactly | head learned nothing beyond baseline — **this is what happened**, see Step 12b |
| routed expert weights near zero, loss still falling | shared MLP/attention swallowed the block |
| expert selection collapses to a few MLP slots | aux loss weight too low |
| tokens/sec drops after a code change | a host sync entered the step path |
| loss spikes on resume | schedule re-anchoring or `global_offset` wrong |

`_ExpertTracking` guards against activation-checkpoint recompute double counting via `begin_forward(expected_updates)`; samples every 8th forward. If expert counts look wrong after any loop-structure change, check `expected_updates` matches `n_loops`.

---

## Post-training

**Target for Steps 12–13: calibrated abstention, not chain-of-thought.** Calibrated "knows when it doesn't know" is a directly measurable capability (ECE, abstention precision/recall) that the halt/correctness machinery is positioned to deliver. Multi-step reasoning is not learnable at 332M total / 174M active by SFT alone. Reasoning is revisited, gated, in Step 17 — after Step 16's RL gate has evidence to check against.

**Cross-step decision on abstention (from Step 15's Stage 2).** Step 12b-iii's data rebalance and Step 15's Stage 2 no-evidence condition both attack abstention. Doing both risks training the same behaviour twice with different mechanisms. **Decision: retrieval-grounded abstention (Stage 2) owns abstention; 12b-iii is reduced to fixing the SQuAD v2 conversation-count imbalance and stops there.** This ordering also means the frozen-backbone probe in 12b-i is fit against the retrieval-grounded checkpoint, not the pre-repair one.

### Step 12 — SFT — done, passed literal acceptance, failed behaviorally

`scripts/sft.py` ran: 708.9M SFT tokens (2 epochs over a 358.8M-token corpus), 5h40m, reusing `pretrain.train_step` so `p_halt` / `p_correct` supervision stayed identical to pretraining. Val CE 1.990 → 1.785 (ppl 5.96), top-1 62.1%.

| dataset | role | realized share |
|---|---|---|
| `HuggingFaceTB/smoltalk2` (no-think) | general instruction following | 48.9% |
| `HuggingFaceH4/ultrachat_200k` | general instruction following | 27.9% |
| `allenai/tulu-3-sft-personas-math` | short worked solutions | 14.0% |
| `rajpurkar/squad_v2` | **primary abstention supervision** | 7.5% tokens / **25.6% of conversations** |
| `HuggingFaceH4/no_robots` | tone and refusal style | 1.5% |
| `openai/gsm8k` (socratic) | short numbered steps | 0.4% |

**Literal acceptance passed** (ECE change -0.0233, didn't degrade). **Behaviorally it did not**: the model abstains on 80.2% of all SQuAD v2 validation questions, including 78.4% of the *answerable* half. Full breakdown in CONCLUSION.md's "Failure: the abstention mechanism".

### Step 12b — Repair the abstention mechanism (blocks Step 13)

Three independent failures stacked: SFT data rewards refusal, correctness head learned nothing beyond baseline, halt head saturated. Fixes are independent; none alone explains or fixes the collapse. **Acceptance gate throughout**: re-run `scripts/eval_abstention.py` and check the answerable-half false-abstention rate specifically — precision at 0.512 is barely above the 0.501 base rate, so precision alone can pass while behaviour is degenerate.

**Sequence.** 12b-0 (measure) → 12b-iii (repair finetune) → re-measure → then decide whether 12b-i/ii are worth building. 12b-iv and 12b-v apply to whichever finetune runs.

#### 12b-0. Measure before building anything

Every route below is gated on numbers the existing checkpoint can produce in minutes.

1. **Is loop 3 idle or churning?** Per loop, log `‖Δh‖ / ‖h‖`, `cos(Δ_k, Δ_{k-1})`, and top-1 flip rate between loop 2 and loop 3 readouts (`hidden_states_all` in `modules/model/moe.py`). Effective per-loop gate is `(1 - p_halt) * loop_scale` ≈ `[0.38, 0.40, 0.29]`. Loop 3 writes a ~0.3-RMS-relative update while per-loop CE doesn't move (3.109 / 2.969 / 2.969).
   - Flip rate ≈ 0 → loop 3 is a no-op; ship `n_loops=2`, 12b-ii item 1 is settled.
   - Flip rate high with CE flat → loop 3 churns between equally-good predictions; rules out the noise route in 12b-iv.
2. **Oracle minimum sufficient depth.** On held-out data run `n_loops = 1, 2, 3` and per token record the smallest depth whose argmax matches the label, bucketing "never correct" separately. This histogram *is* "do complex tokens need more loops". Given loop 3 contributes ~0 nats, expect "3 helps where 2 doesn't" to be small, roughly cancelled by tokens where 3 hurts. If no headroom, adaptive depth is dead and 12b-ii items 3–6 close for GPU-minutes. If headroom, the same labels are the training signal.
3. **Does any confidence signal carry answerability info?** AUROC of `(1 - p_max)` for unanswerable SQuAD v2 is **0.457 — below chance**; `(1 - p_correct)` is 0.457 too. Anything replacing the correctness head must beat 0.5 before anything is built on it.

#### 12b-i. Correctness head: revert (redesign only if retried)

**What it does.** `self.correct_proj` (zero-init) applied to the final loop's hidden states inside `compute_mtp_loss`, alongside `lm_head`. BCE target `is_correct` computed under `torch.no_grad()` from that chunk's teacher-forced CE logits — literally `argmax(logits) == labels` on the reference continuation.

**Why it failed.** The target is a near-deterministic function of the hidden state `correct_proj` reads: head and `lm_head` share the same input, and the label is derived from `lm_head`'s own output on that input. The BCE optimum is "reproduce `p_max`" — the best predictor available from that hidden state alone, which is exactly what `p_max` computes free. Gradient isolation worked (`tests/test_correctness_head.py` passes); the *targeting* is wrong. On the real checkpoint: answer-level ECE 0.378 vs. `p_max`'s 0.371, AUROC 0.604, and — the actual failure — AUROC of `(1 - p_correct)` for unanswerable = **0.457**, with mean `p_correct` *higher* on abstentions (0.835) than on real answers (0.739). The head is most confident when refusing.

**Revert.** Per PLAN.md's original revert criterion: remove `correct_proj`, the BCE term in `compute_mtp_loss`, `TrainingConfig.lambda_conf`, and substitute `p_max` everywhere the head's output was read (`sft.py` val logging, `eval_abstention.py`, `eval_calibration.py`). Subtraction, not redesign; compute-free either way.

**Preferred replacement — a frozen-backbone probe over cross-loop disagreement.** Entropy and top-2 margin are functions of the same logits `p_max` comes from (the family that lost, ECE 0.371 vs. 0.378). The signal this architecture provides for free and `p_max` cannot contain is disagreement across *depth*: `KL(p_loop2 ‖ p_loop3)`, top-1 agreement between consecutive loops, optionally across `n_loops` overrides. The recurrence is a free depth-ensemble; its spread is a genuine epistemic-uncertainty signal. Shares its computation with 12b-ii's convergence-exit criterion — one measurement, two consumers.

- **Backbone frozen, fitted offline.** Minutes of compute; no `pretrain.train_step` change. Yields a tunable precision/recall operating point.
- **Features:** `[final hidden state, p_max, entropy, top1-top2 margin, cross-loop KL, cross-loop top-1 agreement]`.
- **Targets from sampled generations vs. ground truth, sequence-level.** Not teacher-forced argmax — the leak that sank Step 4b.
- Fit and eval held-out; same Gate 5 bar (beat `p_max` on ECE *and* AUROC). Per 12b-0 item 3, first job is beating 0.5 AUROC on answerability at all.
- Fit **after** 12b-iii and Step 15 Stage 2 — on the current checkpoint the probe would fit a "refuse" policy.

#### 12b-ii. Halt head: give it real compute authority, replace the control loop, or drop it

**What it does.** `moe.py`'s `forward_step` computes `p_halt` from the *incoming* hidden state each loop (`halt_proj`, zero-init weight, bias `-2.0`), applied as an output gate: `hidden_states += (1 - p_halt) * loop_scale[loop] * dropout(post_norm(output))`. **Every expert still runs regardless of `p_halt`** — halting suppresses the *update*, not the *computation*. Only the ponder loss (`lambda_ponder * mean(1 - p_halt)`) pushes `p_halt` away from a constant, and CE has near-zero gradient w.r.t. `p_halt` once `loop_scale` controls output magnitude.

**Why it failed.** `p_halt` collapsed to ~0.004 during zero-λ warmup (pure CE reward for a near-zero gate letting the residual through). Once the ponder ramp engaged, `p_halt` overshot straight to ~0.78 and pinned there for the remaining ~14B tokens. The auto-adjust controller did its job — cut `lambda_ponder` 11 times, 0.15 → 0.01 floor — with **no measurable effect on `p_halt`**. A saturated sigmoid has near-zero gradient w.r.t. its pre-activation; retuning λ's magnitude can't pull it back. Meanwhile `loop_scale` grew from 0.578 to `[1.73, 1.81, 1.32]` — the loop learned to control its own contribution through the multiplicative gate that *does* have a live gradient, leaving `p_halt` dead weight.

**Root cause.** `p_halt` gates output, not compute, so there's no FLOPs saving on the table for CE to trade against. Structural gap, not hyperparameter miss.

**Fixes, cheapest first:**

1. **Measure the honest baseline.** Per-loop CE says loop 3 contributes ~0 nats. A **static `n_loops=2`** config may recover essentially all the quality the halt mechanism was supposed to buy adaptively, for free. Run this ablation first.
2. **Drop the head, exit on convergence** (recommended if 12b-0 says the recurrence converged). "Was the last loop a no-op? then stop" — zero parameters, no `lambda_ponder`, no saturating sigmoid, one tunable threshold.
   - **Measure convergence in the readout, not the hidden state.** `loop_scale[2]=1.32` means loop-3 hidden delta is large while the *prediction* is stationary; use `KL(p_k ‖ p_{k-1})` or top-1 agreement between `lm_head` outputs.
   - **"Vocab projection per loop" objection doesn't apply at generation time** — only the last position needs a readout (1 × 65536, free). Under teacher forcing per-loop readouts already exist.
   - **Trap.** `(1 - p_halt)` is pinned at ~0.22 and `loop_scale` grew to absorb it. Deleting the gate multiplies every loop's delta by ~4.5× and the checkpoint breaks — fold the constant into `loop_scale` instead of dropping the term.
3. **Make halting actually skip compute.** Currently `p_halt` can only be soft because masking output post-hoc changes nothing about cost. Gathering/masking so halted tokens' experts don't run in later loops turns `p_halt` into a real trade-off with a live CE gradient. Most invasive: interacts with grouped-GEMM machinery, needs per-token variable loop count within a batch. Routed MLP path is fine (grouped GEMM already ragged); **non-MLP experts are the dense per-loop cost** and need full K/V with queries only for still-active tokens — varlen supports this but combined with document packing (`cu_seqlens` needs separate q/kv offset arrays) and `shared_attn` on the same path, that's the real work. Also where the savings are, so can't skip.
4. **Replace the λ-nudge controller with a Lagrangian on an explicit budget** (dual ascent) — still an output-gate but constrained optimization instead of a heuristic EMA nudge that can't recover once the primal saturates.
5. **Switch to cumulative ACT** (Graves-style: halting probs accumulate across loops, ≤1, ponder cost on loops taken) instead of the current greedy-per-loop formulation. ACT's cross-loop normalization gives coherent "when do I stop" semantics.
6. **Supervise depth directly.** 12b-0 item 2's oracle histogram is already a per-token label ("smallest depth whose argmax matches the label"); halting can be trained as plain classification. Removes every failure mode above: no λ, no ramp, no saturating sigmoid, no coupling to main loss. Pairs naturally with item 3 (labels say *when*; item 3 makes stopping *cheap*).

Items 1 and 2 are half-day ablations on the existing checkpoint. Items 3–6 are real training-loop changes; only attempt if 12b-0's oracle histogram shows real per-token depth gap.

#### 12b-iii. SFT data: fix the SQuAD imbalance (reduced scope; Step 15 Stage 2 owns abstention)

**Root cause of the 80.2% abstention rate.** SQuAD v2 is 7.5% of SFT tokens but 25.6% of conversations, and its unanswerable third is a ~6-token, extremely low-entropy target. Per-token CE makes a short memorized refusal the cheapest available loss reduction. Nothing in the rest of the mix penalizes refusing an answerable question. Result: 7,786/11,873 completions are literally `"The passage doesn't say."`.

**Reduced scope.** Since Step 15's Stage 2 will teach retrieval-grounded abstention, 12b-iii's job here is just fixing the mechanical data imbalance so it doesn't dominate a Stage 2 finetune. Item 1 alone is the minimum; items 2–4 are optional if the imbalance isn't cleanly cured.

1. Down-sample SQuAD v2 unanswerable rows to ~10–15% of the QA subset (from ~33–50%).
2. Add answerable-only extractive QA (SQuAD 1.1, NQ-open, TriviaQA, HotpotQA) so answerable/QA-shaped-prompt volume isn't dwarfed by SQuAD v2.
3. Weight the loss per conversation, not per token, so a 6-token refusal stops mechanically out-earning a multi-sentence answer.
4. Vary abstention *training* phrasings while keeping the closed set in `modules/data/abstention.py` for eval-side `is_abstention` detection only.

**Cheapest path.** Short repair finetune on the existing SFT checkpoint (~20–50M tokens, `lr=1e-5`, 1 epoch); reuses the Vast.ai runbook's shorter-run template.

#### 12b-iv. Loop refinement: latent noise / dropout (rejected), input injection (open)

The question: *should a further finetune inject a noise vector or extra dropout into the latent between loops to force "actual refinement"?*

**Rejected as specified.** Diagnostics say the loop isn't idle: `(1 - p_halt) * loop_scale` ≈ `[0.38, 0.40, 0.29]`. The failure is "the loop does work `lm_head` is blind to", not "the loop declines to work". Noise adds a *denoising* task at train time only (nothing injected at eval) — train/test mismatch, not refinement pressure. The random-`h₀`-plus-randomized-depth variant that does have a track record is a *pretraining-time* property; can't be installed by a 20–50M-token finetune at `lr=1e-5`.

Related facts:
- **Dropout on the delta already exists** — `moe.py`'s `forward_step` is `self.dropout(self.post_norm(output))`, at `dropout: 0.00` in pretraining and `0.05` in SFT. Raising it is a one-line config change; expect nothing.
- **Input injection already exists, conditionally.** The cross-attn expert receives `other=self._moe_ple(input_ids)` (`transformer.py`) — same trick as "inject `e` at every recurrent step", but it's one routed expert out of 35 at `top_k=2`, so its contribution is gated toward 0 much of the time. **If one architectural change in this family is made, make injection unconditional** (alongside `shared_mlp` / `shared_attn`) rather than adding noise. Gives every loop a stable anchor to refine *against*.
- **The unavoidable tension.** `loop_ce_weights: [0.2, 0.3, 1.0]` trains loop 1 to already be a usable readout. Making later loops matter more means making early readouts worse — which makes 12b-ii item 2's early exit less viable. The next run has to pick one.

Gate: 12b-0 item 1. Flip rate ≈ 0 → ship `n_loops=2`, nothing here applies. Flip rate high with CE flat → noise would make it worse.

**Note on Step 15.** Once the retrieval-conditioned query bias (Step 15 item 1 under "Making >3 loops pay") is added, that's the input-injection story for RAG batches, and the appended-evidence buffer supplies the "reason to differ" between loops. Under RAG, 12b-iv's tension may resolve automatically: `[0,0,0.1,0.2,0.3,1.0]` on RAG batches trains later loops to matter, while `[0.2,0.3,1.0]` on plain-LM batches preserves loop 1 readouts — per-task, not global.

#### 12b-v. MTP: already inference-only as output, but not free

MTP was only ever a training-time objective. `TinyMoETransformer.forward` returns `lm_head(x)` from the final loop, and nothing reads `extra_token_outputs` outside `compute_mtp_loss`. Two things still worth changing:

1. **`_mtp_forward` runs unconditionally** (`transformer.py`) and `scripts/inference.py` discards the result. With `late_token_loss=True` that's only the gate/up/down MLP (no vocab projection), but `inference.py` has no KV cache and re-runs the full prefix per generated token, so the waste is paid every step — `eval_abstention.py`'s batched decode inherits it. Guard the call on `self.training`.
2. **`lambda_mtp: 0.0` does NOT skip compute.** `compute_mtp_loss` gates on `mtp_outputs is not None`, so zero weight still pays the full head plus its `num_extra_tokens` chunked vocab projections — 4× cost per CLAUDE.md. To turn MTP off for a finetune, pass `mtp_outputs=None` / skip `_mtp_forward`; keep weights on disk.

**Recommendation:** drop MTP for a behavioural repair finetune. At 20–50M tokens and `lr=1e-5`, objective drift is negligible against the throughput win. Keep it for any further general pretraining.

#### Acceptance

**For 12b overall:** answerable-half false-abstention rate materially below the current 78.4%. Step 13's own bar is <10%. Abstention precision clearly above the ~0.50 base rate. If the correctness head is kept rather than reverted, `p_correct` must beat `p_max` on both ECE and AUROC.

That gate belongs to 12b-iii + Step 15 Stage 2 jointly, since abstention now flows through Stage 2. Sub-item gates:

| sub-item | gate |
|---|---|
| 12b-0 | none — measurement pass everything else conditions on |
| 12b-i (probe) | beats 0.5 AUROC on answerability, then beats `p_max` on ECE *and* AUROC, held out |
| 12b-ii (exit / depth) | real quality gap between `n_loops=2` and `3` in 12b-0's oracle histogram |
| 12b-iii (data) | SQuAD conversation-count imbalance cured; the behavioural bar is on Step 15 Stage 2 |
| 12b-iv (injection) | 12b-0 item 1 shows the recurrence has *not* converged |
| 12b-v (MTP) | none — correctness/throughput cleanup; measured by tokens/sec and unchanged eval |

### Step 13 — Self-labelled calibration set

**Blocked on Step 12b + Step 15 Stage 2 landing a non-degenerate baseline first.** Building this dataset from the current checkpoint would re-encode the collapse. New script `scripts/build_calibration_set.py`.

1. Sample the repaired checkpoint N=16× at temperature 0.8 on short-answer QA (`trivia_qa`, `nq_open`, `squad_v2`).
2. Label each question by empirical pass rate (normalized exact/alias match against reference).
3. Rewrite targets by pass rate: `>0.8` → the answer; `<0.2` → an abstention; in between → a hedge, from a small fixed set of phrasings.
4. Hold out 10% before rewriting (the calibration eval set).
5. Second SFT pass on the rewritten data.

**Acceptance:** abstention rate on the held-out low-pass-rate bucket >60%; on the high-pass-rate bucket <10% (catches the degenerate "refuse everything" solution — the exact number Step 12 failed); ECE of the abstention signal improves relative to Step 12b.

### Step 15 — RAG-first evidence integration (supersedes the original IR-expert plan)

**Framing.** The original Step 15's 6-item ladder (lower temperature, top-k, grow table, warm-start, IR-only finetune, more IR experts) is a *knowledge-capacity* project. Every item keeps the table parametric, so the corpus can't be updated without training and the exposure requirement (~1000 presentations per fact for long-tail knowledge) remains binding. The real leverage is going non-parametric. This step reframes accordingly.

**Current state** (`modules/model/information_retrieval.py` + `experts.py`): one IR expert, `down_proj` 768→128, learned table of 8192 key/value pairs at `ir_dim=128`, cosine softmax retrieval, `up_proj` 128→768. ~2.3M params, ~5% of forward FLOPs, dense every token every loop.

**Four constraints from the run:**

1. **No retriever to preserve.** Cosine logits at `temperature=1.0` over 8192 entries give a near-uniform softmax; retrieved ≈ `mean(y_values)`. The table stores nothing addressable. Replacing its contents, size, or `ir_dim` costs nothing.
2. **The router likes the slot anyway** (7–9% vs 5.7% uniform). The *pathway* is wanted; ablation decides whether the content matters or the slot is a bias term.
3. **Loop 3 buys ~0 nats** (per-loop CE 3.109 / 2.969 / 2.969). ">3 loops" is not a config change; later loops need a *reason* to differ. RAG provides one only if retrieval re-executes per loop with a moving query.
4. **Depth override already works.** `max_enc_loops=64` ([moe.py:183](modules/model/moe.py#L183)), sinusoidal loop encoding, `loop_scale` clamps. `forward(n_loops=8)` runs today; only training blocks deeper loops.

**The two decisions that must not be conflated:**

- **(a) Where memory content comes from** — learned parameters (today) vs. external swappable data (RAG).
- **(b) How the read refines across loops** — one-shot vs. iterative / multi-hop.

#### 15a. Where evidence enters — three ports

**Option A — IR module reads externally supplied memory.** Add `memory=(K_ext, V_ext)` to [information_retrieval.py](modules/model/information_retrieval.py); concatenate `K = [z_keys ; K_ext]`, `V = [y_values ; V_ext]`. Two properties fall out:

- **Graceful degradation.** No corpus attached → bit-identical to today. One checkpoint, both modes.
- **Groundedness signal.** Softmax mass on external vs. parametric entries is measurable — "I retrieved nothing relevant" stops being guesswork. Directly the feature 12b-i's probe is missing. Biggest cross-workstream win.

Limitation: 128-d per entry is a topic vector, not a passage. Fine for "which fact"; weak for "copy this span".

**Option B — evidence as token-level KV through CrossAttention.** `other` in [transformer.py:341](modules/model/transformer.py#L341) is already a per-call injection port re-read every loop. Swap `_moe_ple(input_ids)` for embedded retrieved chunks → RETRO/FiD-lite. Strongest for extractive/grounded answering — what 332M can actually learn.

Blockers:
- [attention.py:109-116](modules/model/attention.py#L109-L116) passes the same `cu_seqlens` for q and k, so `o_len` must equal `S`. Evidence of length M needs `cu_seqlens_k` / `max_seqlen_k` plumbed (flash supports it) and `causal=False`.
- [gemma4.py:79-81](modules/model/gemma4.py#L79-L81) rotates q and k with the same `cos/sin`, meaningless for evidence positions. Probably: no RoPE on evidence keys, or a short per-chunk position basis.

**Option C — both, with a division of labour. Preferred.** IR expert = *selector* (which candidates matter; long-tail entity memory); CrossAttention expert = *reader* over evidence tokens. Maps the two existing experts onto the standard retriever/reader split.

**Caveat.** Don't let the router decide whether to consult evidence. The router never specialized (aux loss pinned from step 0; routed weight flat across all 35 experts). Make the evidence read **always-on** alongside `shared_mlp` / `shared_attn`, and gate the *content* via retrieval scores.

#### 15b. Getting external keys into the query space

Query = `down_proj(RMSNorm(h)) ∈ R^128`. Keys must live there.

- **B1 — self-encoding.** Encode chunks with the model itself, pool, `down_proj`. Spaces match by construction; but a 128-d pooled state from a model never trained for retrieval is weak, and any trunk change forces re-index.
- **B2 — external embedder + adapter** (bge-small / e5-small, 33M, 384-d). **Pragmatic choice.** Map the query into the embedder's space (standard, reusable ANN index), project retrieved vectors into IR space. Retriever quality leaves the critical path; inference cost = one small forward per query.
- **B3 — contrastive warmup.** Regardless of B1/B2, train the query head with InfoNCE on (context → the chunk containing the continuation) pairs mined from `phase1.bin`. Freeze the document side (also sidesteps index staleness). What makes retrieval *work* instead of hoping CE discovers it.

#### 15c. Granularity

Per-token ANN over a corpus during generation is infeasible; per-sequence-at-prefill kills multi-hop. The design:

> **ANN retrieves k≈32–64 candidates per sequence per loop. The IR module's soft, differentiable read over those candidates stays per-token.**

End-to-end differentiable; ANN cost = loops × sequences, not tokens; matches the two-stage retrieve/read structure the module already implements.

**KV-cache note.** If the evidence set mutates mid-generation, the IR/cross-attn cache goes stale. Make evidence **append-only** — new retrievals extend the KV set, never rewrite it. Cache-friendly, and it hands you the accumulating buffer multi-hop needs anyway.

#### 15d. Making >3 loops actually pay

Cheapest first:

1. **Loop-conditioned query.** Zero-init per-loop bias on the IR query, mirroring `loop_router_bias` (sinusoidal in absolute loop index, clamped). Guarantees loop 3 doesn't re-issue loop 1's query. No-op at init → checkpoint loads unchanged.
2. **Append-only evidence buffer.** Loop L reads the union of retrievals from loops 1..L. This *is* refinement, made concrete; makes depth monotonically informative.
3. **Novelty pressure.** Mask already-retrieved ids from the next loop's ANN (or an MMR term). Otherwise three loops fetch the same top-1 three times.
4. **Depth curriculum.** `sample_n_loops` / `loop_ce_weights_for` in [pretrain.py:239-257](scripts/pretrain.py#L239-L257) already truncate-and-rescale so the deepest loop carries weight 1.0. Extend sampling upward (max 6–8) on retrieval-augmented batches with a **back-loaded** weight vector (e.g. `[0,0,0.1,0.2,0.3,1.0]`), while plain-LM batches keep `[0.2,0.3,1.0]`. Resolves 12b-iv's tension per-task rather than globally.
5. **A real job for the halt head.** Halting was gating output, not compute. Under RAG it becomes "stop retrieving and stop looping" — a real early exit with a real compute payoff. "Keep looping while new evidence is arriving" (buffer stopped growing / delta norm below τ) is a well-posed, learnable criterion. This is the first version of 12b-ii item 6 with genuine authority.
6. **Retrieval-utility diagnostic.** Per-loop CE-with-evidence minus CE-with-evidence-zeroed. Tells you whether depth is buying grounding or churn.

#### 15e. Training recipe

**Stage 0 — measure, no training.** Non-negotiable, hours.
- Retrieval entropy today (expect ≈ ln 8192 = 9.01 nats).
- IR ablation: zero the expert's output, measure ΔCE on held-out.
- Query drift: `cos(down_proj(h_loop1), down_proj(h_loop3))`. If ≈1, 15d item 1 is mandatory.

**Stage 1 — sharpen.** Learned log-temperature + top-k read. **Trap the original Step 15 item 1 understates:** `y_values` were only ever used as a near-uniform mixture, so dropping temperature at inference reads out vectors that were never individually trained → loss spike, not signal. Temperature must be *annealed during finetune* (1.0 → ~0.05).

**Stage 2 — oracle evidence.** Key trick, main training spend. Before any index exists, hand the model evidence you already know is relevant: gold passage for QA, held-out span from the same document for web text. Three conditions mixed evenly:

| condition | what it teaches |
|---|---|
| gold evidence | read the buffer (large, immediate CE gradient) |
| distractor evidence | don't blindly trust the buffer |
| no evidence | **abstain — grounded in retrieval, not memorized as a string** |

The third row is why this stage owns abstention: a principled fix for the 78.4% collapse that doesn't rely on rebalancing SQuAD v2 phrasings.

**Stage 3 — align the retriever.** InfoNCE, query side only, document encoder frozen; one hard-negative mining round.

**Stage 4 — end-to-end with the real ANN index.**

**Stage 5 — depth curriculum** (15d items 1–4). Only here does ">3 loops" get trained.

**Stage 6 — RAG SFT.** [chat.py](modules/data/chat.py) has system/user/assistant only; evidence needs either a new segment or the system turn.

#### 15f. Gates

- **G1**: IR ablation ΔCE > ~0.02 nats. If ~0, the expert is a bias term — fix before building an index.
- **G2**: post-anneal, entropy well below ln N *and* held-out CE not regressed.
- **G3**: gold-vs-no-evidence CE gap ≥ ~0.3 nats on the answer span; abstention rate under no-evidence ≫ under gold. **This is the answerable-half false-abstention gate for 12b overall.**
- **G4**: recall@k beats BM25. If not, take B2 and stop training the retriever.
- **G5**: EM/F1 on NQ-open / TriviaQA / **PopQA** with corpus attached vs. not. Then depth ablation: EM at `n_loops` = 2, 3, 4, 6, 8 with corpus attached. **Flat past 3 → depth story is dead; ship 3.**
- **G6 — honest baseline: put retrieved passages in the prompt as text.** If side-channel RAG only matches that, the architecture claim is unproven. The real target is what in-context evidence *can't* do: attach far more evidence than 4096 tokens can hold, at cost that doesn't grow quadratically.

#### 15g. Trainable dense fact table alongside RAG — encode, don't train

If a trainable in-model fact table is later wanted alongside RAG, the arithmetic decides against training it:

Allen-Zhu & Li's knowledge-capacity scaling law: LM parameters store ~**2 bits of extractable fact per parameter** under ideal conditions (clean, deduplicated data, ~1000 exposures per fact). At ~100 exposures it's ~1 bit/param; junk data degrades further. Table cost = `entries × (ir_dim key + ir_dim value)` = `entries × 256` at `ir_dim=128`:

| entries | table params | bf16 | AdamW state (fp32 m/v) | 2-bit ceiling |
|---|---|---|---|---|
| 8K (today) | 2.1M | 4 MB | 17 MB | ~0.5 MB of fact |
| 256K | 67M | 134 MB | 537 MB | ~17 MB |
| 1M | 268M | 537 MB | **2.1 GB** | ~67 MB |
| 4M | 1.07B | 2.1 GB | 8.6 GB | ~268 MB |

A 1M-entry trained table doubles the model for a theoretical ceiling of ~67 MB of deduplicated facts. English Wikipedia is 20+ GB; you're 2–3 orders short. Exposure requirement is the harder wall: writing a long-tail fact by SGD needs ~1000 presentations.

**This inverts the original Step 15 item 5's gate.** Gating parametric memory on **PopQA** measures it on its worst case — long-tail facts have the fewest natural exposures, i.e. the sub-1-bit/param regime. Long-tail is where non-parametric retrieval is *strongest*. The defensible split is the opposite of the intuitive one:

- **Trained/parametric table → head of distribution** + schema, relation templates, entity-type priors, cross-document aggregates absent from any single passage.
- **RAG → tail.** Updatable, unbounded, attributable.

**The version that scales — encode, don't train:** `InformationRetrievalModule` doesn't care where `(K, V)` came from. Train a small encoder and compute them in one pass over the corpus:

- `K = key_enc(passage)`, `V = value_enc(passage)` — learned compression of each passage into the model's 128-d (or 256-d) space, à la gist/memory tokens.
- Trainable params = encoder + reader. Small, generalize to unseen passages.
- The table becomes **data**: as big as the corpus, updatable by re-encoding, no optimizer state, no exposure requirement.

**If a trained table is still wanted: write it, don't train it.** The read path is linear in values: `softmax weights → mix of V → g_proj → up_proj`. For a sharply-selected entry, the map from value vector to hidden-state contribution is linear — writing a fact is *fitting*, not training:

1. Teacher: run the model with the fact **in context**, record hidden states at the injection point.
2. Student: run with the fact **only in the table**.
3. Solve for the value vector (least squares) minimizing the gap.

Context distillation into a memory slot — ROME/MEMIT family. Minutes per fact-batch instead of 1000 exposures.

**Mechanical requirements past ~64K entries:**
- **Product keys.** Dense `[1M, 128]` cosine matmul per token per loop is not viable. Split the query in half, score each against `√N` sub-keys, combine top-k → `O(√N)`.
- **Sparse gradients** (`EmbeddingBag`-style) if the table is trained at all.
- **Sharpening is prerequisite.** A 1M-entry near-uniform softmax is just a more expensive constant.

**Combining both.** If a trained table is always available and retrieval is noisy, CE teaches the model to ignore retrieval (table is more reliable at training time). Mitigations: distractor/no-evidence conditions from Stage 2, plus dropout on the parametric table during RAG training.

**Recommendation.** Build the encoded table (Option A + encoder). Keep the trained table small — 64K–256K entries, ~17–67M params — for head facts and schema. Gate on: **500 MB trained table vs. 500 MB encoded index over real text, same reader, same budget.** Prediction: index wins on factual QA, trained table wins on latency and on aggregate knowledge stated in no single passage.

#### 15h. Risks

- **128 dims is thin for passage content.** Since the table holds nothing today, raising `ir_dim` to 256 for the RAG variant is nearly free — do it alongside any re-init.
- **332M / 16B tokens is a weak reader**, but grounded *extraction* is the easiest thing to teach at this scale. Copying beats recalling; RAG converts a knowledge problem this model can't solve into a copying problem it can.
- **Retriever/reader co-training staleness** — mitigated by freezing the document side throughout.

**Sequencing.** Step 15 runs on its own short rented session (Vast.ai runbook template scaled down). Independent of Steps 12b/13 subsystems; the abstention repair (12b overall gate) is now measured *after* Stage 2, so Stage 2 finetune sits between 12b-iii and Step 13.

### Step 16 — RL: deferred, gated

**Do not start.** Evidence at this scale is negative: a 135M RLVR study on GSM8K went from SFT base 24/1319 (1.82%) to 21/1319 at 192-token completions and 16/1319 at 320 — GRPO made it *worse*. On Qwen2.5-0.5B, even a format reward stayed below 0.1 after 300 steps. Under a 0/1 reward, a base model that can't sample correct solutions produces no gradient; RLVR only amplifies what's already there — at 174M active / ~16–30B tokens there's little to amplify.

**Gate:** pass@8 on the target task, measured on the checkpoint out of Step 12b/13/15 (a still-degenerate refuser would fail pass@8 for the wrong reason). **Below ~15%, do not proceed** — the budget would be spent confirming the null.

If the gate passes: vanilla GRPO is architecturally mismatched to a looped model (credits output tokens while the computation is latent). Read LoopRPT (arXiv 2603.19714) and RLTT first — they assign reward to per-loop latent states and report improved gate calibration (more early exits, final-step dominance maintained) as a side effect, directly relevant to whatever the halt head became after Step 12b-ii.

### Step 17 — Reasoning training: deferred, gated on Step 16's pass@8

**Do not start before Step 16's gate resolves.** Right now the model isn't fine-tuned for reasoning (SFT used only smoltalk2's `_no_think` splits; `modules/data/chat.py` has no reasoning/answer segment distinction), and the Small Model Learnability Gap — at this scale, long-CoT imitation teaches fluent filler before wrong answers rather than actual multi-step reasoning — applies as much to a dedicated reasoning pass as to Step 12's mix.

**If Step 16's gate fails (pass@8 < ~15%):** stop. Base capability doesn't support the amplification RL or CoT-SFT depends on.

**If Step 16's gate passes**, two tracks, cheapest first:

1. **CoT-SFT distillation probe** (cheap, gate not commitment). Add back smoltalk2's excluded `_think` splits (or a math/code CoT set) as a small fraction of a repair SFT pass, with the chat template extended to mark a reasoning segment distinctly from the final answer. **Gate:** does it move GSM8K / held-out math pass rate measurably above the non-CoT baseline? If not — expected per the Learnability Gap — distillation alone isn't buying real reasoning at this scale.
2. **Loop-aware RLVR** (real reasoning-training track for this architecture). The recurrence is closer to "reasoning as adaptive compute" than "reasoning as a text trace" — `n_loops`, `loop_scale`, and whatever the halt head became in 12b-ii are the actual substrate for iterative refinement, not the token stream. LoopRPT/RLTT-style per-loop reward assignment would let RL reinforce "use more loops on hard problems" directly, rather than optimizing a CoT text policy that reconstructs that signal indirectly through token count. Speculative and unbudgeted — revisit against Step 16's pass@8 number before committing rental hours.

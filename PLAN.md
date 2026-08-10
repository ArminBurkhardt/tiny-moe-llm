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

#### 12b-ii. Halt head: give it real compute authority, or replace the control loop

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
2. **Make halting actually skip compute.** Currently `p_halt` can only ever be a soft signal because
   masking output post-hoc changes nothing about cost. Gathering/masking so halted tokens' experts
   genuinely don't run in a later loop turns `p_halt` into a real compute/quality trade-off with a
   live CE gradient (a token that halts and then would have benefited from another loop actually
   loses accuracy, which is signal CE can use) — this is the only fix that addresses the root cause
   rather than the symptom. It's also the most invasive: it interacts with the mask-multiply /
   grouped-GEMM machinery `moe.py` already uses for expert sparsity, and needs a per-token variable
   loop count within a single batch, not just across batches (`loop_count_sampling` already varies
   depth *per step*, this would need it *per token*).
3. **Replace the λ-nudge controller with a Lagrangian on an explicit compute/halt budget** (dual
   ascent: increase the multiplier when the constraint is violated, decrease when it's slack) —
   still an output-gate, but a properly constrained optimization instead of a heuristic EMA nudge
   that has no way to recover once the primal variable (`p_halt`) saturates.
4. **Switch to cumulative ACT** (Graves-style: halting probabilities that accumulate across loops
   and must sum to ≤1, with a ponder cost on the number of loops actually taken) instead of the
   current greedy-per-loop formulation (`p_halt` recomputed fresh each loop, so a token can halt at
   loop 1 and un-halt at loop 2 — CLAUDE.md flags this explicitly). ACT's cross-loop normalization
   is what gives the halting decision a coherent "when do I stop" semantics; the current design
   doesn't actually ask that question per loop, so it's unsurprising the answer collapsed to a
   constant.

Item 1 is a half-day ablation with the existing checkpoint architecture (just force `n_loops=2` at
inference and re-run eval). Items 2-4 are real training-loop changes and should only be attempted
if item 1's ablation shows a real quality gap between 2 and 3 loops worth recovering adaptively.

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

**Acceptance for 12b overall**: re-run `scripts/eval_abstention.py`; answerable-half false-abstention
rate materially below the current 78.4% (Step 13's own bar is <10%, which is the real target to aim
for even though 12b's job is just to get off the degenerate floor); abstention precision clearly
above the ~0.50 base rate; if the correctness head is kept rather than reverted, `p_correct` must
beat `p_max` on both ECE and AUROC on this checkpoint too.

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

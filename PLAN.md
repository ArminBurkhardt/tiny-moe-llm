# PLAN.md

Plan for `tiny-moe-llm`'s first real pretraining run. **Read [CLAUDE.md](CLAUDE.md) first** — it
documents the invariants this plan produced and is the authoritative reference now that the
build-out below is finished. The step-by-step implementation instructions this file used to carry
(Steps 1-11: loop residual, shared experts, halt head/ponder loss, per-loop CE + correctness head,
config assertions, token-count stop, instrumentation, vocab prune, mmap dataset, FP8 wiring, data
prep) are done and live in the code + CLAUDE.md now — see `git log -p -- PLAN.md` for the
original text if the rationale behind a specific decision is needed.

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

## Status: build-out complete, ready to rent

**Config A' (768x8, M=32, V=65536, n_loops=3)** is live in `config.yaml`: 332M total / 174M active
(104M excl. embeddings) params, ~357 MFLOP/token forward, ~1071 MFLOP/token training. Key sizing
calls, since they're not written down elsewhere:
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

**Local validation gates — all 5 passed:**

| gate | result |
|---|---|
| 1 — env check | pass |
| 2 — `test_attention_equiv.py` | pass |
| 3 — `test_overfit.py` | pass |
| 4 — 45M-token local run, real (not synthetic) Hub-sourced slice | mean loops 2.70, 59702 tok/s, MFU 31.8%, peak mem 25.45GB, 12.60min |
| 5 — `eval_calibration.py` on the Gate 4 checkpoint | ECE(p_correct)=0.013 (passes <0.15), but `p_max` beat `p_correct` on both ECE and abstention AUROC. Literal rule says revert Step 4b; **deferred instead** — only one real cloud run remains and `correct_proj` is proven gradient-isolated/free, so the revert-or-keep call gets re-made against the real final checkpoint (see memory `project_step4b_correctness_head_deferred`) |

**Budget decision made**: `target_tokens: 29900000000` is baked into `config.yaml` (see its own
comment for the derivation — data-limited at the 30B prepared-data cap, not budget-limited).

**Data prep exercised for real against the Hub**: `scripts/prepare_data.py` ran locally against all
seven live sources (including the gated `nvidia/Nemotron-CC-Math-v1`) at smoke scale — phase1 50M
tokens / phase2 10M tokens, `data/prepared/{phase1,phase2}.{bin,idx}`, full source
repo-ids/revisions/per-source token counts recorded in `manifest.json`'s `data_prep` key. This
covered every item that used to be listed as "untested against the real Hub": live file discovery,
gated access, the `.jsonl.gz`/`.jsonl.zst`/parquet decompression paths, column auto-detection,
smoltalk2 rendering + holdout-hash recording, and an end-to-end download→tokenize→write→delete
pass. **Not exercised**: a real mid-download `SIGKILL` interruption (only clean stop/resume via the
checkpoint state file) — low risk given the resume path is the same one already covered by
`tests/test_prepare_data.py`'s synthetic interrupt/resume checks, but worth a wary eye during the
real run if a preemption lands mid-source. The full 30B-token run itself still needs to happen on
the rented box — that's the Vast.ai runbook below, not done yet.

---

## Vast.ai runbook

**Total spend capped at EUR 100** (~47 training hours after prep/smoke-test/extraction at
EUR 2/hr — steps 3-4 below are billed too).

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
3. **Data prep**: `python scripts/prepare_data.py`, in the instance's on-start script (unattended,
   40min-2h, likely download-bound). Verify per-source token counts within 2% of target,
   `phase{1,2}.idx` monotonic with last entry == `len(bin)`, peak disk under ~70GB.
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
   — re-renting later to compute them is wasteful. **This is also the point to re-decide Step 4b**
   (correctness head keep/revert) against real numbers instead of the deferred Gate 4 call above.

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
| `p_correct` tracks `p_max` exactly | head learned nothing beyond baseline — flag for the Gate 5 re-decision |
| routed expert weights near zero, loss still falling | shared MLP/attention swallowed the block |
| expert selection collapses to a few MLP slots | aux loss weight too low |
| tokens/sec drops after a code change | a host sync entered the step path |
| loss spikes on resume | schedule re-anchoring or `global_offset` wrong |

`_ExpertTracking` guards against activation-checkpoint recompute double counting via
`begin_forward(expected_updates)`, samples every 8th forward. If expert counts look wrong after
any loop-structure change, check `expected_updates` matches the current `n_loops`.

---

## Post-training

**Target: calibrated abstention, not chain-of-thought.** Calibrated "knows when it doesn't know"
is a shallower, directly measurable (ECE, abstention precision/recall) capability that the
halt/correctness machinery is actually positioned to deliver — multi-step reasoning is not
learnable at 332M total/174M active. Do not evaluate primarily on GSM8K/MATH; math data is in
the mix for representation quality, not benchmark score (SmolLM2-1.7B scored 3.21 on math after
6T tokens).

### Step 12 — SFT — **built, not yet run**

`scripts/sft.py` (+ `scripts/prepare_sft_data.py`, `scripts/eval_abstention.py`,
`modules/data/{chat,abstention,sft_dataset}.py`,
`tests/test_sft_dataset.py`, config.yaml's `sft:` block). Reuses the model/packing path — in fact it
reuses `pretrain.train_step` verbatim, which is what keeps `p_halt`/`p_correct` supervision
*identical* rather than merely similar; swaps data source, adds loss masking over prompt tokens.
Invariants and the reasoning behind them are in [CLAUDE.md](CLAUDE.md)'s "SFT / post-training"
section. Blocked only on the pretraining run finishing (it needs the final checkpoint and the
manifest's holdout hashes off the Hub); nothing here has been exercised against a real checkpoint
yet, and the data prep has not been run against the live Hub.

| dataset | role |
|---|---|
| `HuggingFaceTB/smoltalk2` (no-think splits) | general instruction following |
| `rajpurkar/squad_v2` | **primary abstention supervision** — unanswerable questions are the point |
| `allenai/tulu-3-sft-personas-math` | short worked solutions |
| `openai/gsm8k` (socratic) | short numbered steps |
| `HuggingFaceH4/no_robots` | human-written; tone and refusal style |

- **Exclude the smoltalk2 holdout ids** recorded in `manifest.json` (phase-2 pretraining already
  saw them).
- Mask loss over prompt/system tokens; train on completions only.
- **Do not** use long-CoT trace datasets (KIMI-K2.5/Claude/Fable 5 sets) — Small Model
  Learnability Gap: at this scale they teach fluent filler before a wrong answer, and typically
  carry provider terms restricting training of competing models.
- Keep `p_correct`/`p_halt` supervision active during SFT (still free). If the Gate 5 re-decision
  reverts the correctness head, substitute `p_max` everywhere below; nothing else changes.

**Acceptance:** SQuAD v2 abstention precision/recall on the unanswerable split both reported; ECE
of the abstention signal doesn't degrade relative to the pretrained checkpoint.

**Measured by `scripts/eval_abstention.py`** — it generates an answer for every question in
`squad_v2`'s *validation* split (deliberately never consumed by `prepare_sft_data.py`, precisely so
it can serve here), classifies each completion with `modules.data.abstention.is_abstention`, and
reports abstention precision/recall/F1 plus the false-abstention rate on the answerable half (the
"refuse everything" tell that precision alone hides). Calibration is reported twice: answer-level
(confidence averaged over the generated tokens, scored against whether the answer was right) and
token-level teacher-forced, the latter runnable against the pretrained checkpoint too via
`--baseline-checkpoint` so the "doesn't degrade" half is an actual delta rather than a comparison
between two differently-defined numbers. ECE/AUROC are imported from `scripts/eval_calibration.py`,
so Gate 5's numbers and these come out of the same code. The caveat is printed with the result: the
pretrained checkpoint is out of distribution on the chat control tokens, so a PASS there is weak
evidence and a FAIL is strong. `sft.py`'s validation pass already logs `p_correct`/`p_max`/top-1 on
`sft_val` at checkpoint cadence, which catches "the head learned nothing beyond `p_max`" early, but
it is not the acceptance number.

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

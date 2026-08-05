I've reviewed the whole run path. Here's what I found.

## 1. The FP8 log line (done)

Added `log_precision_mode()` in [pretrain.py:67](scripts/pretrain.py#L67), called from `pretrain()` at [pretrain.py:474](scripts/pretrain.py#L474). It reports the *resolved recipe* and cross-checks TE's own device probe, not just the env var — because \"`USE_FP8` was in the environment\" and \"the GEMMs run in FP8\" are different claims and nobody watches the first seconds of an unattended run.

Verified both branches on the 5090 under WSL:

```
INFO: Precision: BF16 (USE_FP8 not set -- export USE_FP8=1 to train in FP8)
INFO: Precision: FP8 ENABLED -- te.autocast wraps the forward with DelayedScaling(format=HYBRID,
      amax_history_len=16). Note the routed MoE grouped GEMM stays BF16 by design (see moe.py).
```

`test_supervisor.py` still passes; file parses clean.

---

## 2. Blocking defect: the supervisor rewinds phase 2 on every restart

[run_training.py:76](scripts/run_training.py#L76) iterates `PHASES` unconditionally on every launch. `onstart.sh` re-runs the supervisor after every vast reclaim, so **a preemption during phase 2 re-enters phase 1 first**. Traced through:

1. `pretrain(--phase phase1)` resumes the newest checkpoint — a *phase 2* one, token_count ≈ 27B.
2. `resolve_resume_scope` sees `ckpt_phase=\"phase2\" != \"phase1\"` → resets `global_offset`/step to 0, preserves token_count (correct in isolation).
3. `verify_resume` short-circuits because the recorded phase differs — no guard fires.
4. First log step: `27B >= phase1_target (25.415B)` → **writes `checkpoint_phase1_final.pt`**, exit 0.
5. Phase 2 then launches, `find_resume_checkpoint` picks that file (newest mtime), sees `phase1 != phase2`, and **resets the phase-2 document offset to 0.**

Consequences, in order of how much they hurt:

- `checkpoint_phase1_final.pt` — the artifact you'd want for any later ablation — is silently overwritten with phase-2 weights, locally *and* on the Hub under `checkpoints/final/`.
- Phase 2 re-reads `phase2.bin` from document 0 after every preemption, so the anneal keeps seeing the same head of the corpus while `token_count` marches on. Phase 2 is ~4.5B tokens and ~5 h; two or three reclaims and much of the anneal is duplicate data.
- If you free disk by deleting `phase1.bin` after phase 1, the spurious phase-1 launch dies on `FileNotFoundError` → supervisor counts it as a crash → gives up after 5 restarts in 600 s. **Training stops dead.**

This is worth fixing before you rent — a `run_state.json`-based skip in `main()` (or \"skip a phase whose `checkpoint_{phase}_final.pt` exists\") is a few lines. Say the word and I'll write it.

## 3. Other findings

**Must-do before/at rental**

| # | Finding | Why it matters |
|---|---|---|
| A | `HF_TOKEN` must be set as a vast **instance env var**. `huggingface.key` is gitignored, so a fresh clone has no token; `setup.sh` only *warns*, and `onstart.sh` launches training anyway. | `HFSync` stays enabled (repo id is non-empty), every upload 401s, retention correctly refuses to delete → the disk fills instead. Silent for hours. |
| B | Pick the NGC image tag deliberately and verify TE symbols at minute two. `pretrain.py` hard-requires `te.autocast` **and** constructs `MXFP8BlockScaling` / `NVFP4BlockScaling(disable_rht=…)` at import time (lines 28–38) even though only `fp8_recipe` is ever used. | Local TE is **2.15.0**; `te.autocast` lives in `transformer_engine.pytorch.quantization`, which is recent. An older image's TE → `ImportError`/`AttributeError` at import, *after* you've paid for data prep. `run_env_check.sh` only does `import transformer_engine.pytorch` and would **not** catch it. Run `python -c \"import scripts.pretrain\"` as the real check. |
| C | On first boot `onstart.sh` launches the supervisor before `data/prepared/` exists → `Dataset.__init__` `FileNotFoundError` → 5 restarts → supervisor gives up. | Not damaging, but don't set the onstart hook until after `prepare_data.py` has finished, or you'll be reading a confusing log. |

**Worth knowing**

- **HF repo growth.** At `checkpoint_every_tokens: 400M` you push **75 × 2 GB ≈ 150 GB** of rolling checkpoints to `ikeafisch4/temp-train`, and nothing ever deletes the remote copies. Either prune remote rollings occasionally or raise the cadence to 800M (37 ckpts, ~75 GB, at the cost of a ~55 min loss window per preemption instead of ~27 min).
- **Batch size leaves throughput on the table.** Gate 4 peaked at 25.45 GB on a 32 GB card. On 80 GB, `batch_size: 16` / `grad_accumulation_steps: 8` keeps tokens-per-optimizer-step (and therefore `total_steps`, warmup, and the cosine) *identical* while roughly doubling GEMM sizes. Try it during the smoke test; revert if peak mem passes ~65 GB. This is the single cheapest MFU win available.
- **`lambda_ponder: 0.15` was tuned at a very different schedule.** Your config comment already flags this: the ponder ramp completes at 6.7% of the real run vs ~36% in the Gate 4 test, so `p_halt` gets far longer to collapse under CE pressure before ponder engages, and recovering out of a deeper sigmoid saturation needs more λ. Useful lever: **`config.yaml` is re-read at every `pretrain.py` launch**, so you can raise `lambda_ponder` between preemption restarts without losing progress. Safe to change mid-run: `lambda_ponder`, `checkpoint_every_tokens`, `keep_local_checkpoints`. Not safe: anything under `model:`, `target_tokens`, or the `batch_size × seq_length × grad_accumulation_steps` product.
- **Data-prep RAM.** `load_document_texts` reads a whole shard into a Python `list[str]` and the generator holds it for the file's duration — up to 6 sources live at once in phase 1. Only ever exercised at 50M-token smoke scale. Take an instance with ≥64 GB RAM.
- **Doc inaccuracy.** The runbook says \"TE warns on silent fallback\" for FP8. It doesn't — `te.autocast(enabled=True)` calls `check_recipe_support` and **raises**. Better behaviour than documented, but the new log line now tells you either way.
- **Latent, low probability.** `load_checkpoint` can return `losses=None` (legacy `.get` default); `losses.append(...)` would then crash at the first log step. One-line `losses = losses or []` fixes it.

**Verified sound** (no action)

- **Architecture / trainability.** Every FP8-eligible GEMM has both dims divisible by 16 — I checked all of them: decoder q/k/v/o (768→768/192), MLP (768→2304), shared MLP (768→4608), expert attn head_dim 48, `moe_embed_proj` (32→768), main `SmallLMHead` sub-heads (192→16384), MTP `SmallLMHead` sub-heads (48→8192). `test_model_gpu.py` already exercises the exact `DelayedScaling(HYBRID)` recipe training uses.
- The residual stream is well-conditioned: the ×√768 embedding scale is absorbed by the decoder's final RMSNorm, so the MoE loop sees unit-RMS input and `loop_scale = 1/√3` puts the loop stack on par with the decoder from step 0 — the ponder-deadlock precondition (`loop_scale ≈ 0`) genuinely doesn't apply.
- **Gradient accumulation is correct.** I confirmed against the installed accelerate 1.13.0 that `AcceleratedOptimizer.step` *and* `.zero_grad` are both gated on `sync_gradients`, so the unconditional `optimizer.step()` in `train_step` is a no-op on micro-steps. The fp32-master ordering (clip → copy grads → step → copy values → zero) is right.
- **Interruption path.** Atomic `.pt` writes + `os.replace`, `.pt.tmp` sweep at startup, retention requiring *both* out-of-window and confirmed-uploaded, `find_resume_checkpoint` newest-that-loads, `verify_resume` against `run_state.json`, `HFSync.drain()` waiting on `_busy`, flag-only signal handlers, stale-STOP clearing, bounded crash-loop restarts. All of it holds up. The only hole is §2.
- LR math: `total_steps` 57,029; phase 1 ends at step ~48,475 with LR ≈ 6.0e-5, phase 2 anneals to the 4e-5 floor. Token-anchored re-anchoring makes that survive restarts.

---

## 4. Cost

Measured from your live model rather than PLAN.md (`n_loops=3`, subsampled intermediate heads):

```
332.3M total / 173.1M active params
489.6 MFLOP/token forward   (worst-case fully-packed 4096 attention)
1410.9 MFLOP/token training (3x body+attn, 4x checkpointed heads)
```

Gate 4's real measurement implies ~1116 MFLOP/token effective (real packing is cheaper than worst case). Against the H100 SXM's 990 TFLOPS dense BF16 peak:

| Sustained MFU | tok/s | Hours for 29.9B |
|---|---|---|
| 25% | 175–222 k | 37–47 h |
| 30% | 210–266 k | 31–40 h |
| 32% (your Gate 4 number) | 224–284 k | 29–37 h |

Straight peak-scaling from Gate 4 (59.7 k tok/s × 4.73) gives 282 k tok/s / 29 h — treat that as the optimistic end. MFU usually *drops* moving to a 4.7× faster GPU at the same batch, and the routed grouped GEMM is BF16-locked so FP8 only helps the dense path.

**Billed total**

| Item | Hours |
|---|---|
| setup | 0.2 |
| `prepare_data.py` (download + tokenize, GPU idle) | 1.5–3 |
| smoke test | 0.5 |
| training | 30–45 |
| Gate 5 + extraction | 0.5 |
| **total** | **33–49 h** |

At €2/hr: **€66–98**, plus ~€2–4 disk. At a €1.5/hr interruptible bid: €50–74.

**The number that decides it: you need ≥180 k tok/s sustained to fit 29.9B tokens under €100 at €2/hr.** That's exactly what the smoke test measures — if it comes in under 180 k, cut `target_tokens` right there.

## 5. Storage — yes, buy 180 GB

| Item | Size |
|---|---|
| NGC image unpacked | ~25 GB |
| `phase1.bin` + `.idx` | ~51 GB |
| `phase2.bin` + `.idx` | ~9 GB |
| prep scratch (in-flight shards) | ~5 GB |
| checkpoints (2 rolling + 2 final + in-flight `.tmp`) | ~10 GB |
| pip, tokenizer, hub metadata | ~3 GB |
| **peak** | **~103 GB** |

PLAN.md's 120 GB leaves ~17 GB. That \"works\" until you remember the retention design **deliberately fills the disk when uploads fail** — that's its loud-failure mode. At 120 GB you get ~8 held checkpoints ≈ **3.5 h of grace** before the run dies. At 180 GB you get ~38 ≈ **17 h** — enough to notice a token problem overnight and fix it. That, not the corpus, is the real argument for 180 GB.

Disk is billed continuously (including while a preempted instance is stopped) at roughly €0.10–0.20/GB/month → **~€2–4 for the whole run.** Buy the 180 GB.

## 6. Exact steps

**Before renting (free, on the 5090):**
```bash
bash tests/run_tests.sh tests/test_model_gpu.py     # FP8 DelayedScaling path
bash tests/run_tests.sh tests/test_supervisor.py tests/test_checkpoint_lifecycle.py
```
And decide on the §2 fix.

**1 — Rent.** 1× H100 SXM/NVL 80 GB, **interruptible**, **180 GB disk** (fixed at creation), ≥64 GB RAM, image `nvcr.io/nvidia/pytorch:<recent>-py3`. In the instance config set env var **`HF_TOKEN=hf_...`** (finding A). Leave the onstart command **empty** for now (finding C).

**2 — Verify the environment before downloading 60 GB.**
```bash
cd /workspace && git clone --branch train-build https://github.com/ArminBurkhardt/tiny-llm.git && cd tiny-llm
python -c \"import torch, transformer_engine.pytorch, flash_attn; print(torch.cuda.get_device_capability())\"   # expect (9, 0)
```

**3 — Setup.**
```bash
bash scripts/setup.sh --hf-token $HF_TOKEN
python -c \"import sys; sys.path.insert(0,'.'); import scripts.pretrain\"    # finding B: proves te.autocast + the recipe classes exist
```

**4 — Data prep** (1.5–3 h, resumes itself if interrupted).
```bash
nohup python scripts/prepare_data.py > prep.log 2>&1 &
tail -f prep.log
```
Check: per-source counts within 2%, both `.idx` monotonic with last entry == `len(bin)` (the script asserts this), `du -sh data/prepared` ≈ 60 GB.

**5 — Smoke test** (~5 min). This is the decision point.
```bash
USE_FP8=1 python scripts/pretrain.py --phase phase1
# watch for: \"Precision: FP8 ENABLED\", finite dry-run loss, then ~10 log lines
# Ctrl-C after ~200 steps
rm -rf ckpts/training
```
Read off the log line: **`MFU:`** and **`Tokens/sec:`** (need ≥180 k) and **`Peak Mem:`**. If peak is comfortably under ~35 GB, this is where you set `batch_size: 16` / `grad_accumulation_steps: 8` and re-smoke. If tok/s is below 180 k, cut `target_tokens` now — it's the last cheap moment.

**6 — Set the onstart hook**, now that data exists. Paste `scripts/onstart.sh` as the instance's onstart command so reclaims recover automatically.

**7 — Launch.**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_FP8=1
mkdir -p ckpts/training
nohup python scripts/run_training.py >> ckpts/training/train.log 2>&1 &
```

**8 — Watch.** `tail -f ckpts/training/train.log`, or `cat ckpts/training/status.json`, or just the Hub repo. Per PLAN.md's monitoring table, the three to actually watch are `loop_scale` (all 3 entries growing, not one), `p_halt` (should plateau ~0.28–0.30, not pin at 0 or 1), and per-loop CE (should differ across loops).

**9 — Stop / recover.** `touch ckpts/training/STOP` to end cleanly (exit 10, ~23 s). `kill -USR1 <pid>` to checkpoint without stopping. Reclaims need nothing.

**10 — Before releasing the instance.** Don't skip this, re-renting to compute it is pure waste:
```bash
python scripts/eval_calibration.py -c ckpts/training/checkpoint_phase2_final.pt --phase phase2
```
Record ECE and abstention AUROC for `p_correct` vs `p_max` — that's the deferred Step 4b keep-or-revert decision, now against real numbers. Then scp both `checkpoint_phase{1,2}_final.pt` down (they're on the Hub too, but verify before you release).

---

Want me to fix §2 (the phase-loop rewind) and the `losses = losses or []` hardening? Both are small and I'd rather they land before you're paying by the hour.

# Runbook: the unattended pretraining run

Everything you need to start, watch, stop, and recover a ~40 hour pretraining run on a rented,
interruptible vast.ai box. Read the "If something looks wrong" table before the run, not during it.

Every command runs **from the repo root**. `config.py` opens `config.yaml` with a relative path,
so `cd scripts && python pretrain.py` fails.

---

## 1. What you actually run

Four commands, in this order. Only the last one is long-lived.

| # | Command | Runtime | Interruptible |
|---|---------|---------|---------------|
| 1 | `bash scripts/setup.sh --hf-token hf_xxx` | ~3 min | yes, just re-run |
| 2 | `python scripts/prepare_data.py` | hours | yes, resumes itself |
| 3 | `USE_FP8=1 python scripts/pretrain.py --phase phase1` (smoke, then Ctrl-C) | ~5 min | yes |
| 4 | `python scripts/run_training.py` | ~40 h | yes, restarts itself |

On vast.ai you can paste `scripts/onstart.sh` as the instance's onstart command to survive
reclaims — it clones, runs setup, and launches the supervisor under `nohup`. It deliberately does
**not** run `prepare_data.py`; that has its own multi-hour lifecycle and its own resume state, and
tangling the two makes both harder to diagnose. Run step 2 by hand first: the hook exits
immediately (without launching training) if `data/prepared/phase1.idx` is missing, so pasting it
in before data prep just means it does nothing on the first boot — set it after step 2 completes.

### 1.1 `scripts/setup.sh`

Installs the non-CUDA dependencies, writes `huggingface.key` (mode 600, gitignored), downloads the
tokenizer, and runs two preflight checks.

Do **not** `pip install -r requirements.txt` on the box. The NGC image ships torch,
`transformer_engine` and `flash-attn` prebuilt for the rented GPU; the requirements file's wheels
target the local dev GPU and would clobber them.

The preflight **uploads a file to `ikeafisch4/temp-train` and deletes it again**. This proves the
token has *write* scope, not just that the network works. A read-only token would otherwise fail
every upload silently for 40 hours, and under the retention policy (§3) that ends in a full disk
rather than an error. If this fails, fix the token before doing anything else.

The second preflight probes `nvidia/Nemotron-CC-Math-v1`, the one Hub-gated data source. A warning
here only matters if you have not run `prepare_data.py` yet.

### 1.2 `scripts/prepare_data.py`

Builds `data/prepared/phase1.{bin,idx}` and `phase2.{bin,idx}`. Checkpoints itself every 2000
documents into `_prepare_state_{phase}.json`; if it dies, re-run the identical command and it
truncates back to the last confirmed state and continues. Peak disk stays bounded — it holds one
shard file per source at a time and deletes each after tokenizing.

Peak *RAM* is a separate concern: `load_document_texts` reads a whole shard into a Python
`list[str]` and the generator holds it for that file's duration, and up to 6 sources are live at
once in phase 1. Only ever exercised at 50M-token smoke scale — pick an instance with **>=64GB
RAM** for the real run.

### 1.3 The smoke test

Run phase 1 by hand for ~200 steps and check the log line before committing to the real run:

- `MFU:` should be in the region of the Gate 4 extrapolation. **More than ~20% off and you should
  redo the budget math before starting** — this is the last cheap moment to change `target_tokens`.
- `Peak Mem:` should leave headroom on an 80GB card. Gate 4 peaked at 25.45GB on a 32GB card; on an
  80GB H100 try `batch_size: 16` / `grad_accumulation_steps: 8` in `config.yaml` (keeps
  tokens-per-optimizer-step, and therefore `total_steps`/warmup/the cosine, identical while roughly
  doubling GEMM sizes) — the cheapest available MFU win. Revert if peak mem passes ~65GB.
- FP8 should actually be active. **`te.autocast(enabled=True)` raises if the device rejects the
  recipe — it does not silently fall back.** `log_precision_mode()`'s log line at startup states
  the resolved recipe either way, so check that instead of assuming a missing warning means BF16.

Then **delete `ckpts/training/` and start clean**, so the real run does not resume from a
200-step smoke checkpoint.

### 1.4 `scripts/run_training.py`

The supervisor. Runs `pretrain.py --phase phase1` to its token target, then `--phase phase2`,
relaunching through preemptions. This is the only thing you leave running.

---

## 2. Watching it

```bash
tail -f ckpts/training/train.log      # the log the supervisor writes
cat ckpts/training/status.json        # machine-readable snapshot, rewritten every ~3s
```

`status.json` is the fastest way to answer "how far along is it":

```json
{
  "phase": "phase1",
  "tokens": 12400000000,
  "phase_target": 25415000000,
  "run_target": 29900000000,
  "tokens_per_sec": 198000.0,
  "loss": 2.91,
  "eta_phase": "18h 14m",
  "eta_run": "24h 32m",
  "step": 23640,
  "epoch": 0
}
```

`eta_phase` is time to the end of the current phase; `eta_run` is time to the end of both. Both are
extrapolated from the throughput of the last log interval only, so they jump around early and after
a restart. The same figures appear at the end of every log line as `ETA:`.

Everything is also mirrored to `https://huggingface.co/ikeafisch4/temp-train` at every checkpoint,
so you can check progress without an ssh session:

```
checkpoints/checkpoint_phase1_tok12400M_loss2.9134.pt   rolling
checkpoints/final/checkpoint_phase1_final.pt            terminal, per phase, never deleted
graphs/loss_graph.png                                   overwritten each time
graphs/expert_selection.png                             overwritten each time
logs/train.log
status.json
manifest.json
```

**Remote retention mirrors local retention.** Every rolling checkpoint `prune_checkpoints` deletes
locally (§3) is also deleted from `checkpoints/` on the Hub, and `HFSync` squashes the repo's git
history afterward (throttled to at most once per 30 minutes) so the deleted 2GB blob's storage is
actually reclaimed rather than just hidden from the current tree — a plain Hub file delete alone
leaves it referenced by history and still billed. Both the delete and the squash go through the
same non-fatal retry-and-log path as uploads; a failure never touches the training loop, and a
failed delete just leaves stale clutter on the Hub rather than a local retention problem. Because
history gets rewritten, do not expect old commits on `ikeafisch4/temp-train` to stay browsable —
this repo is a scratch mirror of local checkpoints, not something meant to be read commit-by-commit.

---

## 3. Checkpoints on disk

`ckpts/training/`, ~2GB each, written every `checkpoint_every_tokens` (400M ≈ 30 min ≈ 75 over the
run). Naming is token-keyed, not step-keyed: `checkpoint_phase1_tok12400M_loss2.9134.pt`.

Two rules worth knowing before you go looking for a file that is gone, or wonder why the disk is
filling:

- **A checkpoint is only deleted once it is BOTH outside the keep window (`keep_local_checkpoints`,
  default 2) AND confirmed uploaded.** A checkpoint deleted locally and never uploaded is gone
  forever; one kept because its upload failed only costs disk. If uploads break, the disk fills —
  loud and fixable, which is the intended failure.
- **`checkpoint_{phase}_final.pt` is never pruned.** One per phase, written when the phase ends.

Writes are atomic (write to `.pt.tmp`, `fsync`, then `os.replace`), so a preemption mid-write
cannot leave a truncated file that is also the newest by mtime. On startup any leftover `.pt.tmp`
is cleaned up.

---

## 4. Stopping it

| What you want | Do this | Exit code | Supervisor restarts? |
|---|---|---|---|
| Stop the whole run, cleanly | `touch ckpts/training/STOP` | 10 | no |
| Checkpoint right now, keep training | `kill -USR1 <pid>` | — | n/a |
| Stop this process, expect a restart | `kill <pid>` (SIGTERM) | 20 | yes |
| Stop from an attached terminal | Ctrl-C | 10 | no |

The STOP sentinel is the one to use remotely — it needs no ssh session and no process lookup, just
a file. Both the sentinel and the signal flags are read at the log cadence (~3s), then a 2GB
`torch.save` has to finish. **Measured end to end on a 5090: 23 seconds** from `touch` to the
supervisor exiting 10. Budget for that, not for instant.

A stale `STOP` file is deleted at startup, so a relaunch is never stopped by the previous run's
sentinel. To stop the supervisor *and* the current phase, create the sentinel and let it exit —
killing the supervisor alone leaves `pretrain.py` running.

**Exit codes**, shared between `pretrain.py` and the supervisor:

| Code | Meaning | Supervisor |
|---|---|---|
| 0 | phase finished (target reached or data exhausted) | next phase |
| 10 | you asked it to stop | stops |
| 20 | preempted / SIGTERM | relaunches |
| 30 | **resume verification failed** | stops — see §6 |

---

## 5. Expected events (not problems)

| You see | What it is |
|---|---|
| `checkpoint ... failed to load ... trying the next oldest` | A checkpoint was truncated by a preemption. Working as designed; the run continues from the previous one. |
| `WARNING: stop requested (signal 15)` | vast reclaimed the instance. The supervisor relaunches. |
| `phase1 preempted; relaunching in 30s` | Normal for an interruptible instance. Backoff doubles, capped at 300s. |
| `pruned N uploaded checkpoint(s) past the newest 2` | Retention doing its job. |
| `phase1 data exhausted at ... saving final checkpoint` | Legitimate end of a phase. Phase 1's corpus is only ~0.3% larger than its token target, so either this or "reached target" can fire first; both save a final checkpoint and exit 0. |
| `checkpoint was written during phase1, now training phase2: resetting the document offset` | The phase handoff. The token count is deliberately preserved so the LR schedule continues rather than restarting. |
| `skipping already-complete phase(s) on disk: phase1` | A reclaimed instance came back and `onstart.sh` started a brand new `run_training.py`. `resume_phase_index` (`modules/runtime/checkpoints.py`) checked `ckpts/training` before looping and found phase 1 already finished, so it starts straight at phase 2 instead of re-entering phase 1 and overwriting `checkpoint_phase1_final.pt` with phase-2 weights. |
| ETA swinging wildly in the first few minutes after a restart | It extrapolates from one log interval. It settles. |
| `mean loops` below `n_loops` | `loop_count_sampling` runs 30% of steps at a reduced depth on purpose. Log steps are pinned to full depth. |
| `ponder auto-adjust: p_halt too low/high ... lambda_ponder X -> Y` | `PonderController` (`modules/runtime/ponder.py`) nudging the ponder weight to keep `p_halt`'s steady state in the healthy band (`ponder_target_p_halt` +/- `ponder_p_halt_band` in `config.yaml`). Only fires after the warmup+ramp finishes, at most once per `ponder_adjust_cooldown_tokens`. The adjusted value is checkpointed, so it survives every preemption restart; disable with `ponder_auto_adjust: false`. |

---

## 6. If something looks wrong

| Symptom | Almost certainly | Do this |
|---|---|---|
| **Exit code 30**, `resume verification failed` | The process resumed materially behind what `run_state.json` recorded — the wrong checkpoint loaded, or the state was lost. | **Do not just restart.** Look at `ckpts/training/` and `run_state.json` and work out which checkpoint is real. Restarting blindly retrains ground already covered, at full cost. |
| Disk filling up | Uploads are failing, so retention refuses to delete. | Check the log for `upload of ... failed after 3 attempts`. Usually a token/network problem. Fix it; uploads resume at the next checkpoint. |
| `keeping N checkpoint(s) past the retention window because their upload has not succeeded` | Same as above, caught earlier. | Same. |
| Loss is `nan` | Almost always precision. | Re-run without `USE_FP8=1`. The dry run asserts a finite loss at startup, so a `nan` appearing later is a training-dynamics problem, not a config one. |
| Training "succeeded" in minutes having done nothing | A phase-2 run whose document offset was not reset. | This is the bug `resolve_resume_scope` exists to prevent — check the checkpoint's `phase` field. |
| `upload queue full; dropping ...` | Uploads are slower than checkpointing. | Only rolling checkpoints are ever dropped, and a dropped one stays on disk un-pruned. If it persists, the box's uplink cannot keep up with a 2GB/30min cadence — raise `checkpoint_every_tokens`. |
| Supervisor gave up: `restarted N times in 600s` | A crash loop, not a preemption. | Read the last child traceback in the log. The flap limit exists so a crash loop does not silently bill you for hours. |
| `setup: FATAL: no HF token, but config.yaml's hf_upload_repo resolves to ...` | `huggingface.key`/`$HF_TOKEN` missing or empty, and `config.yaml` has uploads enabled. `setup.sh` now exits non-zero here rather than only warning, so `onstart.sh` (which runs under `set -euo pipefail`) never launches training with a token that would 401 every upload for 40 hours. | Set `$HF_TOKEN` (a vast.ai **instance env var** survives a reclaim; `huggingface.key` alone does not, since it's gitignored and a fresh clone won't have it) or pass `--hf-token hf_xxx`. Or set `hf_upload_repo: ""` to run local-only on purpose. |

---

## 7. Recovering after the instance is reclaimed

If `onstart.sh` is configured, nothing — the box comes back, clones, sets up, and the supervisor
resumes from the newest valid checkpoint.

By hand:

```bash
cd /workspace/tiny-llm
bash scripts/setup.sh --hf-token hf_xxx
# if ckpts/training is empty (fresh disk), pull the last checkpoint back down first:
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('ikeafisch4/temp-train', allow_patterns='checkpoints/*', local_dir='ckpts/training')"
python scripts/run_training.py
```

The checkpoint carries `token_count` and `phase`, and the LR schedule is re-anchored by token
count rather than by saved step, so resuming with a different batch size or grad-accumulation
setting still lands on the right point of the cosine.

**Do not rsync `data/prepared/` or the box's `manifest.json` down.** The corpus is 60GB and
rebuildable; the local `manifest.json` describes a 45M-token *test* corpus and does not match
either the box's data or the real config.

---

## 8. Phase boundaries

| | phase 1 | phase 2 |
|---|---|---|
| corpus | `phase1.bin` (~25.5B tokens) | `phase2.bin` (~4.5B tokens) |
| stops at | 25,415,000,000 tokens (85% of `target_tokens`) | 29,900,000,000 (the combined target) |
| LR | warmup → cosine, still descending | continues the same cosine to its floor |

`target_tokens` is the **combined** budget, and `token_count` carries across the boundary. That is
what makes phase 2 an anneal rather than a second run. A phase 1 that ends early (data exhausted
below target) is self-correcting: phase 2 simply gets the remaining tokens.

---

## 9. Files the run reads and writes

```
ckpts/training/
  checkpoint_{phase}_tok{N}M_loss{L}.pt   rolling, pruned
  checkpoint_{phase}_final.pt             terminal per phase, never pruned
  run_state.json                          {phase, token_count, checkpoint} -- read at startup to verify the resume
  status.json                             progress + ETA, rewritten every log interval
  loss_graph.png                          overwritten (not per step any more)
  expert_selection.png                    overwritten
  train.log                               only when launched via onstart.sh
  STOP                                    you create this; deleted at startup
data/prepared/{phase1,phase2}.{bin,idx}   from prepare_data.py
huggingface.key                           your token, mode 600, gitignored
```

All of `ckpts/` and `data/prepared/` are gitignored, as is every `*.json` — `run_state.json` and
`status.json` are runtime state, not tracked files.

---

## 10. Running SFT (PLAN.md Step 12) on a rented box

SFT is written for the local dev GPU, but it runs unattended on vast.ai. It is a *different shape*
of job from pretraining: ~300M tokens x 2 epochs, so **1-3 hours, not 40**, and there is no
supervisor because there are no phases — only epochs, and the epoch is in the checkpoint.

**Rent on-demand, not interruptible.** The interruptible discount is worth ~EUR 1-3 on a run this
short, against a real chance of babysitting restarts. `sft.py` does honour the same stop contract
as pretraining (SIGTERM → checkpoint + exit 20, `ckpts/sft/STOP` → exit 10, SIGUSR1 → checkpoint and
keep going), so if you do take an interruptible box, wrap it: `until python scripts/sft.py
--upload-repo ikeafisch4/temp-train; do sleep 10; done` — exit 10 and 0 both stop the loop, exit 20
relaunches and resumes from the newest loadable checkpoint in `ckpts/sft`.

### 10.1 On the same instance that just finished pretraining (preferred)

The tokenizer, `manifest.json` and the final checkpoint are all already on disk, and
`checkpoint_phase2_final.pt` is right there — no downloads.

```bash
# stop the onstart hook from relaunching the supervisor on the next boot (clear it in the vast
# console). It is harmless if it does run -- resume_phase_index sees checkpoint_phase2_final.pt
# and phase 2 exits immediately -- but it churns a final checkpoint save for nothing.
unset USE_FP8                      # see 10.3
python scripts/prepare_sft_data.py --target-tokens 300000000
python scripts/sft.py -c ckpts/training/checkpoint_phase2_final.pt \
                      --upload-repo ikeafisch4/temp-train
```

Disk: the SFT corpus is ~1GB (bin + mask) plus a few GB of shard scratch, against ~60GB already
used by `phase{1,2}.bin`. Fits in 120GB with room; delete the phase bins only if you need to.

### 10.2 On a fresh instance

```bash
bash scripts/setup.sh --hf-token X          # deps, token, tokenizer, preflight
bash tests/run_env_check.sh
python scripts/prepare_sft_data.py --pull-manifest   # 20-40 min, download-bound
python scripts/sft.py --from-hub --upload-repo ikeafisch4/temp-train
```

`--pull-manifest` is **not optional here.** `manifest.json` is gitignored (`*.json`), so a fresh
clone has no `smoltalk2_holdout_hashes`, and without them the SFT corpus would re-train on the
smoltalk2 conversations phase-2 pretraining already saw. The script refuses to run with an empty
holdout list rather than doing that silently; `--ignore-holdout` overrides, but do not.

### 10.3 What to change from the local defaults

| | why |
|---|---|
| **`--upload-repo <repo>`** | `sft.hf_upload_repo` is `""` (uploads off) because SFT is a local run. On a rented box that means the checkpoints die with the instance. This is the one you must not forget. |
| **`unset USE_FP8`** | `onstart.sh` exports `USE_FP8=1`, and `sft.py` imports the same recipe resolution from `pretrain.py`, so it would inherit FP8. At `lr=3e-5` the FP8 GEMM noise is large relative to the update — the same margin argument that forces fp32 masters for every parameter (see `build_sft_param_groups`). The run is short enough that the throughput does not matter. |
| **`sft.batch_size: 16`, `grad_accumulation_steps: 2`** | 4 x 8 is sized for a 32GB 5090. On an 80GB H100 this keeps tokens/step at 131k and is far faster. Both are in `config.yaml`'s `sft:` block. |
| `--target-tokens` | 300M is the default corpus size. Raising it costs prep time and disk, not much else. |

Everything else carries over unchanged. `ckpts/sft/` is a separate directory from `ckpts/training/`
on purpose, so `resume_phase_index` can never mistake an SFT checkpoint for a pretraining phase.

### 10.4 Watching it

`ckpts/sft/status.json` and the log line are the same shape as pretraining's, plus an `[eval]` line
every `sft.eval_every_tokens` (25M) reporting val CE, `p_correct`, `p_max` and top-1 on `sft_val`.
The one to watch: **`p_correct` tracking `p_max` exactly** means the correctness head learned
nothing beyond the free baseline — that is the Gate 5 / Step 4b re-decision showing up early, not a
crash. Loss should start near where pretraining ended and drop quickly in the first few hundred
steps as the model learns the chat format, then flatten.

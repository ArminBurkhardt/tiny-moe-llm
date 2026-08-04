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

On vast.ai you can skip straight to step 4 by pasting `scripts/onstart.sh` as the instance's
onstart command — it clones, runs setup, and launches the supervisor under `nohup`. It deliberately
does **not** run `prepare_data.py`; that has its own multi-hour lifecycle and its own resume state,
and tangling the two makes both harder to diagnose.

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

### 1.3 The smoke test

Run phase 1 by hand for ~200 steps and check the log line before committing to the real run:

- `MFU:` should be in the region of the Gate 4 extrapolation. **More than ~20% off and you should
  redo the budget math before starting** — this is the last cheap moment to change `target_tokens`.
- `Peak Mem:` should leave headroom on an 80GB card.
- FP8 should actually be active. TE warns when it silently falls back.

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
| ETA swinging wildly in the first few minutes after a restart | It extrapolates from one log interval. It settles. |
| `mean loops` below `n_loops` | `loop_count_sampling` runs 30% of steps at a reduced depth on purpose. Log steps are pinned to full depth. |

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
| `no HF token` at setup | `huggingface.key` missing or empty. | `bash scripts/setup.sh --hf-token hf_xxx`. |

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

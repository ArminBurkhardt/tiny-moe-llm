# Unattended training run — design

**Date:** 2026-08-04
**Scope:** everything needed to launch `tiny-moe-llm` pretraining on a rented, interruptible
vast.ai box and have it survive to completion without a human watching it.

## Problem

The model and data pipeline are ready — 12/12 tracked tests pass, and the invariants in
`CLAUDE.md` hold. What does not exist is the operational layer around them. A fresh clone on a
rented box today has no tokenizer, writes a 2 GB checkpoint every ~4 minutes with no retention
(~1.2 TB against a 120 GB disk), writes checkpoints non-atomically, blocks on `input()` in its
only interrupt path, never uploads anything off-box, and silently trains zero batches if you
switch to phase 2. Each of those alone ends the run.

Run shape the design is sized against: `batch_size 8 × seq_length 4096 × grad_accum 16` =
524,288 tokens per optimizer step; `target_tokens` 29.9e9 ⇒ ~57,000 optimizer steps ⇒ **~40 h**
wall clock at 200K tok/s. Phase 1 is 85% of that budget (25.4B tokens), phase 2 the remaining
15% (4.5B).

## Non-goals

- Steps 12–13 (SFT, calibration set). They run later on cheap hardware and do not gate renting.
- Changing training dynamics. One dynamics question is *flagged* below and deliberately left
  unfixed.
- Multi-GPU / distributed. Single-GPU box, as PLAN.md assumes throughout.

## Architecture

New package `modules/runtime/`, four small single-purpose modules. `scripts/pretrain.py` imports
them and stays a training loop rather than growing another 400 lines.

```
modules/runtime/__init__.py      empty, per repo convention (imports are fully qualified)
modules/runtime/checkpoints.py   naming, latest-VALID selection, retention/pruning
modules/runtime/hf_sync.py       background upload thread + queue
modules/runtime/control.py       STOP sentinel + SIGTERM/SIGUSR1 -> one flag
modules/runtime/status.py        status.json writer (phase, tokens, ETA, restarts)
```

`utils.py` keeps `save_checkpoint` / `load_checkpoint` — `tests/test_checkpoint_roundtrip.py`
imports them and moving them is pure churn — but gains atomicity and a `phase` field.

New scripts:

```
scripts/fetch_tokenizer.py   snapshot_download of the public tokenizer repo
scripts/run_training.py      supervisor: phase1 -> phase2, restart, ETA, flap limit
scripts/setup.sh             deps + HF token + tokenizer + env check (replaces vast_init)
scripts/onstart.sh           vast onstart one-liner: clone -> setup -> nohup run_training
```

Deleted: `vast_init` (superseded by `scripts/setup.sh`).

## 1. Tokenizer distribution

Both HF repos exist and are **public**:

- `ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536` — contains `tokenizer.json`,
  `tokenizer_config.json`, `config.json`, `id_remap.json`, `chat_template.jinja`. Public, so
  fetching needs **no** token.
- `ikeafisch4/temp-train` — empty but for `.gitattributes`. Upload target.

No publish script is needed.

**One constant replaces four hardcodes.** `utils.py` gains:

```python
TOKENIZER_REPO = "ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536"
TOKENIZER_DIR = os.environ.get(
    "TINY_LLM_TOKENIZER",
    os.path.join(BASE_DIR, "ckpts", "pretrained", "DeepSeek-V4-Pro-tokenizer-65536"),
)
```

Call sites updated: `scripts/pretrain.py:416`, `scripts/inference.py:104`,
`scripts/eval_calibration.py:223`, `scripts/prepare_data.py:55`, and
`tests/run_env_check.sh:22` (which currently points at the *unpruned* `DeepSeek-V4-Pro-tokenizer`
that will not exist on the box at all).

`scripts/fetch_tokenizer.py` does `snapshot_download(TOKENIZER_REPO, local_dir=TOKENIZER_DIR)`
and is idempotent (skips if `tokenizer.json` is already present unless `--force`).

## 2. HF token

One resolver, `utils.get_hf_token()`, in this order:

1. `$HF_TOKEN`
2. `<repo root>/huggingface.key` (already gitignored via `*.key`, already the local convention)
3. the `huggingface_hub` login cache
4. otherwise `None`, with callers raising a clear error naming `scripts/setup.sh --hf-token`

`setup.sh --hf-token hf_xxx` writes `huggingface.key` (mode 0600) and exports `HF_TOKEN` for the
rest of the session. The token is needed for **uploads** and for `prepare_data.py`'s one gated
source (`nvidia/Nemotron-CC-Math-v1`); not for the tokenizer fetch.

## 3. Checkpoint lifecycle

### 3.1 Atomic writes

`save_checkpoint` writes to `<path>.tmp`, `flush()` + `os.fsync()` on the file handle, then
`os.replace(tmp, path)`. `os.replace` is atomic on both POSIX and Windows. A preemption mid-write
can then leave a stray `.tmp` (cleaned on next startup) but never a truncated `.pt` that is also
newest-by-mtime.

### 3.2 Cadence in tokens

New `config.yaml` key under `training:`:

```yaml
checkpoint_every_tokens: 400000000   # ~30 min at 200K tok/s -> ~75 checkpoints over the run
```

surfaced as `TrainingConfig.checkpoint_every_tokens`. Checked inside the existing `LOG_INTERVAL`
block, where the token counter is already being drained for logging — **no new host sync**. The
trainer tracks `next_checkpoint_tokens` and fires when `n_tokens >= next_checkpoint_tokens`.

This replaces `step % 1500`, which counted *micro*-steps: at `grad_accum 16` that was every 93.75
optimizer steps ≈ 49M tokens ≈ every 4 minutes ≈ 608 checkpoints ≈ 1.2 TB. Expressing the cadence
in tokens also makes it invariant to batch-size and grad-accum changes, which the step form was
not.

### 3.3 Naming

```
checkpoint_{phase}_tok{N}M_loss{L:.4f}.pt        rolling   (N = token_count // 1_000_000)
checkpoint_{phase}_final.pt                       terminal, per phase
```

`checkpoint_{phase}_final.pt` is written for both terminal cases — phase target reached and
dataset exhausted — since from a downstream consumer's point of view they are the same event:
this is the end of that phase's training.

Dropping `epoch{E}_idx{STEP}` from the name: `num_epochs` is now a safety net (PLAN.md Step 6) and
`dataset_idx` is not the resume key anymore (`global_offset` is). Token count is the only figure
that identifies a checkpoint's position in the run. `load_checkpoint` is unchanged and still reads
old-format files; only newly written names change.

### 3.4 Retention

`keep_local_checkpoints: 2` in `config.yaml`. After each successful save, prune rolling
checkpoints that are **both**:

- outside the newest-N window, **and**
- confirmed uploaded to HF (`hf_sync` marks each path on success).

If uploads are failing, the disk fills instead of history vanishing, and each skipped deletion
logs a warning naming the un-uploaded file. This is deliberate: a silently-deleted-and-never-
uploaded checkpoint is unrecoverable, a full disk is a loud, fixable failure.

`checkpoint_*_final.pt` is **never pruned**, regardless of the keep window.

### 3.5 Latest-VALID selection

`get_latest_checkpoint_epoch` is replaced by `modules/runtime/checkpoints.py:find_resume_checkpoint`,
which sorts candidates newest-mtime-first and returns the first that loads, logging an `ERROR` for
each it skips. It raises only when **no** candidate loads.

This preserves the existing invariant — "a checkpoint exists but will not load" must never
degrade into "start from token 0" — while surviving a single corrupt file (disk-level corruption;
§3.1 already makes truncation impossible). Stray `*.tmp` files are deleted at startup, not
considered as candidates.

### 3.6 Graphs

`ckpts/training/loss_graph.png` and `ckpts/training/expert_selection.png`, fixed filenames,
overwritten at the checkpoint cadence rather than at `step % sliding_window_size` (which would
produce ~7,100 files). Pre-existing per-step PNGs are cleaned up on startup.

## 4. HF sync

`modules/runtime/hf_sync.py` exposes an `HFSync` object owning one background `threading.Thread`
and a `queue.Queue` of `(local_path, repo_path, delete_after)` jobs. `pretrain.py` enqueues and
returns immediately — a 2 GB upload at 200 Mbps is ~80 s, and blocking the loop for that every
30 min is ~4% throughput for no benefit.

Repo layout on `ikeafisch4/temp-train`:

```
checkpoints/checkpoint_phase1_tok25400M_loss2.9134.pt
checkpoints/final/checkpoint_phase1_final.pt
logs/train.log
graphs/loss_graph.png
graphs/expert_selection.png
status.json
manifest.json
```

Uploaded on every checkpoint: the `.pt`, `train.log`, both PNGs, `status.json`, and
`manifest.json` **if it exists on the box**. That last is the fresh one `prepare_data.py` writes
during the box's own data prep — it carries the smoltalk2 holdout hashes Step 12 SFT needs, and
dies with the instance otherwise. The local 45M-token test manifest is never involved; the runbook
carries an explicit do-not-rsync note for `data/prepared/` and the local `manifest.json` (whose
`_prepare_state_*.json` sidecars would make `prepare_data.py` think it had already finished).

Failure handling: 3 retries with exponential backoff. **An upload failure never propagates into
the training loop.** It logs an error, leaves the path unmarked (blocking its deletion per §3.4),
and the thread continues with the next job. Queue depth is capped; if uploads fall far enough
behind that the queue is full, the oldest *rolling* checkpoint job is dropped (logged) rather than
blocking training — `final` and non-checkpoint jobs are never dropped. A dropped job leaves its
file unmarked, so §3.4 will refuse to delete it: sustained upload failure converges on a full
disk, which is the loud, recoverable failure, not on silently discarded history.

Shutdown: `HFSync.drain(timeout)` is called on clean exit and after the final per-phase
checkpoint, so a STOP or a phase transition does not race the uploader.

## 5. Stopping and restarting

`modules/runtime/control.py` installs handlers and exposes a single `should_stop()` /
`should_checkpoint_now()` pair backed by module-level flags. Signal handlers do nothing but set a
flag — no I/O, no allocation.

| trigger | effect | exit code |
|---|---|---|
| `touch ckpts/training/STOP` | checkpoint, drain uploads, exit | **10** (supervisor does not restart) |
| `SIGTERM` (vast preemption, `kill`) | checkpoint, drain uploads, exit | **20** (supervisor restarts) |
| `SIGUSR1` (`kill -USR1 <pid>`) | checkpoint now, keep training | — |
| `Ctrl-C`, `sys.stdin.isatty()` true | prompt, then save (today's behaviour) | 10 |
| `Ctrl-C`, no tty | save immediately, no `input()` | 10 |
| phase token target reached | save `checkpoint_{phase}_final.pt`, exit | **0** |
| dataset exhausted before target | save `checkpoint_{phase}_final.pt`, exit | **0** |

The flag and the STOP sentinel (`os.path.exists`) are both read in the `LOG_INTERVAL` block — at
~0.16 s per micro-step and `LOG_INTERVAL 20`, that is ~3 s granularity, inside vast's SIGTERM
grace, and it costs no GPU sync. Best-effort by nature: a 2 GB `torch.save` still takes time, and
a short enough grace period can cut it off. §3.1's atomicity means a cut-off save loses that one
checkpoint, not the previous one.

The dataset-exhaustion case is a real current bug: phase 1's data runs out at ~25.5B, below
`target_tokens` 29.9B, and today the epoch loop just ends with no final save — losing up to a full
checkpoint interval.

### 5.1 Resume verification

Verification lives in `pretrain.py`, not the supervisor, because that is where the checkpoint is
loaded and where the token count becomes known.

`ckpts/training/run_state.json` is written at every checkpoint with `{phase, token_count, ckpt}`.
On startup, after the resume, `pretrain.py` compares:

- If `run_state.json` records `phase == current phase` and `token_count = X > 0`, and the resumed
  count is below `X - 2 * checkpoint_every_tokens`, **abort with a non-retryable exit code (30)**.
  That slack covers the legitimate case of falling back to an older checkpoint per §3.5; anything
  beyond it means the wrong file was loaded or the state was lost, and continuing burns 40
  GPU-hours re-covering ground.
- If `run_state.json` records a *different* phase, no comparison is made (§6 handles the
  transition).
- Missing `run_state.json` with no checkpoint present is a normal cold start.

`tests/` gets a case for the abort path — the whole point is that it fires rather than being
comfortable.

## 6. Phase orchestration

### 6.1 `--phase` flag

`pretrain.py` gains `--phase phase1|phase2`, defaulting to `TrainingConfig.phase` from
`config.yaml`. This removes the need to edit a tracked file on the box.

### 6.2 Phase-scoped resume

`save_checkpoint` gains `phase=` and stores it. On load:

- **`global_offset` resets to 0** when the checkpoint's phase differs from the current phase. The
  phase-2 corpus is a different document stream (~4M docs) and phase 1's offset (~23M docs) makes
  `range(first, num_docs, num_workers)` empty — the dataloader yields zero batches and training
  exits looking successful. This is the current silent-failure bug.
- **`token_count` carries over.** The LR cosine is anchored to the *combined* 29.9B budget, so
  phase 2 must continue the decay from 25.4B into the anneal, not restart it.
- `epoch` resets to 0.

Legacy checkpoints without a `phase` key are treated as belonging to the current phase (matching
today's behaviour) and logged.

### 6.3 Per-phase targets

New `phase1_fraction: 0.85` in `config.yaml` (PLAN.md Step 5/6's 85/15 split), giving
`TrainingConfig.phase_target_tokens(phase)`:

- `phase1` → `int(target_tokens * 0.85)` = 25.415B
- `phase2` → `target_tokens` = 29.9B

`target_tokens` stays the combined figure, so `total_steps` and the LR anchor are untouched. The
stop condition in the `LOG_INTERVAL` block compares against the *phase* target.

### 6.4 Supervisor — `scripts/run_training.py`

```
for phase in (phase1, phase2):
    loop:
        rc = subprocess.run([python, scripts/pretrain.py, --phase, phase])
        rc == 0   -> phase complete, advance
        rc == 10  -> user stop, exit supervisor
        rc == 30  -> resume verification failed, exit supervisor (do NOT retry)
        rc == 20  -> preempted, relaunch after 30 s
        else      -> crash, relaunch after 30 s backoff (capped at 300 s)
        flap limit: >5 relaunches within 10 min -> exit supervisor
```

It writes `ckpts/training/status.json` throughout: phase, tokens done / phase target / run target,
tokens/sec (EMA from the child's log), **ETA to phase end and to run end**, restart count, and last
error. `pretrain.py` also logs the ETA on each log line, since that is where you actually look.

## 7. Setup

### 7.1 `scripts/setup.sh` (replaces `vast_init`)

1. `pip install --no-cache-dir` of non-CUDA deps only. torch / transformer_engine / flash_attn
   come from the NGC image and must not be touched.
   **Removed from the current list:** `sentence-transformers` (unimported anywhere, and it pulls a
   torch dependency that can clobber the NGC prebuilt torch and break TE/flash-attn with it),
   `bitsandbytes` (only a commented-out line at `pretrain.py:451`), `fastparquet`
   (`prepare_data.py` forces the pyarrow engine). Torch-dependent installs get `--no-deps`.
2. `--hf-token` → `huggingface.key` (0600) + `export HF_TOKEN`.
3. `python scripts/fetch_tokenizer.py`.
4. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
5. `bash tests/run_env_check.sh` (now pointing at the correct tokenizer dir).

### 7.2 `scripts/onstart.sh`

The single line pasted into vast's onstart box: clone the repo at the pinned branch, run
`setup.sh`, then `nohup python scripts/run_training.py`. Data prep stays a separate manual command
— it has its own hours-long interruptible lifecycle and its own resume state, and entangling a
data-prep failure with the training launch makes both harder to diagnose.

### 7.3 Dataset revision pinning

`prepare_data.py` already resolves `info.sha` into the manifest but never passes `revision=` to
`hf_hub_download`. It will. If a source repo updates mid-run, the resume state's `file_idx` would
otherwise point into a differently-sorted file list — a real hazard for a multi-hour unattended
job.

## 8. Documentation

- **`docs/runbook.md`** (new): every command in order, how to stop each, what normal log output
  looks like, which lines are alarming, and what to do when they appear. Covers the STOP/SIGUSR1
  controls, the exit-code table, where to find checkpoints on HF, and the do-not-rsync note.
- **`README.md`**: full rewrite. Currently says "~243M parameters" (now 332M) and advertises the
  identity expert removed in PLAN.md Step 3c.
- **`docs/configuration.md`**: stale defaults (`batch_size: 3`, `target_tokens: 5e9`,
  `lambda_ponder: 3e-3`); missing `loop_ce_subsample`, `loop_count_sampling`, `data_dir`, `phase`,
  and this spec's new keys.
- **`docs/training.md`**: still describes the pre-Step-9 `data_config.json` / parquet dataset.
- **`CLAUDE.md`**: new sections for `modules/runtime/`, the control/exit-code contract, the
  checkpoint lifecycle invariants, and the phase-scoped-offset rule.

## Testing

Plain scripts under `tests/`, matching the existing no-pytest convention:

- `tests/test_checkpoint_lifecycle.py` — atomic write survives a simulated mid-write kill;
  retention keeps N and never deletes un-uploaded or `final` checkpoints; `find_resume_checkpoint`
  skips a corrupt newest file and picks the next, and raises when all are corrupt.
- `tests/test_phase_resume.py` — a phase-1 checkpoint loaded under `--phase phase2` resets
  `global_offset` to 0 and preserves `token_count`; same-phase load preserves both.
- `tests/test_control.py` — STOP sentinel and SIGTERM each set the flag and map to the right exit
  code; `isatty()` false skips `input()`.
- `tests/test_resume_verification.py` — a `run_state.json` ahead of the resumed token count aborts
  with exit 30.
- `tests/test_hf_sync.py` — against a stubbed `HfApi`: retry/backoff, upload failure does not
  raise into the caller, un-uploaded paths stay unmarked, `drain()` completes.

None of these need a GPU or Transformer Engine; they exercise `modules/runtime/` and `utils.py`
only. Existing tests must continue to pass unchanged — `test_checkpoint_roundtrip.py` in
particular pins `save_checkpoint`/`load_checkpoint`'s signature compatibility.

## Flagged, deliberately not fixed

PLAN.md Step 6 describes phase 2 as annealing "LR → ~0", but `build_scheduler`'s cosine floor is
`0.1 * lr`. Changing it alters training dynamics rather than plumbing, so it stays as-is pending
an explicit decision.

## Implementation order

One commit each, in dependency order:

1. Tokenizer constant + `fetch_tokenizer.py` + `get_hf_token()` + env-check fix
2. Checkpoint lifecycle: atomic writes, token cadence, naming, retention, latest-valid, graphs
3. `hf_sync` + `status.json`
4. `control` + preemption + save-on-exhaustion + resume verification
5. Phase orchestration: `--phase`, phase-scoped offset, per-phase targets, `run_training.py`
6. `setup.sh` + `onstart.sh` + `vast_init` removal + revision pinning
7. Docs: `runbook.md`, README rewrite, `configuration.md` / `training.md` / `CLAUDE.md` refresh

Steps 1–4 are the ones where an error costs money rather than an hour.

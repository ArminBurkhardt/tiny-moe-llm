# Unattended Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `tiny-moe-llm` pretraining survive a 40-hour unattended run on a rented,
interruptible vast.ai box — atomic checkpoints with retention and HF upload, preemption-safe
stopping, verified resume, and automatic phase 1 → phase 2 orchestration.

**Architecture:** A new `modules/runtime/` package holds four small single-purpose modules
(checkpoint lifecycle, HF upload thread, stop control, status reporting) that are unit-testable
without a GPU. `scripts/pretrain.py` imports them and stays a training loop. A new
`scripts/run_training.py` supervises the two phases as subprocesses, restarting on preemption.

**Tech Stack:** Python 3, PyTorch, `huggingface_hub`, `accelerate`, Transformer Engine (already a
hard dependency of `modules/model/`, but *not* of the new `modules/runtime/` code).

**Spec:** [docs/superpowers/specs/2026-08-04-unattended-training-run-design.md](../specs/2026-08-04-unattended-training-run-design.md)

## Global Constraints

- **Commit messages are a single line**, `type: short description`. No body, no bullet list, no
  `Co-Authored-By` trailer, no plan/step numbers, no hyphenated compound modifiers in the subject
  ("per loop", not "per-loop").
- **Every script is launched from the repo root.** `config.py` opens `"config.yaml"` with a
  relative path.
- **`modules/runtime/` must not import `transformer_engine`, `torch.nn`, or anything under
  `modules/model/`.** Its tests must run on a machine with no GPU. `import torch` is allowed only
  where genuinely needed (it is not, in any of these four modules).
- **`modules/runtime/__init__.py` is empty.** Imports are always fully qualified
  (`from modules.runtime.checkpoints import ...`), matching `modules/model/` and `modules/data/`.
- **Comments are lowercase, explanatory, and justify *why*.** Match the density of the surrounding
  code; do not strip existing comments.
- **Google-style docstrings with an `Args:` block** on public functions in `modules/`.
- **Tests are plain scripts, not pytest.** Each `sys.path.insert`s the repo root, asserts, prints
  `[ok] ...` lines, and ends with a `PASSED` line. Run with `python tests/test_x.py` from the repo
  root.
- **`.gitignore` swallows `*.json`** — any new JSON file (`status.json`, `run_state.json`) is
  untracked by design and lives under the already-ignored `ckpts/`.
- **Do not change training dynamics.** The cosine LR floor stays `0.1 * lr` (flagged in the spec,
  deliberately unfixed).
- Exact repo ids, verified live: tokenizer `ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536` (public,
  model repo, contains `tokenizer.json`, `tokenizer_config.json`, `config.json`, `id_remap.json`,
  `chat_template.jinja`); uploads `ikeafisch4/temp-train` (public, model repo, currently empty).
- Exit-code contract, used by Tasks 6, 7 and 9 and by the supervisor: `0` phase complete,
  `10` user stop, `20` preempted, `30` resume verification failed.

---

### Task 1: Tokenizer constant, HF token resolver, fetch script

**Files:**
- Modify: `utils.py` (append constants + `get_hf_token`)
- Create: `scripts/fetch_tokenizer.py`
- Modify: `scripts/pretrain.py:416`
- Modify: `scripts/inference.py:104`
- Modify: `scripts/eval_calibration.py:223`
- Modify: `scripts/prepare_data.py:55`
- Modify: `tests/run_env_check.sh:22`
- Test: `tests/test_hf_token.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `utils.TOKENIZER_REPO: str`, `utils.TOKENIZER_DIR: str`,
  `utils.HF_UPLOAD_REPO: str`, `utils.get_hf_token() -> str | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hf_token.py`:

```python
"""get_hf_token resolution order and TOKENIZER_DIR env override. No GPU, no TE."""
import os, sys, tempfile, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils

# 1. $HF_TOKEN wins over everything else
os.environ["HF_TOKEN"] = "  env_token  "
assert utils.get_hf_token() == "env_token", "HF_TOKEN must win and be stripped"
print("[ok] $HF_TOKEN takes precedence and is stripped")

# 2. with HF_TOKEN unset, the repo-root huggingface.key is read
del os.environ["HF_TOKEN"]
key_path = os.path.join(utils.BASE_DIR, "huggingface.key")
backup = None
if os.path.isfile(key_path):
    with open(key_path) as f:
        backup = f.read()
try:
    with open(key_path, "w") as f:
        f.write("file_token\n")
    assert utils.get_hf_token() == "file_token", "huggingface.key must be the second source"
    print("[ok] huggingface.key is read when HF_TOKEN is unset")

    # 3. an empty key file must not shadow the cache/None fallback
    with open(key_path, "w") as f:
        f.write("\n")
    assert utils.get_hf_token() != "", "an empty huggingface.key must not resolve to empty string"
    print("[ok] empty huggingface.key falls through instead of returning ''")
finally:
    if backup is None:
        os.remove(key_path)
    else:
        with open(key_path, "w") as f:
            f.write(backup)

# 4. TOKENIZER_DIR honours the env override at import time
os.environ["TINY_LLM_TOKENIZER"] = os.path.join(tempfile.mkdtemp(), "tok")
importlib.reload(utils)
assert utils.TOKENIZER_DIR == os.environ["TINY_LLM_TOKENIZER"], utils.TOKENIZER_DIR
print("[ok] TINY_LLM_TOKENIZER overrides TOKENIZER_DIR")

del os.environ["TINY_LLM_TOKENIZER"]
importlib.reload(utils)
assert utils.TOKENIZER_DIR.endswith("DeepSeek-V4-Pro-tokenizer-65536"), utils.TOKENIZER_DIR
assert utils.TOKENIZER_REPO == "ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536"
assert utils.HF_UPLOAD_REPO == "ikeafisch4/temp-train"
print("[ok] defaults point at the pruned tokenizer and the right repos")

print("\nHF TOKEN / TOKENIZER CONSTANT CHECKS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_hf_token.py`
Expected: FAIL with `AttributeError: module 'utils' has no attribute 'get_hf_token'`

- [ ] **Step 3: Add the constants and resolver to `utils.py`**

Insert after the `FP32/FP16/BF16` aliases in `utils.py`:

```python
# one place for the tokenizer location. every script used to hardcode this path independently,
# which meant a fresh clone on the rented box (ckpts/ is gitignored) failed in four different
# places with four different messages. TINY_LLM_TOKENIZER overrides it for one-off runs.
TOKENIZER_REPO = "ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536"
TOKENIZER_DIR = os.environ.get(
    "TINY_LLM_TOKENIZER",
    os.path.join(BASE_DIR, "ckpts", "pretrained", "DeepSeek-V4-Pro-tokenizer-65536"),
)
# checkpoints/logs/graphs are pushed here so a reclaimed instance doesn't take the run with it
HF_UPLOAD_REPO = "ikeafisch4/temp-train"


def get_hf_token():
    """Resolve the Hugging Face token from the one place it is allowed to live.

    Order: ``$HF_TOKEN`` -> ``<repo root>/huggingface.key`` (gitignored via ``*.key``, and already
    the local convention) -> the ``huggingface_hub`` login cache. Returns ``None`` rather than
    raising, so read-only callers that don't need a token (the tokenizer repo is public) work
    without one; upload callers raise their own error naming ``scripts/setup.sh --hf-token``.

    Returns:
        The token string, or None if no source provided one.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    key_path = os.path.join(BASE_DIR, "huggingface.key")
    if os.path.isfile(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            token = f.read().strip()
        if token:
            return token

    # last resort: whatever `huggingface-cli login` left behind. wrapped because huggingface_hub
    # is not a hard dependency of every entry point that imports utils.
    try:
        from huggingface_hub import get_token as _cached_token
        token = (_cached_token() or "").strip()
    except Exception:
        token = ""
    return token or None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_hf_token.py`
Expected: PASS, ending in `HF TOKEN / TOKENIZER CONSTANT CHECKS PASSED`

- [ ] **Step 5: Write `scripts/fetch_tokenizer.py`**

```python
"""Download the pruned 65536-token tokenizer from the Hub into ckpts/pretrained/.

ckpts/ is gitignored, so a fresh clone on a rented box has no tokenizer at all and every entry
point dies on line one. The repo is public, so no token is required -- get_hf_token() is passed
only so a private mirror would also work.
"""
import argparse
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from huggingface_hub import snapshot_download

from utils import TOKENIZER_DIR, TOKENIZER_REPO, get_hf_token, logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=TOKENIZER_REPO)
    parser.add_argument("--dest", default=TOKENIZER_DIR)
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()

    marker = os.path.join(args.dest, "tokenizer.json")
    if os.path.isfile(marker) and not args.force:
        logger.info(f"tokenizer already present at {args.dest} (use --force to re-download)")
        return 0

    os.makedirs(args.dest, exist_ok=True)
    logger.info(f"downloading {args.repo} -> {args.dest}")
    snapshot_download(repo_id=args.repo, local_dir=args.dest, token=get_hf_token())

    if not os.path.isfile(marker):
        raise RuntimeError(f"{args.repo} downloaded but {marker} is missing -- wrong repo?")
    logger.info(f"tokenizer ready at {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Replace the four hardcoded paths**

In `scripts/pretrain.py`, change the import line to add `TOKENIZER_DIR`:

```python
from utils import save_checkpoint, load_checkpoint, BASE_DIR, logger, BF16, TOKENIZER_DIR
```

and replace lines 416-419:

```python
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    logger.info(f"Tokenizer loaded from {TOKENIZER_DIR} with vocab size {tokenizer.vocab_size}")
```

In `scripts/inference.py:104`, replace the `default=` expression with `TOKENIZER_DIR` (adding it
to the existing `from utils import ...` line).

In `scripts/eval_calibration.py:223`, same substitution.

In `scripts/prepare_data.py:55`, replace the whole line with:

```python
DEFAULT_TOKENIZER_DIR = TOKENIZER_DIR
```

adding `TOKENIZER_DIR` to its existing `from utils import ...` line.

- [ ] **Step 7: Fix `tests/run_env_check.sh`**

Replace line 22's `AutoTokenizer.from_pretrained("ckpts/pretrained/DeepSeek-V4-Pro-tokenizer")`
with a read of the shared constant, so the check cannot drift from the trainer again:

```python
import sys; sys.path.insert(0, ".")
from utils import TOKENIZER_DIR
tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
```

- [ ] **Step 8: Verify the substitutions caught everything**

Run: `grep -rn "DeepSeek-V4-Pro-tokenizer" scripts/ tests/ utils.py`
Expected: exactly two hits, both in `utils.py` (`TOKENIZER_REPO`, `TOKENIZER_DIR`), plus
`scripts/prune_vocab.py:35`'s `SRC_TOKENIZER_DIR`, which points at the *unpruned* source tokenizer
and must stay as-is — it is the prune script's input, not the trained model's tokenizer.

- [ ] **Step 9: Commit**

```bash
git add utils.py scripts/fetch_tokenizer.py scripts/pretrain.py scripts/inference.py scripts/eval_calibration.py scripts/prepare_data.py tests/run_env_check.sh tests/test_hf_token.py
git commit -m "feat: single tokenizer constant, hf token resolver and fetch script"
```

---

### Task 2: Atomic checkpoint writes and the phase field

**Files:**
- Modify: `utils.py` (`save_checkpoint`, `load_checkpoint`)
- Modify: `scripts/pretrain.py` (the `load_checkpoint` call site, ~line 477)
- Modify: `tests/test_checkpoint_roundtrip.py` (the `load_checkpoint` call site, ~line 35)
- Test: `tests/test_checkpoint_atomic.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `save_checkpoint(model, optimizer, scheduler, epoch, dataset_idx, path, token_count=0,
  global_offset=0, losses=None, phase=None)` writing atomically;
  `load_checkpoint(model, optimizer, scheduler, path) -> (epoch, dataset_idx, token_count,
  global_offset, losses, phase)` — **a 6-tuple now**, phase last, `None` for legacy checkpoints.

- [ ] **Step 1: Write the failing test**

Create `tests/test_checkpoint_atomic.py`:

```python
"""save_checkpoint must be atomic and must round-trip `phase`. No GPU, no TE -- uses tiny
nn.Module stand-ins rather than TinyMoETransformer, so this runs on the dev box and in CI-less
environments alike. test_checkpoint_roundtrip.py covers the real model.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn, optim

from utils import save_checkpoint, load_checkpoint

d = tempfile.mkdtemp()
m = nn.Linear(4, 4)
opt = optim.AdamW(m.parameters(), lr=1e-3)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

path = os.path.join(d, "checkpoint_phase1_tok100M_loss1.2345.pt")
save_checkpoint(m, opt, sched, epoch=1, dataset_idx=7, path=path,
                token_count=100_000_000, global_offset=4242, losses=[3.0, 2.0], phase="phase1")

# no .tmp left behind
assert not os.path.exists(path + ".tmp"), "the temp file must be renamed away, not left on disk"
assert os.path.isfile(path)
print("[ok] save leaves no .tmp behind")

m2 = nn.Linear(4, 4)
opt2 = optim.AdamW(m2.parameters(), lr=1e-3)
epoch, idx, tokens, offset, losses, phase = load_checkpoint(m2, opt2, sched, path)
assert (epoch, idx, tokens, offset, phase) == (1, 7, 100_000_000, 4242, "phase1"), \
    (epoch, idx, tokens, offset, phase)
assert losses == [3.0, 2.0]
print("[ok] phase and global_offset round-trip")

# legacy checkpoint (no phase key) loads with phase=None rather than raising
legacy = os.path.join(d, "checkpoint_epoch0_idx1_loss9.9999.pt")
torch.save({
    "model_state_dict": m.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "scheduler_state_dict": sched.state_dict(),
    "dataset_idx": 3, "epoch": 0,
}, legacy)
epoch, idx, tokens, offset, losses, phase = load_checkpoint(m2, opt2, sched, legacy)
assert phase is None and tokens == 0 and offset == 0, (phase, tokens, offset)
print("[ok] legacy checkpoints still load, phase is None")

# a truncated .tmp is never mistaken for a real checkpoint
stray = os.path.join(d, "checkpoint_phase1_tok200M_loss1.0000.pt.tmp")
with open(stray, "wb") as f:
    f.write(b"\x00" * 32)
assert not stray.endswith(".pt"), "a partial write must not end in .pt"
print("[ok] partial writes cannot masquerade as .pt files")

print("\nATOMIC CHECKPOINT CHECKS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_checkpoint_atomic.py`
Expected: FAIL with `TypeError: save_checkpoint() got an unexpected keyword argument 'phase'`

- [ ] **Step 3: Make `save_checkpoint` atomic and phase-aware**

Replace `utils.py`'s `save_checkpoint` body:

```python
def save_checkpoint(model, optimizer, scheduler, epoch, dataset_idx, path, token_count=0,
                    global_offset=0, losses=None, phase=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "dataset_idx": dataset_idx,
        "epoch": epoch,
        "token_count": token_count,
        # single resume point into the flat, unshuffled document stream (PLAN.md Step 9):
        # doc sharding across workers is pure doc_idx % num_workers arithmetic, so one
        # conservative (min-across-workers) scalar is enough -- no per-worker/file bookkeeping
        "global_offset": global_offset,
        # which {phase}.bin/.idx corpus that offset indexes into. without this, resuming a
        # phase-1 checkpoint under phase 2 feeds a ~23M doc offset into a ~4M doc corpus and the
        # dataloader silently yields zero batches
        "phase": phase,
        "losses": losses
    }
    # write-then-rename: a preemption mid-write must not be able to leave a truncated .pt that is
    # also the newest file by mtime. os.replace is atomic on POSIX and on Windows.
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    logger.info(f"Checkpoint saved at {path}")
```

- [ ] **Step 4: Return `phase` from `load_checkpoint`**

Append to `load_checkpoint`, before the `return`:

```python
    # legacy (pre phase scoping) checkpoints have no phase -- the caller treats None as "same
    # phase as the one being launched", matching the old behaviour
    phase = checkpoint.get("phase", None)
```

and change the return to:

```python
    return epoch, dataset_idx, token_count, global_offset, losses, phase
```

- [ ] **Step 5: Update the two call sites**

`scripts/pretrain.py` ~line 477:

```python
            start_epoch, dataset_idx, resume_token_count, start_doc_idx, losses, ckpt_phase = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
```

`tests/test_checkpoint_roundtrip.py` ~line 35:

```python
epoch, idx, tokens, global_offset, _, _ = load_checkpoint(m2, opt2, scheduler, path)
```

`ckpt_phase` is unused until Task 6; that is fine and intentional.

- [ ] **Step 6: Run both tests to verify they pass**

Run: `python tests/test_checkpoint_atomic.py`
Expected: PASS, ending in `ATOMIC CHECKPOINT CHECKS PASSED`

Run (GPU box only): `bash tests/run_tests.sh tests/test_checkpoint_roundtrip.py`
Expected: PASS, ending in `CHECKPOINT ROUNDTRIP CHECKS PASSED`

- [ ] **Step 7: Commit**

```bash
git add utils.py scripts/pretrain.py tests/test_checkpoint_roundtrip.py tests/test_checkpoint_atomic.py
git commit -m "feat: atomic checkpoint writes and a phase field"
```

---

### Task 3: Checkpoint lifecycle module

**Files:**
- Create: `modules/runtime/__init__.py` (empty)
- Create: `modules/runtime/checkpoints.py`
- Test: `tests/test_checkpoint_lifecycle.py`

The spec's testing section lists `test_phase_resume.py` and `test_resume_verification.py` as
separate files; they are folded into `test_checkpoint_lifecycle.py` here because all three exercise
the same module and the repo's tests are one-file-per-module, not one-file-per-behaviour.

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `rolling_name(phase: str, token_count: int, loss: float) -> str`
  - `final_name(phase: str) -> str`
  - `is_final(filename: str) -> bool`
  - `cleanup_stale_files(checkpoint_dir: str) -> int`
  - `find_resume_checkpoint(checkpoint_dir: str, try_load) -> tuple[str, object] | None`
  - `prune_checkpoints(checkpoint_dir: str, keep: int, is_uploaded) -> list[str]`
  - `resolve_resume_scope(ckpt_phase, phase, start_epoch, dataset_idx, start_doc_idx) ->
    tuple[int, int, int]`
  - `write_run_state(path: str, phase: str, token_count: int, checkpoint: str) -> None`
  - `read_run_state(path: str) -> dict | None`
  - `verify_resume(state: dict | None, phase: str, resumed_tokens: int, slack_tokens: int) -> None`
  - `class ResumeVerificationError(RuntimeError)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_checkpoint_lifecycle.py`:

```python
"""Checkpoint naming, stale-file cleanup, latest-VALID resume selection, retention, and resume
verification. No GPU, no TE, no torch -- pure file bookkeeping.
"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.runtime.checkpoints import (
    ResumeVerificationError, cleanup_stale_files, final_name, find_resume_checkpoint, is_final,
    prune_checkpoints, read_run_state, resolve_resume_scope, rolling_name, verify_resume,
    write_run_state,
)


def touch(d, name, age=0.0):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write("x")
    if age:
        t = time.time() - age
        os.utime(p, (t, t))
    return p


# --- naming ---------------------------------------------------------------
assert rolling_name("phase1", 25_415_000_000, 2.91337) == "checkpoint_phase1_tok25415M_loss2.9134.pt"
assert final_name("phase2") == "checkpoint_phase2_final.pt"
assert is_final("checkpoint_phase2_final.pt")
assert not is_final("checkpoint_phase1_tok100M_loss1.0000.pt")
print("[ok] naming is token keyed and final checkpoints are distinguishable")

# --- stale cleanup --------------------------------------------------------
d = tempfile.mkdtemp()
touch(d, "checkpoint_phase1_tok1M_loss1.0000.pt.tmp")
touch(d, "loss_graph_epoch0_step256.png")
touch(d, "expert_selection_epoch0_step512.png")
keep_me = touch(d, "checkpoint_phase1_tok1M_loss1.0000.pt")
removed = cleanup_stale_files(d)
assert removed == 3, removed
assert os.path.isfile(keep_me)
print(f"[ok] cleanup_stale_files removed {removed} stale artifacts, left real checkpoints alone")

# --- latest VALID selection ----------------------------------------------
d = tempfile.mkdtemp()
old = touch(d, "checkpoint_phase1_tok100M_loss3.0000.pt", age=100)
new = touch(d, "checkpoint_phase1_tok200M_loss2.0000.pt", age=10)

def try_load(p):
    if p == new:
        raise RuntimeError("simulated corruption")
    return {"path": p}

got = find_resume_checkpoint(d, try_load)
assert got is not None and got[0] == old, got
print("[ok] a corrupt newest checkpoint falls back to the next one")

def always_fail(p):
    raise RuntimeError("simulated corruption")

try:
    find_resume_checkpoint(d, always_fail)
    raise AssertionError("must raise when NO candidate loads -- silently starting from token 0 is the bug")
except RuntimeError as e:
    assert "none of them load" in str(e), str(e)
    assert "simulated corruption" in str(e), "the underlying error must survive into the message"
print("[ok] all candidates corrupt raises instead of restarting from scratch")

assert find_resume_checkpoint(tempfile.mkdtemp(), try_load) is None
print("[ok] an empty directory is a normal cold start, not an error")

# --- phase scoping --------------------------------------------------------
# same phase: everything is carried through untouched
assert resolve_resume_scope("phase1", "phase1", 2, 900, 23_000_000) == (2, 900, 23_000_000)
# legacy checkpoint with no phase recorded is treated as the current phase (old behaviour)
assert resolve_resume_scope(None, "phase1", 2, 900, 23_000_000) == (2, 900, 23_000_000)
# crossing phases: the doc stream resets, because phase 2's corpus is ~4M docs and a ~23M offset
# makes every worker's range() empty -- zero batches, and training "succeeds" having trained nothing
assert resolve_resume_scope("phase1", "phase2", 2, 900, 23_000_000) == (0, 0, 0)
print("[ok] crossing a phase boundary resets the document offset and leaves nothing else to chance")

# --- retention ------------------------------------------------------------
d = tempfile.mkdtemp()
a = touch(d, "checkpoint_phase1_tok100M_loss3.0000.pt", age=400)
b = touch(d, "checkpoint_phase1_tok200M_loss2.5000.pt", age=300)
c = touch(d, "checkpoint_phase1_tok300M_loss2.2000.pt", age=200)
fin = touch(d, "checkpoint_phase1_final.pt", age=500)

deleted = prune_checkpoints(d, keep=2, is_uploaded=lambda p: True)
assert deleted == [a], deleted
assert os.path.isfile(b) and os.path.isfile(c)
assert os.path.isfile(fin), "final checkpoints are never pruned, regardless of the keep window"
print("[ok] retention keeps the newest N and never touches the final checkpoint")

d = tempfile.mkdtemp()
a = touch(d, "checkpoint_phase1_tok100M_loss3.0000.pt", age=400)
touch(d, "checkpoint_phase1_tok200M_loss2.5000.pt", age=300)
touch(d, "checkpoint_phase1_tok300M_loss2.2000.pt", age=200)
deleted = prune_checkpoints(d, keep=2, is_uploaded=lambda p: False)
assert deleted == [], deleted
assert os.path.isfile(a), "an un-uploaded checkpoint must survive -- a full disk is recoverable, a lost checkpoint is not"
print("[ok] retention refuses to delete anything not confirmed uploaded")

# --- run state + resume verification -------------------------------------
d = tempfile.mkdtemp()
sp = os.path.join(d, "run_state.json")
assert read_run_state(sp) is None
write_run_state(sp, phase="phase1", token_count=12_400_000_000, checkpoint="ckpt.pt")
st = read_run_state(sp)
assert st["phase"] == "phase1" and st["token_count"] == 12_400_000_000
print("[ok] run_state round-trips")

slack = 800_000_000  # 2 * checkpoint_every_tokens
verify_resume(st, "phase1", 12_400_000_000, slack)          # exact
verify_resume(st, "phase1", 12_000_000_000, slack)          # inside slack (fell back one ckpt)
verify_resume(st, "phase2", 0, slack)                       # different phase -> no comparison
verify_resume(None, "phase1", 0, slack)                     # cold start
print("[ok] legitimate resumes pass verification")

for bad in (0, 11_000_000_000):
    try:
        verify_resume(st, "phase1", bad, slack)
        raise AssertionError(f"resuming at {bad} against a recorded 12.4B must abort")
    except ResumeVerificationError:
        pass
print("[ok] a resume materially behind the recorded token count aborts")

print("\nCHECKPOINT LIFECYCLE CHECKS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_checkpoint_lifecycle.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.runtime'`

- [ ] **Step 3: Create the package and module**

Create empty `modules/runtime/__init__.py`.

Create `modules/runtime/checkpoints.py`:

```python
"""Checkpoint file bookkeeping for an unattended, interruptible run.

Naming, stale-file cleanup, picking a checkpoint that actually loads, retention, and the
run-state sidecar used to verify that a resume landed where it should have. Deliberately free of
torch/TE imports so it can be tested on a machine with no GPU.
"""
import json
import os

from utils import logger

FINAL_SUFFIX = "_final.pt"
CKPT_PREFIX = "checkpoint_"


class ResumeVerificationError(RuntimeError):
    """Raised when a resumed run is materially behind the last recorded token count."""


def rolling_name(phase: str, token_count: int, loss: float) -> str:
    """Name a periodic checkpoint.

    Args:
        phase: "phase1" or "phase2" -- which corpus ``global_offset`` indexes into.
        token_count: real (non pad) tokens trained so far.
        loss: the loss value at the moment of saving, for eyeballing the directory.

    Returns:
        A filename. epoch/step are deliberately absent: num_epochs is a safety net now and
        dataset_idx stopped being the resume key when global_offset landed, so the token count is
        the only figure that says where a checkpoint sits in the run.
    """
    return f"{CKPT_PREFIX}{phase}_tok{token_count // 1_000_000}M_loss{loss:.4f}.pt"


def final_name(phase: str) -> str:
    """Name the terminal checkpoint of a phase (target reached or data exhausted -- same event
    from a consumer's point of view). Never pruned."""
    return f"{CKPT_PREFIX}{phase}{FINAL_SUFFIX}"


def is_final(filename: str) -> bool:
    """True if this filename is a phase's terminal checkpoint."""
    return os.path.basename(filename).endswith(FINAL_SUFFIX)


def _checkpoint_files(checkpoint_dir: str) -> list:
    """All real checkpoint files, newest mtime first."""
    if not os.path.isdir(checkpoint_dir):
        return []
    names = [
        n for n in os.listdir(checkpoint_dir)
        if n.startswith(CKPT_PREFIX) and n.endswith(".pt")
    ]
    paths = [os.path.join(checkpoint_dir, n) for n in names]
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths


def cleanup_stale_files(checkpoint_dir: str) -> int:
    """Delete artifacts a previous run had no chance to clean up.

    Two kinds: ``*.pt.tmp`` left by a save that was preempted mid-write (harmless, but they eat
    2GB each), and the per-step ``loss_graph_epoch*``/``expert_selection_epoch*`` PNGs the old
    plotting cadence produced by the thousand.

    Args:
        checkpoint_dir: ckpts/training.

    Returns:
        Number of files removed.
    """
    if not os.path.isdir(checkpoint_dir):
        return 0
    removed = 0
    for name in os.listdir(checkpoint_dir):
        stale = (
            name.endswith(".pt.tmp")
            or (name.startswith("loss_graph_epoch") and name.endswith(".png"))
            or (name.startswith("expert_selection_epoch") and name.endswith(".png"))
        )
        if not stale:
            continue
        try:
            os.remove(os.path.join(checkpoint_dir, name))
            removed += 1
        except OSError as e:
            logger.warning(f"could not remove stale file {name}: {e}")
    if removed:
        logger.info(f"cleaned up {removed} stale files in {checkpoint_dir}")
    return removed


def find_resume_checkpoint(checkpoint_dir: str, try_load):
    """Return the newest checkpoint that actually loads.

    "A checkpoint exists but will not load" must never degrade into "start from token 0" -- that
    is how an unattended run silently throws away days of compute behind a plausible looking loss
    curve. So a failure to load the newest file is logged loudly and the next-newest is tried;
    only when *every* candidate fails does this raise.

    Args:
        checkpoint_dir: ckpts/training.
        try_load: callable taking a path and either returning a result or raising. Note it may
            partially mutate the model it loads into before raising; callers resume from the
            returned path's state, which is the last successful load, so that is benign.

    Returns:
        ``(path, try_load's result)``, or None if the directory holds no checkpoints at all.

    Raises:
        RuntimeError: if candidates exist but none of them load.
    """
    candidates = _checkpoint_files(checkpoint_dir)
    if not candidates:
        return None

    last_error = None
    for path in candidates:
        try:
            return path, try_load(path)
        except Exception as e:
            last_error = e
            logger.error(
                f"checkpoint {os.path.basename(path)} failed to load "
                f"({type(e).__name__}: {e}) -- trying the next oldest"
            )
    raise RuntimeError(
        f"found {len(candidates)} checkpoint(s) in {checkpoint_dir} but none of them load; "
        f"refusing to silently restart from scratch. Last error: {last_error}"
    ) from last_error


def prune_checkpoints(checkpoint_dir: str, keep: int, is_uploaded) -> list:
    """Delete rolling checkpoints outside the keep window that are confirmed uploaded.

    Both conditions are required. A checkpoint that was deleted locally and never uploaded is
    gone for good; one that is kept because its upload failed only costs disk, and a full disk is
    a loud, fixable failure. Final checkpoints are exempt entirely.

    Args:
        checkpoint_dir: ckpts/training.
        keep: how many rolling checkpoints to retain locally, newest first.
        is_uploaded: callable taking a path, returning whether HF has it.

    Returns:
        The paths actually deleted.
    """
    rolling = [p for p in _checkpoint_files(checkpoint_dir) if not is_final(p)]
    deleted, held = [], []
    for path in rolling[keep:]:
        if not is_uploaded(path):
            held.append(os.path.basename(path))
            continue
        try:
            os.remove(path)
            deleted.append(path)
        except OSError as e:
            logger.warning(f"could not prune {path}: {e}")
    if held:
        logger.warning(
            f"keeping {len(held)} checkpoint(s) past the retention window because their upload "
            f"has not succeeded: {', '.join(held)} -- check disk space and the uploader log"
        )
    if deleted:
        logger.info(f"pruned {len(deleted)} uploaded checkpoint(s) past the newest {keep}")
    return deleted


def resolve_resume_scope(ckpt_phase, phase: str, start_epoch: int, dataset_idx: int,
                         start_doc_idx: int):
    """Decide what survives a resume when the checkpoint may come from the other phase.

    ``global_offset`` is an index into one specific ``{phase}.bin`` document stream. Feeding phase
    1's offset (~23M docs) into phase 2's corpus (~4M docs) makes every worker's
    ``range(first, num_docs, num_workers)`` empty: the dataloader yields zero batches and the run
    exits looking like a success having trained nothing. So it resets. The token count does NOT
    reset -- that is the caller's business and it must carry over, because the cosine LR is
    anchored to the combined budget and phase 2 continues the decay rather than restarting it.

    Args:
        ckpt_phase: phase recorded in the checkpoint, or None for a legacy checkpoint.
        phase: phase being launched.
        start_epoch: epoch from the checkpoint.
        dataset_idx: step index from the checkpoint.
        start_doc_idx: global_offset from the checkpoint.

    Returns:
        ``(start_epoch, dataset_idx, start_doc_idx)`` to actually resume with.
    """
    if ckpt_phase is None or ckpt_phase == phase:
        return start_epoch, dataset_idx, start_doc_idx
    logger.warning(
        f"checkpoint was written during {ckpt_phase}, now training {phase}: resetting the "
        f"document offset (was {start_doc_idx:,}) and epoch/step; the token count is preserved "
        f"so the LR schedule continues rather than restarting"
    )
    return 0, 0, 0


def write_run_state(path: str, phase: str, token_count: int, checkpoint: str) -> None:
    """Record where the run had got to, for the next process to verify itself against.

    Args:
        path: ckpts/training/run_state.json.
        phase: the phase that was training.
        token_count: real tokens trained at the moment of the checkpoint.
        checkpoint: basename of the checkpoint just written.
    """
    payload = {"phase": phase, "token_count": int(token_count), "checkpoint": os.path.basename(checkpoint)}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_run_state(path: str):
    """Read the run-state sidecar, or None if it is absent or unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"could not read run state {path}: {e} -- treating as absent")
        return None


def verify_resume(state, phase: str, resumed_tokens: int, slack_tokens: int) -> None:
    """Abort if this process resumed materially behind where the last one got to.

    The failure this exists for: a preempted box relaunches, fails to find (or silently discards)
    its checkpoint, and starts again from token 0 with a loss curve that looks entirely plausible
    for another forty GPU-hours. Slack covers the legitimate case of falling back to an older
    checkpoint per find_resume_checkpoint.

    Args:
        state: read_run_state's output, or None on a cold start.
        phase: the phase being launched.
        resumed_tokens: token count restored from the checkpoint (0 if starting fresh).
        slack_tokens: how far behind is acceptable, normally 2 * checkpoint_every_tokens.

    Raises:
        ResumeVerificationError: if the gap exceeds the slack.
    """
    if state is None:
        return
    if state.get("phase") != phase:
        # a phase transition legitimately resets the doc stream and starts this phase's own
        # bookkeeping; there is nothing to compare against
        return
    recorded = int(state.get("token_count", 0))
    if recorded <= 0:
        return
    if resumed_tokens >= recorded - slack_tokens:
        return
    raise ResumeVerificationError(
        f"resume verification failed for {phase}: run_state.json records {recorded:,} tokens but "
        f"this process resumed at {resumed_tokens:,} (slack {slack_tokens:,}). Refusing to retrain "
        f"ground already covered -- inspect ckpts/training before restarting."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_checkpoint_lifecycle.py`
Expected: PASS, ending in `CHECKPOINT LIFECYCLE CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add modules/runtime/__init__.py modules/runtime/checkpoints.py tests/test_checkpoint_lifecycle.py
git commit -m "feat: checkpoint lifecycle module with retention and resume verification"
```

---

### Task 4: Background HF uploader and status writer

**Files:**
- Create: `modules/runtime/hf_sync.py`
- Create: `modules/runtime/status.py`
- Test: `tests/test_hf_sync.py`

**Interfaces:**
- Consumes: `utils.get_hf_token`, `utils.HF_UPLOAD_REPO` (Task 1).
- Produces:
  - `class HFSync` with `__init__(repo_id, token=None, api=None, max_queue=8, retries=3,
    backoff=5.0)`, `upload(local_path, repo_path, droppable=False)`, `is_uploaded(local_path) ->
    bool`, `drain(timeout=600.0) -> bool`, `close()`, and a `.enabled` bool.
  - `status.write_status(path, **fields) -> None`
  - `status.eta_seconds(tokens_done, tokens_target, tokens_per_sec) -> float | None`
  - `status.format_duration(seconds) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hf_sync.py`:

```python
"""HFSync retry/backoff, non-fatal failure, upload marking, drop policy and drain. Uses a stub
api object -- no network, no GPU.
"""
import os, sys, tempfile, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.runtime.hf_sync import HFSync
from modules.runtime.status import eta_seconds, format_duration, write_status


class StubApi:
    """Records calls; fails the first `fail_times` attempts for any path in `flaky`."""

    def __init__(self, flaky=(), fail_times=0, always_fail=()):
        self.calls = []
        self.flaky = set(flaky)
        self.always_fail = set(always_fail)
        self.fail_times = fail_times
        self.attempts = {}
        self.lock = threading.Lock()

    def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, token=None, **kw):
        with self.lock:
            self.attempts[path_in_repo] = self.attempts.get(path_in_repo, 0) + 1
            n = self.attempts[path_in_repo]
            self.calls.append(path_in_repo)
        if path_in_repo in self.always_fail:
            raise RuntimeError("permanent failure")
        if path_in_repo in self.flaky and n <= self.fail_times:
            raise RuntimeError("transient failure")
        return "ok"


d = tempfile.mkdtemp()

def mkfile(name, body="x"):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(body)
    return p

# --- happy path -----------------------------------------------------------
api = StubApi()
sync = HFSync("owner/repo", token="t", api=api)
p = mkfile("a.pt")
sync.upload(p, "checkpoints/a.pt")
assert sync.drain(timeout=10), "drain must complete"
assert api.calls == ["checkpoints/a.pt"], api.calls
assert sync.is_uploaded(p)
sync.close()
print("[ok] a queued file uploads once and is marked uploaded")

# --- retry then succeed ---------------------------------------------------
api = StubApi(flaky={"checkpoints/b.pt"}, fail_times=2)
sync = HFSync("owner/repo", token="t", api=api, backoff=0.01)
p = mkfile("b.pt")
sync.upload(p, "checkpoints/b.pt")
assert sync.drain(timeout=10)
assert api.attempts["checkpoints/b.pt"] == 3, api.attempts
assert sync.is_uploaded(p)
sync.close()
print("[ok] transient failures are retried and eventually succeed")

# --- permanent failure is non fatal and leaves the file unmarked ----------
api = StubApi(always_fail={"checkpoints/c.pt"})
sync = HFSync("owner/repo", token="t", api=api, backoff=0.01)
p = mkfile("c.pt")
sync.upload(p, "checkpoints/c.pt")   # must not raise
assert sync.drain(timeout=10)
assert not sync.is_uploaded(p), "a failed upload must NOT be marked uploaded -- retention reads this"
sync.close()
print("[ok] a permanently failing upload never raises into the caller and stays unmarked")

# --- disabled sync is a no-op that reports nothing uploaded ---------------
sync = HFSync("", token=None, api=StubApi())
assert not sync.enabled
p = mkfile("d.pt")
sync.upload(p, "checkpoints/d.pt")
assert not sync.is_uploaded(p)
assert sync.drain(timeout=1)
sync.close()
print("[ok] an empty repo id disables uploads without breaking callers")

# --- status helpers -------------------------------------------------------
assert eta_seconds(0, 100, 0) is None
assert eta_seconds(50, 100, 10) == 5.0
assert eta_seconds(150, 100, 10) == 0.0
assert format_duration(0) == "0m"
assert format_duration(3661) == "1h 1m"
assert format_duration(90000) == "25h 0m"
sp = os.path.join(d, "status.json")
write_status(sp, phase="phase1", tokens=1234)
import json
with open(sp) as f:
    assert json.load(f)["tokens"] == 1234
assert not os.path.exists(sp + ".tmp")
print("[ok] status helpers compute ETA and write atomically")

print("\nHF SYNC CHECKS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_hf_sync.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.runtime.hf_sync'`

- [ ] **Step 3: Write `modules/runtime/status.py`**

```python
"""Run status reporting: atomic JSON writes and ETA arithmetic.

Kept separate from hf_sync so the trainer can write status without an uploader, and so the
supervisor can read it without importing upload machinery.
"""
import json
import os


def write_status(path: str, **fields) -> None:
    """Write a status snapshot atomically.

    Something is always reading this file -- you over ssh, the supervisor, or the uploader about
    to push it -- so a half-written file is a real hazard, hence write-then-rename.

    Args:
        path: ckpts/training/status.json.
        **fields: arbitrary JSON-serialisable status fields.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def eta_seconds(tokens_done: int, tokens_target: int, tokens_per_sec: float):
    """Seconds remaining at the current rate.

    Args:
        tokens_done: real tokens trained so far.
        tokens_target: the token count this phase or run stops at.
        tokens_per_sec: recent throughput.

    Returns:
        Seconds remaining, 0.0 if already past target, or None if the rate is unknown/zero.
    """
    if not tokens_per_sec or tokens_per_sec <= 0:
        return None
    remaining = tokens_target - tokens_done
    if remaining <= 0:
        return 0.0
    return remaining / tokens_per_sec


def format_duration(seconds) -> str:
    """Render a duration as "Nh Mm" (or "Mm" under an hour). "unknown" for None."""
    if seconds is None:
        return "unknown"
    seconds = max(int(seconds), 0)
    hours, minutes = divmod(seconds // 60, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"
```

- [ ] **Step 4: Write `modules/runtime/hf_sync.py`**

```python
"""Background uploader for checkpoints, logs, graphs and status.

A 2GB checkpoint at 200 Mbps takes ~80s. Doing that inline every 30 minutes costs ~4% of the run
for nothing, so uploads go to one worker thread and the training loop never waits on the network.
Upload failures are logged and dropped, never raised: losing an upload is survivable, crashing a
40 hour training run over a transient 503 is not.
"""
import os
import threading
import time
from collections import deque

from utils import logger


class _Job:
    __slots__ = ("local_path", "repo_path", "droppable")

    def __init__(self, local_path, repo_path, droppable):
        self.local_path = local_path
        self.repo_path = repo_path
        self.droppable = droppable


class HFSync:
    """Queue-backed uploader to a Hugging Face repo.

    Args:
        repo_id: e.g. "ikeafisch4/temp-train". An empty string disables uploading entirely, so a
            local run needs no special casing at the call sites.
        token: HF token with write access to repo_id.
        api: an object exposing ``upload_file(path_or_fileobj, path_in_repo, repo_id, token)``.
            Defaults to a real ``huggingface_hub.HfApi``; injectable for tests.
        max_queue: soft cap on pending jobs before droppable ones start being discarded.
        retries: attempts per file before giving up.
        backoff: seconds before the first retry, tripled each attempt.
    """

    def __init__(self, repo_id, token=None, api=None, max_queue=8, retries=3, backoff=5.0):
        self.repo_id = repo_id or ""
        self.enabled = bool(self.repo_id)
        self._token = token
        self._retries = retries
        self._backoff = backoff
        self._max_queue = max_queue

        self._queue = deque()
        self._cv = threading.Condition()
        self._uploaded = set()
        self._closing = False
        self._thread = None
        # true while the worker holds a job it has taken off the queue -- drain() must wait for
        # that job too, not just for the queue to empty
        self._busy = False

        if not self.enabled:
            logger.warning("HF upload disabled (no repo id) -- checkpoints stay local only")
            return

        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self._api = api

        self._thread = threading.Thread(target=self._run, name="hf-sync", daemon=True)
        self._thread.start()
        logger.info(f"HF upload thread started -> {self.repo_id}")

    def upload(self, local_path: str, repo_path: str, droppable: bool = False) -> None:
        """Queue a file for upload and return immediately.

        Args:
            local_path: file on disk. Read by the worker thread, so it must not be deleted before
                the upload completes -- retention only deletes files is_uploaded() confirms.
            repo_path: destination path inside the repo.
            droppable: rolling checkpoints set this. If the queue backs up, the oldest droppable
                job is discarded rather than blocking training. It stays unmarked, so retention
                will refuse to delete it and the disk fills instead -- loud and recoverable.
        """
        if not self.enabled:
            return
        with self._cv:
            if len(self._queue) >= self._max_queue:
                for i, job in enumerate(self._queue):
                    if job.droppable:
                        del self._queue[i]
                        logger.warning(
                            f"upload queue full ({self._max_queue}); dropping {job.repo_path}. "
                            f"It stays on disk and un-pruned -- uploads are falling behind."
                        )
                        break
            self._queue.append(_Job(local_path, repo_path, droppable))
            self._cv.notify()

    def is_uploaded(self, local_path: str) -> bool:
        """Whether this exact path completed an upload in this process."""
        with self._cv:
            return os.path.abspath(local_path) in self._uploaded

    def drain(self, timeout: float = 600.0) -> bool:
        """Block until the queue empties or timeout elapses.

        Called before exiting and at phase boundaries so a stop does not race the uploader.

        Returns:
            True if the queue drained, False on timeout.
        """
        if not self.enabled:
            return True
        deadline = time.time() + timeout
        with self._cv:
            while self._queue or self._busy:
                remaining = deadline - time.time()
                if remaining <= 0:
                    logger.warning(f"upload drain timed out with {len(self._queue)} job(s) pending")
                    return False
                self._cv.wait(timeout=min(remaining, 1.0))
        return True

    def close(self) -> None:
        """Stop the worker thread. Does not wait for pending jobs -- call drain() first."""
        if not self.enabled:
            return
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    # --- worker ----------------------------------------------------------
    def _run(self):
        while True:
            with self._cv:
                while not self._queue and not self._closing:
                    self._cv.wait(timeout=1.0)
                if self._closing and not self._queue:
                    return
                job = self._queue.popleft()
                self._busy = True
            try:
                self._upload_with_retries(job)
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def _upload_with_retries(self, job):
        delay = self._backoff
        for attempt in range(1, self._retries + 1):
            try:
                self._api.upload_file(
                    path_or_fileobj=job.local_path,
                    path_in_repo=job.repo_path,
                    repo_id=self.repo_id,
                    token=self._token,
                )
                with self._cv:
                    self._uploaded.add(os.path.abspath(job.local_path))
                logger.info(f"uploaded {job.repo_path}")
                return
            except Exception as e:
                if attempt == self._retries:
                    # deliberately swallowed: a failed upload must not take the training run with
                    # it. The file stays unmarked, so retention will not delete it either.
                    logger.error(
                        f"upload of {job.repo_path} failed after {self._retries} attempts "
                        f"({type(e).__name__}: {e}); giving up on this file"
                    )
                    return
                logger.warning(
                    f"upload of {job.repo_path} attempt {attempt}/{self._retries} failed "
                    f"({type(e).__name__}: {e}); retrying in {delay:.0f}s"
                )
                time.sleep(delay)
                delay *= 3
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python tests/test_hf_sync.py`
Expected: PASS, ending in `HF SYNC CHECKS PASSED`

- [ ] **Step 6: Commit**

```bash
git add modules/runtime/hf_sync.py modules/runtime/status.py tests/test_hf_sync.py
git commit -m "feat: background hugging face uploader and status writer"
```

---

### Task 5: Stop control module

**Files:**
- Create: `modules/runtime/control.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - Constants `EXIT_OK = 0`, `EXIT_USER_STOP = 10`, `EXIT_PREEMPTED = 20`,
    `EXIT_RESUME_FAILED = 30`, `STOP_SENTINEL = "STOP"`.
  - `class RunControl` with `__init__(checkpoint_dir)`, `install()`, `poll()`,
    `.stop_requested: bool`, `.exit_code: int`, `.reason: str`,
    `.take_checkpoint_request() -> bool`, `clear_sentinel()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_control.py`:

```python
"""Stop control: sentinel file, signal flags, exit-code mapping. No GPU, no TE."""
import os, signal, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.runtime.control import (
    EXIT_OK, EXIT_PREEMPTED, EXIT_RESUME_FAILED, EXIT_USER_STOP, RunControl, STOP_SENTINEL,
)

assert (EXIT_OK, EXIT_USER_STOP, EXIT_PREEMPTED, EXIT_RESUME_FAILED) == (0, 10, 20, 30)
print("[ok] exit-code contract matches the supervisor's expectations")

# --- sentinel file --------------------------------------------------------
d = tempfile.mkdtemp()
c = RunControl(d)
c.poll()
assert not c.stop_requested
with open(os.path.join(d, STOP_SENTINEL), "w") as f:
    f.write("")
c.poll()
assert c.stop_requested and c.exit_code == EXIT_USER_STOP, (c.stop_requested, c.exit_code)
assert "STOP" in c.reason
print("[ok] the STOP sentinel requests a user stop, not a restart")

# the sentinel is removed so a relaunch does not immediately stop again
c.clear_sentinel()
assert not os.path.exists(os.path.join(d, STOP_SENTINEL))
print("[ok] clear_sentinel removes the file")

# --- SIGTERM -> preempted -------------------------------------------------
d = tempfile.mkdtemp()
c = RunControl(d)
c.install()
os.kill(os.getpid(), signal.SIGTERM)
c.poll()
assert c.stop_requested and c.exit_code == EXIT_PREEMPTED, (c.stop_requested, c.exit_code)
print("[ok] SIGTERM maps to the restartable exit code")

# --- checkpoint request is one shot --------------------------------------
d = tempfile.mkdtemp()
c = RunControl(d)
c.install()
assert not c.take_checkpoint_request()
if hasattr(signal, "SIGUSR1"):
    os.kill(os.getpid(), signal.SIGUSR1)
    assert c.take_checkpoint_request(), "SIGUSR1 must request an immediate checkpoint"
    assert not c.take_checkpoint_request(), "the request must be consumed, not sticky"
    assert not c.stop_requested, "SIGUSR1 checkpoints but does NOT stop"
    print("[ok] SIGUSR1 requests exactly one checkpoint and does not stop the run")
else:
    print("[skip] SIGUSR1 not available on this platform (Windows); handler is guarded")

# --- an existing STOP file must not be honoured before install ------------
d = tempfile.mkdtemp()
with open(os.path.join(d, STOP_SENTINEL), "w") as f:
    f.write("")
c = RunControl(d)
c.clear_sentinel()
c.poll()
assert not c.stop_requested, "a stale STOP from a previous run must not stop the new one"
print("[ok] a stale STOP sentinel is cleared at startup rather than honoured")

print("\nCONTROL CHECKS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_control.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.runtime.control'`

- [ ] **Step 3: Write `modules/runtime/control.py`**

```python
"""Stopping an unattended run cleanly, from a signal or from a file.

The old interrupt path blocked on input() before saving, which on a box with no tty raises
EOFError into a bare except and saves nothing -- and vast preemption sends SIGTERM, which never
raised KeyboardInterrupt in the first place, so that path never even ran.

Handlers here only set flags: no I/O, no allocation, nothing that can deadlock inside a signal
context. The training loop reads the flags at its existing log cadence.
"""
import os
import signal

from utils import logger

STOP_SENTINEL = "STOP"

# exit-code contract shared with scripts/run_training.py
EXIT_OK = 0             # phase finished (target reached or data exhausted)
EXIT_USER_STOP = 10     # you asked for it -- supervisor must NOT restart
EXIT_PREEMPTED = 20     # SIGTERM/preemption -- supervisor restarts
EXIT_RESUME_FAILED = 30 # resume verification failed -- supervisor must NOT restart


class RunControl:
    """Tracks stop and checkpoint requests from signals and the STOP sentinel file.

    Args:
        checkpoint_dir: ckpts/training -- where the STOP sentinel is looked for. Chosen over a
            PID file because it can be created from vast's web console with no ssh session and no
            process lookup.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.stop_requested = False
        self.exit_code = EXIT_OK
        self.reason = ""
        self._checkpoint_requested = False

    @property
    def sentinel_path(self) -> str:
        return os.path.join(self.checkpoint_dir, STOP_SENTINEL)

    def install(self) -> None:
        """Install signal handlers. Guarded per signal because SIGUSR1 does not exist on Windows
        and the dev box is Windows."""
        signal.signal(signal.SIGTERM, self._on_terminate)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, self._on_checkpoint)
        logger.info(
            f"stop control armed: touch {self.sentinel_path} to stop, SIGTERM to checkpoint and "
            f"exit for restart" + (", SIGUSR1 to checkpoint without stopping" if hasattr(signal, "SIGUSR1") else "")
        )

    def clear_sentinel(self) -> None:
        """Remove a leftover STOP file. Called at startup: a sentinel from a previous run must
        not stop the new one before it has trained a single step."""
        try:
            os.remove(self.sentinel_path)
            logger.info("removed a stale STOP sentinel from a previous run")
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"could not remove {self.sentinel_path}: {e}")

    def poll(self) -> None:
        """Check the sentinel file. Called at the log cadence -- a stat every ~3s costs nothing
        and needs no GPU sync."""
        if self.stop_requested:
            return
        if os.path.exists(self.sentinel_path):
            self._request_stop(EXIT_USER_STOP, f"STOP sentinel found at {self.sentinel_path}")

    def take_checkpoint_request(self) -> bool:
        """Consume a pending "checkpoint now" request. One shot: returns True at most once per
        signal, so a single SIGUSR1 does not checkpoint on every subsequent log step."""
        requested = self._checkpoint_requested
        self._checkpoint_requested = False
        return requested

    def request_stop_preempted(self, reason: str) -> None:
        """Mark a restartable stop from a non-signal source (e.g. an unexpected exception)."""
        self._request_stop(EXIT_PREEMPTED, reason)

    def _request_stop(self, exit_code: int, reason: str) -> None:
        self.stop_requested = True
        self.exit_code = exit_code
        self.reason = reason
        logger.warning(f"stop requested ({reason}); will checkpoint and exit {exit_code}")

    def _on_terminate(self, signum, frame):
        # flag only -- logging from a signal handler can deadlock against the logging lock
        self.stop_requested = True
        self.exit_code = EXIT_PREEMPTED
        self.reason = f"signal {signum}"

    def _on_checkpoint(self, signum, frame):
        self._checkpoint_requested = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_control.py`
Expected: PASS, ending in `CONTROL CHECKS PASSED`

Note: the SIGTERM sub-test replaces the default handler and then delivers SIGTERM to the test
process itself. If the handler were not installed, the test process would die — a failure of this
test can look like the test "hanging up" rather than asserting.

- [ ] **Step 5: Commit**

```bash
git add modules/runtime/control.py tests/test_control.py
git commit -m "feat: stop control for sentinel file and signals"
```

---

### Task 6: Wire the runtime modules into `pretrain.py`

**Files:**
- Modify: `config.yaml` (`training:` block)
- Modify: `config.py` (`TrainingConfig`)
- Modify: `scripts/pretrain.py` (imports, `pretrain()` signature, resume block, log block,
  checkpoint block, graph block, interrupt handler, `__main__`)
- Test: `tests/test_phase_targets.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `pretrain(phase: str | None = None) -> int` returning an exit code;
  `TrainingConfig.checkpoint_every_tokens`, `.keep_local_checkpoints`, `.phase1_fraction`,
  `.hf_upload_repo`, `.phase_target_tokens(phase) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_phase_targets.py`:

```python
"""Per phase token targets and the new training config keys. No GPU, no TE."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainingConfig as T

assert T.checkpoint_every_tokens > 0
assert T.keep_local_checkpoints >= 1
assert 0.0 < T.phase1_fraction < 1.0
print(f"[ok] new keys load: every {T.checkpoint_every_tokens:,} tokens, keep {T.keep_local_checkpoints}, phase1 {T.phase1_fraction}")

p1 = T.phase_target_tokens("phase1")
p2 = T.phase_target_tokens("phase2")
assert p1 == int(T.target_tokens * T.phase1_fraction), (p1, T.target_tokens)
assert p2 == T.target_tokens, (p2, T.target_tokens)
assert p1 < p2, "phase 1 must stop before the combined budget so phase 2 has the anneal"
print(f"[ok] phase1 stops at {p1/1e9:.2f}B, phase2 runs to the combined {p2/1e9:.2f}B")

# the LR schedule stays anchored to the COMBINED budget -- phase 2 continues the cosine, it does
# not restart it
tokens_per_step = T.Batch_size * T.Seq_length * T.grad_accumulation_steps
assert T.total_steps == T.target_tokens // tokens_per_step
print(f"[ok] total_steps ({T.total_steps:,}) still derives from the combined target, not the phase target")

try:
    T.phase_target_tokens("phase3")
    raise AssertionError("an unknown phase must raise rather than silently returning a target")
except ValueError:
    pass
print("[ok] an unknown phase name raises")

print("\nPHASE TARGET CHECKS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_phase_targets.py`
Expected: FAIL with `AttributeError: type object 'TrainingConfig' has no attribute 'checkpoint_every_tokens'`

- [ ] **Step 3: Add the config keys**

Append to `config.yaml`'s `training:` block:

```yaml
  checkpoint_every_tokens: 400000000 # ~30 min at 200K tok/s. expressed in TOKENS, not steps: the
                                     # old `step % 1500` counted MICRO steps, so at grad_accum 16
                                     # it fired every ~49M tokens (~4 min, ~608 checkpoints,
                                     # ~1.2TB against a 120GB disk). tokens also make the cadence
                                     # invariant to batch size / grad accum changes.
  keep_local_checkpoints: 2 # rolling checkpoints kept on disk. a checkpoint is only deleted once
                            # it is BOTH outside this window AND confirmed uploaded -- see
                            # modules/runtime/checkpoints.prune_checkpoints for why.
  phase1_fraction: 0.85 # PLAN.md's 85/15 split. phase 1 stops here; phase 2 runs to target_tokens.
  hf_upload_repo: "ikeafisch4/temp-train" # "" disables uploads (local runs)
```

Append to `config.py`'s `TrainingConfig`:

```python
    # checkpoint lifecycle for the unattended run
    checkpoint_every_tokens = int(Config["training"].get("checkpoint_every_tokens", 400_000_000))
    keep_local_checkpoints = int(Config["training"].get("keep_local_checkpoints", 2))
    hf_upload_repo = str(Config["training"].get("hf_upload_repo", ""))

    # phase 1 gets this fraction of target_tokens, phase 2 the rest. target_tokens itself stays
    # the COMBINED budget so total_steps and the cosine LR anchor are unchanged -- phase 2 must
    # continue the decay from where phase 1 left it, not restart it.
    phase1_fraction = float(Config["training"].get("phase1_fraction", 0.85))

    @classmethod
    def phase_target_tokens(cls, phase: str) -> int:
        """Token count at which the given phase stops training."""
        if phase == "phase1":
            return int(cls.target_tokens * cls.phase1_fraction)
        if phase == "phase2":
            return cls.target_tokens
        raise ValueError(f"unknown phase {phase!r}; expected 'phase1' or 'phase2'")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_phase_targets.py`
Expected: PASS, ending in `PHASE TARGET CHECKS PASSED`

- [ ] **Step 5: Add imports and the `--phase` entry point to `pretrain.py`**

Extend the existing `utils` import and add the runtime imports:

```python
from utils import save_checkpoint, load_checkpoint, BASE_DIR, logger, BF16, TOKENIZER_DIR, get_hf_token, HF_UPLOAD_REPO
from modules.runtime import checkpoints as ckpt_lib
from modules.runtime.control import EXIT_OK, EXIT_PREEMPTED, EXIT_RESUME_FAILED, EXIT_USER_STOP, RunControl
from modules.runtime.hf_sync import HFSync
from modules.runtime.status import eta_seconds, format_duration, write_status
```

Replace the `__main__` block at the bottom:

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="pretrain tiny-moe-llm")
    parser.add_argument(
        "--phase", choices=("phase1", "phase2"), default=TrainingConfig.phase,
        help="which {phase}.bin/.idx corpus to train on (default: config.yaml's training.phase)",
    )
    args = parser.parse_args()
    raise SystemExit(pretrain(phase=args.phase))
```

and change the function signature to `def pretrain(phase=None):`, with a first line of:

```python
    phase = phase or TrainingConfig.phase
    phase_target = TrainingConfig.phase_target_tokens(phase)
```

Replace `split=TrainingConfig.phase` in the `Dataset(...)` construction with `split=phase`.

- [ ] **Step 6: Rework the resume block for phase scoping, valid-checkpoint selection and verification**

Replace the whole `latest = get_latest_checkpoint_epoch(...)` / `if latest is None: ... else: ...`
block (currently ~lines 466-490) with:

```python
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_lib.cleanup_stale_files(checkpoint_dir)
    run_state_path = os.path.join(checkpoint_dir, "run_state.json")
    run_state = ckpt_lib.read_run_state(run_state_path)

    def _try_load(path):
        return load_checkpoint(model, optimizer, scheduler, path)

    # "no checkpoint exists" and "a checkpoint exists but would not load" are NOT the same event.
    # find_resume_checkpoint walks newest-first and only raises when NOTHING loads, which keeps
    # that distinction while surviving one corrupt file.
    found = ckpt_lib.find_resume_checkpoint(checkpoint_dir, _try_load)
    resume_token_count = 0
    if found is None:
        logger.warning("No checkpoint found in ckpts/training. Starting training from scratch")
    else:
        checkpoint_path, loaded = found
        start_epoch, dataset_idx, resume_token_count, start_doc_idx, losses, ckpt_phase = loaded

        # a checkpoint from the OTHER phase indexes a different corpus -- see resolve_resume_scope
        # for why that silently trains zero batches if left alone. The token count is deliberately
        # NOT part of what it resets.
        start_epoch, dataset_idx, start_doc_idx = ckpt_lib.resolve_resume_scope(
            ckpt_phase, phase, start_epoch, dataset_idx, start_doc_idx
        )

        model._token_tracker.num_tokens = resume_token_count
        resumed = True

        # the fp32 masters aren't part of model_state_dict (they're optimizer-only shadows built
        # at optimizer-construction time, before this resume) -- reseed them from the just-loaded
        # bf16 weights so the no_decay group keeps moving from where this checkpoint left off,
        # rather than from the pre-resume random init they were cloned from. Adam's exp_avg/
        # exp_avg_sq for the masters already came back correctly via optimizer.load_state_dict
        # above (restored by param-group position, not by identity, so unaffected by this copy).
        with torch.no_grad():
            for bf16_param, master in no_decay_master_pairs:
                master.data.copy_(bf16_param.data.float())

        # total_steps moves with batch size / grad accum, so reanchor the LR schedule by tokens
        tokens_per_step = TrainingConfig.Batch_size * TrainingConfig.Seq_length * TrainingConfig.grad_accumulation_steps
        resumed_sched_step = min(resume_token_count // tokens_per_step, TrainingConfig.total_steps)
        scheduler = build_scheduler(optimizer)
        with warnings.catch_warnings():
            # silence the "step() called before optimizer.step()" warning during fast forward
            warnings.simplefilter("ignore")
            for _ in range(resumed_sched_step):
                scheduler.step()
        logger.info(
            f"Reanchored LR scheduler to step {resumed_sched_step:,}/{TrainingConfig.total_steps:,} "
            f"({resume_token_count / 1e9:.3f}B tokens trained, current LR {scheduler.get_last_lr()[0]:.3e})"
        )

    # the check this whole file exists to make possible: if the last process reached 12.4B tokens
    # and this one thinks it is at 0, something ate the checkpoint and continuing burns 40
    # GPU-hours retraining ground already covered.
    try:
        ckpt_lib.verify_resume(run_state, phase, resume_token_count, 2 * TrainingConfig.checkpoint_every_tokens)
    except ckpt_lib.ResumeVerificationError as e:
        logger.error(str(e))
        return EXIT_RESUME_FAILED
```

Delete the now-unused `get_latest_checkpoint_epoch` and `checkpoint_name` functions
(`scripts/pretrain.py:302-316`).

- [ ] **Step 7: Start the uploader and control, and add a shared save helper**

Immediately after the resume block (before `dry_run`), insert:

```python
    control = RunControl(checkpoint_dir)
    control.clear_sentinel()
    control.install()

    hf = HFSync(TrainingConfig.hf_upload_repo or HF_UPLOAD_REPO, token=get_hf_token())
    log_path = os.path.join(checkpoint_dir, "train.log")
    loss_png = os.path.join(checkpoint_dir, "loss_graph.png")
    experts_png = os.path.join(checkpoint_dir, "expert_selection.png")
    status_path = os.path.join(checkpoint_dir, "status.json")
    manifest_path = os.path.join(BASE_DIR, "manifest.json")
```

Then, **immediately after the nested `snapshot_global_offset()` definition** (~line 572 — it must
come after, since `save_and_sync` calls it), define the single save path used by every caller:

```python
    def save_and_sync(epoch, dataset_idx, loss_value, token_count, final=False):
        """Write a checkpoint, refresh the graphs, upload everything, then prune.

        Order matters: prune runs last and only deletes what hf.is_uploaded confirms, so a
        checkpoint can never be removed before its replacement is safely off-box.
        """
        name = ckpt_lib.final_name(phase) if final else ckpt_lib.rolling_name(phase, token_count, loss_value)
        path = os.path.join(checkpoint_dir, name)
        save_checkpoint(
            unwrapped_model, optimizer, scheduler, epoch, dataset_idx, path=path,
            token_count=token_count, global_offset=snapshot_global_offset(), losses=losses,
            phase=phase,
        )
        ckpt_lib.write_run_state(run_state_path, phase, token_count, name)

        try:
            save_loss_graph(losses, loss_png)
            save_expert_selection_graph(unwrapped_model.moe.expert_tracker.get_stats(), experts_png)
        except Exception as e:
            # plotting must never take the run down
            logger.error(f"Error occurred while saving graphs: {e}")

        repo_dir = "checkpoints/final" if final else "checkpoints"
        hf.upload(path, f"{repo_dir}/{name}", droppable=not final)
        for local, remote in (
            (log_path, "logs/train.log"),
            (loss_png, "graphs/loss_graph.png"),
            (experts_png, "graphs/expert_selection.png"),
            (status_path, "status.json"),
            (manifest_path, "manifest.json"),
        ):
            if os.path.isfile(local):
                hf.upload(local, remote)

        ckpt_lib.prune_checkpoints(checkpoint_dir, TrainingConfig.keep_local_checkpoints, hf.is_uploaded)
```

- [ ] **Step 8: Replace the cadence, stop checks and graph block inside the log interval**

Inside `if step % LOG_INTERVAL == 0:`, after `n_tokens = unwrapped_model._token_tracker.sync()`,
add the ETA fields to the existing `logger.info(...)` f-string by appending:

```python
                    eta_phase = eta_seconds(n_tokens, phase_target, tokens_per_sec)
```

and adding ` | ETA: {format_duration(eta_phase)}` to the end of the log line's format string.

Then replace the `if n_tokens >= TrainingConfig.target_tokens:` block, the
`if step % ...sliding_window_size == 0:` graph block, and the `if (step % 1500 == 0)` checkpoint
block (currently ~lines 730-775) with:

```python
                    write_status(
                        status_path, phase=phase, tokens=n_tokens, phase_target=phase_target,
                        run_target=TrainingConfig.target_tokens, tokens_per_sec=tokens_per_sec,
                        loss=val_loss, eta_phase=format_duration(eta_phase),
                        eta_run=format_duration(eta_seconds(n_tokens, TrainingConfig.target_tokens, tokens_per_sec)),
                        step=step, epoch=epoch,
                    )

                    control.poll()

                    # stop at the PHASE's token budget. phase 1 stops at 85% of target_tokens so
                    # phase 2 still has the anneal; the LR schedule stays anchored to the combined
                    # figure either way.
                    if n_tokens >= phase_target:
                        logger.info(f"Reached {phase} target ({phase_target:,} tokens); saving final checkpoint.")
                        save_and_sync(epoch, dataset_idx, val_loss, n_tokens, final=True)
                        stop_training = True
                        break

                    if control.stop_requested:
                        logger.info(f"Stopping: {control.reason}. Saving checkpoint...")
                        save_and_sync(epoch, dataset_idx, val_loss, n_tokens)
                        exit_code = control.exit_code
                        stop_training = True
                        break

                    # checkpoint cadence in TOKENS, checked here because the counter is already
                    # drained for logging -- no extra host sync. SIGUSR1 forces one immediately.
                    if n_tokens >= next_checkpoint_tokens or control.take_checkpoint_request():
                        save_and_sync(epoch, dataset_idx, val_loss, n_tokens)
                        next_checkpoint_tokens = n_tokens + TrainingConfig.checkpoint_every_tokens
```

Initialise the cadence counter next to `last_token_count` near the top of the loop setup:

```python
    # first periodic checkpoint one interval after wherever this process picked up
    next_checkpoint_tokens = last_token_count + TrainingConfig.checkpoint_every_tokens
    exit_code = EXIT_OK
```

- [ ] **Step 9: Handle data exhaustion and replace the interrupt handler**

After the inner `for local_step, batch in enumerate(dataloader):` loop ends (where
`dataset_idx = 0` currently sits), add:

```python
            # the dataloader ran dry. phase 1's corpus ends at ~25.5B, below the combined
            # target_tokens, so this is the NORMAL end of a phase -- previously it fell out of the
            # loop with no final checkpoint, losing everything since the last periodic save.
            if not stop_training:
                final_tokens = unwrapped_model._token_tracker.sync()
                logger.info(f"{phase} data exhausted at {final_tokens:,} tokens; saving final checkpoint.")
                save_and_sync(epoch, dataset_idx, losses[-1] if losses else float("nan"), final_tokens, final=True)
                stop_training = True

            dataset_idx = 0
            if stop_training:
                break
```

Replace the `except KeyboardInterrupt:` block with:

```python
    except KeyboardInterrupt:
        # a tty gets the old confirm-then-save prompt; an unattended box must never block on
        # stdin -- input() there raises EOFError, which the old bare except swallowed, saving
        # nothing.
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                input("Training interrupted. Press Enter to save checkpoint and exit...")
            except EOFError:
                pass
        logger.info("Training interrupted. Saving checkpoint...")
        exit_code = EXIT_USER_STOP
        try:
            save_and_sync(epoch, dataset_idx, losses[-1] if losses else float("nan"),
                          unwrapped_model._token_tracker.sync())
        except Exception as e:
            logger.error(f"Failed to save the interrupt checkpoint: {e}")
    finally:
        # never let a stop race the uploader: drain before the process goes away
        hf.drain(timeout=900)
        hf.close()

    return exit_code
```

- [ ] **Step 10: Verify nothing references the deleted helpers**

Run: `grep -n "get_latest_checkpoint_epoch\|checkpoint_name(" scripts/ tests/ -r`
Expected: no hits.

- [ ] **Step 11: Smoke test on the GPU box**

Run: `bash tests/run_tests.sh tests/test_train_smoke.py tests/test_checkpoint_roundtrip.py tests/test_dataset_resume.py`
Expected: all PASS. `test_train_smoke.py` exercises the modified loop end to end.

- [ ] **Step 12: Commit**

```bash
git add config.yaml config.py scripts/pretrain.py tests/test_phase_targets.py
git commit -m "feat: token based checkpoint cadence, uploads, phase scoping and clean stops"
```

---

### Task 7: Training supervisor

**Files:**
- Create: `scripts/run_training.py`
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: the exit-code contract from Task 5, `TrainingConfig.phase_target_tokens` from Task 6.
- Produces: `run_phase(phase, launch, max_restarts=5, restart_window=600.0, backoff=30.0,
  sleep=time.sleep) -> int` and `main() -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_supervisor.py`:

```python
"""Supervisor restart policy: which exit codes restart, which stop, and the flap limit.
`launch` is injected so no real training process is spawned. No GPU, no TE.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.runtime.control import EXIT_OK, EXIT_PREEMPTED, EXIT_RESUME_FAILED, EXIT_USER_STOP
from scripts.run_training import run_phase


def scripted(codes):
    """A launch() stand-in returning the given exit codes in order."""
    seq = list(codes)
    calls = []

    def launch(phase):
        calls.append(phase)
        return seq.pop(0)

    launch.calls = calls
    return launch


noop_sleep = lambda s: None

# preemption restarts until the phase completes
lz = scripted([EXIT_PREEMPTED, EXIT_PREEMPTED, EXIT_OK])
assert run_phase("phase1", lz, sleep=noop_sleep) == EXIT_OK
assert len(lz.calls) == 3, lz.calls
print("[ok] preemption relaunches until the phase completes")

# a user stop is final
lz = scripted([EXIT_USER_STOP, EXIT_OK])
assert run_phase("phase1", lz, sleep=noop_sleep) == EXIT_USER_STOP
assert len(lz.calls) == 1, "a user stop must NOT be restarted"
print("[ok] a user stop ends the supervisor")

# a failed resume verification is final -- retrying would retrain the same ground
lz = scripted([EXIT_RESUME_FAILED, EXIT_OK])
assert run_phase("phase1", lz, sleep=noop_sleep) == EXIT_RESUME_FAILED
assert len(lz.calls) == 1, "a resume verification failure must NOT be restarted"
print("[ok] a resume verification failure ends the supervisor")

# an unknown crash code restarts
lz = scripted([1, EXIT_OK])
assert run_phase("phase1", lz, sleep=noop_sleep) == EXIT_OK
assert len(lz.calls) == 2
print("[ok] an unexpected crash restarts")

# flapping stops the supervisor rather than burning the rental on a crash loop
lz = scripted([1] * 20)
rc = run_phase("phase1", lz, max_restarts=3, sleep=noop_sleep)
assert rc != EXIT_OK, rc
assert len(lz.calls) == 4, lz.calls  # initial run + 3 restarts
print("[ok] the flap limit stops a crash loop")

print("\nSUPERVISOR CHECKS PASSED")
```

Note: `from scripts.run_training import run_phase` requires `scripts/__init__.py`. Create it as an
empty file in Step 3 — it does not change how the scripts are launched (they are still run as
files from the repo root).

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_supervisor.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_training'`

- [ ] **Step 3: Write `scripts/run_training.py`**

Create empty `scripts/__init__.py`, then:

```python
"""Supervise a full pretraining run: phase 1 -> phase 2, restarting through preemptions.

Runs pretrain.py as a subprocess rather than importing it, so a CUDA-level crash or an OOM kills
only the child. The exit-code contract in modules/runtime/control.py decides what happens next.
"""
import os
import subprocess
import sys
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config import TrainingConfig
from modules.runtime.control import EXIT_OK, EXIT_PREEMPTED, EXIT_RESUME_FAILED, EXIT_USER_STOP
from modules.runtime.status import format_duration
from utils import BASE_DIR, logger

PHASES = ("phase1", "phase2")
# codes that mean "a human or a verification check decided this run stops" -- restarting would
# either ignore an explicit instruction or retrain ground already covered
TERMINAL_CODES = (EXIT_USER_STOP, EXIT_RESUME_FAILED)


def launch_pretrain(phase: str) -> int:
    """Run scripts/pretrain.py for one phase, streaming its output, and return its exit code."""
    cmd = [sys.executable, os.path.join(BASE_DIR, "scripts", "pretrain.py"), "--phase", phase]
    logger.info(f"launching: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=BASE_DIR)


def run_phase(phase, launch=launch_pretrain, max_restarts=5, restart_window=600.0,
              backoff=30.0, sleep=time.sleep) -> int:
    """Run one phase to completion, relaunching on preemption or crash.

    Args:
        phase: "phase1" or "phase2".
        launch: callable taking the phase and returning an exit code. Injected for tests.
        max_restarts: relaunches allowed inside restart_window before giving up. A crash loop on
            a rented box costs money for nothing, so it is bounded.
        restart_window: seconds over which max_restarts is counted.
        backoff: seconds before the first relaunch, doubled each time, capped at 300.
        sleep: injected for tests.

    Returns:
        The exit code the supervisor should propagate.
    """
    restarts = []
    delay = backoff
    while True:
        code = launch(phase)
        if code == EXIT_OK:
            logger.info(f"{phase} complete")
            return EXIT_OK
        if code in TERMINAL_CODES:
            logger.warning(f"{phase} exited {code}; not restarting")
            return code

        now = time.time()
        restarts = [t for t in restarts if now - t < restart_window] + [now]
        if len(restarts) > max_restarts:
            logger.error(
                f"{phase} restarted {len(restarts)} times in {restart_window:.0f}s -- giving up "
                f"rather than burning the rental on a crash loop"
            )
            return code

        kind = "preempted" if code == EXIT_PREEMPTED else f"crashed (exit {code})"
        logger.warning(f"{phase} {kind}; relaunching in {delay:.0f}s (restart {len(restarts)})")
        sleep(delay)
        delay = min(delay * 2, 300.0)


def main() -> int:
    started = time.time()
    for phase in PHASES:
        target = TrainingConfig.phase_target_tokens(phase)
        logger.info(f"=== {phase}: training to {target:,} tokens ===")
        code = run_phase(phase)
        if code != EXIT_OK:
            logger.error(f"stopping after {phase} (exit {code})")
            return code
    logger.info(f"both phases complete in {format_duration(time.time() - started)}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_supervisor.py`
Expected: PASS, ending in `SUPERVISOR CHECKS PASSED`

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/run_training.py tests/test_supervisor.py
git commit -m "feat: training supervisor with phase handoff and restart policy"
```

---

### Task 8: Setup script, onstart, dependency cleanup, revision pinning

**Files:**
- Create: `scripts/setup.sh`
- Create: `scripts/onstart.sh`
- Delete: `vast_init`
- Modify: `requirements.txt`
- Modify: `scripts/prepare_data.py` (revision pinning, ~lines 169-187 and ~399-420)

**Interfaces:**
- Consumes: `scripts/fetch_tokenizer.py` (Task 1), `utils.get_hf_token` (Task 1),
  `scripts/run_training.py` (Task 7).
- Produces: nothing importable.

- [ ] **Step 1: Pin dataset revisions in `prepare_data.py`**

`make_hf_generator_factory` currently takes no revision. Add one parameter and pass it through:

```python
def make_hf_generator_factory(spec: SourceSpec, files: list, scratch_dir: str,
                              hf_token: Optional[str], seed: int, revision: Optional[str]) -> Callable:
```

and in the `hf_hub_download` call inside it:

```python
                # pin the revision the file list was taken from. without this, a source repo that
                # updates mid-run reshuffles the sorted file list and the resume state's file_idx
                # silently points at a different file.
                local_path = hf_hub_download(repo_id=spec.repo_id, filename=filename, repo_type="dataset",
                                              local_dir=scratch_dir, token=hf_token, revision=revision)
```

At the call site (~line 417), pass the already-recorded sha:

```python
                "generator_factory": make_hf_generator_factory(
                    spec, files_by_source[spec.key], scratch_dir, args.hf_token, args.seed,
                    revision_by_source[spec.key],
                ),
```

Also pin the file listing itself so the list and the downloads agree (~line 385):

```python
            all_files = hf_api.list_repo_files(spec.repo_id, repo_type="dataset", revision=info.sha)
```

- [ ] **Step 2: Verify the revision threading is complete**

Run: `grep -n "revision" scripts/prepare_data.py`
Expected: hits at the `make_hf_generator_factory` signature, the `hf_hub_download` call, the
`list_repo_files` call, `revision_by_source[spec.key] = info.sha`, the manifest write, and the
generator-factory call site — six places, all consistent.

- [ ] **Step 3: Clean up `requirements.txt`**

Remove `sentence-transformers>=5.2.0`, `fastparquet`, and `bitsandbytes>=0.30.0`. None is imported
anywhere in the repo (`bitsandbytes` is a commented-out line in `pretrain.py`; `prepare_data.py`
forces the pyarrow engine), and `sentence-transformers` pulls a torch dependency that can clobber
the NGC image's prebuilt torch and take Transformer Engine and flash-attn down with it.

- [ ] **Step 4: Write `scripts/setup.sh`**

```bash
#!/bin/bash
# One-shot environment setup for the rented box. Replaces the old vast_init.
#
# The NGC image (nvcr.io/nvidia/pytorch:25.xx-py3) ships torch, transformer_engine and flash-attn
# prebuilt for the rented GPU -- do NOT `pip install -r requirements.txt` wholesale, its
# TE/flash-attn wheels target the local dev GPU and would clobber the working prebuilt ones.
#
# Usage: bash scripts/setup.sh --hf-token hf_xxx
set -euo pipefail

HF_TOKEN_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --hf-token) HF_TOKEN_ARG="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

# 1. non-CUDA deps only. torch/transformer_engine/flash_attn come from the image.
pip install --no-cache-dir \
    "transformers>=5.5.4" \
    "numpy>=1.21.0" \
    "pandas>=3.0.0" \
    "pyarrow>=14.0.0" \
    "huggingface_hub>=0.27.0" \
    "zstandard>=0.23.0" \
    "matplotlib>=3.10.0" \
    "accelerate>=1.13.0"

# 2. one place for the token. *.key is gitignored.
if [ -n "$HF_TOKEN_ARG" ]; then
  printf '%s' "$HF_TOKEN_ARG" > huggingface.key
  chmod 600 huggingface.key
  echo "setup: wrote huggingface.key"
fi
if [ -f huggingface.key ]; then
  HF_TOKEN="$(cat huggingface.key)"
  export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "setup: no HF token. Uploads and the gated Nemotron-Math source will fail." \
       "Re-run with --hf-token hf_xxx." >&2
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. tokenizer (public repo, no token needed)
python scripts/fetch_tokenizer.py

# 4. preflight: prove the token's SCOPE, not just connectivity. A read-only token fails every
#    upload, and under the retention policy that means a full disk 40 hours in rather than an
#    error now. Also probe the one gated dataset for the same reason.
python - <<'EOF'
import os, sys
sys.path.insert(0, ".")
from utils import HF_UPLOAD_REPO, get_hf_token, logger

token = get_hf_token()
if not token:
    logger.warning("preflight skipped: no HF token")
    sys.exit(0)

from huggingface_hub import HfApi
api = HfApi(token=token)

try:
    api.upload_file(path_or_fileobj=b"preflight", path_in_repo="preflight.txt",
                    repo_id=HF_UPLOAD_REPO, token=token)
    api.delete_file(path_in_repo="preflight.txt", repo_id=HF_UPLOAD_REPO, token=token)
    logger.info(f"preflight ok: write access to {HF_UPLOAD_REPO} confirmed")
except Exception as e:
    raise SystemExit(
        f"preflight FAILED: cannot write to {HF_UPLOAD_REPO} ({type(e).__name__}: {e}). "
        f"The token most likely lacks write scope -- fix it before starting a 40 hour run."
    )

gated = "nvidia/Nemotron-CC-Math-v1"
try:
    api.dataset_info(gated)
    logger.info(f"preflight ok: {gated} is accessible")
except Exception as e:
    logger.warning(
        f"preflight: {gated} is not accessible ({type(e).__name__}: {e}). Accept its access "
        f"request on huggingface.co before running scripts/prepare_data.py."
    )
EOF

# 5. environment sanity
TINY_LLM_ROOT="$(pwd)" TINY_LLM_ENV_INIT=/dev/null bash tests/run_env_check.sh

echo "setup: done. Next: python scripts/prepare_data.py, then python scripts/run_training.py"
```

- [ ] **Step 5: Write `scripts/onstart.sh`**

```bash
#!/bin/bash
# vast.ai onstart script. Paste as the instance's onstart command.
#
# Deliberately does NOT run prepare_data.py: that has its own hours-long interruptible lifecycle
# and its own resume state, and entangling a data-prep failure with the training launch makes both
# harder to diagnose. Run it once by hand, then let this bring training back after every reclaim.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ArminBurkhardt/tiny-llm.git}"
BRANCH="${BRANCH:-train-build}"
WORKDIR="${WORKDIR:-/workspace/tiny-llm}"

if [ ! -d "$WORKDIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
git fetch origin "$BRANCH" && git checkout "$BRANCH" && git pull --ff-only

bash scripts/setup.sh ${HF_TOKEN:+--hf-token "$HF_TOKEN"}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_FP8=1   # H100: switches chosen_recipe to fp8_recipe in pretrain.py

# run_training.py already restarts pretrain.py through preemptions; nohup keeps it alive after
# the ssh session that started it goes away.
nohup python scripts/run_training.py >> ckpts/training/train.log 2>&1 &
echo "onstart: training supervisor launched, log at $WORKDIR/ckpts/training/train.log"
```

- [ ] **Step 6: Delete `vast_init` and verify nothing references it**

```bash
git rm vast_init
grep -rn "vast_init" . --exclude-dir=.git --exclude-dir=venv
```

Expected hits only in `tests/run_tests.sh` / `tests/run_env_check.sh` comments (which mention it
as an example value for `TINY_LLM_ENV_INIT`), `PLAN.md`, and `CLAUDE.md`. Update those references
to `scripts/setup.sh`.

- [ ] **Step 7: Verify the shell scripts parse**

Run: `bash -n scripts/setup.sh && bash -n scripts/onstart.sh && echo "syntax ok"`
Expected: `syntax ok`

- [ ] **Step 8: Verify nothing was broken by the dependency removal**

Run: `grep -rn "sentence_transformers\|bitsandbytes\|fastparquet" scripts/ modules/ tests/ *.py`
Expected: only the commented-out `bitsandbytes` line in `scripts/pretrain.py`.

- [ ] **Step 9: Commit**

```bash
git add scripts/setup.sh scripts/onstart.sh requirements.txt scripts/prepare_data.py tests/run_tests.sh tests/run_env_check.sh
git rm --cached vast_init 2>/dev/null || true
git commit -m "feat: setup and onstart scripts, pinned dataset revisions, dependency cleanup"
```

---

### Task 9: Documentation

**Files:**
- Create: `docs/runbook.md`
- Rewrite: `README.md`
- Modify: `docs/configuration.md`
- Modify: `docs/training.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything.
- Produces: nothing importable.

- [ ] **Step 1: Gather the current numbers rather than trusting the old docs**

Run: `python -c "import sys; sys.path.insert(0,'.'); from config import ModelConfig, TrainingConfig as T; print(T.total_steps, T.target_tokens, T.phase_target_tokens('phase1'))"`

Run (GPU box): `python -c "import sys; sys.path.insert(0,'.'); from modules.model.transformer import TinyMoETransformer; from config import ModelConfig; TinyMoETransformer(**ModelConfig.Params)"`
— its `__init__` prints total/active parameter counts and the FLOP/token estimate. Use those
printed numbers in the README, not the stale "~243M".

- [ ] **Step 2: Write `docs/runbook.md`**

It must contain, in this order:

1. **Prerequisites** — an HF token with *write* scope on `ikeafisch4/temp-train`, and an accepted
   access request for `nvidia/Nemotron-CC-Math-v1`.
2. **Commands in order**, each with what it prints when it works and roughly how long it takes:
   ```bash
   bash scripts/setup.sh --hf-token hf_xxx     # ~3 min
   python scripts/prepare_data.py              # hours; resumable, safe to kill and rerun
   python scripts/run_training.py              # ~40 h; phase1 -> phase2
   ```
3. **How to stop each**, as the table from `modules/runtime/control.py`:
   `touch ckpts/training/STOP` (clean stop, no restart), `kill -TERM` (stop and let the supervisor
   restart), `kill -USR1` (checkpoint without stopping), Ctrl-C.
4. **Exit codes**: 0 phase complete, 10 user stop, 20 preempted, 30 resume verification failed.
5. **What normal looks like** — a real log line from the run, and which fields to watch:
   `Loss`, `p_halt`, `loop_scale` (all three entries), `MFU`, `Tokens/sec`, `ETA`.
6. **What is alarming and what to do**:
   - `keeping N checkpoint(s) past the retention window because their upload has not succeeded`
     → the token lost write scope or the network is down; fix it before the disk fills.
   - `resume verification failed` (exit 30) → the supervisor has stopped on purpose. Inspect
     `ckpts/training/` and `run_state.json` before restarting anything.
   - `found N checkpoint(s) ... but none of them load` → disk corruption; pull the newest
     checkpoint back down from HF.
   - `p_halt` pinned near 0.00–0.02 → the ponder deadlock in `CLAUDE.md`; the run is still
     training but the loop is dead.
   - `loop_scale` entries collapsing toward 0 → same family of problem, check per-loop CE.
7. **Where the artifacts are**: `ikeafisch4/temp-train` — `checkpoints/`, `checkpoints/final/`,
   `logs/train.log`, `graphs/`, `status.json`, `manifest.json`.
8. **Do not rsync `data/prepared/` or the local `manifest.json` up.** The local copies are a 45M
   token smoke corpus, and the `_prepare_state_*.json` sidecars would make `prepare_data.py`
   believe it had already finished.
9. **Resuming after a reclaim**: relaunch the instance with the same onstart; it re-clones,
   re-fetches, and `run_training.py` picks up from the newest checkpoint. Checkpoints on HF are
   the recovery path if the disk is gone — download the newest into `ckpts/training/` first.

- [ ] **Step 3: Rewrite `README.md`**

Correct facts it currently gets wrong: parameter count (use Step 1's printed number, not "~243M"),
and the identity expert (removed — the halt head replaced it). Cover: what the model is, the
expert-pool layout, the loop recurrence, how to run training locally and on a rented box (pointing
at `docs/runbook.md`), and the layout of `scripts/` including the four new files.

- [ ] **Step 4: Update `docs/configuration.md`**

Fix the stale defaults (`batch_size: 3` → 8, `target_tokens: 5e9` → 29.9e9, `lambda_ponder: 3e-3`
→ 0.15) and document the keys it omits: `loop_ce_subsample`, `loop_count_sampling`, `data_dir`,
`phase`, plus this plan's `checkpoint_every_tokens`, `keep_local_checkpoints`, `phase1_fraction`,
`hf_upload_repo`.

- [ ] **Step 5: Update `docs/training.md`**

It still describes the pre-Step-9 `data_config.json` / parquet dataset. Replace with the mmap
`{phase}.bin`/`.idx` corpus, the supervisor, and the checkpoint/upload lifecycle.

- [ ] **Step 6: Update `CLAUDE.md`**

Add to the layout block:

```
modules/runtime/checkpoints.py  naming, latest-VALID resume, retention, run_state sidecar
modules/runtime/hf_sync.py      background upload thread (failures never raise into training)
modules/runtime/control.py      STOP sentinel + SIGTERM/SIGUSR1 -> flags read at LOG_INTERVAL
modules/runtime/status.py       status.json writer + ETA arithmetic
scripts/run_training.py         supervisor: phase1 -> phase2, restart policy, flap limit
scripts/setup.sh                box setup (replaces vast_init) + upload preflight
scripts/onstart.sh              vast onstart: clone -> setup -> nohup run_training
scripts/fetch_tokenizer.py      pull the pruned tokenizer from the Hub
```

Add a **Run lifecycle** section stating the invariants a future change must not break:

- The exit-code contract (0/10/20/30) is shared between `control.py` and `run_training.py`;
  changing one without the other turns a clean stop into a restart loop.
- **A checkpoint is only deleted once it is both outside the keep window and confirmed uploaded.**
  Relaxing either half makes a lost upload unrecoverable.
- **`global_offset` is phase-scoped.** Loading a checkpoint from the other phase must reset it to
  0 while preserving `token_count` — the corpus differs, the LR anchor does not.
- Signal handlers set flags only; all I/O happens in the `LOG_INTERVAL` block.
- `modules/runtime/` must stay free of `transformer_engine` / `modules.model` imports so its
  tests run without a GPU.
- The checkpoint cadence is in **tokens**, not steps. `step` counts micro-steps, so any step-based
  cadence is silently divided by `grad_accumulation_steps`.

- [ ] **Step 7: Run the full suite**

Run: `python tests/test_hf_token.py && python tests/test_checkpoint_atomic.py && python tests/test_checkpoint_lifecycle.py && python tests/test_hf_sync.py && python tests/test_control.py && python tests/test_phase_targets.py && python tests/test_supervisor.py`
Expected: all seven PASS.

Run (GPU box): `bash tests/run_tests.sh tests/test_attention_equiv.py tests/test_overfit.py tests/test_per_loop_ce.py tests/test_correctness_head.py tests/test_ponder_deadlock.py tests/test_dataset_resume.py tests/test_checkpoint_roundtrip.py tests/test_prepare_data.py tests/test_review_fixes.py tests/test_train_smoke.py`
Expected: all PASS, no regressions against the pre-change baseline.

- [ ] **Step 8: Commit**

```bash
git add docs/runbook.md README.md docs/configuration.md docs/training.md CLAUDE.md
git commit -m "docs: runbook, readme rewrite and refreshed configuration and training docs"
```

---

## Verification Before Declaring Done

- [ ] All seven new no-GPU tests pass from the repo root.
- [ ] All pre-existing tests pass on the GPU box (the baseline is 12/12).
- [ ] `grep -rn "DeepSeek-V4-Pro-tokenizer" scripts/ tests/` shows only `prune_vocab.py`'s
      unpruned source path.
- [ ] `grep -rn "get_latest_checkpoint_epoch\|step % 1500" scripts/` is empty.
- [ ] A real short run (`target_tokens` temporarily lowered, `hf_upload_repo: ""`) produces
      exactly one `checkpoint_phase1_final.pt`, at most `keep_local_checkpoints` rolling
      checkpoints, one `loss_graph.png`, and no `*_epoch*_step*.png`.
- [ ] `touch ckpts/training/STOP` during that run stops it within ~5 seconds with exit code 10.

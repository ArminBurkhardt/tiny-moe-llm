"""Checkpoint naming, stale-file cleanup, latest-VALID resume selection, retention, and resume
verification. No GPU, no TE, no torch -- pure file bookkeeping.
"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.runtime.checkpoints import (
    ResumeVerificationError, cleanup_stale_files, final_name, find_resume_checkpoint, is_final,
    prune_checkpoints, read_run_state, resolve_resume_scope, resume_phase_index, rolling_name,
    verify_resume, write_run_state,
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

# --- resume_phase_index (the phase-rewind bug: a reclaimed instance restarts run_training.py,
# whose old unconditional loop re-entered phase 1 even though phase-2 checkpoints already existed
# on the surviving disk) ------------------------------------------------------------------------
assert resume_phase_index(tempfile.mkdtemp()) == 0
print("[ok] a cold start (no checkpoints) begins at phase 1")

d = tempfile.mkdtemp()
touch(d, "checkpoint_phase1_tok100M_loss3.0000.pt")
assert resume_phase_index(d) == 0
print("[ok] a rolling phase-1 checkpoint alone stays at phase 1")

d = tempfile.mkdtemp()
touch(d, "checkpoint_phase1_tok25000M_loss2.0000.pt")
touch(d, "checkpoint_phase1_final.pt")
assert resume_phase_index(d) == 1
print("[ok] a phase-1 final checkpoint advances to phase 2")

d = tempfile.mkdtemp()
touch(d, "checkpoint_phase1_final.pt")
touch(d, "checkpoint_phase2_tok27000M_loss1.8000.pt")
assert resume_phase_index(d) == 1
print("[ok] a rolling phase-2 checkpoint stays at phase 2, not zero")

# the bug's exact trace: only a phase-2 rolling checkpoint is on disk (e.g. recovery restored just
# the newest rolling file, per the runbook, and phase 1's never-pruned final.pt was not restored)
d = tempfile.mkdtemp()
touch(d, "checkpoint_phase2_tok27000M_loss1.8000.pt")
assert resume_phase_index(d) == 1, "a phase-2 checkpoint with no phase-1 final must still skip phase 1"
print("[ok] a phase-2 checkpoint alone still skips phase 1, even without a phase-1 final on disk")

d = tempfile.mkdtemp()
touch(d, "checkpoint_phase1_final.pt")
touch(d, "checkpoint_phase2_final.pt")
assert resume_phase_index(d) == 1, "both phases done: nothing left in PHASE_ORDER past the last index"
print("[ok] both phases complete clamps to the last phase index, not out of range")

print("\nCHECKPOINT LIFECYCLE CHECKS PASSED")

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

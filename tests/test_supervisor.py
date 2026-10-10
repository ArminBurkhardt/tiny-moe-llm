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

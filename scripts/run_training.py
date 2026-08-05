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
from modules.runtime import checkpoints as ckpt_lib
from modules.runtime.control import EXIT_OK, EXIT_PREEMPTED, EXIT_RESUME_FAILED, EXIT_USER_STOP
from modules.runtime.status import format_duration
from utils import BASE_DIR, logger

PHASES = ckpt_lib.PHASE_ORDER
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
    checkpoint_dir = os.path.join(BASE_DIR, "ckpts", "training")
    start_index = ckpt_lib.resume_phase_index(checkpoint_dir)
    if start_index > 0:
        skipped = ", ".join(PHASES[:start_index])
        logger.info(
            f"skipping already-complete phase(s) on disk: {skipped} -- see "
            f"checkpoints.resume_phase_index if this looks wrong"
        )
    for phase in PHASES[start_index:]:
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

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

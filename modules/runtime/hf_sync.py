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
    __slots__ = ("op", "local_path", "repo_path", "droppable")

    def __init__(self, op, local_path, repo_path, droppable):
        self.op = op
        self.local_path = local_path
        self.repo_path = repo_path
        self.droppable = droppable


class HFSync:
    """Queue-backed uploader (and pruner) for a Hugging Face repo.

    Args:
        repo_id: e.g. "ikeafisch4/temp-train". An empty string disables uploading entirely, so a
            local run needs no special casing at the call sites.
        token: HF token with write access to repo_id.
        api: an object exposing ``upload_file``/``delete_file``/``super_squash_history`` with the
            real ``huggingface_hub.HfApi`` signatures. Defaults to a real ``HfApi``; injectable
            for tests.
        max_queue: soft cap on pending jobs before droppable ones start being discarded.
        retries: attempts per file before giving up.
        backoff: seconds before the first retry, tripled each attempt.
        squash_min_interval: minimum seconds between history squashes (see delete()'s docstring).
    """

    def __init__(self, repo_id, token=None, api=None, max_queue=8, retries=3, backoff=5.0,
                 squash_min_interval=1800.0):
        self.repo_id = repo_id or ""
        self.enabled = bool(self.repo_id)
        self._token = token
        self._retries = retries
        self._backoff = backoff
        self._max_queue = max_queue
        self._squash_min_interval = squash_min_interval

        self._queue = deque()
        self._cv = threading.Condition()
        self._uploaded = set()
        self._closing = False
        self._thread = None
        # true while the worker holds a job it has taken off the queue -- drain() must wait for
        # that job too, not just for the queue to empty
        self._busy = False
        # squash_min_interval below covers repeated bursts; -inf so the first delete after
        # startup can always trigger one
        self._last_squash = float("-inf")

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

    def _enqueue(self, job: _Job) -> None:
        with self._cv:
            if len(self._queue) >= self._max_queue:
                for i, queued in enumerate(self._queue):
                    if queued.droppable:
                        del self._queue[i]
                        logger.warning(
                            f"upload queue full ({self._max_queue}); dropping "
                            f"{queued.op} {queued.repo_path}. A dropped upload stays on disk and "
                            f"un-pruned; a dropped delete just leaves stale clutter on the Hub -- "
                            f"uploads are falling behind either way."
                        )
                        break
            self._queue.append(job)
            self._cv.notify()

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
        self._enqueue(_Job("upload", local_path, repo_path, droppable))

    def delete(self, repo_path: str) -> None:
        """Queue removal of a file already confirmed uploaded, and reclaim its storage.

        Only meant for checkpoints modules.runtime.checkpoints.prune_checkpoints has already
        deleted locally -- never for `checkpoints/final/*`, which is never pruned. A plain
        ``delete_file`` only removes the blob from the repo's current tree; the git history still
        references it, so the Hub keeps charging storage for a 2GB checkpoint until the history
        itself is rewritten. So every successful delete also (throttled) triggers
        ``super_squash_history``, which squashes the whole branch into one commit and actually
        frees the space -- fine for this repo since it exists only as a scratch mirror of local
        checkpoints, not something anyone reads commit-by-commit. Throttled to at most once per
        ``squash_min_interval`` so a burst of deletes (several prunes back to back) costs one
        squash, not one per file.
        """
        if not self.enabled:
            return
        self._enqueue(_Job("delete", None, repo_path, droppable=True))

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
                if job.op == "upload":
                    self._upload_with_retries(job)
                else:
                    self._delete_with_retries(job)
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

    def _delete_with_retries(self, job):
        delay = self._backoff
        for attempt in range(1, self._retries + 1):
            try:
                self._api.delete_file(
                    path_in_repo=job.repo_path, repo_id=self.repo_id, token=self._token,
                )
                logger.info(f"deleted {job.repo_path} from {self.repo_id}")
                self._maybe_squash_history()
                return
            except Exception as e:
                if attempt == self._retries:
                    # same swallow-and-log policy as upload: a stale file left on the Hub is
                    # clutter, not a correctness problem, and must not take the run down with it.
                    logger.error(
                        f"delete of {job.repo_path} failed after {self._retries} attempts "
                        f"({type(e).__name__}: {e}); leaving it on the Hub"
                    )
                    return
                logger.warning(
                    f"delete of {job.repo_path} attempt {attempt}/{self._retries} failed "
                    f"({type(e).__name__}: {e}); retrying in {delay:.0f}s"
                )
                time.sleep(delay)
                delay *= 3

    def _maybe_squash_history(self):
        now = time.time()
        if now - self._last_squash < self._squash_min_interval:
            return
        try:
            self._api.super_squash_history(repo_id=self.repo_id, token=self._token)
            self._last_squash = now
            logger.info(f"squashed {self.repo_id} history to reclaim deleted-checkpoint storage")
        except Exception as e:
            # non-fatal: the file is still gone from the tree, it just keeps costing storage in
            # history until a future delete's squash succeeds
            logger.warning(
                f"history squash on {self.repo_id} failed ({type(e).__name__}: {e}); deleted "
                f"checkpoints stay in git history (and billed storage) until the next attempt"
            )

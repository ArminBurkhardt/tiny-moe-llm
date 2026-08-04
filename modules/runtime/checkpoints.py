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

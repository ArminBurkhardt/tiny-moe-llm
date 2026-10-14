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

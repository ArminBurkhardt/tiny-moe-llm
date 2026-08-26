import os
import logging
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# set warning color to yellow
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("\033[93m%(levelname)s: %(message)s\033[0m")
ch.setFormatter(formatter)
logger.addHandler(ch)

FP32 = torch.float32
FP16 = torch.float16
BF16 = torch.bfloat16

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
        "losses": losses,
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


def model_params_for_state_dict(state_dict, params: dict) -> dict:
    """``ModelConfig.Params`` with the IR table's shape taken from the checkpoint, not from yaml.

    The 65536 x 256 reshape changed both the table's dimensions and whether it is read through
    centroids, and every eval in this repo has to keep scoring the pre-reshape checkpoints -- they
    are the baseline each gate is quoted against. Constructing from ``config.yaml`` alone would
    make an old checkpoint fail to load the moment the yaml is flipped, which is the wrong way
    round: the checkpoint is the authority on its own shape.

    Reads three things off the state dict: the key table's ``[entries, dim]``, and whether a
    ``centroids`` buffer exists at all (absent => the exact full-table path the old checkpoints
    were trained under, so clusters are forced to 0).

    Args:
        state_dict: the checkpoint's ``model_state_dict``.
        params: the yaml-derived kwargs to start from; not mutated.

    Returns:
        A copy with ``num_ir_entries`` / ``ir_dim`` / ``ir_num_clusters`` set to match.
    """
    out = dict(params)
    keys = [k for k in state_dict if k.endswith("ir_module.z_keys")]
    if not keys:
        return out
    entries, dim = tuple(state_dict[keys[0]].shape)
    out["num_ir_entries"] = int(entries)
    out["ir_dim"] = int(dim)
    centroids = [k for k in state_dict if k.endswith("ir_module.centroids")]
    if not centroids:
        out["ir_num_clusters"] = 0
    else:
        out["ir_num_clusters"] = int(state_dict[centroids[0]].shape[0])
    return out


def load_checkpoint(model, optimizer, scheduler, path):
    """Restore a pretraining run. Returns a 6-tuple; pre-Phase-0 checkpoints returned 7.

    ``strict=True`` is deliberate: a checkpoint written before the halt/correctness heads were
    deleted still carries ``moe.halt_proj.*`` and ``correct_proj.*``, and silently dropping them
    would also silently keep a ``loop_scale`` that is ~4x too large for an ungated loop. Run it
    through ``scripts/migrate_phase0.py`` first -- that is what folds the gate in.
    """
    checkpoint = torch.load(path)
    if "optimizer_state_dict" not in checkpoint:
        # scripts/migrate_phase0.py drops the optimizer state on purpose (its moments are indexed
        # by param-group position and two tensors left the model). Say that, rather than letting a
        # bare KeyError surface as "this checkpoint would not load" during a resume scan.
        raise ValueError(
            f"{os.path.basename(path)} carries no optimizer state"
            + (" -- it is a Phase 0 migration output, which is a finetune SEED, not a resume "
               "point. Pass it with -c to initialize from it."
               if "phase0_migration" in checkpoint else "")
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    dataset_idx = checkpoint["dataset_idx"]
    token_count = checkpoint.get("token_count", 0)
    # legacy (pre Step 9) checkpoints have no global_offset -- there's no sound mapping from the
    # old per-file position to a doc index in the new flat corpus, so they just restart the doc
    # stream from 0
    global_offset = checkpoint.get("global_offset", 0)
    losses = checkpoint.get("losses", None)
    # legacy (pre phase scoping) checkpoints have no phase -- the caller treats None as "same
    # phase as the one being launched", matching the old behaviour
    phase = checkpoint.get("phase", None)
    logger.info(f"Checkpoint loaded from {path}")
    return epoch, dataset_idx, token_count, global_offset, losses, phase

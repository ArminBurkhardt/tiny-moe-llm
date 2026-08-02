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



def save_checkpoint(model, optimizer, scheduler, epoch, dataset_idx, path, token_count=0, global_offset=0, losses=None):
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
        "losses": losses
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved at {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
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
    logger.info(f"Checkpoint loaded from {path}")
    return epoch, dataset_idx, token_count, global_offset, losses

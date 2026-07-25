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



def save_checkpoint(model, optimizer, scheduler, epoch, dataset_idx, path, token_count=0, file_idx=0, losses=None, file_order=None, shard_token_count=0, record_idx=0, worker_positions=None, num_data_workers=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "dataset_idx": dataset_idx,
        "epoch": epoch,
        "token_count": token_count,
        # global resume position (conservative min across workers): file, record within it, and tokens taken from it. used as the fallback when the worker count changes
        "file_idx": file_idx,
        "record_idx": record_idx,
        "shard_token_count": shard_token_count,
        # {worker_id: (file_idx, record_idx, shard_token_count)} plus the worker count they were produced with. 
        # only usable when resuming with that same count, else fall back to above
        "worker_positions": worker_positions,
        "num_data_workers": num_data_workers,
        # the epochs shuffled file ordering. file_idx indexes into it, so restoring it is what keeps consumed shards from reappearing
        "file_order": file_order,
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
    file_idx = checkpoint.get("file_idx", 0)
    # legacy checkpoints predate these; the defaults fall back to the old behaviour:
    # 0 re-reads the resume shard from the top, None sends the trainer down the global path
    record_idx = checkpoint.get("record_idx", 0)
    shard_token_count = checkpoint.get("shard_token_count", 0)
    worker_positions = checkpoint.get("worker_positions", None)
    num_data_workers = checkpoint.get("num_data_workers", None)
    file_order = checkpoint.get("file_order", None)
    losses = checkpoint.get("losses", None)
    logger.info(f"Checkpoint loaded from {path}")
    return epoch, dataset_idx, token_count, file_idx, record_idx, losses, file_order, shard_token_count, worker_positions, num_data_workers

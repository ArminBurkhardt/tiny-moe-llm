"""
Show dataset progress from the latest training checkpoint.

Reads only checkpoint metadata (no model weights) and reports progress through the flat,
unshuffled document stream (PLAN.md Step 9): the checkpoint's global_offset is a doc index
into {data_dir}/{phase}.idx.
"""

import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    best_ts, best_path = 0, None
    if not os.path.isdir(checkpoint_dir):
        return None
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("checkpoint") and fname.endswith(".pt"):
            fpath = os.path.join(checkpoint_dir, fname)
            ts = os.path.getmtime(fpath)
            if ts > best_ts:
                best_ts, best_path = ts, fpath
    return best_path


def load_checkpoint_meta(path: str) -> dict:
    """Load a checkpoint without pulling in the model/optimizer tensors."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch":         ckpt.get("epoch", 0),
        "dataset_idx":   ckpt.get("dataset_idx", 0),
        "token_count":   ckpt.get("token_count", 0),
        "global_offset": ckpt.get("global_offset", 0),
    }


def num_docs(idx_path: str) -> int | None:
    if not os.path.exists(idx_path):
        return None
    return os.path.getsize(idx_path) // 8 - 1


def main():
    from config import TrainingConfig

    checkpoint_dir = os.path.join(BASE_DIR, "ckpts", "training")
    data_dir = os.path.join(BASE_DIR, TrainingConfig.data_dir)
    idx_path = os.path.join(data_dir, f"{TrainingConfig.phase}.idx")

    ckpt_path = find_latest_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        print("No checkpoint found in", checkpoint_dir)
        return

    print(f"Checkpoint     : {ckpt_path}")
    meta = load_checkpoint_meta(ckpt_path)
    total_docs = num_docs(idx_path)

    print(f"Epoch          : {meta['epoch']}")
    print(f"Step           : {meta['dataset_idx']:,}")
    print(f"Tokens         : {meta['token_count'] / 1e9:.3f}B")
    print(f"Phase / split  : {TrainingConfig.phase}")
    print(f"Global offset  : doc {meta['global_offset']:,}", end="")
    if total_docs is not None:
        pct = meta["global_offset"] / total_docs * 100 if total_docs else 0
        print(f" / {total_docs:,} ({pct:.1f}% consumed)")
    else:
        print(f"  ({idx_path} not found -- can't compute total docs)")


if __name__ == "__main__":
    main()

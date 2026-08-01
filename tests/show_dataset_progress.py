"""
Show dataset progress from the latest training checkpoint.

Reads only checkpoint metadata (no model weights) and reconstructs the effective
file ordering to print:
  - all already-consumed files
  - the current file (at file_idx)
  - all remaining files

Ordering source:
  - new checkpoints save the exact shuffled ``file_order``; it is used verbatim (ground truth).
  - legacy checkpoints (no ``file_order``) are reconstructed the same way the trainer's
    ``Dataset.build_legacy_order`` does: the first ``file_idx`` sorted files are consumed and the
    remaining unused shards are shuffled with seed ``config seed + epoch``.
"""

import os
import sys
import json
import random
from pathlib import Path

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


def build_file_list(sources: list[dict]) -> list[tuple[str, str]]:
    """Canonical, sorted (path, column) list. Mirrors Dataset._discover_files."""
    files = []
    for src in sources:
        root = src.get("root")
        column = src.get("column", "text")
        pattern = src.get("glob", "*")
        if root is None:
            continue
        abs_root = root if os.path.isabs(root) else os.path.join(BASE_DIR, root)
        if not os.path.exists(abs_root):
            print(f"  [MISSING] {abs_root}")
            continue
        for path in sorted(Path(abs_root).rglob(pattern)):
            if path.is_file() and path.suffix in {".parquet", ".jsonl", ".json"}:
                files.append((str(path), column))
    return files


def read_config_seed(default: int = 42) -> int:
    """Read the shuffle seed from config.yaml (mirrors TrainingConfig.seed); fall back to default."""
    try:
        import yaml
        with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return int(cfg["training"].get("seed", default))
    except Exception:
        return default


def resolve_order(file_order, file_idx: int, sources: list[dict], epoch: int):
    """Return (order, is_legacy): the effective [(path, column), ...] iteration order.

    Prefers the saved ``file_order``; otherwise rebuilds the legacy ordering exactly as the
    trainer's ``Dataset.build_legacy_order`` does (consumed sorted prefix + shuffled remainder).
    """
    if file_order:
        return [tuple(e) for e in file_order], False
    all_files = build_file_list(sources)
    seed = read_config_seed() + epoch
    consumed = all_files[:file_idx]
    remaining = list(all_files[file_idx:])
    random.Random(seed).shuffle(remaining)
    return consumed + remaining, True


def load_checkpoint_meta(path: str) -> dict:
    """Load a checkpoint without pulling in the model/optimizer tensors."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch":       ckpt.get("epoch", 0),
        "dataset_idx": ckpt.get("dataset_idx", 0),
        "token_count": ckpt.get("token_count", 0),
        "file_idx":    ckpt.get("file_idx", 0),
        "file_order":  ckpt.get("file_order", None),
    }


def main():
    checkpoint_dir = os.path.join(BASE_DIR, "ckpts", "training")
    config_path    = os.path.join(BASE_DIR, "data_config.json")

    # --- checkpoint ---
    ckpt_path = find_latest_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        print("No checkpoint found in", checkpoint_dir)
        return

    print(f"Checkpoint : {ckpt_path}")
    meta = load_checkpoint_meta(ckpt_path)
    epoch       = meta["epoch"]
    dataset_idx = meta["dataset_idx"]
    token_count = meta["token_count"]
    file_idx    = meta["file_idx"]
    file_order  = meta["file_order"]

    # --- file list ---
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    sources = config.get("pretrain", [])
    files, is_legacy = resolve_order(file_order, file_idx, sources, epoch)

    print(f"Epoch      : {epoch}")
    print(f"Step       : {dataset_idx:,}")
    print(f"Tokens     : {token_count / 1e9:.3f}B")
    print(f"File index : {file_idx}")
    print(f"Ordering   : {'legacy (reconstructed: sorted prefix + shuffled remainder)' if is_legacy else 'saved shuffle from checkpoint'}")
    print()

    if not files:
        print("No data files found.")
        return

    total = len(files)
    pct   = file_idx / total * 100 if total else 0

    print(f"Total files : {total}  ({pct:.1f}% consumed)\n")

    col_w = max(len(p) for p, _ in files) + 2

    # already consumed
    if file_idx > 0:
        print(f"--- DONE ({file_idx} file{'s' if file_idx != 1 else ''}) ---")
        for i, (path, col) in enumerate(files[:file_idx]):
            rel = os.path.relpath(path, BASE_DIR)
            print(f"  [{i:>5}] {rel:<{col_w}}  col={col}")
        print()

    # current file
    if file_idx < total:
        path, col = files[file_idx]
        rel = os.path.relpath(path, BASE_DIR)
        size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
        print(f"--- CURRENT ---")
        print(f"  [{file_idx:>5}] {rel:<{col_w}}  col={col}  size={size_mb:.1f} MB")
        print()

    # remaining
    remaining = files[file_idx + 1:]
    if remaining:
        print(f"--- REMAINING ({len(remaining)} file{'s' if len(remaining) != 1 else ''}) ---")
        for i, (path, col) in enumerate(remaining, start=file_idx + 1):
            rel = os.path.relpath(path, BASE_DIR)
            size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
            print(f"  [{i:>5}] {rel:<{col_w}}  col={col}  size={size_mb:.1f} MB")
    else:
        print("--- All files consumed ---")


if __name__ == "__main__":
    main()

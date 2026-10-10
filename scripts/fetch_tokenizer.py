"""Download the pruned 65536-token tokenizer from the Hub into ckpts/pretrained/.

ckpts/ is gitignored, so a fresh clone on a rented box has no tokenizer at all and every entry
point dies on line one. The repo is public, so no token is required -- get_hf_token() is passed
only so a private mirror would also work.
"""
import argparse
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from huggingface_hub import snapshot_download

from utils import TOKENIZER_DIR, TOKENIZER_REPO, get_hf_token, logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=TOKENIZER_REPO)
    parser.add_argument("--dest", default=TOKENIZER_DIR)
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()

    marker = os.path.join(args.dest, "tokenizer.json")
    if os.path.isfile(marker) and not args.force:
        logger.info(f"tokenizer already present at {args.dest} (use --force to re-download)")
        return 0

    os.makedirs(args.dest, exist_ok=True)
    logger.info(f"downloading {args.repo} -> {args.dest}")
    snapshot_download(repo_id=args.repo, local_dir=args.dest, token=get_hf_token())

    if not os.path.isfile(marker):
        raise RuntimeError(f"{args.repo} downloaded but {marker} is missing -- wrong repo?")
    logger.info(f"tokenizer ready at {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

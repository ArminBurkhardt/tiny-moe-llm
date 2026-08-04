#!/bin/bash
# One-shot environment setup for the rented box. Replaces the old vast_init.
#
# The NGC image (nvcr.io/nvidia/pytorch:25.xx-py3) ships torch, transformer_engine and flash-attn
# prebuilt for the rented GPU -- do NOT `pip install -r requirements.txt` wholesale, its
# TE/flash-attn wheels target the local dev GPU and would clobber the working prebuilt ones.
#
# Usage: bash scripts/setup.sh --hf-token hf_xxx
set -euo pipefail

HF_TOKEN_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --hf-token) HF_TOKEN_ARG="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

# 1. non-CUDA deps only. torch/transformer_engine/flash_attn come from the image.
pip install --no-cache-dir \
    "transformers>=5.5.4" \
    "numpy>=1.21.0" \
    "pandas>=3.0.0" \
    "pyarrow>=14.0.0" \
    "huggingface_hub>=0.27.0" \
    "zstandard>=0.23.0" \
    "matplotlib>=3.10.0" \
    "accelerate>=1.13.0"

# 2. one place for the token. *.key is gitignored.
if [ -n "$HF_TOKEN_ARG" ]; then
  printf '%s' "$HF_TOKEN_ARG" > huggingface.key
  chmod 600 huggingface.key
  echo "setup: wrote huggingface.key"
fi
if [ -f huggingface.key ]; then
  HF_TOKEN="$(cat huggingface.key)"
  export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "setup: no HF token. Uploads and the gated Nemotron-Math source will fail." \
       "Re-run with --hf-token hf_xxx." >&2
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. tokenizer (public repo, no token needed)
python scripts/fetch_tokenizer.py

# 4. preflight: prove the token's SCOPE, not just connectivity. A read-only token fails every
#    upload, and under the retention policy that means a full disk 40 hours in rather than an
#    error now. Also probe the one gated dataset for the same reason.
python - <<'EOF'
import os, sys
sys.path.insert(0, ".")
from utils import HF_UPLOAD_REPO, get_hf_token, logger

token = get_hf_token()
if not token:
    logger.warning("preflight skipped: no HF token")
    sys.exit(0)

from huggingface_hub import HfApi
api = HfApi(token=token)

try:
    api.upload_file(path_or_fileobj=b"preflight", path_in_repo="preflight.txt",
                    repo_id=HF_UPLOAD_REPO, token=token)
    api.delete_file(path_in_repo="preflight.txt", repo_id=HF_UPLOAD_REPO, token=token)
    logger.info(f"preflight ok: write access to {HF_UPLOAD_REPO} confirmed")
except Exception as e:
    raise SystemExit(
        f"preflight FAILED: cannot write to {HF_UPLOAD_REPO} ({type(e).__name__}: {e}). "
        f"The token most likely lacks write scope -- fix it before starting a 40 hour run."
    )

gated = "nvidia/Nemotron-CC-Math-v1"
try:
    api.dataset_info(gated)
    logger.info(f"preflight ok: {gated} is accessible")
except Exception as e:
    logger.warning(
        f"preflight: {gated} is not accessible ({type(e).__name__}: {e}). Accept its access "
        f"request on huggingface.co before running scripts/prepare_data.py."
    )
EOF

# 5. environment sanity
TINY_LLM_ROOT="$(pwd)" TINY_LLM_ENV_INIT=/dev/null bash tests/run_env_check.sh

echo "setup: done. Next: python scripts/prepare_data.py, then python scripts/run_training.py"

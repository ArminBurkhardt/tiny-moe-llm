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
  # a missing token only truly doesn't matter if config.yaml has uploads disabled -- otherwise
  # HFSync stays "enabled" (repo id is non-empty), every upload 401s, and retention correctly
  # refuses to delete the un-uploaded files, so the failure mode is a silently full disk found
  # hours into an unattended run rather than an error now, at minute two.
  UPLOAD_REPO="$(python - <<'EOF'
import sys
sys.path.insert(0, ".")
from config import TrainingConfig
from utils import HF_UPLOAD_REPO
print(TrainingConfig.upload_repo(HF_UPLOAD_REPO))
EOF
)"
  if [ -n "$UPLOAD_REPO" ]; then
    echo "setup: FATAL: no HF token, but config.yaml's hf_upload_repo resolves to '$UPLOAD_REPO'." \
         "Every checkpoint upload would fail for the whole run. Set \$HF_TOKEN (e.g. as a vast.ai" \
         "instance env var) or pass --hf-token hf_xxx, or set hf_upload_repo: \"\" to run local-only." >&2
    exit 1
  fi
  echo "setup: no HF token. Uploads are disabled by config.yaml, so this is fine; the gated" \
       "Nemotron-Math source will still fail without one." >&2
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

# run_env_check.sh only does `import transformer_engine.pytorch`, which is too shallow: pretrain.py
# also constructs `transformer_engine.common.recipe.DelayedScaling` and every modules/model/ file
# does its own `import transformer_engine.pytorch as te` at module scope, so a TE build whose API
# doesn't match this repo's pin (e.g. an older/newer NGC image tag) can still pass run_env_check.sh
# and then fail here -- catch that now, not after prepare_data.py has spent hours of rental.
python -c "import sys; sys.path.insert(0, '.'); import scripts.pretrain" \
  || { echo "setup: FATAL: 'import scripts.pretrain' failed -- almost certainly a" \
            "transformer_engine/flash-attn version mismatch in this NGC image. Fix before running" \
            "prepare_data.py; see CLAUDE.md's 'Environment' section." >&2; exit 1; }
echo "setup: scripts.pretrain imports cleanly (transformer_engine/flash-attn API check)"

echo "setup: done. Next: python scripts/prepare_data.py, then python scripts/run_training.py"

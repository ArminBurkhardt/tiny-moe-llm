#!/usr/bin/env bash
# one compact JSON about the run on stdout, for the login node: no GPU, a few seconds. usage:
#   bash scripts/slurm/status.sh [--run run2] [--lines 400] [--target-tokens 800e9] [--budget-gpu-hours 4500] [--quota-tb 4] [--account A] [--since 2026-10-28]
# reads squeue and sacct, the trainer log, the run directory, the chain state files and bsc_quota when present.
# python3 with the standard library only (a login node's system python is fine, no container needed).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/mn5_env.sh" >/dev/null 2>&1
cd "$REPO" || { echo '{"error": "REPO not found"}'; exit 2; }
exec "${STATUS_PY:-python3}" "$HERE/status.py" --repo "$REPO" --data-root "$DATA_ROOT" "$@"

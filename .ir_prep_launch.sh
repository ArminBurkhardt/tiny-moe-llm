#!/usr/bin/env bash
set -e
cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm
source env_init
export HF_TOKEN="$(python -c 'from utils import get_hf_token; print(get_hf_token())')"
exec "$@"

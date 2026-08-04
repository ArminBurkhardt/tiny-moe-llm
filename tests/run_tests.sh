#!/bin/bash
# TINY_LLM_ROOT/TINY_LLM_ENV_INIT let this run unchanged on WSL (defaults match the local dev box)
# and on the rented box, where scripts/setup.sh has already installed everything into the system
# python and there is nothing to source (`TINY_LLM_ROOT=/workspace/tiny-llm TINY_LLM_ENV_INIT=/dev/null`).
cd "${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}"
source "${TINY_LLM_ENV_INIT:-env_init}"
for t in "$@"; do
  echo "=== $t ==="
  python "$t" || echo "!!! $t FAILED (exit $?)"
done

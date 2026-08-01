#!/bin/bash
cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm
source env_init
for t in "$@"; do
  echo "=== $t ==="
  python "$t" || echo "!!! $t FAILED (exit $?)"
done

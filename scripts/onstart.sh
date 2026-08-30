#!/bin/bash
# vast.ai onstart script. Paste as the instance's onstart command.
#
# Deliberately does NOT run prepare_data.py: that has its own hours-long interruptible lifecycle
# and its own resume state, and entangling a data-prep failure with the training launch makes both
# harder to diagnose. Run it once by hand, then let this bring training back after every reclaim.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ArminBurkhardt/tiny-llm.git}"
BRANCH="${BRANCH:-train-build}"
WORKDIR="${WORKDIR:-/workspace/tiny-llm}"

if [ ! -d "$WORKDIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
git fetch origin "$BRANCH" && git checkout "$BRANCH" && git pull --ff-only

# run_training.py has no data of its own to fall back on: on a fresh box (this hook re-runs on
# every reclaim, including the very first boot) Dataset.__init__ dies on a missing phase1.idx,
# the supervisor crash-loops 5x in 600s, and gives up -- silently, since nothing here is watching.
# scripts/prepare_data.py is a separate, hours-long, self-resuming step that must be run by hand
# once before this hook is pasted in; catch the ordering mistake here instead of losing 40 minutes
# to a dead supervisor.
if [ ! -f data/prepared/phase1.idx ]; then
  echo "onstart: data/prepared/phase1.idx missing -- run scripts/prepare_data.py first," \
       "then set this onstart hook. Not launching training." >&2
  exit 0
fi

bash scripts/setup.sh ${HF_TOKEN:+--hf-token "$HF_TOKEN"}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_FP8=1   # H100: switches chosen_recipe to fp8_recipe in pretrain.py

# run_training.py already restarts pretrain.py through preemptions; nohup keeps it alive after
# the ssh session that started it goes away.
mkdir -p ckpts/training
nohup python scripts/run_training.py >> ckpts/training/train.log 2>&1 &
echo "onstart: training supervisor launched, log at $WORKDIR/ckpts/training/train.log"

#!/usr/bin/env bash
# Unattended chain: wait out the running pretraining supervisor, then SFT, then the Step 12
# acceptance eval. Exists because there is deliberately no phase supervisor for SFT (see CLAUDE.md
# "SFT / post-training") -- this is the minimum wrapper that lets the whole post-training step run
# while nobody is awake to type the second command.
#
# Launch detached, so it survives the SSH session that started it:
#   setsid nohup bash scripts/run_sft_after_pretrain.sh > sft_chain.log 2>&1 < /dev/null &
#
# Preconditions, all checked below before anything touches the GPU:
#   - data/prepared/sft_train.{bin,idx,mask} already built (run prepare_sft_data.py BEFORE this,
#     while pretraining is still going -- it is CPU-only and does not contend for the GPU)
#   - the repo is at a commit that actually has scripts/sft.py
#
# No `set -e`: every failure mode here wants a logged message and a specific exit code, not a bare
# non-zero from whichever command happened to trip first.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

PRETRAIN_FINAL="ckpts/training/checkpoint_phase2_final.pt"
SFT_DIR="ckpts/sft"
UPLOAD_REPO="${SFT_UPLOAD_REPO:-ikeafisch4/temp-train}"
MAX_ATTEMPTS="${SFT_MAX_ATTEMPTS:-10}"

log() { echo "[chain $(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

# --- 0. preconditions ------------------------------------------------------------------------
for f in scripts/sft.py scripts/eval_abstention.py; do
  [ -f "$f" ] || { log "FATAL: $f missing -- the box is on a commit older than 'feat: add sft script'"; exit 1; }
done
for f in data/prepared/sft_train.bin data/prepared/sft_train.idx data/prepared/sft_train.mask \
         data/prepared/sft_val.bin data/prepared/sft_val.idx data/prepared/sft_val.mask; do
  [ -f "$f" ] || { log "FATAL: $f missing -- run scripts/prepare_sft_data.py first"; exit 1; }
done
log "preconditions ok"

# --- 1. wait out pretraining -----------------------------------------------------------------
# match the PID once and then poll it, rather than re-running pgrep in the loop: this script's own
# command line contains the pattern, so a repeated name match would wait on itself forever.
PRETRAIN_PID="$(pgrep -f 'scripts/run_training\.py' | head -1 || true)"
if [ -n "$PRETRAIN_PID" ]; then
  log "waiting on pretraining supervisor pid $PRETRAIN_PID"
  while kill -0 "$PRETRAIN_PID" 2>/dev/null; do sleep 60; done
  log "pretraining supervisor exited"
else
  log "no run_training.py process found -- assuming pretraining already finished"
fi

# --- 2. only proceed on a genuine completion --------------------------------------------------
# run_training.py writes phase 2's final checkpoint only when phase 2 actually reached its token
# target. A STOP sentinel (exit 10), a failed resume verification (exit 30) or a crash all leave it
# absent -- and fine-tuning a half-trained model unattended is strictly worse than doing nothing,
# because it burns the window AND produces a checkpoint that looks like a result.
if [ ! -f "$PRETRAIN_FINAL" ]; then
  log "FATAL: $PRETRAIN_FINAL missing -- pretraining did not complete cleanly. Not starting SFT."
  exit 1
fi
log "found $PRETRAIN_FINAL"

# --- 3. SFT ------------------------------------------------------------------------------------
# same exit-code contract as pretraining (modules/runtime/control.py): 0 done, 10 user STOP,
# 30 resume verification failed -- all terminal. Anything else (20 = SIGTERM, or a crash) gets
# another attempt; -c is ignored once ckpts/sft holds a resumable checkpoint, so a relaunch
# continues rather than restarting from the pretrained weights.
rc=1
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  log "sft.py attempt $attempt/$MAX_ATTEMPTS"
  python scripts/sft.py -c "$PRETRAIN_FINAL" --upload-repo "$UPLOAD_REPO"
  rc=$?
  log "sft.py exited $rc"
  case "$rc" in
    0|10|30) break ;;
    *) sleep 30 ;;
  esac
done

if [ "$rc" -ne 0 ]; then
  log "SFT did not finish cleanly (exit $rc) -- skipping the eval"
  exit "$rc"
fi

# --- 4. the Step 12 acceptance number ----------------------------------------------------------
# --baseline-checkpoint makes the calibration result a DELTA against the pretrained model rather
# than a bare number (see eval_abstention.py's docstring on why only the teacher-forced pass can be
# read off a pretrained checkpoint at all).
if [ ! -f "$SFT_DIR/checkpoint_sft_final.pt" ]; then
  log "no $SFT_DIR/checkpoint_sft_final.pt -- skipping the eval"
  exit 1
fi
log "running the abstention eval"
python scripts/eval_abstention.py \
  -c "$SFT_DIR/checkpoint_sft_final.pt" \
  --baseline-checkpoint "$PRETRAIN_FINAL" \
  --json-out "$SFT_DIR/abstention_eval.json"
log "eval exited $? -- chain complete"

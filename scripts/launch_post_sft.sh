#!/usr/bin/env bash
# the consolidation SFT after the main run (docs/plans/RUN2_handover.md, "After the run: post-training on the PC"), one GPU,
# detached. the mix is scripts/build_post_sft_mix.py's; the read per save is scripts/reads/read_post_sft_save.sh.
#
# usage: bash scripts/launch_post_sft.sh CONFIG SEED SPLIT RUN [--dry-run]
#   CONFIG  the yaml (ckpts/inject/config_pilot_postsft.yaml for the rehearsal, config_run2_postsft.yaml for the real one)
#   SEED    the checkpoint to train from: a run checkpoint or an export of scripts/export_weights.py
#   SPLIT   the train split in DATA_DIR (post_sft_train)
#   RUN     the run name: ckpts/evidence_RUN, log ckpts/inject/RUN.log
# env: DATA_DIR  where the mix is (default data/prepared_post_pilot; data/prepared_post_run2 for the real one)
#      VAL       the val split (default post_sft_val)
#      FIXED_DIR where evidence_fixed.* sits, linked into DATA_DIR when missing (default data/prepared_pilot; DATA_ROOT/prepared for the real one)
#      TINY_LLM_ROOT, TINY_LLM_ENV_INIT as tests/run_tests.sh
# the same line on an existing run directory resumes it (the checkpoint's own optimizer, data position and rng); a stop leaves
# STOP in the run directory and this script removes it. watch the log with  tail -n 0 -F ckpts/inject/RUN.log
# run it from an open WSL terminal or tmux. called as `wsl ...` from Windows, the trainer is killed when that call returns
# (observed 2026-10-28), setsid and nohup notwithstanding.
# --dry-run prints the command and touches nothing.
DRY=0
for x in "$@"; do [ "$x" = "--dry-run" ] && DRY=1; done
ARGS=()
for x in "$@"; do [ "$x" = "--dry-run" ] || ARGS+=("$x"); done
CONFIG=${ARGS[0]}; SEED=${ARGS[1]}; SPLIT=${ARGS[2]}; RUN=${ARGS[3]}
[ -n "$CONFIG" ] && [ -n "$SEED" ] && [ -n "$SPLIT" ] && [ -n "$RUN" ] || { echo "usage: launch_post_sft.sh CONFIG SEED SPLIT RUN [--dry-run]" >&2; exit 1; }
ROOT=${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}
ENVI=${TINY_LLM_ENV_INIT:-env_init}
DATA_DIR=${DATA_DIR:-data/prepared_post_pilot}
VAL=${VAL:-post_sft_val}
FIXED_DIR=${FIXED_DIR:-data/prepared_pilot}
L=ckpts/inject/$RUN.log
LAUNCH="python scripts/sft.py --evidence --reader-kv -c $SEED --run-name $RUN --data-dir $DATA_DIR --train-split $SPLIT --val-split $VAL"
if [ "$DRY" = 1 ]; then
  echo "cd $ROOT && source $ENVI"
  echo "TINY_LLM_CONFIG=$CONFIG"
  echo "link: $FIXED_DIR/evidence_fixed.* -> $DATA_DIR/ (when missing)"
  echo "$LAUNCH"
  exit 0
fi
cd "$ROOT" || exit 1
source "$ENVI"
export TINY_LLM_CONFIG=$CONFIG
[ -e "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }
[ -e "$SEED" ] || { echo "no such seed: $SEED" >&2; exit 1; }
[ -e "$DATA_DIR/$SPLIT.bin" ] || { echo "no such split: $DATA_DIR/$SPLIT (scripts/build_post_sft_mix.py)" >&2; exit 1; }
mkdir -p ckpts/inject ckpts/evidence_$RUN
rm -f ckpts/evidence_$RUN/STOP
# the trainer reads every split from one data dir
for f in "$FIXED_DIR"/evidence_fixed.*; do
  [ -e "$f" ] && [ ! -e "$DATA_DIR/$(basename "$f")" ] && ln -s "$(readlink -f "$f")" "$DATA_DIR/$(basename "$f")"
done
ls "$DATA_DIR"/evidence_fixed.* > /dev/null 2>&1 || { echo "no evidence_fixed.* in $DATA_DIR and none in $FIXED_DIR to link (set FIXED_DIR); the trainer needs it" >&2; exit 1; }
echo "launch$(date -Is) config $CONFIG seed $SEED split $SPLIT" >> $L
setsid nohup bash -c "$LAUNCH >> $L 2>&1; echo \"trainer exit \$? \$(date -Is)\" >> $L" > /dev/null 2>&1 < /dev/null &
disown
echo "launched, log $L"

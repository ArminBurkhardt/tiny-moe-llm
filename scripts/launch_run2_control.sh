#!/usr/bin/env bash
# the 10B control of the real run (docs/plans/RUN2.md row 8) on a multi GPU node: the same shape and
# schedule, no retrieval, one launch to the end of the 10B split (no switch, no stop).
#
#   DATA_ROOT=/data/run2 bash scripts/launch_run2_control.sh
# print the command and touch nothing: bash scripts/launch_run2_control.sh --dry-run
#
# env: DATA_ROOT    the corpus root of scripts/build_run2_corpus.py (merges in DATA_ROOT/prepared_run2,
#                   held-out splits in DATA_ROOT/prepared); default data/run2
#      NPROC        processes, one per GPU (default 8) on one node; the accumulation is computed as
#                   524,288 / (8 x 4096 x NPROC) and passed as --grad-accumulation-steps (the yaml's 1 is
#                   overridden), so the global batch stays 524,288 tokens; a count that does not divide
#                   16 is refused. the multi node cluster path is scripts/slurm/ (sbatch), not this file
#      SPLIT        the train split (default run2_control_train)
#      SEED         the grounded seed (default ckpts/inject/seed_run2_grounded.pt)
#      UPLOAD       an HF repo id passed as --upload-repo (default none: uploads off)
#      TINY_LLM_ROOT, TINY_LLM_ENV_INIT as tests/run_tests.sh (/dev/null on the rented box)
# the launch is detached (setsid nohup) and writes ckpts/inject/run2_control.log; watch it with
#   tail -n 0 -F ckpts/inject/run2_control.log
# the control reads a plain split with SFTDataset (no --reader-kv, as the pilot's arm C). the accelerate line must match the form in the sft.py docstring ("Data parallel training"); if
# that form changes, change LAUNCH below and tests/test_run2_launchers.py with it.
DRY=0
[ "$1" = "--dry-run" ] && DRY=1
ROOT=${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}
ENVI=${TINY_LLM_ENV_INIT:-env_init}
source "$(cd "$(dirname "$0")" && pwd)/slurm/lib.sh"
DATA_ROOT=${DATA_ROOT:-data/run2}
NPROC=${NPROC:-8}
ACCUM=$(accum_for "$NPROC") || exit 1
SPLIT=${SPLIT:-run2_control_train}
SEED=${SEED:-ckpts/inject/seed_run2_grounded.pt}
RUN=run2_control
D=$DATA_ROOT/prepared_run2
L=ckpts/inject/$RUN.log
EXTRA=""
[ -n "$UPLOAD" ] && EXTRA="$EXTRA --upload-repo $UPLOAD"
LAUNCH="accelerate launch --num_processes $NPROC --mixed_precision no scripts/sft.py --evidence -c $SEED --run-name $RUN --data-dir $D --train-split $SPLIT --val-split inject_val --grad-accumulation-steps $ACCUM$EXTRA"
if [ "$DRY" = 1 ]; then
  echo "cd $ROOT && source $ENVI"
  echo "TINY_LLM_CONFIG=config_run2_control.yaml"
  echo "link: $DATA_ROOT/prepared/evidence_fixed.* -> $D/ (when missing)"
  echo "$LAUNCH"
  exit 0
fi
cd "$ROOT" || exit 1
source "$ENVI"
export TINY_LLM_CONFIG=config_run2_control.yaml
mkdir -p ckpts/inject ckpts/evidence_$RUN
rm -f ckpts/evidence_$RUN/STOP
# the fixed split must sit beside the merges: the trainer reads every split from one data dir
for f in "$DATA_ROOT"/prepared/evidence_fixed.*; do
  [ -e "$f" ] && [ ! -e "$D/$(basename "$f")" ] && ln -s "$(readlink -f "$f")" "$D/$(basename "$f")"
done
echo "launch $(date -Is) split $SPLIT nproc $NPROC" >> $L
setsid nohup bash -c "$LAUNCH >> $L 2>&1; echo \"trainer exit \$? \$(date -Is)\" >> $L" > /dev/null 2>&1 < /dev/null &
disown
echo "launched, log $L"

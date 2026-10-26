#!/usr/bin/env bash
# the real run (docs/plans/RUN2.md) on a multi GPU node: warm-up, then the switch to the main split.
#
# first launch (warm-up split, a stop after the switch save):
#   DATA_ROOT=/data/run2 bash scripts/launch_run2.sh
# after the switch criterion holds (closed book `delta` and gold top-1 reads, RUN2 Schedule), the
# same line on the main split; the run directory ckpts/evidence_run2 resumes from its latest save:
#   DATA_ROOT=/data/run2 SPLIT=run2_main_train bash scripts/launch_run2.sh
# the relaunch passes --switch-split (a relaunch on another split than the checkpoint recorded is
# refused without it). print the command and touch nothing: bash scripts/launch_run2.sh --dry-run
#
# env: DATA_ROOT    the corpus root of scripts/build_run2_corpus.py (merges in DATA_ROOT/prepared_run2,
#                   held-out splits in DATA_ROOT/prepared); default data/run2
#      NPROC        processes, one per GPU (default 8); the per GPU batch is the yaml's, so the global
#                   batch is 8 x 4096 x accumulation 2 x NPROC tokens (524,288 at 8; at another
#                   NPROC edit grad_accumulation_steps in config_run2.yaml to keep 524,288)
#      SPLIT        the train split (default run2_warm_train)
#      STOP_AFTER   regex on the token tag of the save after which STOP is dropped, warm-up only
#                   (default empty: no automatic stop; the warm-up is 2B to 5B, read every 2B,
#                   so pass e.g. STOP_AFTER='4000M' once the read at 4B is wanted)
#      SEED         the grounded seed (default ckpts/inject/seed_run2_grounded.pt)
#      UPLOAD       an HF repo id passed as --upload-repo (default none: uploads off)
#      TINY_LLM_ROOT, TINY_LLM_ENV_INIT as tests/run_tests.sh (/dev/null on the rented box)
# the launch is detached (setsid nohup) and writes ckpts/inject/run2.log; watch it with
#   tail -n 0 -F ckpts/inject/run2.log
# the accelerate line must match the form in the sft.py docstring ("Data parallel training"); if
# that form changes, change LAUNCH below and tests/test_run2_launchers.py with it.
DRY=0
[ "$1" = "--dry-run" ] && DRY=1
ROOT=${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}
ENVI=${TINY_LLM_ENV_INIT:-env_init}
DATA_ROOT=${DATA_ROOT:-data/run2}
NPROC=${NPROC:-8}
SPLIT=${SPLIT:-run2_warm_train}
SEED=${SEED:-ckpts/inject/seed_run2_grounded.pt}
RUN=run2
D=$DATA_ROOT/prepared_run2
L=ckpts/inject/$RUN.log
EXTRA=""
[ "$SPLIT" != run2_warm_train ] && EXTRA="$EXTRA --switch-split"
[ -n "$UPLOAD" ] && EXTRA="$EXTRA --upload-repo $UPLOAD"
LAUNCH="accelerate launch --num_processes $NPROC --mixed_precision no scripts/sft.py --evidence --reader-kv -c $SEED --run-name $RUN --data-dir $D --train-split $SPLIT --val-split inject_val$EXTRA"
if [ "$DRY" = 1 ]; then
  echo "cd $ROOT && source $ENVI"
  echo "TINY_LLM_CONFIG=config_run2.yaml"
  echo "link: $DATA_ROOT/prepared/evidence_fixed.* -> $D/ (when missing)"
  echo "$LAUNCH"
  [ -n "$STOP_AFTER" ] && echo "bash scripts/reads/stop_after_save_run2.sh $RUN '$STOP_AFTER'"
  exit 0
fi
cd "$ROOT" || exit 1
source "$ENVI"
export TINY_LLM_CONFIG=config_run2.yaml
mkdir -p ckpts/inject ckpts/evidence_$RUN
rm -f ckpts/evidence_$RUN/STOP
# the fixed split must sit beside the merges: the trainer reads every split from one data dir
for f in "$DATA_ROOT"/prepared/evidence_fixed.*; do
  [ -e "$f" ] && [ ! -e "$D/$(basename "$f")" ] && ln -s "$(readlink -f "$f")" "$D/$(basename "$f")"
done
echo "launch $(date -Is) split $SPLIT nproc $NPROC" >> $L
if [ -n "$STOP_AFTER" ]; then
  setsid nohup bash scripts/reads/stop_after_save_run2.sh $RUN "$STOP_AFTER" >> $L 2>&1 < /dev/null &
fi
setsid nohup bash -c "$LAUNCH >> $L 2>&1; echo \"trainer exit \$? \$(date -Is)\" >> $L" > /dev/null 2>&1 < /dev/null &
disown
echo "launched, log $L"

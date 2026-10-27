#!/usr/bin/env bash
# the body of the training sbatch scripts (train_run2.sbatch, train_run2_control.sbatch): sourced by them with
# MODE=run|control, JOB_SCRIPT (the sbatch script's path under the repo) and the job's own variables set.
# one job = one srun of one process per GPU over NODES x GPUS_PER_NODE GPUs. the chain: scripts/slurm/submit_chain.sh
# queued every job from the login node (afterany on the one before, one chain id), so a job queues nothing itself
# (CHAIN_MODE=self is the old in-job successor). a job takes the run directory lock first (one trainer per run
# directory), checks its chain id against the run chain state, runs the precheck, then the srun. the signal USR1
# twenty minutes before the time limit (or the squeue based timer) makes it ask the ranks to stop through
# `scancel --signal=TERM` on the step (never a signal to srun itself: srun treats SIGTERM as a forced termination and
# may SIGKILL the step, which would lose the save), the ranks save and exit 20, and at the end it decides from files
# and rank 0 exit code whether the chain goes on or ends (chain_state ended=..., HALT with the reason).
# env (all optional; Slurm's own values win inside a job):
#   NODES, GPUS_PER_NODE  4 and 4 outside Slurm (the dry run); the job gets them from the allocation
#   SPLIT       train split (run: run2_warm_train; main: run2_main_train adds --switch-split; control: run2_control_train)
#   RUN         run name (default run2 or run2_control); a different name makes a separate run directory (the 2 GPU check)
#   SEED        seed checkpoint (default ckpts/inject/seed_run2_grounded.pt)
#   STOP_AFTER  the token tag after whose first save at or past it STOP is dropped (warm-up reads): 4000M, 4B or a number
#               of tokens, as stop_after_save_run2.sh (numeric, a save lands a log step past its grid point)
#   MAX_JOBS    chain length cap (default 20)
#   CPUS_PER_TASK  default 20 (80 cores over 4 GPUs)
#   TRAIN_EXTRA extra sft.py flags, appended as given. PLAN_SPLIT_FLAG and WARM_EPOCH_FLAG name the two flags every run launch passes
#               (--plan-split run2_main_train; the warm-up also --max-epochs-this-split 1)
#   CHAIN_MODE  presubmitted (default) or self (this job queues its successor before it starts)
#   STALL_MIN 15, SAVE_MIN (derived: max(45, 2.5 save intervals at 54k tokens per second per GPU), 49 at 16 GPUs, 97 at 8),
#   STARTUP_MIN 30, LOSS_CEILING 20, DIVERGE_K 3, RISE_K 30, RISE_MARGIN 1.0, RISE_POLLS 5, RISE_MIN 15: the progress
#   watchdog (watchdog.sh); CKPT_EVERY_TOKENS overrides the save interval read from the config (1B)
#   DRY_RUN=1   print the plan and the srun line, touch nothing
#   CHAIN_INDEX, CHAIN_ID, CHAIN_FIRST  set by submit_chain.sh; unset on a manual sbatch (index 0: clears STOP, HALT and
#               the chain state; without a chain id the id check is skipped for a later index)
# assumption behind the 20 minute lead: a log step is seconds, a save is 4.6 GB to the project disk (a minute or
# two), a save every 1B tokens at about 1M tokens per second (16 minutes) is the cadence the rolling saves keep.
MODE=${MODE:-run}
SUBMIT=${SLURM_SUBMIT_DIR:-$PWD}
# shellcheck disable=SC1091
source "$SUBMIT/scripts/slurm/mn5_env.sh"
if [ "$MODE" = control ]; then
  RUN=${RUN:-run2_control}; CONFIG=config_run2_control.yaml; SPLIT=${SPLIT:-run2_control_train}; KV=""
else
  RUN=${RUN:-run2}; CONFIG=config_run2.yaml; SPLIT=${SPLIT:-run2_warm_train}; KV=" --reader-kv"
fi
# a failure that every later job of the chain would repeat (the environment, a bad variable, a missing cache) writes a
# HALT with the reason before it exits, so the chain's next jobs skip at once (the precheck honours HALT) and the
# supervisor reports it instead of resubmitting for ever. the run directory is under REPO; when REPO itself does not
# resolve the HALT goes under the submit directory (the repo root of a normal submission). a dry run touches nothing
halt_early() {
  local base=$REPO d
  [ -d "$base" ] || base=$SUBMIT
  d=$base/ckpts/evidence_$RUN
  echo "job ${SLURM_JOB_ID:-0}: not started: $1" >&2
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  mkdir -p "$d" 2>/dev/null && echo "job ${SLURM_JOB_ID:-0} (before the trainer started): $1" > "$d/HALT"
  return 0
}
cd "$REPO" || { halt_early "cannot cd to REPO=$REPO"; exit 2; }
source scripts/slurm/lib.sh
[ -z "$CTR_FAIL" ] || { halt_early "SIF=$SIF is set but no singularity or apptainer was found (module load singularity in mn5_env.local.sh)"; exit 2; }
if [ -n "$STOP_AFTER" ]; then
  stop_after_m "$STOP_AFTER" >/dev/null || { halt_early "STOP_AFTER='$STOP_AFTER' is not 4000M, 4B or a token count"; exit 2; }
fi

NODES=${SLURM_JOB_NUM_NODES:-${NODES:-2}}
GPN=${SLURM_NTASKS_PER_NODE:-${GPUS_PER_NODE:-4}}
GPUS=$(( NODES * GPN ))
ACCUM=$(accum_for "$GPUS") || { halt_early "$GPUS GPUs do not divide the global batch (accum_for)"; exit 2; }
CPUS=${CPUS_PER_TASK:-20}
SEED=${SEED:-ckpts/inject/seed_run2_grounded.pt}
D=$DATA_ROOT/prepared_run2
# the watchdog's save limit scales with the save interval and the job's GPU count: 2.5 intervals at the throughput floor,
# never under 45 minutes (2 nodes take about 39 minutes per 1B tokens, plus the sharded evals). the interval comes from
# the config's evidence block (the last checkpoint_every_tokens line), 1B when it cannot be read
CKPT_TOKENS=${CKPT_EVERY_TOKENS:-$(grep -o 'checkpoint_every_tokens: *[0-9]*' "$CONFIG" 2>/dev/null | tail -n 1 | grep -o '[0-9]*$')}
CKPT_TOKENS=${CKPT_TOKENS:-1000000000}
export SAVE_MIN=${SAVE_MIN:-$(save_min_for "$GPUS" "$CKPT_TOKENS")}
export STARTUP_MIN=${STARTUP_MIN:-30}
# the precheck has already handled a STOP present at submit time; one that appears during the minutes of setup is the
# user's and the trainer honours it (exit 10) instead of clearing it
EXTRA=" --keep-stop"
[ "$MODE" != control ] && [ "$SPLIT" != run2_warm_train ] && EXTRA="$EXTRA --switch-split"
# the trainer review's contracts: every launch of the main run spans the LR schedule over the whole run (--plan-split), a
# warm-up launch stops at its split's epoch end instead of wrapping (--max-epochs-this-split, log line "warm split
# exhausted"); the control passes neither. flag names are overridable until sft.py's are final
if [ "$MODE" != control ]; then
  EXTRA="$EXTRA ${PLAN_SPLIT_FLAG---plan-split run2_main_train}"
  [ "$SPLIT" = run2_warm_train ] && EXTRA="$EXTRA ${WARM_EPOCH_FLAG---max-epochs-this-split 1}"
fi
[ -n "$TRAIN_EXTRA" ] && EXTRA="$EXTRA $TRAIN_EXTRA"
RUNDIR=ckpts/evidence_$RUN
RUNLOG=ckpts/inject/$RUN.log
LOGDIR=$REPO/logs
JOBID=${SLURM_JOB_ID:-0}
INDEX=${CHAIN_INDEX:-0}
MAXJOBS=${MAX_JOBS:-20}
CHAIN_MODE=${CHAIN_MODE:-presubmitted}
HIST=$RUNDIR/chain_history
STATE=$RUNDIR/chain_state
TRAIN="${CTR}python scripts/sft.py --evidence$KV -c $SEED --run-name $RUN --data-dir $D --train-split $SPLIT --val-split inject_val --grad-accumulation-steps $ACCUM$EXTRA"
# one task per GPU; --gpu-bind=none so every task sees the node's 4 GPUs and LOCAL_RANK picks the device (a site default
# that binds one GPU per task would leave LOCAL_RANK 1 to 3 without a device)
SRUN="srun --nodes=$NODES --ntasks=$GPUS --ntasks-per-node=$GPN --gpus-per-node=$GPN --gpu-bind=none --cpus-per-task=$CPUS --wait=300 -u --output=$RUNLOG --open-mode=append bash scripts/slurm/rank_exec.sh $LOGDIR/rc_$JOBID $TRAIN"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "mode $MODE run $RUN nodes $NODES gpus $GPUS accum $ACCUM chain_index $INDEX"
  echo "TINY_LLM_CONFIG=$CONFIG"
  echo "MASTER_ADDR=<first node> MASTER_PORT=<20000 + job id % 20000>"
  echo "$SRUN"
  echo "chain_mode $CHAIN_MODE"
  echo "watchdog save_min $SAVE_MIN startup_min $STARTUP_MIN"
  [ -n "$STOP_AFTER" ] && echo "bash scripts/reads/stop_after_save_run2.sh $RUN '$STOP_AFTER'"
  exit 0
fi
SITE_ERR=$(require_site_values 2>&1) || { halt_early "$SITE_ERR"; exit 2; }

say() { echo "[$(date -Is)] job $JOBID: $*" | tee -a "$RUNLOG"; }
mkdir -p "$LOGDIR" ckpts/inject "$RUNDIR"
marker() { [ -f "$RUNDIR/run_state.json" ] && md5sum "$RUNDIR/run_state.json" | cut -d' ' -f1 || echo none; }
flag() { [ -e "$1" ] && echo 1 || echo 0; }
FINALF=$RUNDIR/checkpoint_evidence_final.pt
RC=255; WDV=ok; DECISION=""; NEXT=""
halt_chain() {
  echo "job $JOBID rc $RC watchdog $WDV: $1" > "$RUNDIR/HALT"
  state_put "$STATE" ended halted
  [ -n "$NEXT" ] && timeout 30 scancel "$NEXT"
  say "chain halted: $1; read the log, then rm $RUNDIR/HALT and submit a fresh chain"
}

# one trainer per run directory: a second job (a stale one of an old chain, a doubled submission) exits here, before it
# touches the chain state. the lock is held by the batch shell and its children (srun) for the whole job
exec 9>"$RUNDIR/.trainer.lock"
flock -n -E 75 9
LRC=$?
if [ "$LRC" = 75 ]; then say "not running: another trainer holds the lock of $RUNDIR (one trainer per run directory)"; exit 0; fi
if [ "$LRC" != 0 ]; then
  say "cannot take the lock of $RUNDIR (flock exit $LRC); the run directory file system must support flock"
  halt_early "cannot take the lock of $RUNDIR (flock exit $LRC); the run directory file system must support flock"
  exit 2
fi

if [ "$INDEX" = 0 ]; then
  rm -f "$RUNDIR/STOP" "$RUNDIR/HALT" "$STATE" "$HIST"
fi
# the chain id: index 0 and the first job of a resume (CHAIN_FIRST) write theirs into the state, every other job must
# carry the same one, so a surplus job of a chain that was replaced exits at once
if [ "$INDEX" = 0 ] && [ -z "$CHAIN_ID" ]; then CHAIN_ID=job$JOBID; fi
if [ "$INDEX" = 0 ] || [ "${CHAIN_FIRST:-0}" = 1 ]; then
  [ -n "$CHAIN_ID" ] && state_put "$STATE" chain_id "$CHAIN_ID"
  # the lowest job id of this chain: status.py ignores the exit codes of older jobs (a replaced chain's rc 30)
  state_put "$STATE" chain_first_job "$JOBID"
elif [ -n "$CHAIN_ID" ]; then
  CUR_ID=$(state_get "$STATE" chain_id)
  if [ "$CUR_ID" != "$CHAIN_ID" ]; then
    say "not running: this job belongs to chain $CHAIN_ID, the run chain is ${CUR_ID:-unknown} (a replaced chain); no successor queued"
    exit 0
  fi
fi
M0=$(marker)
ENDED=$(state_get "$STATE" ended)
if [ "$INDEX" -gt 0 ] && [ -n "$ENDED" ]; then
  say "not running: the chain ended ($ENDED); no successor queued"
  exit 0
fi
verdict=$(chain_precheck "$(flag "$FINALF")" "$(flag "$RUNDIR/STOP")" "$(flag "$RUNDIR/HALT")" "$M0" \
  "$(state_get "$STATE" last_marker)" "$(state_get "$STATE" stalls)" "$INDEX" "$MAXJOBS")
set -- $verdict
if [ "$1" = skip ]; then
  case "$2" in
    done) state_put "$STATE" ended done;;
    stop-file) state_put "$STATE" ended stop;;
    halted) state_put "$STATE" ended halted;;
    *) state_put "$STATE" ended "$2"
       [ -e "$RUNDIR/HALT" ] || echo "job $JOBID: chain stopped by the precheck: $2 (chain index $INDEX); read $RUNLOG" > "$RUNDIR/HALT";;
  esac
  say "not running: $2 (chain index $INDEX); no successor queued"
  exit 0
fi
STALLS0=$2
state_put "$STATE" last_marker "$M0"
state_put "$STATE" stalls "$STALLS0"
state_put "$STATE" run "$RUN"
state_put "$STATE" mode "$MODE"
state_put "$STATE" split "$SPLIT"
state_put "$STATE" nodes "$NODES"
state_put "$STATE" gpus "$GPUS"
state_put "$STATE" stop_after "$STOP_AFTER"
state_put "$STATE" save_min "$SAVE_MIN"
state_put "$STATE" startup_min "$STARTUP_MIN"

if [ "$CHAIN_MODE" = self ] && [ -n "$JOB_SCRIPT" ] && [ "$JOBID" != 0 ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:$JOBID --job-name="$SLURM_JOB_NAME" --nodes=$NODES --ntasks-per-node=$GPN --gres=gpu:$GPN \
    --export=ALL,CHAIN_INDEX=$((INDEX + 1)),CHAIN_FIRST=0 "$REPO/$JOB_SCRIPT") || NEXT=""
  say "successor queued: ${NEXT:-FAILED to queue}"
fi

export TINY_LLM_CONFIG=$CONFIG
# a dead rank must kill the job, not hang it: NCCL errors abort the process group, and sft.py collectives time out
# after 30 minutes (InitProcessGroupKwargs); the heartbeat watchdog catches a hung collective on the way
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-900}
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + JOBID % 20000 ))
for f in "$DATA_ROOT"/prepared/evidence_fixed.*; do
  [ -e "$f" ] && [ ! -e "$D/$(basename "$f")" ] && ln -s "$(readlink -f "$f")" "$D/$(basename "$f")"
done
rm -f "$LOGDIR"/rc_$JOBID.rank*
say "start: mode $MODE split $SPLIT nodes $NODES gpus $GPUS accum $ACCUM chain index $INDEX chain $CHAIN_ID stalls $STALLS0 marker $M0"
T0=$(date +%s)

WATCH=""
if [ -n "$STOP_AFTER" ]; then
  FROM=$(stat -c %s "$RUNLOG" 2>/dev/null || echo 0)
  FROM=$FROM LOG=$RUNLOG bash scripts/reads/stop_after_save_run2.sh "$RUN" "$STOP_AFTER" >> "$RUNLOG" 2>&1 &
  WATCH=$!
fi

FROM_OFF=$(stat -c %s "$RUNLOG" 2>/dev/null || echo 0)
rm -f "$LOGDIR/wd_$JOBID"
KILLER=""
SRUN_PID=""
CANCELLED=0
# the save path never signals srun: srun treats SIGTERM as a forced termination and may SIGKILL the step (no save, no
# rc file). scancel --signal=TERM on the step reaches every task rank_exec.sh, which forwards it to the trainer
terminate() {
  timeout 30 scancel --signal=TERM "$(train_step_id "$JOBID")" 2>/dev/null || say "scancel --signal=TERM on the step failed"
}
# the kill path (a hung step, a diverged loss): the step first, then srun
hard_kill() {
  timeout 30 scancel --signal=KILL "$(train_step_id "$JOBID")" 2>/dev/null
  [ -n "$SRUN_PID" ] && kill -KILL "$SRUN_PID" 2>/dev/null
}
on_usr1() { say "time limit near: SIGTERM to the trainer through the step, it saves and exits 20"; terminate; }
on_usr2() {
  local v
  v=$(cut -d' ' -f1 "$LOGDIR/wd_$JOBID" 2>/dev/null)
  if [ "$v" = diverged ]; then
    say "watchdog: $(cat "$LOGDIR/wd_$JOBID" 2>/dev/null); SIGKILL to the step with no save (a save would write the diverged state)"
    hard_kill
  else
    say "watchdog: $(cat "$LOGDIR/wd_$JOBID" 2>/dev/null); SIGTERM to the ranks, kill in 180 s"
    terminate
    ( sleep "${KILL_AFTER_S:-180}"; hard_kill ) &
    KILLER=$!
  fi
}
# a cancelled or limit killed job still leaves its history line; Slurm gives KillWait seconds, so the line is written
# first and the ranks asked to save after
on_term() {
  CANCELLED=1
  # a TERM with much of the time limit left is a user's scancel: the chain ends (ended=cancelled, so the afterany
  # successors exit at once and the supervisor, which never restarts a chain a person stopped, only alerts). a TERM in
  # the last minutes is the time limit, and the chain goes on. an unreadable squeue leaves the chain state alone
  local left
  if [ "$JOBID" != 0 ]; then
    left=$(slurm_time_left_seconds "$(timeout 30 squeue -h -j "$JOBID" -o %L 2>/dev/null)") || left=""
    if [ -n "$left" ] && [ "$left" -gt "${CANCEL_MIN_LEFT_S:-300}" ]; then
      state_put "$STATE" ended cancelled
      say "cancelled by hand ($left s of the limit left): the chain ends and is not resubmitted"
    fi
  fi
  hist_append "$HIST" "$JOBID" cancelled "$(( $(date +%s) - T0 ))" cancelled
  say "SIGTERM to the batch shell (scancel or the limit): history line written, SIGTERM to the ranks (save, exit 20)"
  terminate
}
trap on_usr1 USR1
trap on_usr2 USR2
trap on_term TERM

# the watchdog first: a trainer without divergence and stall detection is not started
WD=""
if [ "${WATCHDOG:-1}" = 1 ] && [ "$JOBID" != 0 ]; then
  LOG=$RUNLOG RUNDIR=$RUNDIR JOBID=$JOBID LOGDIR=$LOGDIR START=$T0 FROM=$FROM_OFF BATCH_PID=$$ bash scripts/slurm/watchdog.sh >> "$RUNLOG" 2>&1 &
  WD=$!
  sleep "${WD_CHECK_S:-3}"
  if ! kill -0 "$WD" 2>/dev/null; then
    say "FATAL: the progress watchdog did not start (see $RUNLOG); the trainer is not started"
    halt_chain "the progress watchdog did not start"
    [ -n "$WATCH" ] && kill "$WATCH" 2>/dev/null
    exit 3
  fi
fi

$SRUN &
SRUN_PID=$!
# a save timer for a site that does not honour --signal: the time left comes from squeue
TIMER=""
if [ "$JOBID" != 0 ]; then
  LEFT=$(slurm_time_left_seconds "$(timeout 30 squeue -h -j "$JOBID" -o %L 2>/dev/null)") || LEFT=""
  if [ -n "$LEFT" ] && [ "$LEFT" -gt 1500 ]; then
    ( sleep $(( LEFT - 1200 )); kill -USR1 $$ 2>/dev/null ) &
    TIMER=$!
  fi
fi
while kill -0 "$SRUN_PID" 2>/dev/null; do wait "$SRUN_PID"; done
# ignored, not reset: a signal arriving while the sidecars are killed must not take the default action (kill the shell)
trap '' USR1 USR2
for p in $WD $TIMER $KILLER; do kill "$p" 2>/dev/null; done
[ -n "$WATCH" ] && kill "$WATCH" 2>/dev/null
[ -f "$LOGDIR/wd_$JOBID" ] && WDV=$(cut -d' ' -f1 "$LOGDIR/wd_$JOBID")

[ -f "$LOGDIR/rc_$JOBID.rank0" ] && RC=$(cat "$LOGDIR/rc_$JOBID.rank0")
M1=$(marker)
if [ "$M1" = "$M0" ]; then STALLS1=$(( STALLS0 + 1 )); else STALLS1=0; fi
DECISION=$(chain_decide "$RC" "$(flag "$FINALF")" "$(flag "$RUNDIR/STOP")" "$STALLS1")
REASON=""
# lines of this job that explain an exit 10 or 30 without a STOP file: the trainer kill rule, a warm split that ran out
# (needs the switch decision, the chain ends like a stop), a refused switch or resume (exit 30)
NEWLOG=$(tail -c +$(( FROM_OFF + 1 )) "$RUNLOG" 2>/dev/null)
KILLLINE=$(echo "$NEWLOG" | grep -m 1 'KILL:' | cut -c1-200)
EXHAUSTED=$(echo "$NEWLOG" | grep -m 1 'warm split exhausted' | cut -c1-200)
if [ "$DECISION" = stop ] && [ "$(flag "$RUNDIR/STOP")" = 0 ]; then
  if [ -z "$EXHAUSTED" ] && [ -n "$KILLLINE" ]; then
    DECISION=fail; REASON="the trainer kill rule fired: $KILLLINE"
  fi
fi
if [ "$RC" = 30 ]; then
  LASTERR=$(echo "$NEWLOG" | grep -i -E 'error|refus|verify|mismatch' | tail -n 1 | cut -c1-200)
  REASON="exit 30 (a refused resume or switch): ${LASTERR:-see $RUNLOG}"
fi
# exit 2 is the trainer refusing to start (a missing rows cache, a bad argument): every rerun would repeat it, so the
# chain halts at once instead of burning the crash loop allowance
if { [ "$RC" = 1 ] || [ "$RC" = 2 ]; } && [ "$DECISION" = fail ]; then
  LASTERR=$(echo "$NEWLOG" | grep -i -E 'error|rows cache|refus|not found|missing|mismatch' | tail -n 1 | cut -c1-200)
  if [ "$RC" = 2 ]; then
    REASON="trainer exit 2, rows cache missing: run the rows stage (scripts/sft.py --evidence --estimate-rows-only --estimate-world-sizes $GPUS ...); deterministic, not rerun: ${LASTERR:-see $RUNLOG}"
  else
    REASON="trainer exit 1 (a refusal such as a seed mismatch, a missing evidence_fixed or a refused head; deterministic, not rerun): ${LASTERR:-see $RUNLOG}"
  fi
fi
[ "$WDV" = diverged ] && DECISION=diverged
ELAPSED=$(( $(date +%s) - T0 ))
JSTATE=$(job_state "$RC" "$WDV")
[ "$DECISION" = nofinal ] && JSTATE=failed
[ "$CANCELLED" = 1 ] || hist_append "$HIST" "$JOBID" "$JSTATE" "$ELAPSED" "$DECISION"
state_put "$STATE" last_job_state "$JSTATE"
say "trainer rank 0 exit $RC after ${ELAPSED}s, new save: $([ "$M1" != "$M0" ] && echo yes || echo no), watchdog $WDV, decision $DECISION"
if [ "$DECISION" = resubmit ]; then
  CL=$(crashloop_check "$HIST")
  case "$CL" in halt*) DECISION=fail; REASON=${CL#halt };; esac
fi
# the GPU hours line carries the final decision
echo "$JOBID $GPUS $ELAPSED $(awk -v g=$GPUS -v s=$ELAPSED 'BEGIN{printf "%.1f", g*s/3600}') rc=$RC $DECISION" >> "ckpts/inject/${RUN}_gpu_hours.log"
case "$DECISION" in
  resubmit) ;;
  fail) halt_chain "${REASON:-rank 0 exit $RC, $STALLS1 job(s) without a new save}";;
  nofinal) halt_chain "rank 0 exited 0 but $FINALF does not exist (not finished)";;
  diverged) halt_chain "diverged: $(cut -d' ' -f2- "$LOGDIR/wd_$JOBID")";;
  *)
    if [ "$DECISION" = stop ] && [ -n "$EXHAUSTED" ]; then
      echo "job $JOBID: warm split exhausted, the switch decision is needed ($EXHAUSTED)" > "$RUNDIR/HALT"
    fi
    ENDVAL=$DECISION
    [ "$DECISION" = stop ] && [ -n "$EXHAUSTED" ] && ENDVAL=warm-exhausted
    state_put "$STATE" ended "$ENDVAL"
    [ -n "$NEXT" ] && timeout 30 scancel "$NEXT"
    say "chain ends: $ENDVAL"
    ;;
esac
exit 0

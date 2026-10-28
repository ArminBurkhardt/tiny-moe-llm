#!/usr/bin/env bash
#SBATCH --job-name=run2_corpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=71:50:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=logs/%x_%j.out
# the corpus driver (scripts/build_run2_corpus.py, docs/plans/RUN2_corpus.md) on one whole node (--exclusive, --mem=0, all
# 80 CPUs: the chain builders need up to about 124 GB each and nodes bill whole anyway). submit it through
# scripts/slurm/submit_chain.sh CORPUS N (STAGES=... in the environment), which names the job run2_corpus_<stages>,
# chains N jobs and gives them one chain id. by hand from the repo root after `source scripts/slurm/mn5_env.sh` and
# `mkdir -p logs`:
#   STAGES=bios_main,bios_warm sbatch --job-name=run2_corpus_bios_main_bios_warm scripts/slurm/corpus.sbatch   GPU stages
#   STAGES=merge_warm,merge_main DEVICE=cpu sbatch -p CPU_PARTITION --gres=none --cpus-per-task=112 scripts/slurm/corpus.sbatch
# the CPU form is for the stages that need no GPU (merges) if the site has a CPU partition; otherwise use the GPU
# form. no node, login nodes included, has outgoing internet, so the filler stage reads its shards through the mirror: the
# PC downloads them in consumption order and pushes them through the transfer nodes into an inbox while this job runs
#   STAGES=filler CORPUS_ARGS="--mirror-inbox $DATA_ROOT/inbox --mirror-listing $DATA_ROOT/listing.json --offline-stage $HF_HOME" \
#     bash scripts/slurm/submit_chain.sh CORPUS 3
# with `python scripts/mirror_sources.py push --listing listing.json --remote USER@transfer1.bsc.es:$DATA_ROOT/inbox` running on
# the PC (recipe: scripts/slurm/README.md, "The filler through the mirror").
# env: STAGES (default all), DEVICE (cuda), MAX_JOBS (30), CORPUS_ARGS (extra driver flags), DRY_RUN=1 (print only),
#      CHAIN_MODE (presubmitted by default; self queues the successor from the job),
#      STALL_MIN (120: a builder is stopped, SIGTERM then kill after 3 minutes, and marked halted, when
#      $DATA_ROOT/run2_logs/last_child_output (the driver touches it when a child prints a line or a stage starts or ends;
#      its own 300 s heartbeat does not) is older than that; until the file exists the log and DATA_ROOT file times are used.
#      md5, model loads and merge checks are silent for up to an hour),
#      MIRROR_WAIT_MAX (120: a log whose last line is a mirror "waiting Ns" line with N over that many minutes means the
#      pushed inbox has stopped feeding the driver, the mirror logs every 300 s so the file looks alive; stopped and
#      halted with its own reason).
# the driver is idempotent with stage markers and exits 20 on SIGTERM: this job sends it SIGTERM on the USR1 signal or
# on a cancel, the chain then goes on until the driver exits 0; an exit 1 or any other code ends the chain (no blind
# rerun). one driver per stage group: the job takes $DATA_ROOT/slurm_state/<stages>.lock, and a job whose chain id is
# not the one in <stages>.chain (a replaced chain) exits at once.
SUBMIT=${SLURM_SUBMIT_DIR:-$PWD}
source "$SUBMIT/scripts/slurm/mn5_env.sh"
cd "$REPO" || exit 2
source scripts/slurm/lib.sh
[ -z "$CTR_FAIL" ] || exit 2
STAGES=${STAGES:-all}
DEVICE=${DEVICE:-cuda}
INDEX=${CHAIN_INDEX:-0}
MAXJOBS=${MAX_JOBS:-30}
JOBID=${SLURM_JOB_ID:-0}
CHAIN_MODE=${CHAIN_MODE:-presubmitted}
STALL_MIN=${STALL_MIN:-120}
MIRROR_WAIT_MAX=${MIRROR_WAIT_MAX:-120}
TAG=$(echo "$STAGES" | tr ',' '_')
ST=$DATA_ROOT/slurm_state
CMD="${CTR}python scripts/build_run2_corpus.py build --data-root $DATA_ROOT --stage $STAGES --device $DEVICE $CORPUS_ARGS"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "stages $STAGES device $DEVICE chain_index $INDEX"
  echo "$CMD"
  exit 0
fi
require_site_values || exit 2
mkdir -p logs "$ST"
LOG=logs/corpus_$TAG.log
say() { echo "[$(date -Is)] job $JOBID: $*" | tee -a "$LOG"; }

exec 9>"$ST/$TAG.lock"
flock -n -E 75 9
LRC=$?
if [ "$LRC" = 75 ]; then say "not running: another driver holds $ST/$TAG.lock (one driver per stage group)"; exit 0; fi
if [ "$LRC" != 0 ]; then say "cannot take $ST/$TAG.lock (flock exit $LRC)"; exit 2; fi

if [ "$INDEX" = 0 ]; then rm -f "$ST/$TAG.halt" "$ST/$TAG.done" "$ST/$TAG.stallreason"; fi
if [ "$INDEX" = 0 ] && [ -z "$CHAIN_ID" ]; then CHAIN_ID=job$JOBID; fi
if [ "$INDEX" = 0 ] || [ "${CHAIN_FIRST:-0}" = 1 ]; then
  [ -n "$CHAIN_ID" ] && echo "$CHAIN_ID" > "$ST/$TAG.chain"
elif [ -n "$CHAIN_ID" ] && [ "$(cat "$ST/$TAG.chain" 2>/dev/null)" != "$CHAIN_ID" ]; then
  say "not running: chain $CHAIN_ID is not the current chain of $TAG ($(cat "$ST/$TAG.chain" 2>/dev/null)); no successor queued"
  exit 0
fi
if [ -e "$ST/$TAG.done" ] || [ -e "$ST/$TAG.halt" ] || [ "$INDEX" -ge "$MAXJOBS" ]; then
  [ "$INDEX" -ge "$MAXJOBS" ] && [ ! -e "$ST/$TAG.done" ] && [ ! -e "$ST/$TAG.halt" ] && echo "job cap reached (chain index $INDEX of $MAXJOBS)" > "$ST/$TAG.halt"
  say "not running (done, halted or at the job cap, chain index $INDEX); no successor queued"
  exit 0
fi
NEXT=""
if [ "$CHAIN_MODE" = self ] && [ "$JOBID" != 0 ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:$JOBID --job-name="$SLURM_JOB_NAME" --export=ALL,CHAIN_INDEX=$((INDEX + 1)),CHAIN_FIRST=0 \
    "$REPO/scripts/slurm/corpus.sbatch") || NEXT=""
  say "successor queued: ${NEXT:-FAILED to queue}"
fi
say "start: $CMD"
T0=$(date +%s)
PID=""
KILLER=""
SIG=0
STALLED=0
on_usr1() { SIG=1; say "time limit near: SIGTERM to the driver"; [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null; }
on_term() { SIG=1; say "SIGTERM to the batch shell (scancel or the limit): SIGTERM to the driver"; [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null; }
on_usr2() {
  SIG=1
  STALLED=1
  say "watchdog: $(cat "$ST/$TAG.stallreason" 2>/dev/null); SIGTERM to the driver, kill in 180 s"
  [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null
  ( sleep "${KILL_AFTER_S:-180}"; kill -KILL "$PID" 2>/dev/null ) &
  KILLER=$!
}
trap on_usr1 USR1
trap on_usr2 USR2
trap on_term TERM
$CMD >> "$LOG" 2>&1 &
PID=$!
(
  while kill -0 "$PID" 2>/dev/null; do
    sleep "${WD_POLL_S:-300}"
    if [ "$(corpus_mirror_wait "$LOG" "$MIRROR_WAIT_MAX")" = mirror-stalled ]; then
      echo "mirror wait: the last log line is a mirror waiting line older than ${MIRROR_WAIT_MAX} minutes (the push of the filler inbox stopped)" > "$ST/$TAG.stallreason"
      kill -USR2 $$; exit 0
    fi
    if [ "$(corpus_stalled "$LOG" "$DATA_ROOT" "$STALL_MIN")" = stalled ]; then
      echo "stalled: no child output (run2_logs/last_child_output; the log and DATA_ROOT files before it exists) for ${STALL_MIN} minutes" > "$ST/$TAG.stallreason"
      kill -USR2 $$; exit 0
    fi
  done
) &
WD=$!
# a trapped signal interrupts wait with 128+N whether or not the driver ended; wait again then, which returns its own
# status even when it ended in between (the exit code is never the trap's)
RC=255
while :; do
  SIG=0
  wait "$PID"
  r=$?
  [ "$r" = 127 ] && break
  RC=$r
  [ "$SIG" = 1 ] && [ "$r" -gt 128 ] && continue
  break
done
trap '' USR1 USR2
for p in $WD $KILLER; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
D=$(corpus_decide "$RC")
[ "$STALLED" = 1 ] && D=stalled
say "driver exit $RC after $(( $(date +%s) - T0 ))s, decision $D"
case "$D" in
  resubmit) ;;
  done) touch "$ST/$TAG.done"; [ -n "$NEXT" ] && scancel "$NEXT";;
  stalled) cat "$ST/$TAG.stallreason" > "$ST/$TAG.halt" 2>/dev/null || echo "stalled" > "$ST/$TAG.halt"; [ -n "$NEXT" ] && scancel "$NEXT"; say "chain stopped: the builder stalled; read $LOG, fix, then submit fresh";;
  *) echo "driver exit $RC" > "$ST/$TAG.halt"; [ -n "$NEXT" ] && scancel "$NEXT"; say "chain stopped: read $LOG, fix, then submit fresh";;
esac
exit 0

#!/usr/bin/env bash
# submit a chain of N dependent jobs from the login node: bash scripts/slurm/submit_chain.sh TRAIN|CONTROL|CORPUS N
# each job is queued with --dependency=afterany on the one before, CHAIN_INDEX and one CHAIN_ID set, so the chain exists
# before the first job starts and no job needs to call sbatch from a compute node. a surplus job (the run is done,
# stopped, halted, stalled twice) runs the precheck in train_job.sh / corpus.sbatch and exits at once: about 10 to 30
# seconds of the allocation, so 16 GPUs bill about 0.05 to 0.15 GPU hours per surplus job. the job queues nothing itself
# unless CHAIN_MODE=self.
# every job gets an explicit name: the run name for TRAIN and CONTROL (run2, run2_control, or RUN), run2_corpus_<stages>
# for CORPUS. the submit REFUSES while squeue lists a job of that name (a chain is alive or still pending: a second
# chain would put two trainers on one run directory) and when squeue itself fails (an unreachable slurmctld says
# nothing about the queue).
# env: START_INDEX (default 0: a fresh chain, the first job clears STOP and HALT; 1 = a resume of a chain that broke,
#        the first job honours STOP, HALT and the stall rules and adopts the new chain id; with START_INDEX >= 1 and
#        SPLIT, NODES or STOP_AFTER unset they are read from the chain state file of the last job), SPLIT, RUN,
#        STOP_AFTER, NODES, STAGES, DEVICE and the job's own variables (they travel with sbatch --export=ALL),
#        SBATCH_CMD (default sbatch), SQUEUE_CMD (default squeue), DRY_RUN=1 prints the sbatch lines (no squeue).
#        prints one line per job id. TRAIN uses train_run2.sbatch, CONTROL train_run2_control.sbatch, CORPUS corpus.sbatch.
KIND=$1; N=${2:-3}
case "$KIND" in
  TRAIN) SCRIPT=scripts/slurm/train_run2.sbatch; RUNDEF=run2;;
  CONTROL) SCRIPT=scripts/slurm/train_run2_control.sbatch; RUNDEF=run2_control;;
  CORPUS) SCRIPT=scripts/slurm/corpus.sbatch; RUNDEF=;;
  *) echo "usage: submit_chain.sh TRAIN|CONTROL|CORPUS N" >&2; exit 2;;
esac
case "$N" in ''|*[!0-9]*|0) echo "submit_chain: N must be a positive integer" >&2; exit 2;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/mn5_env.sh"
cd "$REPO" || exit 2
[ "${DRY_RUN:-0}" = 1 ] || require_site_values || exit 2
START=${START_INDEX:-0}
SB=${SBATCH_CMD:-sbatch}
SQ=${SQUEUE_CMD:-squeue}
if [ "$KIND" = CORPUS ]; then
  JOBNAME=run2_corpus_$(echo "${STAGES:-all}" | tr ',' '_')
else
  JOBNAME=${RUN:-$RUNDEF}
fi
if [ "$KIND" != CORPUS ] && [ "$START" -ge 1 ]; then
  ST=ckpts/evidence_${RUN:-$RUNDEF}/chain_state
  [ -z "$SPLIT" ] && SPLIT=$(state_get "$ST" split)
  [ -z "$NODES" ] && NODES=$(state_get "$ST" nodes)
  [ -z "$STOP_AFTER" ] && STOP_AFTER=$(state_get "$ST" stop_after)
  export SPLIT NODES STOP_AFTER
fi
if [ "${DRY_RUN:-0}" != 1 ]; then
  QOUT=$($SQ -h -n "$JOBNAME" -u "${USER:-$(id -un)}" -o "%i %T" 2>&1) || { echo "submit_chain: squeue failed, not submitting ($QOUT)" >&2; exit 1; }
  if [ -n "$QOUT" ]; then
    echo "submit_chain: refusing, jobs named $JOBNAME are already in the queue (a chain is alive or pending):" >&2
    echo "$QOUT" >&2
    echo "scancel the whole chain first (squeue -n $JOBNAME), or wait; never cancel a single job of a chain" >&2
    exit 1
  fi
fi
export CHAIN_MODE=presubmitted
CHAIN_ID=${CHAIN_ID_OVERRIDE:-$(date +%s)-$$}
echo "submit_chain: job name $JOBNAME chain id $CHAIN_ID" >&2
mkdir -p logs
PREV=""
IDS=""
i=0
while [ "$i" -lt "$N" ]; do
  IDX=$(( START + i ))
  FIRST=0; [ "$i" = 0 ] && FIRST=1
  ARGS="--parsable --job-name=$JOBNAME --export=ALL,CHAIN_INDEX=$IDX,CHAIN_MODE=presubmitted,CHAIN_ID=$CHAIN_ID,CHAIN_FIRST=$FIRST"
  [ -n "$PREV" ] && ARGS="$ARGS --dependency=afterany:$PREV"
  [ -n "$NODES" ] && ARGS="$ARGS --nodes=$NODES --ntasks-per-node=${GPUS_PER_NODE:-4} --gres=gpu:${GPUS_PER_NODE:-4}"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "$SB $ARGS $SCRIPT"
    PREV=$(( 1000 + IDX ))
  else
    PREV=$($SB $ARGS "$SCRIPT") || {
      echo "submit_chain: sbatch failed at job $i; cancelling the jobs of this chain already queued:$IDS" >&2
      [ -n "$IDS" ] && scancel $IDS
      exit 1
    }
    PREV=${PREV%%;*}
    IDS="$IDS $PREV"
    echo "$IDX $PREV"
  fi
  i=$(( i + 1 ))
done

#!/usr/bin/env bash
# the one supported way to end a chain by hand: bash scripts/slurm/cancel_chain.sh [RUN]   (RUN is the job name and run
# name, default run2; the control is run2_control)
# it marks the chain ended=cancelled in the run's chain_state and adds a history line first, then scancels every job of
# that name (pending and running). a running job sees the mark in its TERM handler, asks the ranks to save and the
# afterany successors exit at once; the supervisor never resubmits a chain marked cancelled. a bare scancel is read as an
# interruption (node failure, admin kill): the chain goes on.
# env: SQUEUE_CMD (default squeue), SCANCEL_CMD (default scancel)
RUN=${1:-run2}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/mn5_env.sh"
cd "$REPO" || exit 2
SQ=${SQUEUE_CMD:-squeue}
SC=${SCANCEL_CMD:-scancel}
IDS=$($SQ -h -n "$RUN" -u "${USER:-$(id -un)}" -o %i 2>&1) || { echo "cancel_chain: squeue failed, nothing changed ($IDS)" >&2; exit 1; }
RUNDIR=ckpts/evidence_$RUN
mkdir -p "$RUNDIR"
state_put "$RUNDIR/chain_state" ended cancelled
hist_append "$RUNDIR/chain_history" 0 cancelled 0 cancelled
if [ -z "$IDS" ]; then
  echo "cancel_chain: no job named $RUN in the queue; $RUNDIR/chain_state marked ended=cancelled"
  exit 0
fi
$SC $IDS
SRC=$?
LEFT=$($SQ -h -n "$RUN" -u "${USER:-$(id -un)}" -o %i 2>&1) || LEFT="(squeue failed)"
if [ "$SRC" != 0 ] || [ -n "$LEFT" ]; then
  echo "cancel_chain: WARNING: scancel exit $SRC and jobs of $RUN still listed: $(echo $LEFT); $RUNDIR/chain_state is marked ended=cancelled, run this script again" >&2
  exit 1
fi
echo "cancel_chain: marked $RUNDIR/chain_state ended=cancelled, cancelled:" $IDS

#!/usr/bin/env bash
# one training process under srun (one task per GPU). usage: rank_exec.sh RCPREFIX COMMAND...
# sets the rank variables sft.py and accelerate read from Slurm's, runs the command, forwards SIGTERM to it
# (the process must see the stop request itself, not only this shell) and writes its exit code to
# RCPREFIX.rank<N>, because the batch script decides done, stopped, preempted or failed from rank 0's code
# and srun reports only the worst code of all tasks. MASTER_ADDR and MASTER_PORT come from the batch script.
RC=$1; shift
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
"$@" &
P=$!
TERMED=0
trap 'TERMED=1; kill -TERM $P 2>/dev/null' TERM
# a trapped signal makes wait return at once with 128+N whether or not the process ended; wait again then, which
# returns the process's own status even when it ended in between (the exit code is never the trap's 143).
# wait returns 127 for "no such child" too, so a 127 with the child still alive or never seen is told from a real 127
# of the command (python or singularity not found) by whether the child is gone: a real 127 is recorded as 127
rc=255
while :; do
  TERMED=0
  wait $P
  r=$?
  if [ "$r" = 127 ]; then
    if kill -0 "$P" 2>/dev/null; then continue; fi
    echo "rank_exec: the command exited 127 (not found: python or singularity missing on this node?): $*" >&2
    rc=127
    break
  fi
  rc=$r
  [ "$TERMED" = 1 ] && [ "$r" -gt 128 ] && continue
  break
done
echo $rc > "$RC.rank$RANK"
exit $rc

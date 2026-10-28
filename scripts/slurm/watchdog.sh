#!/usr/bin/env bash
# the training job's progress watchdog, a sidecar of the batch shell (no GPU). every POLL seconds it reads what the
# trainer log gained since the job started and the newest save in the run directory, and decides with lib.sh's
# wd_decide and diverge_check. on stalled-step, stalled-save or diverged it writes LOGDIR/wd_JOBID ("VERDICT detail"),
# and signals the batch shell (USR2), whose trap sends SIGTERM to the ranks and kills them after three minutes.
# every poll it rewrites RUNDIR/watchdog_last: "epoch|verdict|detail|step_age_s|save_age_s" (status.sh reads it).
# env: LOG RUNDIR JOBID LOGDIR START (epoch) FROM (log byte offset at job start) BATCH_PID
#      STALL_MIN (30: the trainer prints an eval only at its end, so the step line gap must cover the longest eval) SAVE_MIN (45) STARTUP_MIN (30) LOSS_CEILING (20) DIVERGE_K (3) POLL (60)
#      RISE_K (30 step lines per window median) RISE_MARGIN (1.0 nats) RISE_POLLS (5 polls in a row) RISE_MIN (15
#      window medians before the rule can fire, so a job's first quarter hour is grace)
# the relative divergence rule: every poll that saw new step lines appends the median CE of the last RISE_K step lines
# to a window medians file; when the last RISE_POLLS medians all sit more than RISE_MARGIN above the lowest median of the
# job, the verdict is diverged (a finite loss that climbs). a healthy run's loss falls and its noise (about 0.02 nats
# per step line) is far under the median and the margin, so loss noise, an eval line or the warm to main switch (a new
# job: the history restarts) never trips it. like the ceiling rule it kills the step with no save and halts the chain,
# and deletes nothing: the saves already on disk stay for the recovery
#      ONCE=1 one poll, then exit without signalling (tests); NOW overrides the clock (tests)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
STALL_S=$(( ${STALL_MIN:-30} * 60 )); SAVE_S=$(( ${SAVE_MIN:-45} * 60 )); STARTUP_S=$(( ${STARTUP_MIN:-30} * 60 ))
OFF=${FROM:-0}; LAST_STEP=0
# the temporary file lives in LOGDIR (a project disk path), never in an inherited TMPDIR that may not exist: a watchdog
# that cannot create it must not run silently without divergence detection, so it says so, marks watchdog_last dead
# and exits 1 (train_job.sh checks that the sidecar is alive and refuses to run the trainer without it)
BUF=$(mktemp -p "$LOGDIR" wd_buf.XXXXXX) || {
  echo "watchdog: cannot create a temporary file in $LOGDIR" >&2
  mkdir -p "$RUNDIR"
  echo "$(date +%s)|dead|mktemp failed in $LOGDIR|-1|-1" > "$RUNDIR/watchdog_last"
  exit 1
}
trap 'rm -f "$BUF" "$BUF.new" "$BUF.t" "$BUF.ce" "$BUF.med"' EXIT
while true; do
  NOW=${NOW:-$(date +%s)}
  SIZE=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  if [ "$SIZE" -gt "$OFF" ]; then
    tail -c +$(( OFF + 1 )) "$LOG" | grep 'Step [0-9]' | tail -n 50 > "$BUF.new"
    if [ -s "$BUF.new" ]; then
      LAST_STEP=$NOW; cat "$BUF.new" >> "$BUF"; tail -n 50 "$BUF" > "$BUF.t"; mv "$BUF.t" "$BUF"
      grep -o 'Loss (CE): [^ ]*' "$BUF.new" | sed 's/Loss (CE): //' >> "$BUF.ce"
      tail -n 200 "$BUF.ce" > "$BUF.t"; mv "$BUF.t" "$BUF.ce"
      MED=$(ce_median "${RISE_K:-30}" < "$BUF.ce")
      [ -n "$MED" ] && echo "$MED" >> "$BUF.med"
    fi
    rm -f "$BUF.new"
    OFF=$SIZE
  fi
  EXT=0
  tail -n 1 "$LOG" 2>/dev/null | grep -q 'Step [0-9]' || EXT=1
  [ "$LAST_STEP" = 0 ] && EXT=0
  SAVE=$(find "$RUNDIR" -maxdepth 1 -name 'checkpoint_evidence_*.pt' -printf '%T@\n' 2>/dev/null | sort -n | tail -n 1 | cut -d. -f1)
  SAVE=${SAVE:-0}
  V=$(wd_decide "$NOW" "$START" "$LAST_STEP" "$SAVE" "$STALL_S" "$SAVE_S" "$STARTUP_S" "$EXT")
  DETAIL="step age $(( LAST_STEP > 0 ? NOW - LAST_STEP : NOW - START ))s, save age $(( SAVE > 0 ? NOW - SAVE : NOW - START ))s"
  DV=$(diverge_check "${LOSS_CEILING:-20}" "${DIVERGE_K:-3}" < "$BUF")
  case "$DV" in diverged*) V=diverged; DETAIL=${DV#diverged };; esac
  if [ "$V" != diverged ] && [ -s "$BUF.med" ]; then
    RV=$(rise_check "${RISE_MARGIN:-1.0}" "${RISE_POLLS:-5}" "${RISE_MIN:-15}" < "$BUF.med")
    case "$RV" in diverged*) V=diverged; DETAIL=${RV#diverged };; esac
  fi
  mkdir -p "$RUNDIR"
  echo "$NOW|$V|$DETAIL|$(( LAST_STEP > 0 ? NOW - LAST_STEP : -1 ))|$(( SAVE > 0 ? NOW - SAVE : -1 ))" > "$RUNDIR/watchdog_last"
  if [ "$V" != ok ]; then
    if [ "${ONCE:-0}" = 1 ]; then echo "$V $DETAIL"; exit 0; fi
    echo "$V $DETAIL" > "$LOGDIR/wd_$JOBID"
    kill -USR2 "$BATCH_PID" 2>/dev/null
    exit 0
  fi
  if [ "${ONCE:-0}" = 1 ]; then echo "ok $DETAIL"; exit 0; fi
  unset NOW
  sleep "${POLL:-60}"
done

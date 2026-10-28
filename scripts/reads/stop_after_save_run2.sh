#!/usr/bin/env bash
# usage: stop_after_save_run2.sh RUN TOKENS   (RUN the --run-name, e.g. run2; TOKENS the token tag after which to stop:
# 4000M, 4B, or a bare number, 7 or more digits = tokens, fewer = millions)
# waits for a "Checkpoint saved at" line in the run log whose token tag is at or past TOKENS, then drops the STOP file
# in the run directory. the rule is numeric, never an exact tag: a save lands at the first log step past its grid
# point (tok4003M for 4000M, tok10005M at 16 GPUs), so a match on the exact tag would miss and the run would never
# stop. the load and retention lines also carry checkpoint names and are not read.
# the generalization of stop_after_save.sh (which is pinned to the pilot's pilot_ARM log and run names).
# a stop leaves STOP to remove before the relaunch (scripts/launch_run2.sh removes it).
# env: TINY_LLM_ROOT (default the local checkout), LOG (default ckpts/inject/RUN.log), POLL seconds (default 5: a longer poll lets the trainer pass the target save and write a second off grid save),
#      FROM a byte offset of the log: only text past it counts (the Slurm job and the launcher pass the log size at
#      their start, so a save line of an earlier job does not trigger a stop)
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../slurm/lib.sh"
cd "${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}" || exit 1
RUN=$1
N=$(stop_after_m "$2") || { echo "usage: stop_after_save_run2.sh RUN TOKENS (4000M, 4B or a token count; got '$2')" >&2; exit 1; }
[ -n "$RUN" ] || { echo "usage: stop_after_save_run2.sh RUN TOKENS" >&2; exit 1; }
L=${LOG:-ckpts/inject/${RUN}.log}
mkdir -p "$(dirname "$L")"
TMPBUF=$(mktemp -p "$(dirname "$L")" stop_after_buf.XXXXXX) || exit 1
trap 'rm -f "$TMPBUF" "$TMPBUF.t"' EXIT
# only the bytes written since the last poll are read, complete lines only (a half written line is read next time)
OFF=${FROM:-0}
while :; do
  SIZE=$(stat -c %s "$L" 2>/dev/null || echo 0)
  if [ "$SIZE" -gt "$OFF" ]; then
    tail -c +$(( OFF + 1 )) "$L" 2>/dev/null | head -c $(( SIZE - OFF )) > "$TMPBUF"
    BYTES=$(wc -c < "$TMPBUF")
    PART=0
    [ "$(tail -c 1 "$TMPBUF" | wc -l)" = 1 ] || PART=$(tail -n 1 "$TMPBUF" | wc -c)
    head -c $(( BYTES - PART )) "$TMPBUF" > "$TMPBUF.t"
    OFF=$(( OFF + BYTES - PART ))
    if save_reached "$N" < "$TMPBUF.t"; then break; fi
  fi
  sleep "${POLL:-5}"
done
rm -f "$TMPBUF" "$TMPBUF.t"
touch "ckpts/evidence_${RUN}/STOP"
echo "STOP dropped $(date -Is) after a save at or past ${N}M"

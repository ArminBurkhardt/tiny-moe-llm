#!/usr/bin/env bash
# usage: stop_after_save_run2.sh RUN PATTERN   (RUN the --run-name, e.g. run2; PATTERN a regex on the token tag,
# e.g. '[2-5][0-9]*[0-9]M' is too loose: use the exact switch window, '2000M' or '(2|3)[0-9]{3}M')
# waits for a matching checkpoint line in the run's log, then drops the STOP file in its run directory.
# the generalization of stop_after_save.sh (which is pinned to the pilot's pilot_ARM log and run names).
# saves land every 1B tokens from the last resume point; the trainer's grid is fixed in tokens since
# 2026-10-24, so a pattern on the exact tag is safe. a stop leaves STOP to remove before the relaunch
# (scripts/launch_run2.sh removes it).
# env: TINY_LLM_ROOT (default the local checkout), LOG (default ckpts/inject/RUN.log), POLL seconds (default 5)
cd "${TINY_LLM_ROOT:-/mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm}" || exit 1
RUN=$1
PAT=$2
[ -n "$RUN" ] && [ -n "$PAT" ] || { echo "usage: stop_after_save_run2.sh RUN PATTERN"; exit 1; }
L=${LOG:-ckpts/inject/${RUN}.log}
until grep -qE "checkpoint_evidence_tok(${PAT})_" "$L" 2>/dev/null; do sleep "${POLL:-5}"; done
touch "ckpts/evidence_${RUN}/STOP"
echo "STOP dropped $(date -Is) after a save matching ${PAT}"

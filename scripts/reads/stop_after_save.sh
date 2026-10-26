#!/usr/bin/env bash
# usage: stop_after_save.sh ARM PATTERN   (ARM p|c|pi|e4, PATTERN a regex on the token tag, e.g. '4[0-4][0-9]M')
# waits for a matching checkpoint line in the arm's log, then drops the STOP file.
# saves land every 50M from the last resume point, so after a stop the grid shifts by about 1M;
# pass a range, not a single number.
cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm
ARM=$1
PAT=$2
L=ckpts/inject/pilot_${ARM}.log
until grep -qE "checkpoint_evidence_tok(${PAT})_" "$L"; do sleep 5; done
touch ckpts/evidence_pilot_${ARM}/STOP
echo "STOP dropped $(date -Is) after a save matching ${PAT}"

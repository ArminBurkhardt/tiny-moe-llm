#!/usr/bin/env bash
# push a locally built corpus into REMOTE_DATA_ROOT, resumable (rsync --partial-dir; rerun the same line after a drop).
# run on the PC (WSL). usage: bash scripts/ops/push_corpus.sh LOCAL_DATA_ROOT [--only-done] [--dry-run] [SUBDIR ...]
#   SUBDIR   directories of LOCAL_DATA_ROOT to send (default: prepared_run2 prepared index run2_done)
#   --only-done  refuse the push unless run2_done/ holds a marker for every stage in STAGES (comma list, env), e.g.
#                STAGES=merge_warm,merge_main bash scripts/ops/push_corpus.sh /d/run2 --only-done: a stage is finished
#                when its marker exists, so an unfinished build is never sent
# the byte total of what rsync would send is checked against the transfer budget in the ledger before anything
# moves, and the bytes rsync really moved are appended after it, a failed or interrupted push too (the allowance
# counts what was sent). an interrupted file waits in .rsync-partial, never under its final name. the 2 TB allowance
# counts both directions. needs the transfer host master socket (ops_lib.sh).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/ops_lib.sh"
LOCAL=$1; shift
[ -d "$LOCAL" ] || { echo "usage: push_corpus.sh LOCAL_DATA_ROOT [--only-done] [--dry-run] [SUBDIR ...]" >&2; exit 2; }
DRY=0; ONLY=0; DIRS=""
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --only-done) ONLY=1;; *) DIRS="$DIRS $a";; esac; done
[ -n "$DIRS" ] || DIRS="prepared_run2 prepared index run2_done"
if [ "$ONLY" = 1 ]; then
  for s in $(echo "${STAGES:?--only-done needs STAGES=a,b}" | tr ',' ' '); do
    [ -f "$LOCAL/run2_done/$s.json" ] || { echo "refused: stage $s has no marker in $LOCAL/run2_done" >&2; exit 1; }
  done
fi
SSH="ssh $(ops_ssh_opts)"
DEST=$MN5_USER@$MN5_TRANSFER_HOST
SRC=""
for d in $DIRS; do [ -e "$LOCAL/$d" ] && SRC="$SRC $d" || echo "skip $d (not in $LOCAL)"; done
[ -n "$SRC" ] || { echo "nothing to send" >&2; exit 1; }
cd "$LOCAL" || exit 2
OUT=$(rsync -a --stats --dry-run -R -e "$SSH" $SRC "$DEST:$REMOTE_DATA_ROOT/" 2>&1) || { echo "$OUT" >&2; exit 1; }
BYTES=$(rsync_bytes "$OUT"); BYTES=${BYTES:-0}
echo "to send: $BYTES bytes ($(( BYTES / 1000000000 )) GB); ledger $(( $(ledger_total_bytes) / 1000000000 )) GB of ${TRANSFER_BUDGET_GB} GB"
[ "$DRY" = 1 ] && exit 0
budget_check "$BYTES" || exit 1
OUTF=$(mktemp)
run_rsync "$OUTF" -a --info=progress2 -R -e "$SSH" $SRC "$DEST:$REMOTE_DATA_ROOT/"
RC=$?
ledger_record_rsync in "corpus $SRC" "$RC" "$(cat "$OUTF")" "$BYTES" sent
rm -f "$OUTF"
[ "$RC" = 0 ] || { echo "rsync stopped (exit $RC); the bytes it moved are in the ledger; rerun the same line to resume" >&2; exit 1; }
echo "pushed"

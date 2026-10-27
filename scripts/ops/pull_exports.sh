#!/usr/bin/env bash
# pull exported weights and the run's small logs and json out of the cluster, verify the sha256, optionally upload to
# a private HF repo. run on the PC (WSL). usage: bash scripts/ops/pull_exports.sh [LOCAL_DIR] [--dry-run] [--hf]
#   LOCAL_DIR  default ./run2_pulled
#   --hf       after verifying, `hf upload` the weights to HF_EXPORT_REPO (private; the token stays on this PC)
# pulls REMOTE_DATA_ROOT/exports/*.pt with their .sha256, and from REMOTE_REPO/ckpts/inject the run2_* json and log
# files (reads of the run and of run2_control) and the gpu hours logs. the bytes count in the transfer ledger (both
# directions count against the allowance), a failed pull too. a file already present with a matching hash is not
# fetched again. a .pt without a .sha256 beside it is refused (never verified, never uploaded). an interrupted file
# waits in .rsync-partial. needs the transfer host master socket (ops_lib.sh).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/ops_lib.sh"
LOCAL=./run2_pulled; DRY=0; HF=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --hf) HF=1;; *) LOCAL=$a;; esac; done
SSH="ssh $(ops_ssh_opts)"
DEST=$MN5_USER@$MN5_TRANSFER_HOST
mkdir -p "$LOCAL/exports" "$LOCAL/reads"
EXP_ARGS=(--include='*.pt' --include='*.sha256' --exclude='*')
READ_ARGS=(--include='run2*.json' --include='run2*.log' --include='*_gpu_hours.log' --exclude='*')
OUT=$(rsync -a --stats --dry-run -e "$SSH" "${EXP_ARGS[@]}" "$DEST:$REMOTE_DATA_ROOT/exports/" "$LOCAL/exports/" 2>&1) || { echo "$OUT" >&2; exit 1; }
B1=$(rsync_bytes "$OUT"); B1=${B1:-0}
OUT=$(rsync -a --stats --dry-run -e "$SSH" "${READ_ARGS[@]}" "$DEST:$REMOTE_REPO/ckpts/inject/" "$LOCAL/reads/" 2>&1) || { echo "$OUT" >&2; exit 1; }
B2=$(rsync_bytes "$OUT"); B2=${B2:-0}
# the trainer logs are ckpts/inject/RUN.log (run2.log, run2_control.log: the run2*.log pattern above); the chain's own
# files are the sbatch outputs logs/RUN_JOBID.out and the reads logs logs/reads_*.log under REMOTE_REPO/logs
JOB_ARGS=(--include='run2*.out' --include='reads_*.log' --include='corpus_*.log' --exclude='*')
mkdir -p "$LOCAL/joblogs"
OUT=$(rsync -a --stats --dry-run -e "$SSH" "${JOB_ARGS[@]}" "$DEST:$REMOTE_REPO/logs/" "$LOCAL/joblogs/" 2>&1) || { echo "$OUT" >&2; exit 1; }
B3=$(rsync_bytes "$OUT"); B3=${B3:-0}
BYTES=$(( B1 + B2 + B3 ))
echo "to pull: $B1 bytes of weights, $B2 bytes of reads and trainer logs, $B3 bytes of job logs"
[ "$DRY" = 1 ] && exit 0
budget_check "$BYTES" || exit 1
OUTF=$(mktemp)
run_rsync "$OUTF" -a -e "$SSH" "${EXP_ARGS[@]}" "$DEST:$REMOTE_DATA_ROOT/exports/" "$LOCAL/exports/"
RC1=$?
ledger_record_rsync out "exports to $LOCAL" "$RC1" "$(cat "$OUTF")" "$B1" received
run_rsync "$OUTF" -a -e "$SSH" "${READ_ARGS[@]}" "$DEST:$REMOTE_REPO/ckpts/inject/" "$LOCAL/reads/"
RC2=$?
ledger_record_rsync out "reads to $LOCAL" "$RC2" "$(cat "$OUTF")" "$B2" received
run_rsync "$OUTF" -a -e "$SSH" "${JOB_ARGS[@]}" "$DEST:$REMOTE_REPO/logs/" "$LOCAL/joblogs/"
RC3=$?
ledger_record_rsync out "job logs to $LOCAL" "$RC3" "$(cat "$OUTF")" "$B3" received
rm -f "$OUTF"
[ "$RC1" = 0 ] && [ "$RC2" = 0 ] && [ "$RC3" = 0 ] || { echo "rsync failed (exit $RC1 / $RC2 / $RC3); the bytes it moved are in the ledger; rerun to resume" >&2; exit 1; }
BAD=0
for f in "$LOCAL"/exports/*.pt; do
  [ -e "$f" ] || continue
  [ -e "$f.sha256" ] || { echo "refused: $(basename "$f") has no .sha256 beside it (an export is written with one; this file is partial or foreign)" >&2; BAD=1; }
done
for s in "$LOCAL"/exports/*.sha256; do
  [ -e "$s" ] || continue
  ( cd "$LOCAL/exports" && sha256sum -c "$(basename "$s")" ) || BAD=1
done
[ "$BAD" = 0 ] || { echo "sha256 problem: do not use those files, rerun to refetch" >&2; exit 1; }
if [ "$HF" = 1 ]; then
  [ -n "$HF_EXPORT_REPO" ] || { echo "--hf needs HF_EXPORT_REPO in ops.env" >&2; exit 2; }
  hf repo create "$HF_EXPORT_REPO" --private --exist-ok >/dev/null
  for f in "$LOCAL"/exports/*.pt; do hf upload "$HF_EXPORT_REPO" "$f" "$(basename "$f")" --repo-type model; done
fi
echo "pulled to $LOCAL"

#!/usr/bin/env bash
# push the repo to the cluster (the compute and login nodes have no internet, so code arrives from here).
# run on the PC (WSL), from anywhere. usage: bash scripts/ops/push_code.sh [worktree|head|bundle] [--dry-run]
#   worktree (default)  rsync of the tracked and untracked-but-not-ignored files as they are now (uncommitted work included)
#   head                git archive HEAD piped through ssh into REMOTE_REPO (committed state only)
#   bundle              a git bundle of the current branch copied to REMOTE_REPO/../tiny-llm.bundle (clone it on the login node)
# data, ckpts, venv and every gitignored file are never sent. scripts/slurm/mn5_env.local.sh (the cluster site
# values) is never sent in any mode: it is not in the repo, and the head mode extracts over the checkout, so a tracked
# file of that name would overwrite it. each push appends to the transfer ledger, a failed one too.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/ops_lib.sh"
MODE=worktree; DRY=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; worktree|head|bundle) MODE=$a;; *) echo "usage: push_code.sh [worktree|head|bundle] [--dry-run]" >&2; exit 2;; esac; done
cd "$(git -C "$HERE" rev-parse --show-toplevel)" || exit 2
SSH="ssh $(ops_ssh_opts)"
DEST=$MN5_USER@$MN5_TRANSFER_HOST
LOCALONLY='^(data|ckpts|venv|logs)/|(^|/)mn5_env\.local\.sh$'
OUTF=$(mktemp)
case "$MODE" in
  worktree)
    LIST=$(mktemp)
    git ls-files -z -c -o --exclude-standard | tr '\0' '\n' | grep -vE "$LOCALONLY" > "$LIST"
    OUT=$(rsync -a --stats --dry-run -e "$SSH" --files-from="$LIST" ./ "$DEST:$REMOTE_REPO/" 2>&1) || { echo "$OUT" >&2; exit 1; }
    BYTES=$(rsync_bytes "$OUT"); BYTES=${BYTES:-0}
    echo "worktree: $(wc -l < "$LIST") files, $BYTES bytes to send"
    CRS=$(crlf_files "$LIST") || { echo "refused: carriage returns in shipped shell files (the cluster's bash would fail on them; fix with: sed -i 's/\r\$//' FILE):" >&2; echo "$CRS" >&2; rm -f "$LIST" "$OUTF"; exit 1; }
    [ "$DRY" = 1 ] && exit 0
    budget_check "$BYTES" || exit 1
    run_rsync "$OUTF" -a -e "$SSH" --files-from="$LIST" ./ "$DEST:$REMOTE_REPO/"
    RC=$?
    ledger_record_rsync in "code worktree $(git rev-parse --short HEAD)" "$RC" "$(cat "$OUTF")" "$BYTES" sent
    rm -f "$LIST" "$OUTF"
    [ "$RC" = 0 ] || exit 1
    ;;
  head)
    CRS=$(crlf_head) || { echo "refused: carriage returns in committed shell files:" >&2; echo "$CRS" >&2; exit 1; }
    TMP=$(mktemp)
    git archive --format=tar HEAD > "$TMP"
    BYTES=$(stat -c %s "$TMP")
    echo "head: $BYTES bytes"
    [ "$DRY" = 1 ] && { rm -f "$TMP"; exit 0; }
    budget_check "$BYTES" || { rm -f "$TMP"; exit 1; }
    # the site overlay survives an extract only because it is not in the archive; refuse if it ever is
    if tar -tf "$TMP" | grep -qE 'mn5_env\.local\.sh$'; then echo "refused: mn5_env.local.sh is in the archive" >&2; rm -f "$TMP"; exit 1; fi
    $SSH "$DEST" "mkdir -p $REMOTE_REPO && tar -x -C $REMOTE_REPO" < "$TMP"
    RC=$?
    ledger_append in "code head $(git rev-parse --short HEAD)$([ "$RC" = 0 ] || echo " FAILED rc $RC") (tar size, upper bound)" "$BYTES"
    rm -f "$TMP"
    [ "$RC" = 0 ] || exit 1
    ;;
  bundle)
    CRS=$(crlf_head) || { echo "refused: carriage returns in committed shell files:" >&2; echo "$CRS" >&2; exit 1; }
    TMP=$(mktemp -u).bundle
    git bundle create "$TMP" HEAD >/dev/null || exit 1
    BYTES=$(stat -c %s "$TMP")
    echo "bundle: $BYTES bytes"
    [ "$DRY" = 1 ] && { rm -f "$TMP"; exit 0; }
    budget_check "$BYTES" || { rm -f "$TMP"; exit 1; }
    run_rsync "$OUTF" -a -e "$SSH" "$TMP" "$DEST:$REMOTE_REPO.bundle"
    RC=$?
    ledger_record_rsync in "code bundle $(git rev-parse --short HEAD)" "$RC" "$(cat "$OUTF")" "$BYTES" sent
    rm -f "$TMP" "$OUTF"
    [ "$RC" = 0 ] || exit 1
    ;;
esac
echo "pushed ($MODE) to $DEST:$REMOTE_REPO"

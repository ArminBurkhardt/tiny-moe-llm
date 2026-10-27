#!/usr/bin/env bash
# shared by the push and pull scripts (run in WSL or Linux on the user's PC, never on the cluster, never in the cloud).
# sourced, never run. settings come from scripts/ops/ops.env (copy ops.env.example) or the environment.
OPS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$OPS_DIR/ops.env" ] && source "$OPS_DIR/ops.env"
MN5_USER=${MN5_USER:-CHANGE_USER}
MN5_TRANSFER_HOST=${MN5_TRANSFER_HOST:-transfer1.bsc.es}
REMOTE_REPO=${REMOTE_REPO:-/gpfs/projects/CHANGE_PROJECT/tiny-llm}
REMOTE_DATA_ROOT=${REMOTE_DATA_ROOT:-/gpfs/projects/CHANGE_PROJECT/run2}
SSH_SOCK=${SSH_SOCK:-$HOME/.ssh/mn5.sock}
# the transfer host has its own master socket: a ControlPath of the login host would send the rsync through alogin1
# (the login node's bandwidth and policy, and the wrong host). open it once by hand, README "Supervision":
#   ssh -M -S ~/.ssh/mn5-transfer.sock -o ControlPersist=yes USER@transfer1.bsc.es
SSH_SOCK_TRANSFER=${SSH_SOCK_TRANSFER:-$HOME/.ssh/mn5-transfer.sock}
TRANSFER_BUDGET_GB=${TRANSFER_BUDGET_GB:-2000}
LEDGER=${LEDGER:-ops_transfer_ledger.tsv}

# ops_ssh_opts: ssh options for rsync and ssh to the transfer host: through its own master socket when alive, never a
# prompt (BatchMode). without the socket it warns and goes on (key authentication only)
ops_ssh_opts() {
  local o="-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30"
  if [ -S "$SSH_SOCK_TRANSFER" ]; then
    o="$o -o ControlPath=$SSH_SOCK_TRANSFER"
  else
    echo "ops: no transfer master socket $SSH_SOCK_TRANSFER (open it, README Supervision); trying a plain BatchMode connection" >&2
  fi
  echo "$o"
}

# ledger_total_bytes: bytes in and out so far (column 4 of the ledger)
ledger_total_bytes() {
  [ -f "$LEDGER" ] || { echo 0; return; }
  awk -F'\t' 'NR > 1 && $4 ~ /^[0-9]+$/ { s += $4 } END { printf "%d\n", s }' "$LEDGER"
}

# ledger_append DIRECTION WHAT BYTES: one line, direction is in or out (relative to the cluster)
ledger_append() {
  [ -f "$LEDGER" ] || printf 'time\tdirection\twhat\tbytes\n' > "$LEDGER"
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" >> "$LEDGER"
}

# budget_check BYTES: exit 1 with a message when the ledger plus BYTES would cross TRANSFER_BUDGET_GB
budget_check() {
  local add=$1 total limit
  total=$(ledger_total_bytes)
  limit=$(awk -v g="$TRANSFER_BUDGET_GB" 'BEGIN { printf "%.0f", g * 1e9 }')
  if [ $(( total + add )) -gt "$limit" ]; then
    echo "refused: $(( (total + add) / 1000000000 )) GB would cross the transfer budget of ${TRANSFER_BUDGET_GB} GB (ledger $LEDGER holds $(( total / 1000000000 )) GB)" >&2
    return 1
  fi
}

# rsync_bytes OUTPUT: "Total transfer size" of an rsync --stats --dry-run in bytes
rsync_bytes() {
  sed -n 's/^Total transfer size: \([0-9,]*\) bytes.*/\1/p' <<< "$1" | tr -d ',' | tail -n 1
}

# rsync_moved_bytes OUTPUT sent|received: the bytes a real rsync --stats run moved ("Total bytes sent" / "written" for a
# push, "Total bytes received" / "read" for a pull), empty when the output has no such line (rsync was killed)
rsync_moved_bytes() {
  local pat='sent\|written'
  [ "$2" = received ] && pat='received\|read'
  sed -n "s/^Total bytes \\($pat\\): \\([0-9,]*\\).*/\\2/p" <<< "$1" | tr -d ',' | tail -n 1
}

# ledger_record_rsync DIRECTION WHAT RC OUTPUT ESTIMATE FIELD: appends the transfer to the ledger whether it worked or
# not. the bytes come from the rsync --stats output (FIELD sent for a push, received for a pull); a killed rsync prints
# none, and then the dry run estimate ESTIMATE is recorded as an upper bound (a failed transfer still used the
# allowance). a failed run is labelled in WHAT.
ledger_record_rsync() {
  local dir=$1 what=$2 rc=$3 out=$4 est=$5 field=$6 b
  b=$(rsync_moved_bytes "$out" "$field")
  if [ -z "$b" ]; then b=${est:-0}; what="$what (no rsync stats, dry run estimate)"; fi
  [ "$rc" = 0 ] || what="$what FAILED rc $rc"
  ledger_append "$dir" "$what" "$b"
}

# run_rsync OUTFILE RSYNC_ARGS...: a real rsync with --stats and a partial dir, its output shown and kept in OUTFILE;
# returns rsync's exit code. --partial-dir keeps an interrupted file out of its final name (a plain --partial leaves a
# truncated file under the final name, which a later pull or hash check would take for the file)
run_rsync() {
  local outfile=$1; shift
  rsync --stats --partial-dir=.rsync-partial "$@" 2>&1 | tee "$outfile"
  return "${PIPESTATUS[0]}"
}

# crlf_files LISTFILE: reads file names (one per line, relative to the cwd) from LISTFILE and prints those of the shell,
# sbatch, def, example and Dockerfile kinds that contain a carriage return, which bash on the cluster would choke on (a
# shebang with a trailing CR, a variable value ending in one). exit 1 when there is any
crlf_files() {
  local f bad=0
  while IFS= read -r f; do
    case "$f" in *.sh|*.sbatch|*.def|*.example|*/Dockerfile|Dockerfile) ;; *) continue;; esac
    [ -f "$f" ] || continue
    if grep -q $'\r' "$f"; then echo "$f"; bad=1; fi
  done < "$1"
  return $bad
}

# crlf_head: the same check over the committed files of HEAD (the head and bundle push modes ship those)
crlf_head() {
  local tmp bad=0 f
  tmp=$(mktemp)
  git ls-tree -r --name-only HEAD > "$tmp"
  while IFS= read -r f; do
    case "$f" in *.sh|*.sbatch|*.def|*.example|*/Dockerfile|Dockerfile) ;; *) continue;; esac
    if git show "HEAD:$f" | grep -q $'\r'; then echo "$f"; bad=1; fi
  done < "$tmp"
  rm -f "$tmp"
  return $bad
}

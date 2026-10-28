#!/usr/bin/env bash
# one poll of the real run on the always-on relay VM: supervise, notify through ntfy, pull the exports and reads,
# upload them to a private HF repo. run from the repo root (relay.service does). usage: bash scripts/ops/relay/relay_poll.sh
# settings from scripts/ops/ops.env or the environment: NTFY_TOPIC (required), NTFY_SERVER, RELAY_LOCAL_DIR,
# HF_EXPORT_REPO. the HF token comes from `hf auth login` on the VM, never from ops.env. nothing is deleted, on the
# cluster or here. one poll at a time is systemd's job (a oneshot service never overlaps itself).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../ops_lib.sh"
NTFY_SERVER=${NTFY_SERVER:-https://ntfy.sh}
RELAY_LOCAL_DIR=${RELAY_LOCAL_DIR:-./run2_pulled}
STATE=ops_state
QUIET_SECONDS=21600
if [ -z "${NTFY_TOPIC:-}" ]; then echo "relay: NTFY_TOPIC is not set (ops.env); refusing to run" >&2; exit 2; fi
mkdir -p "$STATE" "$RELAY_LOCAL_DIR"
RELAYLOG=$STATE/relay.log
SUMMARY=()

# ntfy MESSAGE [TITLE] [PRIORITY]
ntfy() {
  local args=(-sS --max-time 30 -d "$1")
  [ -n "${2:-}" ] && args+=(-H "Title: $2")
  [ -n "${3:-}" ] && args+=(-H "Priority: $3")
  curl "${args[@]}" "$NTFY_SERVER/$NTFY_TOPIC" > /dev/null || echo "relay: ntfy failed" >&2
}

# ntfy_limited KEY MESSAGE [TITLE] [PRIORITY]: at most one message per KEY every six hours
ntfy_limited() {
  local f=$STATE/relay_last_$1 now last=0
  now=$(date +%s)
  [ -f "$f" ] && last=$(cat "$f")
  if [ $(( now - last )) -ge $QUIET_SECONDS ]; then
    ntfy "$2" "${3:-}" "${4:-}"
    echo "$now" > "$f"
  fi
}

# a key that recovered may notify again at the next failure
clear_limit() { rm -f "$STATE/relay_last_$1"; }

OUT=$(python3 scripts/ops/supervise.py --act 2>&1)
RC=$?
echo "$OUT"
NOTE=$(grep '^notify: ' <<< "$OUT" | tail -n 1)
NOTE=${NOTE#notify: }
SUMMARY+=("supervise=$RC")
if [ "$RC" = 1 ]; then
  # a new alert text goes out at once; the same alert (a HALT waiting for the user) repeats every six hours, not every poll
  # compared on the alerts after the first " | ": the head of the line carries the token count, new every poll
  ALERTS=${NOTE#* | }; ALERTS=${ALERTS%% | milestone:*}
  LAST_ATTN=$(cat "$STATE/relay_attn_note" 2>/dev/null)
  if [ "$ALERTS" != "$LAST_ATTN" ]; then clear_limit attn; echo "$ALERTS" > "$STATE/relay_attn_note"; fi
  ntfy_limited attn "${NOTE:-attention needed}" "MN5 run: attention" high
elif [ "$RC" = 2 ]; then
  ntfy_limited unreachable "${NOTE:-cluster unreachable}" "MN5 run: unreachable"
elif [[ "$NOTE" == *"milestone:"* ]]; then
  ntfy "$NOTE" "MN5 run: milestone"
fi
[ "$RC" = 2 ] || clear_limit unreachable
[ "$RC" = 1 ] || { clear_limit attn; rm -f "$STATE/relay_attn_note"; }

if [ "$RC" != 2 ]; then
  USED=$(df -P "$RELAY_LOCAL_DIR" | awk 'NR == 2 { gsub("%", "", $5); print $5 }')
  if [ "${USED:-0}" -gt 90 ]; then
    ntfy_limited disk "relay disk ${USED}% full, pull skipped" "relay: disk" high
    SUMMARY+=("pull=skipped-disk")
  else
    if bash scripts/ops/pull_exports.sh "$RELAY_LOCAL_DIR"; then
      SUMMARY+=("pull=ok"); clear_limit pull
    else
      SUMMARY+=("pull=failed")
      ntfy_limited pull "pull_exports.sh failed on the relay, see ops_state/relay.log and journalctl" "relay: pull failed" high
    fi
  fi
fi

# one line per newly landed read or export; more than five at once (the first poll) become one line
SEEN_READS=$STATE/relay_seen_reads.txt
SEEN_EXPORTS=$STATE/relay_seen_exports.txt
touch "$SEEN_READS" "$SEEN_EXPORTS"
announce_new() {
  local seenfile=$1 label=$2; shift 2
  local f name new=()
  for f in "$@"; do
    [ -e "$f" ] || continue
    name=$(basename "$f")
    grep -qxF "$name" "$seenfile" || new+=("$name")
  done
  [ "${#new[@]}" -gt 0 ] || return 0
  if [ "${#new[@]}" -gt 5 ]; then
    ntfy "${#new[@]} ${label}s landed (latest ${new[-1]})" "relay: ${label}s"
  else
    for name in "${new[@]}"; do
      case "$label" in
        read) ntfy "read landed: ${name%_rank_none.json}" "relay: read";;
        *) ntfy "export landed: $name" "relay: export";;
      esac
    done
  fi
  printf '%s\n' "${new[@]}" >> "$seenfile"
}
announce_new "$SEEN_READS" read "$RELAY_LOCAL_DIR"/reads/run2*_rank_none.json
announce_new "$SEEN_EXPORTS" export "$RELAY_LOCAL_DIR"/exports/*.pt

if [ -n "${HF_EXPORT_REPO:-}" ]; then
  MARK=$STATE/relay_hf_repo_created
  UP=ok
  if [ ! -f "$MARK" ]; then
    if hf repo create "$HF_EXPORT_REPO" --private --exist-ok > /dev/null; then touch "$MARK"; else UP=failed; fi
  fi
  if [ "$UP" = ok ]; then
    hf upload-large-folder "$HF_EXPORT_REPO" "$RELAY_LOCAL_DIR" --repo-type model --exclude ".rsync-partial/*" --exclude "*/.rsync-partial/*" --no-bars --no-report || UP=failed
  fi
  SUMMARY+=("upload=$UP")
  if [ "$UP" = ok ]; then clear_limit upload; else
    ntfy_limited upload "upload to $HF_EXPORT_REPO failed, will retry next poll" "relay: upload failed" high
  fi
fi

echo "$(date -Is) ${SUMMARY[*]} :: ${NOTE:-no notify line}" >> "$RELAYLOG"
exit 0

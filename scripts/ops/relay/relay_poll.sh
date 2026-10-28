#!/usr/bin/env bash
# one poll of the real run on the always-on relay VM: supervise, notify through ntfy, pull the exports and reads,
# upload them to a private HF repo. run from the repo root (relay.service does). usage: bash scripts/ops/relay/relay_poll.sh
# settings from scripts/ops/ops.env or the environment: NTFY_TOPIC (required), NTFY_SERVER, RELAY_LOCAL_DIR,
# HF_EXPORT_REPO, HF_PYTHON (the python with huggingface_hub, default python3), HEARTBEAT_HOURS (default 24, 0 = off),
# LOG_UPLOAD_HOURS (default 6). the HF token comes from
# `hf auth login` on the VM, never from ops.env. nothing is deleted on the cluster; here only a pulled export whose hash
# did not match is removed (so it is fetched again). one poll at a time is systemd's job (a oneshot service never
# overlaps itself).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../ops_lib.sh"
NTFY_SERVER=${NTFY_SERVER:-https://ntfy.sh}
RELAY_LOCAL_DIR=${RELAY_LOCAL_DIR:-./run2_pulled}
STATE=ops_state
QUIET_SECONDS=21600
HEARTBEAT_HOURS=${HEARTBEAT_HOURS:-24}
LOG_UPLOAD_HOURS=${LOG_UPLOAD_HOURS:-6}
if [ -z "${NTFY_TOPIC:-}" ]; then echo "relay: NTFY_TOPIC is not set (ops.env); refusing to run" >&2; exit 2; fi
mkdir -p "$STATE" "$RELAY_LOCAL_DIR"
RELAYLOG=$STATE/relay.log
SUMMARY=()

# ntfy MESSAGE [TITLE] [PRIORITY]: returns curl's status, so a caller records a send only when it went out
ntfy() {
  local args=(-sS --fail --max-time 30 -d "$1")
  [ -n "${2:-}" ] && args+=(-H "Title: $2")
  [ -n "${3:-}" ] && args+=(-H "Priority: $3")
  curl "${args[@]}" "$NTFY_SERVER/$NTFY_TOPIC" > /dev/null || { echo "relay: ntfy failed" >&2; return 1; }
}

# ntfy_limited KEY MESSAGE [TITLE] [PRIORITY]: at most one message per KEY every six hours; the time is recorded only
# when the send worked, so a failed send is tried again at the next poll
ntfy_limited() {
  local f=$STATE/relay_last_$1 now last=0
  now=$(date +%s)
  [ -f "$f" ] && last=$(cat "$f")
  if [ $(( now - last )) -ge $QUIET_SECONDS ]; then
    ntfy "$2" "${3:-}" "${4:-}" && echo "$now" > "$f"
  fi
}

# due_hours STAMPFILE HOURS: exit 0 when the stamp is older than HOURS (or missing); HOURS 0 is never due
due_hours() {
  local last=0
  [ "$2" -gt 0 ] 2>/dev/null || return 1
  [ -f "$1" ] && last=$(cat "$1")
  [ $(( $(date +%s) - last )) -ge $(( $2 * 3600 )) ]
}

# a key that recovered may notify again at the next failure
clear_limit() { rm -f "$STATE/relay_last_$1"; }

# startup self-check: the upload path needs huggingface_hub in $HF_PYTHON and the hf CLI; say the real cause once, not a
# misleading "repo not private" at every poll
HF_READY=1
if [ -n "${HF_EXPORT_REPO:-}" ]; then
  if ! HF_WHY=$(hf_python_check); then
    HF_READY=0
    ntfy_limited hfpython "relay cannot upload: $HF_WHY" "relay: HF setup" high
  elif ! command -v hf > /dev/null; then
    HF_READY=0
    ntfy_limited hfpython "relay cannot upload: the hf CLI is not on PATH" "relay: HF setup" high
  fi
fi

OUT=$(python3 scripts/ops/supervise.py --act 2>&1)
RC=$?
echo "$OUT"
NOTE=$(grep '^notify: ' <<< "$OUT" | tail -n 1)
NOTE=${NOTE#notify: }
KEYS=$(grep '^keys: ' <<< "$OUT" | tail -n 1)
KEYS=${KEYS#keys: }
SUMMARY+=("supervise=$RC")
if [ "$RC" = 1 ]; then
  # a new alert goes out at once; the same alert (a HALT waiting for the user, a rate under the floor) repeats every six
  # hours, not every poll. compared on the keys supervise.py prints (the alert texts with only the drifting figures, rate,
  # minutes, CE, projected and pending hours, wiped; an exit code or a save tag stays), so a changed alert is a new one;
  # the message sent still carries the current numbers. an older supervise.py with no keys line falls back to the alert text
  ALERTS=${NOTE#* | }; ALERTS=${ALERTS%% | milestone:*}
  ALERTS_KEY=${KEYS:-$ALERTS}
  LAST_ATTN=$(cat "$STATE/relay_attn_note" 2>/dev/null)
  if [ "$ALERTS_KEY" != "$LAST_ATTN" ]; then clear_limit attn; echo "$ALERTS_KEY" > "$STATE/relay_attn_note"; fi
  ntfy_limited attn "${NOTE:-attention needed}" "MN5 run: attention" high
elif [ "$RC" = 2 ]; then
  ntfy_limited unreachable "${NOTE:-cluster unreachable}" "MN5 run: unreachable"
fi
[ "$RC" = 2 ] || clear_limit unreachable
[ "$RC" = 1 ] || { clear_limit attn; rm -f "$STATE/relay_attn_note"; }

# a milestone goes out on its own whatever the exit code (an alert in force must not swallow it), once per milestone
# a milestone stays in the pending file until its message went out, so a failed send is retried at the next poll
touch "$STATE/relay_milestones.txt" "$STATE/relay_milestones_pending.txt"
if [[ "$NOTE" == *" | milestone: "* ]]; then
  MILESTONE=${NOTE##* | milestone: }
  grep -qxF "$MILESTONE" "$STATE/relay_milestones.txt" || grep -qxF "$MILESTONE" "$STATE/relay_milestones_pending.txt"     || echo "$MILESTONE" >> "$STATE/relay_milestones_pending.txt"
fi
if [ -s "$STATE/relay_milestones_pending.txt" ]; then
  : > "$STATE/relay_milestones_pending.new"
  while IFS= read -r MS; do
    [ -n "$MS" ] || continue
    if ntfy "milestone: $MS (${NOTE%% | *})" "MN5 run: milestone"; then
      echo "$MS" >> "$STATE/relay_milestones.txt"
    else
      echo "$MS" >> "$STATE/relay_milestones_pending.new"
    fi
  done < "$STATE/relay_milestones_pending.txt"
  mv "$STATE/relay_milestones_pending.new" "$STATE/relay_milestones_pending.txt"
fi

PULL_OK=0     # 1: the pull finished and every export with a hash verified
PULL_PART=0   # 1: the pull finished but a hash problem was found (exit 5); the bad file is gone, the others verified
if [ "$RC" != 2 ]; then
  USED=$(df -P "$RELAY_LOCAL_DIR" | awk 'NR == 2 { gsub("%", "", $5); print $5 }')
  if [ "${USED:-0}" -gt 90 ]; then
    ntfy_limited disk "relay disk ${USED}% full, pull skipped" "relay: disk" high
    SUMMARY+=("pull=skipped-disk")
  else
    bash scripts/ops/pull_exports.sh "$RELAY_LOCAL_DIR"
    PRC=$?
    if [ "$PRC" = 0 ]; then
      SUMMARY+=("pull=ok"); PULL_OK=1; clear_limit pull
    elif [ "$PRC" = 5 ]; then
      SUMMARY+=("pull=hash-problem"); PULL_PART=1
      ntfy_limited pull "pull_exports.sh found an export that failed its sha256 or has had no .sha256 for hours; that file is held back, the others go on" "relay: export hash problem" high
    else
      SUMMARY+=("pull=failed")
      ntfy_limited pull "pull_exports.sh failed on the relay, see ops_state/relay.log and journalctl" "relay: pull failed" high
    fi
  fi
fi

# one line per newly landed read or export; more than five at once (the first poll) become one line. a name is marked
# seen only when its message went out
SEEN_READS=$STATE/relay_seen_reads.txt
SEEN_EXPORTS=$STATE/relay_seen_exports.txt
touch "$SEEN_READS" "$SEEN_EXPORTS"
announce_new() {
  local seenfile=$1 label=$2; shift 2
  local f name new=() sent=()
  for f in "$@"; do
    [ -e "$f" ] || continue
    name=$(basename "$f")
    grep -qxF "$name" "$seenfile" || new+=("$name")
  done
  [ "${#new[@]}" -gt 0 ] || return 0
  if [ "${#new[@]}" -gt 5 ]; then
    ntfy "${#new[@]} ${label}s landed (latest ${new[-1]})" "relay: ${label}s" && sent=("${new[@]}")
  else
    for name in "${new[@]}"; do
      case "$label" in
        read) ntfy "read landed: ${name%_rank_none.json}" "relay: read" && sent+=("$name");;
        *) ntfy "export landed: $name" "relay: export" && sent+=("$name");;
      esac
    done
  fi
  [ "${#sent[@]}" -gt 0 ] && printf '%s\n' "${sent[@]}" >> "$seenfile"
  return 0
}
announce_new "$SEEN_READS" read "$RELAY_LOCAL_DIR"/reads/run2*_rank_none.json
# an export is announced and uploaded only after a pull in which every hash verified (pull_exports.sh exits 0 then)
if [ "$PULL_OK" = 1 ] || [ "$PULL_PART" = 1 ]; then
  VERIFIED=()
  for f in "$RELAY_LOCAL_DIR"/exports/*.pt; do
    if [ -e "$f" ] && [ -e "$f.sha256" ]; then VERIFIED+=("$f"); fi
  done
  [ "${#VERIFIED[@]}" -gt 0 ] && announce_new "$SEEN_EXPORTS" export "${VERIFIED[@]}"
fi

if [ -n "${HF_EXPORT_REPO:-}" ] && [ "$PULL_OK" != 1 ] && [ "$PULL_PART" != 1 ]; then
  SUMMARY+=("upload=skipped-no-complete-pull")
elif [ -n "${HF_EXPORT_REPO:-}" ] && [ "$HF_READY" != 1 ]; then
  SUMMARY+=("upload=skipped-hf-setup")
elif [ -n "${HF_EXPORT_REPO:-}" ]; then
  MARK=$STATE/relay_hf_repo_created
  UP=ok
  if [ ! -f "$MARK" ]; then
    if hf repo create "$HF_EXPORT_REPO" --private --exist-ok > /dev/null; then touch "$MARK"; else UP=failed; fi
  fi
  # --exist-ok leaves an existing public repo public: check the visibility before every upload
  if [ "$UP" = ok ]; then
    hf_repo_is_private "$HF_EXPORT_REPO"
    case $? in
      0) ;;
      3) UP=refused; ntfy_limited visibility "HF repo $HF_EXPORT_REPO is PUBLIC: uploads refused until it is private" "relay: HF repo not private" urgent;;
      *) UP=refused; ntfy_limited visibility "HF visibility check failed, uploads refused: $HF_PRIV_WHY" "relay: HF check failed" high;;
    esac
  fi
  if [ "$UP" = ok ]; then
    # the logs grow every poll: they go up every LOG_UPLOAD_HOURS hours, exports and json reads at every poll
    EXCL=(--exclude ".rsync-partial/*" --exclude "*/.rsync-partial/*")
    # an export without its hash (still landing, or stuck) is held back; the others and the json reads go up
    for f in "$RELAY_LOCAL_DIR"/exports/*.pt; do
      [ -e "$f" ] && [ ! -e "$f.sha256" ] && EXCL+=(--exclude "exports/$(basename "$f")")
    done
    LOGSTAMP=$STATE/relay_last_log_upload
    WITH_LOGS=0
    if due_hours "$LOGSTAMP" "$LOG_UPLOAD_HOURS"; then WITH_LOGS=1; else EXCL+=(--exclude "*.log" --exclude "joblogs/*"); fi
    if hf upload-large-folder "$HF_EXPORT_REPO" "$RELAY_LOCAL_DIR" --repo-type model "${EXCL[@]}" --no-bars --no-report; then
      [ "$WITH_LOGS" = 1 ] && date +%s > "$LOGSTAMP"
    else
      UP=failed
    fi
  fi
  SUMMARY+=("upload=$UP")
  if [ "$UP" = ok ]; then clear_limit upload; elif [ "$UP" = failed ]; then
    ntfy_limited upload "upload to $HF_EXPORT_REPO failed, will retry next poll" "relay: upload failed" high
  fi
fi

# heartbeat: one low priority message per HEARTBEAT_HOURS with the latest status line, so silence means the relay is down
if due_hours "$STATE/relay_last_heartbeat" "$HEARTBEAT_HOURS"; then
  ntfy "relay alive: ${NOTE%% | *} (${SUMMARY[*]})" "MN5 relay heartbeat" low && date +%s > "$STATE/relay_last_heartbeat"
fi

echo "$(date -Is) ${SUMMARY[*]} :: ${NOTE:-no notify line}" >> "$RELAYLOG"
exit 0

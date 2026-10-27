#!/usr/bin/env bash
# shared pure functions of the real run's launchers and Slurm jobs (docs/plans/RUN2.md, scripts/slurm/README.md).
# sourced, never run; no Slurm needed to call any of them (tests/test_run2_launchers.py does).

GLOBAL_TOKENS=${GLOBAL_TOKENS:-524288}
BATCH_PER_GPU=${BATCH_PER_GPU:-8}
SEQ_LEN=${SEQ_LEN:-4096}

# accum_for GPUS: the gradient accumulation that keeps GLOBAL_TOKENS per update, or exit 1 when it is not an integer
accum_for() {
  local gpus=$1 per
  case "$gpus" in ''|*[!0-9]*|0) echo "accum_for: bad GPU count '$gpus'" >&2; return 1;; esac
  per=$(( BATCH_PER_GPU * SEQ_LEN * gpus ))
  if [ "$per" -gt "$GLOBAL_TOKENS" ]; then
    echo "accum_for: $gpus GPUs at $BATCH_PER_GPU x $SEQ_LEN each already exceed $GLOBAL_TOKENS tokens per update (16 GPUs is the most at accumulation 1); lower the per GPU batch with BATCH_PER_GPU=N and the yaml's batch_size to match, or use 16 GPUs or fewer" >&2
    return 1
  fi
  if [ $(( GLOBAL_TOKENS % per )) -ne 0 ]; then
    echo "accum_for: $GLOBAL_TOKENS tokens per update is not a whole number of micro batches on $gpus GPUs ($BATCH_PER_GPU x $SEQ_LEN each); use a GPU count dividing $(( GLOBAL_TOKENS / (BATCH_PER_GPU * SEQ_LEN) ))" >&2
    return 1
  fi
  echo $(( GLOBAL_TOKENS / per ))
}

# chain_decide RC FINAL STOP STALLS: what a finished job does next. RC is rank 0's exit code (255 when unknown),
# FINAL and STOP are 1 when checkpoint_evidence_final.pt / the STOP file exist, STALLS the consecutive jobs that
# made no new save. prints done | stop | fail | nofinal | resubmit. done needs the final checkpoint: rank 0 exit 0
# without it is nofinal (the chain halts with that reason, it is never read as finished)
chain_decide() {
  local rc=$1 final=$2 stop=$3 stalls=$4
  if [ "$final" = 1 ]; then echo done; return; fi
  if [ "$rc" = 0 ]; then echo nofinal; return; fi
  if [ "$rc" = 10 ] || [ "$stop" = 1 ]; then echo stop; return; fi
  if [ "$rc" = 30 ]; then echo fail; return; fi
  # rc 1 is a SystemExit with a message (seed mismatch, missing evidence_fixed, a refused head), rc 2 the missing rows
  # cache or a bad argument: every rerun repeats them, so they halt at once
  if [ "$rc" = 1 ] || [ "$rc" = 2 ]; then echo fail; return; fi
  if [ "$stalls" -ge 2 ]; then echo fail; return; fi
  echo resubmit
}

# chain_precheck FINAL STOP HALT MARKER LAST STALLS INDEX MAXJOBS: run at the start of a job, decides whether it
# runs. prints "run NEWSTALLS" or "skip REASON". a job that follows one that made no new save (MARKER == LAST,
# the run_state.json digest) adds one stall; two stalls stop the chain.
chain_precheck() {
  local final=$1 stop=$2 halt=$3 marker=$4 last=$5 stalls=$6 index=$7 maxjobs=$8
  if [ "$final" = 1 ]; then echo "skip done"; return; fi
  if [ "$index" -gt 0 ]; then
    if [ "$stop" = 1 ]; then echo "skip stop-file"; return; fi
    if [ "$halt" = 1 ]; then echo "skip halted"; return; fi
    if [ "$index" -ge "$maxjobs" ]; then echo "skip max-jobs"; return; fi
    if [ "$marker" = "$last" ]; then stalls=$(( stalls + 1 )); else stalls=0; fi
    if [ "$stalls" -ge 2 ]; then echo "skip stalled"; return; fi
    echo "run $stalls"; return
  fi
  echo "run 0"
}

# state_get FILE KEY / state_put FILE KEY VALUE: the chain's key=value state file
state_get() { [ -f "$1" ] && sed -n "s/^$2=//p" "$1" | tail -n 1; }
state_put() {
  local f=$1 k=$2 v=$3
  touch "$f"
  grep -v "^$k=" "$f" > "$f.tmp" 2>/dev/null
  echo "$k=$v" >> "$f.tmp"
  mv "$f.tmp" "$f"
}

# corpus_decide RC: what a corpus job does with the driver's exit code: done (0), resubmit (20, interrupted by
# SIGTERM, rerun the same command) or fail (anything else: do not rerun blindly)
corpus_decide() {
  case "$1" in
    0) echo done;;
    20) echo resubmit;;
    *) echo fail;;
  esac
}

# slurm_time_left_seconds STR: squeue's %L (D-HH:MM:SS, HH:MM:SS, MM:SS, SS, UNLIMITED, INVALID) as seconds, or
# exit 1 when it cannot be read. the job uses it to set its own save timer when --signal is not honoured.
slurm_time_left_seconds() {
  local s=$1 d=0 rest h=0 m=0 sec=0
  case "$s" in *-*) d=${s%%-*}; rest=${s#*-};; *) rest=$s;; esac
  case "$rest" in
    *:*:*) IFS=: read -r h m sec <<< "$rest";;
    *:*) IFS=: read -r m sec <<< "$rest";;
    ''|*[!0-9]*) return 1;;
    *) sec=$rest;;
  esac
  for v in "$d" "$h" "$m" "$sec"; do case "$v" in ''|*[!0-9]*) return 1;; esac; done
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$sec ))
}

# expand_read_pairs: the (checkpoint, tag) pairs of a reads job, one "CKPT TAG" per line, from PAIRS
# ("ckptA:tagA ckptB:tagB", at most 4) or else CKPT and TAG. exit 1 on none, on more than 4, or on a pair without a tag
expand_read_pairs() {
  local p n=0
  if [ -n "$PAIRS" ]; then
    for p in $PAIRS; do
      case "$p" in *:?*) ;; *) echo "expand_read_pairs: '$p' is not CKPT:TAG" >&2; return 1;; esac
      echo "${p%:*} ${p##*:}"; n=$(( n + 1 ))
    done
  elif [ -n "$CKPT" ] && [ -n "$TAG" ]; then
    echo "$CKPT $TAG"; n=1
  fi
  if [ "$n" -lt 1 ]; then echo "expand_read_pairs: give PAIRS='ckpt:tag ...' or CKPT and TAG" >&2; return 1; fi
  if [ "$n" -gt 4 ]; then echo "expand_read_pairs: $n pairs, a node has 4 GPUs" >&2; return 1; fi
}

# container_prefix: the command prefix that runs a command inside the image, empty when SIF is unset. the runtime is
# singularity (module load singularity) or apptainer, whichever exists. BIND is the comma list of bind mounts.
container_prefix() {
  [ -n "$SIF" ] || return 0
  local rt
  rt=$(command -v singularity || command -v apptainer) || { echo "container_prefix: no singularity or apptainer on PATH (module load singularity)" >&2; return 1; }
  echo "$rt exec --nv --bind $BIND $SIF"
}

# wd_decide NOW START LAST_STEP LAST_SAVE STALL_S SAVE_S STARTUP_S EXTENDED: the progress watchdog's verdict, all times
# epoch seconds. LAST_STEP is when a new step line was last seen (0 = none since the job started), LAST_SAVE the mtime
# of the newest save in the run directory (0 = none). a job gets STARTUP_S for its first step line, and STARTUP_S
# on top of SAVE_S for its first save; EXTENDED=1 (the newest log line is not a step line: an eval or a save is
# running) doubles the step limit. prints ok | stalled-step | stalled-save
wd_decide() {
  local now=$1 start=$2 last_step=$3 last_save=$4 stall=$5 save=$6 startup=$7 ext=$8 base limit
  if [ "$last_step" -gt 0 ]; then base=$last_step; limit=$stall; else base=$start; limit=$startup; fi
  [ "$ext" = 1 ] && limit=$(( limit * 2 ))
  if [ $(( now - base )) -gt "$limit" ]; then echo stalled-step; return; fi
  if [ "$last_save" -ge "$start" ] && [ "$last_save" -gt 0 ]; then base=$last_save; limit=$save; else base=$start; limit=$(( save + startup )); fi
  if [ $(( now - base )) -gt "$limit" ]; then echo stalled-save; return; fi
  echo ok
}

# diverge_check CEILING K: reads training log text on stdin; prints "diverged REASON" when the last K "Loss (CE):"
# values are all nan, inf or above CEILING, else "ok". fewer than K values is ok.
diverge_check() {
  local ceiling=$1 k=$2
  grep -o 'Loss (CE): [^ ]*' | awk -v c="$ceiling" -v k="$k" '
    { v[NR] = tolower($3) }
    END {
      if (NR < k) { print "ok"; exit }
      bad = 0
      for (i = NR - k + 1; i <= NR; i++) {
        x = v[i]
        if (x ~ /nan|inf/) bad++
        else if (x + 0 > c + 0) bad++
      }
      if (bad == k) print "diverged last " k " loss values " v[NR - k + 1] " .. " v[NR] " (ceiling " c ")"; else print "ok"
    }'
}

# ce_median K: reads "Loss (CE): X" step lines or bare numbers on stdin; prints the median of the last K numeric values,
# or nothing when there are fewer than K (nan and inf are skipped)
ce_median() {
  local k=$1
  sed -e 's/.*Loss (CE): //' -e 's/ .*//' | awk -v k="$k" '
    $1 ~ /^-?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$/ { v[++n] = $1 + 0 }
    END {
      if (n < k) exit
      for (i = 0; i < k; i++) w[i] = v[n - k + 1 + i]
      for (i = 1; i < k; i++) { x = w[i]; j = i - 1; while (j >= 0 && w[j] > x) { w[j + 1] = w[j]; j-- } w[j + 1] = x }
      printf "%.4f\n", (k % 2) ? w[(k - 1) / 2] : (w[k / 2 - 1] + w[k / 2]) / 2
    }'
}

# rise_check MARGIN M MIN: reads the job's window medians on stdin, one per poll (watchdog.sh); prints "diverged ..." when
# at least MIN medians exist and the last M of them all exceed the best (lowest) median so far by more than MARGIN nats,
# else "ok". a relative rule for a loss that is finite but climbing, which the ceiling rule never sees
rise_check() {
  awk -v margin="$1" -v m="$2" -v min="$3" '
    { v[++n] = $1 + 0 }
    END {
      if (n < min || n < m) { print "ok"; exit }
      best = v[1]
      for (i = 2; i <= n; i++) if (v[i] < best) best = v[i]
      bad = 0
      for (i = n - m + 1; i <= n; i++) if (v[i] > best + margin) bad++
      if (bad == m) printf "diverged CE rose: last %d window medians %.3f .. %.3f against the best %.3f (margin %s)\n", m, v[n - m + 1], v[n], best, margin
      else print "ok"
    }'
}

# save_min_for GPUS TOKENS: the minutes a healthy job may go between saves: 2.5 times one save interval of TOKENS at the
# throughput floor (54k tokens per second per GPU), never under 45
save_min_for() {
  awk -v g="$1" -v t="$2" 'BEGIN { m = 2.5 * t / (g * 54000) / 60; r = int(m); if (r < m) r++; if (r < 45) r = 45; print r }'
}

# hist_append FILE JOBID STATE ELAPSED DECISION: one line per finished job, "JOBID STATE ELAPSED SHORT". STATE is
# ok (rank 0 exit 0, 10 or 20, or a clean stop), failed, stalled or diverged; SHORT is 1 for a job that ended
# in under 600 s without being done or stopped
hist_append() {
  local f=$1 job=$2 state=$3 el=$4 dec=$5 short=0
  case "$dec" in done|stop) ;; *) [ "$el" -lt 600 ] && short=1;; esac
  echo "$job $state $el $short" >> "$f"
}

# crashloop_check FILE: the crash loop rules over the job history. prints "ok" or "halt REASON": three failed,
# stalled or diverged jobs among the last five, or the last two jobs both short
crashloop_check() {
  local f=$1 bad last2
  [ -f "$f" ] || { echo ok; return; }
  bad=$(tail -n 5 "$f" | awk '$2 == "failed" || $2 == "stalled" || $2 == "diverged" { n++ } END { print n + 0 }')
  if [ "$bad" -ge 3 ]; then echo "halt $bad failed or stalled jobs among the last 5"; return; fi
  last2=$(tail -n 2 "$f" | awk '$4 == 1 { n++ } END { print n + 0 }')
  if [ "$(wc -l < "$f")" -ge 2 ] && [ "$last2" -ge 2 ]; then echo "halt two jobs in a row died within 10 minutes of start"; return; fi
  echo ok
}

# job_state RC VERDICT: the history state of a finished job. VERDICT is the watchdog's (ok, stalled-step,
# stalled-save, diverged)
job_state() {
  case "$2" in diverged) echo diverged; return;; stalled-*) echo stalled; return;; esac
  case "$1" in 0|10|20) echo ok;; *) echo failed;; esac
}

# corpus_stalled LOG DIR MINUTES: prints stalled or ok. DIR is DATA_ROOT. the driver touches DIR/run2_logs/last_child_output
# whenever a child process prints a line and when a stage starts or ends (its own 300 s heartbeat lines do not), so when
# that file exists it alone decides: older than MINUTES is a hung child, whatever the heartbeat keeps writing to LOG.
# before the file exists (a driver that has not reached a stage yet) the old rule holds: neither LOG nor any file under
# DIR changed in MINUTES
corpus_stalled() {
  local log=$1 dir=$2 min=$3 mark=$2/run2_logs/last_child_output
  if [ -f "$mark" ]; then
    if [ -n "$(find "$mark" -mmin -"$min" -print 2>/dev/null)" ]; then echo ok; else echo stalled; fi
    return
  fi
  if [ -f "$log" ] && [ -n "$(find "$log" -mmin -"$min" -print 2>/dev/null)" ]; then echo ok; return; fi
  if [ -d "$dir" ] && [ -n "$(find "$dir" -type f -mmin -"$min" -print -quit 2>/dev/null)" ]; then echo ok; return; fi
  echo stalled
}

# stop_after_m STR: the token tag (millions) a STOP_AFTER value names: 4000M, 4B (billions), or a bare number (7 or
# more digits = tokens, fewer = millions). exit 1 on anything else (a regex is not accepted: a save lands a log step
# past its grid point, tok4003M for 4000M, so the rule is numeric, first save at or past the number)
stop_after_m() {
  local s=$1 n
  case "$s" in
    *[Bb]) n=${s%?}; case "$n" in ''|*[!0-9]*) return 1;; esac; n=$(( 10#$n * 1000 ));;
    *[Mm]) n=${s%?}; case "$n" in ''|*[!0-9]*) return 1;; esac; n=$(( 10#$n ));;
    ''|*[!0-9]*) return 1;;
    *) if [ "${#s}" -ge 7 ]; then n=$(( 10#$s / 1000000 )); else n=$(( 10#$s )); fi;;
  esac
  [ "$n" -gt 0 ] || return 1
  echo "$n"
}

# save_tags: reads trainer log text on stdin, prints the token tag (millions) of every "Checkpoint saved at" line, in
# order. the load and retention lines also carry checkpoint names and are not saves
save_tags() {
  sed -n 's/.*Checkpoint saved at .*checkpoint_evidence_tok\([0-9][0-9]*\)M_.*/\1/p'
}

# save_reached N: reads trainer log text on stdin; exit 0 when a save with tag >= N (millions) was logged
save_reached() {
  save_tags | awk -v n="$1" '$1 + 0 >= n + 0 { f = 1 } END { exit f ? 0 : 1 }'
}

# require_site_values: exit 1 naming every site value still holding its CHANGE placeholder (the cluster's
# scripts/slurm/mn5_env.local.sh was not filled in)
require_site_values() {
  local v bad=""
  for v in REPO DATA_ROOT SBATCH_ACCOUNT SBATCH_QOS; do
    case "${!v}" in ''|*CHANGE*) bad="$bad $v";; esac
  done
  [ -z "$bad" ] || { echo "site values not filled in:$bad (copy scripts/slurm/mn5_env.local.sh.example to scripts/slurm/mn5_env.local.sh on the cluster and edit it)" >&2; return 1; }
}

# train_step_id JOBID: the job step of the training srun ("JOBID.0" unless squeue lists another non batch step)
train_step_id() {
  local id
  id=$(timeout 30 squeue -s -h -j "$1" -o %i 2>/dev/null | grep -v -e batch -e extern | head -n 1)
  echo "${id:-$1.0}"
}

# corpus_mirror_wait LOG MINUTES: prints mirror-stalled when the last non empty line of LOG is a mirror "waiting Ns"
# line with N over MINUTES minutes (the driver waits on the pushed inbox and logs every 300 s, which keeps the file
# fresh for corpus_stalled), else ok
corpus_mirror_wait() {
  local log=$1 min=$2 last n
  [ -f "$log" ] || { echo ok; return; }
  last=$(tail -n 20 "$log" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -n 1)
  n=$(echo "$last" | sed -n 's/.*mirror: waiting \([0-9][0-9]*\)s for.*/\1/p')
  if [ -n "$n" ] && [ "$n" -gt $(( min * 60 )) ]; then echo mirror-stalled; else echo ok; fi
}

"""One supervision poll of the real run, from the user's own PC (WSL), over an ssh master socket.

Nothing here runs on the cluster and no credential leaves the PC. The user opens the master socket once by hand
(``scripts/slurm/README.md``, "Supervision"); every call here uses it with BatchMode, so it never prompts and exits 2
when the socket is dead. One poll is one ssh command (``scripts/slurm/status.sh`` does everything in one call).

    python scripts/ops/supervise.py [--act] [--summary] [--run run2] [--user U] [--sock ~/.ssh/mn5.sock]

Exit code 0 ok, 1 attention needed, 2 cannot reach the cluster or the queue could not be read (squeue failed: no
action is taken on a queue that was not seen). The last line printed is ``notify: ...``, one line for a phone push.
``--summary`` prints three human lines instead of the full verdict. ``--act`` performs the safe actions only: resubmit
a broken chain (only once the chain has started, with its recorded split, nodes and STOP_AFTER; ``START_INDEX=1`` so STOP,
HALT and the stall rules still apply; never when the chain state names an end, and at most once an hour), hard link the milestone saves (the first save at or past each multiple of
``--pin-every-m``, the retention rule) into ``$DATA_ROOT/pinned``, submit one batched reads job (with weight exports)
for kept saves not yet read, once four are due, the oldest has waited 12 hours or the run is over; a read is attempted
once, one that ran and left no result is reported and never retried by itself (a submission that failed is retried).
An action whose command exits nonzero is an alert. Settings come from flags, then the environment, then
``scripts/ops/ops.env`` (``--env-file``). Everything else, a HALT above all, is reported and never acted on, and so is
a chain a person cancelled (ended=cancelled or a last history line ``cancelled``). One poll at a time: a lock file beside the history
turns an overlapping poll into a no-op.

Args:
    --host, --user, --sock, --repo: where the run lives (env MN5_LOGIN_HOST, MN5_USER, SSH_SOCK, REMOTE_REPO).
    --run: the run name. --kind: TRAIN or CONTROL (what a resubmit queues; default CONTROL for a run named *_control). --chain-n: jobs per resubmitted chain.
    --target-tokens, --budget-gpu-hours, --min-rate-per-gpu, --save-stale-min, --disk-frac: the rule limits.
    --history, --ops-log, --attempted: local files, one JSON line per poll, one text line per poll, the reads
    already attempted. They default to ops_state/RUN_history.jsonl, RUN_ops.log, RUN_attempted.json (per run name).
"""
import argparse
import json
import os
import subprocess
import sys
import re
import shlex
import time

try:
    import fcntl
except ImportError:  # not on Windows; the supervisor runs in WSL
    fcntl = None

SSH_TIMEOUT = 240  # status.py sub timeouts sum to 130 s (tests/test_ops_tools.py checks the margin)
FINISHED = ("COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE",
            "PREEMPTED")
BAD_REASON = re.compile(r"QOSMax|QOSGrp|QOSUsage|AssocGrp|AssocMax|Held|PartitionTimeLimit|InvalidQOS|InvalidAccount|"
                        r"BadConstraints|PartitionConfig|NonZeroExitCode|ReqNodeNotAvail.*Reserved", re.I)
BAD_STATE = ("REQUEUE_HOLD", "SPECIAL_EXIT", "SUSPENDED", "STOPPED")

SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.,:-]+$")

SAFE_SSH = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15"]


class Unreachable(Exception):
    pass


def ssh_base(cfg):
    return ["ssh", "-S", cfg["sock"], "-o", "ControlMaster=no"] + SAFE_SSH + ["%s@%s" % (cfg["user"], cfg["host"])]


def ssh_run(cfg, command, timeout=SSH_TIMEOUT):
    """Run one command on the login node through the master socket; raises Unreachable on any ssh failure."""
    try:
        chk = subprocess.run(["ssh", "-S", cfg["sock"], "-O", "check", "%s@%s" % (cfg["user"], cfg["host"])],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Unreachable("ssh master check failed: %s" % e)
    if chk.returncode != 0:
        raise Unreachable("the ssh master socket %s is dead; open it again (README, Supervision)" % cfg["sock"])
    try:
        r = subprocess.run(ssh_base(cfg) + [command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Unreachable("ssh command timed out after %ds" % timeout)
    if r.returncode == 255:
        raise Unreachable("ssh failed: %s" % r.stderr.strip()[:200])
    return r.returncode, r.stdout, r.stderr


def parse_env_file(text):
    """KEY=VALUE lines of scripts/ops/ops.env (the file is also sourced by bash, so `export KEY=...`, quotes, comments
    and $HOME or ~ in a value are read the way bash would); returns a dict. Lines it cannot read are skipped."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val[:1] in ("'", '"'):
            q = val[0]
            end = val.find(q, 1)
            val = val[1:end] if end > 0 else val[1:]
            if q == "'":
                out[key] = val
                continue
        else:
            val = re.split(r"\s+#", val, 1)[0].strip()
        out[key] = os.path.expanduser(os.path.expandvars(val))
    return out


def load_env_file(path):
    try:
        with open(path) as f:
            return parse_env_file(f.read())
    except OSError:
        return {}


def parse_status(stdout):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("no JSON line in the status output")


# the figures of an alert that drift while its condition stands; everything else (exit codes, save tags, job ids) stays in
# the key, so two different alerts are never one
DRIFT = [
    (re.compile(r"rate \d+ tokens/s per GPU under \d+ for \d+ minutes"), "rate #"),
    (re.compile(r"no new save in \d+ minutes"), "no new save in # minutes"),
    (re.compile(r"CE rose [\d.]+ between polls \([\d.]+ to [\d.]+\)"), "CE rose #"),
    (re.compile(r"CE [\d.]+ is [\d.]+ above the best [\d.]+"), "CE above best #"),
    (re.compile(r"projected .* GPU hours over the .* budget; the token target that fits is \d+B"), "projected over budget #"),
    (re.compile(r"pending for \d+ hours"), "pending for # hours"),
    (re.compile(r"watchdog silent for \d+ minutes"), "watchdog silent for # minutes"),
    (re.compile(r"disk at \d+%"), "disk at #%"),
    (re.compile(r"\d+ step line\(s\)"), "# step line(s)"),
    (re.compile(r"\(\d+ read\(s\)"), "(# read(s)"),
]


def alert_key(text):
    """A stable id of an alert: its text with the drifting figures wiped (the relay pushes a persistent condition once per
    window and a different alert at once)."""
    for rx, sub in DRIFT:
        text = rx.sub(sub, text)
    return text[:200]


def default_kind(run):
    return "CONTROL" if run.endswith("_control") else "TRAIN"


def tag_label(tag_m):
    return "%dB" % (tag_m // 1000)


def per_gpu_rate(st):
    return st.get("tokens_per_sec_per_gpu")


def job_kind(j):
    """train, reads or corpus: status.py sets "kind" from the exact job name; a job without it is read by name."""
    k = j.get("kind")
    if k:
        return k
    name = j.get("name") or ""
    if name.startswith("run2_reads"):
        return "reads"
    if name.startswith("run2_corpus"):
        return "corpus"
    return "train"


def stop_after_m(text):
    """The token tag (millions) a STOP_AFTER value names (scripts/slurm/lib.sh stop_after_m): 760000M, 760B or a number
    of tokens (7 or more digits) or of millions; None when it is empty or not one of those."""
    t = (text or "").strip()
    m = re.match(r"^(\d+)([BbMm]?)$", t)
    if not m:
        return None
    n = int(m.group(1))
    if m.group(2) in ("B", "b"):
        return n * 1000
    if m.group(2) in ("M", "m"):
        return n
    return n // 1000000 if len(t) >= 7 else n


def ls_tag_m(st):
    ls = st.get("last_save") or {}
    return ls.get("tag_m") if ls.get("tag_m") is not None else -1


def tail_switch_due(st, chain, active):
    """The main run reached its planned stop (the first save at or past STOP_AFTER, 760B by default) and waits for the
    user's switch to the tail split: the chain ended on the STOP file, the main split is the chain's split, the last
    save is at or past the planned stop, and no job is queued or running. A decision for the user, not a failure."""
    if active or chain.get("split") != "run2_main_train" or not st.get("stop"):
        return False
    stop_m = stop_after_m(chain.get("stop_after"))
    ls = st.get("last_save") or {}
    return bool(stop_m) and ls.get("tag_m") is not None and ls["tag_m"] >= stop_m


def kept_saves(saves, every_m):
    """The milestone saves: the first save at or past each multiple of every_m (millions), the rule of
    sft.select_retention_deletions (a save lands a log step past its grid point, tok10003M for 10000M, so a tag that is
    a multiple is never the test). A save written from a nan state is never kept."""
    out, prev = [], 0
    for s in sorted(saves, key=lambda x: x["tag_m"]):
        if s.get("nan"):
            continue
        if s["tag_m"] // every_m > prev // every_m:
            out.append(s)
        prev = s["tag_m"]
    return out


def evaluate(st, history, cfg, now, attempted=None):
    """Rules over one status dict and the poll history; returns {"alerts": [...], "notes": [...], "actions": [...],
    "alerted_reads": [...]}. attempted is the local record of reads already tried ({"reads": {label: t}, "alerted":
    {label: t}}): a read is attempted once."""
    alerts, notes, actions = [], [], []
    attempted = attempted or {}
    tried = attempted.get("reads") or {}
    out = {"alerts": alerts, "notes": notes, "actions": actions, "alerted_reads": []}
    if st.get("squeue_ok") is False:
        alerts.append("squeue failed or timed out on the login node: the queue was not seen, no action taken")
        return out
    jobs = st.get("jobs") or []
    train_jobs = [j for j in jobs if job_kind(j) == "train"]
    reads_jobs = [j for j in jobs if job_kind(j) == "reads"]
    running = [j for j in train_jobs if j.get("state") == "RUNNING"]
    # any listed job that is not finished is the chain: pending, held, requeued, completing
    active = [j for j in train_jobs if j.get("state") not in FINISHED]
    chain = st.get("chain_state") or {}
    ended = chain.get("ended")
    done = bool(st.get("final_exists")) or ended == "done"
    halt = st.get("halt")
    # the last job's exit code counts only while no newer job is queued or running (status.py already scopes it to the
    # current chain's first job; a fresh chain pending must not keep alerting on the replaced chain's rc 30)
    exit30 = str(st.get("last_job_rc")) == "30" and not active
    last_err = (st.get("errors") or [None])[-1]
    kills = st.get("kill_lines") or []
    hist_tail = st.get("history_tail") or []
    last_hist = hist_tail[-1].split() if hist_tail else []
    cancelled = (ended == "cancelled") or (len(last_hist) >= 2 and last_hist[1] == "cancelled")
    if halt:
        alerts.append("HALT: %s (never restarted automatically)" % halt)
    elif cancelled and not active and not done:
        alerts.append("the chain was cancelled by hand (%s); never resubmitted automatically. To let the supervisor "
                      "resubmit, a person clears it on the login node: sed -i '/^ended=cancelled$/d' "
                      "ckpts/evidence_%s/chain_state && sed -i '$ s/^\\([0-9]*\\) cancelled /\\1 ok /' "
                      "ckpts/evidence_%s/chain_history" % ("history line: " + " ".join(last_hist) if last_hist else "chain state",
                                                          cfg.get("run", "run2"), cfg.get("run", "run2")))
    elif exit30:
        alerts.append("last job exited 30 (resume or switch verification failed); never rerun blindly%s" %
                      ((": " + last_err[:160]) if last_err else ""))
    elif done:
        notes.append("run finished")
    elif tail_switch_due(st, chain, active):
        alerts.append("tail switch due: the main run stopped after the save at tok%dM (the planned stop at %s, "
                      "STOP_AFTER); the last 5%% of the run is read from the tail split, and the switch is the user's "
                      "decision. The chain ended on purpose and is never resubmitted by the supervisor. To go on: "
                      "rm ckpts/evidence_%s/STOP, then SPLIT=run2_tail_train bash scripts/slurm/submit_chain.sh TRAIN <n> "
                      "(scripts/slurm/README.md, Tail switch)" % (st["last_save"]["tag_m"],
                                                                  tag_label(stop_after_m(chain.get("stop_after"))),
                                                                  cfg.get("run", "run2")))
    elif ended:
        if ended == "stop" and st.get("stop"):
            notes.append("STOP file present, chain ended on purpose")
        elif ended == "warm-exhausted" or st.get("warm_exhausted"):
            alerts.append("the warm split ran out: the switch decision is needed (chain ended, not restarted)")
        else:
            alerts.append("chain ended (%s) and the run is not done; never resubmitted automatically%s" %
                          (ended, (": " + kills[-1][:160]) if kills else ""))
    elif st.get("stop") and not active:
        notes.append("STOP file present, chain ended on purpose")
    elif not active:
        # a chain exists only once a job of it has started (the first job writes chain_id into the chain state and the
        # run saves): before the first launch, during the corpus build or the control, no train job is normal
        started = bool(chain.get("chain_id")) or bool(st.get("saves")) or bool(st.get("last_save"))
        if not started:
            notes.append("chain not started: no train job, no chain state and no save of %s yet" % cfg.get("run", "run2"))
        else:
            alerts.append("chain broken: no job pending or running and the run is not done")
            recent = [h for h in history if now - h.get("t", 0) <= 3600 and "resubmit" in (h.get("acted") or [])]
            split = chain.get("split")
            if recent:
                alerts.append("the chain was already resubmitted within the hour; not again (look at the log)")
            elif stop_after_m(chain.get("stop_after")) and ls_tag_m(st) >= stop_after_m(chain.get("stop_after")):
                alerts.append("the chain's planned stop (STOP_AFTER %s) was reached at the save tok%dM and nothing is "
                              "queued: a switch decision for the user, not resubmitted (a resubmit with that STOP_AFTER "
                              "would stop again at once)" % (chain.get("stop_after"), ls_tag_m(st)))
            elif not split or not SAFE_VALUE.match(split):
                alerts.append("the chain state records no usable split: not resubmitting by itself (a resubmit without "
                              "the recorded split would restart the warm-up from the seed)")
            else:
                act = {"kind": "resubmit", "split": split}
                for key in ("stop_after", "nodes"):
                    val = chain.get(key)
                    if val and SAFE_VALUE.match(val):
                        act[key] = val
                actions.append(act)

    for j in jobs:
        if j.get("state") == "PENDING":
            reason = j.get("reason") or ""
            if BAD_REASON.search(reason):
                alerts.append("job %s (%s) pending with a reason that does not clear by itself: %s" %
                              (j.get("id"), j.get("name"), reason))
            elif (j.get("pending_s") or 0) > cfg.get("pending_hours", 24) * 3600 and "Dependency" not in reason:
                alerts.append("job %s (%s) pending for %d hours (%s)" % (j.get("id"), j.get("name"),
                                                                       j["pending_s"] // 3600, reason))
        elif j.get("state") in BAD_STATE:
            alerts.append("job %s (%s) is %s" % (j.get("id"), j.get("name"), j.get("state")))
    for tag, why in sorted(((st.get("corpus") or {}).get("halt") or {}).items()):
        alerts.append("corpus stage group %s halted: %s" % (tag, why))

    wd = st.get("watchdog")
    if wd and wd.get("verdict") not in (None, "ok") and running:
        alerts.append("watchdog verdict %s: %s" % (wd["verdict"], wd.get("detail")))
    if running:
        job_age = max((_elapsed(j.get("elapsed")) or 0) for j in running)
        if job_age > 600:
            if wd is None:
                alerts.append("no watchdog_last file while a job runs: the watchdog is not running")
            elif wd.get("verdict") == "dead":
                alerts.append("the watchdog could not start: %s" % wd.get("detail"))
            elif wd.get("epoch") is not None and (st.get("time") or now) - wd["epoch"] > 300:
                alerts.append("watchdog silent for %d minutes while a job runs (dead or stuck)" %
                              (((st.get("time") or now) - wd["epoch"]) // 60))
    for line in st.get("history_tail") or []:
        p = line.split()
        if len(p) >= 2 and p[1] in ("stalled", "diverged"):
            notes.append("recent job %s %s" % (p[0], p[1]))

    window = [h for h in history if now - h.get("t", 0) <= 70 * 60 and h.get("running")]
    rate = per_gpu_rate(st)
    if running and rate is not None:
        recent = window + [{"t": now, "rate_per_gpu": rate}]
        span = now - min(h["t"] for h in recent)
        if len(recent) >= 2 and span >= 50 * 60 and all((h.get("rate_per_gpu") or 0) < cfg["min_rate"] for h in recent):
            alerts.append("rate %d tokens/s per GPU under %d for %d minutes" % (rate, cfg["min_rate"], span // 60))

    ls = st.get("last_save")
    if running:
        # the watchdog allowance: a job gets save_stale_min plus the startup allowance for its first save, and
        # save_stale_min between saves after that
        job_age = max((_elapsed(j.get("elapsed")) or 0) for j in running)
        save_age = ls["age_s"] if ls else None
        in_job = save_age is not None and save_age < job_age
        ref = save_age if in_job else job_age
        # the limits the job's own watchdog uses (status.py reports them from the chain state: the save limit scales with
        # the GPU count), never below the configured floor
        lim = st.get("limits") or {}
        stale = max(cfg["save_stale_min"], lim.get("save_min") or 0)
        startup = lim.get("startup_min") or cfg.get("startup_min", 30)
        limit = stale if in_job else stale + startup
        if ref > limit * 60:
            alerts.append("no new save in %d minutes while running" % limit)
    # a finite divergence the watchdog's ceiling rule cannot see: the median CE of the last 20 step lines (status.py) higher
    # than the previous poll's, or than the lowest such median in the log tail. the watchdog halts the job on a sustained
    # +1.0 over its own best; these alert earlier (+0.3 between polls, +0.5 over the tail's best). a healthy loss falls
    ce, best_ce = st.get("last_ce"), st.get("best_ce")
    if running and ce is not None:
        prev = [h for h in history if now - h.get("t", 0) <= 3 * 3600 and h.get("ce") is not None and h.get("running")]
        if prev and ce - prev[-1]["ce"] > cfg.get("ce_rise_poll", 0.3):
            alerts.append("CE rose %.2f between polls (%.3f to %.3f): a finite divergence?" % (ce - prev[-1]["ce"], prev[-1]["ce"], ce))
        elif best_ce is not None and ce - best_ce > cfg.get("ce_rise_best", 0.5):
            alerts.append("CE %.3f is %.2f above the best %.3f of the log tail: a finite divergence?" % (ce, ce - best_ce, best_ce))
    if st.get("nan_saves"):
        alerts.append("save(s) written from a nan loss: %s (never resume from one)" % ", ".join(st["nan_saves"][-2:]))
    if st.get("nan_steps"):
        alerts.append("%d step line(s) with a nan or inf loss in the log tail" % st["nan_steps"])
    disk = st.get("disk") or {}
    frac = disk.get("used_frac")
    if frac is not None and frac >= cfg["disk_frac"]:
        alerts.append("disk at %d%% of the %s TB quota" % (round(frac * 100), disk.get("quota_tb")))
    elif frac is None and disk.get("disk_unknown"):
        alerts.append("disk usage unknown (du timed out and bsc_quota gave nothing): the disk alarm is blind")
    if st.get("fits_budget") is False:
        alerts.append("projected %s GPU hours over the %s budget; the token target that fits is %.0fB" %
                      (st.get("projected_gpu_hours"), st.get("budget_gpu_hours"), (st.get("tokens_that_fit") or 0) / 1e9))
    if st.get("errors_after_last_step") and st.get("errors"):
        alerts.append("error in the trainer log after the last step: %s" % st["errors"][-1][:160])

    exports, pinned, reads_done = set(st.get("exports") or []), set(st.get("pinned") or []), set(st.get("reads_done") or [])
    run = cfg.get("run", "run2")
    kept = kept_saves(st.get("saves") or [], cfg["pin_every_m"])
    for s in kept:
        if os.path.basename(s["file"]) not in pinned:
            actions.append({"kind": "pin", "file": s["file"]})
    reads_pending = bool(reads_jobs)
    todo, failed = [], []
    for s in kept:
        label = tag_label(s["tag_m"])
        have = label in reads_done and ("%s_%s.pt" % (run, label)) in exports
        if have:
            continue
        if label in tried:
            failed.append((label, tried[label]))
        else:
            todo.append(s)
    # a reads job holds a whole 4 GPU exclusive node, and reads.sbatch takes up to four pairs per node: submit when four
    # reads are due, when the oldest due save is older than reads_wait_hours (12), or when the run is over (finished,
    # ended, or no train job left), so no read waits for ever and no node is billed for one pair every 2.8 hours
    run_over = done or bool(ended) or not active
    oldest_age = max((now - (s.get("mtime") or now) for s in todo), default=0)
    if todo and not reads_pending and (len(todo) >= 4 or oldest_age > cfg.get("reads_wait_hours", 12) * 3600 or run_over):
        actions.append({"kind": "reads", "pairs": [(s["file"], tag_label(s["tag_m"])) for s in todo[:4]]})
    elif todo and not reads_pending:
        notes.append("%d read(s) due, waiting for four or %dh (oldest %dh)" % (
            len(todo), cfg.get("reads_wait_hours", 12), oldest_age // 3600))
    already = attempted.get("alerted") or {}
    if not reads_pending:
        for label, t in failed:
            if now - t > 1800 and label not in already:
                alerts.append("reads for %s were attempted and left no result: see the reads job log; never retried by "
                              "itself (remove the entry from the attempted file or run it by hand)" % label)
                out["alerted_reads"].append(label)
    return out


def _elapsed(s):
    try:
        d = 0
        if "-" in s:
            d, s = s.split("-", 1)
        parts = [int(x) for x in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return int(d) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except (ValueError, AttributeError):
        return None


def milestone(prev, st):
    """A milestone line when the token count crossed a multiple of 100B since the previous poll, or the run finished."""
    if st.get("final_exists") and not (prev or {}).get("final"):
        return "run finished"
    a, b = (prev or {}).get("tokens"), st.get("tokens")
    if a is not None and b is not None and int(b // 100e9) > int(a // 100e9):
        return "%dB tokens reached" % (int(b // 100e9) * 100)
    return None


def summary_lines(st, verdict):
    state = "HALT" if st.get("halt") else ("DONE" if st.get("final_exists") else
            ("RUNNING" if any(j.get("state") == "RUNNING" and job_kind(j) == "train" for j in st.get("jobs") or []) else "IDLE"))
    tok = (st.get("tokens") or 0) / 1e9
    rate = st.get("tokens_per_sec")
    l1 = "%s %s: %.1fB of %.0fB tokens, %s tok/s (%s per GPU)" % (
        st.get("run"), state, tok, (st.get("target_tokens") or 0) / 1e9,
        "%.2fM" % (rate / 1e6) if rate else "n/a", "%.1fk" % (st["tokens_per_sec_per_gpu"] / 1e3) if st.get("tokens_per_sec_per_gpu") else "n/a")
    l2 = "ETA %s h, GPU hours %s of %s (projected %s)" % (st.get("eta_hours"), st.get("gpu_hours_used"),
                                                          st.get("budget_gpu_hours"), st.get("projected_gpu_hours"))
    l3 = "alerts: " + ("; ".join(verdict["alerts"]) if verdict["alerts"] else "none")
    return [l1, l2, l3]


def perform(actions, cfg, runner):
    """Run the safe actions through runner(command) -> (rc, out, err); returns a list of result strings."""
    out = []
    repo, kind, run = cfg["repo"], cfg["kind"], cfg["run"]
    # the environment is sourced with `;`, not `&&`: nothing depends on the exit code of a sourced file (mn5_env.sh used
    # to return 1 whenever CTR was empty, and the action silently never ran). a failed cd stops the action (97), and an
    # environment that left DATA_ROOT empty is refused (98) rather than writing under /
    head = "cd %s || exit 97; source scripts/slurm/mn5_env.sh >/dev/null; " % repo
    for a in actions:
        if a["kind"] == "resubmit":
            carried = "".join(" %s=%s" % (env_name, shlex.quote(a[key])) for key, env_name in
                              (("split", "SPLIT"), ("stop_after", "STOP_AFTER"), ("nodes", "NODES")) if a.get(key))
            cmd = head + "START_INDEX=1 RUN=%s%s bash scripts/slurm/submit_chain.sh %s %d" % (run, carried, kind, cfg["chain_n"])
        elif a["kind"] == "pin":
            cmd = head + ("[ -n \"$DATA_ROOT\" ] || exit 98; mkdir -p $DATA_ROOT/pinned && "
                          "( ln -f %s $DATA_ROOT/pinned/ || cp -n %s $DATA_ROOT/pinned/ )") % (a["file"], a["file"])
        elif a["kind"] == "reads":
            pairs = " ".join("%s:%s" % p for p in a["pairs"])
            read_config = cfg.get("read_config")
            cfg_env = (" READ_CONFIG=%s" % shlex.quote(read_config)) if read_config else ""
            cmd = head + ("mkdir -p logs && RUN=%s%s PAIRS='%s' EXPORT=1 sbatch --parsable "
                          "--job-name=run2_reads_%s scripts/slurm/reads.sbatch") % (run, cfg_env, pairs, run)
        else:
            continue
        rc, so, se = runner(cmd)
        out.append("%s rc %d %s" % (a["kind"], rc, (so or se).strip()[:120]))
    return out


def failed_actions(actions, results):
    """The actions whose result line says rc other than 0; results[i] belongs to actions[i] (perform keeps the order)."""
    bad = []
    for a, r in zip(actions, results):
        m = re.match(r"^\w+ rc (-?\d+)\b", r)
        if not m or int(m.group(1)) != 0:
            bad.append((a, r))
    return bad


def load_history(path):
    rows = []
    try:
        with open(path) as f:
            for line in f.readlines()[-200:]:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def load_attempted(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_attempted(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


def take_lock(path):
    """Non blocking exclusive lock on path; returns the open file (keep it referenced) or None when another poll holds it."""
    if fcntl is None:
        return open(os.devnull)
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def main(argv=None):
    # settings: a CLI flag wins over the process environment, which wins over scripts/ops/ops.env (the file bash scripts
    # source; its lines have no `export`, so a poll started from a plain shell would otherwise see none of them)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops.env"))
    known, _ = pre.parse_known_args(argv)
    env = load_env_file(known.env_file)
    env.update(os.environ)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--env-file", default=known.env_file, help="settings file (default scripts/ops/ops.env)")
    p.add_argument("--host", default=env.get("MN5_LOGIN_HOST", "alogin1.bsc.es"))
    p.add_argument("--user", default=env.get("MN5_USER", ""))
    p.add_argument("--sock", default=os.path.expanduser(env.get("SSH_SOCK", "~/.ssh/mn5.sock")))
    p.add_argument("--repo", default=env.get("REMOTE_REPO", ""))
    p.add_argument("--run", default="run2")
    p.add_argument("--kind", default=None, choices=["TRAIN", "CONTROL"],
                   help="what a resubmit queues (default: CONTROL for a run name ending in _control, else TRAIN)")
    p.add_argument("--chain-n", type=int, default=4)
    p.add_argument("--target-tokens", type=float, default=800e9)
    p.add_argument("--budget-gpu-hours", type=float, default=4500.0)
    p.add_argument("--quota-tb", type=float, default=4.0)
    p.add_argument("--min-rate-per-gpu", type=float, default=54000.0)
    p.add_argument("--save-stale-min", type=float, default=45.0)
    p.add_argument("--startup-min", type=float, default=30.0, help="the watchdog startup allowance for a job's first save")
    p.add_argument("--pending-hours", type=float, default=24.0)
    p.add_argument("--ce-rise-poll", type=float, default=0.3, help="alert when the CE median rose this much since the previous poll")
    p.add_argument("--ce-rise-best", type=float, default=0.5, help="alert when the CE median is this far above the log tail's best")
    p.add_argument("--reads-wait-hours", type=float, default=12.0, help="submit a reads job for fewer than four due saves after this long")
    p.add_argument("--disk-frac", type=float, default=0.9)
    p.add_argument("--pin-every-m", type=int, default=10000, help="pin and read the first save at or past each multiple of this (millions of tokens)")
    p.add_argument("--history", default=None)
    p.add_argument("--ops-log", default=None)
    p.add_argument("--attempted", default=None)
    p.add_argument("--act", action="store_true")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args(argv)
    if not a.user or not a.repo:
        print("set --user and --repo (or MN5_USER and REMOTE_REPO, in the environment or in %s)" % a.env_file, file=sys.stderr)
        print("notify: supervise not configured: set MN5_USER and REMOTE_REPO")
        return 2
    a.kind = a.kind or default_kind(a.run)
    a.history = a.history or "ops_state/%s_history.jsonl" % a.run
    a.ops_log = a.ops_log or "ops_state/%s_ops.log" % a.run
    a.attempted = a.attempted or "ops_state/%s_attempted.json" % a.run
    cfg = {"host": a.host, "user": a.user, "sock": a.sock, "repo": a.repo, "run": a.run, "kind": a.kind,
           "chain_n": a.chain_n, "min_rate": a.min_rate_per_gpu, "save_stale_min": a.save_stale_min,
           "startup_min": a.startup_min, "pending_hours": a.pending_hours, "ce_rise_poll": a.ce_rise_poll,
           "ce_rise_best": a.ce_rise_best, "reads_wait_hours": a.reads_wait_hours,
           "disk_frac": a.disk_frac, "pin_every_m": a.pin_every_m,
           "read_config": "config_run2_control.yaml" if a.kind == "CONTROL" else None}
    for path in (a.history, a.ops_log, a.attempted):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lock = take_lock(os.path.join(os.path.dirname(os.path.abspath(a.history)), "%s_supervise.lock" % a.run))
    if lock is None:
        print("another supervise poll of %s is running; nothing done" % a.run)
        print("notify: poll skipped, another poll is running")
        return 0
    now = int(time.time())
    cmd = "cd %s && bash scripts/slurm/status.sh --run %s --target-tokens %g --budget-gpu-hours %g --quota-tb %g" % (
        a.repo, a.run, a.target_tokens, a.budget_gpu_hours, a.quota_tb)
    try:
        rc, so, se = ssh_run(cfg, cmd)
        st = parse_status(so)
    except Unreachable as e:
        print("cannot reach the cluster: %s" % e)
        print("notify: MN5 unreachable: %s" % e)
        return 2
    except ValueError as e:
        print("status.sh gave no usable output (%s): %s" % (e, (so + se)[:300]))
        print("notify: status.sh failed")
        return 2
    if st.get("squeue_ok") is False:
        print("squeue failed on the login node: the queue state is unknown, no action taken")
        print("notify: ATTN %s squeue failed, queue unknown, nothing done" % a.run)
        return 2
    hist = load_history(a.history)
    attempted = load_attempted(a.attempted)
    verdict = evaluate(st, hist, cfg, now, attempted)
    ms = milestone(hist[-1] if hist else None, st)
    results = []
    acted = []
    if a.act and verdict["actions"]:
        try:
            results = perform(verdict["actions"], cfg, lambda c: ssh_run(cfg, c))
            bad = failed_actions(verdict["actions"], results)
        except Unreachable as e:
            results = ["action failed: %s" % e]
            bad = [(act, results[0]) for act in verdict["actions"]]
        # an action that did not run (rc other than 0: a failed cd, sbatch or ln, a refusing submit) is an alert, is not
        # recorded as acted (a resubmit may be tried again next poll) and a reads job whose sbatch failed is not marked
        # attempted, so the next poll submits it again
        failed_ids = [id(act) for act, _ in bad]
        for act, r in bad:
            verdict["alerts"].append("action %s failed: %s" % (act["kind"], r[:160]))
        for act in verdict["actions"]:
            if id(act) in failed_ids:
                continue
            acted.append(act["kind"])
            if act["kind"] == "reads":
                # recorded once the submission worked: a read that ran and left no result is never retried by itself
                for _, label in act["pairs"]:
                    attempted.setdefault("reads", {})[label] = now
    for label in verdict.get("alerted_reads") or []:
        attempted.setdefault("alerted", {})[label] = now
    if acted or verdict.get("alerted_reads"):
        save_attempted(a.attempted, attempted)
    with open(a.history, "a") as f:
        f.write(json.dumps({"t": now, "running": any(j.get("state") == "RUNNING" and job_kind(j) == "train"
                                                     for j in st.get("jobs") or []),
                            "tokens": st.get("tokens"), "rate_per_gpu": per_gpu_rate(st), "ce": st.get("last_ce"),
                            "final": bool(st.get("final_exists")), "alerts": verdict["alerts"],
                            "acted": acted}) + "\n")
    lines = summary_lines(st, verdict)
    attn = bool(verdict["alerts"])
    notify = ("ATTN " if attn else "OK ") + lines[0] + ((" | " + "; ".join(verdict["alerts"])) if attn else "")
    if ms:
        notify += " | milestone: " + ms
    with open(a.ops_log, "a") as f:
        f.write("%s exit %d %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), 1 if attn else 0, notify))
    if a.summary:
        print("\n".join(lines))
    else:
        print("\n".join(lines))
        for n in verdict["notes"]:
            print("note: " + n)
        wd = st.get("watchdog")
        if wd:
            print("watchdog: %s (%s)" % (wd["verdict"], wd["detail"]))
        if st.get("halt"):
            print("HALT reason: " + st["halt"])
        for act in verdict["actions"]:
            print("action %s: %s" % ("done" if a.act else "recommended", json.dumps(act)))
        for r in results:
            print("result: " + r)
    if attn:
        print("keys: " + " | ".join(alert_key(x) for x in verdict["alerts"]))
    print("notify: " + notify)
    return 1 if attn else 0


if __name__ == "__main__":
    sys.exit(main())

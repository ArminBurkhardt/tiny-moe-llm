"""One JSON line about the run, from the login node (stdlib only, Python 3.6 compatible).

Parsers are pure functions over text so tests/test_ops_tools.py can run them on fake logs.

Every external command runs under a hard timeout (``SUB_TIMEOUTS``); their sum stays well under the supervisor's ssh
timeout (240 s), so one status call always answers. A command that fails or times out is reported
(``squeue_ok``, ``sacct_ok``, ``disk.disk_unknown``), never replaced by an empty or invented answer.
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import time

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_TOKENS = re.compile(r"evidence tokens: ([0-9.]+)M")
STEP_TPS = re.compile(r"Tokens/sec: ([0-9.]+)")
STEP_NO = re.compile(r"Epoch (\d+) \| Step (\d+)")
SAVE_NAME = re.compile(r"checkpoint_evidence_tok(\d+)M_loss(nan|inf|-inf|[0-9.]+)\.pt$", re.I)
# trainer failures only: a traceback, a CUDA or NCCL error, an out of memory, a kill. the eval lines and NCCL WARN or
# process group teardown lines are never errors, and a nan loss counts on step lines only (NAN_STEP)
ERROR_PAT = re.compile(r"Traceback \(most recent|CUDA error|NCCL error|ncclSystemError|ncclInternalError|"
                       r"OutOfMemoryError|out of memory|\bOOM\b|RuntimeError|Killed|DUE TO TIME LIMIT|CANCELLED", re.I)
IGNORED = re.compile(r"\[eval|NCCL WARN|ProcessGroupNCCL.*(destroy|not been destroyed|not destroyed)|destroy_process_group",
                     re.I)
NAN_STEP = re.compile(r"Loss(?: \(CE\))?: (?:nan|inf|-inf)\b", re.I)
KILL_LINE = re.compile(r"\bKILL:")

SUB_TIMEOUTS = {"squeue": 15, "sacct": 25, "quota": 20, "du": 70}
RUN_KINDS = ("train", "reads", "corpus")


def strip(line):
    return ANSI.sub("", line).rstrip("\n")


def is_step(line):
    return bool(STEP_NO.search(line)) and "Tokens/sec" in line


def parse_steps(lines):
    """Step lines as (epoch, step, tokens_M or None, tokens_per_sec or None). Tokens are cumulative, the step number
    resets every epoch: anything about progress or ETA is computed from tokens, never from steps."""
    out = []
    for raw in lines:
        line = strip(raw)
        if not is_step(line):
            continue
        m = STEP_NO.search(line)
        t = STEP_TOKENS.search(line)
        r = STEP_TPS.search(line)
        out.append((int(m.group(1)), int(m.group(2)), float(t.group(1)) if t else None,
                    float(r.group(1)) if r else None))
    return out


def rate_over(steps, n):
    """Median Tokens/sec of the last n step lines that report a positive rate, or None."""
    rates = sorted(s[3] for s in steps[-n:] if s[3])
    if not rates:
        return None
    return rates[len(rates) // 2]


def last_eval(lines):
    for raw in reversed(lines):
        line = strip(raw)
        if "[eval" in line:
            return line[:300]
    return None


def is_error(line):
    return bool(ERROR_PAT.search(line)) and not IGNORED.search(line)


def error_lines(lines, keep=8):
    """The last `keep` error lines and whether any came after the last step line."""
    last_step = -1
    hits = []
    for i, raw in enumerate(lines):
        line = strip(raw)
        if is_step(line):
            last_step = i
        elif is_error(line):
            hits.append((i, line[:240]))
    recent = hits[-keep:]
    return [h[1] for h in recent], any(h[0] > last_step for h in recent)


def nan_steps(lines):
    """How many step lines in `lines` carry a nan or inf loss."""
    return sum(1 for raw in lines if is_step(strip(raw)) and NAN_STEP.search(strip(raw)))


def trainer_flags(lines):
    """KILL lines (the trainer's own kill rule) and whether a warm split ran out ("warm split exhausted")."""
    kills = [strip(r)[:240] for r in lines if KILL_LINE.search(strip(r))][-3:]
    exhausted = any("warm split exhausted" in r for r in lines)
    return kills, exhausted


def job_kind(name, run):
    """train (the run's own name), reads (run2_reads_<run>), corpus (run2_corpus_*) or None for any other job."""
    if name == run:
        return "train"
    if name == "run2_reads_" + run:
        return "reads"
    if name.startswith("run2_corpus_"):
        return "corpus"
    return None


def gpn_from_gres(text, default=4):
    """GPUs per node from squeue's %b (gres/gpu:4, gpu:h100:4, gres:gpu:4); default 4 (the billing unit is the node)."""
    m = re.search(r"gpu(?::[A-Za-z0-9_]+)?:(\d+)", text or "")
    return int(m.group(1)) if m else default


def submit_age(text, now):
    """Seconds since squeue's %V (YYYY-MM-DDTHH:MM:SS, the cluster's local time), or None."""
    try:
        return max(int(now - time.mktime(time.strptime(text.strip(), "%Y-%m-%dT%H:%M:%S"))), 0)
    except (ValueError, OverflowError):
        return None


def parse_squeue(text, now=None):
    """Jobs from squeue -o "%i|%j|%T|%M|%l|%D|%R|%V|%b". The last two fields are optional."""
    jobs = []
    now = time.time() if now is None else now
    for line in text.splitlines():
        p = line.strip().split("|")
        if len(p) >= 7:
            j = {"id": p[0], "name": p[1], "state": p[2], "elapsed": p[3], "limit": p[4], "nodes": p[5],
                 "reason": p[6]}
            try:
                n = int(p[5])
            except ValueError:
                n = 0
            j["gpus"] = n * gpn_from_gres(p[8] if len(p) > 8 else "")
            if len(p) > 7:
                j["pending_s"] = submit_age(p[7], now) if p[2] == "PENDING" else None
            jobs.append(j)
    return jobs


def parse_sacct(text):
    jobs = []
    for line in text.splitlines():
        p = line.strip().split("|")
        if len(p) >= 5 and "." not in p[0]:
            jobs.append({"id": p[0], "name": p[1], "state": p[2], "elapsed": p[3], "exit": p[4]})
    return jobs


def parse_tres_gpus(tres):
    """GPUs a sacct AllocTRES string (billing=160,cpu=80,gres/gpu=16,node=4) is billed for: the larger of the gres count
    and 4 per node (billing is per whole node: an exclusive one GPU job such as the corpus job bills its node's 4); None
    when neither is present."""
    m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", tres or "")
    gres = int(m.group(1)) if m else None
    m = re.search(r"(?:^|,)node=(\d+)", tres or "")
    nodes = int(m.group(1)) * 4 if m else None
    if gres is None and nodes is None:
        return None
    return max(gres or 0, nodes or 0)


def sacct_gpu_hours(text, run):
    """GPU hours from ``sacct -X -n -P -o JobID,JobName,ElapsedRaw,AllocTRES,State`` over the account: the whole account
    (every job counts against the allocation, killed, surplus, reads, corpus and control included), and the part of
    this run's own jobs. Returns (total, run_hours, by_kind)."""
    total, mine, by_kind = 0.0, 0.0, {}
    for line in text.splitlines():
        p = line.strip().split("|")
        if len(p) < 5 or "." in p[0]:
            continue
        try:
            sec = int(p[2])
        except ValueError:
            continue
        gpus = parse_tres_gpus(p[3])
        if gpus is None:
            gpus = 4
        h = sec * gpus / 3600.0
        total += h
        kind = job_kind(p[1], run) or "other"
        by_kind[kind] = by_kind.get(kind, 0.0) + h
        if kind == "train":
            mine += h
    return total, mine, {k: round(v, 1) for k, v in by_kind.items()}


def elapsed_seconds(s):
    """squeue %M / sacct Elapsed (D-HH:MM:SS, HH:MM:SS, MM:SS) as seconds, or None."""
    try:
        d = 0
        if "-" in s:
            d, s = s.split("-", 1)
        parts = [int(x) for x in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return int(d) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except ValueError:
        return None


def parse_gpu_hours(text, min_job=0):
    """Lines 'JOBID GPUS SECONDS GPUHOURS rc=N decision' as (hours total, last rc or None, last decision). The hours
    count every line; the rc and the decision are those of the last job whose id is at least min_job (the first job of
    the current chain, so an rc 30 of a replaced chain stops alerting once a fresh chain has started)."""
    total, rc, dec = 0.0, None, None
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 6:
            try:
                total += float(p[3])
            except ValueError:
                continue
            try:
                job = int(p[0])
            except ValueError:
                job = 0
            if job >= min_job:
                rc = p[4].replace("rc=", "")
                dec = p[5]
    return total, rc, dec


CE_VALUE = re.compile(r"Loss \(CE\): (nan|inf|-inf|-?[0-9.]+(?:[eE][-+]?[0-9]+)?)", re.I)


def ce_values(lines):
    """The CE of every step line, in order (nan and inf are skipped: nan_steps counts those)."""
    out = []
    for raw in lines:
        line = strip(raw)
        if not is_step(line):
            continue
        m = CE_VALUE.search(line)
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if v == v and abs(v) != float("inf"):
                out.append(v)
    return out


def median(vals):
    s = sorted(vals)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def ce_stats(values, k=20):
    """(latest, best): the median CE of the last k step lines, and the lowest such median over the values given (windows
    of k stepped by k/2). Both None with fewer than k values. A median, so one spiked step line moves nothing."""
    if len(values) < k:
        return None, None
    last = median(values[-k:])
    best = last
    for i in range(k, len(values) + 1, max(k // 2, 1)):
        m = median(values[i - k:i])
        if m is not None and m < best:
            best = m
    return round(last, 4), round(best, 4)


def save_min_for(gpus, tokens=1e9):
    """The minutes a healthy job may go between saves (train_job.sh, lib.sh save_min_for): 2.5 save intervals of `tokens`
    at the throughput floor of 54k tokens per second per GPU, never under 45."""
    if not gpus:
        return 45
    return max(45, int(-(-(2.5 * tokens / (gpus * 54000.0) / 60.0) // 1)))


def parse_state(text):
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def parse_watchdog(text):
    p = text.strip().split("|")
    if len(p) < 5:
        return None
    try:
        return {"epoch": int(p[0]), "verdict": p[1], "detail": p[2], "step_age_s": int(p[3]), "save_age_s": int(p[4])}
    except ValueError:
        return None


def projection(tokens_done, rate_total, gpus, gpu_hours_used, target, budget):
    """ETA to the target at the current rate, projected GPU hours, and the token target the budget fits. Tokens and the
    rate only: the step counter resets every epoch."""
    if not rate_total or not gpus or tokens_done is None:
        return {"eta_hours": None, "projected_gpu_hours": None, "fits_budget": None, "tokens_that_fit": None}
    remaining = max(target - tokens_done, 0.0)
    eta_h = remaining / rate_total / 3600.0
    proj = gpu_hours_used + eta_h * gpus
    per_gpu_hour_tokens = rate_total * 3600.0 / gpus
    fit = tokens_done + max(budget - gpu_hours_used, 0.0) * per_gpu_hour_tokens
    return {"eta_hours": round(eta_h, 1), "projected_gpu_hours": round(proj, 0), "fits_budget": proj <= budget,
            "tokens_that_fit": int(fit)}


UNITS = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}
SIZE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])(?:i?B)?$", re.I)


def parse_size(tok):
    m = SIZE.match(tok.strip())
    return float(m.group(1)) * UNITS[m.group(2).upper()] if m else None


def parse_quota(text, data_root):
    """(used_bytes, limit_bytes) of the bsc_quota row for the project disk, or None. Format unconfirmed: the row is the
    first line naming the project path prefix (the first four components of DATA_ROOT) with two sizes that carry a
    unit, used first and limit second; anything else returns None and the caller falls back to du."""
    prefix = "/".join(data_root.rstrip("/").split("/")[:4])
    for line in text.splitlines():
        if prefix and prefix in line:
            sizes = [parse_size(t) for t in re.split(r"[\s|]+", line)]
            sizes = [s for s in sizes if s is not None]
            if len(sizes) >= 2 and sizes[1] > 0:
                return sizes[0], sizes[1]
    return None


def run(cmd, timeout):
    """(returncode, stdout); returncode None on a timeout (the whole process group is killed). cmd is a shell string or
    an argument list."""
    try:
        p = subprocess.Popen(cmd, shell=isinstance(cmd, str), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, start_new_session=True)
    except OSError:
        return 127, ""
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            p.communicate(timeout=5)
        except Exception:
            pass
        return None, ""


def tail_lines(path, nbytes):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(size - nbytes, 0))
            data = f.read().decode("utf-8", "replace")
        lines = data.splitlines()
        return lines[1:] if size > nbytes else lines
    except OSError:
        return []


def read_text(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def du_total_kb(paths, timeout):
    """One du over every path (a hard link between the trees counts once). Returns {path: kb} and ok; ok is False on a
    timeout or when nothing was read."""
    rc, out = run(["du", "-sk"] + list(paths), timeout)
    got = {}
    for line in (out or "").splitlines():
        p = line.split("\t", 1)
        if len(p) == 2 and p[0].isdigit():
            got[p[1].strip()] = int(p[0])
    return got, rc is not None and bool(got)


def collect(a):
    now = time.time()
    run_name = a.run
    rundir = os.path.join(a.repo, "ckpts", "evidence_" + run_name)
    runlog = os.path.join(a.repo, "ckpts", "inject", run_name + ".log")
    user = os.environ.get("USER") or ""
    st = {"time": int(now), "run": run_name}
    rc, sq_txt = run('squeue -h %s -o "%%i|%%j|%%T|%%M|%%l|%%D|%%R|%%V|%%b"' % (("-u " + user) if user else ""),
                     SUB_TIMEOUTS["squeue"])
    st["squeue_ok"] = rc == 0
    sq = parse_squeue(sq_txt, now) if rc == 0 else []
    jobs = []
    for j in sq:
        k = job_kind(j["name"], run_name)
        if k:
            j["kind"] = k
            jobs.append(j)
    st["jobs"] = jobs
    st["jobs_other"] = len(sq) - len(jobs)
    st["recent_jobs"] = parse_sacct(run(
        'sacct -n -P -u %s -S now-3days -o JobID,JobName,State,Elapsed,ExitCode 2>/dev/null | grep run2 | tail -n 12' % user,
        SUB_TIMEOUTS["sacct"])[1])
    lines = tail_lines(runlog, 4 * 1024 * 1024)
    steps = parse_steps(lines)
    state = parse_state(read_text(os.path.join(rundir, "chain_state")))
    gpus = int(state.get("gpus", 0) or 0) or None
    train_running = [j for j in jobs if j["kind"] == "train" and j["state"] == "RUNNING"]
    if train_running and train_running[0].get("gpus"):
        gpus = train_running[0]["gpus"]
    gh_text = read_text(os.path.join(a.repo, "ckpts", "inject", run_name + "_gpu_hours.log"))
    try:
        first_job = int(state.get("chain_first_job") or 0)
    except ValueError:
        first_job = 0
    used, last_rc, last_dec = parse_gpu_hours(gh_text, first_job)
    used_source = "log"
    for j in train_running:
        sec = elapsed_seconds(j["elapsed"])
        if sec and j.get("gpus"):
            used += sec * j["gpus"] / 3600.0
    account = a.account or os.environ.get("SBATCH_ACCOUNT", "")
    if account and "CHANGE" not in account:
        src, sa_txt = run("sacct -X -n -P -a -A %s -S %s -o JobID,JobName,ElapsedRaw,AllocTRES,State 2>/dev/null"
                          % (account, a.since), SUB_TIMEOUTS["sacct"])
        st["sacct_ok"] = src == 0 and bool(sa_txt.strip())
        if st["sacct_ok"]:
            used, mine, by_kind = sacct_gpu_hours(sa_txt, run_name)
            used_source = "sacct"
            st["gpu_hours_by_kind"] = by_kind
            st["gpu_hours_run"] = round(mine, 1)
    else:
        st["sacct_ok"] = None
    st["gpu_hours_source"] = used_source
    tokens = steps[-1][2] * 1e6 if steps and steps[-1][2] is not None else None
    tps = rate_over(steps, a.lines)
    st["tokens"] = tokens
    st["tokens_per_sec"] = tps
    st["tokens_per_sec_per_gpu"] = round(tps / gpus) if tps and gpus else None
    st["gpus"] = gpus
    # the limits the job's own watchdog uses (train_job.sh stores them in the chain state), so the supervisor judges a
    # slow save interval by the same number; derived from the GPU count when an older state has none
    def state_int(key):
        try:
            return int(state.get(key) or 0) or None
        except ValueError:
            return None
    st["limits"] = {"save_min": state_int("save_min") or save_min_for(gpus), "startup_min": state_int("startup_min") or 30}
    all_ce = ce_values(lines)
    st["last_ce"], st["best_ce"] = ce_stats(all_ce)
    st["gpu_hours_used"] = round(used, 1)
    st["last_job_rc"] = last_rc
    st["last_job_decision"] = last_dec
    st.update({"target_tokens": a.target_tokens, "budget_gpu_hours": a.budget_gpu_hours})
    st.update(projection(tokens, tps, gpus, used, a.target_tokens, a.budget_gpu_hours))
    saves = []
    for p in glob.glob(os.path.join(rundir, "checkpoint_evidence_tok*M_loss*.pt")):
        m = SAVE_NAME.search(p)
        if m:
            saves.append({"tag_m": int(m.group(1)), "file": p, "mtime": int(os.path.getmtime(p)),
                          "nan": m.group(2).lower() in ("nan", "inf", "-inf")})
    saves.sort(key=lambda s: s["tag_m"])
    st["saves"] = saves
    st["last_save"] = dict(saves[-1], age_s=int(now - saves[-1]["mtime"])) if saves else None
    st["nan_saves"] = [os.path.basename(s["file"]) for s in saves if s["nan"]]
    st["final_exists"] = os.path.exists(os.path.join(rundir, "checkpoint_evidence_final.pt"))
    st["stop"] = os.path.exists(os.path.join(rundir, "STOP"))
    halt = os.path.join(rundir, "HALT")
    st["halt"] = read_text(halt).strip() if os.path.exists(halt) else None
    st["chain_state"] = state
    st["history_tail"] = read_text(os.path.join(rundir, "chain_history")).splitlines()[-5:]
    st["watchdog"] = parse_watchdog(read_text(os.path.join(rundir, "watchdog_last")))
    errs, after = error_lines(lines[-a.lines:])
    st["errors"] = errs
    st["errors_after_last_step"] = after
    st["nan_steps"] = nan_steps(lines[-a.lines:])
    kills, exhausted = trainer_flags(lines[-a.lines:])
    st["kill_lines"] = kills
    st["warm_exhausted"] = exhausted
    st["last_eval"] = last_eval(lines)
    st["exports"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.data_root, "exports", run_name + "_*.pt")))
    st["pinned"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.data_root, "pinned", "*.pt")))
    reads = []
    suffix = "_rank_none.json"
    for p in glob.glob(os.path.join(a.repo, "ckpts", "inject", run_name + "_*" + suffix)):
        label = os.path.basename(p)[len(run_name) + 1:-len(suffix)]
        if re.match(r"^(\d+B|final)$", label):
            reads.append(label)
    st["reads_done"] = sorted(reads)
    corpus = {"done": [], "halt": {}}
    for p in glob.glob(os.path.join(a.data_root, "slurm_state", "*.done")):
        corpus["done"].append(os.path.basename(p)[:-len(".done")])
    for p in glob.glob(os.path.join(a.data_root, "slurm_state", "*.halt")):
        corpus["halt"][os.path.basename(p)[:-len(".halt")]] = read_text(p).strip()[:200]
    corpus["done"].sort()
    st["corpus"] = corpus
    qrc, quota_txt = run("command -v bsc_quota >/dev/null && bsc_quota 2>/dev/null | head -n 30", SUB_TIMEOUTS["quota"])
    quota_txt = (quota_txt or "").strip()
    st["quota_text"] = quota_txt or None
    got, du_ok = du_total_kb([rundir, a.data_root], SUB_TIMEOUTS["du"])
    run_kb, data_kb = got.get(rundir), got.get(a.data_root)
    disk = {"run_dir_gb": round(run_kb / 1e6, 1) if run_kb else None,
            "data_root_gb": round(data_kb / 1e6, 1) if data_kb else None, "quota_tb": a.quota_tb,
            "used_frac": None, "source": None, "disk_unknown": True}
    parsed = parse_quota(quota_txt, a.data_root) if quota_txt else None
    if parsed:
        disk.update({"used_frac": round(parsed[0] / parsed[1], 3), "quota_tb": round(parsed[1] / 1e12, 2),
                     "source": "bsc_quota", "disk_unknown": False})
    elif du_ok and run_kb is not None and data_kb is not None:
        disk.update({"used_frac": round(((run_kb or 0) + (data_kb or 0)) / 1e9 / a.quota_tb, 3), "source": "du",
                     "disk_unknown": False})
    st["disk"] = disk
    return st


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--run", default="run2")
    p.add_argument("--lines", type=int, default=400)
    p.add_argument("--target-tokens", type=float, default=800e9)
    p.add_argument("--budget-gpu-hours", type=float, default=4500.0)
    p.add_argument("--quota-tb", type=float, default=4.0)
    p.add_argument("--account", default="", help="sacct account for the GPU hours (default $SBATCH_ACCOUNT)")
    p.add_argument("--since", default="2026-10-28", help="sacct start date of the allocation")
    a = p.parse_args(argv)
    print(json.dumps(collect(a), separators=(",", ":")))


if __name__ == "__main__":
    main()

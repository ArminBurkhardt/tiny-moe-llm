"""The status parser (scripts/slurm/status.py), the supervisor's rules and actions (scripts/ops/supervise.py) and the
transfer ledger and scripts (scripts/ops/), on fake logs, fake status dicts, a fake ssh runner and fake rsync, squeue
and sacct. No cluster, no GPU. Needs bash.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


status = load("status_mod", "scripts/slurm/status.py")
sup = load("supervise_mod", "scripts/ops/supervise.py")

STEP = ("\x1b[93mINFO: Epoch 0 | Step {s} | Loss: 5.0 | Loss (CE): {ce} | LR: 1e-4 | evidence tokens: {t}M | "
        "evidence stream: 1.0M | Tokens/sec: {r} | Peak Mem: 40 GB | Time: 1 min\x1b[0m")


def step_line(s, t, r, ce="5.0"):
    return STEP.format(s=s, t=t, r=r, ce=ce)


def write_exe(path, text):
    with open(path, "w", newline="\n") as f:
        f.write(text)
    os.chmod(path, 0o755)


def sh(script, **env):
    e = dict(os.environ)
    e.update(env)
    return subprocess.run(["bash", "-c", script], cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, timeout=120)


def test_status_parsers():
    lines = [step_line(10, 100.0, 900000), "INFO: [eval] epoch 0 step 10 | CE: 5.1000 | ppl: 1",
             step_line(20, 200.5, 1100000), "Traceback (most recent call last):", "RuntimeError: NCCL error",
             step_line(30, 301.0, 1000000), "INFO: Checkpoint saved at x"]
    steps = status.parse_steps(lines)
    assert [s[1] for s in steps] == [10, 20, 30] and steps[-1][2] == 301.0
    assert status.rate_over(steps, 400) == 1000000
    assert status.last_eval(lines).startswith("INFO: [eval]")
    errs, after = status.error_lines(lines)
    assert len(errs) == 2 and after is False
    errs, after = status.error_lines(lines + ["CUDA error: an illegal memory access"])
    assert after is True
    jobs = status.parse_squeue("123|run2|RUNNING|1-02:00:00|3-00:00:00|4|node1\n124|run2|PENDING|0:00|3-00:00:00|4|Dependency\n")
    assert [j["state"] for j in jobs] == ["RUNNING", "PENDING"] and jobs[0]["gpus"] == 16
    assert status.elapsed_seconds("1-02:00:00") == 93600 and status.elapsed_seconds("05:30") == 330
    total, rc, dec = status.parse_gpu_hours("11 16 3600 16.0 rc=20 resubmit\n12 16 7200 32.0 rc=0 done\n")
    assert total == 48.0 and rc == "0" and dec == "done"
    assert status.parse_state("a=1\nb=2\na=3\n")["a"] == "3"
    wd = status.parse_watchdog("1700000000|stalled-step|step age 1000s|1000|-1")
    assert wd["verdict"] == "stalled-step" and wd["save_age_s"] == -1 and status.parse_watchdog("x") is None
    p = status.projection(100e9, 1.0e6, 16, 400.0, 800e9, 4500.0)
    assert p["eta_hours"] == round(700e9 / 1e6 / 3600, 1) and p["fits_budget"] is True
    p = status.projection(100e9, 0.5e6, 16, 400.0, 800e9, 4500.0)
    assert p["fits_budget"] is False and p["tokens_that_fit"] < 800e9
    assert status.projection(None, None, None, 0, 800e9, 4500.0)["fits_budget"] is None
    assert status.SAVE_NAME.search("checkpoint_evidence_tok10000M_loss3.2000.pt").group(1) == "10000"
    assert status.SAVE_NAME.search("checkpoint_evidence_tok10003M_lossnan.pt").group(2) == "nan"

    # ETA from tokens and the rate: the step counter resets every epoch and must not matter
    e0 = status.parse_steps([step_line(900, 500000.0, 1.0e6)])
    e1 = status.parse_steps([step_line(3, 500000.0, 1.0e6)])
    assert e0[0][2] == e1[0][2] == 500000.0

    # L11: eval lines with nan, NCCL WARN and process group teardown warnings are not errors; a nan step loss is counted apart
    quiet = ["INFO: [eval] epoch 0 | CE: nan | selection: nan",
             "node1:123:456 [0] NCCL INFO ...", "NCCL WARN Cuda failure 'peer access is not supported'",
             "[W1027 ProcessGroupNCCL.cpp:1496] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. "
             "On normal program exit, the application should call destroy_process_group", "loss went to nan briefly"]
    errs, after = status.error_lines(quiet + [step_line(40, 1.0, 1)])
    assert errs == [] and after is False, errs
    for bad in ("Traceback (most recent call last):", "torch.OutOfMemoryError: CUDA out of memory", "RuntimeError: boom",
                "CUDA error: device-side assert triggered", "NCCL error: unhandled system error", "Killed"):
        assert status.error_lines([bad])[0] == [bad], bad
    assert status.nan_steps([step_line(1, 1.0, 1, ce="nan"), step_line(2, 1.0, 1), "INFO: [eval] Loss: nan"]) == 1
    kills, ex = status.trainer_flags(["INFO: KILL: fixed-target gain 0.1 under 0.5", "INFO: warm split exhausted"])
    assert len(kills) == 1 and ex is True and status.trainer_flags(["INFO: ok"]) == ([], False)


def test_status_collect_pieces():
    # H4: exact job names, GPU counts from the allocation, pending age
    now = time.mktime(time.strptime("2026-10-28T12:00:00", "%Y-%m-%dT%H:%M:%S"))
    txt = ("1|run2|RUNNING|10:00:00|3-00:00:00|4|n1|2026-10-28T01:00:00|gres/gpu:4\n"
           "2|run2|PENDING|0:00|3-00:00:00|4|Dependency|2026-10-28T01:00:00|gres/gpu:4\n"
           "3|run2_control|RUNNING|1:00:00|3-00:00:00|2|n2|2026-10-28T01:00:00|gres/gpu:4\n"
           "4|run2_corpus_qa|PENDING|0:00|3-00:00:00|1|QOSMaxWallDurationPerJobLimit|2026-10-27T10:00:00|gres/gpu:1\n"
           "5|run2_reads_run2|RUNNING|0:10:00|6:00:00|1|n3|2026-10-28T11:00:00|gpu:4\n"
           "6|other|RUNNING|0:10:00|6:00:00|1|n3|2026-10-28T11:00:00|\n")
    jobs = status.parse_squeue(txt, now)
    assert [j["gpus"] for j in jobs] == [16, 16, 8, 1, 4, 4]
    assert jobs[3]["pending_s"] == 26 * 3600 and jobs[0]["pending_s"] is None
    kinds = [status.job_kind(j["name"], "run2") for j in jobs]
    assert kinds == ["train", "train", None, "corpus", "reads", None], kinds
    assert status.job_kind("run2_control", "run2_control") == "train"
    assert status.job_kind("run2_reads_run2_control", "run2_control") == "reads"

    # M5: GPU hours from sacct over the account
    sacct = ("100|run2|7200|billing=160,cpu=80,gres/gpu=16,mem=1T,node=4|TIMEOUT\n"
             "101|run2_control|3600|billing=40,cpu=20,node=1|COMPLETED\n"
             "102|run2_reads_run2|3600|gres/gpu:h100=4,node=1|COMPLETED\n"
             "103|run2|abc|node=4|RUNNING\n"
             "100.batch|batch|7200|cpu=1|TIMEOUT\n")
    total, mine, by_kind = status.sacct_gpu_hours(sacct, "run2")
    assert total == 32.0 + 4.0 + 4.0 and mine == 32.0, (total, mine)
    assert by_kind == {"train": 32.0, "other": 0.0 + 4.0, "reads": 4.0} or by_kind["train"] == 32.0, by_kind
    assert status.parse_tres_gpus("billing=1,node=2") == 8 and status.parse_tres_gpus("cpu=1") is None

    # M4: bsc_quota rows, du counts a hard link once, a timeout is reported not hidden
    q = "Filesystem  Path  Used  Limit\ngpfs  /gpfs/projects/bsc12/run  1.5T  4.0T  37%\ngpfs  /gpfs/scratch/bsc12  1G  2G\n"
    assert status.parse_quota(q, "/gpfs/projects/bsc12/run2") == (1.5e12, 4.0e12)
    assert status.parse_quota("nothing here", "/gpfs/projects/bsc12/run2") is None
    d = tempfile.mkdtemp()
    a, b = os.path.join(d, "a"), os.path.join(d, "b")
    os.makedirs(a)
    os.makedirs(b)
    with open(os.path.join(a, "f"), "wb") as f:
        f.write(b"x" * 4 * 1024 * 1024)
    os.link(os.path.join(a, "f"), os.path.join(b, "f"))
    got, ok = status.du_total_kb([a, b], 20)
    assert ok and abs((got[a] + got[b]) - 4096) < 64, got
    rc, out = status.run("sleep 5; echo late", 1)
    assert rc is None and out == ""
    assert status.run("echo hi", 5) == (0, "hi\n")
    # the sub timeouts stay well under the supervisor's ssh timeout
    assert sum(status.SUB_TIMEOUTS.values()) + 40 <= sup.SSH_TIMEOUT, (status.SUB_TIMEOUTS, sup.SSH_TIMEOUT)

    # collect() with fake squeue / sacct on PATH: squeue failing is reported, not an empty queue
    fb = os.path.join(d, "bin")
    os.makedirs(fb)
    write_exe(os.path.join(fb, "squeue"), '#!/usr/bin/env bash\n[ -n "$FAKE_FAIL" ] && exit 1\ncat "$FAKE_SQ"\n')
    write_exe(os.path.join(fb, "sacct"), '#!/usr/bin/env bash\ncat "$FAKE_SA"\n')
    sq = os.path.join(d, "sq.txt")
    sa = os.path.join(d, "sa.txt")
    with open(sq, "w") as f:
        f.write(txt)
    with open(sa, "w") as f:
        f.write(sacct)
    repo = os.path.join(d, "repo")
    data = os.path.join(d, "data")
    rundir = os.path.join(repo, "ckpts", "evidence_run2")
    os.makedirs(rundir)
    os.makedirs(os.path.join(repo, "ckpts", "inject"))
    os.makedirs(os.path.join(data, "slurm_state"))
    os.makedirs(os.path.join(data, "exports"))
    open(os.path.join(rundir, "checkpoint_evidence_tok10003M_loss3.0000.pt"), "w").close()
    open(os.path.join(rundir, "checkpoint_evidence_tok10009M_lossnan.pt"), "w").close()
    with open(os.path.join(rundir, "chain_state"), "w") as f:
        f.write("gpus=16\nended=stop\n")
    open(os.path.join(data, "slurm_state", "qa.done"), "w").close()
    with open(os.path.join(data, "slurm_state", "bios.halt"), "w") as f:
        f.write("stalled: no growth\n")
    open(os.path.join(repo, "ckpts", "inject", "run2_10B_rank_none.json"), "w").close()
    open(os.path.join(repo, "ckpts", "inject", "run2_control_10B_rank_none.json"), "w").close()
    open(os.path.join(data, "exports", "run2_10B.pt"), "w").close()

    class A:
        pass
    args = A()
    args.repo, args.data_root, args.run, args.lines = repo, data, "run2", 400
    args.target_tokens, args.budget_gpu_hours, args.quota_tb, args.account, args.since = 800e9, 4500.0, 4.0, "acct", "2026-10-28"
    old_path = os.environ["PATH"]
    os.environ.update(PATH=fb + os.pathsep + old_path, FAKE_SQ=sq, FAKE_SA=sa, USER="u")
    try:
        st = status.collect(args)
        assert st["squeue_ok"] is True and [j["kind"] for j in st["jobs"]] == ["train", "train", "corpus", "reads"], st["jobs"]
        assert st["jobs_other"] == 2 and st["gpus"] == 16 and st["gpu_hours_source"] == "sacct" and st["gpu_hours_used"] == 40.0, st
        assert st["nan_saves"] == ["checkpoint_evidence_tok10009M_lossnan.pt"] and st["saves"][1]["nan"] is True
        assert st["corpus"] == {"done": ["qa"], "halt": {"bios": "stalled: no growth"}}, st["corpus"]
        assert st["reads_done"] == ["10B"] and st["exports"] == ["run2_10B.pt"], (st["reads_done"], st["exports"])
        assert st["disk"]["disk_unknown"] is False and st["disk"]["source"] == "du"
        os.environ["FAKE_FAIL"] = "1"
        st = status.collect(args)
        assert st["squeue_ok"] is False and st["jobs"] == []
    finally:
        os.environ["PATH"] = old_path
        os.environ.pop("FAKE_FAIL", None)


def base_status(**kw):
    now = int(time.time())
    st = {"run": "run2", "time": now, "squeue_ok": True,
          "jobs": [{"id": "1", "name": "run2", "state": "RUNNING", "elapsed": "10:00:00"}],
          "tokens": 150e9, "tokens_per_sec": 1.0e6, "tokens_per_sec_per_gpu": 62500, "gpus": 16,
          "gpu_hours_used": 400.0, "budget_gpu_hours": 4500.0, "target_tokens": 800e9, "eta_hours": 180.0,
          "projected_gpu_hours": 3300.0, "fits_budget": True, "tokens_that_fit": 900e9,
          "saves": [], "last_save": {"tag_m": 150000, "age_s": 600, "file": "x"}, "final_exists": False, "stop": False,
          "halt": None, "chain_state": {}, "history_tail": [],
          "watchdog": {"epoch": now - 30, "verdict": "ok", "detail": "", "step_age_s": 5, "save_age_s": 600},
          "errors": [], "errors_after_last_step": False, "nan_saves": [], "nan_steps": 0, "kill_lines": [],
          "exports": [], "pinned": [], "reads_done": [], "last_job_rc": "20", "corpus": {"done": [], "halt": {}},
          "disk": {"used_frac": 0.3, "quota_tb": 4, "disk_unknown": False}}
    st.update(kw)
    return st


CFG = {"min_rate": 54000, "save_stale_min": 45, "disk_frac": 0.9, "pin_every_m": 10000, "repo": "/r", "run": "run2",
       "kind": "TRAIN", "chain_n": 4}


def test_rules():
    now = int(time.time())
    v = sup.evaluate(base_status(), [], CFG, now)
    assert v["alerts"] == [] and v["actions"] == []
    # chain broken: no job, not done -> alert and resubmit; never twice within the hour
    v = sup.evaluate(base_status(jobs=[]), [], CFG, now)
    assert any("chain broken" in a for a in v["alerts"]) and v["actions"] == [{"kind": "resubmit"}]
    v = sup.evaluate(base_status(jobs=[]), [{"t": now - 600, "acted": ["resubmit"]}], CFG, now)
    assert v["actions"] == [] and any("already resubmitted" in a for a in v["alerts"])
    # a pending job counts as an active chain
    v = sup.evaluate(base_status(jobs=[{"name": "run2", "state": "PENDING", "elapsed": "0:00", "reason": "Priority"}]), [], CFG, now)
    assert v["alerts"] == []
    # HALT and exit 30: alert, never resubmit; the trainer's last error line rides along
    v = sup.evaluate(base_status(jobs=[], halt="3 failed or stalled jobs"), [], CFG, now)
    assert any("HALT" in a for a in v["alerts"]) and v["actions"] == []
    v = sup.evaluate(base_status(jobs=[], last_job_rc="30", errors=["RuntimeError: switch refused"]), [], CFG, now)
    assert any("exited 30" in a and "switch refused" in a for a in v["alerts"]) and v["actions"] == []
    # a STOP file with no job is on purpose; a finished run is not broken
    v = sup.evaluate(base_status(jobs=[], stop=True), [], CFG, now)
    assert v["alerts"] == [] and v["actions"] == []
    v = sup.evaluate(base_status(jobs=[], final_exists=True), [], CFG, now)
    assert v["alerts"] == []
    # H1: a chain state that names an end is never resubmitted: halted leftovers, the job cap, a kill rule exit 10
    for ended in ("halted", "max-jobs", "stalled"):
        v = sup.evaluate(base_status(jobs=[], chain_state={"ended": ended}), [], CFG, now)
        assert v["actions"] == [] and any("chain ended (%s)" % ended in a for a in v["alerts"]), (ended, v)
    v = sup.evaluate(base_status(jobs=[], chain_state={"ended": "stop"}, kill_lines=["INFO: KILL: fixed-target gain 0.0"]), [], CFG, now)
    assert v["actions"] == [] and any("KILL" in a for a in v["alerts"]), v
    v = sup.evaluate(base_status(jobs=[], chain_state={"ended": "stop"}, stop=True), [], CFG, now)
    assert v["alerts"] == [] and v["actions"] == [] and v["notes"]
    v = sup.evaluate(base_status(jobs=[], chain_state={"ended": "warm-exhausted"}), [], CFG, now)
    assert v["actions"] == [] and any("warm split" in a for a in v["alerts"])
    v = sup.evaluate(base_status(jobs=[], chain_state={"ended": "done"}), [], CFG, now)
    assert v["alerts"] == [] and v["actions"] == []
    # C3: a failed squeue says nothing about the queue: alert, no action (the supervisor exits 2)
    v = sup.evaluate(base_status(jobs=[], squeue_ok=False), [], CFG, now)
    assert v["actions"] == [] and any("squeue" in a for a in v["alerts"])
    # slow for an hour: needs history spanning 50 minutes, all below the floor
    slow = base_status(tokens_per_sec_per_gpu=40000)
    hist = [{"t": now - 3500, "running": True, "rate_per_gpu": 41000}, {"t": now - 1800, "running": True, "rate_per_gpu": 39000}]
    assert any("rate" in a for a in sup.evaluate(slow, hist, CFG, now)["alerts"])
    assert not any("rate" in a for a in sup.evaluate(slow, hist[1:], CFG, now)["alerts"])
    hist[0]["rate_per_gpu"] = 60000
    assert not any("rate" in a for a in sup.evaluate(slow, hist, CFG, now)["alerts"])
    # no save: 45 minutes between saves once the job has one, 45 + 30 for its first (the watchdog allowance)
    v = sup.evaluate(base_status(last_save={"tag_m": 1, "age_s": 3000, "file": "x"}), [], CFG, now)
    assert any("no new save" in a for a in v["alerts"])
    run_job = lambda el: [{"name": "run2", "state": "RUNNING", "elapsed": el}]
    v = sup.evaluate(base_status(jobs=run_job("00:20:00"), last_save=None), [], CFG, now)
    assert not any("no new save" in a for a in v["alerts"])
    v = sup.evaluate(base_status(jobs=run_job("01:10:00"), last_save=None), [], CFG, now)
    assert not any("no new save" in a for a in v["alerts"]), "70 min is inside the 75 min first save allowance"
    v = sup.evaluate(base_status(jobs=run_job("01:20:00"), last_save=None), [], CFG, now)
    assert any("no new save in 75" in a for a in v["alerts"])
    v = sup.evaluate(base_status(jobs=run_job("01:10:00"), last_save={"tag_m": 1, "age_s": 3000, "file": "x"}), [], CFG, now)
    assert any("no new save in 45" in a for a in v["alerts"]), "a save inside this job: the 45 minute rule"
    # disk, budget, watchdog, errors, nan
    v = sup.evaluate(base_status(disk={"used_frac": 0.93, "quota_tb": 4, "disk_unknown": False}), [], CFG, now)
    assert any("disk" in a for a in v["alerts"])
    v = sup.evaluate(base_status(disk={"used_frac": None, "quota_tb": 4, "disk_unknown": True}), [], CFG, now)
    assert any("disk usage unknown" in a for a in v["alerts"])
    v = sup.evaluate(base_status(fits_budget=False, projected_gpu_hours=5200, tokens_that_fit=690e9), [], CFG, now)
    assert any("690B" in a for a in v["alerts"])
    v = sup.evaluate(base_status(watchdog={"epoch": now, "verdict": "stalled-step", "detail": "step age 1000s"}), [], CFG, now)
    assert any("watchdog" in a for a in v["alerts"])
    v = sup.evaluate(base_status(errors=["RuntimeError: NCCL"], errors_after_last_step=True), [], CFG, now)
    assert any("NCCL" in a for a in v["alerts"])
    v = sup.evaluate(base_status(nan_saves=["checkpoint_evidence_tok10003M_lossnan.pt"], nan_steps=3), [], CFG, now)
    assert any("nan loss" in a for a in v["alerts"]) and any("3 step line" in a for a in v["alerts"])
    # M7: a dead or silent watchdog while a job runs
    v = sup.evaluate(base_status(watchdog={"epoch": now - 900, "verdict": "ok", "detail": "", "step_age_s": 1, "save_age_s": 1}), [], CFG, now)
    assert any("silent" in a for a in v["alerts"])
    v = sup.evaluate(base_status(watchdog=None), [], CFG, now)
    assert any("no watchdog_last" in a for a in v["alerts"])
    v = sup.evaluate(base_status(watchdog={"epoch": now, "verdict": "dead", "detail": "mktemp failed"}), [], CFG, now)
    assert any("could not start" in a for a in v["alerts"])
    # M1: a pending job that cannot clear, or pending for a day
    v = sup.evaluate(base_status(jobs=[{"id": "9", "name": "run2", "state": "PENDING", "elapsed": "0:00",
                                        "reason": "QOSMaxWallDurationPerJobLimit"}]), [], CFG, now)
    assert any("does not clear" in a for a in v["alerts"]) and v["actions"] == []
    v = sup.evaluate(base_status(jobs=[{"id": "9", "name": "run2", "state": "PENDING", "elapsed": "0:00", "reason": "Priority",
                                        "pending_s": 30 * 3600}]), [], CFG, now)
    assert any("pending for 30 hours" in a for a in v["alerts"])
    v = sup.evaluate(base_status(jobs=[{"id": "9", "name": "run2", "state": "REQUEUE_HOLD", "elapsed": "0:00", "reason": ""}]), [], CFG, now)
    assert any("REQUEUE_HOLD" in a for a in v["alerts"]) and not any("chain broken" in a for a in v["alerts"])
    # H4: a corpus job or the control does not keep the main chain alive; a corpus halt is reported
    v = sup.evaluate(base_status(jobs=[{"name": "run2_corpus_qa", "state": "RUNNING", "elapsed": "1:00:00", "kind": "corpus"}]), [], CFG, now)
    assert any("chain broken" in a for a in v["alerts"])
    v = sup.evaluate(base_status(corpus={"done": [], "halt": {"bios": "mirror wait"}}), [], CFG, now)
    assert any("corpus stage group bios halted" in a for a in v["alerts"])

    # C2: the milestone is the first save at or past each multiple, never a tag that is a multiple
    saves = [{"tag_m": t, "file": "/r/ckpts/evidence_run2/checkpoint_evidence_tok%dM_loss3.0.pt" % t, "mtime": 1}
             for t in (9998, 10003, 10009, 19998, 20004, 20010)]
    assert [s["tag_m"] for s in sup.kept_saves(saves, 10000)] == [10003, 20004]
    nan = saves + [{"tag_m": 30001, "file": "x", "mtime": 1, "nan": True}]
    assert [s["tag_m"] for s in sup.kept_saves(nan, 10000)] == [10003, 20004]
    exact = [{"tag_m": t, "file": "f%d" % t, "mtime": 1} for t in (9000, 10000, 20000, 21000)]
    assert [s["tag_m"] for s in sup.kept_saves(exact, 10000)] == [10000, 20000]
    # pins and one batched reads job for the kept saves that lack reads or an export
    v = sup.evaluate(base_status(saves=saves, pinned=["checkpoint_evidence_tok10003M_loss3.0.pt"], reads_done=["10B"],
                                 exports=["run2_10B.pt"]), [], CFG, now)
    assert [a["kind"] for a in v["actions"]] == ["pin", "reads"], v["actions"]
    assert v["actions"][1]["pairs"] == [(saves[4]["file"], "20B")]
    v = sup.evaluate(base_status(saves=saves, jobs=[{"name": "run2", "state": "RUNNING", "elapsed": "1:00:00"},
                                                    {"name": "run2_reads_run2", "state": "PENDING", "elapsed": "0:00"}]), [], CFG, now)
    assert "reads" not in [a["kind"] for a in v["actions"]]
    # H2: a read is attempted once; a failure alerts once and is never resubmitted
    att = {"reads": {"10B": now - 7200, "20B": now - 7200}}
    v = sup.evaluate(base_status(saves=saves), [], CFG, now, att)
    assert [a["kind"] for a in v["actions"]] == ["pin", "pin"], v["actions"]
    assert sorted(v["alerted_reads"]) == ["10B", "20B"] and any("attempted and left no result" in a for a in v["alerts"])
    att["alerted"] = {"10B": now, "20B": now}
    v = sup.evaluate(base_status(saves=saves), [], CFG, now, att)
    assert v["alerted_reads"] == [] and not any("attempted" in a for a in v["alerts"])
    v = sup.evaluate(base_status(saves=saves), [], CFG, now, {"reads": {"10B": now - 60, "20B": now - 60}})
    assert not any("attempted" in a for a in v["alerts"]), "a read just submitted is still running"
    # milestones
    assert sup.milestone({"tokens": 99e9}, base_status(tokens=101e9)) == "100B tokens reached"
    assert sup.milestone({"tokens": 101e9}, base_status(tokens=150e9)) is None
    assert sup.milestone({"tokens": 5e9}, base_status(final_exists=True)) == "run finished"
    lines = sup.summary_lines(base_status(), {"alerts": []})
    assert len(lines) == 3 and "150.0B of 800B" in lines[0] and lines[2] == "alerts: none"


def test_actions_and_parse():
    seen = []

    def runner(cmd):
        seen.append(cmd)
        return 0, "4242\n", ""
    acts = [{"kind": "resubmit"}, {"kind": "pin", "file": "/r/a.pt"}, {"kind": "reads", "pairs": [("/r/a.pt", "10B")]}]
    res = sup.perform(acts, CFG, runner)
    assert len(res) == 3 and "START_INDEX=1" in seen[0] and "submit_chain.sh TRAIN 4" in seen[0]
    assert "ln -f /r/a.pt" in seen[1]
    assert "RUN=run2 PAIRS='/r/a.pt:10B' EXPORT=1 sbatch --parsable --job-name=run2_reads_run2 scripts/slurm/reads.sbatch" in seen[2], seen[2]
    # M8: the control run names its reads and jobs by its own run name
    seen.clear()
    sup.perform([{"kind": "reads", "pairs": [("/r/c.pt", "10B")]}], dict(CFG, run="run2_control", kind="CONTROL"), runner)
    assert "RUN=run2_control PAIRS='/r/c.pt:10B'" in seen[0] and "--job-name=run2_reads_run2_control" in seen[0], seen[0]
    st = sup.parse_status("junk\n" + json.dumps({"a": 1}) + "\n")
    assert st == {"a": 1}
    try:
        sup.parse_status("no json here")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_main_paths():
    """supervise.main with a fake ssh: squeue failing exits 2 with no action, a second poll is a no op, history and the
    attempted record are per run name, a reads action is recorded as attempted."""
    d = tempfile.mkdtemp()
    args = ["--user", "u", "--repo", "/r", "--sock", os.path.join(d, "s"), "--act",
            "--history", os.path.join(d, "ops", "x_history.jsonl"), "--ops-log", os.path.join(d, "ops", "x.log"),
            "--attempted", os.path.join(d, "ops", "x_attempted.json")]
    calls = []
    real = sup.ssh_run
    payload = {}

    def fake(cfg, command, timeout=sup.SSH_TIMEOUT):
        calls.append(command)
        if "status.sh" in command:
            return 0, json.dumps(payload) + "\n", ""
        return 0, "4242\n", ""
    sup.ssh_run = fake
    try:
        payload.update(base_status(jobs=[], squeue_ok=False))
        assert sup.main(args) == 2 and len(calls) == 1, calls
        # a chain broken with the queue read fine: one resubmit, recorded; the next poll will not repeat it
        calls[:] = []
        payload.clear()
        payload.update(base_status(jobs=[]))
        assert sup.main(args) == 1
        assert any("submit_chain.sh TRAIN" in c for c in calls), calls
        hist = [json.loads(line) for line in open(args[args.index("--history") + 1])]
        assert hist[-1]["acted"] == ["resubmit"]
        calls[:] = []
        assert sup.main(args) == 1 and not any("submit_chain.sh" in c for c in calls), calls
        # a reads action is recorded; the following poll does not submit it again
        saves = [{"tag_m": 10003, "file": "/r/ckpts/evidence_run2/checkpoint_evidence_tok10003M_loss3.0.pt", "mtime": 1}]
        payload.clear()
        payload.update(base_status(saves=saves, pinned=["checkpoint_evidence_tok10003M_loss3.0.pt"]))
        calls[:] = []
        sup.main(args)
        assert any("reads.sbatch" in c for c in calls), calls
        att = json.load(open(args[args.index("--attempted") + 1]))
        assert "10B" in att["reads"], att
        calls[:] = []
        sup.main(args)
        assert not any("reads.sbatch" in c for c in calls), calls
        # a poll that finds the lock taken does nothing
        if sup.fcntl is not None:
            held = sup.take_lock(os.path.join(d, "ops", "run2_supervise.lock"))
            assert held is not None
            calls[:] = []
            assert sup.main(args) == 0 and calls == [], calls
            assert sup.take_lock(os.path.join(d, "ops", "run2_supervise.lock")) is None
            held.close()
    finally:
        sup.ssh_run = real


def test_unreachable_exit_2():
    d = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ops", "supervise.py"), "--user", "u", "--repo", "/r",
                        "--sock", os.path.join(d, "dead.sock"), "--history", os.path.join(d, "h.jsonl"),
                        "--ops-log", os.path.join(d, "o.log"), "--attempted", os.path.join(d, "a.json")],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert "notify:" in r.stdout and "unreachable" in r.stdout.lower()


def test_ledger_and_transfers():
    d = tempfile.mkdtemp()
    led = os.path.join(d, "l.tsv")
    script = ("source scripts/ops/ops_lib.sh; LEDGER=%s; TRANSFER_BUDGET_GB=1; ledger_append in a 400000000; "
              "ledger_append out b 500000000; ledger_total_bytes; budget_check 50000000 && echo fits; "
              "budget_check 200000000 || echo refused") % led
    r = sh(script)
    assert r.stdout.split() == ["900000000", "fits", "refused"], (r.stdout, r.stderr)
    assert "would cross the transfer budget" in r.stderr
    assert open(led).read().splitlines()[0] == "time\tdirection\twhat\tbytes"
    r = sh("source scripts/ops/ops_lib.sh; rsync_bytes 'Number of files: 3\nTotal transfer size: 1,234,567 bytes\n'")
    assert r.stdout.strip() == "1234567"

    # M6: the bytes are recorded on failure too: from the stats when there are any, else the dry run estimate
    led2 = os.path.join(d, "l2.tsv")
    stats = "Total bytes sent: 2,000,000\\nTotal bytes received: 321\\n"
    r = sh("source scripts/ops/ops_lib.sh; LEDGER=%s; "
           "ledger_record_rsync in 'corpus x' 23 \"$(printf '%%b' '%s')\" 9000 sent; "
           "ledger_record_rsync in 'corpus y' 23 'rsync: connection unexpectedly closed' 9000 sent; "
           "ledger_record_rsync out 'pull z' 0 \"$(printf '%%b' '%s')\" 77 received" % (led2, stats, stats))
    rows = [ln.split("\t") for ln in open(led2).read().splitlines()[1:]]
    assert rows[0][1:] == ["in", "corpus x FAILED rc 23", "2000000"], rows
    assert rows[1][3] == "9000" and "FAILED rc 23" in rows[1][2] and "estimate" in rows[1][2], rows
    assert rows[2][1:] == ["out", "pull z", "321"], rows

    # the scripts with a fake rsync: a failed push still counts, a pull of a .pt with no .sha256 is refused
    fb = os.path.join(d, "bin")
    os.makedirs(fb)
    write_exe(os.path.join(fb, "rsync"), (
        '#!/usr/bin/env bash\n'
        'echo "$@" >> "$FAKE_RSYNC_LOG"\n'
        'case "$*" in *--dry-run*) echo "Total transfer size: 5,000 bytes"; exit 0;; esac\n'
        'last=${@: -1}\n'
        'case "$*" in\n'
        '  *"--include=*.pt"*) mkdir -p "$last"; echo w > "$last/run2_1B.pt"; [ -n "$FAKE_SHA" ] && (cd "$last" && sha256sum run2_1B.pt > run2_1B.pt.sha256);;\n'
        'esac\n'
        '[ -n "$FAKE_RSYNC_FAIL" ] && { echo "rsync error: unexplained error"; exit 23; }\n'
        'echo "Total bytes sent: 111"; echo "Total bytes received: 222"\n'))
    env = dict(PATH=fb + os.pathsep + os.environ["PATH"], LEDGER=os.path.join(d, "led.tsv"), FAKE_RSYNC_LOG=os.path.join(d, "rsync.log"),
               HOME=d)
    src = os.path.join(d, "corpus")
    os.makedirs(os.path.join(src, "prepared_run2"))
    r = sh("bash scripts/ops/push_corpus.sh %s prepared_run2" % src, FAKE_RSYNC_FAIL="1", **env)
    assert r.returncode == 1 and "rerun the same line" in r.stderr, (r.stdout, r.stderr)
    rows = open(env["LEDGER"]).read().splitlines()
    assert rows[-1].split("\t")[1] == "in" and "FAILED rc 23" in rows[-1] and rows[-1].split("\t")[3] == "5000", rows
    assert "--partial-dir=.rsync-partial" in open(env["FAKE_RSYNC_LOG"]).read()
    assert "--partial " not in open(env["FAKE_RSYNC_LOG"]).read()
    r = sh("bash scripts/ops/push_corpus.sh %s prepared_run2" % src, **env)
    assert r.returncode == 0 and open(env["LEDGER"]).read().splitlines()[-1].split("\t")[3] == "111", (r.stdout, r.stderr)
    # the transfer host has its own socket option, never the login one
    assert "ControlPath" not in open(env["FAKE_RSYNC_LOG"]).read()
    r = sh("source scripts/ops/ops_lib.sh; SSH_SOCK_TRANSFER=/x/t.sock SSH_SOCK=/x/l.sock; ops_ssh_opts", **env)
    assert "ControlPath" not in r.stdout and "no transfer master socket" in r.stderr
    sock = os.path.join(d, "t.sock")
    import socket
    s = socket.socket(socket.AF_UNIX)
    s.bind(sock)
    r = sh("source scripts/ops/ops_lib.sh; SSH_SOCK_TRANSFER=%s SSH_SOCK=/x/l.sock; ops_ssh_opts" % sock, **env)
    assert "ControlPath=%s" % sock in r.stdout and "/x/l.sock" not in r.stdout
    s.close()
    # L9: a pulled .pt without its .sha256 is refused (and never uploaded)
    local = os.path.join(d, "pulled")
    r = sh("bash scripts/ops/pull_exports.sh %s" % local, **env)
    assert r.returncode == 1 and "has no .sha256" in r.stderr, (r.returncode, r.stdout, r.stderr)
    r = sh("bash scripts/ops/pull_exports.sh %s" % local, FAKE_SHA="1", **env)
    assert r.returncode == 0 and "pulled to" in r.stdout, (r.stdout, r.stderr)
    # H5: push_code never sends the cluster site overlay
    assert "mn5_env\\.local\\.sh" in open(os.path.join(ROOT, "scripts", "ops", "push_code.sh")).read()


def test_review_fixes():
    now = int(time.time())
    # ops.env: assignments without export, quotes, comments, $HOME
    env = sup.parse_env_file("# c\nMN5_USER=bob\nexport REMOTE_REPO='/p/a b'  \nSSH_SOCK=$HOME/.ssh/s # note\nQ=\"x y\"\nbad line\n")
    assert env["MN5_USER"] == "bob" and env["REMOTE_REPO"] == "/p/a b" and env["Q"] == "x y", env
    assert env["SSH_SOCK"] == os.path.expandvars("$HOME/.ssh/s"), env
    d = tempfile.mkdtemp()
    ef = os.path.join(d, "ops.env")
    with open(ef, "w") as f:
        f.write("MN5_USER=u\nREMOTE_REPO=/r\n")
    real, calls = sup.ssh_run, []
    sup.ssh_run = lambda cfg, c, timeout=0: (calls.append((cfg, c)) or (_ for _ in ()).throw(sup.Unreachable("dead")))
    old = {k: os.environ.pop(k, None) for k in ("MN5_USER", "REMOTE_REPO")}
    try:
        rc = sup.main(["--env-file", ef, "--history", os.path.join(d, "h.jsonl"), "--ops-log", os.path.join(d, "o.log"),
                       "--attempted", os.path.join(d, "a.json")])
        assert rc == 2 and calls and calls[0][0]["user"] == "u" and calls[0][0]["repo"] == "/r", calls
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ops", "supervise.py"), "--env-file", os.path.join(d, "none")],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=dict(os.environ))
        assert r.returncode == 2 and "notify:" in r.stdout, (r.stdout, r.stderr)
    finally:
        sup.ssh_run = real
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v

    # finding 1: the action commands do not chain on the sourced file's exit code; a failed action alerts and is not acted
    cmds = []
    sup.perform([{"kind": "resubmit"}, {"kind": "pin", "file": "/r/a.pt"}, {"kind": "reads", "pairs": [("/r/a.pt", "10B")]}],
                CFG, lambda c: cmds.append(c) or (0, "", ""))
    for c in cmds:
        assert "mn5_env.sh >/dev/null; " in c and "mn5_env.sh >/dev/null &&" not in c and c.startswith("cd /r || exit 97;"), c
    acts = [{"kind": "resubmit"}, {"kind": "reads", "pairs": [("/r/a.pt", "10B")]}]
    assert [a["kind"] for a, _ in sup.failed_actions(acts, ["resubmit rc 1 x", "reads rc 0 42"])] == ["resubmit"]
    d = tempfile.mkdtemp()
    args = ["--user", "u", "--repo", "/r", "--sock", os.path.join(d, "s"), "--act", "--env-file", os.path.join(d, "none"),
            "--history", os.path.join(d, "h.jsonl"), "--ops-log", os.path.join(d, "o.log"), "--attempted", os.path.join(d, "a.json")]
    saves = [{"tag_m": 10003, "file": "/r/ckpts/evidence_run2/checkpoint_evidence_tok10003M_loss3.0.pt", "mtime": 1}]
    payload = base_status(saves=saves, pinned=["checkpoint_evidence_tok10003M_loss3.0.pt"])

    def fake(cfg, command, timeout=0):
        if "status.sh" in command:
            return 0, json.dumps(payload) + "\n", ""
        return 1, "", "sbatch: error"
    sup.ssh_run = fake
    try:
        assert sup.main(args) == 1
        assert not os.path.exists(os.path.join(d, "a.json")) or "10B" not in json.load(open(os.path.join(d, "a.json"))).get("reads", {})
        assert json.loads(open(os.path.join(d, "h.jsonl")).read().splitlines()[-1])["acted"] == []
    finally:
        sup.ssh_run = real

    # finding 2: the CE rules, and the status fields
    ce_lines = [step_line(i, 100.0 + i, 1.0e6, ce="3.0") for i in range(40)]
    assert status.ce_values(ce_lines) == [3.0] * 40
    assert status.ce_values([step_line(1, 1.0, 1.0, ce="nan")]) == []
    assert status.ce_stats([3.0] * 10) == (None, None)
    last, best = status.ce_stats([3.0] * 40 + [4.0] * 20)
    assert last == 4.0 and best == 3.0, (last, best)
    v = sup.evaluate(base_status(last_ce=3.9, best_ce=3.0), [], CFG, now)
    assert any("above the best" in a for a in v["alerts"])
    v = sup.evaluate(base_status(last_ce=3.4, best_ce=3.3), [{"t": now - 1800, "running": True, "ce": 3.0}], CFG, now)
    assert any("between polls" in a for a in v["alerts"])
    v = sup.evaluate(base_status(last_ce=3.01, best_ce=3.0), [{"t": now - 1800, "running": True, "ce": 3.05}], CFG, now)
    assert not any("CE" in a for a in v["alerts"]), "a falling or flat loss never alerts"

    # finding 3: a chain a person cancelled is reported, never resubmitted
    v = sup.evaluate(base_status(jobs=[], chain_state={"ended": "cancelled"}), [], CFG, now)
    assert v["actions"] == [] and any("cancelled by hand" in a for a in v["alerts"]), v
    v = sup.evaluate(base_status(jobs=[], history_tail=["5 ok 100 0", "6 cancelled 900 0"]), [], CFG, now)
    assert v["actions"] == [] and any("cancelled by hand" in a for a in v["alerts"]), v

    # finding 5: the save limit follows the job's own watchdog limits
    run_job = lambda el: [{"name": "run2", "state": "RUNNING", "elapsed": el}]
    st = base_status(jobs=run_job("02:00:00"), last_save={"tag_m": 1, "age_s": 3000, "file": "x"}, limits={"save_min": 97, "startup_min": 30})
    assert not any("no new save" in a for a in sup.evaluate(st, [], CFG, now)["alerts"])
    st["last_save"]["age_s"] = 98 * 60
    assert any("no new save in 97" in a for a in sup.evaluate(st, [], CFG, now)["alerts"])
    assert status.save_min_for(8) == 97 and status.save_min_for(16) == 49 and status.save_min_for(None) == 45

    # finding 6: an exclusive one GPU job bills its node
    assert status.parse_tres_gpus("cpu=80,gres/gpu=1,node=1") == 4 and status.parse_tres_gpus("gres/gpu=16,node=4") == 16
    assert status.parse_tres_gpus("gres/gpu=2") == 2

    # finding 10: reads wait for four due saves, 12 hours, or the end of the run
    fresh = [{"tag_m": 10003 + 10000 * i, "file": "f%d" % i, "mtime": now - 3600} for i in range(3)]
    v = sup.evaluate(base_status(saves=fresh, pinned=["f0", "f1", "f2"]), [], CFG, now)
    assert "reads" not in [a["kind"] for a in v["actions"]] and any("waiting for four" in n for n in v["notes"]), v
    v = sup.evaluate(base_status(saves=fresh + [{"tag_m": 40003, "file": "f3", "mtime": now - 60}], pinned=["f0", "f1", "f2", "f3"]), [], CFG, now)
    assert "reads" in [a["kind"] for a in v["actions"]]
    old_saves = [dict(s, mtime=now - 13 * 3600) for s in fresh[:1]]
    v = sup.evaluate(base_status(saves=old_saves, pinned=["f0"]), [], CFG, now)
    assert "reads" in [a["kind"] for a in v["actions"]]
    v = sup.evaluate(base_status(saves=fresh[:1], pinned=["f0"], final_exists=True), [], CFG, now)
    assert "reads" in [a["kind"] for a in v["actions"]]

    # finding 13: a stale rc 30 of a replaced chain does not alert while a newer job is queued, and status scopes it by job id
    v = sup.evaluate(base_status(last_job_rc="30", jobs=[{"name": "run2", "state": "PENDING", "elapsed": "0:00", "reason": "Priority"}]), [], CFG, now)
    assert not any("exited 30" in a for a in v["alerts"])
    txt = "11 16 3600 16.0 rc=30 fail\n12 16 3600 16.0 rc=20 resubmit\n"
    assert status.parse_gpu_hours(txt, 12)[1:] == ("20", "resubmit") and status.parse_gpu_hours(txt, 13)[1:] == (None, None)
    assert status.parse_gpu_hours(txt, 13)[0] == 32.0 and status.parse_gpu_hours(txt)[1] == "20"

    # finding 9: shipped shell files with carriage returns are refused
    d = tempfile.mkdtemp()
    for name, data in (("ok.sh", b"echo 1\n"), ("bad.sbatch", b"echo 1\r\n"), ("notes.txt", b"a\r\n")):
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
    with open(os.path.join(d, "list"), "w") as f:
        f.write("ok.sh\nbad.sbatch\nnotes.txt\n")
    r = sh("source scripts/ops/ops_lib.sh; cd %s; crlf_files list; echo rc=$?" % d)
    assert r.stdout.split() == ["bad.sbatch", "rc=1"], r.stdout
    ga = open(os.path.join(ROOT, ".gitattributes")).read()
    for pat in ("*.sbatch text eol=lf", "*.example text eol=lf", "*.def text eol=lf", "scripts/slurm/** text eol=lf", "scripts/ops/** text eol=lf"):
        assert pat in ga, pat
    assert "crlf_files" in open(os.path.join(ROOT, "scripts", "ops", "push_code.sh")).read()
    # finding B: the trainer and chain logs are pulled
    pe = open(os.path.join(ROOT, "scripts", "ops", "pull_exports.sh")).read()
    assert "'run2*.log'" in pe and "run2*.out" in pe and "'run2_*.log'" not in pe


def main():
    test_review_fixes()
    test_status_parsers()
    test_status_collect_pieces()
    test_rules()
    test_actions_and_parse()
    test_main_paths()
    test_ledger_and_transfers()
    test_unreachable_exit_2()
    print("test_ops_tools: ok")


if __name__ == "__main__":
    main()

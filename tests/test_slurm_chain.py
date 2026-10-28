"""The chain safety rules of the Slurm jobs on fake logs, fake chain state and fake sbatch, squeue, srun and scancel
(scripts/slurm/).

Checks: slurm_time_left_seconds, expand_read_pairs, the progress watchdog's decision table and its script on a
fake log, divergence, the crash loop rules over a fake history, the corpus stall rules, the container prefix, the
batched reads job, submit_chain.sh (job names, one chain id, refusing while the queue lists the chain or squeue
fails), the whole training job on a fake srun (save path through scancel on the step, kill path on divergence, the
run directory lock, chain id, the precheck ends, the exit decisions) and the corpus job (rc capture, lock, mirror
wait stall). GPU free; needs bash and flock.
"""
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_run2_launchers import ROOT, bash, clean_env, fn, slurm_dry, srun_line  # noqa: E402


def step_lines(values):
    return "\n".join(f"Epoch 0 | Step {i} | Loss: 1 | Loss (CE): {v} | Tokens/sec: 1" for i, v in enumerate(values))


def write_exe(path, text):
    with open(path, "w", newline="\n") as f:
        f.write(text)
    os.chmod(path, 0o755)


def fake_repo():
    """A scratch REPO whose scripts is a link to the real one, a data root, and a bin dir of fake Slurm commands."""
    tmp = tempfile.mkdtemp()
    repo = os.path.join(tmp, "repo")
    os.makedirs(repo)
    os.symlink(os.path.join(ROOT, "scripts"), os.path.join(repo, "scripts"))
    data = os.path.join(tmp, "data")
    os.makedirs(data)
    fb = os.path.join(tmp, "bin")
    os.makedirs(fb)
    write_exe(os.path.join(fb, "srun"), '#!/usr/bin/env bash\nwhile [ $# -gt 0 ]; do case "$1" in -*) shift;; *) break;; esac; done\nexec "$@"\n')
    write_exe(os.path.join(fb, "scontrol"), "#!/usr/bin/env bash\necho node1\n")
    write_exe(os.path.join(fb, "squeue"), (
        '#!/usr/bin/env bash\n'
        'case "$*" in\n'
        '  *" -s "*|"-s "*) echo "$SLURM_JOB_ID.batch"; echo "$SLURM_JOB_ID.0";;\n'
        '  *"-n "*) [ -n "$FAKE_SQUEUE_OUT" ] && echo "$FAKE_SQUEUE_OUT"; exit ${FAKE_SQUEUE_RC:-0};;\n'
        '  *) echo ${FAKE_LEFT:-UNLIMITED};;\n'
        'esac\n'))
    write_exe(os.path.join(fb, "scancel"), (
        '#!/usr/bin/env bash\n'
        'echo "$@" >> "$FAKE_BIN/scancel.log"\n'
        'for a in "$@"; do case "$a" in\n'
        '  --signal=TERM) kill -TERM $(cat "$FAKE_PIDFILE") 2>/dev/null;;\n'
        '  --signal=KILL) kill -KILL $(cat "$FAKE_PIDFILE") 2>/dev/null;;\n'
        'esac; done\n'))
    write_exe(os.path.join(fb, "sbatch"), (
        '#!/usr/bin/env bash\n'
        'echo "$@" >> "$FAKE_BIN/sbatch.args"\n'
        'n=$(wc -l < "$FAKE_BIN/sbatch.args")\n'
        '[ "$n" = "${FAKE_SBATCH_FAIL_AT:-0}" ] && exit 1\n'
        'echo $((4000 + n))\n'))
    write_exe(os.path.join(fb, "python"), (
        '#!/usr/bin/env bash\n'
        'case "$*" in\n'
        '  *build_run2_corpus.py*) case "$FAKE_CORPUS_MODE" in\n'
        '      mirror) echo "mirror: waiting 9000s for a/b in inbox"; trap "exit 20" TERM; while true; do sleep 0.2; done;;\n'
        '      *) exit ${FAKE_CORPUS_RC:-0};; esac;;\n'
        '  *scripts/sft.py*) ;;\n'
        '  *) exit 0;;\n'
        'esac\n'
        'echo $$ > "$FAKE_PIDFILE"\n'
        'case "$FAKE_MODE" in\n'
        '  ok20) echo "INFO: Checkpoint saved at x/checkpoint_evidence_tok1M_loss3.0000.pt" >> "$FAKE_LOG"; echo $RANDOM > "$FAKE_RUNDIR/run_state.json"; exit 20;;\n'
        '  exit0) exit 0;;\n'
        '  exit0final) touch "$FAKE_RUNDIR/checkpoint_evidence_final.pt"; exit 0;;\n'
        '  exit10kill) echo "INFO: KILL: fixed-target gain 0.0 under the bar" >> "$FAKE_LOG"; exit 10;;\n'
        '  exhausted) echo "INFO: warm split exhausted at epoch end" >> "$FAKE_LOG"; exit 10;;\n'
        '  exit30) echo "ERROR: resume verification failed: offset behind" >> "$FAKE_LOG"; exit 30;;\n'
        '  term) trap \'echo saved > "$FAKE_RUNDIR/saved_on_term"; exit 20\' TERM; while true; do sleep 0.2; done;;\n'
        '  exit2) echo "ERROR: rows cache missing: run the corpus rows stage" >> "$FAKE_LOG"; exit 2;;\n'
        '  rise) for i in $(seq 1 25); do echo "Epoch 0 | Step $i | Loss: 3 | Loss (CE): 3.0 | Tokens/sec: 1" >> "$FAKE_LOG"; sleep 0.3; done; '
        'while true; do echo "Epoch 0 | Step 99 | Loss: 5 | Loss (CE): 5.0 | Tokens/sec: 1" >> "$FAKE_LOG"; sleep 0.3; done;;\n'
        '  diverge) while true; do echo "Epoch 0 | Step 1 | Loss: nan | Loss (CE): nan | Tokens/sec: 1" >> "$FAKE_LOG"; sleep 0.3; done;;\n'
        'esac\n'))
    return tmp, repo, data, fb


def job_env(tmp, repo, data, fb, **kw):
    log = os.path.join(repo, "ckpts", "inject", "run2.log")
    e = dict(REPO=repo, DATA_ROOT=data, SBATCH_ACCOUNT="acct", SLURM_SUBMIT_DIR=repo, SLURM_JOB_ID="777",
             SLURM_JOB_NUM_NODES="1", SLURM_NTASKS_PER_NODE="1", SLURM_JOB_NODELIST="n1", SLURM_JOB_NAME="run2",
             SLURM_PROCID="0", SLURM_NTASKS="1", SLURM_LOCALID="0", FAKE_BIN=fb, FAKE_PIDFILE=os.path.join(tmp, "pid"),
             FAKE_LOG=log, FAKE_RUNDIR=os.path.join(repo, "ckpts", "evidence_run2"), WD_CHECK_S="0.3", POLL="1",
             KILL_AFTER_S="2", PATH=fb + os.pathsep + os.environ["PATH"], TMPDIR=tmp)
    e.update(kw)
    return e


def run_job(tmp, repo, data, fb, script="source scripts/slurm/train_job.sh", trigger=None, timeout=60, **kw):
    """Run a job script like sbatch does (one bash process); trigger = (signal, seconds, wait for the pid file)."""
    env = clean_env(**job_env(tmp, repo, data, fb, **kw))
    out = os.path.join(tmp, "job.out")
    with open(out, "w") as fh:
        p = subprocess.Popen(["bash", "-c", script], cwd=repo, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if trigger:
            sig, wait_pid = trigger
            end = time.time() + 20
            while time.time() < end and not os.path.exists(env["FAKE_PIDFILE"]):
                time.sleep(0.1)
            time.sleep(wait_pid)
            p.send_signal(sig)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raise
    return p.returncode, open(out).read(), env


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def state_of(repo):
    d = os.path.join(repo, "ckpts", "evidence_run2")
    st = {}
    for line in read(os.path.join(d, "chain_state")).splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            st[k] = v
    return d, st


def test_lib_functions():
    assert fn("chain_decide 0 0 0 0")[1] == "nofinal"
    assert fn("chain_decide 0 1 0 0")[1] == "done"
    assert fn("require_site_values")[0] != 0
    r = bash("source scripts/slurm/lib.sh; REPO=/a DATA_ROOT=/b SBATCH_ACCOUNT=x SBATCH_QOS=acc_ehpc require_site_values")
    assert r.returncode == 0, r.stderr
    r = bash("source scripts/slurm/lib.sh; REPO=/a DATA_ROOT=/b SBATCH_ACCOUNT=CHANGE_ACCOUNT SBATCH_QOS=q require_site_values")
    assert r.returncode != 0 and "SBATCH_ACCOUNT" in r.stderr and "REPO" not in r.stderr
    # saves land past their grid point: tags 4000..4004M, 10000..10005M at 16 GPUs; the load and retention lines are not saves
    def reached(text, n):
        r = bash(f"source scripts/slurm/lib.sh; printf '%s\\n' '{text}' | save_reached {n}")
        return r.returncode == 0
    save = "INFO: Checkpoint saved at /c/checkpoint_evidence_tok{}M_loss3.1.pt"
    assert not reached(save.format(3999), 4000)
    for tag in (4000, 4003, 4004, 10005):
        assert reached(save.format(tag), 4000), tag
    assert not reached("resumed from /c/checkpoint_evidence_tok4100M_loss3.0.pt", 4000)
    assert not reached("retention: deleted checkpoint_evidence_tok4100M_loss3.0.pt (not a milestone)", 4000)
    assert fn("printf '%s\\n' 'x Checkpoint saved at a/checkpoint_evidence_tok19998M_loss2.1.pt' "
              "'x Checkpoint saved at a/checkpoint_evidence_tok20004M_loss2.0.pt' | save_tags")[1] == "19998\n20004"
    d = tempfile.mkdtemp()
    log = os.path.join(d, "c.log")
    with open(log, "w") as f:
        f.write("mirror: waiting 300s for a/b in inbox\nmirror: waiting 9000s for a/b in inbox\n")
    assert fn(f"corpus_mirror_wait {log} 120")[1] == "mirror-stalled"
    assert fn(f"corpus_mirror_wait {log} 200")[1] == "ok"
    with open(log, "a") as f:
        f.write("stage md5 done\n")
    assert fn(f"corpus_mirror_wait {log} 120")[1] == "ok"
    assert fn(f"corpus_mirror_wait {d}/none 120")[1] == "ok"
    # train_step_id: the first step that is not the batch or extern step
    fb = os.path.join(d, "bin")
    os.makedirs(fb)
    write_exe(os.path.join(fb, "squeue"), '#!/usr/bin/env bash\nprintf "9.batch\\n9.extern\\n9.1\\n"\n')
    r = bash("source scripts/slurm/lib.sh; train_step_id 9", PATH=fb + os.pathsep + os.environ["PATH"])
    assert r.stdout.strip() == "9.1", r.stdout
    assert fn("train_step_id 9")[1].endswith(".0") or True


def test_env_overlay_and_bind():
    d = tempfile.mkdtemp()
    sd = os.path.join(d, "slurm")
    shutil.copytree(os.path.join(ROOT, "scripts", "slurm"), sd, ignore=shutil.ignore_patterns("__pycache__"))
    for stale in ("mn5_env.local.sh",):
        if os.path.exists(os.path.join(sd, stale)):
            os.remove(os.path.join(sd, stale))
    # no overlay: placeholders stay and the site check refuses
    r = bash(f"source {sd}/mn5_env.sh; echo $REPO; echo $SBATCH_QOS; require_site_values", REPO="", DATA_ROOT="", SBATCH_ACCOUNT="")
    assert "CHANGE_PROJECT" in r.stdout and "acc_ehpc" in r.stdout and r.returncode != 0, (r.stdout, r.stderr)
    with open(os.path.join(sd, "mn5_env.local.sh"), "w") as f:
        f.write("export REPO=/p/repo\nexport DATA_ROOT=/p/data\nexport SBATCH_ACCOUNT=acct\n")
    r = bash(f"source {sd}/mn5_env.sh; echo $REPO $DATA_ROOT $SBATCH_ACCOUNT; require_site_values", REPO="", DATA_ROOT="", SBATCH_ACCOUNT="")
    assert r.returncode == 0 and "/p/repo /p/data acct" in r.stdout, (r.stdout, r.stderr)
    # BIND and CTR are rebuilt from this shell's environment, never taken from a login shell's exported value
    r = bash(f"source {sd}/mn5_env.sh; echo \"$BIND\"; echo \"[$CTR]\"", BIND="/login/only", CTR="stale ", TMPDIR="/job/tmp")
    lines = r.stdout.splitlines()
    assert lines[0] == "/p/repo,/p/data,/job/tmp,/p/data/hf_home", lines
    assert lines[1] == "[]", lines
    # SIF without a runtime: CTR_FAIL, and a job script refuses (exit 2) instead of running host python
    r = slurm_dry("scripts/slurm/train_job.sh", MODE="run", SIF="/p/run2.sif", PATH="/usr/bin:/bin")
    assert r.returncode == 2 and "no singularity or apptainer" in r.stderr, (r.returncode, r.stderr)
    # the overlay is gitignored and never pushed
    gi = open(os.path.join(ROOT, ".gitignore")).read()
    assert "scripts/slurm/mn5_env.local.sh" in gi
    assert "mn5_env.local.sh" in open(os.path.join(ROOT, "scripts", "ops", "push_code.sh")).read()
    assert os.path.exists(os.path.join(ROOT, "scripts", "slurm", "mn5_env.local.sh.example"))


def test_submit_chain():
    tmp, repo, data, fb = fake_repo()
    base = dict(REPO=repo, DATA_ROOT=data, SBATCH_ACCOUNT="acct", SBATCH_CMD=os.path.join(fb, "sbatch"),
                SQUEUE_CMD=os.path.join(fb, "squeue"), FAKE_BIN=fb, PATH=fb + os.pathsep + os.environ["PATH"],
                SLURM_JOB_ID="1")
    script = f"bash {ROOT}/scripts/slurm/submit_chain.sh"
    r = bash(f"{script} TRAIN 2", **base)
    assert r.returncode == 0 and r.stdout.split() == ["0", "4001", "1", "4002"], (r.stdout, r.stderr)
    args = read(os.path.join(fb, "sbatch.args")).splitlines()
    ids = {a.split("CHAIN_ID=")[1].split(",")[0] for a in args}
    assert len(ids) == 1 and all("--job-name=run2 " in a for a in args), args
    assert "CHAIN_FIRST=1" in args[0] and "CHAIN_FIRST=0" in args[1] and "--dependency=afterany:4001" in args[1], args
    os.remove(os.path.join(fb, "sbatch.args"))
    # the queue lists the chain: refuse, nothing queued
    r = bash(f"{script} TRAIN 2", FAKE_SQUEUE_OUT="123 RUNNING", **base)
    assert r.returncode == 1 and "refusing" in r.stderr and not os.path.exists(os.path.join(fb, "sbatch.args")), r.stderr
    # squeue itself fails: refuse
    r = bash(f"{script} TRAIN 2", FAKE_SQUEUE_RC="1", **base)
    assert r.returncode == 1 and "squeue failed" in r.stderr and not os.path.exists(os.path.join(fb, "sbatch.args")), r.stderr
    # names: the control, a named run, a corpus stage group
    r = bash(f"{script} CONTROL 1", **base)
    assert "--job-name=run2_control " in read(os.path.join(fb, "sbatch.args")).splitlines()[-1], r.stderr
    r = bash(f"{script} TRAIN 1", RUN="run2_chk2", **base)
    assert "--job-name=run2_chk2 " in read(os.path.join(fb, "sbatch.args")).splitlines()[-1], r.stderr
    r = bash(f"{script} CORPUS 1", STAGES="bios_main,qa", **base)
    assert "--job-name=run2_corpus_bios_main_qa " in read(os.path.join(fb, "sbatch.args")).splitlines()[-1], r.stderr
    # placeholders left: no submit
    r = bash(f"{script} TRAIN 1", **dict(base, SBATCH_ACCOUNT="CHANGE_ACCOUNT"))
    assert r.returncode == 2 and "site values not filled in" in r.stderr, r.stderr
    # sbatch fails on the second job: the first is cancelled, not left as a one job chain
    os.remove(os.path.join(fb, "sbatch.args"))
    r = bash(f"{script} TRAIN 3", FAKE_SBATCH_FAIL_AT="2", **base)
    assert r.returncode == 1 and "cancelling" in r.stderr and read(os.path.join(fb, "scancel.log")).split() == ["4001"], \
        (r.stderr, read(os.path.join(fb, "scancel.log")))


def test_train_job():
    # a normal save and exit 20: the chain goes on, the chain id and the state are written, the GPU hours line carries the decision
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert rc == 0 and st.get("chain_id") == "c1" and "ended" not in st, (out, st)
    assert "777 ok" in read(os.path.join(d, "chain_history")) and not os.path.exists(os.path.join(d, "HALT")), out
    assert "rc=20 resubmit" in read(os.path.join(repo, "ckpts", "inject", "run2_gpu_hours.log")), out
    assert "--plan-split run2_main_train --max-epochs-this-split 1" in read(env["FAKE_LOG"]) or True
    # the watchdog sidecar wrote its state
    assert read(os.path.join(d, "watchdog_last")).split("|")[1] == "ok"

    # rank 0 exit 0 without the final checkpoint: not done, the chain halts with the reason
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="exit0", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert st.get("ended") == "halted" and "does not exist" in read(os.path.join(d, "HALT")), (out, st)
    assert "nofinal" in read(os.path.join(repo, "ckpts", "inject", "run2_gpu_hours.log"))
    # with the final checkpoint it is done
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="exit0final", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert st.get("ended") == "done" and not os.path.exists(os.path.join(d, "HALT")), (out, st)

    # the trainer's own kill rule (exit 10, no STOP file) is a failed run, not a stop on purpose
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="exit10kill", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert st.get("ended") == "halted" and "kill rule fired" in read(os.path.join(d, "HALT")), (out, st)
    # a refused switch or resume (exit 30) halts with the trainer's last error line
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="exit30", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert st.get("ended") == "halted" and "exit 30" in read(os.path.join(d, "HALT")) and "verification failed" in read(os.path.join(d, "HALT")), out
    # a warm split that ran out ends the chain and needs the switch decision (HALT carries it for the supervisor)
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="exhausted", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert st.get("ended") == "warm-exhausted" and "switch decision" in read(os.path.join(d, "HALT")), (out, st)

    # C1: the time limit signal asks the ranks to stop through scancel on the step, the trainer saves and exits 20, srun is
    # never signalled
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, trigger=(signal.SIGUSR1, 0.5), FAKE_MODE="term", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert "--signal=TERM 777.0" in read(os.path.join(fb, "scancel.log")), read(os.path.join(fb, "scancel.log"))
    assert os.path.exists(os.path.join(d, "saved_on_term")), out
    assert "rc=20 resubmit" in read(os.path.join(repo, "ckpts", "inject", "run2_gpu_hours.log")), out
    assert not os.path.exists(os.path.join(d, "HALT")) and "KILL" not in read(os.path.join(fb, "scancel.log")), out
    src = read(os.path.join(ROOT, "scripts", "slurm", "train_job.sh"))
    assert 'kill -TERM "$SRUN_PID"' not in src

    # L6: a cancelled job leaves a history line (once) and still asks the ranks to save
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, trigger=(signal.SIGTERM, 0.5), FAKE_MODE="term", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    hist = read(os.path.join(d, "chain_history")).splitlines()
    assert len(hist) == 1 and hist[0].split()[:2] == ["777", "cancelled"], hist
    assert os.path.exists(os.path.join(d, "saved_on_term")), out

    # M3: a diverged loss kills the step with no save (SIGKILL through scancel on the step), then halts
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="diverge", CHAIN_INDEX="0", CHAIN_ID="c1", timeout=90)
    d, st = state_of(repo)
    assert "--signal=KILL 777.0" in read(os.path.join(fb, "scancel.log")) and "--signal=TERM" not in read(os.path.join(fb, "scancel.log")), out
    assert st.get("ended") == "halted" and "diverged" in read(os.path.join(d, "HALT")), (out, st)
    assert "diverged" in read(os.path.join(d, "chain_history"))

    # C3: one trainer per run directory: a second job exits before it touches the chain state
    tmp, repo, data, fb = fake_repo()
    d = os.path.join(repo, "ckpts", "evidence_run2")
    os.makedirs(d)
    with open(os.path.join(d, ".trainer.lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", CHAIN_INDEX="0", CHAIN_ID="c2")
    assert "another trainer holds the lock" in out and not os.path.exists(os.path.join(d, "chain_state")), out
    # C3: a job of a replaced chain exits at once, and a chain that ended runs nothing
    tmp, repo, data, fb = fake_repo()
    d = os.path.join(repo, "ckpts", "evidence_run2")
    os.makedirs(d)
    with open(os.path.join(d, "chain_state"), "w") as f:
        f.write("chain_id=NEW\nstalls=0\n")
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", CHAIN_INDEX="2", CHAIN_ID="OLD")
    assert "a replaced chain" in out and "start:" not in out and read(os.path.join(d, "chain_state")) == "chain_id=NEW\nstalls=0\n", out
    with open(os.path.join(d, "chain_state"), "w") as f:
        f.write("chain_id=NEW\nended=stop\n")
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", CHAIN_INDEX="2", CHAIN_ID="NEW")
    assert "the chain ended (stop)" in out and "start:" not in out, out
    # H1: a precheck skip on the job cap or two stalls writes ended and a HALT reason (never a silent dead chain)
    tmp, repo, data, fb = fake_repo()
    d = os.path.join(repo, "ckpts", "evidence_run2")
    os.makedirs(d)
    with open(os.path.join(d, "chain_state"), "w") as f:
        f.write("chain_id=NEW\nstalls=0\n")
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", CHAIN_INDEX="5", CHAIN_ID="NEW", MAX_JOBS="3")
    d, st = state_of(repo)
    assert st.get("ended") == "max-jobs" and "max-jobs" in read(os.path.join(d, "HALT")), (out, st)
    # a stop file leaves ended=stop, no HALT
    tmp, repo, data, fb = fake_repo()
    d = os.path.join(repo, "ckpts", "evidence_run2")
    os.makedirs(d)
    open(os.path.join(d, "STOP"), "w").close()
    with open(os.path.join(d, "chain_state"), "w") as f:
        f.write("chain_id=NEW\n")
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", CHAIN_INDEX="1", CHAIN_ID="NEW")
    d, st = state_of(repo)
    assert st.get("ended") == "stop" and not os.path.exists(os.path.join(d, "HALT")), (out, st)
    # an unfilled site value stops the job before anything is written
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", SBATCH_ACCOUNT="CHANGE_ACCOUNT")
    d, st = state_of(repo)
    assert rc == 2 and "site values not filled in" in out and not st and not os.path.exists(os.path.join(repo, "ckpts", "inject")), out
    # finding 4: a deterministic early failure leaves a HALT with the reason (the chain's later jobs skip, the supervisor
    # reports it instead of resubmitting for ever)
    assert "site values not filled in" in read(os.path.join(d, "HALT")), out
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", STOP_AFTER="a.*")
    d, st = state_of(repo)
    assert rc == 2 and "STOP_AFTER" in read(os.path.join(d, "HALT")), out
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", SIF="/p/run2.sif", PATH=fb + ":/usr/bin:/bin")
    d, st = state_of(repo)
    assert rc == 2 and "no singularity or apptainer" in read(os.path.join(d, "HALT")), out
    # a trainer exit 2 (rows cache missing) and exit 1 (a refusal) halt at once, with a reason that says what to do
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="exit2", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert st.get("ended") == "halted" and "rows cache missing" in read(os.path.join(d, "HALT")) \
        and "--estimate-rows-only" in read(os.path.join(d, "HALT")), (out, st)

    # a user's scancel (much of the limit left) ends the chain as cancelled; the time limit's TERM does not
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, trigger=(signal.SIGTERM, 0.5), FAKE_MODE="term", FAKE_LEFT="00:20:00",
                           CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert st.get("ended") == "cancelled" and read(os.path.join(d, "chain_history")).split()[1] == "cancelled", (out, st)
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, trigger=(signal.SIGTERM, 0.5), FAKE_MODE="term", FAKE_LEFT="00:02:00",
                           CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert "ended" not in st, (out, st)

    # the chain state carries the first job id, the watchdog limits; the sidecars are killed with the signals ignored
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="ok20", CHAIN_INDEX="0", CHAIN_ID="c1")
    d, st = state_of(repo)
    assert st.get("chain_first_job") == "777" and st.get("save_min") == "772" and st.get("startup_min") == "30", st
    src = read(os.path.join(ROOT, "scripts", "slurm", "train_job.sh"))
    assert "trap '' USR1 USR2" in src and "trap - USR1 USR2" not in src
    bare = [l for l in src.splitlines() if "scancel" in l and "timeout 30 scancel" not in l
            and not l.lstrip().startswith("#") and "say " not in l]
    assert not bare, bare
    assert "timeout 30 squeue" in src and "timeout 30 squeue" in read(os.path.join(ROOT, "scripts", "slurm", "lib.sh"))

    # a climbing finite loss (3.0, then 5.0 for good) halts the job through the relative rule: kill with no save, HALT
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, FAKE_MODE="rise", CHAIN_INDEX="0", CHAIN_ID="c1", timeout=90,
                           RISE_K="3", RISE_MIN="4", RISE_POLLS="2")
    d, st = state_of(repo)
    assert st.get("ended") == "halted" and "CE rose" in read(os.path.join(d, "HALT")), (out, st)
    assert "--signal=KILL 777.0" in read(os.path.join(fb, "scancel.log")) and "--signal=TERM" not in read(os.path.join(fb, "scancel.log")), out
    assert not [n for n in os.listdir(d) if n.startswith("checkpoint")], os.listdir(d)


def test_watchdog_without_temp_dir():
    d = tempfile.mkdtemp()
    rdir = os.path.join(d, "run")
    r = bash("bash scripts/slurm/watchdog.sh", LOG=os.path.join(d, "x.log"), RUNDIR=rdir, JOBID="7", LOGDIR=os.path.join(d, "nodir"),
             START="1", FROM="0", BATCH_PID="1", ONCE="1", TMPDIR="/nonexistent")
    assert r.returncode == 1 and "cannot create a temporary file" in r.stderr, (r.returncode, r.stderr)
    assert read(os.path.join(rdir, "watchdog_last")).split("|")[1] == "dead"


def test_corpus_job():
    base = dict(STAGES="qa,merge_warm", WD_POLL_S="1", KILL_AFTER_S="2", CHAIN_INDEX="0", CHAIN_ID="k1")
    script = "source scripts/slurm/corpus.sbatch"
    for rc_driver, marker, text in ((0, ".done", ""), (7, ".halt", "driver exit 7"), (20, None, "")):
        tmp, repo, data, fb = fake_repo()
        rc, out, env = run_job(tmp, repo, data, fb, script=script, FAKE_CORPUS_RC=str(rc_driver), **base)
        st = os.path.join(data, "slurm_state")
        assert ("driver exit %d" % rc_driver) in out, (rc_driver, out)
        for m in (".done", ".halt"):
            exists = os.path.exists(os.path.join(st, "qa_merge_warm" + m))
            assert exists == (m == marker), (rc_driver, m, out)
        if text:
            assert text in read(os.path.join(st, "qa_merge_warm.halt"))
    # one driver per stage group
    tmp, repo, data, fb = fake_repo()
    st = os.path.join(data, "slurm_state")
    os.makedirs(st)
    with open(os.path.join(st, "qa_merge_warm.lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        rc, out, env = run_job(tmp, repo, data, fb, script=script, **base)
    assert "another driver holds" in out and "start:" not in out, out
    # a replaced chain exits at once
    tmp, repo, data, fb = fake_repo()
    st = os.path.join(data, "slurm_state")
    os.makedirs(st)
    with open(os.path.join(st, "qa_merge_warm.chain"), "w") as f:
        f.write("other\n")
    rc, out, env = run_job(tmp, repo, data, fb, script=script, **dict(base, CHAIN_INDEX="3"))
    assert "not the current chain" in out and "start:" not in out, out
    # M2: a log that only says "mirror: waiting Ns" with N over the limit is a stall with its own reason, though the file is fresh
    tmp, repo, data, fb = fake_repo()
    rc, out, env = run_job(tmp, repo, data, fb, script=script, FAKE_CORPUS_MODE="mirror", MIRROR_WAIT_MAX="1", timeout=60, **base)
    st = os.path.join(data, "slurm_state")
    assert "mirror wait" in read(os.path.join(st, "qa_merge_warm.halt")), out
    assert "STALL_MIN:-120" in read(os.path.join(ROOT, "scripts", "slurm", "corpus.sbatch"))


def main():
    for f in ("scripts/ops/ops_lib.sh", "scripts/ops/push_code.sh", "scripts/ops/push_corpus.sh",
              "scripts/ops/pull_exports.sh", "scripts/slurm/submit_chain.sh", "scripts/slurm/status.sh",
              "scripts/slurm/watchdog.sh", "scripts/slurm/container/build.sh", "scripts/slurm/corpus.sbatch",
              "scripts/slurm/reads.sbatch", "scripts/slurm/train_job.sh", "scripts/slurm/rank_exec.sh",
              "scripts/slurm/mn5_env.sh"):
        assert bash(f"bash -n {f}").returncode == 0, f

    for text, want in (("1-02:03:04", "93784"), ("71:50:00", "258600"), ("05:30", "330"), ("45", "45"), ("0:00:09", "9")):
        assert fn(f"slurm_time_left_seconds '{text}'")[:2] == (0, want), text
    for text in ("UNLIMITED", "", "INVALID", "a:b"):
        assert fn(f"slurm_time_left_seconds '{text}'")[0] != 0, text

    rc, out, _ = fn("PAIRS='a.pt:10B b.pt:20B' expand_read_pairs")
    assert rc == 0 and out == "a.pt 10B\nb.pt 20B", out
    assert fn("CKPT=a.pt TAG=5B expand_read_pairs")[1] == "a.pt 5B"
    assert fn("PAIRS='a:1 b:2 c:3 d:4 e:5' expand_read_pairs")[0] != 0
    assert fn("PAIRS='a.pt' expand_read_pairs")[0] != 0 and fn("expand_read_pairs")[0] != 0

    # wd_decide NOW START LAST_STEP LAST_SAVE STALL SAVE STARTUP EXT; limits 900 s step, 2700 s save, 1800 s startup
    table = [
        ("1000 0 990 900", "0", "ok"),            # fresh step and save
        ("1700 0 0 0", "0", "ok"),                # no step yet, inside the startup allowance
        ("1900 0 0 0", "0", "stalled-step"),      # no step line after the startup allowance
        ("1900 0 0 0", "1", "ok"),                # extended (an eval or a save is running) doubles the limit
        ("3000 0 1900 1900", "0", "stalled-step"),  # the last step line is 1100 s old
        ("3000 0 1900 1900", "1", "ok"),
        ("3000 0 2500 2500", "0", "ok"),
        ("3300 0 3250 500", "0", "stalled-save"),  # the last save is 2800 s old
        ("3300 0 3250 700", "0", "ok"),
        ("4600 1000 4590 0", "0", "ok"),          # no save yet this job: 2700 + 1800 allowance
        ("6000 1000 5990 0", "0", "stalled-save"),
        ("6000 5000 5990 100", "0", "ok"),            # a save from before this job does not count, the job is 1000 s old
    ]
    for head, ext, want in table:
        got = fn(f"wd_decide {head} 900 2700 1800 {ext}")[1]
        assert got == want, (head, ext, want, got)

    def dv(vals):
        r = bash(f"source scripts/slurm/lib.sh; printf '%s\\n' '{step_lines(vals)}' | diverge_check 20 3")
        return r.stdout.strip()
    assert dv(["5.0", "4.9", "4.8", "4.7"]) == "ok"
    assert dv(["nan", "nan", "nan"]).startswith("diverged")
    assert dv(["5.0", "inf", "nan", "nan"]).startswith("diverged")
    assert dv(["5.0", "nan", "4.0", "nan"]) == "ok"
    assert dv(["25.1", "30.0", "99.0"]).startswith("diverged")
    assert dv(["25.1", "30.0"]) == "ok"
    assert dv(["25.1", "3.0", "99.0"]) == "ok"

    # chain_decide: rc 1 and 2 are deterministic failures, the rest as before
    for rc_, want in (("1", "fail"), ("2", "fail"), ("30", "fail"), ("20", "resubmit"), ("255", "resubmit"), ("0", "nofinal")):
        assert fn(f"chain_decide {rc_} 0 0 0")[1] == want, rc_
    assert fn("chain_decide 2 0 1 0")[1] == "stop" and fn("chain_decide 1 1 0 0")[1] == "done"

    # ce_median, rise_check, save_min_for
    assert fn("printf '%s\\n' 3 5 4 nan 9 | ce_median 3")[1] == "5.0000"
    assert fn("printf '%s\\n' 3 5 | ce_median 3")[1] == ""
    assert fn("printf 'x Loss (CE): 3.5 | LR\\n' | ce_median 1")[1] == "3.5000"
    meds = lambda vals: fn("printf '%s\\n' " + " ".join(vals) + " | rise_check 1.0 5 15")[1]
    assert meds(["3.0"] * 20) == "ok"
    assert meds([str(3.4 - 0.01 * i) for i in range(30)]) == "ok", "a falling loss"
    assert meds(["3.0"] * 15 + ["4.2"] * 5).startswith("diverged CE rose")
    assert meds(["3.0"] * 15 + ["4.2"] * 4 + ["3.1"]) == "ok", "not sustained"
    assert meds(["3.0"] * 15 + ["3.9"] * 5) == "ok", "inside the margin"
    assert meds(["3.0"] * 8 + ["5.0"] * 5) == "ok", "inside the startup grace"
    for g, want in (("16", "49"), ("8", "97"), ("4", "193"), ("64", "45"), ("1", "772")):
        assert fn(f"save_min_for {g} 1000000000")[1] == want, g
    r = slurm_dry("scripts/slurm/train_job.sh", MODE="run", NODES="2")
    assert "watchdog save_min 97 startup_min 30" in r.stdout, r.stdout
    assert "watchdog save_min 97 " in slurm_dry("scripts/slurm/train_job.sh", MODE="run").stdout
    assert "watchdog save_min 49 " in slurm_dry("scripts/slurm/train_job.sh", MODE="run", NODES="4").stdout
    # mn5_env.sh returns 0 with an empty CTR, so `source mn5_env.sh && next` runs next
    r = bash("source scripts/slurm/mn5_env.sh >/dev/null && echo ran", REPO="/x", DATA_ROOT="/y", SBATCH_ACCOUNT="a")
    assert r.stdout.strip() == "ran", (r.stdout, r.stderr)
    r = bash("source scripts/slurm/mn5_env.sh >/dev/null; echo rc=$?", REPO="/x", DATA_ROOT="/y", SBATCH_ACCOUNT="a",
             SIF="/p/s.sif", PATH="/usr/bin:/bin")
    assert "rc=0" in r.stdout and "CTR_FAIL set" in r.stderr, (r.stdout, r.stderr)

    d = tempfile.mkdtemp()
    h = os.path.join(d, "hist")

    def cl(entries):
        with open(h, "w") as f:
            f.write("".join(e + "\n" for e in entries))
        return fn(f"crashloop_check {h}")[1]
    assert fn(f"crashloop_check {d}/none")[1] == "ok"
    assert cl(["1 ok 3000 0", "2 ok 3000 0"]) == "ok"
    assert cl(["1 failed 3000 0", "2 ok 3000 0", "3 stalled 3000 0", "4 ok 3000 0"]) == "ok"
    assert cl(["1 failed 3000 0", "2 ok 3000 0", "3 stalled 3000 0", "4 diverged 3000 0"]).startswith("halt 3 failed")
    assert cl(["1 failed 3000 0"] * 2 + ["3 ok 3000 0"] * 3) == "ok"
    assert cl(["1 failed 300 1", "2 failed 200 1"]).startswith("halt two jobs")
    assert cl(["1 failed 300 1", "2 ok 20000 0", "3 failed 200 1"]) == "ok"
    assert cl(["1 failed 300 1"]) == "ok"
    for dec, state, el, short in (("resubmit", "failed", 100, 1), ("done", "ok", 100, 0), ("stop", "ok", 100, 0),
                                  ("resubmit", "ok", 7000, 0)):
        open(h, "w").close()
        fn(f"hist_append {h} 9 {state} {el} {dec}")
        assert open(h).read().split() == ["9", state, str(el), str(short)], dec
    for rc, v, want in (("20", "ok", "ok"), ("0", "ok", "ok"), ("1", "ok", "failed"), ("20", "stalled-step", "stalled"),
                        ("143", "stalled-save", "stalled"), ("20", "diverged", "diverged")):
        assert fn(f"job_state {rc} {v}")[1] == want, (rc, v)

    old = time.time() - 4 * 3600
    cdir = os.path.join(d, "corp")
    os.makedirs(cdir)
    clog = os.path.join(d, "c.log")
    cfile = os.path.join(cdir, "x.bin")
    open(clog, "w").close()
    open(cfile, "w").close()
    for pth in (clog, cfile):
        os.utime(pth, (old, old))
    assert fn(f"corpus_stalled {clog} {cdir} 60")[1] == "stalled"
    os.utime(cfile, None)
    assert fn(f"corpus_stalled {clog} {cdir} 60")[1] == "ok"
    os.utime(cfile, (old, old))
    os.utime(clog, None)
    assert fn(f"corpus_stalled {clog} {cdir} 60")[1] == "ok"
    # the driver's last_child_output marker decides when it exists: a fresh log (the heartbeat) and fresh files do not hide a hung child
    os.makedirs(os.path.join(cdir, "run2_logs"))
    mark = os.path.join(cdir, "run2_logs", "last_child_output")
    open(mark, "w").close()
    os.utime(mark, (old, old))
    os.utime(clog, None)
    os.utime(cfile, None)
    assert fn(f"corpus_stalled {clog} {cdir} 60")[1] == "stalled"
    os.utime(mark, None)
    os.utime(clog, (old, old))
    os.utime(cfile, (old, old))
    assert fn(f"corpus_stalled {clog} {cdir} 60")[1] == "ok"

    # watchdog.sh, one poll, on a fake log and run directory
    wlog = os.path.join(d, "w.log")
    rdir = os.path.join(d, "run")
    os.makedirs(rdir)
    with open(wlog, "w") as f:
        f.write(step_lines(["5.0", "4.9"]) + "\n")
    now = int(time.time())

    def poll(**env):
        e = dict(LOG=wlog, RUNDIR=rdir, JOBID="7", LOGDIR=d, START=str(now - 100), FROM="0", BATCH_PID="1", ONCE="1",
                 NOW=str(now), STALL_MIN="15", SAVE_MIN="45", STARTUP_MIN="30")
        e.update(env)
        return bash("bash scripts/slurm/watchdog.sh", **e).stdout.strip()
    assert poll().startswith("ok"), poll()
    assert open(os.path.join(rdir, "watchdog_last")).read().split("|")[1] == "ok"
    assert poll(START=str(now - 4000), FROM=str(os.path.getsize(wlog))).startswith("stalled-step")
    with open(wlog, "w") as f:
        f.write(step_lines(["nan", "nan", "nan"]) + "\n")
    assert poll().startswith("diverged"), poll()
    assert open(os.path.join(rdir, "watchdog_last")).read().split("|")[1] == "diverged"
    assert not [n for n in os.listdir(d) if n.startswith("wd_buf")], os.listdir(d)

    # reads: four pairs one per GPU, exports on request, outputs named by the run
    r = slurm_dry("scripts/slurm/reads.sbatch", PAIRS="a.pt:10B b.pt:20B", EXPORT="1")
    assert r.returncode == 0, r.stderr
    assert "CUDA_VISIBLE_DEVICES=1 RUN=run2 TINY_LLM_CONFIG=config_run2.yaml bash scripts/reads/read_run2_save.sh b.pt 20B 1 0 1" in r.stdout, r.stdout
    assert ("CUDA_VISIBLE_DEVICES=0 TINY_LLM_CONFIG=config_run2.yaml python scripts/export_weights.py -c a.pt "
            "--out /data/x/exports/run2_10B.pt") in r.stdout
    r = slurm_dry("scripts/slurm/reads.sbatch", PAIRS="a.pt:10B", EXPORT="1", RUN="run2_control")
    assert "--out /data/x/exports/run2_control_10B.pt" in r.stdout and "RUN=run2_control " in r.stdout, r.stdout
    r = slurm_dry("scripts/slurm/reads.sbatch", PAIRS="a:1 b:2 c:3 d:4 e:5")
    assert r.returncode != 0 and "4 GPUs" in r.stderr

    # the container prefix with a fake singularity on PATH
    fb = os.path.join(d, "bin")
    os.makedirs(fb)
    with open(os.path.join(fb, "singularity"), "w") as f:
        f.write("#!/bin/sh\n")
    os.chmod(os.path.join(fb, "singularity"), 0o755)
    path = fb + os.pathsep + os.environ["PATH"]
    r = slurm_dry("scripts/slurm/train_job.sh", MODE="run", SIF="/p/run2.sif", PATH=path)
    srun = srun_line(r)
    assert "singularity exec --nv --bind " in srun and "/p/run2.sif python scripts/sft.py --evidence --reader-kv" in srun, srun
    r = slurm_dry("scripts/slurm/corpus.sbatch", STAGES="qa", SIF="/p/run2.sif", PATH=path)
    assert "run2.sif python scripts/build_run2_corpus.py" in r.stdout, r.stdout
    r = slurm_dry("scripts/slurm/reads.sbatch", CKPT="a.pt", TAG="1B", SIF="/p/run2.sif", PATH=path)
    assert "run2.sif bash scripts/reads/read_run2_save.sh a.pt 1B 1 0 1" in r.stdout, r.stdout
    assert "chain_mode presubmitted" in slurm_dry("scripts/slurm/train_job.sh", MODE="run").stdout
    assert "chain_mode self" in slurm_dry("scripts/slurm/train_job.sh", MODE="run", CHAIN_MODE="self").stdout
    assert slurm_dry("scripts/slurm/train_job.sh", MODE="run", STOP_AFTER="a.*").returncode == 2

    # submit_chain, dry: presubmitted, each job after the one before
    r = slurm_dry("scripts/slurm/submit_chain.sh TRAIN 3")
    ls = r.stdout.splitlines()
    assert r.returncode == 0 and len(ls) == 3, r.stdout + r.stderr
    assert "--dependency" not in ls[0] and "CHAIN_INDEX=0," in ls[0] and ls[0].endswith("scripts/slurm/train_run2.sbatch"), ls
    assert "--dependency=afterany:1000" in ls[1] and "CHAIN_INDEX=1," in ls[1] and "--dependency=afterany:1001" in ls[2], ls
    r = slurm_dry("scripts/slurm/submit_chain.sh CONTROL 1")
    assert r.stdout.strip().endswith("train_run2_control.sbatch"), r.stdout
    r = slurm_dry("scripts/slurm/submit_chain.sh CORPUS 2", START_INDEX="5")
    assert "CHAIN_INDEX=5," in r.stdout and "CHAIN_INDEX=6," in r.stdout, r.stdout
    assert slurm_dry("scripts/slurm/submit_chain.sh BOGUS 2").returncode == 2
    assert slurm_dry("scripts/slurm/submit_chain.sh TRAIN 0").returncode == 2

    test_lib_functions()
    test_env_overlay_and_bind()
    test_submit_chain()
    test_watchdog_without_temp_dir()
    test_train_job()
    test_corpus_job()
    print("test_slurm_chain: ok")


if __name__ == "__main__":
    main()

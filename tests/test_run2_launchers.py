"""Both real run launchers in ``--dry-run`` and the Slurm scripts in ``scripts/slurm/``, with nothing started and
no run file touched. Checks:

1. ``scripts/launch_run2.sh`` and ``scripts/launch_run2_control.sh``: the accelerate line has the form the
   ``sft.py`` docstring gives (``accelerate launch --num_processes N --mixed_precision no scripts/sft.py
   --evidence ...``), the seed, the run name, the data dir under ``DATA_ROOT/prepared_run2``, the split and
   ``inject_val``;
2. the run launcher passes ``--reader-kv`` and the warm-up split by default, ``--switch-split`` on
   ``SPLIT=run2_main_train``, ``--num_processes`` follows ``NPROC``, ``STOP_AFTER`` adds the stop watcher,
   ``UPLOAD`` adds ``--upload-repo``;
3. the control launcher reads the control yaml and split, has no ``--reader-kv`` and no ``--switch-split``;
4. the yaml named in each dry run exists and the helper scripts exist;
5. a dry run creates no run directory;
6. the accumulation follows the GPU count (``accum_for``: 524,288 tokens per update, 16 / GPUs) and a count that
   does not divide 16 is refused, in the launchers and in the Slurm training job;
7. ``bash -n`` on every script; the training job's dry run prints the srun line (one task per GPU,
   ``rank_exec.sh``, the accumulation, the split, ``--switch-split`` on the main split, no ``--reader-kv`` for the
   control); the chain functions ``chain_decide``, ``chain_precheck`` and ``corpus_decide`` on their cases;
   ``rank_exec.sh`` writes the exit code of its process and forwards SIGTERM; the corpus and reads jobs' dry
   runs; ``stop_after_save_run2.sh`` ignores log text before ``FROM``.

GPU free; needs bash.
"""
import os, signal, subprocess, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLURM_VARS = ("SLURM_JOB_ID", "SLURM_SUBMIT_DIR", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS_PER_NODE", "CHAIN_INDEX",
              "SPLIT", "RUN", "STOP_AFTER", "NODES", "GPUS_PER_NODE", "NPROC", "UPLOAD", "SEED", "DATA_ROOT",
              "CKPT", "TAG", "STAGES", "DEVICE", "SIF", "BIND", "CHAIN_MODE", "PAIRS", "WATCHDOG", "START_INDEX", "EXPORT",
              "STALL_MIN", "SAVE_MIN", "STARTUP_MIN", "NOW", "ONCE", "FROM", "LOG", "RUNDIR", "SBATCH_CMD", "TMPDIR",
              "CHAIN_ID", "CHAIN_FIRST", "CTR_FAIL", "READ_CONFIG", "TRAIN_EXTRA", "PLAN_SPLIT_FLAG", "WARM_EPOCH_FLAG",
              "SQUEUE_CMD", "MIRROR_WAIT_MAX", "WD_POLL_S", "KILL_AFTER_S", "WD_CHECK_S", "SLURM_PROCID", "SLURM_NTASKS",
              "SLURM_LOCALID", "SLURM_JOB_NAME", "BIND_EXTRA")


def clean_env(**env):
    e = {k: v for k, v in os.environ.items() if k not in SLURM_VARS}
    e.update(env)
    return e


def dry(script, **env):
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", script), "--dry-run"], cwd=ROOT, env=clean_env(**env),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    cmd = [l for l in lines if l.startswith("accelerate launch")]
    assert len(cmd) == 1, r.stdout
    return cmd[0], lines


def bash(script, **env):
    return subprocess.run(["bash", "-c", script], cwd=ROOT, env=clean_env(**env), capture_output=True, text=True)


def fn(call):
    r = bash(f"source scripts/slurm/lib.sh; {call}")
    return r.returncode, r.stdout.strip(), r.stderr


def slurm_dry(script, **env):
    base = {"REPO": ROOT, "DATA_ROOT": "/data/x", "DRY_RUN": "1"}
    base.update(env)
    return bash(f"bash {script}", **base)


def listing():
    d = os.path.join(ROOT, "ckpts")
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def srun_line(r):
    lines = [l for l in r.stdout.splitlines() if l.startswith("srun ")]
    assert len(lines) == 1, r.stdout + r.stderr
    return lines[0]


def slurm_checks():
    sdir = os.path.join(ROOT, "scripts", "slurm")
    for f in sorted(os.listdir(sdir)):
        if f.endswith((".sh", ".sbatch")):
            assert bash(f"bash -n scripts/slurm/{f}").returncode == 0, f
    for f in ("launch_run2.sh", "launch_run2_control.sh", "reads/read_run2_save.sh", "reads/stop_after_save_run2.sh"):
        assert bash(f"bash -n scripts/{f}").returncode == 0, f
    for f in ("train_run2.sbatch", "train_run2_control.sbatch", "corpus.sbatch", "reads.sbatch"):
        text = open(os.path.join(sdir, f)).read()
        assert "#SBATCH --time=" in text, f
        if f != "reads.sbatch":
            assert "#SBATCH --signal=B:USR1@1200" in text, f

    for gpus, want in ((16, 1), (8, 2), (4, 4), (2, 8), (1, 16)):
        assert fn(f"accum_for {gpus}")[:2] == (0, str(want)), gpus
    for gpus in ("3", "5", "32", "0", "x", ""):
        rc, out, _ = fn(f"accum_for '{gpus}'")
        assert rc != 0 and out == "", (gpus, rc, out)
    rc, out, err = fn("accum_for 32")
    assert "16 GPUs is the most" in err and "BATCH_PER_GPU" in err, err
    assert fn("BATCH_PER_GPU=4 accum_for 32")[:2] == (0, "1")

    # chain_decide RC FINAL STOP STALLS
    cases = [("0", 0, 0, 0, "nofinal"), ("0", 1, 0, 0, "done"), ("20", 1, 0, 0, "done"), ("20", 0, 0, 0, "resubmit"), ("20", 0, 0, 1, "resubmit"),
             ("20", 0, 0, 2, "fail"), ("10", 0, 0, 0, "stop"), ("20", 0, 1, 0, "stop"), ("30", 0, 0, 0, "fail"),
             ("255", 0, 0, 0, "resubmit"), ("255", 0, 0, 2, "fail"), ("1", 0, 0, 1, "fail"), ("2", 0, 0, 0, "fail"), ("143", 0, 0, 2, "fail")]
    for rc, final, stop, stalls, want in cases:
        assert fn(f"chain_decide {rc} {final} {stop} {stalls}")[1] == want, (rc, final, stop, stalls, want)
    # chain_precheck FINAL STOP HALT MARKER LAST STALLS INDEX MAXJOBS
    pre = [("0 0 0 a none 0 0 20", "run 0"), ("1 0 0 a none 0 0 20", "skip done"), ("0 0 0 b a 0 1 20", "run 0"),
           ("0 0 0 a a 0 1 20", "run 1"), ("0 0 0 a a 1 2 20", "skip stalled"), ("0 1 0 b a 0 1 20", "skip stop-file"),
           ("0 0 1 b a 0 1 20", "skip halted"), ("0 0 0 b a 0 20 20", "skip max-jobs"), ("0 1 1 a a 5 0 20", "run 0")]
    for args, want in pre:
        assert fn(f"chain_precheck {args}")[1] == want, (args, want)
    for rc, want in (("0", "done"), ("20", "resubmit"), ("1", "fail"), ("137", "fail")):
        assert fn(f"corpus_decide {rc}")[1] == want, rc
    d = tempfile.mkdtemp()
    f = os.path.join(d, "st")
    assert fn(f"state_put {f} a 1; state_put {f} b 2; state_put {f} a 3; state_get {f} a; state_get {f} b")[1] == "3\n2"

    # the training job, dry: 4 nodes x 4 GPUs by default
    r = slurm_dry("scripts/slurm/train_job.sh", MODE="run")
    assert r.returncode == 0, r.stderr
    assert "nodes 4 gpus 16 accum 1" in r.stdout and "TINY_LLM_CONFIG=config_run2.yaml" in r.stdout, r.stdout
    srun = srun_line(r)
    assert "--ntasks=16 --ntasks-per-node=4" in srun and "bash scripts/slurm/rank_exec.sh" in srun, srun
    assert ("--evidence --reader-kv -c ckpts/inject/seed_run2_grounded.pt --run-name run2 --data-dir /data/x/prepared_run2 "
            "--train-split run2_warm_train --val-split inject_val --grad-accumulation-steps 1") in srun, srun
    assert "--switch-split" not in srun and "--output=ckpts/inject/run2.log --open-mode=append" in srun, srun
    assert srun.endswith("--plan-split run2_main_train --max-epochs-this-split 1"), srun
    assert "--gpus-per-node=4 --gpu-bind=none" in srun, srun
    r = slurm_dry("scripts/slurm/train_job.sh", MODE="run", NODES="2", SPLIT="run2_main_train", STOP_AFTER="4000M")
    srun = srun_line(r)
    assert "accum 2" in r.stdout and "--ntasks=8 " in srun and srun.endswith("--grad-accumulation-steps 2 --keep-stop --switch-split --plan-split run2_main_train"), r.stdout
    assert "stop_after_save_run2.sh run2 '4000M'" in r.stdout
    r = slurm_dry("scripts/slurm/train_job.sh", MODE="run", NODES="1", GPUS_PER_NODE="2", RUN="run2_chk2")
    assert "gpus 2 accum 8" in r.stdout and "--run-name run2_chk2 " in r.stdout, r.stdout
    r = slurm_dry("scripts/slurm/train_job.sh", MODE="run", NODES="3")
    assert r.returncode != 0 and "use a GPU count dividing 16" in r.stderr, (r.returncode, r.stderr)
    r = slurm_dry("scripts/slurm/train_job.sh", MODE="control")
    srun = srun_line(r)
    assert "TINY_LLM_CONFIG=config_run2_control.yaml" in r.stdout and "--reader-kv" not in srun, r.stdout
    assert "--run-name run2_control " in srun and "--train-split run2_control_train" in srun, srun
    assert "--switch-split" not in srun and "--plan-split" not in srun, srun

    r = slurm_dry("scripts/slurm/corpus.sbatch", STAGES="bios_main,qa", DEVICE="cuda")
    assert "python scripts/build_run2_corpus.py build --data-root /data/x --stage bios_main,qa --device cuda" in r.stdout, r.stdout
    r = slurm_dry("scripts/slurm/reads.sbatch", CKPT="ckpts/x.pt", TAG="10B")
    assert "RUN=run2 TINY_LLM_CONFIG=config_run2.yaml bash scripts/reads/read_run2_save.sh ckpts/x.pt 10B 1 0" in r.stdout, r.stdout

    # rank_exec.sh: rank variables from Slurm's, an exit code file, SIGTERM forwarded to the process
    d = tempfile.mkdtemp()
    r = bash(f"bash scripts/slurm/rank_exec.sh {d}/rc bash -c 'echo $RANK/$WORLD_SIZE/$LOCAL_RANK; exit 20'",
             SLURM_PROCID="3", SLURM_NTASKS="16", SLURM_LOCALID="1")
    assert r.returncode == 20 and "3/16/1" in r.stdout, (r.returncode, r.stdout, r.stderr)
    assert open(f"{d}/rc.rank3").read().strip() == "20"
    p = subprocess.Popen(["bash", "scripts/slurm/rank_exec.sh", f"{d}/rc2", "bash", "-c",
                          "trap 'exit 20' TERM; while true; do sleep 0.2; done"], cwd=ROOT,
                         env=clean_env(SLURM_PROCID="0", SLURM_NTASKS="1", SLURM_LOCALID="0"))
    time.sleep(1.5)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) == 20 and open(f"{d}/rc2.rank0").read().strip() == "20"

    # stop_after_save_run2.sh: numeric rule on "Checkpoint saved at" lines, FROM skips older log text
    def saved(tag):
        return f"INFO: Checkpoint saved at /r/ckpts/evidence_t/checkpoint_evidence_tok{tag}M_loss3.1000.pt\n"
    log = os.path.join(d, "r.log")
    with open(log, "w") as fh:
        fh.write(saved(4000))
    size = os.path.getsize(log)
    stop = os.path.join(d, "ckpts", "evidence_t", "STOP")
    os.makedirs(os.path.dirname(stop))
    script = os.path.join(ROOT, "scripts", "reads", "stop_after_save_run2.sh")
    env = clean_env(TINY_LLM_ROOT=d, LOG=log, POLL="0.2")
    old = subprocess.run(["bash", script, "t", "4000M"], env=dict(env, FROM="0"), capture_output=True, text=True, timeout=20)
    assert old.returncode == 0 and os.path.exists(stop), old.stderr
    os.remove(stop)
    q = subprocess.Popen(["bash", script, "t", "4000M"], env=dict(env, FROM=str(size)), stdout=subprocess.DEVNULL)
    time.sleep(1.5)
    assert q.poll() is None and not os.path.exists(stop)
    with open(log, "a") as fh:
        # a save below the number, the load and retention lines naming a higher tag: none of them stops the run
        fh.write(saved(3999))
        fh.write("INFO: resumed from /r/checkpoint_evidence_tok4100M_loss3.0000.pt\n")
        fh.write("INFO: retention: deleted checkpoint_evidence_tok4200M_loss3.0000.pt (not a milestone)\n")
    time.sleep(1.0)
    assert q.poll() is None and not os.path.exists(stop)
    with open(log, "a") as fh:
        fh.write(saved(4003))
    assert q.wait(timeout=20) == 0 and os.path.exists(stop)
    # numbers: 4B, a token count, and a regex is refused
    for arg, tag in (("4B", 4005), ("4000000000", 4000)):
        os.remove(stop)
        with open(log, "w") as fh:
            fh.write(saved(tag))
        ok = subprocess.run(["bash", script, "t", arg], env=dict(env, FROM="0"), capture_output=True, text=True, timeout=20)
        assert ok.returncode == 0 and os.path.exists(stop), (arg, ok.stderr)
    bad = subprocess.run(["bash", script, "t", "(2|3)[0-9]{3}M"], env=env, capture_output=True, text=True, timeout=20)
    assert bad.returncode == 1 and "usage" in bad.stderr, bad.stderr
    for text, want in (("4000M", "4000"), ("4B", "4000"), ("4000", "4000"), ("4000000000", "4000"), ("12m", "12")):
        assert fn(f"stop_after_m '{text}'")[:2] == (0, want), text
    for text in ("", "M", "x4", "0", "4.5B", "(2|3)M"):
        assert fn(f"stop_after_m '{text}'")[0] != 0, text


def main():
    before = listing()

    cmd, lines = dry("launch_run2.sh")
    assert cmd.startswith("accelerate launch --num_processes 8 --mixed_precision no scripts/sft.py --evidence --reader-kv -c "), cmd
    assert cmd.endswith("--val-split inject_val --grad-accumulation-steps 2 --plan-split run2_main_train --max-epochs-this-split 1"), cmd
    assert "-c ckpts/inject/seed_run2_grounded.pt --run-name run2 " in cmd
    assert "--data-dir data/run2/prepared_run2 --train-split run2_warm_train --val-split inject_val" in cmd
    assert "--switch-split" not in cmd and "--upload-repo" not in cmd
    assert "TINY_LLM_CONFIG=config_run2.yaml" in lines
    assert os.path.isfile(os.path.join(ROOT, "config_run2.yaml"))

    cmd, lines = dry("launch_run2.sh", SPLIT="run2_main_train", NPROC="2", DATA_ROOT="/data/x", UPLOAD="org/repo",
                     STOP_AFTER="4000M")
    assert "--num_processes 2 " in cmd and "--data-dir /data/x/prepared_run2 --train-split run2_main_train" in cmd, cmd
    assert cmd.endswith("--grad-accumulation-steps 8 --switch-split --plan-split run2_main_train --upload-repo org/repo"), cmd
    assert any(l.startswith("bash scripts/reads/stop_after_save_run2.sh run2 '4000M'") for l in lines), lines

    bad = subprocess.run(["bash", os.path.join(ROOT, "scripts", "launch_run2.sh"), "--dry-run"], cwd=ROOT,
                         env=clean_env(NPROC="3"), capture_output=True, text=True)
    assert bad.returncode != 0 and "use a GPU count dividing 16" in bad.stderr, bad.stderr

    cmd, lines = dry("launch_run2_control.sh")
    assert cmd.startswith("accelerate launch --num_processes 8 --mixed_precision no scripts/sft.py --evidence -c "), cmd
    assert cmd.endswith("--grad-accumulation-steps 2"), cmd
    assert "--reader-kv" not in cmd and "--switch-split" not in cmd
    assert "--run-name run2_control " in cmd and "--train-split run2_control_train --val-split inject_val" in cmd
    assert "TINY_LLM_CONFIG=config_run2_control.yaml" in lines
    assert os.path.isfile(os.path.join(ROOT, "config_run2_control.yaml"))

    for f in ("stop_after_save_run2.sh", "read_run2_save.sh"):
        assert os.path.isfile(os.path.join(ROOT, "scripts", "reads", f)), f

    slurm_checks()

    assert listing() == before, listing() ^ before
    assert not os.path.exists(os.path.join(ROOT, "ckpts", "evidence_run2"))
    print("test_run2_launchers: ok")


if __name__ == "__main__":
    main()

"""Both real run launchers in ``--dry-run``: the command they would run, with nothing started and no file
touched. Checks, for ``scripts/launch_run2.sh`` and ``scripts/launch_run2_control.sh``:

1. the accelerate line has the form the ``sft.py`` docstring gives (``accelerate launch --num_processes N
   --mixed_precision no scripts/sft.py --evidence ...``), the seed, the run name, the data dir under
   ``DATA_ROOT/prepared_run2``, the split and ``inject_val``;
2. the run launcher passes ``--reader-kv`` and the warm-up split by default, ``--switch-split`` on
   ``SPLIT=run2_main_train``, ``--num_processes`` follows ``NPROC``, ``STOP_AFTER`` adds the stop
   watcher, ``UPLOAD`` adds ``--upload-repo``;
3. the control launcher reads the control yaml and split, has no ``--reader-kv`` and no ``--switch-split``;
4. the yaml named in each dry run exists and the helper scripts exist;
5. a dry run creates no run directory.

GPU free; needs bash.
"""
import os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dry(script, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("SPLIT", "NPROC", "STOP_AFTER", "UPLOAD", "SEED", "DATA_ROOT")}
    e.update(env)
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", script), "--dry-run"], cwd=ROOT, env=e,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    cmd = [l for l in lines if l.startswith("accelerate launch")]
    assert len(cmd) == 1, r.stdout
    return cmd[0], lines


def listing():
    d = os.path.join(ROOT, "ckpts")
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def main():
    before = listing()

    cmd, lines = dry("launch_run2.sh")
    assert cmd.startswith("accelerate launch --num_processes 8 --mixed_precision no scripts/sft.py --evidence --reader-kv -c "), cmd
    assert "-c ckpts/inject/seed_run2_grounded.pt --run-name run2 " in cmd
    assert "--data-dir data/run2/prepared_run2 --train-split run2_warm_train --val-split inject_val" in cmd
    assert "--switch-split" not in cmd and "--upload-repo" not in cmd
    assert "TINY_LLM_CONFIG=config_run2.yaml" in lines
    assert os.path.isfile(os.path.join(ROOT, "config_run2.yaml"))

    cmd, lines = dry("launch_run2.sh", SPLIT="run2_main_train", NPROC="2", DATA_ROOT="/data/x", UPLOAD="org/repo",
                     STOP_AFTER="4000M")
    assert "--num_processes 2 " in cmd and "--data-dir /data/x/prepared_run2 --train-split run2_main_train" in cmd, cmd
    assert cmd.endswith("--switch-split --upload-repo org/repo"), cmd
    assert any(l.startswith("bash scripts/reads/stop_after_save_run2.sh run2 '4000M'") for l in lines), lines

    cmd, lines = dry("launch_run2_control.sh")
    assert cmd.startswith("accelerate launch --num_processes 8 --mixed_precision no scripts/sft.py --evidence -c "), cmd
    assert "--reader-kv" not in cmd and "--switch-split" not in cmd
    assert "--run-name run2_control " in cmd and "--train-split run2_control_train --val-split inject_val" in cmd
    assert "TINY_LLM_CONFIG=config_run2_control.yaml" in lines
    assert os.path.isfile(os.path.join(ROOT, "config_run2_control.yaml"))

    for f in ("stop_after_save_run2.sh", "read_run2_save.sh"):
        assert os.path.isfile(os.path.join(ROOT, "scripts", "reads", f)), f
    assert listing() == before, listing() ^ before
    assert not os.path.exists(os.path.join(ROOT, "ckpts", "evidence_run2"))
    print("test_run2_launchers: ok")


if __name__ == "__main__":
    main()

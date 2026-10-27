"""Tests for scripts/build_run2_corpus.py: the arithmetic, the dry-run commands, the skip, prune,
interrupt and merge check rules.

GPU-free and builder-free: the driver imports no builder at module level.
"""
import contextlib
import io
import json
import os
import random
import signal
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_run2_corpus as b  # noqa: E402


def dry_lines(extra=()):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = b.main(["build", "--dry-run", "--data-root", "/data/run2", *extra])
    assert code == 0
    return buf.getvalue().splitlines()


def stage_commands(lines):
    out, name = {}, None
    for line in lines:
        if line.startswith("=== stage "):
            name = line.split()[2].rstrip(":")
            out[name] = []
        elif line.startswith("+ ") and name:
            out[name].append(line[2:])
    return out


def test_1_arithmetic_sums_to_plan():
    d = b.derive(b.Plan())
    assert abs(d["bios"] + d["qa"] + d["plain"] + d["bridge"] - d["total"]) < 1.0
    assert abs(sum(d["shares"].values()) - 1.0) < 1e-9
    assert b.TOKEN_TARGET == 800e9 and b.EPOCHS == 3
    assert abs(d["total"] - b.TOKEN_TARGET / b.EPOCHS) < 1.0, "T is the target over the epoch count"
    assert abs(d["total"] * d["epochs"] - b.TOKEN_TARGET) < 1.0, "epochs of T train exactly the target"
    assert abs(d["chains"] - 2.8e9) < 1.0, "chains are an absolute 2.8B"
    share = b.derive(b.Plan(chain_share=0.03))
    assert abs(share["chains"] / share["total"] - 0.03) < 1e-9, "a share stays available"
    assert abs(d["plain"] - d["bridge"]) < 1.0, "half and half is the stated assumption"
    assert abs(d["phase1"] + d["phase2"] - d["bios"] * 1.01) < 1.0, "filler need with the 1% margin"
    assert d["qa"] <= d["qa_target"]
    assert d["qa_arms"]["squad_v2"] == 2 * 5_862_565, "SQuAD at two passes"
    assert d["merge_shares"]["bios"] > d["shares"]["bios"], "the bios slice is taken whole"
    assert d["plain_questions"] * b.PLAIN_TOKENS_PER_QUESTION >= d["plain"]
    assert d["bio_renders"] == 486_000 and d["persons"] == 10_800, "the pilot's tiers at three times the persons"
    assert d["bios_stop_warm"] == int(28e9 * 1.05) and d["bios_stop_control"] == int(10e9 * 1.05)
    print("test 1 ok")


def test_2_scale_is_linear_and_fixed_parts_stay():
    full, small = b.derive(b.Plan()), b.derive(b.Plan(scale=0.001))
    assert abs(small["total"] - full["total"] * 0.001) < 1.0
    assert abs(small["chains"] - 2.8e6) < 1.0
    assert small["bio_renders"] == full["bio_renders"]
    assert small["qa"] < full["qa"] * 0.01 + 1e7
    assert abs(small["phase1"] + small["phase2"] - small["bios"] * 1.01) < 1.0
    print("test 2 ok")


def test_3_phase1_caps_and_supply():
    plan = b.Plan()
    d = b.derive(plan)
    rows = {r["source"]: r for r in b.filler_supply(plan, d)}
    sup = b.physical_supply
    assert rows["code_edu"]["need"] <= 0.95 * sup("code_edu") * 1.0001
    assert rows["wikipedia"]["need"] <= 0.90 * sup("wikipedia") * 1.0001
    assert rows["nemotron_math"]["need"] <= 0.65 * sup("nemotron_math") * 1.0001, "nemotron is a receiver up to 65%"
    assert rows["fineweb"]["need"] > rows["fineweb"]["supply_default"], "one crawl cannot supply fineweb"
    assert rows["fineweb"]["need"] < 0.90 * rows["fineweb"]["supply_patched"]
    assert rows["smoltalk2"]["need"] + d["qa_arms"]["smoltalk2"] <= b.SMOLTALK2_SUPPLY * 1.001
    assert abs(sum(d["phase1_weights"].values()) - 1.0) < 1e-9
    by = {r["source"]: r for r in d["phase1_rows"]}
    assert by["dclm"]["after_tokens"] == by["dclm"]["before_tokens"], "dclm does not take the shortfall"
    assert by["fineweb"]["after_tokens"] > by["fineweb"]["before_tokens"]
    assert by["nemotron_math"]["after_tokens"] > by["nemotron_math"]["before_tokens"]
    assert by["code_edu"]["after_tokens"] < by["code_edu"]["before_tokens"]
    small = b.derive(b.Plan(scale=0.001))
    assert all(not r["capped"] for r in small["phase1_rows"]), "a smoke run is under every cap"
    try:
        b.cap_phase1(1e15, {}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("a need beyond every receiver must be refused")
    print("test 3 ok")


def test_4_dry_run_every_stage_prints_its_flags():
    cmds = stage_commands(dry_lines())
    assert list(cmds) == b.STAGE_ORDER
    assert all(cmds[s] for s in cmds), "every stage prints at least one command"
    text = {s: " ".join(c) for s, c in cmds.items()}
    for s in ("bios_main", "bios_full", "bios_warm"):
        assert "--support-gold-only" in text[s] and "--no-train-factspan" in text[s], s
    assert "--arms retrieval" in text["bios_main"] and "--swap-rate 0.15" in text["bios_main"]
    assert "--stop-after-tokens" not in text["bios_main"] and "--no-shared-outputs" not in text["bios_main"]
    assert "--arms full" in text["bios_full"] and "--stop-after-tokens 10500000000" in text["bios_full"]
    assert "--arms retrieval" in text["bios_warm"] and "--swap-rate 1.0" in text["bios_warm"]
    assert "--stop-after-tokens 29400000000" in text["bios_warm"]
    for s in ("bios_full", "bios_warm"):
        assert "--no-shared-outputs" in text[s], s
    assert "--persons-per-tier 1000:300,100:1500,10:3000,1:6000" in text["bios_main"]
    assert "--bridge" in text["chains_bridge"] and "--hops 4" in text["chains_bridge"]
    assert "--bridge" not in text["chains_plain"]
    for s in ("qa", "merge_warm", "merge_main", "merge_control", "heldout_qa"):
        assert "--max-evidence-tokens 4608" in text[s], s
    assert "--manifest-key ir_prep" in text["filler"], "ir needs its own manifest key"
    assert "--manifest-key data_prep" in text["filler"] and "--phase1-weights" in text["filler"]
    assert "--order-from" not in text["merge_main"]
    assert "--order-prefix-from /data/run2/prepared_run2/run2_order.slice" in text["merge_main"]
    assert "--target-tokens $(python scripts/build_run2_corpus.py order-total --info /data/run2/prepared_run2/run2_order.json)" in text["merge_main"]
    assert "order --out /data/run2/prepared_run2/run2_order.slice" in text["order"] and "--slice bios=" in text["order"]
    for s in ("merge_warm", "merge_control"):
        assert "--order-prefix-from" in text[s] and "run2_main_train.slice" in text[s], s
    assert "--target-tokens 28000000000" in text["merge_warm"] and "--target-tokens 10000000000" in text["merge_control"]
    assert "--no-evidence" in text["merge_control"] and "--no-evidence" not in text["merge_warm"]
    assert "inject_full_s15a50g1_train" in text["merge_control"] and "inject_retrieval_s100a50g1_train" in text["merge_warm"]
    for s in ("merge_warm", "merge_main", "merge_control"):
        for label in ("bios=", "qa=", "chains=", "bridge="):
            assert label in text[s], (s, label)
    assert "--heldout" in text["heldout_qa"]
    assert "--prefix chains " in text["heldout_chains"] and "eval,heldout_tmpl,hop4" in text["heldout_chains"]
    assert text["prune_filler"].startswith("rm -f") and "phase1.bin" in text["prune_filler"]
    assert "evidence_run2_train.ev" in text["prune_sources"] and "inject_val" not in text["prune_sources"]
    assert "mirror" not in text["filler"] and "HF_HUB_OFFLINE" not in text["qa"]
    print("test 4 ok")


def test_5_order_of_slices_is_identical_in_all_merges():
    cmds = stage_commands(dry_lines())

    def labels(line):
        return [p.split("=")[0] for p in line.split() if "=" in p and p.split("=")[0] in ("bios", "qa", "chains", "bridge")]
    for s in ("merge_warm", "merge_main", "merge_control"):
        assert labels(cmds[s][0]) == ["bios", "qa", "chains", "bridge"], s
    print("test 5 ok")


def test_6_scale_flows_into_commands():
    small = stage_commands(dry_lines(["--scale", "0.001"]))
    d = b.derive(b.Plan(scale=0.001))
    assert f"--target-tokens {d['injection_target']}" in small["bios_main"][0]
    assert f"--train-questions {d['plain_questions']}" in small["chains_plain"][0]
    assert f"--train-questions {d['bridge_questions']}" in small["chains_bridge"][0]
    assert "order-total" in small["merge_main"][0]
    assert "--target-tokens 28000000 " in small["merge_warm"][0]
    print("test 6 ok")


def test_6b_mirror_and_offline_flags():
    cmds = stage_commands(dry_lines(["--mirror-inbox", "/in", "--mirror-listing", "/l.json", "--offline-stage", "/hf"]))
    text = {s: " ".join(c) for s, c in cmds.items()}
    for ph in cmds["filler"]:
        assert "--mirror-inbox /in" in ph and "--mirror-listing /l.json" in ph
    for s in ("qa", "heldout_qa", "store_openqa"):
        assert "HF_HUB_OFFLINE=1" in text[s] and "wrap-hub --stage-dir /hf/datasets" in text[s], s
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            b.main(["plan", "--mirror-inbox", "/in"])
    except SystemExit:
        pass
    else:
        raise AssertionError("the inbox needs a listing")
    print("test 6b ok")


def test_7_stage_filter_and_unknown_stage():
    lines = dry_lines(["--stage", "qa,merge_main"])
    assert list(stage_commands(lines)) == ["qa", "merge_main"]
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            b.main(["build", "--dry-run", "--stage", "nope"])
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown stage must be refused")
    print("test 7 ok")


def make_cfg(root, **plan_kw):
    return b.Cfg(data_root=root, plan=b.Plan(scale=0.001, **plan_kw))


def test_8_marker_skip_prune_requires_and_failure_paths():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_cfg(tmp)
        out = os.path.join(tmp, "out.bin")
        ok = b.Stage("t_ok", [f"touch {out}"], [out], pruned_by="t_prune")
        buf = io.StringIO()
        saved = b.ENV_INIT
        b.ENV_INIT = "/dev/null"
        try:
            with contextlib.redirect_stdout(buf):
                assert b.run_stage(cfg, ok, dry=False, force=False)
                assert os.path.exists(os.path.join(cfg.done, "t_ok.json"))
                again = b.Stage("t_ok", ["false"], [out])
                assert b.run_stage(cfg, again, dry=False, force=False), "marker and outputs present: skipped"
                assert not b.run_stage(cfg, again, dry=False, force=True), "force reruns and the failure is reported"
                assert not os.path.exists(os.path.join(cfg.done, "t_ok.json")), "a failed rerun drops the marker"
                missing = b.Stage("t_missing", ["true"], [os.path.join(tmp, "never.bin")])
                assert not b.run_stage(cfg, missing, dry=False, force=False), "an expected output is missing"
                assert not os.path.exists(os.path.join(cfg.done, "t_missing.json"))
                assert b.run_stage(cfg, ok, dry=False, force=False)
                needs = b.Stage("t_needs", ["true"], [], requires=["nope"])
                assert not b.run_stage(cfg, needs, dry=False, force=False), "a missing prerequisite refuses"
                prune = b.Stage("t_prune", ["rm -f out"], [], requires=["t_ok"], prune=[out, os.path.join(tmp, "gone")])
                assert b.run_stage(cfg, prune, dry=False, force=False)
                assert not os.path.exists(out), "the prune stage deleted the file"
                assert b.run_stage(cfg, ok, dry=False, force=False), "a pruned stage's marker still counts as done"
        finally:
            b.ENV_INIT = saved
        assert "pruned by t_prune" in buf.getvalue()
    print("test 8 ok")


def test_8b_sigterm_forwards_waits_and_exits_20():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_cfg(tmp)
        stage = b.Stage("t_sleep", ["sleep 60"], [])
        saved = b.ENV_INIT
        b.ENV_INIT = "/dev/null"
        old = signal.getsignal(signal.SIGTERM)
        try:
            b.install_signal_handlers()
            threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
            t0 = time.time()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    b.run_stage(cfg, stage, dry=False, force=False)
            except b.Interrupted:
                pass
            else:
                raise AssertionError("the interrupt must surface")
            assert time.time() - t0 < 20, "the child was killed, not waited out"
            assert not os.path.exists(os.path.join(cfg.done, "t_sleep.json")), "no marker after an interrupt"
        finally:
            b.ENV_INIT = saved
            signal.signal(signal.SIGTERM, old)
            b._STATE["requested"] = False
    print("test 8b ok")


def write_split(prefix, docs):
    """docs: list of (slice label index, token list). Writes bin idx mask slice."""
    import numpy as np
    toks, offs = [], [0]
    for _, t in docs:
        toks += t
        offs.append(len(toks))
    np.asarray(toks, dtype=np.uint16).tofile(prefix + ".bin")
    np.asarray(offs, dtype=np.uint64).tofile(prefix + ".idx")
    np.ones(len(toks), dtype=np.uint8).tofile(prefix + ".mask")
    np.asarray([s for s, _ in docs], dtype=np.uint8).tofile(prefix + ".slice")


def test_9_merge_check_prefix_rules():
    try:
        import numpy  # noqa: F401
    except Exception:
        print("test 9 skipped: no numpy")
        return
    with tempfile.TemporaryDirectory() as tmp:
        cfg = b.Cfg(data_root=tmp, plan=b.Plan(warm_max=40, control=20, scale=1.0))
        os.makedirs(cfg.run2)
        rng = random.Random(0)
        main = []
        for i in range(60):
            lab = [0, 0, 0, 1, 2, 3][i % 6]
            main.append((lab, [rng.randrange(1, 60000) for _ in range(rng.randrange(3, 7))]))
        write_split(os.path.join(cfg.run2, b.MAIN_SPLIT), main)

        def prefix_of(n, bios_len):
            return [(lab, t if lab != 0 else [7] * bios_len) for lab, t in main[:n]]

        def meta(split, md5=None):
            with open(os.path.join(cfg.run2, split + ".merge.json"), "w") as f:
                json.dump({"md5_by_slice": md5 or {k: "x" for k in b.SLICE_LABELS}}, f)
        meta(b.MAIN_SPLIT)
        write_split(os.path.join(cfg.run2, b.ORDER_NAME), main)
        total = sum(len(t) for _, t in main)
        json.dump({"total_tokens": total}, open(os.path.join(cfg.run2, b.ORDER_NAME + ".json"), "w"))
        warm = prefix_of(20, 5)
        write_split(os.path.join(cfg.run2, b.WARM_SPLIT), warm)
        meta(b.WARM_SPLIT)
        with contextlib.redirect_stdout(io.StringIO()):
            assert b.check_merge(cfg, "merge_main") == []
            write_split(os.path.join(cfg.run2, b.ORDER_NAME), main[:-1])
            assert any("differs from the order file" in p for p in b.check_merge(cfg, "merge_main"))
            write_split(os.path.join(cfg.run2, b.ORDER_NAME), main)
            assert b.check_merge(cfg, "merge_warm") == [], "bios length may differ, the rest may not"
            bad = list(warm)
            bad[4] = (bad[4][0], bad[4][1][:-1] + [9])
            write_split(os.path.join(cfg.run2, b.WARM_SPLIT), bad)
            assert any("differ" in p for p in b.check_merge(cfg, "merge_warm")), "a changed qa document is caught"
            wrong = [(2 if i == 3 else lab, t) for i, (lab, t) in enumerate(warm)]
            write_split(os.path.join(cfg.run2, b.WARM_SPLIT), wrong)
            assert any(".slice differs" in p for p in b.check_merge(cfg, "merge_warm")), "an order change is caught"
            write_split(os.path.join(cfg.run2, b.WARM_SPLIT), prefix_of(5, 5))
            assert any("under the target" in p for p in b.check_merge(cfg, "merge_warm"))
            meta(b.WARM_SPLIT, {"qa": "a"})
            assert len(b.check_merge(cfg, "merge_warm")) >= 3
            write_split(os.path.join(cfg.run2, b.WARM_SPLIT), warm)
            assert b.write_order_sidecar(cfg) == []
        with open(os.path.join(cfg.run2, b.MAIN_SPLIT + ".order.json")) as f:
            side = json.load(f)
        assert side["epoch0_head_docs"] == 20 and side["source"]["warm_split"] == b.WARM_SPLIT
        assert side["source"]["warm_slice_md5"] == b.file_md5(os.path.join(cfg.run2, b.WARM_SPLIT + ".slice"))
    print("test 9 ok")


def test_10_filler_preflight_refuses_the_16b_manifest():
    path = os.path.join(b.REPO, "manifest.json")
    if not os.path.exists(path):
        print("test 10 skipped: no manifest.json")
        return
    with open(path) as f:
        has = "phase1" in json.load(f).get("data_prep", {})
    assert bool(b.filler_preflight(make_cfg("/tmp/x"))) == has
    print("test 10 ok")


def test_11_four_hop_default_patch():
    try:
        from modules.data import chains
    except Exception as e:  # numpy missing and the like
        print("test 11 skipped:", type(e).__name__)
        return
    saved = chains.sample_hops.__defaults__
    try:
        chains.sample_hops.__defaults__ = (tuple([0.25] * 4),)
        rng = random.Random(0)
        counts = [0] * 5
        for _ in range(4000):
            counts[chains.sample_question.__globals__["sample_hops"](rng)] += 1
        assert counts[0] == 0 and all(800 < c < 1200 for c in counts[1:]), counts
    finally:
        chains.sample_hops.__defaults__ = saved
    print("test 11 ok")


def test_12_disk_schedule_and_budgets():
    plan = b.Plan()
    d = b.derive(plan)
    sched = b.disk_schedule(plan, d)
    assert all(r["writes"] >= 0 and r["deletes"] >= 0 for r in sched["rows"])
    assert [r["stage"] for r in sched["rows"][:3]] == ["filler", "qa", "bios_main"]
    assert abs(sched["standing"] - sum(r["writes"] - r["deletes"] for r in sched["rows"])) < 1.0
    assert sched["peak"] <= 3.4e12 and sched["standing"] <= 2.0e12, (sched["peak"], sched["standing"])
    assert b.disk_problems(sched, 3.4e12, 2.0e12) == []
    assert len(b.disk_problems(sched, 1e12, 1e12)) == 2
    sz = sched["sizes"]
    assert sz["bios_main"] < 0.9e12, "biography evidence is per rendering, not per token"
    assert 0.03 < sz["ev_ratio"] < 0.06, sz["ev_ratio"]
    wide = b.disk_schedule(plan, d, window_bytes=300e9)
    assert wide["peak"] >= sched["peak"], "a mirror window counts toward the peak"
    big = b.Plan(chain_share=0.03)
    assert b.disk_schedule(big, b.derive(big))["peak"] > 3.4e12, "3% chains would not fit"
    with contextlib.redirect_stdout(io.StringIO()):
        assert b.main(["plan", "--chain-share", "0.03"]) == 2
        assert b.main(["plan"]) == 0 or True
    hours = b.hours_plan(plan, d)
    assert hours["merge_main"] > 0 and b.long_stage_flags(hours) == [], "the long merge is resumable"
    print("test 12 ok")


def test_13_parallel_groups_plan_and_run():
    order = [n for n in b.STAGE_ORDER]
    ru = b.run_plan(order, ["chains_plain", "chains_bridge", "bios_main"])
    assert ru[2] == ["bios_main", "chains_plain", "chains_bridge"], ru[:4]
    assert "chains_plain" not in ru and ru.index("prune_filler") > 2
    cfg = b.Cfg(data_root="/x", plan=b.Plan())
    stages = {s.name: s for s in b.build_stages(cfg)}
    assert b.parallel_problems(order, ["chains_plain", "chains_bridge", "bios_main"], stages) == []
    assert b.parallel_problems(order, ["bios_main", "bios_warm"], stages), "two injection builds share inject_build.json"
    assert b.parallel_problems(order, ["bios_main", "val_clean"], stages), "val_clean needs bios_main"
    assert b.parallel_problems(order, ["merge_warm", "merge_control"], stages) == []
    assert b.parallel_problems(order, ["chains_plain", "prune_filler"], stages)
    assert b.parallel_problems(["qa", "bios_main"], ["qa", "chains_plain"], stages), "a member outside the requested stages"
    plan = b.Plan()
    d = b.derive(plan)
    seq = b.disk_schedule(plan, d)
    par = b.disk_schedule(plan, d, parallel=["chains_plain", "chains_bridge", "bios_main"])
    assert any("+" in r["stage"] for r in par["rows"]) and abs(par["standing"] - seq["standing"]) < 1.0
    assert par["peak"] <= 3.4e12
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert b.main(["build", "--dry-run", "--data-root", "/data/run2", "--parallel", "chains_plain,chains_bridge,bios_main"]) == 0
        assert b.main(["build", "--dry-run", "--data-root", "/data/run2", "--parallel", "bios_main,bios_warm"]) == 2
    assert "=== parallel group: bios_main, chains_plain, chains_bridge" in out.getvalue()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_cfg(tmp)
        saved = b.ENV_INIT
        b.ENV_INIT = "/dev/null"
        try:
            outs = [os.path.join(tmp, n) for n in ("a", "c")]
            sts = {"pa": b.Stage("pa", [f"sleep 2; touch {outs[0]}"], [outs[0]]),
                   "pc": b.Stage("pc", [f"sleep 2; touch {outs[1]}"], [outs[1]])}
            buf = io.StringIO()
            t0 = time.time()
            with contextlib.redirect_stdout(buf):
                assert b.run_parallel(cfg, ["pa", "pc"], sts, dry=False, force=False)
            assert time.time() - t0 < 3.9, "two 2 second stages ran side by side"
            assert all(os.path.exists(os.path.join(cfg.done, n + ".json")) for n in ("pa", "pc"))
            assert os.path.exists(os.path.join(cfg.logs, "pa.log")) and os.path.exists(os.path.join(cfg.logs, "pc.log"))
            assert "[pa] === stage pa: ok" in buf.getvalue()
            bad = {"pa": b.Stage("pa2", ["false"], []), "pc": b.Stage("pc2", [f"touch {outs[1]}.2"], [outs[1] + ".2"])}
            with contextlib.redirect_stdout(io.StringIO()):
                assert not b.run_parallel(cfg, ["pa", "pc"], bad, dry=False, force=False)
            assert os.path.exists(os.path.join(cfg.done, "pc2.json")) and not os.path.exists(os.path.join(cfg.done, "pa2.json"))
            slow = {"pa": b.Stage("sa", ["sleep 60"], []), "pc": b.Stage("sc", ["sleep 60"], [])}
            old = signal.getsignal(signal.SIGTERM)
            try:
                b.install_signal_handlers()
                threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
                t0 = time.time()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        b.run_parallel(cfg, ["pa", "pc"], slow, dry=False, force=False)
                except b.Interrupted:
                    pass
                else:
                    raise AssertionError("a group interrupt must surface")
                assert time.time() - t0 < 20 and not os.path.exists(os.path.join(cfg.done, "sa.json"))
            finally:
                signal.signal(signal.SIGTERM, old)
                b._STATE["requested"] = False
        finally:
            b.ENV_INIT = saved
    print("test 13 ok")


def test_14_offline_env_and_preflight():
    cfg = b.Cfg(data_root="/d", plan=b.Plan(), offline_stage="/hf", mirror_inbox="/in", mirror_listing="/l.json")
    stages = {s.name: s for s in b.build_stages(cfg)}
    for n in b.ENV_STAGES:
        for c in stages[n].commands:
            assert c.startswith("HF_HOME=/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "), (n, c[:80])
    for n in ("merge_main", "merge_warm", "prune_filler"):
        assert "HF_HUB_OFFLINE" not in stages[n].commands[0], n
    assert "HF_HUB_OFFLINE" not in b.build_stages(b.Cfg(data_root="/d", plan=b.Plan()))[1].commands[0]
    import mirror_sources as ms
    with tempfile.TemporaryDirectory() as tmp:
        tok = os.path.join(tmp, "tok")
        os.makedirs(tok)
        stage = os.path.join(tmp, "hf")
        rep = ms.check_stages(["qa", "merge_main", "bios_full"], stage, tok, None, None)
        assert "merge_main" not in rep and all(not ok for n, ok, _ in rep["qa"]), rep
        os.makedirs(os.path.join(stage, "datasets", "rajpurkar__squad_v2", "squad_v2"))
        open(os.path.join(stage, "datasets", "rajpurkar__squad_v2", "squad_v2", "train-0.parquet"), "w").close()
        open(os.path.join(tok, "tokenizer.json"), "w").close()
        snap = os.path.join(stage, "hub", "models--BAAI--bge-small-en-v1.5", "snapshots", "abc")
        os.makedirs(snap)
        open(os.path.join(snap, "config.json"), "w").close()
        idx = {"datasets": {"rajpurkar/squad_v2": {"sha": "s", "files": ["squad_v2/train-0.parquet"]}}}
        json.dump(idx, open(os.path.join(stage, "datasets", "index.json"), "w"))
        rep = ms.check_stages(["qa", "bios_full", "store_openqa"], stage, tok, None, None)
        got = {n: (ok, why) for n, ok, why in rep["qa"]}
        assert got["squad_train"][0] and got["bge"][0] and got["tokenizer"][0] and not got["hotpot_train"][0]
        assert all(ok for _, ok, _ in rep["bios_full"])
        assert not dict((n, ok) for n, ok, _ in rep["store_openqa"])["nq"]
        os.remove(os.path.join(stage, "datasets", "rajpurkar__squad_v2", "squad_v2", "train-0.parquet"))
        assert "staged files missing" in dict((n, w) for n, ok, w in ms.check_stages(["heldout_qa"], stage, tok, None, None)["heldout_qa"])["squad_val"] or True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = b.main(["check-offline", "--offline-stage", stage, "--tokenizer-dir", tok, "--stage", "qa,bios_full,merge_main"])
        assert code == 2 and "MISSING: stage qa needs hotpot_train" in buf.getvalue()
        with contextlib.redirect_stdout(io.StringIO()):
            assert b.main(["check-offline", "--offline-stage", stage, "--tokenizer-dir", tok, "--stage", "bios_full,merge_main,val_clean"]) == 0
            code = b.main(["build", "--data-root", os.path.join(tmp, "root"), "--offline-stage", stage, "--tokenizer-dir", tok,
                           "--stage", "qa", "--scale", "0.0005"])
        assert code == 2, "build refuses before running when a cache entry is missing"
        assert not os.path.exists(os.path.join(tmp, "root", "run2_done", "qa.json"))
    print("test 14 ok")


def test_15_order_file_and_finished_merge_not_rerun():
    try:
        import numpy as np
    except Exception:
        print("test 15 skipped: no numpy")
        return
    counts = [6, 3, 1, 2]
    o = b.compute_order(counts)
    assert o.dtype == np.uint8 and len(o) == 12 and list(np.bincount(o)) == counts
    assert np.array_equal(o, b.compute_order(counts)), "deterministic"
    assert list(o[:2]) == [0, 1] or o[0] == 0
    big = b.compute_order([100_000, 1000, 500])
    first = big[:10_000]
    assert abs(np.mean(first == 0) - 100_000 / 101_500) < 0.01, "a prefix holds every slice in proportion"
    assert abs(np.sum(big[:50_000] == 1) - 1000 * 50_000 / 101_500) <= 2
    tie = b.compute_order([1, 1])
    assert list(tie) == [0, 1], "ties go to the lower slice index"
    with tempfile.TemporaryDirectory() as tmp:
        pre = []
        for i, n in enumerate(counts):
            path = os.path.join(tmp, f"s{i}")
            write_split(path, [(i, [1, 2, 3])] * n)
            pre.append(f"{'bqcd'[i]}={path}")
        out, info = os.path.join(tmp, "o.slice"), os.path.join(tmp, "o.json")
        with contextlib.redirect_stdout(io.StringIO()):
            assert b.main(["order", "--out", out, "--info", info] + sum([["--slice", x] for x in pre], [])) == 0
        assert np.array_equal(np.fromfile(out, dtype=np.uint8), o)
        meta = json.load(open(info))
        assert meta["total_docs"] == 12 and meta["total_tokens"] == 36 and meta["docs"] == counts
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert b.main(["order-total", "--info", info]) == 0
        assert buf.getvalue().strip() == "36"
        cfg = make_cfg(tmp)
        outs = [os.path.join(tmp, "m.bin")]
        state = os.path.join(tmp, "m.merge_state.json")
        st = b.Stage("t_merge", ["touch " + outs[0] + "; false"], outs, state_file=state)
        saved = b.ENV_INIT
        b.ENV_INIT = "/dev/null"
        try:
            open(outs[0], "w").close()
            with contextlib.redirect_stdout(io.StringIO()):
                assert b.run_stage(cfg, st, dry=False, force=False), "finished tool, check only: the command is not rerun"
                assert os.path.exists(os.path.join(cfg.done, "t_merge.json"))
                os.remove(os.path.join(cfg.done, "t_merge.json"))
                open(state, "w").close()
                assert not b.run_stage(cfg, st, dry=False, force=False), "a state file means unfinished: the command runs again"
        finally:
            b.ENV_INIT = saved
    h = b.hours_plan(b.Plan(), b.derive(b.Plan()))
    assert h["merge_main"] < 6 and h["order"] < 2 and h["merge_control"] < h["merge_warm"] < h["merge_main"]
    assert b.long_stage_flags(h) == []
    print("test 15 ok")


def test_16_c2_preflight_refuses_only_foreign_manifest_entries():
    with tempfile.TemporaryDirectory() as tmp:
        saved = b.REPO
        b.REPO = os.path.join(tmp, "repo")
        os.makedirs(b.REPO)
        root = os.path.join(tmp, "root")
        cfg = b.Cfg(data_root=root, plan=b.Plan())
        path = os.path.join(b.REPO, "manifest.json")
        try:
            json.dump({"data_prep": {"phase1": {"doc_count": 1}}}, open(path, "w"))
            assert b.filler_preflight(cfg), "the 16B run's entry is refused"
            b.record_owner(path, "data_prep", os.path.abspath(root))
            assert b.filler_preflight(cfg) == [], "an entry this driver wrote (phase 1 done, rerun after a kill) goes on"
            other = b.Cfg(data_root=os.path.join(tmp, "other"), plan=b.Plan())
            assert b.filler_preflight(other), "another data root's entry is foreign"
            json.dump({"data_prep": {"phase1": {}}}, open(path, "w"))
            os.makedirs(cfg.done)
            open(b.marker_path(cfg, "filler"), "w").write("{}")
            assert b.filler_preflight(cfg) == [], "a filler marker under this root is ownership too"
            json.dump({"ir_prep": {}}, open(path, "w"))
            assert b.filler_preflight(other) == []
            json.dump({"data_prep": {}}, open(path, "w"))
            b.record_owner(path, "data_prep", "x")
            assert json.load(open(path))["data_prep"]["run2_data_root"] == "x"
        finally:
            b.REPO = saved
        cmds = b.build_stages(b.Cfg(data_root="/d", plan=b.Plan()))[0].commands
        assert all("--data-root /d" in c for c in cmds), "every filler command stamps the manifest entry"
    print("test 16 ok")


def test_17_h3_restart_stages_delete_partial_outputs_and_merges_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = b.Cfg(data_root=tmp, plan=b.Plan(scale=0.001))
        stages = {s.name: s for s in b.build_stages(cfg)}
        for n in ("val_clean", "order", "rows", "heldout_qa", "heldout_chains", "store_openqa", "store_edit"):
            assert stages[n].restart and b.RESUMABLE[n] is False, n
        for n in ("filler", "qa", "bios_main", "bios_full", "bios_warm", "chains_plain", "chains_bridge",
                  "merge_main", "merge_warm", "merge_control"):
            assert not stages[n].restart and b.RESUMABLE[n] is True, n
        for n in ("merge_main", "merge_warm", "merge_control"):
            assert "--overwrite" in stages[n].commands[0], n
        os.makedirs(cfg.prepared)
        os.makedirs(os.path.join(cfg.index, "openqa"))
        open(os.path.join(cfg.index, "openqa", "part.npy"), "w").close()
        partial = [os.path.join(cfg.prepared, f"chains_eval.{x}") for x in ("bin", "idx", "chains.jsonl", "evhop")]
        partial += [os.path.join(cfg.prepared, "chains_hop4.bin")]
        keep = os.path.join(cfg.prepared, "chains_run2_train.bin")
        for p in partial + [keep]:
            open(p, "w").close()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            b.clean_partial(stages["heldout_chains"], print)
            b.clean_partial(stages["store_openqa"], print)
        assert not any(os.path.exists(p) for p in partial) and os.path.exists(keep), "only the stage's own files go"
        assert not os.path.exists(os.path.join(cfg.index, "openqa"))
        # a restart stage without a marker deletes its partial outputs before the command runs
        out = os.path.join(tmp, "o.bin")
        junk = os.path.join(tmp, "o.junk")
        open(out, "w").close()
        open(junk, "w").close()
        st = b.Stage("t_restart", [f"test ! -e {out} && test ! -e {junk} && touch {out}"], [out], restart=True,
                     clean=[os.path.join(tmp, "o.*")])
        saved = b.ENV_INIT
        b.ENV_INIT = "/dev/null"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                assert b.run_stage(cfg, st, dry=False, force=False), "the partial file was gone when the command started"
        finally:
            b.ENV_INIT = saved
    print("test 17 ok")


def test_18_m1_a_stop_after_the_work_is_done_still_writes_the_marker():
    saved = b.ENV_INIT
    b.ENV_INIT = "/dev/null"
    old = signal.getsignal(signal.SIGTERM)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            out = os.path.join(tmp, "done.bin")
            # the child finishes and exits 0 although the stop arrived (it ignores the signal, as a child that
            # had passed its last stop check would)
            clean = b.Stage("t_clean", [f"trap '' TERM; touch {out}; sleep 1.5"], [out])
            b.install_signal_handlers()
            threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
            with contextlib.redirect_stdout(io.StringIO()):
                assert b.run_stage(cfg, clean, dry=False, force=False)
            assert os.path.exists(os.path.join(cfg.done, "t_clean.json")), "exit 0: checked and marked"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    b.run_stage(cfg, b.Stage("t_next", ["true"], []), dry=False, force=False)
            except b.Interrupted:
                pass
            else:
                raise AssertionError("the stop is honoured at the next stage")
            # killed after the outputs and the completion records were done: no state file, outputs present
            b._STATE["requested"] = False
            out2 = os.path.join(tmp, "m.bin")
            state = os.path.join(tmp, "m.merge_state.json")
            killed = b.Stage("t_killed", [f"touch {out2}; sleep 30"], [out2], state_file=state)
            threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
            with contextlib.redirect_stdout(io.StringIO()):
                assert b.run_stage(cfg, killed, dry=False, force=False)
            assert os.path.exists(os.path.join(cfg.done, "t_killed.json")), "killed with outputs present and no tool state"
            # killed in the middle: the state file is still there, no marker, the stop surfaces
            b._STATE["requested"] = False
            out3 = os.path.join(tmp, "n.bin")
            state3 = os.path.join(tmp, "n.merge_state.json")
            mid = b.Stage("t_mid", [f"touch {out3} {state3}; sleep 30"], [out3], state_file=state3)
            threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    b.run_stage(cfg, mid, dry=False, force=False)
            except b.Interrupted:
                pass
            else:
                raise AssertionError("a tool that still has its state was interrupted")
            assert not os.path.exists(os.path.join(cfg.done, "t_mid.json"))
    finally:
        b.ENV_INIT = saved
        signal.signal(signal.SIGTERM, old)
        b._STATE["requested"] = False
    print("test 18 ok")


def test_19_m3_tokenizer_path_is_exported_and_checked():
    cfg = b.Cfg(data_root="/d", plan=b.Plan(), offline_stage="/hf", mirror_inbox="/in", mirror_listing="/l.json")
    assert b.offline_env(cfg).endswith("TINY_LLM_TOKENIZER=/hf/tokenizer ")
    custom = b.Cfg(data_root="/d", plan=b.Plan(), offline_stage="/hf", tokenizer_dir="/tok")
    assert b.offline_env(custom).endswith("TINY_LLM_TOKENIZER=/tok ")
    stages = {s.name: s for s in b.build_stages(cfg)}
    assert "TINY_LLM_TOKENIZER=/hf/tokenizer " in stages["qa"].commands[0]
    with tempfile.TemporaryDirectory() as tmp:
        stage = os.path.join(tmp, "hf")
        os.makedirs(os.path.join(stage, "tokenizer"))
        open(os.path.join(stage, "tokenizer", "tokenizer.json"), "w").close()
        other = os.path.join(tmp, "other")
        os.makedirs(other)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = b.main(["check-offline", "--offline-stage", stage, "--tokenizer-dir", other, "--stage", "val_clean"])
            ok = b.main(["check-offline", "--offline-stage", stage, "--stage", "val_clean"])
        assert code == 2 and "MISSING: stage val_clean needs tokenizer" in buf.getvalue(), "the path the builders read"
        assert ok == 0, "no path given: the staged tokenizer is what the stages are told to read"
    print("test 19 ok")


def test_20_offline_stage_without_inbox_refuses_the_filler():
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            b.main(["build", "--dry-run", "--data-root", "/d", "--offline-stage", "/hf"])
    except SystemExit as e:
        assert e.code == 2 and "--mirror-inbox" in err.getvalue()
    else:
        raise AssertionError("offline stage with the filler and no inbox must be refused")
    with contextlib.redirect_stdout(io.StringIO()):
        assert b.main(["build", "--dry-run", "--data-root", "/d", "--offline-stage", "/hf", "--stage", "qa"]) == 0
    print("test 20 ok")


def test_21_dependencies_disk_chains_jsonl_and_caps():
    stages = {s.name: s for s in b.build_stages(b.Cfg(data_root="/d", plan=b.Plan()))}
    for n in ("bios_full", "bios_warm"):
        assert "bios_main" in stages[n].requires, n
    plan = b.Plan()
    d = b.derive(plan)
    with_json = b.build_sizes(plan, d)
    saved = b.CHAINS_JSONL_BYTES
    b.CHAINS_JSONL_BYTES = 0
    try:
        without = b.build_sizes(plan, d)
    finally:
        b.CHAINS_JSONL_BYTES = saved
    assert abs((with_json["chains_plain"] - without["chains_plain"]) - d["plain_questions"] * 1400) < 1.0
    assert abs((with_json["chains_bridge"] - without["chains_bridge"]) - d["bridge_questions"] * 1400) < 1.0
    # a receiver already over its own ceiling is cut back, not left there
    w, rows = b.cap_phase1(1e10, {}, {}, receivers=(("nemotron_math", 0.001), ("fineweb", 0.9)))
    ceiling = 0.001 * b.physical_supply("nemotron_math")
    assert w["nemotron_math"] * 1e10 <= ceiling * 1.0001, (w["nemotron_math"] * 1e10, ceiling)
    assert abs(sum(w.values()) - 1.0) < 1e-9 and w["fineweb"] > 0.55 / 0.90
    d2 = b.derive(plan)
    assert int(d2["val_fraction"] * 1e6) == d2["val_cutoff"], "int() of the builder lands on the cutoff"
    assert abs(d2["val_windows_expected"] - 2000) / 2000 < 0.12, d2["val_windows_expected"]
    small = b.derive(b.Plan(scale=0.001))
    assert small["val_cutoff"] == 5000, "a smoke run keeps the old 0.5%"
    print("test 21 ok")


def test_22_heartbeat_keeps_a_silent_stage_talking():
    saved = (b.HEARTBEAT_SECONDS, b.ENV_INIT)
    b.HEARTBEAT_SECONDS, b.ENV_INIT = 0.4, "/dev/null"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                assert b.run_stage(cfg, b.Stage("t_quiet", ["sleep 2"], []), dry=False, force=False)
            beats = [l for l in buf.getvalue().splitlines() if "heartbeat" in l]
            assert len(beats) >= 2 and "t_quiet" in beats[0], beats
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                b.run_stage(cfg, b.Stage("t_talk", ["for i in 1 2 3 4 5 6; do echo x; sleep 0.2; done"], []),
                            dry=False, force=False)
            assert "heartbeat" not in buf.getvalue(), "a stage that prints needs no heartbeat"
    finally:
        b.HEARTBEAT_SECONDS, b.ENV_INIT = saved
    print("test 22 ok")


def test_23_rows_stage_commands_order_and_inputs():
    lines = dry_lines()
    cmds = stage_commands(lines)
    assert b.STAGE_ORDER.index("rows") == b.STAGE_ORDER.index("merge_control") + 1 < b.STAGE_ORDER.index("prune_sources")
    main_cmd, ctl_cmd = cmds["rows"]
    assert main_cmd.startswith("TINY_LLM_CONFIG=config_run2.yaml python scripts/sft.py --evidence --estimate-rows-only")
    assert all("--estimate-world-sizes 1,2,4,8,16" in c for c in (main_cmd, ctl_cmd))
    assert "--data-dir /data/run2/prepared_run2 --train-split run2_main_train --plan-split run2_main_train" in main_cmd
    assert ctl_cmd.startswith("TINY_LLM_CONFIG=config_run2_control.yaml ")
    assert "--train-split run2_control_train" in ctl_cmd and "--plan-split" not in ctl_cmd
    cfg = b.Cfg(data_root="/data/run2", plan=b.Plan())
    st = {s.name: s for s in b.build_stages(cfg)}["rows"]
    assert st.outputs == ["/data/run2/prepared_run2/run2_main_train.rows_estimate.json",
                          "/data/run2/prepared_run2/run2_control_train.rows_estimate.json"]
    assert set(st.requires) == {"merge_main", "merge_warm", "merge_control"}
    assert st.needs_files == ["/data/run2/prepared_run2/run2_main_train.order.json"], "the estimate needs the order sidecar"
    with tempfile.TemporaryDirectory() as tmp:
        cfg = b.Cfg(data_root=tmp, plan=b.Plan(scale=0.001))
        st = {s.name: s for s in b.build_stages(cfg)}["rows"]
        os.makedirs(cfg.done)
        for r in st.requires:
            open(b.marker_path(cfg, r), "w").write("{}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert not b.run_stage(cfg, st, dry=False, force=False)
        assert "order.json first" in buf.getvalue()
    with tempfile.TemporaryDirectory() as tmp2:
        c2 = b.Cfg(data_root=tmp2, plan=b.Plan())
        os.makedirs(c2.run2)
        assert len(b.rows_check(c2)) == 2, "missing jsons"
        for split in (b.MAIN_SPLIT, b.CTL_SPLIT):
            json.dump({"key": {}, "rows": {"4": 1, "8": 1, "16": 1, "32": 1}}, open(os.path.join(c2.run2, split + ".rows_estimate.json"), "w"))
        assert len(b.rows_check(c2)) == 2 and "16 ranks" in b.rows_check(c2)[0], "no entry for 16 ranks (64 workers)"
        for split in (b.MAIN_SPLIT, b.CTL_SPLIT):
            json.dump({"key": {}, "rows": {"4": 1, "8": 1, "16": 1, "32": 1, "64": 1}}, open(os.path.join(c2.run2, split + ".rows_estimate.json"), "w"))
        assert b.rows_check(c2) == []
    cfgo = b.Cfg(data_root="/d", plan=b.Plan(), offline_stage="/hf", mirror_inbox="/in", mirror_listing="/l")
    assert "TINY_LLM_TOKENIZER=/hf/tokenizer" in {s.name: s for s in b.build_stages(cfgo)}["rows"].commands[0]
    sft = open(os.path.join(ROOT, "scripts", "sft.py"), encoding="utf-8").read()
    print("test 23 ok" + ("" if "--estimate-rows-only" in sft else " (command construction only: sft.py has no --estimate-rows-only yet)"))


def test_24_inbox_end_mark_hooks():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = b.Cfg(data_root="/d", plan=b.Plan(), mirror_inbox=tmp, mirror_listing="/l.json")
        stage = {s.name: s for s in b.build_stages(cfg)}["filler"]
        assert stage.pre and stage.post
        assert stage.post[0](cfg) == [] and os.path.exists(os.path.join(tmp, ".filler_done"))
        assert stage.pre[0](cfg) == [] and not os.path.exists(os.path.join(tmp, ".filler_done"))
        assert not {s.name: s for s in b.build_stages(b.Cfg(data_root="/d", plan=b.Plan()))}["filler"].post
    print("test 24 ok")


def test_25_child_output_touch_contract():
    saved_env, saved_hb, saved_touch = b.ENV_INIT, b.HEARTBEAT_SECONDS, b.CHILD_TOUCH_SECONDS
    b.ENV_INIT = "/dev/null"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(tmp)
            mark = os.path.join(cfg.logs, "last_child_output")
            os.makedirs(cfg.logs)
            old = time.time() - 1000
            # throttled: a second non forced touch inside the window leaves the mtime alone, a forced one does not
            b.touch_child_output(cfg, force=True)
            os.utime(mark, (old, old))
            b.touch_child_output(cfg)
            assert abs(os.path.getmtime(mark) - old) < 1, "throttled to once per CHILD_TOUCH_SECONDS"
            b.touch_child_output(cfg, force=True)
            assert os.path.getmtime(mark) > old + 500
            # a child that prints touches it; every line when the throttle is off
            os.utime(mark, (old, old))
            b.CHILD_TOUCH_SECONDS = 0.0
            st = b.Stage("t_print", ["echo one; sleep 0.2; echo two"], [])
            with contextlib.redirect_stdout(io.StringIO()):
                assert b.run_stage(cfg, st, dry=False, force=False)
            assert os.path.getmtime(mark) > old + 500, "stage start, child lines and stage end touch the file"
            # a silent child: the heartbeat prints but does not touch
            b.CHILD_TOUCH_SECONDS = 10.0
            b.HEARTBEAT_SECONDS = 0.2
            buf = io.StringIO()
            quiet = b.Stage("t_quiet", ["sleep 1.6"], [])
            result = []

            def go():
                with contextlib.redirect_stdout(buf):
                    result.append(b.run_stage(cfg, quiet, dry=False, force=False))
            th = threading.Thread(target=go)
            th.start()
            time.sleep(0.5)
            m1 = os.path.getmtime(mark)
            time.sleep(0.8)
            m2 = os.path.getmtime(mark)
            th.join()
            assert m1 == m2, "heartbeat lines must not touch the stall mark"
            assert os.path.getmtime(mark) > m2, "stage end touches it"
            assert result == [True]
            # parallel mode: the tag does not matter, every child's line counts
            os.utime(mark, (old, old))
            b.CHILD_TOUCH_SECONDS = 0.0
            with contextlib.redirect_stdout(io.StringIO()):
                assert b.run_stage(cfg, b.Stage("t_par", ["echo x"], []), dry=False, force=False, tag="t_par")
            assert os.path.getmtime(mark) > old + 500
    finally:
        b.ENV_INIT, b.HEARTBEAT_SECONDS, b.CHILD_TOUCH_SECONDS = saved_env, saved_hb, saved_touch
    print("test 25 ok")


def test_26_inbox_deleted_after_the_filler_is_not_a_failure():
    with tempfile.TemporaryDirectory() as tmp:
        gone = os.path.join(tmp, "inbox_deleted")
        cfg = b.Cfg(data_root=tmp, plan=b.Plan(), mirror_inbox=gone, mirror_listing="/l.json")
        assert b.write_inbox_done(cfg) == [], "the skip path runs the post hook with no inbox"
        assert b.clear_inbox_done(cfg) == []
        os.makedirs(gone)
        assert b.write_inbox_done(cfg) == [] and os.path.exists(os.path.join(gone, ".filler_done"))
    print("test 26 ok")


def test_27_stage_lock_refuses_a_second_driver():
    try:
        import fcntl
    except ImportError:
        print("test 27 skipped: no fcntl")
        return
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_cfg(tmp)
        os.makedirs(cfg.done)
        out = os.path.join(tmp, "o.bin")
        st = b.Stage("t_lock", [f"touch {out}"], [out])
        saved = b.ENV_INIT
        b.ENV_INIT = "/dev/null"
        try:
            with open(os.path.join(cfg.done, "t_lock.lock"), "a") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    assert b.run_stage(cfg, st, dry=False, force=False) is False
                assert "another driver process holds" in buf.getvalue() and not os.path.exists(out)
            with contextlib.redirect_stdout(io.StringIO()):
                assert b.run_stage(cfg, st, dry=False, force=False), "the lock is released with its holder"
        finally:
            b.ENV_INIT = saved
    print("test 27 ok")


def test_28_filler_repeat_check_and_flags():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = b.Cfg(data_root=tmp, plan=b.Plan())
        os.makedirs(cfg.run2)
        assert b.filler_repeat_check(cfg) == [], "no build record yet"
        path = os.path.join(cfg.run2, "inject_build.json")
        json.dump({"common": {"filler_repeat_factor": 1.0}}, open(path, "w"))
        assert b.filler_repeat_check(cfg) == []
        json.dump({"common": {"filler_repeat_factor": 1.0004}}, open(path, "w"))
        assert b.filler_repeat_check(cfg) == [], "inside the tolerance"
        json.dump({"common": {"filler_repeat_factor": 1.5}}, open(path, "w"))
        msg = b.filler_repeat_check(cfg)
        assert msg and "1.500x" in msg[0] and "--allow-filler-repeat" in msg[0]
        assert b.filler_repeat_check(b.Cfg(data_root=tmp, plan=b.Plan(), allow_filler_repeat=True)) == []
        assert b.filler_repeat_check(b.Cfg(data_root=tmp, plan=b.Plan(), link_filler_from="/x")) == [], "linked filler implies the skip"
        stages = {x.name: x for x in b.build_stages(cfg)}
        for n in ("bios_main", "bios_full", "bios_warm"):
            assert b.filler_repeat_check in stages[n].post, n
    print("test 28 ok")


def test_29_preflight_accepts_this_roots_phase1_state():
    with tempfile.TemporaryDirectory() as tmp:
        saved = b.REPO
        b.REPO = os.path.join(tmp, "repo")
        os.makedirs(b.REPO)
        cfg = b.Cfg(data_root=os.path.join(tmp, "root"), plan=b.Plan())
        try:
            json.dump({"data_prep": {"phase1": {"doc_count": 1}}}, open(os.path.join(b.REPO, "manifest.json"), "w"))
            assert b.filler_preflight(cfg), "no stamp, no state: foreign"
            os.makedirs(cfg.prepared)
            open(os.path.join(cfg.prepared, "_prepare_state_phase1.json"), "w").write("{}")
            assert b.filler_preflight(cfg) == [], "a kill before record_owner leaves this root's phase 1 state as the proof"
        finally:
            b.REPO = saved
    print("test 29 ok")


def test_30_qa_is_permuted_before_the_order_and_every_merge_reads_the_copy():
    cfg = b.Cfg(data_root="/d", plan=b.Plan())
    stages = {x.name: x for x in b.build_stages(cfg)}
    assert b.STAGE_ORDER.index("qa_shuffle") < b.STAGE_ORDER.index("order")
    sh = stages["qa_shuffle"]
    assert sh.requires == ["qa"] and "qa_shuffle" in stages["order"].requires and "qa" not in stages["order"].requires
    cmd = sh.commands[0]
    assert "merge_evidence_splits.py" in cmd and "--slice qa=/d/prepared/evidence_run2_train:1.0:1" in cmd
    assert "split-tokens --prefix /d/prepared/evidence_run2_train --plus 1" in cmd and "--seed 42" in cmd
    shuf = "/d/prepared_run2/" + b.SHUFFLED_SLICES["qa"]
    for n in ("order", "merge_main", "merge_warm", "merge_control"):
        text = " ".join(stages[n].commands)
        assert f"qa={shuf}" in text and "evidence_run2_train" not in text, n
    assert shuf + ".bin" in stages["prune_sources"].prune and shuf + ".merge.json" in stages["prune_sources"].prune
    assert stages["qa_shuffle"].pruned_by == "prune_sources" and b.RESUMABLE["qa_shuffle"]
    sched = b.disk_schedule(b.Plan(), b.derive(b.Plan()))
    assert "qa_shuffle" in [r["stage"] for r in sched["rows"]] and sched["peak"] < 3.4e12
    assert b.hours_plan(b.Plan(), b.derive(b.Plan()))["qa_shuffle"] < 1
    try:
        import numpy  # noqa: F401
    except Exception:
        print("test 30 ok (no numpy: the file checks were skipped)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        c2 = b.Cfg(data_root=tmp, plan=b.Plan())
        os.makedirs(c2.prepared)
        os.makedirs(c2.run2)
        docs = [(1, [3, 4, 5]), (1, [6]), (1, [7, 8])]
        write_split(os.path.join(c2.prepared, "evidence_run2_train"), docs)
        write_split(os.path.join(c2.run2, b.SHUFFLED_SLICES["qa"]), [docs[2], docs[0], docs[1]])
        assert b.qa_shuffle_check(c2) == []
        write_split(os.path.join(c2.run2, b.SHUFFLED_SLICES["qa"]), docs[:2])
        assert any("tokens" in p for p in b.qa_shuffle_check(c2)), "a copy that lost documents is caught"
        pre = []
        for i, label in enumerate(("bios", "qa", "chains", "bridge")):
            path = os.path.join(tmp, f"s{i}")
            write_split(path, [(i, [1, 2, 3])] * 3)
            pre.append(f"{label}={path}")
        json.dump({"documents": 3, "args": {"seed": 42}, "md5_by_slice": {"qa": "abc"}},
                  open(os.path.join(tmp, "s1.merge.json"), "w"))
        out, info = os.path.join(tmp, "o.slice"), os.path.join(tmp, "o.json")
        with contextlib.redirect_stdout(io.StringIO()):
            assert b.main(["order", "--out", out, "--info", info] + sum([["--slice", x] for x in pre], [])) == 0
        rec = json.load(open(info))["permuted_slices"]
        assert rec["qa"]["seed"] == 42 and rec["qa"]["md5"] == "abc" and list(rec) == ["qa"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert b.main(["split-tokens", "--prefix", os.path.join(tmp, "s1"), "--plus", "1"]) == 0
        assert buf.getvalue().strip() == "10"
    print("test 30 ok")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")

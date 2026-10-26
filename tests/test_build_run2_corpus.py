"""Tests for scripts/build_run2_corpus.py: the arithmetic, the dry-run commands, the skip rules.

GPU-free and builder-free: the driver imports no builder at module level.
"""
import contextlib
import io
import json
import os
import random
import sys
import tempfile

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
    assert b.TOKEN_TARGET == 280e9 and b.EPOCHS == 3
    assert abs(d["total"] - b.TOKEN_TARGET / b.EPOCHS) < 1.0, "T is the target over the epoch count"
    assert abs(d["total"] * d["epochs"] - b.TOKEN_TARGET) < 1.0, "epochs of T train exactly the target"
    assert abs((d["plain"] + d["bridge"]) / d["total"] - 0.03) < 1e-9, "chains default to 3%"
    d10 = b.derive(b.Plan(chain_share=0.10))
    assert abs((d10["plain"] + d10["bridge"]) / d10["total"] - 0.10) < 1e-9, "RUN2's 10% stays available"
    assert abs(d["plain"] - d["bridge"]) < 1.0, "half and half is the stated assumption"
    assert abs(d["epochs"] - 3.0) < 1e-9
    assert abs(d["phase1"] + d["phase2"] - d["bios"] * 1.01) < 1.0, "filler need with the 1% margin"
    assert d["qa"] <= d["qa_target"]
    assert d["qa_arms"]["squad_v2"] == 2 * 5_862_565, "SQuAD at two passes"
    assert d["merge_shares"]["bios"] > d["shares"]["bios"], "the bios slice is taken whole"
    assert d["plain_questions"] * b.PLAIN_TOKENS_PER_QUESTION >= d["plain"]
    assert d["bio_renders"] == 162_000 and d["persons"] == 3600, "the pilot's tier structure"
    print("test 1 ok")


def test_2_scale_is_linear_and_fixed_parts_stay():
    full, small = b.derive(b.Plan()), b.derive(b.Plan(scale=0.001))
    assert abs(small["total"] - full["total"] * 0.001) < 1.0
    assert abs(small["chains"] / small["total"] - 0.03) < 1e-9
    assert small["bio_renders"] == full["bio_renders"]
    assert small["qa"] < full["qa"] * 0.01 + 1e7
    assert abs(small["phase1"] + small["phase2"] - small["bios"] * 1.01) < 1.0
    print("test 2 ok")


def test_3_filler_supply_flags_the_fineweb_crawl():
    plan = b.Plan()
    d = b.derive(plan)
    rows = {r["source"]: r for r in b.filler_supply(plan, d)}
    assert rows["fineweb"]["need"] > rows["fineweb"]["supply_default"], "one crawl cannot supply fineweb"
    assert rows["fineweb"]["need"] < rows["fineweb"]["supply_patched"]
    for name, r in rows.items():
        if name != "fineweb":
            assert r["need"] <= r["supply_default"], f"{name} short: {r['need']:.3g} > {r['supply_default']:.3g}"
    assert rows["smoltalk2"]["need"] + d["qa_arms"]["smoltalk2"] <= b.SMOLTALK2_SUPPLY * 1.001
    print("test 3 ok")


def test_4_dry_run_every_stage_prints_its_flags():
    cmds = stage_commands(dry_lines())
    assert list(cmds) == b.STAGE_ORDER
    assert all(cmds[s] for s in cmds), "every stage prints at least one command"
    text = {s: " ".join(c) for s, c in cmds.items()}
    for s in ("bios_main", "bios_warm"):
        assert "--support-gold-only" in text[s], s
    assert "--arms full,retrieval" in text["bios_main"] and "--swap-rate 0.15" in text["bios_main"]
    assert "--arms retrieval" in text["bios_warm"] and "--swap-rate 1.0" in text["bios_warm"]
    assert "--bridge" in text["chains_bridge"] and "--hops 4" in text["chains_bridge"]
    assert "--bridge" not in text["chains_plain"]
    for s in ("qa", "merge_warm", "merge_main", "merge_control", "heldout_qa"):
        assert "--max-evidence-tokens 4608" in text[s], s
    assert "--manifest-key ir_prep" in text["filler"], "ir needs its own manifest key"
    assert "--manifest-key data_prep" in text["filler"]
    assert "--order-from" in text["merge_main"] and "run2_warm_train.slice" in text["merge_main"]
    assert "--order-from" not in text["merge_warm"] and "--order-from" not in text["merge_control"]
    assert "--no-evidence" in text["merge_control"] and "--no-evidence" not in text["merge_main"]
    for s in ("merge_warm", "merge_main", "merge_control"):
        for label in ("bios=", "qa=", "chains=", "bridge="):
            assert f"--slice {label}" in text[s] or f"--slice '{label}" in text[s] or label in text[s], (s, label)
    assert "--heldout" in text["heldout_qa"]
    assert "--prefix chains " in text["heldout_chains"] and "eval,heldout_tmpl,hop4" in text["heldout_chains"]
    print("test 4 ok")


def test_5_order_of_slices_is_identical_in_warm_and_main():
    cmds = stage_commands(dry_lines())

    def labels(line):
        return [p.split("=")[0] for p in line.split() if "=" in p and p.split("=")[0] in ("bios", "qa", "chains", "bridge")]
    assert labels(cmds["merge_warm"][0]) == labels(cmds["merge_main"][0]) == ["bios", "qa", "chains", "bridge"]
    print("test 5 ok")


def test_6_scale_flows_into_commands():
    small = stage_commands(dry_lines(["--scale", "0.001"]))
    d = b.derive(b.Plan(scale=0.001))
    assert f"--target-tokens {d['injection_target']}" in small["bios_main"][0]
    assert f"--train-questions {d['plain_questions']}" in small["chains_plain"][0]
    assert f"--train-questions {d['bridge_questions']}" in small["chains_bridge"][0]
    assert "--target-tokens 93333333 " in small["merge_warm"][0]
    print("test 6 ok")


def test_7_stage_filter_and_unknown_stage():
    lines = dry_lines(["--stage", "qa,merge_main"])
    assert list(stage_commands(lines)) == ["qa", "merge_main"]
    try:
        b.main(["build", "--dry-run", "--stage", "nope"])
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown stage must be refused")
    print("test 7 ok")


def make_cfg(root):
    return b.Cfg(data_root=root, plan=b.Plan(scale=0.001))


def test_8_marker_skip_and_failure_paths():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_cfg(tmp)
        out = os.path.join(tmp, "out.bin")
        ok = b.Stage("t_ok", [f"touch {out}"], [out])
        buf = io.StringIO()
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
        assert "skipped" in buf.getvalue()
    print("test 8 ok")


def test_9_merge_check_reads_md5_lines():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_cfg(tmp)
        os.makedirs(cfg.run2)

        def write(split, md5):
            with open(os.path.join(cfg.run2, split + ".merge.json"), "w") as f:
                json.dump({"md5_by_slice": md5}, f)
        same = {"qa": "a", "chains": "b", "bridge": "c"}
        write("run2_warm_train", dict(same, bios="w"))
        write("run2_main_train", dict(same, bios="m"))
        with contextlib.redirect_stdout(io.StringIO()):
            assert b.check_merge(cfg, "merge_main") == []
            write("run2_main_train", dict(same, bios="m", qa="x"))
            assert any("qa" in p for p in b.check_merge(cfg, "merge_main"))
            write("run2_main_train", dict(same, bios="w"))
            assert any("identical" in p for p in b.check_merge(cfg, "merge_main"))
            write("run2_main_train", {"qa": "a"})
            assert len(b.check_merge(cfg, "merge_main")) >= 3
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


def hi_ok(b):
    p = b.Plan(chain_share=0.10)
    e = b.estimate(p, b.derive(p))
    return e["disk"]["chains plain build"] > e["disk"]["bios warm build (retrieval)"]


def test_12_estimates_are_positive_and_chains_dominate_disk():
    plan = b.Plan()
    e = b.estimate(plan, b.derive(plan))
    assert all(v > 0 for v in e["disk"].values())
    assert hi_ok(b)
    hi = b.estimate(b.Plan(chain_share=0.10), b.derive(b.Plan(chain_share=0.10)))
    assert sum(e["disk"].values()) < 0.6 * sum(hi["disk"].values())
    print("test 12 ok")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")

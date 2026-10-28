"""Tests for scripts/mirror_sources.py and the driver's cluster-side patches. No network: fakes only."""
import io
import json
import os
import sys
import tempfile
import contextlib
from types import ModuleType, SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import mirror_sources as m  # noqa: E402
import build_run2_corpus as b  # noqa: E402


def spec(key, repo, prefix, suffix, filt=None, p1=0.5, p2=0.0):
    return SimpleNamespace(key=key, repo_id=repo, file_prefix=prefix, file_suffix=suffix, file_filter=filt,
                           phase1_weight=p1, phase2_weight=p2, ir_weight=0.0)


class FakeApi:
    def __init__(self, files):
        self.files = files

    def dataset_info(self, repo, files_metadata=False):
        return SimpleNamespace(sha="sha_" + repo, siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in self.files[repo].items()])


def make_listing():
    files = {
        "a/web": {"data/z.parquet": 10, "data/a.parquet": 10, "data/b.parquet": 10, "data/c.parquet": 10,
                  "data/readme.md": 1, "other/x.parquet": 10},
        "b/chat": {"SFT/f_no_think.parquet": 5, "SFT/g_think.parquet": 5, "SFT/h_no_think.parquet": 5},
    }
    specs = [spec("web", "a/web", "data/", (".parquet",)),
             spec("chat", "b/chat", "SFT/", (".parquet",), lambda f: "_no_think" in f, p1=0.0, p2=0.5)]
    plan = {"web": {"per_file": 10e6, "need": {"phase1": 20e6, "phase2": 10e6, "ir": 0.0}},
            "chat": {"per_file": 5e6, "need": {"phase1": 0.0, "phase2": 5e6, "ir": 5e6}}}
    return m.build_listing(specs, FakeApi(files), plan, 0.0, None, created="t")


def test_1_listing_filter_order_and_need():
    lst = make_listing()
    web = lst["sources"]["web"]
    assert [f["name"] for f in web["files"]] == ["data/a.parquet", "data/b.parquet", "data/c.parquet", "data/z.parquet"]
    assert web["revision"] == "sha_a/web" and web["phase_files"] == [2, 3, 3] and web["needed"] == 4
    chat = lst["sources"]["chat"]
    assert [f["name"] for f in chat["files"]] == ["SFT/f_no_think.parquet", "SFT/h_no_think.parquet"]
    assert chat["phase_files"] == [0, 1, 2] and chat["needed"] == 2 and not chat["short"]
    assert m.total_bytes(lst) == {"web": 40, "chat": 10}
    short = m.build_listing([spec("web", "a/web", "data/", (".parquet",))], FakeApi({"a/web": {"data/a.parquet": 1}}),
                            {"web": {"per_file": 1.0, "need": {"phase1": 5.0}}}, 0.0, None)
    assert short["sources"]["web"]["short"]
    wide = m.build_listing([spec("fineweb", "a/web", "data/", (".parquet",))], FakeApi({"a/web": {"other/a.parquet": 1, "data/b.parquet": 1}}),
                           {"fineweb": {"per_file": 1.0, "need": {"phase1": 1.0}}}, 0.0, "other/")
    assert [f["name"] for f in wide["sources"]["fineweb"]["files"]] == ["other/a.parquet"], "the fineweb prefix override"
    print("test 1 ok")


def test_2_consumption_order_is_phase_then_progress_and_sorted_per_source():
    lst = make_listing()
    order = m.consumption_order(lst)
    web = [r["name"] for r in order if r["source"] == "web"]
    assert web == sorted(web), "per source the sorted list is consumed front to back"
    assert [r["phase"] for r in order if r["source"] == "web"] == [0, 0, 1, 2]
    phases = [r["phase"] for r in order]
    assert phases == sorted(phases), "every phase 0 file before any phase 1 file"
    assert len(order) == 6
    print("test 2 ok")


class FakeRemote:
    def __init__(self):
        self.files = {}
        self.sent = []

    def ls(self):
        return dict(self.files)

    def send(self, local, rel, size):
        self.files[rel] = size
        self.sent.append(rel)

    def consume(self, rel):
        self.files.pop(rel)


def make_pusher(tmp, lst, remote, window=1e9, limit=1e12, max_present=None):
    removed = []
    p = m.Pusher(lst, os.path.join(tmp, "state.json"), fetch=lambda row: "/local/" + row["name"], send=remote.send,
                 remote_ls=remote.ls, window_bytes=window, transfer_limit=limit, max_present=max_present,
                 remove=removed.append, log=lambda *a: None)
    p.removed = removed
    return p


def long_listing(n=10, needed=6, per_file=10e6, sizes=None, repo="a/web", key="web"):
    files = [{"name": f"data/{i:03d}.parquet", "size": (sizes[i] if sizes else 10)} for i in range(n)]
    src = {"repo_id": repo, "revision": "r", "prefix": "data/", "files": files, "per_file_tokens": per_file,
           "need_tokens": {"phase1": needed * per_file}, "phase_files": [needed, needed, needed], "needed": needed,
           "short": False}
    src["tail"] = m.source_tail(src)
    src["max_present"] = m.cap_for_tail(src["tail"])
    return {"version": 1, "sources": {key: src}}


class InboxRemote:
    """The inbox as a directory: what the pusher lists and sends into is what the fetcher reads and deletes."""

    def __init__(self, root):
        self.root = root

    def ls(self):
        out = {}
        for d, _, fs in os.walk(self.root):
            for f in fs:
                out[os.path.relpath(os.path.join(d, f), self.root).replace(os.sep, "/")] = os.path.getsize(os.path.join(d, f))
        return out

    def send(self, local, rel, size):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * size)


def pair(tmp, lst, **kw):
    """A pusher and a fetcher on one inbox directory; the fetcher's wait polls the pusher."""
    inbox = os.path.join(tmp, "inbox")
    os.makedirs(inbox, exist_ok=True)
    remote = InboxRemote(inbox)
    pusher = m.Pusher(lst, os.path.join(tmp, "state.json"), fetch=lambda row: "/local/" + row["name"], send=remote.send,
                      remote_ls=remote.ls, window_bytes=1e9, remove=lambda p: None, log=lambda *a: None, **kw)
    waits = {"n": 0}

    def sleep(s):
        waits["n"] += 1
        assert waits["n"] < 200, "deadlock: the consumer waits and the pusher will not send"
        pusher.step()
    fetcher = m.InboxFetcher(inbox, m.MirrorListing(lst), sleep=sleep, log=lambda *a: None)
    return pusher, fetcher, remote, inbox


def test_3_pusher_is_demand_driven_and_resumable():
    with tempfile.TemporaryDirectory() as tmp:
        lst = long_listing(n=10, needed=10)
        assert lst["sources"]["web"]["tail"] == 2 and lst["sources"]["web"]["max_present"] == 5
        remote = FakeRemote()
        p = make_pusher(tmp, lst, remote, max_present=None)
        assert [p.step() for _ in range(6)] == ["pushed"] * 5 + ["wait"], "cap of five present for the source"
        assert remote.sent == [f"a__web/data/{i:03d}.parquet" for i in range(5)]
        assert len(p.removed) == 5, "local copies are deleted after a push"
        remote.consume("a__web/data/000.parquet")
        assert p.step() == "pushed" and remote.sent[-1] == "a__web/data/005.parquet"
        p2 = make_pusher(tmp, lst, remote, max_present=None)
        assert len(p2.done) == 6, "the done list survives a restart"
        while p2.step() != "finished":
            for rel in list(remote.files):
                remote.consume(rel)
        assert sorted(remote.sent) == sorted(set(remote.sent)), "nothing is pushed twice"
        assert len(remote.sent) == 10 and p2.state["pushed_bytes"] == 100
        two = make_listing()
        remote = FakeRemote()
        q = make_pusher(tempfile.mkdtemp(), two, remote, max_present=None)
        assert [q.step() for _ in range(7)] == ["pushed"] * 6 + ["finished"], "two sources, every needed file once"
    print("test 3 ok")


def test_4_pusher_window_limit_and_dot_files():
    with tempfile.TemporaryDirectory() as tmp:
        lst = make_listing()
        remote = FakeRemote()
        p = make_pusher(tmp, lst, remote, window=30, max_present=5)
        assert [p.step() for _ in range(3)] == ["pushed"] * 3, "two web files and one chat file, 25 bytes"
        assert p.step() == "wait", "25 bytes in the inbox plus the next 10 exceeds the window of 30"
        remote.files["a__web/data/.b.parquet.XYZ"] = 3
        remote.consume("a__web/data/a.parquet")
        assert p.step() == "pushed"
        assert remote.ls().get("a__web/data/.b.parquet.XYZ") == 3
        tight = make_pusher(tempfile.mkdtemp(), lst, FakeRemote(), limit=15)
        tight.step()
        try:
            tight.step()
        except RuntimeError:
            pass
        else:
            raise AssertionError("the transfer limit must stop the push")
        assert p.worst_case_window() == 5 * 10 + 5 * 5
    print("test 4 ok")


def test_5_inbox_fetcher_waits_and_deletes_by_listing_index():
    with tempfile.TemporaryDirectory() as tmp:
        lst = make_listing()
        ml = m.MirrorListing(lst)
        clock = {"t": 0.0}
        logs = []

        def sleep(s):
            clock["t"] += s
            if clock["t"] >= 12:
                with open(os.path.join(tmp, "a__web", "data", "a.parquet"), "wb") as f:
                    f.write(b"x" * 10)
        os.makedirs(os.path.join(tmp, "a__web", "data"))
        fetcher = m.InboxFetcher(tmp, ml, sleep=sleep, clock=lambda: clock["t"], log=logs.append, wait_log_seconds=5, poll_seconds=5)
        with open(os.path.join(tmp, "a__web", "data", "a.parquet.part"), "wb") as f:
            f.write(b"y")
        path = fetcher.download("a/web", "data/a.parquet")
        assert os.path.getsize(path) == 10 and clock["t"] >= 12 and logs and "waiting" in logs[0]
        for n in ("b", "c", "z"):
            with open(os.path.join(tmp, "a__web", "data", n + ".parquet"), "wb") as f:
                f.write(b"x" * 10)
        fetcher.download("a/web", "data/b.parquet")
        fetcher.download("a/web", "data/c.parquet")
        assert os.path.exists(os.path.join(tmp, "a__web", "data", "a.parquet")), "the file two before is kept"
        fetcher.download("a/web", "data/z.parquet")
        assert not os.path.exists(os.path.join(tmp, "a__web", "data", "a.parquet")), "three before is deleted"
        assert os.path.exists(os.path.join(tmp, "a__web", "data", "b.parquet"))
        assert os.path.exists(os.path.join(tmp, "a__web", "data", "a.parquet.part")), "rsync temp files are never touched"
        assert fetcher.tail == 2
        fake_os = m.NoRemoveOs(os, tmp)
        keep = os.path.join(tmp, "a__web", "data", "keep")
        open(keep, "w").close()
        assert fake_os.remove(keep) is None and os.path.exists(keep), "the generator's own remove is ignored in the inbox"
        outside = tempfile.NamedTemporaryFile(delete=False)
        outside.close()
        fake_os.remove(outside.name)
        assert not os.path.exists(outside.name)
        assert fake_os.path.join("a", "b") == os.path.join("a", "b")
    print("test 5 ok")


def test_5b_c3_files_of_a_dead_process_are_deleted_by_the_next_open():
    with tempfile.TemporaryDirectory() as tmp:
        lst = long_listing(n=10, needed=10)
        pusher, fetcher, remote, inbox = pair(tmp, lst)
        for i in range(6):
            remote.send("", f"a__web/data/{i:03d}.parquet", 10)
        for i in range(6, 8):
            remote.send("", f"a__web/data/{i:03d}.parquet", 10)
        fresh = m.InboxFetcher(inbox, m.MirrorListing(lst), sleep=lambda s: None, log=lambda *a: None)
        fresh.download("a/web", "data/007.parquet")
        left = sorted(remote.ls())
        assert left == [f"a__web/data/{i:03d}.parquet" for i in (5, 6, 7)], left
        assert pusher.caps["web"] == 5 and len(left) < pusher.caps["web"], "the cap is free again"
    print("test 5b ok")


def test_5c_m4_phase_end_clear_keeps_the_highest_opened_file():
    with tempfile.TemporaryDirectory() as tmp:
        lst = long_listing(n=10, needed=10)
        pusher, fetcher, remote, inbox = pair(tmp, lst)
        for i in range(8):
            remote.send("", f"a__web/data/{i:03d}.parquet", 10)
        for i in (3, 4, 5):
            fetcher.download("a/web", f"data/{i:03d}.parquet")
        fetcher.clear()
        left = sorted(remote.ls())
        assert left[0] == "a__web/data/005.parquet" and "a__web/data/007.parquet" in left, left
        assert "a__web/data/003.parquet" not in left and "a__web/data/004.parquet" not in left
        assert "a__web/data/005.parquet" in left, "the file opened last may be uncommitted and the next phase reads it"
    print("test 5c ok")


def test_5d_h1_tail_is_per_repo_and_the_cap_follows_it():
    with tempfile.TemporaryDirectory() as tmp:
        sizes = [10] * 9 + [1]
        small = long_listing(n=10, needed=10, sizes=sizes, repo="s/small", key="small")
        src = small["sources"]["small"]
        assert src["tail"] > 2 and src["max_present"] >= src["tail"] + 2, "one small shard raises that repo's tail and cap"
        big = long_listing(n=10, needed=10)
        assert big["sources"]["web"]["tail"] == 2 and big["sources"]["web"]["max_present"] == 5
        both = {"version": 1, "sources": {"small": src, "web": big["sources"]["web"]}}
        p = m.Pusher(both, os.path.join(tmp, "s.json"), fetch=None, send=None, remote_ls=lambda: {}, window_bytes=1e9)
        assert p.caps["web"] == 5 and p.caps["small"] == src["max_present"], "caps are per repo"
        try:
            m.Pusher(both, os.path.join(tmp, "s2.json"), fetch=None, send=None, remote_ls=lambda: {}, window_bytes=1e9,
                     max_present=5)
        except ValueError as e:
            assert "small" in str(e)
        else:
            raise AssertionError("a cap under tail + 2 must fail at construction")
        ml = m.MirrorListing(both)
        f = m.InboxFetcher(tmp, ml, sleep=lambda s: None, log=lambda *a: None)
        f.note_rows(10_000, "s/small")
        assert f.tail_for("a/web") == 2
        f.note_rows(1_200, "a/web")
        assert f.tail_for("a/web") == 1 + 2 and f.tail_for("s/small") == src["tail"], "a short file moves only its own repo"
        try:
            f.note_rows(10, "a/web")
        except RuntimeError as e:
            assert "cap" in str(e)
        else:
            raise AssertionError("a tail the cap cannot hold must stop the build, not hang it")
        bad = {"version": 1, "sources": {"web": dict(big["sources"]["web"], max_present=3)}}
        try:
            m.check_caps(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("check_caps must refuse a listing whose cap cannot hold its tail")
    print("test 5d ok")


def test_5e_consumer_and_pusher_do_not_deadlock_across_kills_and_small_shards():
    with tempfile.TemporaryDirectory() as tmp:
        sizes = [10, 10, 1, 10, 10, 10, 10, 1, 10, 10, 10, 10]
        lst = long_listing(n=12, needed=12, sizes=sizes)
        pusher, fetcher, remote, inbox = pair(tmp, lst)
        names = [f["name"] for f in lst["sources"]["web"]["files"]]
        for n in names[:7]:
            fetcher.download("a/web", n)
        # the consumer process dies; a new one resumes at the file the generator was on
        fresh = m.InboxFetcher(inbox, m.MirrorListing(lst), sleep=fetcher.sleep, log=lambda *a: None)
        for n in names[6:]:
            fresh.download("a/web", n)
        assert len(remote.ls()) <= pusher.caps["web"], remote.ls()
    print("test 5e ok")


def test_5f_h2_a_wrong_estimate_does_not_hang_the_build():
    with tempfile.TemporaryDirectory() as tmp:
        lst = long_listing(n=12, needed=3)
        pusher, fetcher, remote, inbox = pair(tmp, lst)
        names = [f["name"] for f in lst["sources"]["web"]["files"]]
        for n in names[:10]:
            fetcher.download("a/web", n)
        sent = pusher.state["pushed_bytes"] // 10
        assert sent >= 10, "the pusher went on past needed while the consumer kept asking"
        assert sent <= 10 + 3, "and only a little ahead of it"
        with open(os.path.join(inbox, m.DONE_MARK), "w") as f:
            f.write("done")
        assert pusher.step() == "finished", "the filler's end mark ends the pusher"
        many = long_listing(n=12, needed=3)
        sub = os.path.join(tmp, "b")
        os.makedirs(sub)
        p2, _, r2, _ = pair(sub, many)
        steps = [p2.step() for _ in range(40)]
        assert steps.count("pushed") <= 5 and steps[-1] == "wait", "nobody consumes: the pusher stops at the cap"
    print("test 5f ok")


def test_5g_m5_a_file_already_in_the_inbox_is_not_sent_twice():
    with tempfile.TemporaryDirectory() as tmp:
        lst = long_listing(n=6, needed=6)
        pusher, fetcher, remote, inbox = pair(tmp, lst)
        remote.send("", "a__web/data/000.parquet", 10)       # sent, then the pusher died before saving
        sent = []
        real = remote.send
        pusher.send = lambda local, rel, size: (sent.append(rel), real(local, rel, size))
        assert pusher.step() == "pushed"
        assert sent == ["a__web/data/001.parquet"], sent
        assert pusher.state["pushed_bytes"] == 10, "the present file is not counted twice"
        assert "a__web/data/000.parquet" in pusher.done
    print("test 5g ok")


def test_5h_list_takes_the_drivers_plan_flags():
    seen = {}
    real = m.cmd_list
    m.cmd_list = lambda a: seen.update(vars(a)) or 0
    try:
        assert m.main(["list", "--out", "x.json", "--scale", "0.5", "--qa-target", "1e9", "--margin", "0.2"]) == 0
    finally:
        m.cmd_list = real
    import argparse
    plan = b.plan_from_args(argparse.Namespace(**seen))
    assert plan.scale == 0.5 and plan.qa_target == 1e9 and seen["margin"] == 0.2
    assert "needed" in m.__doc__ and "not a stop" in m.__doc__
    print("test 5h ok")


def test_5i_tokenizer_check_is_exactly_the_path_given():
    with tempfile.TemporaryDirectory() as tmp:
        stage = os.path.join(tmp, "hf")
        os.makedirs(os.path.join(stage, "tokenizer"))
        open(os.path.join(stage, "tokenizer", "tokenizer.json"), "w").close()
        other = os.path.join(tmp, "elsewhere")
        os.makedirs(other)
        ok, why = m.check_need("tokenizer", stage, other, None, None)
        assert not ok and "elsewhere" in why, "a staged tokenizer does not stand in for the path the builders read"
        assert m.check_need("tokenizer", stage, None, None, None)[0], "no path given: the staged copy"
        assert m.check_need("tokenizer", stage, os.path.join(stage, "tokenizer"), None, None)[0]
    print("test 5i ok")


def test_6_fake_api_answers_from_the_listing():
    ml = m.MirrorListing(make_listing())
    api = m.FakeHfApi(ml, token=None)
    assert api.dataset_info("a/web").sha == "sha_a/web"
    assert api.list_repo_files("a/web", repo_type="dataset", revision="x")[0] == "data/a.parquet"
    print("test 6 ok")


def test_7_stage_small_and_offline_hub():
    with tempfile.TemporaryDirectory() as tmp:
        names = {
            "rajpurkar/squad_v2": ["squad_v2/train-0.parquet", "squad_v2/validation-0.parquet", "README.md"],
            "hotpotqa/hotpot_qa": ["distractor/train-0.parquet", "distractor/validation-0.parquet", "fullwiki/x.parquet"],
            "HuggingFaceTB/smoltalk2": ["SFT/a_no_think.parquet", "SFT/b_think.parquet"],
            "HuggingFaceH4/ultrachat_200k": ["data/train_sft-0.parquet", "data/test_sft-0.parquet"],
            "HuggingFaceH4/no_robots": ["data/train-0.parquet", "data/test-0.parquet"],
            "florin-hf/nq_open_gold": ["test_dataset.json", "train.json"],
            "mandarjoshi/trivia_qa": ["rc.wikipedia/validation-0.parquet", "rc.wikipedia/train-0.parquet"],
        }
        for repo, rev, prefixes, _ in m.BENCH_DATASETS:
            ext = "" if repo == "akariasai/PopQA" else ".parquet"
            names.setdefault(repo, []).extend(p + ext for p in prefixes)
            names[repo].append("README.md")
        names["mandarjoshi/trivia_qa"].append("rc.nocontext/train-0.parquet")
        api = SimpleNamespace(dataset_info=lambda r: SimpleNamespace(sha="s_" + r),
                              list_repo_files=lambda r, repo_type=None, revision=None: names[r])

        def download(repo, name, sha, local_dir):
            path = os.path.join(local_dir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(b"data:" + name.encode())
            return path
        snaps = []
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            idx = m.stage_small(tmp, api, download, lambda repo, rev, ld, cd: snaps.append(repo) or "rev")
        assert idx["datasets"]["hotpotqa/hotpot_qa"]["files"] == ["distractor/train-0.parquet", "distractor/validation-0.parquet"]
        assert idx["datasets"]["HuggingFaceTB/smoltalk2"]["files"] == ["SFT/a_no_think.parquet"]
        assert idx["datasets"]["florin-hf/nq_open_gold"]["files"] == ["test_dataset.json"]
        trivia = idx["datasets"]["mandarjoshi/trivia_qa"]["files"]
        assert trivia == ["rc.nocontext/validation.parquet", "rc.wikipedia/validation-0.parquet"], trivia
        assert idx["datasets"]["ybisk/piqa"]["files"] == ["plain_text/validation.parquet"]
        assert idx["datasets"]["akariasai/PopQA"]["files"] == ["test.tsv"]
        assert snaps == [m.BGE_REPO, m.TOKENIZER_REPO]
        hub = ModuleType("fake_hub")
        hub.HfApi = type("HfApi", (), {})
        hub.hf_hub_download = None
        m.install_offline_hub(os.path.join(tmp, "datasets"), hub=hub)
        a = hub.HfApi()
        assert a.dataset_info("rajpurkar/squad_v2").sha == "s_rajpurkar/squad_v2"
        files = a.list_repo_files("rajpurkar/squad_v2", repo_type="dataset", revision="s")
        assert files == ["squad_v2/train-0.parquet", "squad_v2/validation-0.parquet"]
        scratch = os.path.join(tmp, "scratch")
        got = hub.hf_hub_download(repo_id="rajpurkar/squad_v2", filename=files[0], repo_type="dataset", local_dir=scratch,
                                  token=None, revision="s")
        assert open(got, "rb").read() == b"data:squad_v2/train-0.parquet"
        os.remove(got)
        assert os.path.exists(os.path.join(tmp, "datasets", "rajpurkar__squad_v2", files[0])), "the staged copy survives the builder's delete"
        try:
            hub.hf_hub_download(repo_id="rajpurkar/squad_v2", filename="nope", local_dir=scratch)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a file outside the stage must fail loudly")
    print("test 7 ok")


def test_8_phase1_weight_patch_and_driver_plan_feed():
    specs = [spec("fineweb", "x", "", ()), spec("code_edu", "y", "", ()), spec("smoltalk2", "z", "", (), p1=0.0, p2=0.3)]
    specs[0].phase1_weight, specs[1].phase1_weight = 0.5, 0.5
    lines = []
    b.apply_phase1_weights(specs, {"fineweb": 0.8, "code_edu": 0.2}, lines.append)
    assert specs[0].phase1_weight == 0.8 and specs[1].phase1_weight == 0.2 and specs[2].phase1_weight == 0.0
    assert len(lines) == 2 and "0.5000 -> 0.8000" in lines[0]
    assert b.parse_weights("a=0.5,b=0.25") == {"a": 0.5, "b": 0.25}
    plan = b.Plan()
    d = b.derive(plan)
    fp = b.filler_file_need(plan, d)
    assert set(fp) == set(b.SOURCE_SUPPLY) and fp["fineweb"]["need"]["phase1"] > fp["fineweb"]["need"]["phase2"]
    t = b.transfer_estimate(plan, d)
    assert sum(v["bytes"] for v in t.values()) < 2e12
    print("test 8 ok")


def test_9_earliest_file_is_sent_when_later_lookahead_fills_the_window():
    with tempfile.TemporaryDirectory() as tmp:
        lst = make_listing()
        remote = FakeRemote()
        remote.files["a__web/data/z.parquet"] = 10     # a later phase file of the same source sits in the inbox
        p = make_pusher(tmp, lst, remote, window=12, max_present=5)
        assert p.step() == "pushed", "the file the consumer needs first must not wait behind later lookahead"
        assert "a__web/data/a.parquet" in remote.sent
        # an earlier file still in the inbox keeps the window in force
        assert p.step() == "wait", "10 + 10 bytes present, the next file is not the one the consumer waits on"
    print("test 9 ok")


def test_10_shared_ledger_ssh_options_and_partial_dir():
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        led = os.path.join(tmp, "shared.tsv")
        m.ledger_append(led, "in", "code worktree", 1_500_000_000)
        m.ledger_append(led, "out", "exports", 600_000_000)
        assert m.ledger_total(led) == 2_100_000_000
        assert m.default_ledger({"OPS_LEDGER": led}) == led
        assert os.path.isabs(m.default_ledger({})) and m.default_ledger({}).endswith("ops_transfer_ledger.tsv")
        # the budget is checked against the ledger's total, so another script's transfers count
        lst = make_listing()
        remote = FakeRemote()
        p = m.Pusher(lst, os.path.join(tmp, "s.json"), fetch=lambda row: "/local/" + row["name"], send=remote.send,
                     remote_ls=remote.ls, window_bytes=1e9, transfer_limit=2_100_000_005, max_present=5,
                     remove=lambda f: None, log=lambda *a: None, ledger_path=led)
        try:
            p.step()
            raise AssertionError("expected the shared budget to refuse a 10 byte file")
        except RuntimeError as e:
            assert "transfer limit" in str(e)
        p.limit = 5e12
        assert p.step() == "pushed"
        rows = open(led).read().splitlines()
        assert rows[-1].split("	")[1] == "in" and rows[-1].split("	")[2].startswith("mirror ") and rows[-1].split("	")[3] == "10", rows
        # a failed send is counted too: the partial's size in the inbox, else the full size
        remote.files["a__web/.rsync-partial/b.parquet"] = 4

        def broken(local, rel, size):
            raise OSError("connection dropped")
        p.send = broken
        before = m.ledger_total(led)
        try:
            p.step()
            raise AssertionError("expected the failed send to raise")
        except OSError:
            pass
        assert m.ledger_total(led) > before and "FAILED" in open(led).read().splitlines()[-1]
        p.send = remote.send
        # the rsync goes through the ops ssh options and a partial dir, never a plain --partial
        sock = os.path.join(tmp, "t.sock")
        open(sock, "w").close()
        opts = m.ssh_opts({"SSH_SOCK_TRANSFER": sock})
        assert "BatchMode=yes" in opts and "ControlPath=" + sock in opts
        seen = []
        real = subprocess.run
        subprocess.run = lambda cmd, **kw: (seen.append(cmd), SimpleNamespace(stdout=""))[1]
        try:
            m.rsync_send("u@h", "/inbox", opts)("/local/f", "repo/f", 1)
            m.ssh_remote_ls("u@h", "/inbox", opts)()
        finally:
            subprocess.run = real
        rs = [c for c in seen if c[0] == "rsync"][0]
        assert "--partial-dir=.rsync-partial" in rs and "--partial" not in rs and "BatchMode=yes" in rs[rs.index("-e") + 1], rs
        assert all("BatchMode=yes" in c for c in seen if c[0] == "ssh"), seen
        # a partial file under .rsync-partial is not a finished inbox file
        assert m.is_temp("repo/.rsync-partial/x.parquet") and not m.is_temp("repo/data/x.parquet")
    print("test 10 ok")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all tests passed")

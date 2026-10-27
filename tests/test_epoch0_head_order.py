"""Head-first epoch 0 order, the grad accumulation override and the head check on resume.

A split with ``{split}.order.json`` (``epoch0_head_docs`` H) reads its first H documents in epoch 0
in the order a split of just those H documents would, then the rest from its own stream. Without
the sidecar the order is bit identical to the plain seeded permutation.

Run: `python tests/test_epoch0_head_order.py` (the last test imports scripts.sft, so it needs the
WSL environment; the rest is GPU-free).
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.data.sft_dataset import SFTDataset, epoch_order, read_epoch0_head_docs


class Tok:
    bos_token_id = 500
    eos_token_id = 501
    pad_token_id = 501


def write_split(directory, split, num_docs):
    ids = []
    idx = [0]
    for d in range(num_docs):
        ids += [500, 1000 + d, 7]
        idx.append(len(ids))
    np.asarray(ids, dtype=np.uint16).tofile(os.path.join(directory, f"{split}.bin"))
    np.asarray(idx, dtype=np.uint64).tofile(os.path.join(directory, f"{split}.idx"))
    np.ones(len(ids), dtype=np.uint8).tofile(os.path.join(directory, f"{split}.mask"))


def write_order(directory, split, head, **extra):
    with open(os.path.join(directory, f"{split}.order.json"), "w") as f:
        json.dump({"epoch0_head_docs": head, **extra}, f)


def make(directory, split, **kwargs):
    return SFTDataset(directory, Tok(), batch_size=1, max_length=4, split=split,
                      num_mtp_tokens=1, seed=42, **kwargs)


def read_docs(ds):
    out = []
    for batch in ds._batch_iterator():
        out += [int(t) - 1000 for t in batch["input_ids"].flatten().tolist() if 1000 <= t < 1500]
    return out


def test_order_function():
    plain = np.random.default_rng([42, 0]).permutation(50)
    assert (epoch_order(42, 0, 50, None) == plain).all()
    assert (epoch_order(42, 3, 50, 10) == np.random.default_rng([42, 3]).permutation(50)).all()
    order = epoch_order(42, 0, 50, 10)
    assert sorted(order.tolist()) == list(range(50))
    assert (order[:10] == np.random.default_rng([42, 0]).permutation(10)).all()
    assert sorted(order[:10].tolist()) == list(range(10))
    assert sorted(order[10:].tolist()) == list(range(10, 50))
    assert (epoch_order(42, 0, 50, 50) == np.random.default_rng([42, 0]).permutation(50)).all()
    for bad in (0, 51, -1):
        try:
            epoch_order(42, 0, 50, bad)
        except ValueError:
            continue
        raise AssertionError(f"head {bad} accepted")


def test_sidecar_and_validation():
    d = tempfile.mkdtemp()
    try:
        write_split(d, "main", 40)
        assert read_epoch0_head_docs(os.path.join(d, "main.order.json"), 40) is None
        write_order(d, "main", 12, other="kept")
        assert make(d, "main").epoch0_head_docs == 12
        assert make(d, "main", shuffle=False).epoch0_head_docs is None
        for bad in (0, 41, "x", 1.5, None, True):
            write_order(d, "main", bad)
            try:
                make(d, "main")
            except ValueError:
                continue
            raise AssertionError(f"sidecar head {bad!r} accepted")
        with open(os.path.join(d, "main.order.json"), "w") as f:
            f.write("{}")
        try:
            make(d, "main")
        except ValueError:
            pass
        else:
            raise AssertionError("missing key accepted")
    finally:
        shutil.rmtree(d)


def test_warm_prefix_and_switch():
    d = tempfile.mkdtemp()
    try:
        write_split(d, "warm", 12)
        write_split(d, "main", 40)
        write_order(d, "main", 12)
        warm = read_docs(make(d, "warm"))
        main = read_docs(make(d, "main"))
        assert main[:12] == warm and sorted(main) == list(range(40))
        # a switch at position p visits the unseen head documents, then the rest, no repeats
        for p in (0, 5, 11):
            resumed = read_docs(make(d, "main", start_doc_idx=p))
            assert resumed == main[p:], p
            assert set(warm[:p]).isdisjoint(resumed)
        # epoch 1 unchanged by the sidecar
        e1 = make(d, "main")
        e1.set_epoch(1)
        plain = make(d, "warm")
        assert (e1.document_order(40) == np.random.default_rng([42, 1]).permutation(40)).all()
        del plain
    finally:
        shutil.rmtree(d)


def test_no_sidecar_identical():
    d = tempfile.mkdtemp()
    try:
        write_split(d, "main", 30)
        ds = make(d, "main")
        assert ds.epoch0_head_docs is None
        assert (ds.document_order(30) == np.random.default_rng([42, 0]).permutation(30)).all()
    finally:
        shutil.rmtree(d)


def test_rank_streams_cover_once():
    d = tempfile.mkdtemp()
    try:
        write_split(d, "main", 37)
        write_order(d, "main", 11)
        for world in (1, 2, 4):
            seen = []
            for rank in range(world):
                seen += read_docs(make(d, "main", rank=rank, world_size=world))
            assert sorted(seen) == list(range(37)), world
    finally:
        shutil.rmtree(d)


def test_resume_check():
    from scripts.sft import check_head_docs, plan_changed
    check_head_docs(None, None, True)
    check_head_docs(12, 12, True)
    check_head_docs(None, 12, False)
    check_head_docs(12, None, False)
    for saved, current in ((None, 12), (12, None), (12, 13)):
        try:
            check_head_docs(saved, current, True)
        except SystemExit:
            continue
        raise AssertionError((saved, current))
    moved = plan_changed({"total_steps": 10, "grad_accumulation_steps": 2},
                         {"total_steps": 10, "grad_accumulation_steps": 1})
    assert moved == [("grad_accumulation_steps", 2, 1)], moved
    assert plan_changed({"total_steps": 10}, {"total_steps": 10, "grad_accumulation_steps": 1}) == []


def test_order_memory_form_identical():
    # the in place form against the two permutations joined, which the order was defined by
    for n, h in ((1, 1), (2, 1), (7, 3), (50, 10), (50, 50), (1000, 1), (1000, 999), (4097, 1500)):
        for seed in (0, 42):
            head = np.random.default_rng([seed, 0]).permutation(h)
            rest = h + np.random.default_rng([seed, 0, 1]).permutation(n - h)
            want = np.concatenate([head, rest]).astype(np.int64, copy=False)
            got = epoch_order(seed, 0, n, h)
            assert got.dtype == np.int64 and got.tobytes() == want.tobytes(), (n, h, seed)
    # the plain path is the permutation itself, for epoch 0 without a head and every later epoch
    assert epoch_order(42, 0, 50).tobytes() == np.random.default_rng([42, 0]).permutation(50).tobytes()
    assert epoch_order(42, 2, 50, 10).tobytes() == np.random.default_rng([42, 2]).permutation(50).tobytes()


def test_switch_state():
    from scripts.sft import check_switch_state as check
    # sound: epoch 0, inside the head, the warm split's count is the new head
    assert check(0, 500, 1000, 4000, 1000) is None
    assert check(0, 1000, 1000, 4000, 1000) is None
    # pilot style: no sidecar, equal counts
    assert check(0, 700, 1000, 1000, None) is None
    # H2: the warm run crossed its epoch boundary
    assert "epoch 1" in check(1, 10, 1000, 4000, 1000)
    # position past the head
    assert "past" in check(0, 1001, 1000, 4000, 1000)
    # M1: stale or missing sidecar, naming both numbers
    msg = check(0, 500, 1000, 4000, 1200)
    assert "1,000" in msg and "1,200" in msg and "4,000" in msg, msg
    msg = check(0, 500, 1000, 4000, None)
    assert "1,000" in msg and "4,000" in msg, msg
    assert check(0, 500, None, 4000, 1000) is not None


def test_plan_split():
    from types import SimpleNamespace
    import torch
    from scripts.sft import build_sft_scheduler, plan_changed, plan_total_steps
    # the main plan: 800B tokens at 524,288 per update
    rows_main = int(800e9 / 4096 / 3 * 1.0)
    steps_main = plan_total_steps(rows_main, 3, 8, 1, 16)
    steps_warm = plan_total_steps(int(rows_main / 9.5), 3, 8, 1, 16)
    cfg = SimpleNamespace(warmup_fraction=0.000655, lr_min_factor=0.1)

    def schedule(total):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1.0)
        sched = build_sft_scheduler(opt, total, cfg)
        out = {}
        for step in range(1, 1201):
            opt.step()
            sched.step()
            out[step] = opt.param_groups[0]["lr"]
        return out
    main_lr = schedule(steps_main)
    assert abs(steps_main * 0.000655 - 1000) < 80, steps_main
    assert 0.9 < main_lr[1000] <= 1.0 and 0.4 < main_lr[500] < 0.6, (main_lr[500], main_lr[1000])
    # fitted to the warm split instead (the defect), the warmup is about a tenth as long
    assert steps_warm * 0.000655 < 150
    # the same plan on the warm launch and the main launch: the switch moves nothing
    plan = {"total_steps": steps_main, "rows_per_epoch": rows_main, "world_size": 16,
            "grad_accumulation_steps": 1, "plan_split": "run2_main_train"}
    assert plan_changed(dict(plan), dict(plan)) == []
    assert plan_changed({"plan_split": None, "total_steps": 5}, {"plan_split": "m", "total_steps": 5}) == []
    assert plan_changed(plan, {**plan, "plan_split": None}) == [("plan_split", "run2_main_train", None)]
    assert plan_changed(plan, {**plan, "total_steps": steps_warm})[0][0] == "total_steps"


def test_cli_overrides():
    from types import SimpleNamespace
    from scripts.sft import cli_overrides
    ns = SimpleNamespace
    assert cli_overrides(ns(data_dir=None, train_split="", val_split=None, grad_accumulation_steps=None)) == {}
    assert cli_overrides(ns(data_dir="d", train_split="t", val_split=None,
                            grad_accumulation_steps=2)) == {"data_dir": "d", "train_split": "t",
                                                            "grad_accumulation_steps": 2}
    for bad in (0, -1):
        try:
            cli_overrides(ns(grad_accumulation_steps=bad))
        except SystemExit:
            continue
        raise AssertionError(f"accumulation {bad} accepted")


def test_destroy_process_group_guarded():
    import torch
    from scripts import sft as S
    calls = []
    real = (torch.distributed.is_initialized, torch.distributed.destroy_process_group)
    try:
        torch.distributed.is_initialized = lambda: True
        torch.distributed.destroy_process_group = lambda: calls.append(1)
        S.destroy_process_group_quietly()
        assert calls == [1]

        def boom():
            raise RuntimeError("nccl")
        torch.distributed.destroy_process_group = boom
        S.destroy_process_group_quietly()          # a failure never propagates
        # sft() destroys on every way out of a multi process run, including an exception
        os.environ["WORLD_SIZE"], os.environ["RANK"] = "2", "0"
        torch.distributed.destroy_process_group = lambda: calls.append(2)
        real_impl = S._sft_impl
        try:
            S._sft_impl = lambda args: 30
            assert S.sft(None) == 30 and calls[-1] == 2

            def raiser(args):
                raise SystemExit("refused")
            S._sft_impl = raiser
            try:
                S.sft(None)
            except SystemExit:
                pass
            assert calls.count(2) == 2
        finally:
            S._sft_impl = real_impl
            os.environ.pop("WORLD_SIZE"), os.environ.pop("RANK")
        # one process: no teardown
        S._sft_impl = lambda args: 0
        n = len(calls)
        assert S.sft(None) == 0 and len(calls) == n
        S._sft_impl = real_impl
    finally:
        torch.distributed.is_initialized, torch.distributed.destroy_process_group = real


def test_final_checkpoint_data():
    import torch
    from scripts.sft import final_checkpoint_data
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "final.pt")
        torch.save({"sft": {"data_dir": "x", "train_split": "warm", "val_split": "v", "seed": 1},
                    "w": torch.zeros(3)}, path)
        assert final_checkpoint_data(path) == {"data_dir": "x", "train_split": "warm", "val_split": "v"}
        torch.save({"w": torch.zeros(3)}, path)
        assert final_checkpoint_data(path) == {}
    finally:
        shutil.rmtree(d)


def reference_rows(idx_path, evidx_path, max_length, mtp, max_ev, order, workers, row_limit):
    """The packing replayed document by document in plain Python ints, no numpy tricks."""
    idx = np.fromfile(idx_path, dtype=np.uint64).tolist()
    ev = np.fromfile(evidx_path, dtype=np.uint64).tolist() if evidx_path else None
    n = len(idx) - 1
    seq = [int(d) for d in order] if order is not None else list(range(n))
    rows = 0
    for w in range(workers):
        used = used_ev = 0
        open_row = False
        for doc in seq[w::workers]:
            length = idx[doc + 1] - idx[doc]
            if length <= 0:
                continue
            length += mtp
            evl = (ev[doc + 1] - ev[doc]) if ev else 0
            if length > row_limit or (max_ev and evl > max_ev):
                continue
            if open_row and (used + length > row_limit or (max_ev and used_ev + evl > max_ev)):
                rows += 1
                used = used_ev = 0
            used, used_ev, open_row = used + length, used_ev + evl, True
        rows += int(open_row)
    return max(1, rows)


def write_evidx(directory, split, lengths):
    off = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint64)
    off.tofile(os.path.join(directory, f"{split}.evidx"))


def test_rows_estimate_identical():
    from scripts.sft import estimate_packed_rows
    d = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(3)
        n = 3000
        lens = rng.integers(0, 40, n)
        lens[::97] = 300                    # over the row limit, skipped
        offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.uint64)
        offs.tofile(os.path.join(d, "s.idx"))
        write_evidx(d, "s", rng.integers(0, 90, n))
        order = epoch_order(42, 0, n, 700)
        for workers in (1, 4, 32):
            for evidx, max_ev in (("s", 0), ("s", 120), (None, 0)):
                ev_path = os.path.join(d, "s.evidx") if evidx else None
                row_limit = 99 if ev_path else 100
                got = estimate_packed_rows(os.path.join(d, "s.idx"), 100, 2, evidx_path=ev_path,
                                           max_evidence_tokens=max_ev, order=order, num_workers=workers)
                want = reference_rows(os.path.join(d, "s.idx"), ev_path if max_ev else None, 100, 2,
                                      max_ev, order, workers, row_limit)
                assert got == want, (workers, evidx, max_ev, got, want)
        # the pretraining reader: tokens over the row length, never a loop
        got = estimate_packed_rows(os.path.join(d, "s.idx"), 100, 2, split_documents=True)
        assert got == max(1, int((lens + 2).sum() // 100) + 1)
    finally:
        shutil.rmtree(d)


def test_rows_cache():
    from scripts.sft import (read_rows_cache, resolve_rows_estimate, rows_cache_key,
                             rows_cache_path, write_rows_cache)
    d = tempfile.mkdtemp()
    try:
        write_split(d, "main", 40)
        write_order(d, "main", 12)
        ds = make(d, "main")
        key = rows_cache_key(ds, 4, 1)
        path = rows_cache_path(d, "main")
        assert path.endswith("main.rows_estimate.json")
        assert read_rows_cache(path, key, 4) is None            # no file
        write_rows_cache(path, key, 4, 111)
        write_rows_cache(path, key, 8, 222)                      # another rank count, same key
        assert read_rows_cache(path, key, 4) == 111 and read_rows_cache(path, key, 8) == 222
        assert read_rows_cache(path, key, 16) is None
        assert not os.path.exists(path + ".tmp")
        # invalidation: each input of the estimate moves the key
        assert rows_cache_key(ds, 4, 1) == key
        assert rows_cache_key(ds, 8, 1) != key and rows_cache_key(ds, 4, 2) != key
        write_order(d, "main", 13)
        assert rows_cache_key(make(d, "main"), 4, 1) != key      # head
        write_order(d, "main", 12)
        assert rows_cache_key(make(d, "main"), 4, 1) == key
        assert rows_cache_key(SFTDataset(d, Tok(), 1, 4, "main", 1, seed=7), 4, 1) != key   # seed
        ds_other = make(d, "main", shuffle=False)
        assert rows_cache_key(ds_other, 4, 1) != key             # reader order
        os.utime(os.path.join(d, "main.idx"), (1, 1))            # a copy keeps the bytes, not the mtime
        assert rows_cache_key(make(d, "main"), 4, 1) == key
        with open(os.path.join(d, "main.idx"), "r+b") as f:      # index rewritten
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([last[0] ^ 1]))
        assert rows_cache_key(make(d, "main"), 4, 1) != key
        assert read_rows_cache(path, rows_cache_key(make(d, "main"), 4, 1), 4) is None
        with open(path, "w") as f:
            f.write("not json")
        assert read_rows_cache(path, key, 4) is None
        # a new key replaces the old entries instead of mixing with them
        new_key = rows_cache_key(make(d, "main"), 4, 1)
        write_rows_cache(path, new_key, 4, 5)
        assert read_rows_cache(path, new_key, 4) == 5
        # resolve: a single process computes and writes, several ranks without a cache refuse
        os.remove(path)
        calls = []
        assert resolve_rows_estimate(path, new_key, 1, lambda w: calls.append(w) or 77) == (77, "computed")
        assert calls == [4] and read_rows_cache(path, new_key, 4) == 77
        assert resolve_rows_estimate(path, new_key, 1, lambda w: 1 / 0) == (77, "cache")
        assert resolve_rows_estimate(path, new_key, 2, lambda w: 1 / 0) == (None, "refused")
        write_rows_cache(path, new_key, 8, 9)
        assert resolve_rows_estimate(path, new_key, 2, lambda w: 1 / 0) == (9, "cache")
    finally:
        shutil.rmtree(d)


def test_estimate_rows_only_mode():
    from types import SimpleNamespace
    import scripts.sft as S
    d = tempfile.mkdtemp()
    real = S.Accelerator
    try:
        write_split(d, "main", 60)
        args = SimpleNamespace(
            repair=False, ir=False, evidence=True, run_name=None, data_dir=d, train_split="main",
            val_split=None, grad_accumulation_steps=None, ceiling_json=None, plan_split=None,
            estimate_rows_only=True, estimate_world_sizes="1,4", switch_split=False,
            max_epochs_this_split=0)

        def no_accelerator(*a, **k):
            raise AssertionError("the estimate only mode built an accelerator")
        S.Accelerator = no_accelerator
        assert S._sft_impl(args) == 0
        path = S.rows_cache_path(d, "main")
        assert os.path.isfile(path)
        saved = json.load(open(path))
        assert set(saved["rows"]) == {str(S.NUM_DATA_WORKERS), str(4 * S.NUM_DATA_WORKERS)}, saved
        before = open(path).read()
        assert S._sft_impl(args) == 0 and open(path).read() == before     # cached, untouched
        # the plan split is the one cached
        write_split(d, "warm", 10)
        args.train_split, args.plan_split = "warm", "main"
        os.remove(path)
        assert S._sft_impl(args) == 0 and os.path.isfile(path)
        assert not os.path.exists(S.rows_cache_path(d, "warm"))
    finally:
        S.Accelerator = real
        shutil.rmtree(d)


def main():
    test_rows_estimate_identical()
    test_rows_cache()
    test_estimate_rows_only_mode()
    test_order_function()
    test_order_memory_form_identical()
    test_switch_state()
    test_plan_split()
    test_cli_overrides()
    test_destroy_process_group_guarded()
    test_final_checkpoint_data()
    test_sidecar_and_validation()
    test_warm_prefix_and_switch()
    test_no_sidecar_identical()
    test_rank_streams_cover_once()
    test_resume_check()
    print("test_epoch0_head_order: all passed")


if __name__ == "__main__":
    main()

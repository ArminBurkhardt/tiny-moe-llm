"""Contracts the multi-node resume path relies on (CPU, run in WSL after env_init).

1. A corpus moved to another directory is a plain resume: ``check_split_switch`` does not refuse
   it without ``--switch-split`` and only a split name change counts as other data.
2. The rows cache key ignores mtimes (a copy or restage keeps it valid) and changes with the
   index bytes, the document count and the packing parameters.
3. ``all_reduce_grads`` leaves a gradient that is None on every rank as None.
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from scripts.sft import check_split_switch, rows_cache_key, split_changes


class FakeDataset:
    def __init__(self, idx_path, num_docs):
        self.idx_path = idx_path
        self.num_docs = num_docs
        self.seed = 1
        self.shuffle = True
        self.epoch0_head_docs = None


def test_moved_corpus():
    saved = {"data_dir": "/scratch/a/prepared_run2", "train_split": "run2_main_train",
             "val_split": "inject_val"}
    current = dict(saved, data_dir="/mnt/b/prepared_run2")
    assert [c[0] for c in split_changes(saved, current, "/repo")] == ["data_dir"]
    assert check_split_switch(saved, current, False, "/repo")[0][0] == "data_dir"
    warm = dict(saved, train_split="run2_warm_train")
    try:
        check_split_switch(warm, current, False, "/repo")
    except SystemExit:
        pass
    else:
        raise AssertionError("a split name change must still be refused")
    print("1. a moved corpus resumes without --switch-split, a split change is still refused   PASS")


def test_cache_key():
    tmp = tempfile.mkdtemp()
    try:
        a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        os.makedirs(a)
        os.makedirs(b)
        data = np.arange(500_000, dtype=np.uint64)
        data.tofile(os.path.join(a, "s.idx"))
        shutil.copyfile(os.path.join(a, "s.idx"), os.path.join(b, "s.idx"))
        os.utime(os.path.join(b, "s.idx"), ns=(time.time_ns() + 10**12,) * 2)
        key_a = rows_cache_key(FakeDataset(os.path.join(a, "s.idx"), 499_999), 4096, 2)
        key_b = rows_cache_key(FakeDataset(os.path.join(b, "s.idx"), 499_999), 4096, 2)
        assert key_a == key_b, "a copy with other mtimes must keep the cache valid"
        assert rows_cache_key(FakeDataset(os.path.join(b, "s.idx"), 499_998), 4096, 2) != key_a
        assert rows_cache_key(FakeDataset(os.path.join(b, "s.idx"), 499_999), 8192, 2) != key_a
        data[-1] += 1
        data.tofile(os.path.join(b, "s.idx"))
        assert rows_cache_key(FakeDataset(os.path.join(b, "s.idx"), 499_999), 4096, 2) != key_a
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("2. the rows cache key ignores mtimes and tracks bytes, documents and packing   PASS")


def test_none_grads_stay_none():
    from scripts.pretrain import all_reduce_grads
    p_used = torch.nn.Parameter(torch.zeros(4))
    p_unused = torch.nn.Parameter(torch.zeros(3))
    p_used.grad = torch.ones(4)
    calls = []

    def reduce(t):
        calls.append(t.numel())
        if len(calls) == 1:                       # flags: the other rank also has no grad for p_unused
            t.add_(torch.tensor([1.0, 0.0]))
        else:
            t.add_(torch.ones_like(t))

    all_reduce_grads([p_used, p_unused], reduce, 2)
    assert p_unused.grad is None and calls == [2, 4], calls
    assert torch.allclose(p_used.grad, torch.ones(4))
    print("3. a gradient absent on every rank stays None   PASS")


if __name__ == "__main__":
    test_moved_corpus()
    test_cache_key()
    test_none_grads_stay_none()

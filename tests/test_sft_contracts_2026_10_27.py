"""Contracts the multi-node resume path relies on (CPU, run in WSL after env_init).

1. A corpus moved to another directory is a plain resume: ``check_split_switch`` does not refuse
   it without ``--switch-split`` and only a split name change counts as other data.
2. The rows cache key ignores mtimes (a copy or restage keeps it valid) and changes with the
   index bytes, the document count and the packing parameters.
3. ``all_reduce_grads`` leaves a gradient that is None on every rank as None.
4. The tail switch: ``check_tail_switch`` lets ``--switch-from-start`` restart only the last epoch of the plan the
   checkpoint was planned on, and ``--end-at-plan`` and ``--switch-from-start`` are parsed and wired into the loop.
5. A data worker survives a SIGTERM sent to it from outside (a step wide scancel).
6. A string ``SystemExit`` exits 3, an int passes through, None is 0.
7. A changed row shape is reported, an unrecorded one is not; ``--expect-tokens-per-update`` is checked.
8. The ``--probe-peak`` batch has the layout the train step reads, with every term active.
"""
import os
import signal
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from scripts.sft import (
    EXIT_REFUSED, build_probe_batch, build_train_loader, check_split_switch, check_tail_switch,
    check_tokens_per_update, exit_code_for, rows_cache_key, run_shape_changes, split_changes,
)


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


def test_tail_switch():
    assert check_tail_switch(2, 3, "run2_main_train", "run2_main_train") is None
    assert "epoch 1 of 3" in check_tail_switch(1, 3, "run2_main_train", "run2_main_train")
    assert "epoch 0 of 3" in check_tail_switch(0, 3, "run2_main_train", "run2_main_train")
    assert "planned on None" in check_tail_switch(2, 3, None, "run2_main_train")
    assert "planned on 'run2_main_train'" in check_tail_switch(2, 3, "run2_main_train", None)
    assert "--plan-split 'other'" in check_tail_switch(2, 3, "run2_main_train", "other")
    assert check_tail_switch(0, 1, "m", "m") is None, "a one epoch profile switches in its only epoch"
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "sft.py"),
               encoding="utf-8").read()
    for frag in ('"--switch-from-start"', '"--end-at-plan"', "check_tail_switch(state[\"epoch\"]", "plan_reached = True",
                 "if stop_training or plan_reached:", "start_doc_idx, step_offset = 0, 0"):
        assert frag in src, frag
    # a split switch from the warm-up (epoch 0) is still the old rule: the new flag only lifts it
    from scripts.sft import check_switch_state
    assert "not 0" in check_switch_state(2, 5, 100, 100, None)
    print("4. the tail switch is refused outside the plan's last epoch and the loop ends at the plan   PASS")


class SlowStream(torch.utils.data.IterableDataset):
    """Yields the worker pid and an index, slowly, so a signal lands mid iteration."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        for i in range(self.n):
            time.sleep(0.05)
            yield {"pid": os.getpid(), "i": i, "worker": info.id}


def test_worker_survives_sigterm():
    loader = build_train_loader(SlowStream(30), 2)
    seen, signalled = [], False
    for batch in loader:
        seen.append((int(batch["worker"]), int(batch["i"])))
        if not signalled and len(seen) == 4:
            for pid in {int(batch["pid"])}:
                os.kill(pid, signal.SIGTERM)
            signalled = True
    assert signalled and len(seen) == 60, len(seen)
    loader = build_train_loader(SlowStream(8), 2)
    assert len(list(loader)) == 16, "normal shutdown after exhaustion"
    plain = torch.utils.data.DataLoader(SlowStream(30), batch_size=None, num_workers=2)
    died = False
    try:
        for n, batch in enumerate(plain):
            if n == 3:
                os.kill(int(batch["pid"]), signal.SIGTERM)
    except RuntimeError:
        died = True
    assert died, "control: an unguarded worker dies of the same signal"
    print("5. a data worker ignores SIGTERM from outside and iteration continues   PASS")


def test_exit_codes():
    assert EXIT_REFUSED == 3
    assert exit_code_for(None) == 0 and exit_code_for(10) == 10 and exit_code_for(2) == 2
    assert exit_code_for("refused: reason") == 3
    print("6. a string SystemExit exits 3, ints pass through   PASS")


def test_shape_and_update_checks():
    saved = {"max_evidence_tokens": 12288, "seq_length": 4096, "mtp_tokens": 2}
    assert run_shape_changes(saved, dict(saved)) == []
    assert run_shape_changes(saved, dict(saved, max_evidence_tokens=8192)) == [
        ("max_evidence_tokens", 12288, 8192)]
    old = {"max_evidence_tokens": None, "seq_length": None, "mtp_tokens": None}
    assert run_shape_changes(old, dict(saved)) == [], "an old checkpoint records nothing"
    assert check_tokens_per_update(None, 8, 4096, 8, 2) == 524288
    assert check_tokens_per_update(524288, 8, 4096, 8, 2) == 524288
    try:
        check_tokens_per_update(524288, 4, 4096, 8, 2)
    except SystemExit as e:
        assert "262,144" in str(e) and "batch 4" in str(e) and "accumulation 2" in str(e)
    else:
        raise AssertionError("a different global batch must be refused")
    print("7. row shape changes and the tokens per update are checked   PASS")


def test_probe_batch():
    b = build_probe_batch(2, 256, 400, 25, 5000, 1, conversations=4)
    assert b["input_ids"].shape == (2, 256) and (b["input_ids"] != 1).all()
    assert b["evidence_ids"].shape == (2, 400) and b["chunk_slot"].shape == (2, 16)
    assert b["evidence_chunk_ids"].max() == 15 and b["chunk_keys"].shape == (2, 16, 384)
    assert b["chunk_gold"].sum(dim=1).tolist() == [4, 4]
    assert set(b["evidence_doc_slot"][0].tolist()) == {0, 1, 2, 3}
    assert (b["labels"][:, 1:] != -100).any() and (b["labels"][:, 0] == -100).all()
    assert b["answerable_ids"].shape == (2, 256)
    c = build_probe_batch(2, 400, 40 * 12 * 5, 5, 5000, 1, conversations=40)
    assert c["chunk_slot"].shape == (2, 480) and set(c["chunk_slot"][0].tolist()) == set(range(40))
    assert c["chunk_gold"].sum(dim=1).tolist() == [40, 40]
    assert (c["document_ids"][0].unique().numel()) == 40
    assert ((c["labels"] != -100).sum(dim=1) > 0).all()
    print("8. the probe batch carries full evidence, gold chunks and supervised answers   PASS")


if __name__ == "__main__":
    test_moved_corpus()
    test_cache_key()
    test_none_grads_stay_none()
    test_tail_switch()
    test_worker_survives_sigterm()
    test_exit_codes()
    test_shape_and_update_checks()
    test_probe_batch()

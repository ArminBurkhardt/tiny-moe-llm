"""Int32 epoch order, tail integrity checks, evidence cap count at startup, masked_fill and the
fp32 balance shadow in routing.

Run: `python tests/test_data_integrity_2026_10_28.py` (CPU only; the routing test imports
modules.model.moe, so it needs the WSL environment).
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from modules.data.sft_dataset import SFTDataset, check_index_total, epoch_order


class Tok:
    bos_token_id = 500
    eos_token_id = 501
    pad_token_id = 501


def old_order(seed, epoch, n, head):
    # the previous int64 implementation, verbatim
    if head is None or epoch != 0:
        return np.random.default_rng([seed, epoch]).permutation(n)
    out = np.arange(n, dtype=np.int64)
    np.random.default_rng([seed, 0]).shuffle(out[:head])
    np.random.default_rng([seed, 0, 1]).shuffle(out[head:])
    return out


def test_order_int32_identical():
    for n in (1, 2, 7, 50, 1000, 4097, 100003):
        for seed in (0, 42, 7):
            for epoch in (0, 1, 3):
                for head in (None, 1, n // 2 or 1, n):
                    got = epoch_order(seed, epoch, n, head)
                    want = old_order(seed, epoch, n, head)
                    assert got.dtype == np.int32, got.dtype
                    assert np.array_equal(got.astype(np.int64), want), (n, seed, epoch, head)


def write_sft(directory, split, n_docs, truncate_tail=0):
    ids, idx = [], [0]
    for d in range(n_docs):
        ids += [500, 1000 + d, 7]
        idx.append(len(ids))
    arr = np.asarray(ids, dtype=np.uint16)
    mask = np.ones(len(ids), dtype=np.uint8)
    if truncate_tail:
        arr, mask = arr[:-truncate_tail], mask[:-truncate_tail]
    arr.tofile(os.path.join(directory, f"{split}.bin"))
    mask.tofile(os.path.join(directory, f"{split}.mask"))
    np.asarray(idx, dtype=np.uint64).tofile(os.path.join(directory, f"{split}.idx"))


def test_sft_tail_check():
    with tempfile.TemporaryDirectory() as d:
        write_sft(d, "ok", 10)
        SFTDataset(d, Tok(), split="ok", max_length=64)
        write_sft(d, "cut", 10, truncate_tail=2)
        try:
            SFTDataset(d, Tok(), split="cut", max_length=64)
        except ValueError as e:
            assert "cut.idx" in str(e) and "28" in str(e) and "30" in str(e), str(e)
        else:
            raise AssertionError("truncated bin and mask pair passed")


def test_check_index_total():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.idx")
        np.asarray([0, 3, 9], dtype=np.uint64).tofile(p)
        check_index_total(p, "x.bin", 9)
        try:
            check_index_total(p, "x.bin", 8)
        except ValueError as e:
            assert "x.bin" in str(e) and "9" in str(e) and "8" in str(e)
        else:
            raise AssertionError


def test_evidence_dataset_checks_and_count():
    from modules.data.evidence_dataset import EMBED_DIM, EvidenceDataset, _index_lengths

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "o.idx")
        offs = np.asarray([0, 3, 3, 10, 14], dtype=np.uint64)
        offs.tofile(p)
        lens = _index_lengths(p)
        assert lens.dtype == np.int32 and lens.tolist() == [3, 0, 7, 4], lens

        def build(split, ev_cut=0, key_cut=0):
            n_docs = 4
            ids, idx, ev, evidx, keyidx = [], [0], [], [0], [0]
            ev_lens = [4, 0, 9, 2]
            chunks = 0
            for k in range(n_docs):
                ids += [500, 1000 + k, 7]
                idx.append(len(ids))
                ev += [11] * ev_lens[k]
                evidx.append(len(ev))
                chunks += 1 if ev_lens[k] else 0
                keyidx.append(chunks)
            pj = lambda s: os.path.join(d, f"{split}.{s}")
            np.asarray(ids, dtype=np.uint16).tofile(pj("bin"))
            np.ones(len(ids), dtype=np.uint8).tofile(pj("mask"))
            np.asarray(idx, dtype=np.uint64).tofile(pj("idx"))
            evarr = np.asarray(ev, dtype=np.uint16)
            evarr[: len(evarr) - ev_cut].tofile(pj("ev"))
            np.zeros(len(evarr) - ev_cut, dtype=np.uint16).tofile(pj("evchunk"))
            np.asarray(evidx, dtype=np.uint64).tofile(pj("evidx"))
            keys = np.zeros((chunks - key_cut, EMBED_DIM), dtype=np.float16)
            keys.tofile(pj("evkey"))
            np.asarray(keyidx, dtype=np.uint64).tofile(pj("evkeyidx"))

        build("good")
        ds = EvidenceDataset(d, Tok(), split="good", max_length=64, max_evidence_tokens=5)
        assert ds.startup_skipped_evidence == 1, ds.startup_skipped_evidence
        build("shortev", ev_cut=2)
        try:
            EvidenceDataset(d, Tok(), split="shortev", max_length=64)
        except ValueError as e:
            assert "shortev.evidx" in str(e) and "shortev.ev" in str(e), str(e)
        else:
            raise AssertionError("truncated ev passed")
        build("shortkey", key_cut=1)
        try:
            EvidenceDataset(d, Tok(), split="shortkey", max_length=64)
        except ValueError as e:
            assert "shortkey.evkeyidx" in str(e), str(e)
        else:
            raise AssertionError("truncated evkey passed")


def test_masked_fill_identical():
    labels = torch.randint(0, 100, (3, 17))
    pad = torch.rand(3, 17) > 0.6
    a = labels.clone()
    a[pad] = -100
    b = labels.clone().masked_fill(pad, -100)
    assert torch.equal(a, b)


def test_route_uses_current_shadow():
    from modules.model.moe import LoopMixtureOfExperts

    n_exp, top_k = 4, 1
    bias = torch.zeros(2, n_exp, dtype=torch.bfloat16)
    shadow = torch.tensor([[0.0, 0.2515, 0.0, 0.0], [0.0] * 4])
    # bf16 rounds 0.2515 to 0.25 (step 2**-10 * 2 near 0.25 is 0.00195 above, 0.00098 below)
    bias[0, 1] = shadow[0, 1].to(torch.bfloat16)
    assert float(bias[0, 1]) != float(shadow[0, 1])

    class Stub:
        def __init__(self):
            self.balance_bias = bias
            self.top_k = top_k
            self.num_experts = n_exp
            self.balance_tracking = False
            self.training = False
            self._balance_bias_fp32 = None
            self.router = lambda h, temperature=1.0: h
            self.loop_bias = lambda i: 0.0
            self.expert_tracker = types.SimpleNamespace(update=lambda *a, **k: None)

    stub = Stub()
    # logits chosen so expert 0 beats expert 1 by a margin that sits between the bf16 bias
    # (0.25) and the fp32 bias (0.2515)
    probs0 = torch.tensor([0.40, 0.15, 0.20, 0.25])
    probs0 = probs0.log().view(1, 1, n_exp)
    # route() softmaxes; fix the gap in the post softmax domain instead
    p = torch.softmax(probs0, dim=-1)[0, 0]
    gap = float(p[0] - p[1])
    shadow[0, 1] = gap + 0.0005  # fp32 selects expert 1, the rounded buffer may not
    bias[0, 1] = shadow[0, 1].to(torch.bfloat16)
    stub.balance_bias = bias
    _, idx_buf, _ = LoopMixtureOfExperts.route(stub, probs0, temperature=1.0, loop_idx=0)
    stub._balance_bias_fp32 = shadow
    stub._balance_stamp = (bias._version, bias.data_ptr(), shadow.device)
    _, idx_shadow, _ = LoopMixtureOfExperts.route(stub, probs0, temperature=1.0, loop_idx=0)
    assert idx_shadow.item() == 1, idx_shadow
    # a stale stamp falls back to the buffer
    stub._balance_stamp = (bias._version + 1, bias.data_ptr(), shadow.device)
    _, idx_stale, _ = LoopMixtureOfExperts.route(stub, probs0, temperature=1.0, loop_idx=0)
    assert idx_stale.item() == idx_buf.item()
    # zero bias, with and without a shadow, picks the plain argmax
    stub.balance_bias = torch.zeros(2, n_exp, dtype=torch.bfloat16)
    stub._balance_bias_fp32 = torch.zeros(2, n_exp)
    stub._balance_stamp = (stub.balance_bias._version, stub.balance_bias.data_ptr(), torch.device("cpu"))
    _, idx_zero, _ = LoopMixtureOfExperts.route(stub, probs0, temperature=1.0, loop_idx=0)
    assert idx_zero.item() == 0


if __name__ == "__main__":
    test_order_int32_identical()
    test_check_index_total()
    test_sft_tail_check()
    test_evidence_dataset_checks_and_count()
    test_masked_fill_identical()
    test_route_uses_current_shadow()
    print("ok")

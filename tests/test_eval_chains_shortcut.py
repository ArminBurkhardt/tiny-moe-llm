"""The shortcut line in ``scripts/eval_chains.py`` and the once per batch evidence of the bridge decode.

1. ``chains.question_from_meta`` on a sidecar line carrying ``chunk_facts`` gives a question whose
   ``shortcut_baselines`` equal the build time numbers; a line without ``chunk_facts`` gives None
   and ``split_shortcuts`` maps that hop count to None (the old sidecar stays readable);
2. ``chains.z_vs_line`` and ``shortcut_stats``: the z against the line, n/a without baselines;
3. ``accuracy_table`` prints the shortcut columns only when a cell carries them, and keeps the old
   cell format otherwise; ``mark_subhop=False`` drops the ``*``;
4. ``generate_bridges`` builds the evidence once per batch and the tensors it hands the model at
   every step equal the ones a per step rebuild gives (the layout of one pad run and one real run
   per row does not change as the real runs grow).

GPU free; the packer and the attention helpers import through ``eval_abstention``.
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from modules.data import chains
from modules.model.attention import _segment_ids, cu_seqlens_from_doc_ids
from scripts import eval_chains as ec
from scripts.eval_abstention import _pack_evidence_batch
from scripts.prepare_evidence_data import EMBED_DIM


def _meta(q, with_facts=True):
    m = {"qid": "x", "question": q.question, "answer": q.answer, "answer_type": q.answer_type,
         "hops": q.hops, "composition": list(q.composition), "held_out": False,
         "candidates": q.candidates, "chunk_kind": q.chunk_kind, "entities": q.entities}
    if with_facts:
        m["chunk_facts"] = [list(f) for f in q.chunk_facts]
    return json.loads(json.dumps(m))


def test_meta_roundtrip():
    _, held = chains.split_compositions(42)
    rng = random.Random(5)
    qs = [chains.sample_question(rng, held, hops=2) for _ in range(60)]
    direct = chains.shortcut_baselines(qs)
    rebuilt = [chains.question_from_meta(_meta(q)) for q in qs]
    assert all(r is not None for r in rebuilt)
    again = chains.shortcut_baselines(rebuilt)
    assert direct == again, (direct, again)
    assert chains.question_from_meta(_meta(qs[0], with_facts=False)) is None

    class Split:
        meta = [_meta(q) for q in qs] + [_meta(chains.sample_question(rng, held, hops=1), with_facts=False)]

    out = ec.split_shortcuts(Split, list(range(len(Split.meta))))
    assert out[2] == direct and out[1] is None
    assert direct["penultimate_fed"] > direct["random"] + 0.02, direct
    print("1. sidecar line rebuilds the shortcut baselines; no chunk_facts gives None  PASS")


def test_z_and_stats():
    z = chains.z_vs_line(0.30, 0.25, 1000)
    assert abs(z - 0.05 / np.sqrt(0.25 * 0.75 / 1000)) < 1e-9
    assert np.isnan(chains.z_vs_line(0.3, 0.0, 10)) and np.isnan(chains.z_vs_line(0.3, 0.5, 0))
    base = {"penultimate_fed": 0.256, "random": 0.185}
    s = ec.shortcut_stats([True] * 30 + [False] * 70, base)
    assert s["shortcut"] == 0.256 and abs(s["z_vs_shortcut"] - chains.z_vs_line(0.3, 0.256, 100)) < 1e-12
    n = ec.shortcut_stats([True], None)
    assert n["shortcut"] is None and np.isnan(n["z_vs_shortcut"])
    print("2. z against the line, n/a without baselines  PASS")


def test_table():
    stats = {"acc": 0.178, "sigma": 0.012, "chance": 0.185, "z_vs_chance": -0.5, "n": 10}
    plain = ec.accuracy_table({0: {2: dict(stats)}, 2: {2: dict(stats)}})
    assert "|" in plain[1] and plain[1].count("|") == 2 and "*" in plain[1]
    with_line = dict(stats, shortcut=0.256, z_vs_shortcut=-1.9)
    lines = ec.accuracy_table({0: {2: dict(with_line)}, 2: {2: dict(with_line)}})
    assert "0.256 |   -1.9" in lines[1] and lines[1].count("|") == 4
    assert len(lines[0]) == len(lines[1]) == len(lines[2])
    missing = ec.accuracy_table({0: {2: dict(stats, shortcut=None)}})
    assert "n/a" in missing[1]
    bridge = ec.accuracy_table({0: {2: dict(with_line)}}, mark_subhop=False)
    assert "*" not in bridge[1]
    print("3. table: shortcut columns only when present, no star in bridge mode  PASS")


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i % 20)) for i in ids)


class _FakeModel:
    def __init__(self, vocab=20):
        self.vocab = vocab
        self.calls = []
        self.lm_head = lambda h: h

    def build_evidence(self, ids, chunk_ids, segments, num_segments, chunk_keys=None, chunk_segments=None):
        return dict(ids=ids, chunk_ids=chunk_ids, segments=segments, num_segments=num_segments,
                    chunk_keys=chunk_keys, chunk_segments=chunk_segments)

    def __call__(self, input_ids, cu_seqlens, max_seqlen, return_hidden, skip_mtp, evidence, n_loops,
                 reader_sites_kept):
        self.calls.append((evidence, cu_seqlens.clone(), input_ids.shape[1]))
        b, s = input_ids.shape
        hidden = torch.zeros(1, b, s, self.vocab)
        step = len(self.calls)
        hidden[0, :, :, (step * 3) % self.vocab] = 1.0
        return hidden


def _rows():
    keys = np.ones((2, EMBED_DIM), dtype=np.float32)
    return [{"ids": [11, 12, 13, 14, 15], "chunk_ids": [0, 0, 0, 1, 1], "keys": keys},
            {"ids": [21, 22, 23], "chunk_ids": [0, 0, 1], "keys": keys},
            {"ids": [31], "chunk_ids": [0], "keys": keys[:1]}]


def test_evidence_once_per_batch():
    items = [{"prompt": [1, 2, 3, 4, 5], "evidence": r} for r in _rows()]
    model = _FakeModel()
    steps = 6
    out = ec.generate_bridges(model, _FakeTokenizer(), items, depth=3, kept=None, pad_id=0, eos_id=99,
                              batch_size=3, device="cpu", max_seq_len=64, max_new_tokens=steps)
    assert len(model.calls) == steps and all(len(g) == steps for g in out)
    first = model.calls[0][0]
    for evidence, cu, width in model.calls:
        assert evidence is first, "the evidence batch was rebuilt"
        b = len(items)
        # the per step rebuild the old code did, from this step's own layout
        doc = torch.zeros((b, width), dtype=torch.long)
        for r, it in enumerate(items):
            p = len(it["prompt"]) + (width - 6)
            doc[r, width - p:] = 1
        cu_ref, _ = cu_seqlens_from_doc_ids(doc)
        assert torch.equal(cu_ref, cu)
        seg = _segment_ids(cu_ref, b, width, "cpu")
        ref = _pack_evidence_batch(model, seg[:, -1], seg[:, 0], [it["evidence"] for it in items],
                                   num_segments=int(cu_ref.numel() - 1), device="cpu")
        for key in ("ids", "chunk_ids", "segments", "chunk_keys", "chunk_segments"):
            a, c = evidence[key], ref[key]
            assert (a is None and c is None) or torch.equal(torch.as_tensor(a), torch.as_tensor(c)), key
        assert evidence["num_segments"] == ref["num_segments"]
    print("4. bridge decode builds the evidence once; every step's tensors equal a rebuild  PASS")


if __name__ == "__main__":
    test_meta_roundtrip()
    test_z_and_stats()
    test_table()
    test_evidence_once_per_batch()
    print("test_eval_chains_shortcut: ok")

"""The eval side evidence packer lays every row's tokens out sorted by segment id.

``evidence_cu_seqlens`` counts tokens per segment and assumes the flattened ``[B, S_ev]`` axis is
sorted by segment id. A right padded batch gives each row its real segment first and its dump
segment last, a left padded batch the other way round, and the packer has to place the row's own
tokens accordingly: before the dump width in the first case, after it in the second. Before
2026-10-24 the left padded layout put them first, so every left padded decode with evidence
attached read its buffer shifted by the dump width. GPU free: a fake model records what the packer
hands to ``build_evidence``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from modules.model.evidence import evidence_cu_seqlens
from scripts.eval_abstention import _pack_evidence_batch
from scripts.prepare_evidence_data import EMBED_DIM


class _FakeModel:
    def build_evidence(self, ids, chunk_ids, segments, num_segments, chunk_keys=None, chunk_segments=None):
        return dict(ids=ids, chunk_ids=chunk_ids, segments=segments, num_segments=num_segments,
                    chunk_keys=chunk_keys, chunk_segments=chunk_segments)


def _rows():
    # rows of unequal evidence width, one with nothing attached, one of width 1
    return [
        {"ids": [11, 12, 13, 14, 15], "chunk_ids": [0, 0, 0, 1, 1], "keys": np.ones((2, EMBED_DIM), dtype=np.float32)},
        {"ids": [21, 22, 23], "chunk_ids": [0, 0, 1], "keys": np.ones((2, EMBED_DIM), dtype=np.float32)},
        {"ids": [31, 32, 33, 34], "chunk_ids": [0, 1, 1, 1], "keys": np.ones((2, EMBED_DIM), dtype=np.float32)},
        None,
        {"ids": [51], "chunk_ids": [0], "keys": np.ones((1, EMBED_DIM), dtype=np.float32)},
    ]


def _check(real, dump, name):
    rows = _rows()
    ev = _pack_evidence_batch(_FakeModel(), torch.tensor(real), torch.tensor(dump), rows,
                              num_segments=10, device="cpu")
    seg = ev["segments"]
    flat = seg.reshape(-1)
    assert bool((flat[1:] >= flat[:-1]).all()), f"{name}: segments not sorted: {flat.tolist()}"
    cu, _ = evidence_cu_seqlens(seg, 10)
    for i, row in enumerate(rows):
        n = len(row["ids"]) if row else 0
        a, b = int(cu[real[i]]), int(cu[real[i] + 1])
        assert b - a == n, f"{name}: row {i} real segment holds {b - a} tokens, expected {n}"
        got = ev["ids"].reshape(-1)[a:b].tolist()
        assert got == (row["ids"] if row else []), f"{name}: row {i} real segment reads {got}"
        chunks = ev["chunk_ids"].reshape(-1)[a:b].tolist()
        assert chunks == (row["chunk_ids"] if row else []), f"{name}: row {i} chunk ids {chunks}"
        da, db = int(cu[dump[i]]), int(cu[dump[i] + 1])
        assert db - da == 5 - n, f"{name}: row {i} dump segment holds {db - da} tokens, expected {5 - n}"
        assert (ev["chunk_ids"].reshape(-1)[da:db] < 0).all(), f"{name}: dump tokens carry chunk ids"
    print(f"{name}: real segments read their own tokens, dump segments the padding  PASS")


# five rows: three of unequal width, one with nothing attached, one of width 1
# right padded: real segment first (0, 2, ...), dump last (1, 3, ...)
_check([0, 2, 4, 6, 8], [1, 3, 5, 7, 9], "1. right padded")
# left padded: dump segment first (0, 2, ...), real last (1, 3, ...)
_check([1, 3, 5, 7, 9], [0, 2, 4, 6, 8], "2. left padded")
print("all evidence pack layout checks passed")

"""Serving-time retrieval: top_k_chunks over a tiny fake store with known vectors. GPU-free."""
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.data.store import EMBED_DIM, load_store, top_k_chunks, write_store


def unit(i: int, j: int = None, w: float = 0.0) -> np.ndarray:
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    v[i] = 1.0
    if j is not None:
        v[j] = w
    return v / np.linalg.norm(v)


def main():
    # chunk i points along axis i, except chunk 4 which leans toward axis 1
    keys = np.stack([unit(0), unit(1), unit(2), unit(3), unit(4, 1, 0.5)])

    # 1. the exact axis wins, best first
    ids, scores = top_k_chunks(unit(2), keys, 3)
    assert ids[0] == 2 and abs(scores[0] - 1.0) < 1e-6, (ids, scores)
    assert len(ids) == 3 and list(scores) == sorted(scores, reverse=True)

    # 2. a query between axes 1 and 4 prefers the chunk that leans on both, and scale does not matter
    query = unit(1, 4, 0.9) * 7.0
    ids, _ = top_k_chunks(query, keys, 2)
    assert list(ids) == [4, 1], ids

    # 3. k above N is capped, ties go to the lower id
    ids, _ = top_k_chunks(unit(10), keys, 99)
    assert len(ids) == 5 and list(ids[:4]) == [0, 1, 2, 3], ids

    # 4. fp16 keys as stored on disk still retrieve the right chunk after a write and load
    with tempfile.TemporaryDirectory() as tmp:
        chunks = [{"chunk_id": i, "text": f"chunk {i}", "source": "t", "doc_id": i, "title": ""} for i in range(5)]
        write_store(tmp, chunks, keys, {"embedder": "fake"})
        store = load_store(tmp)
        assert store.keys.dtype == np.float16
        ids, _ = top_k_chunks(unit(3), store.keys, 1)
        assert list(ids) == [3], ids

    # 5. attach order: shuffle is fixed per question and keeps the set, best_first and best_last keep the rank
    from scripts import inference
    ids = [7, 3, 9, 1, 5, 2]
    assert inference.order_chunks(ids, "best_first", "q") == ids
    assert inference.order_chunks(ids, "best_last", "q") == ids[::-1]
    a1, a2 = inference.order_chunks(ids, "shuffle", "who?"), inference.order_chunks(ids, "shuffle", "who?")
    assert a1 == a2 and sorted(a1) == sorted(ids)
    assert any(inference.order_chunks(ids, "shuffle", f"question {i}") != ids for i in range(20))
    assert inference.DEFAULT_CHUNK_ORDER == "shuffle"
    try:
        inference.order_chunks(ids, "nope", "q"); raise SystemExit("accepted a bad order")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        chunks = [{"chunk_id": i, "text": f"chunk {i}", "source": "t", "doc_id": i, "title": ""} for i in range(5)]
        write_store(tmp, chunks, keys, {"embedder": "fake"})
        seen = {}
        original = inference.build_evidence_batch
        inference.build_evidence_batch = lambda model, tok, texts, device, keys=None: seen.setdefault("texts", texts)
        try:
            for order in inference.CHUNK_ORDERS:
                seen.clear()
                r = inference.StoreRetriever(tmp, 4, "cpu", chunk_order=order)
                r.embed_question = lambda q: unit(2)
                _, cids, report = r.evidence_for(None, None, "any question", show=True)
                best = r.retrieve("any question")
                assert seen["texts"] == [f"chunk {c}" for c in cids] and sorted(cids) == sorted(best), (order, cids, best)
                if order == "best_first":
                    assert cids == best
                if order == "best_last":
                    assert cids == best[::-1]
                assert order in report and "[1]" in report
            try:
                inference.StoreRetriever(tmp, 4, "cpu", chunk_order="bad")
                raise AssertionError("accepted a bad order")
            except SystemExit:
                pass
        finally:
            inference.build_evidence_batch = original

    print("test_inference_retrieval: ok")


if __name__ == "__main__":
    main()

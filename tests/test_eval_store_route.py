"""The edit read's retrieval route: who supplies the query that fills the buffer.

Four checks, all without a GPU, on tiny stores written to a temporary directory:

1. **The bge route returns the original store's query keys** for a qid subset, in the sampled
   order, as float32, looked up by qid in the original store's ``questions.jsonl`` order even when
   the edit store lists its questions in another order.
2. **Those queries rank the raw bge keys** the way the hand-made geometry says, against the
   original store (before) and the edit store (after).
3. **The model route returns the model queries unchanged** (the same object).
4. **Refusals**: a qid absent from the store, a store without query keys, an unknown route and a
   model route without queries each stop with SystemExit.
"""
import os, sys, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.data.store import EMBED_DIM, load_store, write_store
from scripts.eval_store import edit_queries, topk_ids


def unit(*pairs):
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    for dim, value in pairs:
        v[dim] = value
    return v / np.linalg.norm(v)


def chunks(n):
    return [{"chunk_id": i, "text": f"chunk {i}", "source": "nq", "doc_id": f"d{i}", "title": ""}
            for i in range(n)]


def question(qid):
    return {"qid": qid, "source": "nq", "question": f"question {qid}?", "answers": ["a"],
            "gold_chunk_ids": [0], "gold_groups": [[0]], "answer_type": "OTHER"}


def build(tmp):
    keys = np.stack([unit((i, 1.0)) for i in range(4)])
    query_keys = np.stack([unit((1, 1.0), (2, 0.5)), unit((3, 1.0), (0, 0.3)), unit((0, 1.0), (2, 0.6))])
    original_root = os.path.join(tmp, "original")
    write_store(original_root, chunks(4), keys, {"name": "original"},
                questions=[question("q0"), question("q1"), question("q2")], query_keys=query_keys)
    # the edit store reverses the chunk order and lists its questions in another order, without query keys
    edit_root = os.path.join(tmp, "edited")
    write_store(edit_root, chunks(4), keys[::-1].copy(), {"name": "edited"},
                questions=[question("q2"), question("q0"), question("q1")])
    return load_store(original_root), load_store(edit_root), query_keys


def test_bge_route(original, edited, query_keys):
    qids = ["q2", "q0"]
    queries = edit_queries("bge", original, qids)
    assert queries.dtype == np.float32 and queries.shape == (2, EMBED_DIM), (queries.dtype, queries.shape)
    expected = query_keys[[2, 0]].astype(np.float16).astype(np.float32)
    assert np.array_equal(queries, expected)

    before = topk_ids(queries, original.keys.astype(np.float32), 2)
    assert before.tolist() == [[0, 2], [1, 2]], before.tolist()
    after = topk_ids(queries, edited.keys.astype(np.float32), 2)
    assert after.tolist() == [[3, 1], [2, 1]], after.tolist()
    subset = topk_ids(queries[[1]], edited.keys.astype(np.float32), 1)
    assert subset.tolist() == [[2]], subset.tolist()


def test_model_route(original):
    model_queries = np.random.default_rng(0).standard_normal((3, EMBED_DIM)).astype(np.float32)
    out = edit_queries("model", original, ["q0", "q1", "q2"], model_queries)
    assert out is model_queries


def expect_exit(fn):
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError("expected SystemExit")


def test_refusals(original, edited):
    expect_exit(lambda: edit_queries("bge", original, ["q0", "missing"]))
    expect_exit(lambda: edit_queries("bge", edited, ["q0"]))
    expect_exit(lambda: edit_queries("dense", original, ["q0"]))
    expect_exit(lambda: edit_queries("model", original, ["q0"], None))


def main():
    tmp = tempfile.mkdtemp(prefix="evalstoreroute_")
    try:
        original, edited, query_keys = build(tmp)
        test_bge_route(original, edited, query_keys)
        print("1-2. bge route: query rows by qid in sampled order, neighbours before and after: ok")
        test_model_route(original)
        print("3. model route returns the model queries unchanged: ok")
        test_refusals(original, edited)
        print("4. refusals: ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("all eval_store route checks passed")


if __name__ == "__main__":
    main()

"""The retrieval store: chunking, gold labels, the on-disk format, recall arithmetic and edit stores.

Eight checks, all without a GPU (a seeded fake embedder stands in for bge):

1. **Chunking is sentence aligned and under the cap.** Every chunk re-tokenizes to at most the cap,
   a prefix is counted against it, chunks hold whole sentences in order, and a sentence longer
   than the cap is split on token boundaries into pieces that also fit.
2. **Gold labels use normalized aliases.** A chunk is gold when it contains the answer after
   case, punctuation and article normalization; a question with no such chunk is dropped; a
   two-paragraph question keeps one gold group per paragraph.
3. **Exact-text duplicates are one chunk**, owned by the first source that produced it.
4. **write_store / load_store round trip**: dtype float16, shapes, questions, query keys, meta,
   optional files absent, no temporary file left behind, a wrong ``chunk_id`` refused.
5. **Recall arithmetic** on a hand-made score matrix: ids, recall@k per source, both-gold, and the
   paired comparison counts.
6. **Edit classification and flip rates.**
7. **A delete store** removes the gold chunk, a same-document chunk that holds the answer and a
   cosine neighbour, keeps a same-document chunk without the answer, renumbers ids densely and
   writes consistent ``edits.jsonl`` lines.
8. **An edit store** rewrites the answer in the gold chunk and in the neighbours that hold it,
   leaves every other chunk and key untouched, and re-embeds only the edited chunks.

Needs the pruned tokenizer (``utils.TOKENIZER_DIR``); the edit checks need ``modules/data/entity_swap.py``.
"""
import os, sys, shutil, tempfile, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.data.store import EMBED_DIM, contains_answer, load_store, split_sentences, write_store
from utils import TOKENIZER_DIR


class FakeEmbedder:
    """Deterministic unit vectors from the text; ``encode_queries`` mimics the query path."""

    def encode(self, texts):
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "big")
            v = np.random.RandomState(seed).randn(EMBED_DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out

    def encode_queries(self, texts):
        return self.encode(["query: " + t for t in texts])


def n_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_chunking(tokenizer):
    from scripts.build_store import chunk_text

    sentences = [f"The record number {i} was filed in the harbour office by clerk {i * 7} on a quiet day."
                 for i in range(60)]
    text = " ".join(sentences)
    for cap in (48, 128):
        chunks = chunk_text(text, tokenizer, cap)
        assert len(chunks) > 1
        assert all(n_tokens(tokenizer, c) <= cap for c in chunks), [n_tokens(tokenizer, c) for c in chunks]
        assert " ".join(chunks) == text, "chunks are not whole sentences in order"
        assert all(c.endswith(".") for c in chunks)

    prefixed = chunk_text(text, tokenizer, 64, prefix="Harbour Office: ")
    assert all(c.startswith("Harbour Office: ") and n_tokens(tokenizer, c) <= 64 for c in prefixed)

    long_sentence = "Word " + "word " * 300 + "end."
    pieces = chunk_text("Short opener here. " + long_sentence + " Short closer here.", tokenizer, 64)
    assert all(n_tokens(tokenizer, c) <= 64 for c in pieces), [n_tokens(tokenizer, c) for c in pieces]
    assert pieces[0] == "Short opener here." and pieces[-1] == "Short closer here."
    assert len(pieces) >= 6
    assert chunk_text("", tokenizer, 64) == []
    assert split_sentences("Dr. smith went. He said \"Go.\" Then left!") == ["Dr. smith went.", 'He said "Go."', "Then left!"]
    print("1. chunks are sentence aligned and under the cap          PASS")


def make_block(source, docs, questions):
    return {"source": source, "docs": docs, "questions": questions}


def doc(key, text, title="", prefix=False):
    return {"key": key, "doc_id": key, "title": title, "text": text, "title_prefix": prefix}


def test_gold_and_dedup(tokenizer):
    from scripts.build_store import build_chunks

    eiffel = ("The tower opened in 1889. It stands in Paris. Gustave Eiffel's company built the Eiffel "
              "Tower for the world fair. Visitors climb it daily. " * 4)
    plain = "Nothing here concerns towers. It is about gardens. " * 12
    nq = make_block("nq", [doc("nq:1", eiffel), doc("nq:2", plain)], [
        {"qid": "nq:a", "question": "Who built the tower?", "answers": ["gustave eiffel's COMPANY"],
         "gold_docs": [["nq:1", "nq:2"]], "mode": "answer"},
        {"qid": "nq:b", "question": "Who painted it?", "answers": ["Monet"],
         "gold_docs": [["nq:2"]], "mode": "answer"},
    ])
    hotpot = make_block("hotpotqa", [
        doc("h:1", "Alpha is a river. It is long.", "Alpha", True),
        doc("h:2", "Beta is a town on Alpha. It is small.", "Beta", True),
        doc("h:3", plain, "Gamma", True),
    ], [{"qid": "hotpotqa:x", "question": "Which town is on the river?", "answers": ["Beta"],
         "gold_docs": [["h:1"], ["h:2"]], "mode": "support"}])
    dup = make_block("triviaqa", [doc("t:1", plain)], [
        {"qid": "triviaqa:p", "question": "Where?", "answers": ["gardens"], "gold_docs": [["t:1"]],
         "mode": "answer"}])
    chunks, questions, stats = build_chunks([nq, hotpot, dup], tokenizer, 64)

    by_qid = {q["qid"]: q for q in questions}
    assert "nq:b" not in by_qid, "a question with no gold chunk must be dropped"
    assert stats["per_source"]["nq"]["questions_dropped_no_gold"] == 1
    gold = by_qid["nq:a"]["gold_chunk_ids"]
    assert gold and all(contains_answer(chunks[i]["text"], "Gustave Eiffel's company") for i in gold)
    assert all(chunks[i]["doc_id"] == "nq:1" for i in gold), "gold must come from the question's own passages"
    assert any(not contains_answer(c["text"], "eiffel's company") for c in chunks), "labels must discriminate"

    groups = by_qid["hotpotqa:x"]["gold_groups"]
    assert len(groups) == 2 and groups[0] != groups[1]
    assert all(chunks[i]["text"].startswith("Alpha: ") for i in groups[0])
    assert all(chunks[i]["text"].startswith("Beta: ") for i in groups[1])
    assert sorted(by_qid["hotpotqa:x"]["gold_chunk_ids"]) == sorted(groups[0] + groups[1])

    texts = [c["text"] for c in chunks]
    assert len(texts) == len(set(texts)), "exact duplicates must be one chunk"
    plain_chunks = [c for c in chunks if c["text"].startswith("Nothing here")]
    assert plain_chunks and all(c["source"] == "nq" for c in plain_chunks), "first source owns a duplicate"
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert by_qid["triviaqa:p"]["gold_chunk_ids"], "a duplicate chunk still serves the later source's question"
    print("2. gold labels use normalized aliases, groups kept       PASS")
    print("3. exact duplicates are one chunk, first source owns it   PASS")


def test_roundtrip():
    tmp = tempfile.mkdtemp(prefix="store_")
    try:
        chunks = [{"chunk_id": i, "text": f"chunk {i}", "source": "nq", "doc_id": f"d{i}", "title": ""}
                  for i in range(5)]
        keys = FakeEmbedder().encode([c["text"] for c in chunks])
        questions = [{"qid": "q0", "source": "nq", "question": "why?", "answers": ["x"],
                      "gold_chunk_ids": [1], "gold_groups": [[1]], "answer_type": "COMMON"}]
        query_keys = FakeEmbedder().encode_queries(["why?"])
        root = os.path.join(tmp, "s")
        write_store(root, chunks, keys, {"n_chunks": 5}, questions=questions, query_keys=query_keys)
        store = load_store(root)
        assert store.keys.dtype == np.float16 and store.keys.shape == (5, EMBED_DIM)
        assert np.allclose(store.keys.astype(np.float32), keys, atol=2e-3)
        assert store.chunks == chunks and store.questions == questions and store.meta == {"n_chunks": 5}
        assert store.query_keys.shape == (1, EMBED_DIM) and store.query_keys.dtype == np.float16
        assert not [f for f in os.listdir(root) if ".tmp" in f], os.listdir(root)

        bare = os.path.join(tmp, "bare")
        write_store(bare, chunks, keys, {})
        loaded = load_store(bare)
        assert loaded.questions == [] and loaded.query_keys is None and loaded.edits == []

        bad = [dict(c) for c in chunks]
        bad[2]["chunk_id"] = 9
        try:
            write_store(os.path.join(tmp, "bad"), bad, keys, {})
        except ValueError:
            pass
        else:
            raise AssertionError("a chunk_id that is not the line number must be refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("4. store round trip, dtypes, optional files, atomic       PASS")


def test_recall_arithmetic():
    from scripts.eval_store import classify_edit, flip_rates, hit_vector, paired_counts, recall_summary, topk_ids

    index = np.array([[1, 0], [0, 1], [-1, 0], [0.6, 0.8], [0, -1]], dtype=np.float32)
    queries = np.array([[1, 0.1], [0.1, 1], [0.1, -1]], dtype=np.float32)
    top = topk_ids(queries, index, 3)
    assert top.tolist() == [[0, 3, 1], [1, 3, 0], [4, 0, 2]], top.tolist()
    questions = [
        {"source": "nq", "gold_chunk_ids": [3], "gold_groups": [[3]]},
        {"source": "nq", "gold_chunk_ids": [2], "gold_groups": [[2]]},
        {"source": "hotpotqa", "gold_chunk_ids": [4, 0], "gold_groups": [[4], [0]]},
    ]
    assert hit_vector(top, questions, 1).tolist() == [False, False, True]
    assert hit_vector(top, questions, 2).tolist() == [True, False, True]
    assert hit_vector(top, questions, 2, both=True).tolist() == [True, False, True]
    assert hit_vector(top[2:], questions[2:], 1, both=True).tolist() == [False]
    summary = recall_summary(top, questions, [1, 2, 3])
    assert summary["all"][2]["recall"] == 2 / 3 and summary["nq"][2]["recall"] == 0.5
    assert summary["nq"][1]["both"] is None and summary["hotpotqa"][1]["both"] == 0.0
    assert summary["hotpotqa"][2]["both"] == 1.0

    a = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    b = np.array([1, 0, 1, 0, 0, 0], dtype=bool)
    pc = paired_counts(a, b)
    assert (pc["a_only"], pc["b_only"], pc["n"]) == (2, 1, 6)
    assert abs(pc["diff"] - 1 / 6) < 1e-9
    assert abs(pc["sigma"] - np.sqrt(3 - 1 / 6) / 6) < 1e-9
    same = paired_counts(a, a)
    assert same["diff"] == 0.0 and same["z"] == 0.0
    print("5. recall@k arithmetic and the paired counts              PASS")

    em = lambda pred, refs: float(any(pred.strip().lower() == r.strip().lower() for r in refs))
    assert classify_edit("edit", "1900", "1900", ["1887"], em) == "follow"
    assert classify_edit("edit", "1887", "1900", ["1887"], em) == "stuck"
    assert classify_edit("edit", "1850", "1900", ["1887"], em) == "other"
    assert classify_edit("delete", "I don't know.", None, ["1887"], em) == "abstained"
    assert classify_edit("delete", "1887", None, ["1887"], em) == "stuck"
    rates = flip_rates([("edit", "follow")] * 7 + [("edit", "stuck")] * 3 + [("delete", "abstained")] * 4
                       + [("delete", "stuck")] * 6)
    assert rates["edit"]["flip_rate"] == 0.7 and rates["edit"]["pass"] is True
    assert rates["delete"]["flip_rate"] == 0.4 and rates["delete"]["pass"] is False
    print("6. edit classification and flip rates                     PASS")


def unit(rng):
    v = rng.randn(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_edit_source(tmp):
    """Question 0 has a gold chunk, a same-document chunk holding the answer, a same-document chunk
    without it and a cosine neighbour from another document holding it; 24 more questions supply the
    gazetteer and have orthogonal-ish random keys."""
    rng = np.random.RandomState(3)
    chunks, keys, questions = [], [], []

    def add(text, doc_id, title, key):
        chunks.append({"chunk_id": len(chunks), "text": text, "source": "nq", "doc_id": doc_id, "title": title})
        keys.append(key)
        return len(chunks) - 1

    g_key = unit(rng)
    near_key = g_key + 0.05 * unit(rng)
    near_key /= np.linalg.norm(near_key)
    c_gold = add("Orsted Works was founded in 1887 by two brothers.", "doc0", "Orsted Works", g_key)
    c_doc = add("Later history says Orsted Works opened its first plant in 1887 near the coast.",
                "doc0", "Orsted Works", unit(rng))
    c_other = add("The firm employs engineers across three continents.", "doc0", "Orsted Works", unit(rng))
    c_near = add("A trade almanac lists 1887 as the founding year of Orsted Works.", "doc9", "Almanac", near_key)
    questions.append({"qid": "q0", "source": "nq", "question": "In what year was Orsted Works founded?",
                      "answers": ["1887"], "gold_chunk_ids": [c_gold], "gold_groups": [[c_gold]],
                      "answer_type": "YEAR"})
    for i in range(1, 25):
        year = 1900 + i * 3
        c = add(f"Firm {i} Limited was founded in {year} in a small town.", f"doc{i * 10}", f"Firm {i}", unit(rng))
        questions.append({"qid": f"q{i}", "source": "nq", "question": f"In what year was Firm {i} Limited founded?",
                          "answers": [str(year)], "gold_chunk_ids": [c], "gold_groups": [[c]],
                          "answer_type": "YEAR"})
    keys = np.stack(keys)
    query_keys = np.stack([unit(rng) for _ in questions])
    root = os.path.join(tmp, "src")
    write_store(root, chunks, keys, {"embedder": "fake", "chunk_tokens": 128, "n_chunks": len(chunks),
                                     "sources": {"nq": len(chunks)}}, questions=questions,
                query_keys=query_keys)
    return load_store(root), (c_gold, c_doc, c_other, c_near)


def check_dense(result, store, removed):
    chunks = result["chunks"]
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert len(chunks) == len(store.chunks) - len(removed)
    assert result["keys"].shape == (len(chunks), EMBED_DIM)


def test_edit_stores(tmp):
    try:
        from modules.data.entity_swap import answer_type
    except ImportError:
        print("SKIP: modules/data/entity_swap.py is not there yet, edit store checks not run")
        return
    from scripts.build_store import build_edit

    store, (c_gold, c_doc, c_other, c_near) = make_edit_source(tmp)
    if answer_type(store.questions[0]["question"], store.questions[0]["answers"][0]) != "YEAR":
        print("SKIP: entity_swap does not type the synthetic year answers as YEAR, edit store checks not run")
        return
    fake = FakeEmbedder()

    deleted = build_edit(store, fake.encode, fraction=0.0, neighbour_cos=0.85, seed=5)
    removed = {c_gold, c_doc, c_near}
    edit_q0 = next(e for e in deleted["edits"] if e["qid"] == "q0")
    assert edit_q0["kind"] == "delete" and set(edit_q0["removed_chunk_ids"]) == removed, edit_q0
    assert edit_q0["edited_chunk_ids"] == [] and edit_q0["substitute"] is None
    all_removed = {i for e in deleted["edits"] for i in e["removed_chunk_ids"]}
    check_dense(deleted, store, all_removed)
    survivors = [c["text"] for c in deleted["chunks"]]
    assert store.chunks[c_other]["text"] in survivors, "a same-document chunk without the answer must stay"
    assert not any(store.chunks[i]["text"] in survivors for i in removed)
    q0_new = next(q for q in deleted["questions"] if q["qid"] == "q0")
    assert q0_new["gold_chunk_ids"] == [] and len(deleted["questions"]) == len(deleted["edits"])
    assert deleted["query_keys"].shape == (len(deleted["questions"]), EMBED_DIM)
    print("7. delete store: gold, same-doc answer holder and cosine neighbour removed, ids dense  PASS")

    edited = build_edit(store, fake.encode, fraction=1.0, neighbour_cos=0.85, seed=5)
    e0 = next(e for e in edited["edits"] if e["qid"] == "q0")
    assert e0["kind"] == "edit", f"q0 fell back to delete: {e0}"
    assert set(e0["original_chunk_ids"]) == {c_gold, c_doc, c_near}, e0
    assert e0["gold_edited_chunk_ids"] and len(e0["edited_chunk_ids"]) == 3
    assert e0["removed_chunk_ids"] == [] and e0["substitute"] and e0["substitute"] != "1887"
    new_by_old = dict(zip(e0["original_chunk_ids"], e0["edited_chunk_ids"]))
    for old in (c_gold, c_doc, c_near):
        text = edited["chunks"][new_by_old[old]]["text"]
        assert e0["substitute"] in text and "1887" not in text, text
        assert edited["chunks"][new_by_old[old]]["doc_id"] == store.chunks[old]["doc_id"]
        assert np.allclose(edited["keys"][new_by_old[old]], fake.encode([text])[0], atol=1e-6), "edited chunk not re-embedded"
    changed = {new for e in edited["edits"] for new in e["edited_chunk_ids"]}
    old_of = {new: old for e in edited["edits"] for old, new in zip(e["original_chunk_ids"], e["edited_chunk_ids"])}
    for chunk in edited["chunks"]:
        if chunk["chunk_id"] in changed:
            continue
        assert chunk["text"] == store.chunks[chunk["chunk_id"]]["text"], "an untouched chunk changed"
        assert np.array_equal(edited["keys"][chunk["chunk_id"]], store.keys[chunk["chunk_id"]].astype(np.float32))
    check_dense(edited, store, set())
    q0_edit = next(q for q in edited["questions"] if q["qid"] == "q0")
    assert q0_edit["gold_chunk_ids"] == [new_by_old[c_gold]]
    print("8. edit store: answer rewritten in gold and answer-holding neighbours, rest untouched  PASS")


def test_triviaqa_alias_order():
    import random
    import pandas as pd
    from scripts.build_store import triviaqa_block
    from modules.data.entity_swap import SWAPPABLE, answer_type

    frame = pd.DataFrame([{
        "question_id": "tc_1", "question": "Which city hosted the 1900 Olympic Games?",
        "answer": {"value": "Paris", "aliases": np.array(["Paris", "City of Light"]),
                   "normalized_aliases": np.array(["paris", "city of light"])},
        "entity_pages": {"title": np.array(["Paris"]), "wiki_context": np.array(["Paris hosted the games."])},
    }])
    block = triviaqa_block(frame, None, random.Random(0))
    answers = block["questions"][0]["answers"]
    assert answers == ["Paris", "City of Light"], answers
    assert all(a != a.lower() for a in answers), "a lowercase normalized alias came back in"
    assert answer_type(block["questions"][0]["question"], answers[0]) in SWAPPABLE
    print("7b. triviaqa answers lead with the value and drop lowercase duplicates        PASS")


def test_edit_every_alias(tmp):
    from modules.data.entity_swap import alias_remains
    from scripts.build_store import build_edit

    store, (c_gold, c_doc, c_other, c_near) = make_edit_source(tmp)
    store.questions[0]["answers"] = ["1887", "MDCCCLXXXVII"]
    store.chunks[c_other]["text"] = "The firm, dated MDCCCLXXXVII on its charter, employs engineers."
    fake = FakeEmbedder()
    edited = build_edit(store, fake.encode, fraction=1.0, neighbour_cos=0.85, seed=5)
    e0 = next(e for e in edited["edits"] if e["qid"] == "q0")
    assert e0["kind"] == "edit", e0
    assert c_other in e0["original_chunk_ids"], "a chunk holding only the second alias must be swapped too"
    for new in e0["edited_chunk_ids"]:
        assert not alias_remains(edited["chunks"][new]["text"], ["1887", "MDCCCLXXXVII"]),             edited["chunks"][new]["text"]
    assert "alias_remains" in edited["report"]
    print("8b. edit store swaps every alias, leftovers counted                           PASS")


def main():
    if not os.path.isdir(TOKENIZER_DIR):
        print(f"SKIP: no tokenizer at {TOKENIZER_DIR}")
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    test_chunking(tokenizer)
    test_gold_and_dedup(tokenizer)
    test_roundtrip()
    test_recall_arithmetic()
    tmp = tempfile.mkdtemp(prefix="editstore_")
    try:
        test_edit_stores(tmp)
        test_triviaqa_alias_order()
        test_edit_every_alias(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

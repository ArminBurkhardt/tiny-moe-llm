"""Build a retrieval store from open-domain QA sets, and derive an edit store from it.

A store (format in ``modules/data/store.py``) is the external memory the evidence port reads at
serving time: sentence-aligned chunks of the passages behind NQ, TriviaQA and HotpotQA questions,
a bge key per chunk, and the question set with its gold chunk ids. The three sets are held out of
every training corpus here (NQ-open test, TriviaQA and HotpotQA validation), so the store measures
retrieval and the pathway, not recall of training data.

Two modes:

  * ``--sources nq,triviaqa,hotpotqa``: chunk, deduplicate, label gold, embed, write.
    Gold for NQ and TriviaQA is every chunk of the question's own passages that contains a
    normalized answer string (a question with none is dropped). Gold for HotpotQA is the supporting
    paragraphs; all ten paragraphs of every question enter the store, so the store holds the
    distractors the dataset ships with.
  * ``--edit-from NAME``: derive an edit store. A seeded share of the questions becomes ``edit``
    (the answer string is replaced by another entity of the same type in the gold chunk and in every
    near neighbour that contains it) and the rest ``delete`` (the gold chunk and its near neighbours
    are removed). Neighbours are chunks within ``--neighbour-cos`` bge cosine of a gold chunk plus
    chunks of the same document or title that contain the answer. Two questions never touch the same
    chunk: a question whose affected chunks collide with an earlier one is left out. Chunk ids are
    renumbered densely; ``edits.jsonl`` records both id spaces.

``edits.jsonl`` lines: ``{"qid", "kind", "original", "substitute", "edited_chunk_ids" (new ids),
"original_chunk_ids" (old ids of those same chunks, aligned), "gold_edited_chunk_ids" (new ids, the
gold subset), "removed_chunk_ids" (old ids)}``.

The bge pass is the only GPU work (``--device cpu`` works, tens of minutes). Raw downloads go to
``data/benchmarks/``, never under ``data/prepared*``.

```bash
python scripts/build_store.py --name openqa --sources nq,triviaqa,hotpotqa --max-questions 3000 \\
  --chunk-tokens 128 --device cuda --seed 42
python scripts/build_store.py --edit-from openqa --name openqa_edit --edit-fraction 0.5 \\
  --neighbour-cos 0.85 --device cpu --seed 42
```
"""
import os
import sys
import json
import random
import hashlib
import argparse
import datetime
from typing import Dict, List, Optional, Sequence, Tuple

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from modules.data.store import (
    EMBEDDER, EMBED_DIM, Store, contains_answer, load_store, split_sentences, write_jsonl, write_store,
)
from scripts.prepare_evidence_data import ChunkEmbedder
from utils import BASE_DIR, TOKENIZER_DIR, get_hf_token, logger

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DEFAULT_INDEX_ROOT = os.path.join(BASE_DIR, "data", "index")
DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, "data", "benchmarks")

NQ_REPO = "florin-hf/nq_open_gold"
NQ_FILE = "test_dataset.json"
TRIVIA_REPO = "mandarjoshi/trivia_qa"
TRIVIA_PREFIX = "rc.wikipedia/validation"
HOTPOT_REPO = "hotpotqa/hotpot_qa"
HOTPOT_PREFIX = "distractor/validation"
SOURCE_NAMES = ("nq", "triviaqa", "hotpotqa")


class QueryAwareEmbedder(ChunkEmbedder):
    """``ChunkEmbedder`` plus the bge query instruction for short-query to passage retrieval.

    Passage keys come from ``encode`` exactly as the training corpus writes them (CLS, no
    instruction). Question keys come from ``encode_queries``, which prepends the instruction the
    model card prescribes for queries.
    """

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """``[len(texts), 384]`` float32 unit rows for questions.

        Args:
            texts: raw question strings.
        """
        return self.encode([QUERY_INSTRUCTION + t for t in texts])


# ------------------------------------------------------------------------------------ chunking


def chunk_text(text: str, tokenizer, chunk_tokens: int, prefix: str = "") -> List[str]:
    """Sentence-aligned greedy chunks of at most ``chunk_tokens`` model tokens.

    A sentence longer than the budget is split on token boundaries into pieces of their own. Each
    sentence is counted with its leading space, so the true token count of a joined chunk is never
    above the count used here.

    Args:
        text: raw passage.
        tokenizer: the repo tokenizer.
        chunk_tokens: hard cap per chunk, the prefix included.
        prefix: text put in front of every chunk (a HotpotQA title), counted against the cap.

    Returns:
        Chunk strings, each starting with ``prefix``. Empty when the text has no content.
    """
    prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
    budget = chunk_tokens - prefix_len - 1
    if budget < 8:
        raise ValueError(f"chunk_tokens {chunk_tokens} leaves no room after a {prefix_len} token prefix")
    sentences = split_sentences(text)
    if not sentences:
        return []
    lengths = [len(ids) for ids in
               tokenizer([" " + s for s in sentences], add_special_tokens=False)["input_ids"]]

    units: List[Tuple[str, int, bool]] = []     # (text, tokens, stands alone)
    for sentence, n in zip(sentences, lengths):
        if n <= budget:
            units.append((sentence, n, False))
            continue
        ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
        step = max(budget - 2, 1)
        for start in range(0, len(ids), step):
            piece = tokenizer.decode(ids[start:start + step]).strip()
            if piece:
                units.append((piece, min(step, len(ids) - start) + 1, True))

    chunks, current, current_n = [], [], 0

    def flush():
        nonlocal current, current_n
        if current:
            chunks.append(prefix + " ".join(current))
        current, current_n = [], 0

    for piece, n, alone in units:
        if alone:
            flush()
            chunks.append(prefix + piece)
            continue
        if current and current_n + n > budget:
            flush()
        current.append(piece)
        current_n += n
    flush()
    return chunks


# ---------------------------------------------------------------------------------- source loaders


def _as_list(value) -> List[str]:
    """pandas hands arrays back where lists are expected; empty strings are dropped."""
    if value is None:
        return []
    return [str(v) for v in value if str(v).strip()]


def _dedupe(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def load_parquet_files(repo: str, prefix: str, cache_dir: str, token: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """Download every parquet shard under ``prefix`` of a dataset repo and concatenate them.

    Args:
        repo: dataset repo id.
        prefix: path prefix of the split's shards.
        cache_dir: download root; the repo goes in ``<cache_dir>/<repo with / as __>``.
        token: Hub token or None.

    Returns:
        ``(frame, revision_sha)``.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    sha = api.dataset_info(repo).sha
    names = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=sha)
                   if f.startswith(prefix) and f.endswith(".parquet"))
    if not names:
        raise SystemExit(f"no parquet files under {prefix!r} in {repo} @ {sha}")
    local_dir = os.path.join(cache_dir, repo.replace("/", "__"))
    os.makedirs(local_dir, exist_ok=True)
    frames = [pd.read_parquet(hf_hub_download(repo_id=repo, filename=name, repo_type="dataset",
                                              local_dir=local_dir, token=token, revision=sha),
                              engine="pyarrow") for name in names]
    return pd.concat(frames, ignore_index=True), sha


def load_nq_rows(cache_dir: str, token: Optional[str]) -> Tuple[List[dict], str]:
    """The NQ-open test questions with their gold passage text.

    Args:
        cache_dir: download root.
        token: Hub token or None.

    Returns:
        ``(rows, revision_sha)``; rows carry ``example_id, question, answers, text,
        idx_gold_in_corpus``.
    """
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi(token=token).dataset_info(NQ_REPO).sha
    local_dir = os.path.join(cache_dir, NQ_REPO.replace("/", "__"))
    os.makedirs(local_dir, exist_ok=True)
    path = hf_hub_download(repo_id=NQ_REPO, filename=NQ_FILE, repo_type="dataset",
                           local_dir=local_dir, token=token, revision=sha)
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        columns = [k for k, v in raw.items() if isinstance(v, list)]
        if not columns:
            raise SystemExit(f"{NQ_FILE}: expected a list of rows, got keys {list(raw)}")
        raw = [dict(zip(columns, values)) for values in zip(*(raw[c] for c in columns))]
    needed = {"example_id", "question", "answers", "text", "idx_gold_in_corpus"}
    if raw and not needed <= set(raw[0]):
        raise SystemExit(f"{NQ_FILE}: columns are {sorted(raw[0])}, need {sorted(needed)}")
    return raw, sha


def nq_block(rows: List[dict], limit: Optional[int], rng: random.Random) -> dict:
    """Source block for NQ: one document per gold passage, answer-containing chunks are gold.

    Args:
        rows: rows from ``load_nq_rows``.
        limit: question cap after a seeded shuffle, or None.
        rng: seeded generator.
    """
    rows = list(rows)
    rng.shuffle(rows)
    docs, questions = {}, []
    for row in rows:
        if limit is not None and len(questions) >= limit:
            break
        answers = _dedupe(_as_list(row.get("answers")))
        text = str(row.get("text") or "").strip()
        question = str(row.get("question") or "").strip()
        if not answers or not text or not question:
            continue
        key = f"nq:{row['idx_gold_in_corpus']}"
        docs.setdefault(key, {"key": key, "doc_id": key, "title": "", "text": text, "title_prefix": False})
        questions.append({"qid": f"nq:{row['example_id']}", "question": question, "answers": answers,
                          "gold_docs": [[key]], "mode": "answer"})
    return {"source": "nq", "docs": list(docs.values()), "questions": questions}


def triviaqa_block(frame: pd.DataFrame, limit: Optional[int], rng: random.Random) -> dict:
    """Source block for TriviaQA: the question's entity pages, alias-containing chunks are gold.

    Args:
        frame: the ``rc.wikipedia`` validation frame.
        limit: question cap after a seeded shuffle, or None.
        rng: seeded generator.
    """
    rows = frame.to_dict("records")
    rng.shuffle(rows)
    docs, questions = {}, []
    for row in rows:
        if limit is not None and len(questions) >= limit:
            break
        answer = row.get("answer") or {}
        value = str(answer.get("value") or "")
        merged = _dedupe(([value] if value else []) + _as_list(answer.get("aliases"))
                         + _as_list(answer.get("normalized_aliases")))
        seen_lower: set = set()
        answers = [a for a in merged if not (a.lower() in seen_lower or seen_lower.add(a.lower()))]
        pages = row.get("entity_pages") or {}
        # not filtered: a dropped empty title would shift every context after it onto the wrong title
        titles = [str(t) for t in (pages.get("title") if pages.get("title") is not None else [])]
        contexts = list(pages.get("wiki_context") if pages.get("wiki_context") is not None else [])
        question = str(row.get("question") or "").strip()
        if not answers or not titles or not question:
            continue
        keys = []
        for title, context in zip(titles, contexts):
            text = str(context or "").strip()
            if not text:
                continue
            key = f"tqa:{title}"
            docs.setdefault(key, {"key": key, "doc_id": key, "title": title, "text": text,
                                  "title_prefix": False})
            keys.append(key)
        if keys:
            questions.append({"qid": f"triviaqa:{row['question_id']}", "question": question,
                              "answers": answers, "gold_docs": [keys], "mode": "answer"})
    return {"source": "triviaqa", "docs": list(docs.values()), "questions": questions}


def hotpotqa_block(frame: pd.DataFrame, limit: Optional[int], rng: random.Random) -> dict:
    """Source block for HotpotQA: every paragraph is a document, supporting paragraphs are gold.

    A paragraph is ``"Title: body"``, as the training corpus writes it, and splits only above the
    chunk cap. Each supporting paragraph is its own gold group, so a question can ask for both.

    Args:
        frame: the ``distractor`` validation frame.
        limit: question cap after a seeded shuffle, or None.
        rng: seeded generator.
    """
    rows = frame.to_dict("records")
    rng.shuffle(rows)
    docs, questions = {}, []
    for row in rows:
        if limit is not None and len(questions) >= limit:
            break
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        context, supporting = row.get("context"), row.get("supporting_facts")
        if not question or not answer or not isinstance(context, dict):
            continue
        support = {str(t).strip() for t in (supporting.get("title") if isinstance(supporting, dict) else [])
                   if t is not None}
        groups = []
        for title, sentences in zip(context["title"], context["sentences"]):
            body = "".join(str(s) for s in (sentences if sentences is not None else [])).strip()
            title = str(title).strip()
            if not body:
                continue
            key = f"hotpot:{row['id']}:{title}"
            docs[key] = {"key": key, "doc_id": f"hotpot:{title}", "title": title, "text": body,
                         "title_prefix": True}
            if title in support:
                groups.append([key])
        if groups:
            questions.append({"qid": f"hotpotqa:{row['id']}", "question": question, "answers": [answer],
                              "gold_docs": groups, "mode": "support"})
    return {"source": "hotpotqa", "docs": list(docs.values()), "questions": questions}


# ------------------------------------------------------------------------------------ assembly


def build_chunks(blocks: List[dict], tokenizer, chunk_tokens: int) -> Tuple[List[dict], List[dict], dict]:
    """Chunk every document, deduplicate by exact text across the store, and label gold.

    Args:
        blocks: source blocks (``source``, ``docs``, ``questions``) from the loaders above.
        tokenizer: the repo tokenizer.
        chunk_tokens: chunk cap.

    Returns:
        ``(chunks, questions, stats)``. Chunks carry the C5 fields, the source of a deduplicated
        chunk is the first block that produced it. Questions carry ``gold_chunk_ids`` and
        ``gold_groups``; a question with no gold chunk is dropped and counted in ``stats``.
    """
    chunks: List[dict] = []
    by_text: Dict[str, int] = {}
    doc_chunks: Dict[Tuple[str, str], List[int]] = {}
    stats = {"per_source": {}}

    for block in blocks:
        source = block["source"]
        for doc in block["docs"]:
            prefix = f"{doc['title']}: " if doc.get("title_prefix") and doc["title"] else ""
            ids = []
            for text in chunk_text(doc["text"], tokenizer, chunk_tokens, prefix):
                chunk_id = by_text.get(text)
                if chunk_id is None:
                    chunk_id = len(chunks)
                    by_text[text] = chunk_id
                    chunks.append({"chunk_id": chunk_id, "text": text, "source": source,
                                   "doc_id": doc["doc_id"], "title": doc["title"]})
                if chunk_id not in ids:
                    ids.append(chunk_id)
            doc_chunks[(source, doc["key"])] = ids

    questions = []
    for block in blocks:
        source = block["source"]
        kept = dropped = 0
        for q in block["questions"]:
            groups = []
            for keys in q["gold_docs"]:
                ids = [i for key in keys for i in doc_chunks.get((source, key), [])]
                ids = list(dict.fromkeys(ids))
                if q["mode"] == "answer":
                    ids = [i for i in ids if any(contains_answer(chunks[i]["text"], a) for a in q["answers"])]
                if ids:
                    groups.append(sorted(ids))
            if not groups:
                dropped += 1
                continue
            kept += 1
            questions.append({
                "qid": q["qid"], "source": source, "question": q["question"], "answers": q["answers"],
                "gold_chunk_ids": sorted({i for g in groups for i in g}), "gold_groups": groups,
                "answer_type": _answer_type(q["question"], q["answers"][0]),
            })
        stats["per_source"][source] = {"documents": len(block["docs"]), "questions_kept": kept,
                                       "questions_dropped_no_gold": dropped}
    stats["chunks"] = len(chunks)
    return chunks, questions, stats


def _answer_type(question: str, answer: str) -> str:
    try:
        from modules.data.entity_swap import answer_type
    except ImportError:
        return ""
    return answer_type(question, answer)


def embed_blocks(embed, texts: List[str], label: str, block: int = 8192) -> np.ndarray:
    """Embed in blocks so a long build logs progress. Returns float32 ``[len(texts), 384]``."""
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for start in range(0, len(texts), block):
        out[start:start + block] = embed(texts[start:start + block])
        logger.info(f"[{label}] embedded {min(start + block, len(texts)):,}/{len(texts):,}")
    return out


def build_openqa(args, tokenizer) -> None:
    """Build and write the question-answering store."""
    token = args.hf_token or get_hf_token()
    wanted = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SOURCE_NAMES]
    if unknown:
        raise SystemExit(f"unknown source(s) {unknown}; known: {SOURCE_NAMES}")
    revisions, blocks = {}, []
    for name in wanted:
        rng = random.Random(f"{args.seed}:{name}")
        if name == "nq":
            rows, revisions[NQ_REPO] = load_nq_rows(args.cache_dir, token)
            blocks.append(nq_block(rows, args.max_questions, rng))
        elif name == "triviaqa":
            frame, revisions[TRIVIA_REPO] = load_parquet_files(TRIVIA_REPO, TRIVIA_PREFIX, args.cache_dir, token)
            blocks.append(triviaqa_block(frame, args.max_questions, rng))
        else:
            frame, revisions[HOTPOT_REPO] = load_parquet_files(HOTPOT_REPO, HOTPOT_PREFIX, args.cache_dir, token)
            blocks.append(hotpotqa_block(frame, args.max_questions, rng))
        logger.info(f"[{name}] {len(blocks[-1]['questions']):,} questions, {len(blocks[-1]['docs']):,} documents")

    chunks, questions, stats = build_chunks(blocks, tokenizer, args.chunk_tokens)
    for source, row in stats["per_source"].items():
        logger.info(f"[{source}] kept {row['questions_kept']:,} questions, dropped "
                    f"{row['questions_dropped_no_gold']:,} with no gold chunk")
    logger.info(f"{len(chunks):,} chunks after exact-text deduplication")

    embedder = QueryAwareEmbedder(device=args.device, cache_size=4096)
    keys = embed_blocks(embedder.encode, [c["text"] for c in chunks], "chunks")
    query_keys = embed_blocks(embedder.encode_queries, [q["question"] for q in questions], "questions")

    sources = {}
    for chunk in chunks:
        sources[chunk["source"]] = sources.get(chunk["source"], 0) + 1
    meta = {"embedder": EMBEDDER, "chunk_tokens": args.chunk_tokens, "n_chunks": len(chunks),
            "sources": sources, "built": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset_revisions": revisions, "questions": {s: r["questions_kept"] for s, r in
                                                          stats["per_source"].items()},
            "seed": args.seed}
    root = os.path.join(args.index_root, args.name)
    write_store(root, chunks, keys, meta, questions=questions, query_keys=query_keys)
    logger.info(f"wrote {root}: {len(chunks):,} chunks, {len(questions):,} questions")


# ------------------------------------------------------------------------------------ edit store


def _question_rng(seed: int, qid: str) -> random.Random:
    return random.Random(int(hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()[:12], 16))


def neighbour_ids(keys32: np.ndarray, gold_ids: Sequence[int], threshold: float) -> List[int]:
    """Chunks whose key has cosine at least ``threshold`` to any gold chunk, the gold excluded.

    Args:
        keys32: float32 unit rows ``[N, 384]``.
        gold_ids: row indices of the gold chunks.
        threshold: cosine cut.
    """
    sims = keys32[list(gold_ids)] @ keys32.T
    near = np.nonzero((sims >= threshold).any(axis=0))[0]
    gold = set(gold_ids)
    return [int(i) for i in near if int(i) not in gold]


def neighbour_share(keys32: np.ndarray, gold_ids: Sequence[int], thresholds: Sequence[float],
                    block: int = 256) -> Dict[float, float]:
    """Share of the given gold chunks that have another chunk within each cosine threshold.

    Args:
        keys32: float32 unit rows.
        gold_ids: unique gold chunk ids.
        thresholds: cosine cuts to report.
        block: gold rows scored per matmul.
    """
    gold_ids = list(gold_ids)
    best = np.zeros(len(gold_ids), dtype=np.float32)
    for start in range(0, len(gold_ids), block):
        rows = gold_ids[start:start + block]
        sims = keys32[rows] @ keys32.T
        sims[np.arange(len(rows)), rows] = -1.0
        best[start:start + block] = sims.max(axis=1)
    return {t: float((best >= t).mean()) if len(best) else 0.0 for t in thresholds}


def build_edit(store: Store, embed, *, fraction: float, neighbour_cos: float, seed: int) -> dict:
    """Derive an edit store from a question store.

    Args:
        store: a loaded store with questions.
        embed: callable ``texts -> [n, 384]`` float32 unit rows, used to re-embed edited chunks.
        fraction: share of questions made ``edit`` (the rest, and every edit that cannot be
            performed, become ``delete``).
        neighbour_cos: bge cosine at or above which a chunk is a near neighbour of a gold chunk.
        seed: seeds the kind draw, the substitute draw and the claim order.

    Returns:
        ``{"chunks", "keys", "questions", "query_keys", "edits", "report"}`` ready for
        ``write_store`` / ``write_jsonl``; ids in ``chunks`` and ``questions`` are the new dense ids.
    """
    from modules.data.entity_swap import SWAPPABLE, Gazetteer, alias_remains, swap_all_aliases, swap_in_text

    chunks = store.chunks
    keys32 = store.keys.astype(np.float32)
    gazetteer = Gazetteer.from_pairs((q["question"], q["answers"][0]) for q in store.questions if q["answers"])

    by_doc: Dict[str, List[int]] = {}
    by_title: Dict[str, List[int]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk["chunk_id"])
        if chunk["title"]:
            by_title.setdefault(chunk["title"], []).append(chunk["chunk_id"])

    all_gold = sorted({i for q in store.questions for i in q["gold_chunk_ids"]})
    shares = neighbour_share(keys32, all_gold, (0.85, 0.90))
    logger.info(f"gold chunks with a neighbour at cosine 0.85: {shares[0.85]:.3f}, at 0.90: {shares[0.90]:.3f}")

    order = list(range(len(store.questions)))
    random.Random(seed).shuffle(order)
    claimed, plans = set(), []
    counts = {"edit": 0, "delete": 0, "edit_fell_back_to_delete": 0, "collisions": 0, "alias_remains": 0}
    for qi in order:
        q = store.questions[qi]
        gold = list(q["gold_chunk_ids"])
        rng = _question_rng(seed, q["qid"])
        near = set(neighbour_ids(keys32, gold, neighbour_cos))
        for g in gold:
            same = by_doc.get(chunks[g]["doc_id"], []) + by_title.get(chunks[g]["title"], [])
            near.update(i for i in same if i not in gold
                        and any(contains_answer(chunks[i]["text"], a) for a in q["answers"]))
        affected = set(gold) | near
        if affected & claimed:
            counts["collisions"] += 1
            continue

        kind = "edit" if rng.random() < fraction else "delete"
        edited: Dict[int, str] = {}
        original = q["answers"][0]
        substitute = None
        if kind == "edit":
            texts = [chunks[i]["text"] for i in sorted(affected)]
            surface = next((a for a in q["answers"]
                            if any(swap_in_text(chunks[g]["text"], a, "x")[1] for g in gold)), None)
            if q.get("answer_type") in SWAPPABLE and surface is not None:
                substitute = gazetteer.draw(q["answer_type"], q["answers"], rng, avoid_text=" ".join(texts))
            if substitute is not None:
                original = surface
                for i in sorted(affected):
                    new_text, n = swap_all_aliases(chunks[i]["text"], q["answers"], substitute)
                    if n:
                        edited[i] = new_text
                if any(alias_remains(t, q["answers"]) for t in edited.values()):
                    counts["alias_remains"] += 1
                    substitute, edited, original = None, {}, q["answers"][0]
            if substitute is None and kind == "edit":
                kind = "delete"
                counts["edit_fell_back_to_delete"] += 1
        claimed |= affected
        counts[kind] += 1
        plans.append({"qi": qi, "kind": kind, "original": original, "substitute": substitute,
                      "edited": edited, "removed": sorted(affected) if kind == "delete" else []})

    removed = {i for p in plans for i in p["removed"]}
    kept = [i for i in range(len(chunks)) if i not in removed]
    new_id = {old: new for new, old in enumerate(kept)}

    edited_all: Dict[int, str] = {}
    for p in plans:
        edited_all.update(p["edited"])
    new_chunks = []
    for old in kept:
        record = dict(chunks[old])
        record["chunk_id"] = new_id[old]
        if old in edited_all:
            record["text"] = edited_all[old]
        new_chunks.append(record)
    new_keys = keys32[kept].copy()
    edited_old = sorted(edited_all)
    if edited_old:
        new_keys[[new_id[i] for i in edited_old]] = embed_blocks(
            embed, [edited_all[i] for i in edited_old], "edited chunks")

    questions, query_rows, edits = [], [], []
    for p in sorted(plans, key=lambda p: p["qi"]):
        q = store.questions[p["qi"]]
        gold = set(q["gold_chunk_ids"])
        edited_ids = sorted(p["edited"])
        edits.append({
            "qid": q["qid"], "kind": p["kind"], "original": p["original"], "substitute": p["substitute"],
            "edited_chunk_ids": [new_id[i] for i in edited_ids],
            "original_chunk_ids": edited_ids,
            "gold_edited_chunk_ids": [new_id[i] for i in edited_ids if i in gold],
            "removed_chunk_ids": p["removed"],
        })
        mapped = dict(q)
        mapped["gold_chunk_ids"] = [new_id[i] for i in q["gold_chunk_ids"] if i in new_id]
        mapped["gold_groups"] = [[new_id[i] for i in g if i in new_id] for g in q.get("gold_groups", [])]
        questions.append(mapped)
        if store.query_keys is not None:
            query_rows.append(store.query_keys[p["qi"]])
    report = {**counts, "chunks_before": len(chunks), "chunks_after": len(new_chunks),
              "chunks_edited": len(edited_old), "chunks_removed": len(removed),
              "gold_share_with_neighbour_085": shares[0.85], "gold_share_with_neighbour_090": shares[0.90]}
    return {"chunks": new_chunks, "keys": new_keys, "questions": questions,
            "query_keys": np.stack(query_rows) if query_rows else None, "edits": edits, "report": report}


def build_edit_store(args) -> None:
    """Load the source store, derive the edit store and write it."""
    source = load_store(os.path.join(args.index_root, args.edit_from))
    if not source.questions:
        raise SystemExit(f"{args.edit_from} has no questions.jsonl")
    embedder = QueryAwareEmbedder(device=args.device, cache_size=4096)
    result = build_edit(source, embedder.encode, fraction=args.edit_fraction,
                        neighbour_cos=args.neighbour_cos, seed=args.seed)
    logger.info(f"edit store report: {json.dumps(result['report'])}")
    meta = dict(source.meta)
    sources = {}
    for chunk in result["chunks"]:
        sources[chunk["source"]] = sources.get(chunk["source"], 0) + 1
    meta.update({"n_chunks": len(result["chunks"]), "sources": sources,
                 "built": datetime.datetime.now().isoformat(timespec="seconds"),
                 "edit_of": args.edit_from, "edit_fraction": args.edit_fraction,
                 "neighbour_cos": args.neighbour_cos, "edit_seed": args.seed,
                 "edit_report": result["report"]})
    root = os.path.join(args.index_root, args.name)
    write_store(root, result["chunks"], result["keys"], meta, questions=result["questions"],
                query_keys=result["query_keys"])
    write_jsonl(os.path.join(root, "edits.jsonl"), result["edits"])
    logger.info(f"wrote {root}: {len(result['chunks']):,} chunks, {len(result['edits']):,} edited questions")


def main():
    parser = argparse.ArgumentParser(description="build a retrieval store or derive an edit store from one")
    parser.add_argument("--name", required=True, help="output store directory name under --index-root")
    parser.add_argument("--sources", default="nq,triviaqa,hotpotqa")
    parser.add_argument("--max-questions", type=int, default=3000, help="questions per source, seeded")
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--edit-from", default=None, help="derive an edit store from this store instead")
    parser.add_argument("--edit-fraction", type=float, default=0.5)
    parser.add_argument("--neighbour-cos", type=float, default=0.85)
    parser.add_argument("--index-root", default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = os.path.join(args.index_root, args.name)
    if os.path.exists(os.path.join(out, "meta.json")) and not args.overwrite:
        raise SystemExit(f"{out} already holds a store; pass --overwrite to replace it")
    if args.edit_from:
        build_edit_store(args)
        return
    build_openqa(args, AutoTokenizer.from_pretrained(args.tokenizer))


if __name__ == "__main__":
    main()

"""The retrieval store: a directory of text chunks, their bge keys and optional question sets.

A store lives in ``data/index/<name>/`` and is read the same way by every eval and by the corpus
builders that write one. Pure numpy and json: nothing here touches torch, the tokenizer or the
model, so it loads on any machine and from the GPU-free tests.

Files:

  * ``chunks.jsonl``: one object per line, ``{"chunk_id", "text", "source", "doc_id", "title"}``;
    ``chunk_id`` equals the line number.
  * ``keys.npy``: float16 ``[N, 384]``, row ``i`` the unit-length bge-small key of line ``i``.
  * ``meta.json``: embedder, chunk size, counts per source, build time, dataset revisions.
  * ``questions.jsonl`` (optional): ``{"qid", "source", "question", "answers", "gold_chunk_ids",
    "answer_type"}`` plus ``gold_groups`` (the gold ids split per supporting paragraph, so a
    two-hop question can ask for both).
  * ``query_keys_bge.npy`` (optional): float16 ``[Q, 384]``, bge query embedding per question line.
  * ``edits.jsonl`` (edit stores only): see ``scripts/build_store.py``.

Every write goes through a temporary file and ``os.replace``, so a reader never sees a half
written file.
"""
import json
import os
import re
import string
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

EMBED_DIM = 384
EMBEDDER = "BAAI/bge-small-en-v1.5"

_PUNCTUATION = set(string.punctuation)
_ARTICLES = {"a", "an", "the"}


@dataclass
class Store:
    """A loaded store.

    Args:
        root: directory the store was read from.
        chunks: chunk records, list index equals ``chunk_id``.
        keys: float16 ``[N, 384]`` bge keys.
        questions: question records, empty when the store has none.
        meta: the parsed ``meta.json``.
        query_keys: float16 ``[Q, 384]`` bge query keys aligned to ``questions``, or None.
        edits: parsed ``edits.jsonl``, empty for a plain store.
    """
    root: str
    chunks: List[dict]
    keys: np.ndarray
    questions: List[dict]
    meta: dict
    query_keys: Optional[np.ndarray] = None
    edits: List[dict] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace.

    The same rule as the answer normalization the QA evals use, restated here because nothing under
    ``modules/`` may import a script. Gold labeling and the evals must agree on it.
    """
    text = "".join(ch for ch in text.lower() if ch not in _PUNCTUATION)
    return " ".join(t for t in text.split() if t not in _ARTICLES)


def contains_answer(chunk_text: str, answer: str) -> bool:
    """Whether the normalized answer occurs in the normalized chunk on word boundaries.

    Args:
        chunk_text: raw chunk text.
        answer: raw answer string.

    Returns:
        False for an answer that normalizes to nothing.
    """
    needle = normalize_text(answer)
    if not needle:
        return False
    return f" {needle} " in f" {normalize_text(chunk_text)} "


def _atomic_write(path: str, write) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as handle:
            write(handle)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    """Write one json object per line, atomically.

    Args:
        path: target file.
        records: dicts to serialize.
    """
    def write(handle):
        for record in records:
            handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    _atomic_write(path, write)


def read_jsonl(path: str) -> List[dict]:
    """Read a json-lines file; blank lines are skipped.

    Args:
        path: file to read.
    """
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_npy(path: str, array: np.ndarray) -> None:
    _atomic_write(path, lambda handle: np.save(handle, array))


def write_store(root: str, chunks: List[dict], keys: np.ndarray, meta: dict,
                questions: Optional[List[dict]] = None,
                query_keys: Optional[np.ndarray] = None) -> None:
    """Write a store directory, each file atomically.

    Args:
        root: store directory, created when missing.
        chunks: chunk records; ``chunk_id`` must equal the list index.
        keys: ``[N, 384]`` keys, any float dtype, written as float16.
        meta: free-form dict, written as ``meta.json``.
        questions: optional question records.
        query_keys: optional ``[Q, 384]`` query keys, one per question.

    Raises:
        ValueError: on a wrong key width, a row count that does not match, or a ``chunk_id`` that is
            not the line number.
    """
    keys = np.asarray(keys)
    if keys.ndim != 2 or keys.shape[1] != EMBED_DIM:
        raise ValueError(f"keys must be [N, {EMBED_DIM}], got {keys.shape}")
    if keys.shape[0] != len(chunks):
        raise ValueError(f"{keys.shape[0]} keys for {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        if int(chunk["chunk_id"]) != i:
            raise ValueError(f"chunk at line {i} has chunk_id {chunk['chunk_id']}")
    if query_keys is not None:
        query_keys = np.asarray(query_keys)
        if questions is None or query_keys.shape != (len(questions), EMBED_DIM):
            raise ValueError("query_keys need one row per question")

    os.makedirs(root, exist_ok=True)
    write_jsonl(os.path.join(root, "chunks.jsonl"), chunks)
    _write_npy(os.path.join(root, "keys.npy"), keys.astype(np.float16))
    if questions is not None:
        write_jsonl(os.path.join(root, "questions.jsonl"), questions)
    if query_keys is not None:
        _write_npy(os.path.join(root, "query_keys_bge.npy"), query_keys.astype(np.float16))
    # meta last: a directory with meta.json is a finished store
    _atomic_write(os.path.join(root, "meta.json"),
                  lambda handle: handle.write(json.dumps(meta, indent=2).encode("utf-8")))


def load_store(root: str) -> Store:
    """Read a store directory.

    Args:
        root: store directory.

    Returns:
        The ``Store``. Optional files that are absent come back empty or None.

    Raises:
        FileNotFoundError: when ``chunks.jsonl``, ``keys.npy`` or ``meta.json`` is missing.
        ValueError: when the key matrix and the chunk list disagree.
    """
    chunks = read_jsonl(os.path.join(root, "chunks.jsonl"))
    keys = np.load(os.path.join(root, "keys.npy"))
    with open(os.path.join(root, "meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if keys.shape != (len(chunks), EMBED_DIM):
        raise ValueError(f"{root}: keys {keys.shape} do not match {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        if int(chunk["chunk_id"]) != i:
            raise ValueError(f"{root}: chunk at line {i} has chunk_id {chunk['chunk_id']}")

    def optional(name):
        path = os.path.join(root, name)
        return path if os.path.exists(path) else None

    questions_path = optional("questions.jsonl")
    questions = read_jsonl(questions_path) if questions_path else []
    query_path = optional("query_keys_bge.npy")
    query_keys = np.load(query_path) if query_path else None
    if query_keys is not None and query_keys.shape[0] != len(questions):
        raise ValueError(f"{root}: {query_keys.shape[0]} query keys for {len(questions)} questions")
    edits_path = optional("edits.jsonl")
    edits = read_jsonl(edits_path) if edits_path else []
    return Store(root=root, chunks=chunks, keys=keys, questions=questions, meta=meta,
                 query_keys=query_keys, edits=edits)


_SENTENCE_BREAK = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z0-9\"'(\[])|\n+")


def split_sentences(text: str) -> List[str]:
    """Split on sentence-final punctuation followed by a capital, digit or opening quote, and on newlines.

    Args:
        text: raw passage.

    Returns:
        Non-empty stripped sentences. A heuristic, not a parser: an abbreviation can split early,
        which only makes a chunk shorter.
    """
    return [s.strip() for s in _SENTENCE_BREAK.split(text) if s and s.strip()]

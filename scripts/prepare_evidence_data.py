"""Build the evidence-conditioned corpus: four conditions plus replay.

The trick this corpus exists to exploit is that **relevance can be known before any retriever
exists**. For QA the gold passage ships with the dataset; for web text a held-out span of the same
document is relevant by construction. So the port can be trained against an oracle retriever now,
and Phase 5 only has to replace the oracle with a real index later.

What makes it four conditions rather than one is a measurement, not thoroughness. The in-context
ceiling probe (``docs/measurements/evidence_ceiling.md``) found that handing this model a passage
from *another question* costs **0.628 nats more than handing it no passage at all** -- it reads
whatever it is given, uncritically, and is slightly more confident while doing it. A port trained
only on gold evidence would therefore make the model strictly worse the moment a real retriever
handed it something irrelevant, which is what a real retriever mostly does.

| condition | evidence | target |
|---|---|---|
| ``gold`` | the relevant chunk(s) | the real answer |
| ``mixed`` | gold shuffled among distractors | the real answer -- select, then read |
| ``distractors`` | distractors only | abstain: the answer is not in the buffer |
| ``none`` | nothing at all | abstain: grounded in retrieval, not memorized |

A natively unanswerable SQuAD v2 row is a fifth case that falls out free and is the hardest one:
its passage is gold-*shaped* (right topic, retrieved for the right reason) and still does not
answer, so its target is an abstention under **every** condition including ``gold``. Nothing else
in this corpus teaches that "I retrieved something relevant" and "I can answer" are different
claims.

Web text conditions are different on purpose. There is no abstention target for a language modeling
row, so ``distractors`` and ``none`` keep the same continuation as ``gold`` does; what they teach is
that unusable evidence must not *cost* anything, which is the direct fix for the 0.628 nats above.

**The passage leaves the prompt.** In every other corpus in this repo SQuAD's passage is part of the
user turn. Here the user turn is instruction plus question only, and the passage arrives through the
evidence port. That is the whole point -- a corpus that left the passage in the prompt would train
in-context attention, which already works, and leave the port untrained.

### On-disk format

Five files beyond the usual ``{split}.{bin,idx,mask}`` triple, all indexed by the same document
number so a row and its evidence cannot drift apart:

    {split}.ev        uint16   evidence token stream
    {split}.evidx     uint64   per document offsets into .ev (doc_count + 1 entries)
    {split}.evchunk   uint16   per evidence token, its chunk index WITHIN that document
    {split}.evkey     float16  [total chunks, 384] external embedder vectors, flat
    {split}.evkeyidx  uint64   per document offsets into the chunk axis (doc_count + 1 entries)

A document with no evidence writes nothing and leaves both offsets equal to the previous entry.
That is a genuinely empty evidence segment at train time, which flash answers with exact zeros --
so "no evidence" needs no sentinel token and no placeholder chunk, and reads as the absence it is.

Distractor chunks are written per occurrence rather than referenced, so a chunk reused as a
distractor by five documents costs five copies of its 384 floats. Deduplicating would need a global
chunk table and a second level of indirection; at ~0.8 KB a copy the duplication is cheaper than the
machinery, and it keeps a document's evidence contiguous on disk.

### Token accounting

``--target-tokens`` counts **prompt** tokens -- the ``.bin`` stream, which is what the trainer's LR
schedule is anchored to and what the model runs a full forward over. Evidence tokens are counted and
reported separately: they cost an embedding lookup plus the reader's cross attention, not a forward
pass, so folding them into one number would misstate both the run length and the cost.

Usage:

    python scripts/prepare_evidence_data.py                       # 150M prompt tokens
    python scripts/prepare_evidence_data.py --target-tokens 50000000 --no-webtext
"""
import os
import sys
import json
import time
import random
import hashlib
import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download

from modules.data import abstention
from modules.data.chat import ChatTemplate
from scripts.prepare_data import load_state, save_state_atomic
from scripts.prepare_sft_data import SQUAD_INSTRUCTION, is_unanswerable_squad
from utils import BASE_DIR, TOKENIZER_DIR, get_hf_token, logger

EMBED_DIM = 384          # bge-small-en-v1.5, which is also ir_dim -- see the adapters in moe.py
CONDITIONS = ("gold", "mixed", "distractors", "none")


@dataclass
class EvidenceRow:
    """One source row, decomposed into the parts a condition is applied to.

    Splitting it this way is what lets one row serve several conditions: the question and the gold
    answer are fixed, and only which chunks accompany them changes.

    Attributes:
        question: the user turn body, WITHOUT any passage.
        answer: the reference answer, or "" for a natively unanswerable row (which forces an
            abstention under every condition).
        gold: chunk texts that genuinely support the answer.
        near: row-local distractors -- HotpotQA ships eight per question, retrieved for the same
            query and deliberately confusable. Strictly better than a random other row's passage,
            because they are what a real retriever's tail actually looks like.
        lm: True for a language modeling row, where there is no abstention target and the
            unanswerable conditions keep the same continuation.
        prompt_ids: pre-tokenized (ids, mask) for an ``lm`` row, which bypasses the chat template.
    """

    question: str = ""
    answer: str = ""
    gold: List[str] = field(default_factory=list)
    near: List[str] = field(default_factory=list)
    lm: bool = False
    prompt_ids: Optional[Tuple[List[int], List[int]]] = None


@dataclass
class EvidenceSource:
    key: str
    weight: float
    repo_id: str = ""
    file_prefix: str = ""
    file_suffix: tuple = (".parquet",)
    render: str = ""
    # conditions this source draws from, and their relative frequency. QA sources carry all four;
    # replay carries none at all and is the >=20% of the corpus that protects the trunk.
    condition_weights: dict = field(default_factory=dict)
    local_bin: str = ""              # web text reads the prepared corpus instead of the Hub
    qa: bool = False


# Condition mix. ``mixed`` is the largest share because it is the only condition that matches what a
# real retriever delivers -- the others are its endpoints. ``distractors`` and ``none`` together take
# a third, which is what has to carry the abstention policy: Phase 2 established that no data ratio
# of in-prompt refusals moves discrimination, so these two are the replacement lever and a token
# share that made them a rounding error would be testing nothing.
QA_CONDITIONS = {"gold": 0.25, "mixed": 0.40, "distractors": 0.20, "none": 0.15}
# web text has no abstention target, so ``none`` is pure replay-with-a-question-shape and earns
# less; the distractor share is higher because "irrelevant evidence must not cost anything" is the
# lesson this source exists to teach at scale.
LM_CONDITIONS = {"gold": 0.40, "mixed": 0.30, "distractors": 0.20, "none": 0.10}

SOURCES = [
    EvidenceSource("squad_v2", 0.20, "rajpurkar/squad_v2", "squad_v2/train",
                   render="squad_v2", condition_weights=QA_CONDITIONS, qa=True),
    EvidenceSource("hotpot_qa", 0.15, "hotpotqa/hotpot_qa", "distractor/train",
                   render="hotpot_qa", condition_weights=QA_CONDITIONS, qa=True),
    EvidenceSource("webtext", 0.40, render="webtext", condition_weights=LM_CONDITIONS,
                   local_bin="ir"),
    # >=20% replay, no evidence attached at all. With no corpus attached the forward pass is
    # bit-identical to the model before the port existed, so these tokens genuinely protect the
    # trunk rather than quietly training the port on general chat.
    EvidenceSource("smoltalk2", 0.12, "HuggingFaceTB/smoltalk2", "SFT/", render="messages"),
    EvidenceSource("ultrachat", 0.08, "HuggingFaceH4/ultrachat_200k", "data/train_sft",
                   render="messages"),
    EvidenceSource("no_robots", 0.05, "HuggingFaceH4/no_robots", "data/train", render="messages"),
]


def _no_think_sft_split(path: str) -> bool:
    return "_no_think" in path


FILE_FILTERS = {"smoltalk2": _no_think_sft_split}


# --------------------------------------------------------------------------------------- rendering

def evidence_prompt(question: str) -> str:
    """The user turn: instruction and question, no passage.

    ``SQUAD_INSTRUCTION`` is imported rather than restated for the reason every other script in this
    repo imports it -- the instruction is what licenses abstention at all, and a copy that drifted by
    a word would train the model on a prompt the evals never send it. The passage block is simply
    absent; the evidence arrives through the port instead.
    """
    return f"{SQUAD_INSTRUCTION}\n\nQuestion: {question}"


def render_squad_row(row: dict) -> Optional[EvidenceRow]:
    """A SQuAD v2 row: one passage, gold for the answerable two thirds.

    The unanswerable third keeps its passage as ``gold`` deliberately. That passage IS what a
    retriever would return for the question -- right topic, right entities, retrieved for the right
    reason -- and it still does not answer. It is the only supervision in this corpus that separates
    "I retrieved something relevant" from "I can answer", which is exactly the distinction G3b's
    signal has to make.
    """
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    if not context or not question:
        return None
    answers = row.get("answers") or {}
    texts = answers.get("text") if isinstance(answers, dict) else None
    candidates = [str(t).strip() for t in (texts if texts is not None else []) if str(t).strip()]
    return EvidenceRow(
        question=question,
        answer=candidates[0] if candidates else "",
        gold=[context],
    )


def render_hotpot_row(row: dict) -> Optional[EvidenceRow]:
    """A HotpotQA distractor row: 2 supporting paragraphs among 8 retrieved distractors.

    The split into ``gold`` and ``near`` is what makes this source worth its cost. Its distractors
    were retrieved for this question by a real system, so they share entities and phrasing with the
    gold pair -- a random other row's passage is off-topic and trivially rejectable, and a model that
    only ever saw those would learn topic matching rather than selection.
    """
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    context = row.get("context")
    supporting = row.get("supporting_facts")
    if not question or not answer or context is None:
        return None
    titles = context.get("title") if isinstance(context, dict) else None
    sentence_lists = context.get("sentences") if isinstance(context, dict) else None
    if titles is None or sentence_lists is None:
        return None
    support_titles = set()
    if isinstance(supporting, dict) and supporting.get("title") is not None:
        support_titles = {str(t).strip() for t in supporting["title"]}

    gold, near = [], []
    for title, sentences in zip(titles, sentence_lists):
        body = "".join(str(s) for s in (sentences if sentences is not None else [])).strip()
        if not body:
            continue
        chunk = f"{str(title).strip()}: {body}"
        (gold if str(title).strip() in support_titles else near).append(chunk)
    if not gold:
        # no identifiable supporting paragraph -- the condition machinery would have no gold to hide
        # among the distractors, so the row cannot serve its purpose
        return None
    return EvidenceRow(question=question, answer=answer, gold=gold, near=near)


def render_messages_row(row: dict) -> Optional[List[dict]]:
    """A replay conversation, unchanged and with no evidence."""
    msgs = row.get("messages")
    if msgs is None or len(msgs) == 0:
        return None
    conversation = [
        {"role": str(m["role"]), "content": m.get("content")}
        for m in msgs if m is not None and m.get("role") is not None
    ]
    return conversation or None


def split_webtext_document(ids: Sequence[int], rng: random.Random, chunk_tokens: int,
                           min_tokens: int = 384) -> Optional[Tuple[List[int], List[int], List[List[int]]]]:
    """One document -> (prefix, supervised continuation, held-out evidence chunks).

    The document is cut into three consecutive spans -- prefix, evidence, continuation -- and the
    evidence span is *removed* from what the model reads. So the continuation genuinely depends on
    content only the port can supply, which is what makes it pseudo-gold rather than a paraphrase of
    something already visible. Placing the held-out span BETWEEN the two, rather than at the tail,
    is what makes it relevant: it is the text the continuation immediately follows from.

    Returns None for a document too short to cut three ways.
    """
    n = len(ids)
    if n < min_tokens:
        return None
    a = int(n * 0.25)
    b = a + max(chunk_tokens, int(n * 0.25))
    if b >= n - 32:
        return None
    prefix, held, cont = list(ids[:a]), list(ids[a:b]), list(ids[b:])
    chunks = [held[i:i + chunk_tokens] for i in range(0, len(held), chunk_tokens)]
    chunks = [c for c in chunks if len(c) >= 16]
    if not chunks:
        return None
    return prefix, cont, chunks


# ------------------------------------------------------------------------------------- the embedder

class ChunkEmbedder:
    """bge-small CLS embeddings with a text-keyed cache, L2-normalized.

    The cache is what keeps this affordable. A distractor chunk is reused by many documents, and
    re-embedding it each time would multiply the embedding pass by the distractor reuse factor for
    no change in the result -- the vector is a pure function of the text.

    CLS pooling, not mean: bge is trained with a CLS objective, and mean pooling produces vectors
    that still look plausible and cluster far worse.
    """

    REPO = "BAAI/bge-small-en-v1.5"

    def __init__(self, device: str = "cuda", batch_size: int = 256, cache_size: int = 200_000):
        from transformers import AutoModel
        self.tok = AutoTokenizer.from_pretrained(self.REPO)
        self.model = AutoModel.from_pretrained(self.REPO).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.cache = {}
        self.cache_size = cache_size
        self.embedded = 0
        self.cache_hits = 0

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tok(texts[start:start + self.batch_size], padding=True, truncation=True,
                             max_length=256, return_tensors="pt").to(self.device)
            cls = self.model(**batch).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls.float(), p=2, dim=-1).cpu().numpy())
        self.embedded += len(texts)
        return np.concatenate(out, axis=0) if out else np.zeros((0, EMBED_DIM), dtype=np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        """[len(texts), 384] float32, in the order given."""
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        keys = [hashlib.sha1(t.encode("utf-8", "ignore")).digest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        self.cache_hits += len(texts) - len(missing)
        if missing:
            fresh = self._embed([texts[i] for i in missing])
            for i, vec in zip(missing, fresh):
                if len(self.cache) >= self.cache_size:
                    # plain FIFO eviction: the reuse this cache exists for is temporally local (a
                    # distractor is drawn from a reservoir of recent chunks), so recency is the only
                    # signal worth tracking and an LRU's bookkeeping would buy nothing
                    self.cache.pop(next(iter(self.cache)))
                self.cache[keys[i]] = vec
        return np.stack([self.cache[k] for k in keys], axis=0)


# ---------------------------------------------------------------------------------------- the writer

EVIDENCE_SUFFIXES = ("bin", "idx", "mask", "ev", "evidx", "evchunk", "evkey", "evkeyidx")


def truncate_to_state(data_dir: str, split: str, split_state: dict) -> None:
    """Drop bytes past the last confirmed checkpoint, in all eight files at once.

    They are indexed by the same document number, so they have to be trimmed together or a resume
    would pair document *i*'s prompt with document *i+1*'s evidence -- which is not an error any
    length check catches, because both files would still be self-consistent.
    """
    docs = split_state.get("doc_count", 0)
    targets = {
        "bin": split_state.get("tokens_written", 0) * 2,
        "idx": (docs + 1) * 8,
        "mask": split_state.get("tokens_written", 0),
        "ev": split_state.get("ev_tokens", 0) * 2,
        "evidx": (docs + 1) * 8,
        "evchunk": split_state.get("ev_tokens", 0) * 2,
        "evkey": split_state.get("chunks_written", 0) * EMBED_DIM * 2,
        "evkeyidx": (docs + 1) * 8,
    }
    for suffix, target in targets.items():
        path = os.path.join(data_dir, f"{split}.{suffix}")
        if os.path.exists(path) and target < os.path.getsize(path):
            current = os.path.getsize(path)
            with open(path, "r+b") as f:
                f.truncate(target)
            logger.warning(f"truncated {split}.{suffix} from {current} to {target} bytes")


class EvidenceWriter:
    """Append-only writer for one split's eight files."""

    def __init__(self, data_dir: str, split: str, state: dict):
        self.split = split
        self.state = state
        for key in ("doc_count", "tokens_written", "ev_tokens", "chunks_written"):
            state.setdefault(key, 0)
        truncate_to_state(data_dir, split, state)
        self.files = {
            suffix: open(os.path.join(data_dir, f"{split}.{suffix}"), "ab")
            for suffix in EVIDENCE_SUFFIXES
        }
        # the three offset files each carry a leading 0 so a document's span is always
        # offsets[i]..offsets[i+1], with no special case for document 0
        for suffix in ("idx", "evidx", "evkeyidx"):
            if os.path.getsize(os.path.join(data_dir, f"{split}.{suffix}")) == 0:
                self.files[suffix].write(np.array([0], dtype=np.uint64).tobytes())
                self.files[suffix].flush()

    def write(self, ids: Sequence[int], mask: Sequence[int],
              ev_ids: Sequence[int], ev_chunk: Sequence[int], keys: np.ndarray) -> None:
        assert len(ids) == len(mask), "prompt ids and mask disagree"
        assert len(ev_ids) == len(ev_chunk), "evidence ids and chunk ids disagree"
        assert keys.shape[0] == 0 or keys.shape[1] == EMBED_DIM, f"bad key width {keys.shape}"

        self.files["bin"].write(np.asarray(ids, dtype=np.uint16).tobytes())
        self.files["mask"].write(np.asarray(mask, dtype=np.uint8).tobytes())
        self.files["ev"].write(np.asarray(ev_ids, dtype=np.uint16).tobytes())
        self.files["evchunk"].write(np.asarray(ev_chunk, dtype=np.uint16).tobytes())
        self.files["evkey"].write(np.asarray(keys, dtype=np.float16).tobytes())

        self.state["tokens_written"] += len(ids)
        self.state["ev_tokens"] += len(ev_ids)
        self.state["chunks_written"] += int(keys.shape[0])
        self.state["doc_count"] += 1
        self.files["idx"].write(np.asarray([self.state["tokens_written"]], dtype=np.uint64).tobytes())
        self.files["evidx"].write(np.asarray([self.state["ev_tokens"]], dtype=np.uint64).tobytes())
        self.files["evkeyidx"].write(np.asarray([self.state["chunks_written"]], dtype=np.uint64).tobytes())

    def sync(self) -> None:
        for f in self.files.values():
            f.flush()
            os.fsync(f.fileno())

    def close(self) -> None:
        for f in self.files.values():
            f.close()


# ------------------------------------------------------------------------------------ condition mix

def apply_condition(row: EvidenceRow, condition: str, reservoir: deque,
                    rng: random.Random, num_distractors: int) -> Tuple[List[str], str]:
    """Turn one row plus a condition into ``(chunk texts, target answer)``.

    The abstention targets are the load-bearing part. Under ``distractors`` and ``none`` the answer
    is replaced even for a row whose answer is perfectly well known -- that is the point: the model
    must learn that answerability is a property of *the buffer*, not of the question. A corpus that
    kept the real answer whenever it happened to know it would teach exactly the memorization Phase 2
    proved cannot be fixed by ratios.
    """
    unanswerable = not row.answer

    def distractors(k: int) -> List[str]:
        pool = list(row.near)
        rng.shuffle(pool)
        picked = pool[:k]
        if len(picked) < k and reservoir:
            extra = rng.sample(list(reservoir), min(k - len(picked), len(reservoir)))
            picked.extend(extra)
        return picked

    if condition == "gold":
        chunks = list(row.gold)
    elif condition == "mixed":
        chunks = list(row.gold) + distractors(num_distractors)
        rng.shuffle(chunks)
    elif condition == "distractors":
        chunks = distractors(max(1, num_distractors))
    else:
        chunks = []

    if row.lm:
        # no abstention target exists for language modeling: the continuation is the continuation
        # whether or not the buffer helps. What the unanswerable conditions teach here is that
        # unusable evidence must not COST anything, which is the finding the ceiling probe made.
        return chunks, row.answer

    if condition in ("distractors", "none") or unanswerable:
        return chunks, abstention.pick(abstention.ABSTENTIONS_PASSAGE_TRAIN, rng)
    return chunks, row.answer


def pick_condition(weights: dict, rng: random.Random) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


# ------------------------------------------------------------------------------------- the generators

def hub_row_factory(spec: EvidenceSource, files: Sequence[str], scratch_dir: str,
                    hf_token: Optional[str], seed: int, revision: Optional[str]) -> Callable:
    """factory(file_idx, row_idx) -> generator of (file_idx, row_idx, raw row dict).

    One shard in flight, deleted as soon as it is read, and the revision pinned -- same contract and
    same reasons as ``prepare_sft_data.make_generator_factory``. It yields raw rows rather than
    conversations because a row here becomes a different conversation under each condition.
    """
    def factory(start_file_idx: int, start_row_idx: int) -> Iterator:
        for file_idx in range(start_file_idx, len(files)):
            filename = files[file_idx]
            local_path = hf_hub_download(
                repo_id=spec.repo_id, filename=filename, repo_type="dataset",
                local_dir=scratch_dir, token=hf_token, revision=revision,
            )
            frame = pd.read_parquet(local_path, engine="pyarrow")
            try:
                os.remove(local_path)
            except OSError:
                pass
            # sha1 of the filename, not the salted builtin hash(): the shuffle has to be identical
            # across processes or a resume's row_idx points into a different ordering
            digest = hashlib.sha1(filename.encode("utf-8")).digest()
            rng = random.Random(seed ^ int.from_bytes(digest[:4], "big"))
            rows = frame.to_dict("records")
            rng.shuffle(rows)
            del frame
            row_start = start_row_idx if file_idx == start_file_idx else 0
            for row_idx in range(row_start, len(rows)):
                yield file_idx, row_idx, rows[row_idx]
    return factory


def local_document_factory(bin_path: str, idx_path: str) -> Callable:
    """factory(_, doc_idx) -> generator of (0, doc_idx, token id list) over a prepared corpus.

    Reads the flat ``{phase}.bin``/``.idx`` pair that ``prepare_data.py`` writes, in order. The web
    text arm needs *documents*, not the packed stream the trainer reads, which is why it goes through
    the index rather than through ``modules.data.dataset``.
    """
    def factory(_start_file_idx: int, start_row_idx: int) -> Iterator:
        tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        offsets = np.memmap(idx_path, dtype=np.uint64, mode="r")
        for doc_idx in range(int(start_row_idx), len(offsets) - 1):
            start, end = int(offsets[doc_idx]), int(offsets[doc_idx + 1])
            if end > start:
                yield 0, doc_idx, np.asarray(tokens[start:end], dtype=np.int64).tolist()
    return factory


# ------------------------------------------------------------------------------------------- driver

def build_corpus(
    source_entries: List[dict],
    target_tokens: int,
    template: ChatTemplate,
    embedder: ChunkEmbedder,
    data_dir: str,
    state_path: str,
    max_doc_tokens: int = 4094,
    max_evidence_tokens: int = 1536,
    num_distractors: int = 3,
    chunk_tokens: int = 128,
    reservoir_size: int = 4096,
    render_batch: int = 512,
    val_fraction: float = 0.01,
    seed: int = 42,
    checkpoint_docs: int = 2000,
    split_prefix: str = "evidence",
) -> dict:
    """Interleave sources, apply conditions, embed chunks, write both splits.

    Free of Hub calls so a test can drive it with synthetic in-memory sources, exactly as
    ``prepare_sft_data.build_corpus`` and ``prepare_data.run_phase`` are.

    Args:
        source_entries: ``[{"key", "weight", "render", "condition_weights", "row_factory"}]``.
        target_tokens: PROMPT tokens across both splits. Evidence tokens are counted separately.
        embedder: supplies the chunk vectors; any object with ``.encode(list[str]) -> [N, 384]``.
        max_evidence_tokens: rows whose evidence exceeds this are dropped rather than truncated --
            a truncated buffer silently removes the gold chunk some of the time, which would
            mislabel the condition rather than shorten it.
        num_distractors: how many distractors ``mixed`` and ``distractors`` draw when the row does
            not ship its own.
        render_batch: rows rendered before the tokenizer and the external embedder are called. Both
            are near flat in batch size and were the whole cost at one row per call.

    Returns:
        The final resume state, with per-source realized counts and per-condition document counts.
    """
    train_split, val_split = f"{split_prefix}_train", f"{split_prefix}_val"
    state = load_state(state_path)
    state.setdefault("sources", {})
    state.setdefault("splits", {})
    state.setdefault("conditions", {c: 0 for c in CONDITIONS})
    state.setdefault("skipped", {})
    for split in (train_split, val_split):
        state["splits"].setdefault(split, {})
    for entry in source_entries:
        state["sources"].setdefault(
            entry["key"], {"file_idx": 0, "row_idx": 0, "tokens": 0, "ev_tokens": 0,
                           "docs": 0, "done": False},
        )
        state["skipped"].setdefault(entry["key"], {"too_long": 0, "unrenderable": 0, "no_evidence": 0})

    writers = {s: EvidenceWriter(data_dir, s, state["splits"][s]) for s in (train_split, val_split)}
    split_rng = random.Random(seed)
    cond_rng = random.Random(seed + 1)
    chunk_rng = random.Random(seed + 2)
    # recent chunk texts, per source, to draw distractors from. Per source rather than global so a
    # SQuAD question never gets a web text distractor, which would be off-distribution enough to be
    # rejectable on surface form alone and would teach nothing about selection.
    reservoirs = {entry["key"]: deque(maxlen=reservoir_size) for entry in source_entries}

    active = {}
    for entry in source_entries:
        source_state = state["sources"][entry["key"]]
        active[entry["key"]] = {
            "weight": entry["weight"],
            "target": int(target_tokens * entry["weight"]),
            "gen": entry["row_factory"](source_state["file_idx"], source_state["row_idx"]),
            "state": source_state,
            "entry": entry,
        }

    swrr = {key: 0.0 for key in active}
    since_checkpoint = 0

    def total_tokens() -> int:
        return sum(s["tokens_written"] for s in state["splits"].values())

    def checkpoint() -> None:
        for writer in writers.values():
            writer.sync()
        save_state_atomic(state_path, state)

    def live() -> List[str]:
        return [k for k, v in active.items()
                if not v["state"]["done"] and v["state"]["tokens"] < v["target"]]

    t_last = time.time()
    while total_tokens() < target_tokens:
        candidates = live()
        if not candidates:
            logger.warning(
                f"all sources exhausted or at target after {total_tokens():,} of "
                f"{target_tokens:,} prompt tokens -- see the per-source realized counts"
            )
            break

        # Rows are rendered in GROUPS, and the group is the unit of work for both slow things here:
        # the model tokenizer parallelizes across a batch on its own threads, and the external
        # embedder is a GPU forward whose cost is nearly flat up to a few hundred chunks. Committing
        # one row at a time called the embedder with the two or three chunks that row happened to
        # retrieve, which ran a 33M parameter model at batch 3 -- measured at 85 documents/second,
        # against ~1,000 once the calls are grouped.
        renders = []
        while len(renders) < render_batch and candidates:
            total_w = sum(active[k]["weight"] for k in candidates)
            for k in candidates:
                swrr[k] += active[k]["weight"]
            pick = max(candidates, key=lambda k: swrr[k])
            swrr[pick] -= total_w

            slot = active[pick]
            try:
                file_idx, row_idx, raw = next(slot["gen"])
            except StopIteration:
                slot["state"]["done"] = True
                candidates = live()
                continue
            # advanced only after the row is consumed, never at pick time -- the same bug
            # prepare_data.run_phase documents, where a picked-but-uncommitted row is lost on resume
            slot["state"]["file_idx"] = file_idx
            slot["state"]["row_idx"] = row_idx + 1

            rendered = _render_row(
                raw, slot["entry"], template, state, reservoirs[pick], cond_rng, chunk_rng,
                num_distractors=num_distractors, chunk_tokens=chunk_tokens,
            )
            if rendered is not None:
                rendered["source"] = pick
                renders.append(rendered)
            candidates = live()

        if not renders:
            break

        _encode_batch(renders, template, max_doc_tokens, max_evidence_tokens, state)
        _embed_batch(renders, embedder)

        for rendered in renders:
            if rendered.get("dropped"):
                continue
            split = val_split if split_rng.random() < val_fraction else train_split
            writers[split].write(rendered["ids"], rendered["mask"], rendered["ev_ids"],
                                 rendered["ev_chunk"], rendered["keys"])
            source_state = state["sources"][rendered["source"]]
            source_state["tokens"] += len(rendered["ids"])
            source_state["ev_tokens"] += len(rendered["ev_ids"])
            source_state["docs"] += 1
            state["conditions"][rendered["condition"]] = (
                state["conditions"].get(rendered["condition"], 0) + 1
            )
            since_checkpoint += 1
            if total_tokens() >= target_tokens:
                break

        if since_checkpoint >= checkpoint_docs:
            checkpoint()
            now = time.time()
            logger.info(
                f"  {total_tokens():,}/{target_tokens:,} prompt tokens "
                f"({sum(s['ev_tokens'] for s in state['sources'].values()):,} evidence), "
                f"{state['splits'][train_split]['chunks_written']:,} chunks, "
                f"embed cache {embedder.cache_hits / max(1, embedder.cache_hits + embedder.embedded):.0%} hit, "
                f"{since_checkpoint / max(1e-6, now - t_last):.0f} docs/s"
            )
            since_checkpoint = 0
            t_last = now

    checkpoint()
    for writer in writers.values():
        writer.close()
    return state


def _render_row(raw, entry, template, state, reservoir, cond_rng, chunk_rng, *,
                num_distractors, chunk_tokens):
    """One raw row -> a pending record, or None if it is unusable.

    Everything that has to happen **in row order** lives here: the condition draw, and the distractor
    reservoir, which a later row samples from and so cannot be reordered. Everything batchable -- the
    tokenizer and the external embedder -- is deferred to ``_encode_batch``/``_embed_batch``.
    """
    key = entry["key"]
    render = entry["render"]

    if render == "messages":
        conversation = render_messages_row(raw)
        if conversation is None:
            state["skipped"][key]["unrenderable"] += 1
            return None
        return {"conversation": conversation, "chunks": [], "condition": "none", "key": key}

    condition = pick_condition(entry["condition_weights"], cond_rng)

    if render == "webtext":
        pieces = split_webtext_document(raw, chunk_rng, chunk_tokens)
        if pieces is None:
            state["skipped"][key]["unrenderable"] += 1
            return None
        prefix, cont, held = pieces
        # BOS by hand: this row bypasses the chat template (there is no conversation here, it is
        # plain language modeling), and the pretraining dataset prepends one to every document
        ids = [template.bos_id] + prefix + cont
        mask = [0] * (len(prefix) + 1) + [1] * len(cont)
        gold_texts = [template.tokenizer.decode(c, skip_special_tokens=True) for c in held]
        chunk_texts, _ = apply_condition(
            EvidenceRow(gold=gold_texts, lm=True), condition, reservoir, cond_rng, num_distractors
        )
        reservoir.extend(gold_texts)
        record = {"conversation": None, "ids": ids, "mask": mask,
                  "chunks": chunk_texts, "condition": condition, "key": key}
    else:
        row = (render_squad_row if render == "squad_v2" else render_hotpot_row)(raw)
        if row is None:
            state["skipped"][key]["unrenderable"] += 1
            return None
        chunk_texts, answer = apply_condition(row, condition, reservoir, cond_rng, num_distractors)
        reservoir.extend(row.gold)
        reservoir.extend(row.near)
        record = {
            "conversation": [
                {"role": "user", "content": evidence_prompt(row.question)},
                {"role": "assistant", "content": answer},
            ],
            "chunks": chunk_texts, "condition": condition, "key": key,
        }

    if condition != "none" and not chunk_texts:
        # a condition that promised evidence and produced none would be silently relabelled as
        # `none`, which is the one mislabelling this corpus cannot tolerate: the abstention targets
        # differ between them
        state["skipped"][key]["no_evidence"] += 1
        return None
    return record


def _encode_batch(renders, template, max_doc_tokens, max_evidence_tokens, state):
    """Tokenize a group's prompts and evidence in as few calls as possible, marking drops.

    The fast tokenizer parallelizes across a batch on its own Rust threads, so a per-row call is
    mostly FFI overhead -- the same reason ``prepare_sft_data.build_corpus`` batches its
    ``encode_batch``.
    """
    pending = [r for r in renders if r["conversation"] is not None]
    if pending:
        encoded = template.encode_batch([r["conversation"] for r in pending])
        for record, result in zip(pending, encoded):
            if result is None:
                state["skipped"][record["key"]]["unrenderable"] += 1
                record["dropped"] = True
                continue
            record["ids"], record["mask"] = result

    # every chunk of every row in one tokenizer call, then split back by row
    flat, spans = [], []
    for record in renders:
        start = len(flat)
        flat.extend(record["chunks"])
        spans.append((start, len(flat)))
    pieces = template.tokenizer(flat, add_special_tokens=False)["input_ids"] if flat else []

    for record, (start, end) in zip(renders, spans):
        if record.get("dropped"):
            continue
        if len(record["ids"]) > max_doc_tokens:
            state["skipped"][record["key"]]["too_long"] += 1
            record["dropped"] = True
            continue
        # renumbered as they are kept, so a chunk whose text tokenized to nothing does not leave a
        # hole. A hole would be a chunk the SELECTOR can score and win mass on while the READER has
        # no tokens for it -- the two halves of the port disagreeing about what was retrieved.
        ev_ids, ev_chunk, kept = [], [], []
        for text, piece in zip(record["chunks"], pieces[start:end]):
            if not piece:
                continue
            ev_chunk.extend([len(kept)] * len(piece))
            ev_ids.extend(piece)
            kept.append(text)
        if len(ev_ids) > max_evidence_tokens:
            # dropped, never truncated: truncation removes whichever chunk happened to land last,
            # which is the gold one a third of the time under `mixed` -- that mislabels the
            # condition rather than shortening it
            state["skipped"][record["key"]]["too_long"] += 1
            record["dropped"] = True
            continue
        record["ev_ids"], record["ev_chunk"], record["chunks"] = ev_ids, ev_chunk, kept


def _embed_batch(renders, embedder):
    """One external-embedder call for the whole group, split back by row."""
    flat, spans = [], []
    for record in renders:
        if record.get("dropped"):
            spans.append(None)
            continue
        start = len(flat)
        flat.extend(record["chunks"])
        spans.append((start, len(flat)))
    vectors = embedder.encode(flat)
    for record, span in zip(renders, spans):
        if span is None:
            continue
        start, end = span
        record["keys"] = (vectors[start:end] if end > start
                          else np.zeros((0, EMBED_DIM), dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description="Build the evidence-conditioned corpus")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data", "prepared"))
    parser.add_argument("--target-tokens", type=int, default=150_000_000,
                        help="PROMPT tokens across both splits; evidence tokens are reported "
                             "separately (see the module docstring)")
    parser.add_argument("--max-doc-tokens", type=int, default=4094)
    parser.add_argument("--max-evidence-tokens", type=int, default=1536)
    parser.add_argument("--num-distractors", type=int, default=3)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--checkpoint-docs", type=int, default=2000)
    parser.add_argument("--render-batch", type=int, default=512,
                        help="rows per tokenizer/embedder call; the embedder is a GPU forward and "
                             "runs at roughly flat cost up to a few hundred chunks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-prefix", default="evidence")
    parser.add_argument("--webtext-phase", default="ir",
                        help="which prepared {phase}.bin/.idx the web text arm reads")
    parser.add_argument("--no-webtext", action="store_true",
                        help="skip the web text arm (QA and replay only)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    scratch_dir = os.path.join(args.data_dir, "_evidence_scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    template = ChatTemplate(tokenizer)
    hf_token = get_hf_token()

    sources = [s for s in SOURCES if not (args.no_webtext and s.render == "webtext")]
    hf_api = HfApi(token=hf_token)
    entries = []
    for spec in sources:
        if spec.render == "webtext":
            bin_path = os.path.join(args.data_dir, f"{args.webtext_phase}.bin")
            idx_path = os.path.join(args.data_dir, f"{args.webtext_phase}.idx")
            if not os.path.exists(bin_path):
                raise SystemExit(
                    f"{bin_path} does not exist -- the web text arm reads a prepared corpus. "
                    f"Build one with scripts/prepare_data.py, or pass --no-webtext."
                )
            factory = local_document_factory(bin_path, idx_path)
        else:
            info = hf_api.dataset_info(spec.repo_id)
            all_files = sorted(
                f for f in hf_api.list_repo_files(spec.repo_id, repo_type="dataset", revision=info.sha)
                if f.startswith(spec.file_prefix) and f.endswith(spec.file_suffix)
                and (FILE_FILTERS.get(spec.key) is None or FILE_FILTERS[spec.key](f))
            )
            if not all_files:
                raise RuntimeError(f"no files matched {spec.key} under {spec.file_prefix!r}")
            logger.info(f"source {spec.key}: {len(all_files)} files (revision {info.sha[:10]})")
            factory = hub_row_factory(spec, all_files, scratch_dir, hf_token, args.seed, info.sha)
        entries.append({
            "key": spec.key, "weight": spec.weight, "render": spec.render,
            "condition_weights": spec.condition_weights, "row_factory": factory, "qa": spec.qa,
        })

    total_w = sum(e["weight"] for e in entries)
    for e in entries:
        e["weight"] /= total_w

    logger.info(f"loading the external embedder ({ChunkEmbedder.REPO})")
    embedder = ChunkEmbedder(device=args.device)

    state_path = os.path.join(args.data_dir, f"_prepare_state_{args.split_prefix}.json")
    logger.info(f"=== evidence corpus: target {args.target_tokens:,} prompt tokens ===")
    t0 = time.time()
    final = build_corpus(
        entries, args.target_tokens, template, embedder, args.data_dir, state_path,
        max_doc_tokens=args.max_doc_tokens, max_evidence_tokens=args.max_evidence_tokens,
        num_distractors=args.num_distractors, chunk_tokens=args.chunk_tokens,
        render_batch=args.render_batch, val_fraction=args.val_fraction, seed=args.seed,
        checkpoint_docs=args.checkpoint_docs, split_prefix=args.split_prefix,
    )
    elapsed = time.time() - t0

    logger.info(f"=== built in {elapsed / 60:.1f} min ===")
    replay_tokens = 0
    for e in entries:
        s = final["sources"][e["key"]]
        target = int(args.target_tokens * e["weight"])
        logger.info(
            f"  {e['key']}: {s['tokens']:,}/{target:,} prompt tokens, {s['ev_tokens']:,} evidence, "
            f"{s['docs']:,} docs, skipped {final['skipped'][e['key']]}"
        )
        if e["render"] == "messages":
            replay_tokens += s["tokens"]
    total_prompt = sum(sp["tokens_written"] for sp in final["splits"].values())
    logger.info(f"  conditions: {final['conditions']}")
    logger.info(
        f"  replay share: {replay_tokens / max(1, total_prompt):.1%} "
        f"(the floor is 20% -- below it the trunk is not protected)"
    )
    for split, sp in final["splits"].items():
        logger.info(
            f"  {split}: {sp['doc_count']:,} docs, {sp['tokens_written']:,} prompt tokens, "
            f"{sp['ev_tokens']:,} evidence tokens, {sp['chunks_written']:,} chunks"
        )


if __name__ == "__main__":
    main()

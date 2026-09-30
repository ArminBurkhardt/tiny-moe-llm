"""Build the evidence-conditioned corpus: five conditions plus replay.

The trick this corpus exists to exploit is that **relevance can be known before any retriever
exists**. For QA the gold passage ships with the dataset; for web text a held-out span of the same
document is relevant by construction. So the port can be trained against an oracle retriever now,
and Phase 5 only has to replace the oracle with a real index later.

What makes it several conditions rather than one is a measurement, not thoroughness. The in-context
ceiling probe (``docs/measurements/evidence_ceiling.md``) found that handing this model a passage
from *another question* costs **0.628 nats more than handing it no passage at all** -- it reads
whatever it is given, uncritically, and is slightly more confident while doing it. A port trained
only on gold evidence would therefore make the model strictly worse the moment a real retriever
handed it something irrelevant, which is what a real retriever mostly does.

| condition | evidence | target |
|---|---|---|
| ``gold`` | the relevant chunk(s) | the real answer |
| ``mixed`` | gold shuffled among distractors | the real answer -- select, then read |
| ``many`` | gold shuffled among 16-32 distractors | the real answer -- select at eval-scale buffers |
| ``distractors`` | distractors only | abstain: the answer is not in the buffer |
| ``none`` | nothing at all | abstain: grounded in retrieval, not memorized |

``many`` exists because ``mixed``'s few distractors never come close to the buffer sizes the later
retrieval evals run at -- it is QA-only (drawn from ``QA_CONDITIONS``, not the web text mix), so the
selector sees a large buffer somewhere in training rather than only at eval time.

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

Eight files beyond the usual ``{split}.{bin,idx,mask}`` triple. The first five are indexed by the
same document number (or, for the two chunk-level ones, the same chunk axis) so a row and its
evidence cannot drift apart; the last three add a chunk-level and two document-level labels to that
same pair of axes:

    {split}.ev        uint16   evidence token stream
    {split}.evidx     uint64   per document offsets into .ev (doc_count + 1 entries)
    {split}.evchunk   uint16   per evidence token, its chunk index WITHIN that document
    {split}.evkey     float16  [total chunks, 384] external embedder vectors, flat
    {split}.evkeyidx  uint64   per document offsets into the chunk axis (doc_count + 1 entries)
    {split}.evgold    uint8    per chunk (same axis as .evkey/.evkeyidx): 1 = gold, 0 = distractor
    {split}.cond      uint8    per document (same axis as .idx/.evidx): index into CONDITIONS
    {split}.ans       uint8    per document: 1 = the target is the real answer, 0 = a refusal

A document with no evidence writes nothing and leaves both offsets equal to the previous entry.
That is a genuinely empty evidence segment at train time, which flash answers with exact zeros --
so "no evidence" needs no sentinel token and no placeholder chunk, and reads as the absence it is.

Distractor chunks are written per occurrence rather than referenced, so a chunk reused as a
distractor by five documents costs five copies of its 384 floats. Deduplicating would need a global
chunk table and a second level of indirection; at ~0.8 KB a copy the duplication is cheaper than the
machinery, and it keeps a document's evidence contiguous on disk.

``evgold`` exists because ``apply_condition`` shuffles the gold chunk(s) into the distractors for
``mixed``/``many`` and, by design, forgets which one it was -- a selection loss needs the label back,
and it has to live on the chunk axis because the mix itself is what makes selection a real task.
``cond`` exists so validation loss (and any other per-row statistic) can be split by condition without
re-deriving it from the evidence buffer's shape, which is ambiguous for a row where the draw happened
to produce an empty or gold-only buffer under more than one condition. ``ans`` is the other half of
the groundedness label and is NOT a function of ``cond``: a natively unanswerable SQuAD row is built
under ``gold`` like any other and still takes an abstention target, and separating those two is the
whole point of a groundedness readout. Recovering it downstream would mean matching the target text
against the abstention phrasings -- exact, since the set is closed, but a second opinion about a fact
this file already knows when it picks the target.

### Token accounting

``--target-tokens`` counts **prompt** tokens -- the ``.bin`` stream, which is what the trainer's LR
schedule is anchored to and what the model runs a full forward over. Evidence tokens are counted and
reported separately: they cost an embedding lookup plus the reader's cross attention, not a forward
pass, so folding them into one number would misstate both the run length and the cost.

Usage:

    python scripts/prepare_evidence_data.py                       # 150M prompt tokens
    python scripts/prepare_evidence_data.py --target-tokens 50000000 --no-webtext
    python scripts/prepare_evidence_data.py --heldout --max-evidence-tokens 4608

### Held-out splits

The ``{prefix}_val`` split the training build writes is a train-loss slice for QA: it is cut per
rendered row AFTER the QA sources repeat, so every QA question in it also appears in train. It is
still a fair slice of the web text and replay arms and of the loss on rows the trainer has seen.

``--heldout`` builds two splits from SQuAD v2 dev and HotpotQA dev instead, which the training
splits never touch, and never reads or writes the training corpus or its state file:

    {prefix}_dev      every usable dev row once, one QA condition each, targets as in training
    {prefix}_fixed    answerable rows only, each written four times in a row (question-major) under
                      gold, mixed, distractors, none, all with the REAL answer as target and
                      ``ans`` = 1, so one fixed target can be teacher forced under every condition

Rows of both sources are interleaved by one seeded shuffle, so any prefix is a representative
sample. Distractors come from the dev set's own passages, never the row's own gold. Pass the same
``--max-evidence-tokens`` as the training build: the default drops every ``many`` row.
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
from scripts.prepare_data import doc_hash, load_state, render_pretrain_chat, save_state_atomic
from scripts.prepare_sft_data import is_unanswerable_squad
from utils import BASE_DIR, TOKENIZER_DIR, get_hf_token, logger

MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")

EMBED_DIM = 384          # bge-small-en-v1.5, which is also ir_dim -- see the adapters in moe.py
CONDITIONS = ("gold", "mixed", "many", "distractors", "none")


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
    # conditions this source draws from, and their relative frequency. QA sources carry all five;
    # replay carries none at all and is the >=20% of the corpus that protects the trunk.
    condition_weights: dict = field(default_factory=dict)
    local_bin: str = ""              # web text reads the prepared corpus instead of the Hub
    qa: bool = False
    holdout: bool = False            # check each row against the pretraining holdout hashes
    # start another pass over the source when it runs out before reaching its token target. Only
    # the QA sources set it, and only because a second pass over them is not duplication: a row
    # becomes a different conversation under each condition, so a question that arrived as `gold`
    # last pass is likely `distractors` on this one, with distractors drawn from a reservoir that
    # has moved on. That is the corpus's own lesson -- answerability is a property of the buffer,
    # not of the question -- taught on the same question twice. It does repeat the ANSWER text,
    # which is why ``--max-source-epochs`` bounds it rather than letting a small source be looped
    # arbitrarily far. Replay must never repeat (it exists to be ordinary unseen text) and web text
    # has no need to: the prepared corpus behind it is larger than any target here.
    repeat: bool = False


# Condition mix. ``mixed`` is the largest share because it is the only condition that matches what a
# real retriever delivers -- the others are its endpoints. ``distractors`` and ``none`` together take
# a third, which is what has to carry the abstention policy: Phase 2 established that no data ratio
# of in-prompt refusals moves discrimination, so these two are the replacement lever and a token
# share that made them a rounding error would be testing nothing. ``many`` is carved out of
# ``mixed``/``distractors`` rather than added on top, so the four endpoints still sum to 1.0 -- it
# exists to put a buffer size near what the retrieval evals run at somewhere in training, not to
# change how much of the corpus is gold/abstain.
QA_CONDITIONS = {"gold": 0.25, "mixed": 0.36, "many": 0.08, "distractors": 0.16, "none": 0.15}
# web text has no abstention target, so ``none`` is pure replay-with-a-question-shape and earns
# less; the distractor share is higher because "irrelevant evidence must not cost anything" is the
# lesson this source exists to teach at scale. No ``many`` entry -- the large-buffer condition is
# QA-only, where a real answer/abstain target makes "select among many" a task worth training.
LM_CONDITIONS = {"gold": 0.40, "mixed": 0.30, "distractors": 0.20, "none": 0.10}

SOURCES = [
    # both QA sources are FINITE and small: SQuAD v2's train split is ~130k rows (~5.9M prompt
    # tokens) and HotpotQA's ~90k (~4.9M), against weights that ask for 20% and 15% of a corpus
    # sized in the hundreds of millions. Read once they cannot come close, and the first build made
    # that concrete -- QA landed at 10% of tokens and ~18% of the gradient, i.e. the only source
    # that exercises the evidence port was a minority of the training signal. `repeat` is what
    # closes that gap; see the field's own comment for why re-reading these rows is not duplication.
    EvidenceSource("squad_v2", 0.20, "rajpurkar/squad_v2", "squad_v2/train",
                   render="squad_v2", condition_weights=QA_CONDITIONS, qa=True, repeat=True),
    EvidenceSource("hotpot_qa", 0.15, "hotpotqa/hotpot_qa", "distractor/train",
                   render="hotpot_qa", condition_weights=QA_CONDITIONS, qa=True, repeat=True),
    EvidenceSource("webtext", 0.40, render="webtext", condition_weights=LM_CONDITIONS,
                   local_bin="ir"),
    # >=20% replay, no evidence attached at all. With no corpus attached the forward pass is
    # bit-identical to the model before the port existed, so these tokens genuinely protect the
    # trunk rather than quietly training the port on general chat. smoltalk2 alone carries the
    # holdout check: it is the one replay source phase-2 pretraining also drew from.
    EvidenceSource("smoltalk2", 0.12, "HuggingFaceTB/smoltalk2", "SFT/", render="messages",
                   holdout=True),
    EvidenceSource("ultrachat", 0.08, "HuggingFaceH4/ultrachat_200k", "data/train_sft",
                   render="messages"),
    EvidenceSource("no_robots", 0.05, "HuggingFaceH4/no_robots", "data/train", render="messages"),
]


def _no_think_sft_split(path: str) -> bool:
    return "_no_think" in path


FILE_FILTERS = {"smoltalk2": _no_think_sft_split}


# --------------------------------------------------------------------------------------- rendering

# SQUAD_INSTRUCTION says "using only the passage below", which is simply false here -- there is no
# passage below, it arrives through the evidence port instead. Not edited in place: eval_abstention.py
# forces that exact string as a teacher forced reference, and its CE has to stay comparable across
# every checkpoint it scores, so the shared constant has to keep saying what it has always said. This
# is the same instruction otherwise, so abstention is licensed the same way.
EVIDENCE_SQUAD_INSTRUCTION = (
    "Answer the question using only the passage attached as retrieved evidence. If the evidence "
    "does not contain the answer, say so."
)


def evidence_prompt(question: str) -> str:
    """The user turn: instruction and question, no passage.

    Uses ``EVIDENCE_SQUAD_INSTRUCTION`` rather than the shared ``SQUAD_INSTRUCTION`` -- see the
    comment above it. The passage block is simply absent; the evidence arrives through the port
    instead.
    """
    return f"{EVIDENCE_SQUAD_INSTRUCTION}\n\nQuestion: {question}"


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
                           max_doc_tokens: int, held_tokens: int = 384,
                           min_tokens: int = 384) -> Optional[Tuple[List[int], List[int], List[List[int]]]]:
    """One document -> (prefix, supervised continuation, held-out evidence chunks).

    The document is cut into three consecutive spans -- prefix, evidence, continuation -- and the
    evidence span is *removed* from what the model reads. So the continuation genuinely depends on
    content only the port can supply, which is what makes it pseudo-gold rather than a paraphrase of
    something already visible. Placing the held-out span BETWEEN the two, rather than at the tail,
    is what makes it relevant: it is the text the continuation immediately follows from.

    The held span is a FIXED token count (``held_tokens``), not a fraction of the document -- a
    percentage span grows without bound on a long document, and it is exactly the long tail where the
    port needs to carry content the prompt can no longer hold. For a document long enough that
    prefix+continuation would still overflow ``max_doc_tokens`` even with a small fixed span, a
    bounded WINDOW is taken instead of the whole document -- a slice of the prefix ENDING at the held
    span, and a slice of the continuation starting right after it -- so the long tail survives as a
    shorter row instead of being dropped outright.

    Returns None for a document too short to cut three ways, or one the windowing leaves with no
    usable continuation.
    """
    n = len(ids)
    if n < min_tokens:
        return None
    held_len = min(held_tokens, max(16, n - 64))
    if held_len < 16:
        return None
    a = int(n * 0.25)
    b = a + held_len
    if b > n - 32:
        # push the span left so at least 32 tokens of continuation remain after it
        b = n - 32
        a = b - held_len
        if a < 0:
            return None

    # prefix + continuation is the prompt (plus one BOS the caller prepends by hand); a long
    # document can overflow the cap even with a small fixed span, so window it down rather than
    # drop it -- most of the budget goes to the continuation, since that is the supervised half and
    # the reason the row exists, with only a quarter reserved for prefix context
    budget = max(0, max_doc_tokens - 1)
    prefix_len = min(a, budget // 4)
    cont_len = min(n - b, budget - prefix_len)
    if cont_len < 32:
        return None
    prefix, held, cont = list(ids[a - prefix_len:a]), list(ids[a:b]), list(ids[b:b + cont_len])
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
        # this call's own vectors, held separately from the cache so the result cannot depend on
        # what eviction does below. Reading the answer back out of self.cache instead is a real
        # failure and not a rare one: once the cache is full, inserting this batch's misses evicts
        # the OLDEST entries, and the oldest entries are exactly the long-lived keys this batch hit
        # on -- so the lookup at the end raises KeyError on a key that was present when the batch
        # started. It first fires at whatever document fills the cache (~70M tokens in at 200k
        # entries), which is why no smoke-sized build ever saw it.
        vectors = {}
        missing = []
        for i, key in enumerate(keys):
            cached = self.cache.get(key)
            if cached is None:
                missing.append(i)
            else:
                vectors[key] = cached
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
                vectors[keys[i]] = vec
        return np.stack([vectors[k] for k in keys], axis=0)


# ---------------------------------------------------------------------------------------- the writer

EVIDENCE_SUFFIXES = ("bin", "idx", "mask", "ev", "evidx", "evchunk", "evkey", "evkeyidx",
                     "evgold", "cond", "ans")


def truncate_to_state(data_dir: str, split: str, split_state: dict) -> None:
    """Drop bytes past the last confirmed checkpoint, in all ten files at once.

    They are indexed by the same document number (or the same chunk axis for the two chunk-level
    files), so they have to be trimmed together or a resume would pair document *i*'s prompt with
    document *i+1*'s evidence -- which is not an error any length check catches, because both files
    would still be self-consistent.
    """
    docs = split_state.get("doc_count", 0)
    chunks = split_state.get("chunks_written", 0)
    targets = {
        "bin": split_state.get("tokens_written", 0) * 2,
        "idx": (docs + 1) * 8,
        "mask": split_state.get("tokens_written", 0),
        "ev": split_state.get("ev_tokens", 0) * 2,
        "evidx": (docs + 1) * 8,
        "evchunk": split_state.get("ev_tokens", 0) * 2,
        "evkey": chunks * EMBED_DIM * 2,
        "evkeyidx": (docs + 1) * 8,
        "evgold": chunks,
        "cond": docs,
        "ans": docs,
    }
    for suffix, target in targets.items():
        path = os.path.join(data_dir, f"{split}.{suffix}")
        if os.path.exists(path) and target < os.path.getsize(path):
            current = os.path.getsize(path)
            with open(path, "r+b") as f:
                f.truncate(target)
            logger.warning(f"truncated {split}.{suffix} from {current} to {target} bytes")


class EvidenceWriter:
    """Append-only writer for one split's ten files."""

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
        # offsets[i]..offsets[i+1], with no special case for document 0. evgold/cond need no
        # leading entry -- they hold one value per chunk/document directly, not an offset pair
        for suffix in ("idx", "evidx", "evkeyidx"):
            if os.path.getsize(os.path.join(data_dir, f"{split}.{suffix}")) == 0:
                self.files[suffix].write(np.array([0], dtype=np.uint64).tobytes())
                self.files[suffix].flush()

    def write(self, ids: Sequence[int], mask: Sequence[int], ev_ids: Sequence[int],
              ev_chunk: Sequence[int], keys: np.ndarray, ev_gold: Sequence[int],
              condition_idx: int, answerable: int) -> None:
        assert len(ids) == len(mask), "prompt ids and mask disagree"
        assert len(ev_ids) == len(ev_chunk), "evidence ids and chunk ids disagree"
        assert keys.shape[0] == 0 or keys.shape[1] == EMBED_DIM, f"bad key width {keys.shape}"
        assert keys.shape[0] == len(ev_gold), "chunk keys and gold flags disagree"

        self.files["bin"].write(np.asarray(ids, dtype=np.uint16).tobytes())
        self.files["mask"].write(np.asarray(mask, dtype=np.uint8).tobytes())
        self.files["ev"].write(np.asarray(ev_ids, dtype=np.uint16).tobytes())
        self.files["evchunk"].write(np.asarray(ev_chunk, dtype=np.uint16).tobytes())
        self.files["evkey"].write(np.asarray(keys, dtype=np.float16).tobytes())
        self.files["evgold"].write(np.asarray(ev_gold, dtype=np.uint8).tobytes())
        self.files["cond"].write(np.asarray([condition_idx], dtype=np.uint8).tobytes())
        # answerability is a separate axis from the condition and cannot be recovered from it: a
        # natively unanswerable SQuAD row is built under `gold` like any other and still takes an
        # abstention target, which is exactly the pair a groundedness readout has to tell apart.
        # Recorded here, where apply_condition's own decision is still in hand, rather than
        # re-derived downstream by matching the target text against the abstention phrasings
        self.files["ans"].write(np.asarray([answerable], dtype=np.uint8).tobytes())

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

def _shuffle_paired(chunks: List[str], gold_flags: List[bool],
                    rng: random.Random) -> Tuple[List[str], List[bool]]:
    """Shuffle chunk texts and their gold flags together, so the flag still names the right chunk."""
    if not chunks:
        return [], []
    paired = list(zip(chunks, gold_flags))
    rng.shuffle(paired)
    shuffled_chunks, shuffled_flags = zip(*paired)
    return list(shuffled_chunks), list(shuffled_flags)


class DistractorPool:
    """A fixed, deduplicated pool of passage texts to draw distractors from.

    The training build draws from a moving reservoir of recent rows. A held-out split has no such
    stream and needs a pool that is a function of the split alone, and one that can be asked to
    skip a row's own gold: SQuAD dev carries several questions per context, so an unfiltered pool
    would sometimes hand a question its own answer passage inside a buffer whose target says the
    answer is absent.

    Drawing is by rejection, so a draw costs O(k) rather than a copy of a pool that can hold tens of
    thousands of paragraphs.

    Attributes:
        texts: the distinct passages, in first-seen order.
        gold_rejected: how many candidate draws were thrown away for equalling an excluded text.
    """

    def __init__(self, texts: Sequence[str]):
        self.texts = list(dict.fromkeys(texts))
        self.gold_rejected = 0

    def draw(self, k: int, exclude: set, taken: Sequence[str], rng: random.Random) -> List[str]:
        """Up to ``k`` distinct pool texts that are neither in ``exclude`` nor in ``taken``."""
        n = len(self.texts)
        if k <= 0 or n == 0:
            return []
        skip = set(exclude) | set(taken)
        out: List[str] = []
        for _ in range(20 * k + 20):
            if len(out) == k:
                return out
            text = self.texts[rng.randrange(n)]
            if text in exclude:
                self.gold_rejected += 1
            if text in skip:
                continue
            skip.add(text)
            out.append(text)
        if len(out) < k:
            # a pool barely larger than the request: finish exactly instead of rejecting forever
            rest = [t for t in self.texts if t not in skip]
            out.extend(rng.sample(rest, min(k - len(out), len(rest))))
        return out


def apply_condition(row: EvidenceRow, condition: str, reservoir: deque, rng: random.Random,
                    num_distractors: int,
                    many_distractors: Tuple[int, int] = (16, 32),
                    pool: Optional[DistractorPool] = None) -> Tuple[List[str], List[bool], str, bool]:
    """Turn one row plus a condition into ``(chunk texts, gold flags, target answer, answerable)``.

    The abstention targets are the load-bearing part. Under ``distractors`` and ``none`` the answer
    is replaced even for a row whose answer is perfectly well known -- that is the point: the model
    must learn that answerability is a property of *the buffer*, not of the question. A corpus that
    kept the real answer whenever it happened to know it would teach exactly the memorization Phase 2
    proved cannot be fixed by ratios.

    The gold flags are returned alongside the (possibly shuffled) chunks rather than left for the
    caller to re-derive, because ``mixed``/``many`` shuffle gold in among the distractors and forget
    which one it was -- the flag is the only place that distinction survives.

    ``answerable`` is the fourth return for the same reason: it is "this row's target is the real
    answer, not a refusal", which is decided right here and is NOT a function of the condition. A
    natively unanswerable SQuAD row takes an abstention target under ``gold`` exactly like it does
    under ``none``, and telling those two apart is the whole job of a groundedness readout. A
    language modeling row is always answerable -- its continuation is its continuation.
    """
    unanswerable = not row.answer

    def distractors(k: int) -> List[str]:
        near = list(row.near)
        rng.shuffle(near)
        picked = near[:k]
        if pool is not None:
            # held-out path: top up from the split's own pool, never with the row's own gold
            if len(picked) < k:
                picked.extend(pool.draw(k - len(picked), set(row.gold), picked, rng))
            return picked
        if len(picked) < k and reservoir:
            extra = rng.sample(list(reservoir), min(k - len(picked), len(reservoir)))
            picked.extend(extra)
        return picked

    if condition == "gold":
        chunks, gold_flags = list(row.gold), [True] * len(row.gold)
    elif condition == "mixed":
        gold, distract = list(row.gold), distractors(num_distractors)
        chunks, gold_flags = _shuffle_paired(
            gold + distract, [True] * len(gold) + [False] * len(distract), rng,
        )
    elif condition == "many":
        # same shape as `mixed`, at the buffer sizes the retrieval evals actually run at
        k = rng.randint(*many_distractors)
        gold, distract = list(row.gold), distractors(k)
        chunks, gold_flags = _shuffle_paired(
            gold + distract, [True] * len(gold) + [False] * len(distract), rng,
        )
    elif condition == "distractors":
        chunks = distractors(max(1, num_distractors))
        gold_flags = [False] * len(chunks)
    else:
        chunks, gold_flags = [], []

    if row.lm:
        # no abstention target exists for language modeling: the continuation is the continuation
        # whether or not the buffer helps. What the unanswerable conditions teach here is that
        # unusable evidence must not COST anything, which is the finding the ceiling probe made.
        return chunks, gold_flags, row.answer, True

    if condition in ("distractors", "none") or unanswerable:
        return chunks, gold_flags, abstention.pick(abstention.ABSTENTIONS_PASSAGE_TRAIN, rng), False
    return chunks, gold_flags, row.answer, True


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

def _start_next_epoch(slot: dict, max_source_epochs: int) -> bool:
    """Rewind an exhausted source for another pass, or report that it is finished.

    Three conditions, all of them there to make the rewind terminate. The source has to be marked
    ``repeat`` (QA only -- see ``EvidenceSource.repeat``); it has to still be short of its token
    target, so a source that met its share simply stops; and it has to be under the epoch cap,
    which is what bounds how often the same answer text can reappear.

    The fourth guard is the one that is not a policy: **an epoch that produced no new tokens ends
    the source**, whatever the cap says. Without it, a source whose every row is dropped (too long,
    unrenderable, held out) would rewind forever at full speed and the build would never finish or
    fail -- it would simply stop making progress, which is the failure mode hardest to see in a log.

    Returns:
        True if the source was rewound and the caller should keep drawing from it.
    """
    state, entry = slot["state"], slot["entry"]
    if not entry.get("repeat") or state["tokens"] >= slot["target"]:
        return False
    epoch = state.get("epoch", 0)
    if epoch + 1 >= max_source_epochs:
        logger.info(
            f"source {entry['key']}: exhausted after {epoch + 1} pass(es) and "
            f"{state['tokens']:,} of {slot['target']:,} target tokens -- at the epoch cap"
        )
        return False
    if state["tokens"] <= state.get("tokens_at_epoch_start", 0):
        logger.warning(
            f"source {entry['key']}: a whole pass produced no tokens -- stopping it rather than "
            f"rewinding again (every row dropped?)"
        )
        return False
    state["epoch"] = epoch + 1
    state["tokens_at_epoch_start"] = state["tokens"]
    state["file_idx"], state["row_idx"] = 0, 0
    slot["gen"] = entry["row_factory"](0, 0)
    logger.info(
        f"source {entry['key']}: pass {epoch + 2}, at {state['tokens']:,} of "
        f"{slot['target']:,} target tokens. The rows repeat; their conditions and distractors "
        f"do not."
    )
    return True


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
    many_distractors: Tuple[int, int] = (16, 32),
    chunk_tokens: int = 128,
    webtext_held_tokens: int = 384,
    reservoir_size: int = 4096,
    render_batch: int = 512,
    val_fraction: float = 0.01,
    seed: int = 42,
    checkpoint_docs: int = 2000,
    split_prefix: str = "evidence",
    holdout_hashes: Optional[set] = None,
    max_source_epochs: int = 4,
) -> dict:
    """Interleave sources, apply conditions, embed chunks, write both splits.

    Free of Hub calls so a test can drive it with synthetic in-memory sources, exactly as
    ``prepare_sft_data.build_corpus`` and ``prepare_data.run_phase`` are.

    Args:
        source_entries: ``[{"key", "weight", "render", "condition_weights", "row_factory",
            "holdout"}]``.
        target_tokens: PROMPT tokens across both splits. Evidence tokens are counted separately.
        embedder: supplies the chunk vectors; any object with ``.encode(list[str]) -> [N, 384]``.
        max_evidence_tokens: rows whose evidence exceeds this are dropped rather than truncated --
            a truncated buffer silently removes the gold chunk some of the time, which would
            mislabel the condition rather than shorten it.
        num_distractors: how many distractors ``mixed`` and ``distractors`` draw when the row does
            not ship its own.
        many_distractors: ``(min, max)`` distractor count sampled per row for the ``many`` condition.
        webtext_held_tokens: fixed size of the span a web text document gives up as evidence (see
            ``split_webtext_document``), rather than a fraction of the document.
        render_batch: rows rendered before the tokenizer and the external embedder are called. Both
            are near flat in batch size and were the whole cost at one row per call.
        holdout_hashes: pretraining conversation hashes to exclude from sources marked
            ``entry["holdout"]`` (smoltalk2 only, currently) -- see ``main``'s manifest check.

    Returns:
        The final resume state, with per-source realized counts and per-condition document counts.
    """
    train_split, val_split = f"{split_prefix}_train", f"{split_prefix}_val"
    holdout_hashes = holdout_hashes or set()
    state = load_state(state_path)
    state.setdefault("sources", {})
    state.setdefault("splits", {})
    state.setdefault("conditions", {c: 0 for c in CONDITIONS})
    # how many rows carry a real answer rather than a refusal, the second axis of the groundedness
    # label. Read next to the condition counts: `gold` holding fewer answerable rows than it has
    # documents is SQuAD v2's natively unanswerable share showing through, which is expected
    state.setdefault("answerable_docs", 0)
    state.setdefault("skipped", {})
    for split in (train_split, val_split):
        state["splits"].setdefault(split, {})
    for entry in source_entries:
        state["sources"].setdefault(
            entry["key"], {"file_idx": 0, "row_idx": 0, "tokens": 0, "ev_tokens": 0,
                           "docs": 0, "done": False, "epoch": 0, "tokens_at_epoch_start": 0},
        )
        state["skipped"].setdefault(
            entry["key"], {"too_long": 0, "unrenderable": 0, "no_evidence": 0, "holdout": 0},
        )

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
                if _start_next_epoch(slot, max_source_epochs):
                    continue
                slot["state"]["done"] = True
                candidates = live()
                continue
            # advanced only after the row is consumed, never at pick time -- the same bug
            # prepare_data.run_phase documents, where a picked-but-uncommitted row is lost on resume
            slot["state"]["file_idx"] = file_idx
            slot["state"]["row_idx"] = row_idx + 1

            rendered = _render_row(
                raw, slot["entry"], template, state, reservoirs[pick], cond_rng, chunk_rng,
                num_distractors=num_distractors, many_distractors=many_distractors,
                chunk_tokens=chunk_tokens, max_doc_tokens=max_doc_tokens,
                webtext_held_tokens=webtext_held_tokens, holdout_hashes=holdout_hashes,
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
                                 rendered["ev_chunk"], rendered["keys"], rendered["ev_gold"],
                                 CONDITIONS.index(rendered["condition"]),
                                 int(rendered["answerable"]))
            source_state = state["sources"][rendered["source"]]
            source_state["tokens"] += len(rendered["ids"])
            source_state["ev_tokens"] += len(rendered["ev_ids"])
            source_state["docs"] += 1
            state["conditions"][rendered["condition"]] = (
                state["conditions"].get(rendered["condition"], 0) + 1
            )
            state["answerable_docs"] += int(rendered["answerable"])
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
                num_distractors, many_distractors, chunk_tokens, max_doc_tokens,
                webtext_held_tokens, holdout_hashes):
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
        if entry.get("holdout") and holdout_hashes:
            # reproduces prepare_data.py's holdout hash byte for byte (import, not reimplement),
            # so a conversation phase-2 pretraining already saw does not also leak into replay
            msgs = raw.get("messages")
            if msgs is not None and doc_hash(render_pretrain_chat(msgs)) in holdout_hashes:
                state["skipped"][key]["holdout"] += 1
                return None
        # a replay row's target is the assistant's real reply, never a refusal, so it is answerable
        # in the only sense this flag carries. It still labels 0 for groundedness, because that
        # label is "gold present AND answerable" and a replay row retrieved nothing at all
        return {"conversation": conversation, "chunks": [], "gold_flags": [],
                "condition": "none", "key": key, "answerable": True}

    condition = pick_condition(entry["condition_weights"], cond_rng)

    if render == "webtext":
        pieces = split_webtext_document(raw, chunk_rng, chunk_tokens, max_doc_tokens,
                                        held_tokens=webtext_held_tokens)
        if pieces is None:
            state["skipped"][key]["unrenderable"] += 1
            return None
        prefix, cont, held = pieces
        # BOS by hand: this row bypasses the chat template (there is no conversation here, it is
        # plain language modeling), and the pretraining dataset prepends one to every document
        ids = [template.bos_id] + prefix + cont
        mask = [0] * (len(prefix) + 1) + [1] * len(cont)
        gold_texts = [template.tokenizer.decode(c, skip_special_tokens=True) for c in held]
        chunk_texts, gold_flags, _, answerable = apply_condition(
            EvidenceRow(gold=gold_texts, lm=True), condition, reservoir, cond_rng, num_distractors,
            many_distractors=many_distractors,
        )
        reservoir.extend(gold_texts)
        record = {"conversation": None, "ids": ids, "mask": mask, "chunks": chunk_texts,
                  "gold_flags": gold_flags, "condition": condition, "key": key,
                  "answerable": answerable}
    else:
        row = (render_squad_row if render == "squad_v2" else render_hotpot_row)(raw)
        if row is None:
            state["skipped"][key]["unrenderable"] += 1
            return None
        chunk_texts, gold_flags, answer, answerable = apply_condition(
            row, condition, reservoir, cond_rng, num_distractors, many_distractors=many_distractors,
        )
        reservoir.extend(row.gold)
        reservoir.extend(row.near)
        record = {
            "conversation": [
                {"role": "user", "content": evidence_prompt(row.question)},
                {"role": "assistant", "content": answer},
            ],
            "chunks": chunk_texts, "gold_flags": gold_flags, "condition": condition, "key": key,
            "answerable": answerable,
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
        ev_ids, ev_chunk, kept, kept_gold = [], [], [], []
        for text, gold, piece in zip(record["chunks"], record["gold_flags"], pieces[start:end]):
            if not piece:
                continue
            ev_chunk.extend([len(kept)] * len(piece))
            ev_ids.extend(piece)
            kept.append(text)
            kept_gold.append(gold)
        if len(ev_ids) > max_evidence_tokens:
            # dropped, never truncated: truncation removes whichever chunk happened to land last,
            # which is the gold one a third of the time under `mixed` -- that mislabels the
            # condition rather than shortening it
            state["skipped"][record["key"]]["too_long"] += 1
            record["dropped"] = True
            continue
        record["ev_ids"], record["ev_chunk"] = ev_ids, ev_chunk
        record["chunks"], record["ev_gold"] = kept, kept_gold


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


# --------------------------------------------------------------------------------- held-out splits

# the fixed split's conditions, in the order a question's four rows are written
FIXED_CONDITIONS = ("gold", "mixed", "distractors", "none")


def _heldout_record(row: EvidenceRow, key: str, condition: str, chunks: List[str],
                    flags: List[bool], answer: str, answerable: bool) -> dict:
    return {
        "conversation": [
            {"role": "user", "content": evidence_prompt(row.question)},
            {"role": "assistant", "content": answer},
        ],
        "chunks": chunks, "gold_flags": flags, "condition": condition, "key": key,
        "answerable": answerable, "source": key,
    }


def _drop_reason(record: dict, max_doc_tokens: int) -> Optional[str]:
    """Why ``_encode_batch`` dropped a record, or a condition mislabel it would have let through."""
    if record.get("dropped"):
        if "ids" not in record:
            return "unrenderable"
        return "prompt_too_long" if len(record["ids"]) > max_doc_tokens else "evidence_too_long"
    if record["condition"] != "none" and not record["ev_ids"]:
        # the chunks were promised and vanished (tokenized to nothing): writing it would relabel
        # the condition as `none`, whose abstention target differs
        return "no_evidence"
    return None


def _new_split_stats() -> dict:
    return {"sources": {}, "docs": 0, "prompt_tokens": 0, "ev_tokens": 0, "chunks": 0}


def _source_stats(split_stats: dict, key: str) -> dict:
    return split_stats["sources"].setdefault(key, {
        "docs": 0, "questions": 0, "prompt_tokens": 0, "ev_tokens": 0, "answerable": 0,
        "conditions": {c: 0 for c in CONDITIONS}, "drops": {},
    })


def _write_records(records: List[dict], writer: EvidenceWriter, split_stats: dict) -> None:
    for record in records:
        if record.get("dropped"):
            continue
        writer.write(record["ids"], record["mask"], record["ev_ids"], record["ev_chunk"],
                     record["keys"], record["ev_gold"], CONDITIONS.index(record["condition"]),
                     int(record["answerable"]))
        s = _source_stats(split_stats, record["source"])
        s["docs"] += 1
        s["prompt_tokens"] += len(record["ids"])
        s["ev_tokens"] += len(record["ev_ids"])
        s["answerable"] += int(record["answerable"])
        s["conditions"][record["condition"]] += 1
        split_stats["docs"] += 1
        split_stats["prompt_tokens"] += len(record["ids"])
        split_stats["ev_tokens"] += len(record["ev_ids"])
        split_stats["chunks"] += len(record["ev_gold"])


def build_heldout(
    sources: List[dict],
    template: ChatTemplate,
    embedder: ChunkEmbedder,
    data_dir: str,
    max_doc_tokens: int = 4094,
    max_evidence_tokens: int = 1536,
    num_distractors: int = 3,
    many_distractors: Tuple[int, int] = (16, 32),
    render_batch: int = 512,
    seed: int = 42,
    split_prefix: str = "evidence",
    max_questions: Optional[int] = None,
) -> dict:
    """Write the two held-out splits, ``{prefix}_dev`` and ``{prefix}_fixed``.

    Free of Hub calls, like ``build_corpus``, so a test can drive it with synthetic rows.

    ``_dev`` is the held-out copy of the training objective: every usable row once, one condition
    drawn from ``QA_CONDITIONS``, targets exactly as ``apply_condition`` picks them.

    ``_fixed`` is for reading what evidence is worth: each natively answerable question is written
    four times in a row, under ``gold``, ``mixed``, ``distractors``, ``none``, with the REAL answer
    as the target every time. Teacher forcing one fixed target under every condition makes
    CE(none) - CE(gold) a clean measurement, which the per-condition CE of ``_dev`` is not (there an
    answer is compared against a refusal). If any of a question's four rows cannot be written the
    other three are dropped too, so every group stays complete and any prefix stays paired.

    Distractors come from a pool of the split's own passages, deduplicated by text and never
    containing the row's own gold (see ``DistractorPool``).

    Args:
        sources: ``[{"key", "render", "rows"}]``. ``rows`` are raw dataset rows for ``render``
            (``squad_v2`` or ``hotpot_qa``) or already-built ``EvidenceRow`` objects.
        embedder: any object with ``.encode(list[str]) -> [N, 384]``.
        max_questions: cap on usable questions per source, or None for all.

    Returns:
        ``{"splits": {name: stats}, "render_dropped": {key: n}, "gold_rejected": {key: n},
        "pool_sizes": {key: n}}``, where a split's stats hold per-source docs, prompt and evidence
        tokens, per-condition counts, answerable counts and drop reasons.
    """
    dev_split, fixed_split = f"{split_prefix}_dev", f"{split_prefix}_fixed"
    render_dropped, pools, combined = {}, {}, []
    for spec in sources:
        key, rows = spec["key"], []
        render = {"squad_v2": render_squad_row, "hotpot_qa": render_hotpot_row}.get(spec["render"])
        render_dropped[key] = 0
        for raw in spec["rows"]:
            row = raw if isinstance(raw, EvidenceRow) else render(raw)
            if row is None:
                render_dropped[key] += 1
            else:
                rows.append(row)
        if max_questions is not None:
            rows = rows[:max_questions]
        texts = []
        for row in rows:
            texts.extend(row.gold)
            texts.extend(row.near)
        pools[key] = DistractorPool(texts)
        combined.extend((key, row) for row in rows)
    # one seeded shuffle across sources: the trainer's eval reads only the first N batches in
    # on-disk order, so any prefix has to be a representative sample of both sources
    random.Random(seed).shuffle(combined)

    cond_rng, draw_rng = random.Random(seed + 1), random.Random(seed + 2)
    scratch = {"skipped": {s["key"]: {"too_long": 0, "unrenderable": 0} for s in sources}}
    stats = {dev_split: _new_split_stats(), fixed_split: _new_split_stats()}
    writers = {s: EvidenceWriter(data_dir, s, {}) for s in (dev_split, fixed_split)}

    def condition_record(key, row, condition, answer_override=None):
        chunks, flags, answer, answerable = apply_condition(
            row, condition, deque(), draw_rng, num_distractors,
            many_distractors=many_distractors, pool=pools[key],
        )
        if answer_override is not None:
            answer, answerable = answer_override, True
        return _heldout_record(row, key, condition, chunks, flags, answer, answerable)

    def finish(records, group):
        """Encode, decide drops (whole groups of ``group`` rows), embed."""
        _encode_batch(records, template, max_doc_tokens, max_evidence_tokens, scratch)
        reasons = []
        for start in range(0, len(records), group):
            members = records[start:start + group]
            found = [r for r in (_drop_reason(m, max_doc_tokens) for m in members) if r]
            reasons.append(found[0] if found else None)
            if found:
                for m in members:
                    m["dropped"] = True
        _embed_batch(records, embedder)
        return reasons

    for start in range(0, len(combined), render_batch):
        records = []
        for key, row in combined[start:start + render_batch]:
            condition = pick_condition(QA_CONDITIONS, cond_rng)
            records.append(condition_record(key, row, condition))
        reasons = finish(records, 1)
        for record, reason in zip(records, reasons):
            s = _source_stats(stats[dev_split], record["source"])
            s["questions"] += 1
            if reason:
                s["drops"][reason] = s["drops"].get(reason, 0) + 1
        _write_records(records, writers[dev_split], stats[dev_split])

    answerable = [(k, r) for k, r in combined if r.answer]
    step = max(1, render_batch // len(FIXED_CONDITIONS))
    for start in range(0, len(answerable), step):
        records = []
        for key, row in answerable[start:start + step]:
            for condition in FIXED_CONDITIONS:
                records.append(condition_record(key, row, condition, answer_override=row.answer))
        reasons = finish(records, len(FIXED_CONDITIONS))
        for q, reason in enumerate(reasons):
            group = records[q * len(FIXED_CONDITIONS):(q + 1) * len(FIXED_CONDITIONS)]
            s = _source_stats(stats[fixed_split], group[0]["source"])
            s["questions"] += 1
            if reason:
                s["drops"][reason] = s["drops"].get(reason, 0) + 1
            else:
                assert all(m["ids"] == group[0]["ids"] and m["mask"] == group[0]["mask"]
                           for m in group), "a fixed group's prompts differ"
        _write_records(records, writers[fixed_split], stats[fixed_split])

    for writer in writers.values():
        writer.sync()
        writer.close()
    return {
        "splits": stats, "render_dropped": render_dropped,
        "gold_rejected": {k: p.gold_rejected for k, p in pools.items()},
        "pool_sizes": {k: len(p.texts) for k, p in pools.items()},
    }


def log_heldout_report(result: dict) -> None:
    logger.info(f"  unrenderable source rows: {result['render_dropped']}")
    logger.info(f"  distractor pools: {result['pool_sizes']} texts; candidate draws rejected for "
                f"being the row's own gold: {result['gold_rejected']}")
    for split, sp in result["splits"].items():
        logger.info(
            f"  {split}: {sp['docs']:,} docs, {sp['prompt_tokens']:,} prompt tokens, "
            f"{sp['ev_tokens']:,} evidence tokens (ratio {sp['ev_tokens'] / max(1, sp['prompt_tokens']):.2f}), "
            f"{sp['chunks']:,} chunks"
        )
        for key, s in sp["sources"].items():
            logger.info(
                f"    {key}: {s['docs']:,} docs ({s['questions']:,} questions seen), "
                f"{s['prompt_tokens']:,} prompt, {s['ev_tokens']:,} evidence "
                f"(ratio {s['ev_tokens'] / max(1, s['prompt_tokens']):.2f}), "
                f"answerable {s['answerable']:,}, conditions {s['conditions']}, "
                f"dropped {s['drops'] or 0}" + (" (questions, all four rows)" if "fixed" in split else "")
            )


def _delete_split(data_dir: str, split: str) -> None:
    for suffix in EVIDENCE_SUFFIXES:
        path = os.path.join(data_dir, f"{split}.{suffix}")
        if os.path.exists(path):
            os.remove(path)


def _main_heldout(args) -> None:
    splits = [f"{args.split_prefix}_dev", f"{args.split_prefix}_fixed"]
    for split in splits:
        if os.path.exists(os.path.join(args.data_dir, f"{split}.bin")):
            if not args.overwrite:
                raise SystemExit(f"{split}.bin already exists in {args.data_dir}; pass --overwrite")
            _delete_split(args.data_dir, split)

    scratch_dir = os.path.join(args.data_dir, "_evidence_scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    template = ChatTemplate(tokenizer)
    hf_token = get_hf_token()
    hf_api = HfApi(token=hf_token)

    sources = []
    for key, render, repo_id, filename in (
        ("squad_v2", "squad_v2", "rajpurkar/squad_v2", "squad_v2/validation-00000-of-00001.parquet"),
        ("hotpot_qa", "hotpot_qa", "hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet"),
    ):
        info = hf_api.dataset_info(repo_id)
        logger.info(f"source {key} dev: {filename} (revision {info.sha[:10]})")
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset",
                               local_dir=scratch_dir, token=hf_token, revision=info.sha)
        rows = pd.read_parquet(path, engine="pyarrow").to_dict("records")
        try:
            os.remove(path)
        except OSError:
            pass
        sources.append({"key": key, "render": render, "rows": rows})

    logger.info(f"loading the external embedder ({ChunkEmbedder.REPO})")
    embedder = ChunkEmbedder(device=args.device)
    t0 = time.time()
    result = build_heldout(
        sources, template, embedder, args.data_dir, max_doc_tokens=args.max_doc_tokens,
        max_evidence_tokens=args.max_evidence_tokens, num_distractors=args.num_distractors,
        many_distractors=tuple(args.many_distractors), render_batch=args.render_batch,
        seed=args.seed, split_prefix=args.split_prefix, max_questions=args.heldout_max_questions,
    )
    logger.info(f"=== held-out splits built in {(time.time() - t0) / 60:.1f} min ===")
    log_heldout_report(result)


def _shuffled_shard_order(files: List[str], seed: int, source_key: str) -> List[str]:
    """Reproducibly shuffle a source's sorted shard list, from the seed alone.

    Consuming ``sorted()`` order in file-name order handed the first N shards of a source to
    whatever happened to be alphabetically first -- for smoltalk2 that is a 64k-context split, so a
    partial build's kept rows were the shortest conversations of a long-context split rather than a
    representative sample. Shuffling fixes that, but only if it does not itself introduce a new
    resume hazard: the resume state records a ``file_idx`` INTO this list, so the order has to be a
    pure function of ``(seed, source_key)`` and the sorted input -- never of how many shards a prior,
    interrupted run already consumed -- or a resumed run would silently point ``file_idx`` at a
    different file than the one it left off on.
    """
    digest = hashlib.sha1(f"{source_key}:shard-order".encode("utf-8")).digest()
    rng = random.Random(seed ^ int.from_bytes(digest[:4], "big"))
    shuffled = list(files)
    rng.shuffle(shuffled)
    return shuffled


def main():
    parser = argparse.ArgumentParser(description="Build the evidence-conditioned corpus")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data", "prepared"))
    parser.add_argument("--target-tokens", type=int, default=150_000_000,
                        help="PROMPT tokens across both splits; evidence tokens are reported "
                             "separately (see the module docstring)")
    parser.add_argument("--max-doc-tokens", type=int, default=4094)
    parser.add_argument("--max-evidence-tokens", type=int, default=1536)
    parser.add_argument("--num-distractors", type=int, default=3)
    parser.add_argument("--many-distractors", type=int, nargs=2, default=[16, 32],
                        metavar=("MIN", "MAX"),
                        help="distractor count range sampled per row for the 'many' condition")
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--webtext-held-tokens", type=int, default=384,
                        help="fixed size of the span a web text document gives up as evidence, "
                             "clamped to the document (replaces a 25%% held-out fraction, which let "
                             "a long document's held span -- and the prompt either side of it -- "
                             "grow without bound)")
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--checkpoint-docs", type=int, default=2000)
    parser.add_argument("--render-batch", type=int, default=512,
                        help="rows per tokenizer/embedder call; the embedder is a GPU forward and "
                             "runs at roughly flat cost up to a few hundred chunks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-epochs", type=int, default=4,
                        help="how many passes a `repeat` source (the two QA sets) may make before "
                             "it stops, whether or not it reached its token target. They are small "
                             "and finite -- one pass leaves QA at ~10%% of corpus tokens -- and each "
                             "pass redraws every row's condition and distractors, so the repeat is "
                             "a new task on a seen question. 1 disables repeating")
    parser.add_argument("--split-prefix", default="evidence")
    parser.add_argument("--webtext-phase", default="ir",
                        help="which prepared {phase}.bin/.idx the web text arm reads")
    parser.add_argument("--no-webtext", action="store_true",
                        help="skip the web text arm (QA and replay only)")
    parser.add_argument("--ignore-holdout", action="store_true",
                        help="build even if manifest.json has no smoltalk2 holdout hashes (unsafe: "
                             "phase-2 pretraining conversations may leak into the replay arm)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heldout", action="store_true",
                        help="build {prefix}_dev and {prefix}_fixed from SQuAD v2 dev and HotpotQA "
                             "dev instead of the training corpus (see the module docstring)")
    parser.add_argument("--overwrite", action="store_true",
                        help="with --heldout, replace existing held-out splits")
    parser.add_argument("--heldout-max-questions", type=int, default=None,
                        help="with --heldout, cap usable questions per source (default: all)")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    if args.heldout:
        _main_heldout(args)
        return
    scratch_dir = os.path.join(args.data_dir, "_evidence_scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    template = ChatTemplate(tokenizer)
    hf_token = get_hf_token()

    # same manifest and hash function prepare_sft_data.py checks the smoltalk2 replay/SFT overlap
    # against -- imported rather than reimplemented, so a rendering drift can't silently exclude
    # nothing (see render_pretrain_chat's docstring)
    manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    holdout_hashes = set(manifest.get("data_prep", {}).get("smoltalk2_holdout_hashes", []))
    if not holdout_hashes and not args.ignore_holdout:
        raise SystemExit(
            "manifest.json has no data_prep.smoltalk2_holdout_hashes -- pull it from the "
            "pretraining mirror repo (see scripts/prepare_sft_data.py --pull-manifest; "
            "manifest.json is gitignored, so a fresh clone never has it), or pass --ignore-holdout "
            "to build the replay arm without the exclusion."
        )
    logger.info(f"smoltalk2 holdout: {len(holdout_hashes):,} conversations excluded")

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
            # shuffled AFTER sorting, not instead of it: sorting first makes the shuffle a pure
            # function of the file set (independent of whatever order the Hub API happened to list
            # them in), and the shuffle itself is what stops a partial build from only ever seeing
            # the alphabetically first shard
            all_files = _shuffled_shard_order(all_files, args.seed, spec.key)
            logger.info(f"source {spec.key}: {len(all_files)} files (revision {info.sha[:10]})")
            factory = hub_row_factory(spec, all_files, scratch_dir, hf_token, args.seed, info.sha)
        entries.append({
            "key": spec.key, "weight": spec.weight, "render": spec.render,
            "condition_weights": spec.condition_weights, "row_factory": factory, "qa": spec.qa,
            "holdout": spec.holdout, "repeat": spec.repeat,
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
        num_distractors=args.num_distractors, many_distractors=tuple(args.many_distractors),
        chunk_tokens=args.chunk_tokens, webtext_held_tokens=args.webtext_held_tokens,
        render_batch=args.render_batch, val_fraction=args.val_fraction, seed=args.seed,
        checkpoint_docs=args.checkpoint_docs, split_prefix=args.split_prefix,
        holdout_hashes=holdout_hashes, max_source_epochs=args.max_source_epochs,
    )
    elapsed = time.time() - t0

    logger.info(f"=== built in {elapsed / 60:.1f} min ===")
    total_prompt = sum(sp["tokens_written"] for sp in final["splits"].values())
    total_docs = sum(final["sources"][e["key"]]["docs"] for e in entries)
    replay_tokens = 0
    for e in entries:
        s = final["sources"][e["key"]]
        target = int(args.target_tokens * e["weight"])
        token_share = s["tokens"] / max(1, total_prompt)
        # the conversation-count share, not the token share, is what a source actually gets under
        # per-conversation loss weighting (every row's gradient is 1/its own supervised tokens, so a
        # source's pull on the model is its share of ROWS, not its share of tokens) -- printing both
        # next to each other is what makes a token-weighted mix that is a conversation-weighted
        # trap visible at build time instead of after training
        conv_share = s["docs"] / max(1, total_docs)
        passes = s.get("epoch", 0) + 1
        logger.info(
            f"  {e['key']}: {s['tokens']:,}/{target:,} prompt tokens ({token_share:.1%} of corpus "
            f"tokens), {s['ev_tokens']:,} evidence, {s['docs']:,} docs ({conv_share:.1%} of corpus "
            f"conversations)"
            + (f", {passes} passes" if passes > 1 else "")
            + f", skipped {final['skipped'][e['key']]}"
        )
        if e["render"] == "messages":
            replay_tokens += s["tokens"]
    logger.info(f"  conditions: {final['conditions']}")
    answerable = final.get("answerable_docs", 0)
    logger.info(
        f"  answerable rows: {answerable:,}/{total_docs:,} ({answerable / max(1, total_docs):.1%}) "
        f"-- the rest take an abstention target, and the difference between that share and the "
        f"distractors+none share is SQuAD v2's natively unanswerable rows"
    )
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

"""Build the fact injection corpora: one shared document stream, three ways to weigh it.

The experiment asks whether facts can be kept out of the weights and in a store. Every arm trains on
the same documents in the same order: web text windows (filler) and fictional biographies
(``modules/data/biographies.py``) rendered 1, 10, 100 or 1000 times per person. The arms differ only
in how a fact span counts and in what the model is handed to read.

| arm | files | mask | evidence |
|---|---|---|---|
| ``full`` | bin idx mask factspan | 1 on every token after BOS | none |
| ``masked`` | bin idx mask factspan | 0 on every token overlapping an attribute value | none |
| ``retrieval`` | the eleven evidence files and factspan | 1 on language tokens and supported spans | the person's card among distractors |

``full`` and ``masked`` share their ``.bin``, ``.idx`` and ``.factspan`` bytes exactly. ``retrieval``
equals them except where a value is swapped or a name is replaced by a typed placeholder.

**No tagger is needed.** The generator writes every fact span itself and knows its exact character
range, so the loss mask is exact by construction. Background filler text is not tagged at all and is
plain cross entropy in every arm: its own facts are not the experiment.

**How the span weight reaches the trainer.** Through the existing ``.mask`` (uint8, 0 or 1, one entry
per token). Nothing in the trainer or the datasets changes. The weights are binary, binary is what
``.mask`` is, and per-conversation weighting is off for these runs, so ``labels != -100`` is the whole
weight. The multi-token prediction head builds its labels by shifting the main ones, and ``-100`` comes
from mask 0, so its targets carry the same span weight: the weight is complete, not only on the
next-token head. A fractional weight would need a float sidecar multiplied into ``loss_weights``.

Per bio document in the ``retrieval`` arm, in this order:

1. **Buffer.** With probability ``1 - gold_drop_rate`` the person's card plus ``--distractors`` other
   people's cards (half drawn from the nearest same-tier people by key, half uniform); otherwise
   ``distractors + 1`` cards and no gold. Order shuffled, ``.evgold`` marks the person's card.
2. **Support.** An attribute span is supported when its exact value string occurs in a visible card.
   The share supported only by a non-gold card (a value collision) is reported.
3. **Swaps.** With the gold card present, each supported span is swapped with probability
   ``swap_rate`` for another value of the same pool, in the target and in this document's copy of the
   gold card. The card's key stays the canonical one (the selector chooses by person, the reader
   reads the swapped text). With the gold card absent there is no copy to swap, so nothing is.
4. **Anonymization.** With the gold card absent, with probability ``anon_rate`` every mention of the
   name becomes a typed placeholder (``Person KTV``), ``.factspan`` 7. Values are head entities, so
   only the name is anonymized.
5. **Placeholders with the gold card.** With the gold card present, with probability
   ``placeholder_rate`` every mention of the name becomes a typed placeholder in the document and in
   this document's copy of the gold card, and each distractor card is rendered with its own distinct
   placeholder. The keys stay canonical and swaps apply on top; ``.factspan`` is 7 on those mentions.
   The draws come from their own stream, so the buffer, the gold flag and the swaps equal those of a
   build at rate 0.
6. **Mask.** 1 on language tokens and supported spans (swapped included), 0 on unsupported spans.

``.factspan`` is one uint8 per ``.bin`` token: 0 none, 1 to 5 the attribute (index in ``ATTRIBUTES``
plus one), 6 a name mention, 7 a placeholder. It is for evals and build metrics; no trainer reads it.

The build rebuilds a split from scratch; it has no resume. The facts, the pools, the biography store
and the validation split are written once and kept when they already exist, so a second build with
other swap rates (``--suffix``) shares them.

Usage:

    python scripts/prepare_injection_data.py --out-dir data/prepared_inject --target-tokens 300000000 \\
        --filler-phases ir,phase1,phase2 --seed 42 --device cuda
"""
import os
import sys
import json
import time
import zlib
import string
import random
import hashlib
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformers import AutoTokenizer

from modules.data import biographies as bio
from scripts.prepare_evidence_data import (
    CONDITIONS, EMBED_DIM, EVIDENCE_SUFFIXES, ChunkEmbedder, EvidenceWriter,
)
from utils import BASE_DIR, TOKENIZER_DIR, logger

ARMS = ("full", "masked", "retrieval")
CODE_NAME, CODE_PLACEHOLDER = 6, 7
FILLER, BIO = 0, 1
MIN_WINDOW = 32


def split_name(arm: str, suffix: str) -> str:
    """The train split of an arm, e.g. ``inject_retrieval_s15a50_train``.

    Args:
        arm: the arm name, e.g. ``"retrieval"``.
        suffix: the run suffix, empty for none.
    """
    return f"inject_{arm}{'_' + suffix if suffix else ''}_train"


def _rng(seed: int, salt: str, *parts) -> random.Random:
    digest = hashlib.sha1(f"{salt}:{seed}:{':'.join(map(str, parts))}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ------------------------------------------------------------------------------------- tokenizing

def tokenize_with_codes(tokenizer, texts: Sequence[str], spans: Sequence[List[Tuple[int, int, str]]],
                        name_codes: Sequence[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Token ids and a factspan code per token, by character overlap with the spans.

    A token belongs to a span when its character range overlaps the span's, so a token carrying a
    leading space is in the span.

    Args:
        tokenizer: a fast tokenizer (offsets are needed).
        texts: the documents.
        spans: per document, ``(char_start, char_end, label)`` sorted by start.
        name_codes: per document, the code written on name mentions (6, or 7 for a placeholder).

    Returns:
        Per document ``(ids uint16, codes uint8)``, no BOS or EOS.
    """
    encoded = tokenizer(list(texts), add_special_tokens=False, return_offsets_mapping=True)
    attribute_code = {a: i + 1 for i, a in enumerate(bio.ATTRIBUTES)}
    out = []
    for ids, offsets, doc_spans, name_code in zip(encoded["input_ids"], encoded["offset_mapping"],
                                                  spans, name_codes):
        codes = np.zeros(len(ids), dtype=np.uint8)
        i = 0
        for t, (ts, te) in enumerate(offsets):
            while i < len(doc_spans) and doc_spans[i][1] <= ts:
                i += 1
            if i < len(doc_spans) and doc_spans[i][0] < te and te > ts:
                label = doc_spans[i][2]
                codes[t] = name_code if label == "name" else attribute_code[label]
        out.append((np.asarray(ids, dtype=np.uint16), codes))
    return out


# ----------------------------------------------------------------------------------------- filler

def load_filler_windows(sources: Sequence[Tuple[str, str]], window: int, seed: int,
                        val_fraction: float) -> Tuple[List[np.memmap], np.ndarray, np.ndarray]:
    """Cut every document of the filler corpora into windows and set aside the validation ones.

    Args:
        sources: ``(bin_path, idx_path)`` of each prepared corpus.
        window: longest window in tokens; a document's last window is dropped under ``MIN_WINDOW``.
        seed: seeds the validation choice.
        val_fraction: share of windows set aside, chosen by a hash of the window itself so the choice
            never depends on the target size or the order.

    Returns:
        ``(token memmaps, train windows, val windows)`` where a window array is ``[N, 3]`` of
        ``(source, start, length)``.
    """
    memmaps, train, val = [], [], []
    cutoff = int(val_fraction * 1_000_000)
    for source, (bin_path, idx_path) in enumerate(sources):
        memmaps.append(np.memmap(bin_path, dtype=np.uint16, mode="r"))
        offsets = np.memmap(idx_path, dtype=np.uint64, mode="r")
        for d in range(len(offsets) - 1):
            start, end = int(offsets[d]), int(offsets[d + 1])
            for w in range(start, end, window):
                length = min(window, end - w)
                if length < MIN_WINDOW:
                    continue
                is_val = zlib.crc32(f"{seed}:{source}:{w}".encode()) % 1_000_000 < cutoff
                (val if is_val else train).append((source, w, length))
    return memmaps, np.asarray(train, dtype=np.int64).reshape(-1, 3), \
        np.asarray(val, dtype=np.int64).reshape(-1, 3)


def pick_filler(train: np.ndarray, need_tokens: int, seed: int) -> Tuple[np.ndarray, float]:
    """Windows in a seeded shuffled order until ``need_tokens`` are covered, cycling when short.

    Args:
        train: train windows, ``[N, 3]`` of ``(source, start, length)``.
        need_tokens: tokens to cover, BOS and EOS included; zero or less returns no windows.
        seed: the shuffle seed.

    Returns:
        ``(chosen windows [M, 3], repeat factor)`` where the factor is chosen tokens over the tokens
        of the distinct windows used (1.0 means no repeat).
    """
    assert len(train), "no filler windows"
    if need_tokens <= 0:
        return np.zeros((0, 3), dtype=np.int64), 1.0
    rng = np.random.default_rng(seed)
    available = int((train[:, 2] + 2).sum())               # BOS and EOS included
    chosen, total = [], 0
    while total < need_tokens:
        order = rng.permutation(len(train))
        cumulative = total + np.cumsum(train[order, 2] + 2)
        take = min(int(np.searchsorted(cumulative, need_tokens)) + 1, len(order))
        chosen.append(train[order[:take]])
        total = int(cumulative[take - 1])
    return np.concatenate(chosen), float(total / min(total, available))


# ------------------------------------------------------------------------------------- the writers

def _check_free(data_dir: str, split: str, overwrite: bool) -> None:
    for suffix in EVIDENCE_SUFFIXES + ("factspan",):
        path = os.path.join(data_dir, f"{split}.{suffix}")
        if os.path.exists(path):
            if not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to rebuild {split}")
            os.remove(path)


class PlainWriter:
    """``bin idx mask factspan`` for one split, appended document by document."""

    def __init__(self, data_dir: str, split: str):
        self.files = {s: open(os.path.join(data_dir, f"{split}.{s}"), "wb")
                      for s in ("bin", "idx", "mask", "factspan")}
        self.files["idx"].write(np.array([0], dtype=np.uint64).tobytes())
        self.tokens = 0
        self.docs = 0
        self.mask_zero = 0

    def write(self, ids: np.ndarray, mask: np.ndarray, codes: np.ndarray) -> None:
        assert len(ids) == len(mask) == len(codes)
        self.files["bin"].write(ids.astype(np.uint16).tobytes())
        self.files["mask"].write(mask.astype(np.uint8).tobytes())
        self.files["factspan"].write(codes.astype(np.uint8).tobytes())
        self.tokens += len(ids)
        self.docs += 1
        self.mask_zero += int((mask == 0).sum())
        self.files["idx"].write(np.array([self.tokens], dtype=np.uint64).tobytes())

    def close(self) -> None:
        for f in self.files.values():
            f.close()


class RetrievalWriter:
    """The eleven evidence files through ``EvidenceWriter`` plus ``factspan``."""

    def __init__(self, data_dir: str, split: str):
        self.inner = EvidenceWriter(data_dir, split, {})
        self.factspan = open(os.path.join(data_dir, f"{split}.factspan"), "wb")
        self.tokens = 0
        self.docs = 0
        self.mask_zero = 0
        self.ev_tokens = 0

    def write(self, ids, mask, codes, ev_ids, ev_chunk, keys, ev_gold, condition_idx: int) -> None:
        self.inner.write(ids, mask, ev_ids, ev_chunk, keys, ev_gold, condition_idx, 1)
        self.factspan.write(np.asarray(codes, dtype=np.uint8).tobytes())
        self.tokens += len(ids)
        self.docs += 1
        self.mask_zero += int((np.asarray(mask) == 0).sum())
        self.ev_tokens += len(ev_ids)

    def close(self) -> None:
        self.inner.sync()
        self.inner.close()
        self.factspan.close()


# ----------------------------------------------------------------------- the retrieval arm's logic

class RetrievalBuilder:
    """Per bio document decisions and renders for the retrieval arm.

    Args:
        people: every person; ``person_id`` is the index.
        pools: ``{attribute: [values]}``.
        keys: ``[N, 384]`` canonical card keys, row ``i`` for person ``i``.
        tokenizer: the model tokenizer.
        seed: the build seed; every decision is a function of ``(seed, person, exposure)`` and so does
            not depend on document order or on which arms are built.
        swap_rate: chance a supported span is swapped.
        anon_rate: chance a gold-less document anonymizes the name.
        gold_drop_rate: chance a document's buffer has no gold card.
        placeholder_rate: chance a document with the gold card renames every card and the document.
        distractors: distractor cards beside the gold (one more without it).
        neighbors: how many nearest same-tier people the near distractors are drawn from.
    """

    def __init__(self, people: List[bio.Person], pools: Dict[str, List[str]], keys: np.ndarray,
                 tokenizer, seed: int, swap_rate: float, anon_rate: float, gold_drop_rate: float,
                 distractors: int, neighbors: int = 8, *, placeholder_rate: float = 0.0):
        self.people, self.pools, self.keys = people, pools, np.asarray(keys, dtype=np.float32)
        self.tokenizer, self.seed = tokenizer, seed
        self.swap_rate, self.anon_rate, self.gold_drop_rate = swap_rate, anon_rate, gold_drop_rate
        self.placeholder_rate = placeholder_rate
        self.distractors = distractors
        self.card_ids = [np.asarray(x, dtype=np.uint16) for x in
                         tokenizer([p.store_chunk for p in people], add_special_tokens=False)["input_ids"]]
        self.near: Dict[int, List[int]] = {}
        for tier in sorted({p.tier for p in people}):
            members = np.array([p.person_id for p in people if p.tier == tier])
            if len(members) < 2:
                continue
            similarity = self.keys[members] @ self.keys[members].T
            np.fill_diagonal(similarity, -2.0)
            top = np.argsort(-similarity, axis=1)[:, :neighbors]
            for row, pid in enumerate(members):
                self.near[int(pid)] = [int(members[c]) for c in top[row][:len(members) - 1]]
        self.stats = {
            "docs": 0, "gold_in_buffer": 0, "spans": 0, "spans_supported": 0, "span_tokens": 0,
            "span_tokens_supported": 0, "swapped_spans": 0, "anonymized_docs": 0,
            "placeholder_docs": 0, "collision_spans": 0, "spans_in_goldless_docs": 0, "goldless_docs": 0,
            "ev_tokens_max": 0,
        }

    def _buffer(self, person: bio.Person, rng: random.Random, gold: bool) -> List[int]:
        k = self.distractors + (0 if gold else 1)
        near_pool = self.near.get(person.person_id, [])
        n_near = min((k + 1) // 2, len(near_pool))
        chosen = rng.sample(near_pool, n_near) if n_near else []
        while len(chosen) < k:
            other = rng.randrange(len(self.people))
            if other != person.person_id and other not in chosen:
                chosen.append(other)
        cards = ([person.person_id] if gold else []) + chosen
        rng.shuffle(cards)
        return cards

    def build(self, flat_docs: Sequence[Tuple[int, int]]) -> List[dict]:
        """Render a batch of ``(person_id, exposure)`` documents for the retrieval arm.

        Returns:
            Per document a dict with ``ids``, ``mask``, ``codes`` (BOS and EOS included), ``ev_ids``,
            ``ev_chunk``, ``keys``, ``ev_gold`` and ``condition_idx``.
        """
        specs = []
        for pid, j in flat_docs:
            person = self.people[pid]
            rng = _rng(self.seed, "c", pid, j)
            gold = rng.random() >= self.gold_drop_rate
            cards = self._buffer(person, rng, gold)
            texts = {c: self.people[c].store_chunk for c in cards}
            supported, collided = {}, {}
            for attribute in bio.ATTRIBUTES:
                value = person.attributes[attribute]
                by_gold = gold
                by_other = any(value in texts[c] for c in cards if c != pid)
                supported[attribute] = by_gold or by_other
                collided[attribute] = (not by_gold) and by_other
            overrides = {}
            if gold:
                for attribute in bio.ATTRIBUTES:
                    draw = rng.random()
                    if supported[attribute] and draw < self.swap_rate:
                        original = person.attributes[attribute]
                        substitute = original
                        while substitute == original:
                            substitute = rng.choice(self.pools[attribute])
                        overrides[attribute] = substitute
            placeholder = None
            if not gold and rng.random() < self.anon_rate:
                letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.choice((2, 3))))
                placeholder = f"Person {letters}"
            card_placeholders: Dict[int, str] = {}
            if gold:
                prng = _rng(self.seed, "p", pid, j)
                if prng.random() < self.placeholder_rate:
                    taken = set()
                    for card in [pid] + [c for c in cards if c != pid]:
                        while True:
                            letters = "".join(prng.choice(string.ascii_uppercase)
                                              for _ in range(prng.choice((2, 3))))
                            if letters not in taken:
                                break
                        taken.add(letters)
                        card_placeholders[card] = f"Person {letters}"
                    placeholder = card_placeholders[pid]
            text, spans = bio.render_bio(person, _rng(self.seed, "r", pid, j),
                                         name_override=placeholder, value_overrides=overrides or None)
            card_texts = {}
            if card_placeholders:
                for card, name in card_placeholders.items():
                    card_texts[card] = bio.render_store_chunk(
                        self.people[card], (overrides or None) if card == pid else None, name_override=name)
            elif overrides:
                card_texts[pid] = bio.render_store_chunk(person, overrides)
            specs.append({"pid": pid, "gold": gold, "cards": cards, "supported": supported,
                          "collided": collided, "overrides": overrides, "placeholder": placeholder,
                          "card_placeholders": card_placeholders, "text": text, "spans": spans,
                          "card_texts": card_texts})

        tokenized = tokenize_with_codes(
            self.tokenizer, [s["text"] for s in specs], [s["spans"] for s in specs],
            [CODE_PLACEHOLDER if s["placeholder"] else CODE_NAME for s in specs])
        rewritten = [(s, card, t) for s in specs for card, t in s["card_texts"].items()]
        rewritten_ids = (self.tokenizer([t for _, _, t in rewritten],
                                        add_special_tokens=False)["input_ids"] if rewritten else [])
        for spec in specs:
            spec["card_ids"] = {}
        for (spec, card, _), ids in zip(rewritten, rewritten_ids):
            spec["card_ids"][card] = np.asarray(ids, dtype=np.uint16)

        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        out = []
        for spec, (ids, codes) in zip(specs, tokenized):
            # indexed by factspan code: 1 to 5 are the attributes, 0, 6 and 7 are never a value
            supported_by_code = np.zeros(8, dtype=bool)
            supported_by_code[1:6] = [spec["supported"][a] for a in bio.ATTRIBUTES]
            is_value = (codes >= 1) & (codes <= 5)
            token_mask = np.ones(len(ids), dtype=np.uint8)
            token_mask[is_value & ~supported_by_code[codes]] = 0
            full_ids = np.concatenate(([bos], ids, [eos])).astype(np.uint16)
            full_mask = np.concatenate(([0], token_mask, [1])).astype(np.uint8)
            full_codes = np.concatenate(([0], codes, [0])).astype(np.uint8)

            ev_ids, ev_chunk = [], []
            for slot, card in enumerate(spec["cards"]):
                piece = spec["card_ids"].get(card, self.card_ids[card])
                ev_ids.append(piece)
                ev_chunk.append(np.full(len(piece), slot, dtype=np.uint16))
            ev_ids = np.concatenate(ev_ids)
            out.append({
                "ids": full_ids, "mask": full_mask, "codes": full_codes, "ev_ids": ev_ids,
                "ev_chunk": np.concatenate(ev_chunk), "keys": self.keys[spec["cards"]],
                "ev_gold": [int(c == spec["pid"]) for c in spec["cards"]],
                "condition_idx": CONDITIONS.index("mixed" if spec["gold"] else "distractors"),
            })

            s = self.stats
            n_spans = len(bio.ATTRIBUTES)
            s["docs"] += 1
            s["gold_in_buffer"] += int(spec["gold"])
            s["spans"] += n_spans
            s["spans_supported"] += sum(spec["supported"].values())
            s["span_tokens"] += int(is_value.sum())
            s["span_tokens_supported"] += int((is_value & supported_by_code[codes]).sum())
            s["swapped_spans"] += len(spec["overrides"])
            s["anonymized_docs"] += int(spec["placeholder"] is not None and not spec["gold"])
            s["placeholder_docs"] += int(bool(spec["card_placeholders"]))
            s["collision_spans"] += sum(spec["collided"].values())
            s["goldless_docs"] += int(not spec["gold"])
            s["spans_in_goldless_docs"] += 0 if spec["gold"] else n_spans
            s["ev_tokens_max"] = max(s["ev_tokens_max"], len(ev_ids))
        return out


# --------------------------------------------------------------------------------------- the build

def _write_json_atomic(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def build_injection(out_dir: str, tokenizer, embedder, *, persons_per_tier: Dict[int, int],
                    filler: Sequence[Tuple[str, str]], target_tokens: int, seq_length: int = 1024,
                    swap_rate: float = 0.15, anon_rate: float = 0.5, placeholder_rate: float = 0.0,
                    gold_drop_rate: float = 0.2, distractors: int = 3, seed: int = 42, arms: Sequence[str] = ARMS,
                    suffix: str = "", val_fraction: float = 0.005, store_dir: Optional[str] = None,
                    overwrite: bool = False, block: int = 4096) -> dict:
    """Write the arm splits, the shared validation split, the facts, the pools and the store.

    Free of the Hub, so a test can drive it with a fake embedder and a small synthetic filler.

    Args:
        out_dir: where the splits and sidecars go (``data/prepared_inject``).
        tokenizer: the model tokenizer (fast).
        embedder: anything with ``.encode(list[str]) -> [N, 384]``; used for the biography store and
            the card keys. May be None when no retrieval arm is built and the store exists.
        persons_per_tier: ``{tier: people}``.
        filler: ``(bin, idx)`` pairs of prepared corpora, raw documents without BOS or EOS.
        target_tokens: ``.bin`` tokens of one arm's train split, filler plus biographies.
        seq_length: row length the trainer packs to; filler windows leave margin for BOS, EOS and
            the multi-token prediction separators.
        swap_rate: see the module docstring.
        anon_rate: see the module docstring.
        placeholder_rate: see the module docstring.
        gold_drop_rate: see the module docstring.
        distractors: distractor cards beside a gold card.
        seed: seeds everything.
        arms: any of ``ARMS``.
        suffix: appended to the arm split names, so a sweep builds next to the first build.
        val_fraction: share of filler windows that go to validation and never to train.
        store_dir: the biography store directory, or None to skip it.
        overwrite: replace existing splits of these arms (and the shared files).
        block: documents rendered and written together.

    Returns:
        The build metrics, also merged into ``inject_build.json`` in ``out_dir``.
    """
    assert set(arms) <= set(ARMS), arms
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    window = min(1000, seq_length - 24)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id

    pools = bio.make_pools(seed)
    people = bio.make_people(pools, persons_per_tier, seed)
    for path, writer, payload, loader in (
        (os.path.join(out_dir, "inject_facts.jsonl"), bio.save_facts, people, bio.load_facts),
        (os.path.join(out_dir, "inject_pools.json"), bio.save_pools, pools, bio.load_pools),
    ):
        if os.path.exists(path) and not overwrite:
            if loader(path) != payload:
                raise SystemExit(f"{path} differs from this build's seed and tiers; pass --overwrite "
                                 f"or use another --out-dir")
        else:
            writer(path, payload)

    store_ready = store_dir is not None and os.path.exists(os.path.join(store_dir, "meta.json")) \
        and not overwrite
    keys = None
    if "retrieval" in arms or (store_dir is not None and not store_ready):
        assert embedder is not None, "the card keys need an embedder"
        keys = np.asarray(embedder.encode([p.store_chunk for p in people]), dtype=np.float32)
    if store_dir is not None and not store_ready:
        try:
            from modules.data.store import write_store
        except ImportError:
            logger.warning("modules.data.store is missing, so the biography store was NOT written")
        else:
            card_tokens = [len(x) for x in tokenizer([p.store_chunk for p in people],
                                                     add_special_tokens=False)["input_ids"]]
            write_store(
                store_dir,
                [{"chunk_id": p.person_id, "text": p.store_chunk, "source": "bios",
                  "doc_id": str(p.person_id), "title": p.name} for p in people],
                keys,
                {"embedder": ChunkEmbedder.REPO, "chunk_tokens": int(max(card_tokens)),
                 "n_chunks": len(people), "sources": {"bios": len(people)},
                 "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset_revisions": {}},
            )
            logger.info(f"wrote the biography store to {store_dir}")

    # every bio document, canonical render, once
    docs: List[Tuple[int, int]] = [(p.person_id, j) for p in people for j in range(p.tier)]
    texts, spans = [], []
    for pid, j in docs:
        text, doc_spans = bio.render_bio(people[pid], _rng(seed, "r", pid, j))
        texts.append(text)
        spans.append(doc_spans)
    canonical = []
    for lo in range(0, len(docs), 8192):
        canonical.extend(tokenize_with_codes(tokenizer, texts[lo:lo + 8192], spans[lo:lo + 8192],
                                             [CODE_NAME] * len(texts[lo:lo + 8192])))
    del texts, spans
    bio_tokens = sum(len(ids) + 2 for ids, _ in canonical)
    assert max(len(ids) for ids, _ in canonical) + 2 + 2 <= seq_length, "a biography overflows a row"
    exposures: Dict[int, int] = {}
    for pid, _ in docs:
        exposures[pid] = exposures.get(pid, 0) + 1
    assert all(exposures[p.person_id] == p.tier for p in people), "exposures differ from tiers"
    logger.info(f"{len(docs):,} biography documents, {bio_tokens:,} tokens "
                f"({bio_tokens / max(1, target_tokens):.1%} of the target)")

    memmaps, train_windows, val_windows = load_filler_windows(filler, window, seed, val_fraction)
    need = max(0, target_tokens - bio_tokens)
    chosen, repeat = pick_filler(train_windows, need, seed)
    logger.info(f"filler: {len(train_windows):,} train windows, {len(val_windows):,} validation, "
                f"{len(chosen):,} used for {need:,} tokens, repeat factor {repeat:.2f}x")

    kinds = np.concatenate([np.full(len(chosen), FILLER), np.full(len(docs), BIO)])
    indices = np.concatenate([np.arange(len(chosen)), np.arange(len(docs))])
    order = np.random.default_rng(seed + 1).permutation(len(kinds))
    kinds, indices = kinds[order], indices[order]

    names = {arm: split_name(arm, suffix) for arm in arms}
    writers: Dict[str, object] = {}
    for arm, split in names.items():
        _check_free(out_dir, split, overwrite)
        writers[arm] = RetrievalWriter(out_dir, split) if arm == "retrieval" else PlainWriter(out_dir, split)
    builder = None
    if "retrieval" in arms:
        builder = RetrievalBuilder(people, pools, keys, tokenizer, seed, swap_rate, anon_rate,
                                   gold_drop_rate, distractors, placeholder_rate=placeholder_rate)

    def window_doc(row) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        source, start, length = (int(v) for v in row)
        body = np.asarray(memmaps[source][start:start + length], dtype=np.uint16)
        ids = np.concatenate(([bos], body, [eos])).astype(np.uint16)
        mask = np.ones(len(ids), dtype=np.uint8)
        mask[0] = 0
        return ids, mask, np.zeros(len(ids), dtype=np.uint8)

    none_idx = CONDITIONS.index("none")
    empty_keys = np.zeros((0, EMBED_DIM), dtype=np.float32)
    bio_doc_count = filler_doc_count = 0
    for lo in range(0, len(kinds), block):
        kind_block, index_block = kinds[lo:lo + block], indices[lo:lo + block]
        retrieval_docs = {}
        if builder is not None:
            flat = [int(i) for k, i in zip(kind_block, index_block) if k == BIO]
            retrieval_docs = dict(zip(flat, builder.build([docs[i] for i in flat])))
        for kind, index in zip(kind_block, index_block):
            index = int(index)
            if kind == FILLER:
                ids, mask, codes = window_doc(chosen[index])
                filler_doc_count += 1
                for arm, writer in writers.items():
                    if arm == "retrieval":
                        writer.write(ids, mask, codes, [], [], empty_keys, [], none_idx)
                    else:
                        writer.write(ids, mask, codes)
            else:
                bio_doc_count += 1
                ids_body, codes_body = canonical[index]
                ids = np.concatenate(([bos], ids_body, [eos])).astype(np.uint16)
                codes = np.concatenate(([0], codes_body, [0])).astype(np.uint8)
                is_value = (codes >= 1) & (codes <= 5)
                for arm, writer in writers.items():
                    if arm == "full":
                        mask = np.ones(len(ids), dtype=np.uint8)
                        mask[0] = 0
                        writer.write(ids, mask, codes)
                    elif arm == "masked":
                        mask = np.ones(len(ids), dtype=np.uint8)
                        mask[0] = 0
                        mask[is_value] = 0
                        writer.write(ids, mask, codes)
                    else:
                        d = retrieval_docs[index]
                        writer.write(d["ids"], d["mask"], d["codes"], d["ev_ids"], d["ev_chunk"],
                                     d["keys"], d["ev_gold"], d["condition_idx"])
        done = min(lo + block, len(kinds))
        if (lo // block) % 20 == 0:
            logger.info(f"  {done:,}/{len(kinds):,} documents ({time.time() - t0:.0f}s)")
    for writer in writers.values():
        writer.close()

    val_split = "inject_val"
    val_path = os.path.join(out_dir, f"{val_split}.bin")
    if os.path.exists(val_path) and not overwrite:
        logger.info(f"{val_split} exists, kept (identical for every arm)")
    else:
        _check_free(out_dir, val_split, True)
        val_writer = PlainWriter(out_dir, val_split)
        for row in val_windows:
            val_writer.write(*window_doc(row))
        val_writer.close()
        logger.info(f"{val_split}: {val_writer.docs:,} windows, {val_writer.tokens:,} tokens")

    metrics = {"arms": {}, "common": {
        "documents": {"filler": filler_doc_count, "bio": bio_doc_count},
        "bio_tokens": bio_tokens, "bio_token_share": bio_tokens / max(1, writers[arms[0]].tokens),
        "exposures_by_tier": {str(t): {"people": n, "exposures_each": t} for t, n in
                              sorted(persons_per_tier.items())},
        "filler_repeat_factor": repeat, "filler_windows_train": int(len(train_windows)),
        "filler_windows_val": int(len(val_windows)), "seed": seed, "target_tokens": target_tokens,
        "built_seconds": round(time.time() - t0, 1),
    }}
    for arm, writer in writers.items():
        entry = {"split": names[arm], "docs": writer.docs, "tokens": writer.tokens,
                 "mask_zero_share": writer.mask_zero / max(1, writer.tokens)}
        if arm == "retrieval":
            s = builder.stats
            entry.update({
                "swap_rate": swap_rate, "anon_rate": anon_rate, "placeholder_rate": placeholder_rate,
                "gold_drop_rate": gold_drop_rate, "distractors": distractors,
                "gold_in_buffer_share": s["gold_in_buffer"] / max(1, s["docs"]),
                "supported_span_share": s["spans_supported"] / max(1, s["spans"]),
                "supported_fact_token_share": s["span_tokens_supported"] / max(1, s["span_tokens"]),
                "swapped_spans": s["swapped_spans"], "anonymized_docs": s["anonymized_docs"],
                "placeholder_docs": s["placeholder_docs"], "goldless_docs": s["goldless_docs"],
                "value_collision_share_of_spans": s["collision_spans"] / max(1, s["spans"]),
                "value_collision_share_of_goldless_spans":
                    s["collision_spans"] / max(1, s["spans_in_goldless_docs"]),
                "evidence_tokens": writer.ev_tokens,
                "evidence_to_prompt_ratio": writer.ev_tokens / max(1, writer.tokens),
                "evidence_tokens_per_doc_max": s["ev_tokens_max"],
            })
        metrics["arms"][arm] = entry

    path = os.path.join(out_dir, "inject_build.json")
    merged = {"arms": {}, "common": {}}
    if os.path.exists(path):
        with open(path) as f:
            merged = json.load(f)
    merged["arms"].update({names[a]: e for a, e in metrics["arms"].items()})
    merged["common"] = metrics["common"]
    _write_json_atomic(path, merged)
    return metrics


def log_report(metrics: dict) -> None:
    """Log the common figures and each arm's figures.

    Args:
        metrics: the dict returned by ``build_injection``, with ``common`` and ``arms`` keys.
    """
    common = metrics["common"]
    logger.info(f"bio token share {common['bio_token_share']:.2%}, filler repeat "
                f"{common['filler_repeat_factor']:.2f}x, documents {common['documents']}")
    for arm, e in metrics["arms"].items():
        logger.info(f"{arm}: {e['docs']:,} docs, {e['tokens']:,} tokens, mask zero "
                    f"{e['mask_zero_share']:.2%}")
        if arm == "retrieval":
            logger.info(
                f"  gold in buffer {e['gold_in_buffer_share']:.1%}, supported spans "
                f"{e['supported_span_share']:.1%} ({e['supported_fact_token_share']:.1%} of fact tokens), "
                f"swapped {e['swapped_spans']:,}, anonymized docs {e['anonymized_docs']:,}, "
                f"placeholder docs {e.get('placeholder_docs', 0):,}, "
                f"collisions {e['value_collision_share_of_spans']:.2%} of spans "
                f"({e['value_collision_share_of_goldless_spans']:.2%} of gold-less spans), "
                f"evidence ratio {e['evidence_to_prompt_ratio']:.2f}"
            )


def parse_tiers(text: str) -> Dict[int, int]:
    """``"1000:100,100:500"`` to ``{1000: 100, 100: 500}``.

    Args:
        text: comma separated ``tier:count`` pairs.
    """
    return {int(a): int(b) for a, b in (part.split(":") for part in text.split(","))}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "data", "prepared_inject"))
    parser.add_argument("--filler-dir", default=os.path.join(BASE_DIR, "data", "prepared"))
    parser.add_argument("--filler-phases", default="ir,phase1,phase2")
    parser.add_argument("--target-tokens", type=int, default=300_000_000,
                        help="tokens of one arm's train split; filler is cycled when it is short, "
                             "270000000 avoids the repeat")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--swap-rate", type=float, default=0.15)
    parser.add_argument("--anon-rate", type=float, default=0.5)
    parser.add_argument("--placeholder-rate", type=float, default=0.0)
    parser.add_argument("--gold-drop-rate", type=float, default=0.2)
    parser.add_argument("--distractors", type=int, default=3)
    parser.add_argument("--persons-per-tier", default="1000:100,100:500,10:1000,1:2000",
                        help="tier:people pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--suffix", default="", help="appended to the arm split names")
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--store-dir", default=os.path.join(BASE_DIR, "data", "index", "inject_bios"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    filler = []
    for phase in args.filler_phases.split(","):
        pair = (os.path.join(args.filler_dir, f"{phase}.bin"), os.path.join(args.filler_dir, f"{phase}.idx"))
        if not all(os.path.exists(p) for p in pair):
            raise SystemExit(f"{pair[0]} or its idx is missing")
        filler.append(pair)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    logger.info(f"loading the external embedder ({ChunkEmbedder.REPO}) on {args.device}")
    embedder = ChunkEmbedder(device=args.device)
    metrics = build_injection(
        args.out_dir, tokenizer, embedder, persons_per_tier=parse_tiers(args.persons_per_tier),
        filler=filler, target_tokens=args.target_tokens, seq_length=args.seq_length,
        swap_rate=args.swap_rate, anon_rate=args.anon_rate,
        placeholder_rate=args.placeholder_rate, gold_drop_rate=args.gold_drop_rate,
        distractors=args.distractors, seed=args.seed, arms=arms, suffix=args.suffix,
        val_fraction=args.val_fraction, store_dir=args.store_dir, overwrite=args.overwrite,
    )
    log_report(metrics)


if __name__ == "__main__":
    main()

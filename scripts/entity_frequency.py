"""How often an answer string occurs in a token corpus, exactly, on the CPU.

The counterfactual condition stratifies answers by how common they are in the pretraining text,
because a model has more to remember about a head entity than a tail one. The count is of token
sequences in a prepared ``{phase}.bin`` (uint16, the corpus the model was trained on or its local
stand-in), summed over the two ways an answer is tokenized in running text: bare, and with a
leading space.

Counting a few thousand sequences in 200M tokens in pure Python would take hours, so it is a
rolling polynomial hash in numpy. For each length n up to the longest sequence, the hash of every
window of n tokens comes from the hash of every window of n - 1 tokens by one multiply and one add
(uint64 arithmetic that wraps, or a small modulus for the collision test). Windows whose hash is a
target hash are then compared token by token, so a collision can inflate nothing. The corpus is
read in overlapping chunks (overlap longest sequence minus one) and a window is counted by the
chunk it starts in, so a sequence straddling a chunk boundary is counted once. Overlapping
occurrences of the same sequence all count.

Run from the repo root:

    python scripts/entity_frequency.py --bin data/prepared/ir.bin --answers answers.txt --out freq.json
"""
import os
import sys
import json
import hashlib
import argparse
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

MASK64 = (1 << 64) - 1
HASH_BASE = 0x9E3779B97F4A7C15
SMALL_BASE = 31
DEFAULT_CHUNK_TOKENS = 50_000_000


def count_token_sequences(bin_path: str, sequences: List[List[int]], *,
                          chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                          modulus: Optional[int] = None) -> List[int]:
    """Exact occurrence count of each token sequence in a uint16 token file.

    Args:
        bin_path: path to the flat uint16 corpus.
        sequences: token id lists; duplicates and empty lists are allowed (empty counts 0).
        chunk_tokens: windows are counted in chunks of this many start positions.
        modulus: when set (below 2**32), hashes are reduced modulo it, which makes collisions
            common on purpose so the exact check is exercised. None uses wrapping uint64.

    Returns:
        One count per input sequence, in input order.
    """
    counts = [0] * len(sequences)
    by_length: Dict[int, Dict[tuple, List[int]]] = {}
    for i, seq in enumerate(sequences):
        if seq:
            by_length.setdefault(len(seq), {}).setdefault(tuple(seq), []).append(i)
    if not by_length:
        return counts
    max_len = max(by_length)
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    total = tokens.shape[0]
    base_int = HASH_BASE if modulus is None else SMALL_BASE
    base = np.uint64(base_int)
    mod = None if modulus is None else np.uint64(modulus)

    def hash_sequence(seq: Sequence[int]) -> int:
        h = 0
        for t in seq:
            h = (h * base_int + t) & MASK64 if modulus is None else (h * base_int + t) % modulus
        return h

    targets = {}
    for n, seqs in by_length.items():
        table: Dict[int, List[tuple]] = {}
        for seq in seqs:
            table.setdefault(hash_sequence(seq), []).append(seq)
        targets[n] = (np.array(sorted(table), dtype=np.uint64), table)

    for start in range(0, total, chunk_tokens):
        window = np.asarray(tokens[start:start + chunk_tokens + max_len - 1], dtype=np.uint64)
        length = window.shape[0]
        own = min(chunk_tokens, total - start)
        rolling = None
        for n in range(1, max_len + 1):
            valid_starts = length - n + 1
            if valid_starts <= 0:
                break
            if rolling is None:
                rolling = window.copy()
            else:
                rolling = rolling[:valid_starts] * base + window[n - 1:length]
            if mod is not None:
                rolling %= mod
            if n not in targets:
                continue
            hashes, table = targets[n]
            considered = min(own, valid_starts)
            part = rolling[:considered]
            slot = np.searchsorted(hashes, part)
            slot[slot == hashes.shape[0]] = hashes.shape[0] - 1
            positions = np.nonzero(hashes[slot] == part)[0]
            if positions.size == 0:
                continue
            offsets = np.arange(n)
            windows = window[positions[:, None] + offsets]
            position_hash = part[positions]
            order = np.argsort(position_hash, kind="stable")
            uniq, first, size = np.unique(position_hash[order], return_index=True, return_counts=True)
            for h, lo, width in zip(uniq, first, size):
                rows = windows[order[lo:lo + width]]
                for seq in table[int(h)]:
                    hit = int((rows == np.asarray(seq, dtype=np.uint64)).all(axis=1).sum())
                    if hit:
                        for i in by_length[n][seq]:
                            counts[i] += hit
    return counts


def answer_variants(tokenizer, text: str) -> List[List[int]]:
    """The two token sequences an answer takes in running text: bare and with a leading space.

    Args:
        tokenizer: a fast tokenizer.
        text: the answer string.
    """
    bare = tokenizer(text, add_special_tokens=False)["input_ids"]
    spaced = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
    return [bare] if bare == spaced else [bare, spaced]


def answer_frequencies(bin_path: str, answers: Sequence[str], tokenizer,
                       cache_dir: Optional[str] = None) -> Dict[str, int]:
    """Corpus count per answer string, bare plus leading space, cached on disk.

    The cache file is named by the sha1 of the corpus path and size, and holds the counts of every
    answer asked about so far, so a second slice only counts the answers it adds.

    Args:
        bin_path: path to the flat uint16 corpus.
        answers: answer strings.
        tokenizer: a fast tokenizer.
        cache_dir: where the JSON cache lives, or None for no cache.
    """
    cache, cache_path = {}, None
    if cache_dir:
        key = hashlib.sha1(f"{os.path.abspath(bin_path)}:{os.path.getsize(bin_path)}".encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{key}.json")
        if os.path.isfile(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
    todo = sorted({a for a in answers if a and a not in cache})
    if todo:
        flat, owners = [], []
        for i, answer in enumerate(todo):
            for seq in answer_variants(tokenizer, answer):
                flat.append(seq)
                owners.append(i)
        per_sequence = count_token_sequences(bin_path, flat)
        totals = [0] * len(todo)
        for owner, count in zip(owners, per_sequence):
            totals[owner] += count
        cache.update(dict(zip(todo, totals)))
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            part = cache_path + ".part"
            with open(part, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            os.replace(part, cache_path)
    return {a: cache.get(a, 0) for a in answers}


def main():
    from transformers import AutoTokenizer
    from utils import TOKENIZER_DIR, logger

    parser = argparse.ArgumentParser(description="count answer strings in a prepared token corpus")
    parser.add_argument("--bin", required=True, help="flat uint16 corpus, e.g. data/prepared/ir.bin")
    parser.add_argument("--answers", required=True, help="text file, one answer per line")
    parser.add_argument("--out", required=True, help="JSON file to write {answer: count}")
    parser.add_argument("--tokenizer", default=TOKENIZER_DIR)
    args = parser.parse_args()

    with open(args.answers, "r", encoding="utf-8") as f:
        answers = [line.rstrip("\n") for line in f if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    counts = answer_frequencies(args.bin, answers, tokenizer)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2, ensure_ascii=False)
    logger.info(f"counted {len(counts):,} answers in {args.bin}, wrote {args.out}")


if __name__ == "__main__":
    main()

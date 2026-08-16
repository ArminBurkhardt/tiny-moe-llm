"""Prune the DeepSeek tokenizer's vocab from 129280 to 65536 tokens (PLAN.md Step 8).

Target 65536, not smaller: any vocab_size <= 65536 fits a uint16 train.bin, halving the
prepared dataset's disk footprint -- the actual reason for this step, not param count.

Approach: sample text from a local stand-in for the Step 11 phase-1 mix, count how often
the current tokenizer actually emits each token id over that sample, then keep the most
frequent ids plus everything structurally required (every special/added token, and the
256-entry byte-level alphabet that guarantees arbitrary text stays encodable even after
merges are dropped). Kept multi-piece tokens are closed under their BPE merge dependency
so a surviving token never depends on a dropped constituent piece.

Run from the repo root: `python scripts/prune_vocab.py`.
"""

import os
import sys
import json
import glob
import copy
import random
import argparse
import shutil
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from utils import BASE_DIR, logger

SRC_TOKENIZER_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained", "DeepSeek-V4-Pro-tokenizer")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")

# local stand-in for Step 11's phase-1 mix. No DCLM-baseline or FinePDFs-Edu shard exists
# locally, so their combined 17% is folded into fineweb (all three are general web text).
# nemotron-pre-specialized-v1 (InfiniByte-Reasoning) is deliberately excluded: it doesn't
# correspond to any phase-1 row in PLAN.md's mix table, and its instruction-reasoning
# style is closer to SFT data than pretraining text.
MIX_SOURCES = [
    # (weight, root, column)
    (0.80, "data/datasets/parquet/fineweb", "content"),
    (0.1333, "data/datasets/parquet/nemotron/nemotron-pre-specialized-v1.1", "text"),
    (0.0333, "data/datasets/parquet/nemotron/nemotron-pre-math-v1/4plus_MIND", "text"),
    (0.0333, "data/datasets/parquet/wikipedia/en", "text"),
]


def bytes_to_unicode() -> dict[int, str]:
    """the standard GPT2/byte-level-BPE byte -> unicode-char alphabet.

    every one of these 256 chars is a required leaf: with byte_fallback disabled, this
    alphabet is the only thing guaranteeing arbitrary text stays encodable once merges
    are dropped (the ByteLevel pre-tokenizer maps every input byte to one of these chars).
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def discover_files(root: str, pattern: str = "*.parquet") -> list[str]:
    if not os.path.exists(root):
        logger.warning(f"root {root} does not exist, skipping")
        return []
    return sorted(str(p) for p in Path(root).rglob(pattern) if p.is_file())


def iter_texts(files: list[str], column: str, rng: random.Random):
    """yield text records from parquet files, file order and within-file order shuffled"""
    order = list(files)
    rng.shuffle(order)
    for path in order:
        try:
            df = pd.read_parquet(path, columns=[column])
        except Exception as e:
            logger.warning(f"skipping unreadable file {path}: {e}")
            continue
        texts = df[column].dropna().astype(str).tolist()
        rng.shuffle(texts)
        for t in texts:
            if t:
                yield t


def sample_mix(byte_budget: int, seed: int, held_out: bool) -> tuple[dict[str, list[str]], dict[str, int]]:
    """sample text across MIX_SOURCES at their mix weights up to byte_budget total (utf-8 bytes).

    held_out selects the disjoint file half (odd vs even index) so the frequency-count
    corpus and the fertility/round-trip corpus never share a source file.
    """
    per_source_texts, per_source_bytes = {}, {}
    for weight, root, column in MIX_SOURCES:
        name = os.path.basename(root.rstrip("/\\"))
        all_files = discover_files(root)
        files = all_files[1::2] if held_out else all_files[0::2]
        target_bytes = int(byte_budget * weight)
        rng = random.Random(seed ^ (hash(name) & 0xFFFFFFFF))
        texts, total = [], 0
        if files:
            for t in iter_texts(files, column, rng):
                total += len(t.encode("utf-8"))
                texts.append(t)
                if total >= target_bytes:
                    break
        else:
            logger.warning(f"no files available for {name} ({'held-out' if held_out else 'frequency'} half), skipping")
        per_source_texts[name] = texts
        per_source_bytes[name] = total
        logger.info(
            f"[{'held-out' if held_out else 'freq'}] sampled {total / 1e6:.1f}MB "
            f"({len(texts)} docs) from {name} (target {target_bytes / 1e6:.1f}MB)"
        )
    return per_source_texts, per_source_bytes


def count_token_frequencies(tokenizer, per_source_texts: dict[str, list[str]], num_bpe_ids: int, batch_size: int = 256) -> np.ndarray:
    counts = np.zeros(num_bpe_ids, dtype=np.int64)
    for name, texts in per_source_texts.items():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"]
            flat = np.fromiter((tok for ids in enc for tok in ids), dtype=np.int64)
            if flat.size:
                counts += np.bincount(flat, minlength=num_bpe_ids)
        logger.info(f"counted frequencies for {name} ({len(texts)} docs)")
    return counts


def build_merge_graph(vocab: dict[str, int], merges: list[str]) -> dict[int, tuple[int, int]]:
    """token id -> (left parent id, right parent id) for every BPE-merge-derived token.
    ids with no entry here are leaves (the base alphabet)."""
    parents = {}
    for merge in merges:
        left, right = merge.split(" ", 1)
        merged = left + right
        if merged in vocab:
            parents[vocab[merged]] = (vocab[left], vocab[right])
    return parents


def closure(token_id: int, parents: dict[int, tuple[int, int]], cache: dict[int, set]) -> set:
    """full set of ids required to keep token_id: itself plus every ancestor, recursively."""
    cached = cache.get(token_id)
    if cached is not None:
        return cached
    result = {token_id}
    parent_pair = parents.get(token_id)
    if parent_pair is not None:
        left, right = parent_pair
        result |= closure(left, parents, cache)
        result |= closure(right, parents, cache)
    cache[token_id] = result
    return result


def select_kept_ids(vocab: dict[str, int], merges: list[str], freq_counts: np.ndarray, required_ids: set[int], target_size: int) -> set[int]:
    """keep `required_ids` plus the most-frequent remaining tokens, closed under merge
    dependency (PLAN.md: "a kept merge never depends on a dropped one"), reaching exactly
    target_size."""
    parents = build_merge_graph(vocab, merges)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    cache: dict[int, set] = {}

    kept = set(required_ids)
    candidates = [i for i in vocab.values() if i not in kept]
    candidates.sort(key=lambda i: (-int(freq_counts[i]) if i < len(freq_counts) else 0, i))

    def try_add(cid: int) -> bool:
        if cid in kept:
            return False
        needed = closure(cid, parents, cache) - kept
        if needed and len(kept) + len(needed) <= target_size:
            kept.update(needed)
            return True
        return False

    for cid in candidates:
        if len(kept) >= target_size:
            break
        try_add(cid)

    # backfill: candidates skipped earlier can become cheap (or free) once their ancestors
    # were pulled in by something else, so keep re-scanning until nothing more fits exactly.
    changed = True
    while len(kept) < target_size and changed:
        changed = False
        for cid in candidates:
            if len(kept) >= target_size:
                break
            if try_add(cid):
                changed = True

    if len(kept) != target_size:
        raise RuntimeError(f"could not reach exact target vocab size: kept={len(kept)} target={target_size}")
    return kept


def build_new_tokenizer_json(orig: dict, kept_ids: set[int], remap: dict[int, int]) -> dict:
    new = copy.deepcopy(orig)
    orig_vocab = orig["model"]["vocab"]

    new["model"]["vocab"] = {tok: remap[old_id] for tok, old_id in orig_vocab.items() if old_id in kept_ids}
    kept_strs = set(new["model"]["vocab"].keys())

    new_merges = []
    for merge in orig["model"]["merges"]:
        left, right = merge.split(" ", 1)
        if left in kept_strs and right in kept_strs and (left + right) in kept_strs:
            new_merges.append(merge)
    new["model"]["merges"] = new_merges

    new["added_tokens"] = [{**t, "id": remap[t["id"]]} for t in orig["added_tokens"] if t["id"] in kept_ids]
    return new


def write_new_tokenizer(src_dir: str, out_dir: str, new_tokenizer_json: dict, remap: dict[int, int], target_vocab_size: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(new_tokenizer_json, f, ensure_ascii=False)

    for fname in ("tokenizer_config.json", "chat_template.jinja"):
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, fname))

    # config.json in the tokenizer dir describes the full DeepSeek-V4-Pro model, not this
    # project's model -- nothing here reads it, but keep vocab_size consistent for anyone
    # who opens the folder expecting AutoConfig-shaped metadata.
    src_config = os.path.join(src_dir, "config.json")
    if os.path.exists(src_config):
        with open(src_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["vocab_size"] = target_vocab_size
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "id_remap.json"), "w", encoding="utf-8") as f:
        json.dump({str(old): new for old, new in sorted(remap.items())}, f)


def measure_fertility(tokenizer, per_source_texts: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """tokens/byte per source, for a before/after fertility comparison."""
    out = {}
    for name, texts in per_source_texts.items():
        if not texts:
            continue
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_tokens = 0
        for i in range(0, len(texts), 256):
            batch = texts[i:i + 256]
            enc = tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"]
            n_tokens += sum(len(ids) for ids in enc)
        out[name] = {"bytes": n_bytes, "tokens": n_tokens, "tokens_per_byte": n_tokens / n_bytes}
    total_bytes = sum(v["bytes"] for v in out.values())
    total_tokens = sum(v["tokens"] for v in out.values())
    out["_overall"] = {"bytes": total_bytes, "tokens": total_tokens, "tokens_per_byte": total_tokens / total_bytes if total_bytes else 0.0}
    return out


def check_roundtrip(tokenizer, texts: list[str]) -> int:
    """returns the number of documents that failed to round-trip byte-identically"""
    failures = 0
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != t:
            failures += 1
    return failures


def main():
    parser = argparse.ArgumentParser(description="Prune the DeepSeek tokenizer's vocab to fit uint16 (PLAN.md Step 8)")
    parser.add_argument("--src-tokenizer", default=SRC_TOKENIZER_DIR, help="source tokenizer directory")
    parser.add_argument("--output-name", default="DeepSeek-V4-Pro-tokenizer-65536", help="output dir name under ckpts/pretrained/")
    parser.add_argument("--target-vocab-size", type=int, default=65536)
    parser.add_argument("--freq-sample-gb", type=float, default=2.0, help="text sampled for token-frequency counting")
    parser.add_argument("--fertility-sample-mb", type=float, default=200.0, help="held-out text sampled for the fertility/round-trip check")
    parser.add_argument("--roundtrip-docs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.join(BASE_DIR, "ckpts", "pretrained", args.output_name)

    logger.info(f"loading source tokenizer from {args.src_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.src_tokenizer)
    with open(os.path.join(args.src_tokenizer, "tokenizer.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab = raw["model"]["vocab"]
    merges = raw["model"]["merges"]
    added_tokens = raw["added_tokens"]

    byte_alphabet_ids = {vocab[ch] for ch in bytes_to_unicode().values() if ch in vocab}
    added_ids = {t["id"] for t in added_tokens}
    required_ids = byte_alphabet_ids | added_ids
    logger.info(f"required (specials + byte alphabet): {len(required_ids)} ids, base BPE vocab: {len(vocab)}, total (incl. added): {len(tokenizer)}")

    # --- frequency-count corpus (local stand-in for the phase-1 mix) ---
    freq_texts, freq_bytes = sample_mix(int(args.freq_sample_gb * 1024 ** 3), args.seed, held_out=False)
    logger.info(f"total frequency-count corpus: {sum(freq_bytes.values()) / 1e9:.2f}GB")
    freq_counts = count_token_frequencies(tokenizer, freq_texts, num_bpe_ids=len(vocab))

    # --- select kept vocab, closed under merge dependency ---
    kept_ids = select_kept_ids(vocab, merges, freq_counts, required_ids, args.target_vocab_size)
    remap = {old: new for new, old in enumerate(sorted(kept_ids))}
    logger.info(f"kept {len(kept_ids)} ids (target {args.target_vocab_size})")

    new_tokenizer_json = build_new_tokenizer_json(raw, kept_ids, remap)
    write_new_tokenizer(args.src_tokenizer, out_dir, new_tokenizer_json, remap, args.target_vocab_size)
    logger.info(f"wrote pruned tokenizer to {out_dir}")

    new_tokenizer = AutoTokenizer.from_pretrained(out_dir)
    assert len(new_tokenizer) == args.target_vocab_size, f"len(tokenizer)={len(new_tokenizer)} != {args.target_vocab_size}"
    assert new_tokenizer.pad_token_id == new_tokenizer.eos_token_id, "pad/eos id invariant broken by prune"
    assert new_tokenizer.bos_token_id == 0, "bos_token_id invariant broken by prune"
    assert max(new_tokenizer_json["model"]["vocab"].values()) < args.target_vocab_size
    logger.info("acceptance checks passed: len==target, pad==eos, bos==0, max id < target")

    # --- held-out fertility + round-trip check (disjoint files from the frequency corpus) ---
    held_out_texts, held_out_bytes = sample_mix(int(args.fertility_sample_mb * 1024 ** 2), args.seed, held_out=True)
    logger.info(f"total held-out corpus: {sum(held_out_bytes.values()) / 1e6:.1f}MB")

    fertility_before = measure_fertility(tokenizer, held_out_texts)
    fertility_after = measure_fertility(new_tokenizer, held_out_texts)

    regression = {}
    for name in fertility_before:
        before_tpb = fertility_before[name]["tokens_per_byte"]
        after_tpb = fertility_after[name]["tokens_per_byte"]
        regression[name] = (after_tpb - before_tpb) / before_tpb if before_tpb else 0.0
        logger.info(f"fertility[{name}]: before={before_tpb:.4f} tok/byte, after={after_tpb:.4f} tok/byte, regression={regression[name] * 100:.2f}%")

    overall_regression = regression["_overall"]
    if overall_regression > 0.03:
        logger.warning(f"overall fertility regression {overall_regression * 100:.2f}% exceeds the 3% acceptance bound -- keep more tokens")
    else:
        logger.info(f"overall fertility regression {overall_regression * 100:.2f}% within the 3% acceptance bound")

    roundtrip_docs = [t for texts in held_out_texts.values() for t in texts][:args.roundtrip_docs]
    failures = check_roundtrip(new_tokenizer, roundtrip_docs)
    logger.info(f"round-trip check: {failures}/{len(roundtrip_docs)} documents failed to reconstruct byte-identically")
    assert failures == 0, f"{failures} documents failed round-trip"

    manifest = {
        "tokenizer_dir": out_dir,
        "target_vocab_size": args.target_vocab_size,
        "kept_vocab_size": len(kept_ids),
        "required_ids": len(required_ids),
        "freq_sample_bytes": freq_bytes,
        "held_out_sample_bytes": held_out_bytes,
        "fertility_before": fertility_before,
        "fertility_after": fertility_after,
        "fertility_regression": regression,
        "roundtrip_docs_checked": len(roundtrip_docs),
        "roundtrip_failures": failures,
    }
    existing = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing["vocab_prune"] = manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"recorded fertility measurements in {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

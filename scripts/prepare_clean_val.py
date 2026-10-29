"""A filler validation slice that shares no document with any training row of either builder.

``inject_val`` is cut by window from the same ``ir``, ``phase1`` and ``phase2`` documents the
biography builder draws its filler from, and from the ``ir`` documents the evidence builder's web
text arm reads, so some of its tokens appear verbatim as supervised text in training rows. This
script recomputes the validation windows ``prepare_injection_data.load_filler_windows`` sets aside
and keeps only the ones whose document the evidence builder never read.

What the evidence builder used. Its state file (``_prepare_state_{split_prefix}.json``) records one
cursor per source: ``sources[key].row_idx`` is the index of the next document to read, so for the
web text source every document below the cursor is treated as used (the builder reads ``{phase}.idx``
documents in order from 0, skipped ones included). The state does not say which phase the web arm
read, so ``--webtext-phase`` names it (the builder's default is ``ir``). A source whose epoch is
above 0 has wrapped, so every document of it counts as used. Several state files may be given
comma separated; the highest cursor wins. A state with no ``webtext`` entry, or one whose cursor is
missing, raises an error. Phases the web arm did not read are kept whole.

The biography builder's training windows are the complement of the validation windows by
construction (the same hash picks both), so no window is shared with them.

Output ``{name}.bin``, ``.idx``, ``.mask``, ``.factspan`` in ``inject_val``'s layout (BOS, body,
EOS per window; no MTP pads; mask 0 on BOS and 1 elsewhere; factspan zeros) and ``{name}.meta.json``.

Args:
    --out-dir: where the slice is written.
    --filler-dir: holds ``{phase}.bin`` and ``{phase}.idx``.
    --filler-phases: the phases the biography build used, in its order (the order seeds the hash).
    --seq-length: the build's row length; the window is ``min(1000, seq_length - 24)`` as there.
    --seed, --val-fraction: the biography build's, so the windows are the same ones.
    --target-tokens: stop at about this many tokens (default: the size of ``inject_val``).
    --exclude-evidence-state: comma separated evidence builder state files.
    --webtext-phase: the phase the evidence builder's web arm read.
    --name: split name.
    --check-against: a prepared split prefix to measure verbatim overlap against.
    --check-rows: rows of that split to scan (0 = all).
    --check-baseline: another split prefix (such as inject_val) measured the same way.
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.prepare_injection_data import PlainWriter, load_filler_windows
from utils import BASE_DIR, TOKENIZER_DIR, logger

GRAM = 64
STRIDE = 32


def window_size(seq_length: int) -> int:
    """The filler window the biography builder uses for a row length."""
    return min(1000, seq_length - 24)


def used_documents(state_paths: Sequence[str], webtext_phase: str,
                   phases: Sequence[str]) -> Dict[str, Optional[int]]:
    """Per phase, the count of leading documents the evidence builder read.

    Args:
        state_paths: evidence builder state files.
        webtext_phase: the phase its web arm read.
        phases: the filler phases being cut.

    Returns:
        ``{phase: n}``: documents ``0..n-1`` are used; ``None`` means every document is used.
    """
    used: Dict[str, Optional[int]] = {p: 0 for p in phases}
    for path in state_paths:
        with open(path) as f:
            state = json.load(f)
        web = state.get("sources", {}).get("webtext")
        if web is None or "row_idx" not in web:
            raise SystemExit(f"{path} has no webtext cursor; cannot tell which documents were used")
        if webtext_phase not in used:
            continue
        if web.get("epoch", 0) > 0 or used[webtext_phase] is None:
            used[webtext_phase] = None
        else:
            used[webtext_phase] = max(used[webtext_phase], int(web["row_idx"]))
    return used


def filter_windows(val: np.ndarray, offsets_by_source: Sequence[np.ndarray], phases: Sequence[str],
                   used: Dict[str, Optional[int]]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Drop the validation windows whose document the evidence builder read.

    Args:
        val: ``[N, 3]`` of ``(source, start, length)``.
        offsets_by_source: each source's ``.idx`` offsets.
        phases: phase names, aligned with sources.
        used: output of ``used_documents``.

    Returns:
        ``(kept windows, document index of each kept window, per source stats)``.
    """
    keep = np.ones(len(val), dtype=bool)
    docs = np.zeros(len(val), dtype=np.int64)
    stats = []
    for s, phase in enumerate(phases):
        rows = np.where(val[:, 0] == s)[0]
        d = np.searchsorted(offsets_by_source[s], val[rows, 1], side="right") - 1
        docs[rows] = d
        n_used = used[phase]
        drop = np.ones(len(rows), dtype=bool) if n_used is None else d < n_used
        keep[rows[drop]] = False
        stats.append({"phase": phase, "windows": int(len(rows)), "dropped": int(drop.sum()),
                      "kept": int((~drop).sum()),
                      "excluded_documents": "all" if n_used is None else [0, int(n_used)]})
    return val[keep], docs[keep], stats


def write_slice(out_dir: str, name: str, memmaps: Sequence[np.ndarray], windows: np.ndarray,
                bos: int, eos: int) -> PlainWriter:
    """Write windows as BOS, body, EOS documents with ``inject_val``'s mask and factspan."""
    writer = PlainWriter(out_dir, name)
    for source, start, length in windows:
        body = np.asarray(memmaps[int(source)][int(start):int(start) + int(length)], dtype=np.uint16)
        ids = np.concatenate(([bos], body, [eos])).astype(np.uint16)
        mask = np.ones(len(ids), dtype=np.uint8)
        mask[0] = 0
        writer.write(ids, mask, np.zeros(len(ids), dtype=np.uint8))
    writer.close()
    return writer


# ------------------------------------------------------------------------------------- overlap

_BASE = 0x9E3779B97F4A7C15


def gram_hashes(tokens: np.ndarray, length: int = GRAM) -> np.ndarray:
    """Hash of every ``length``-token run of ``tokens`` (uint64, wrapping polynomial)."""
    n = len(tokens) - length + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    one = np.ones(1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        pw = np.concatenate((one, np.cumprod(np.full(len(tokens) - 1, _BASE, dtype=np.uint64))))
        inv_base = pow(_BASE, -1, 1 << 64)
        inv = np.concatenate((one, np.cumprod(np.full(len(tokens) - 1, inv_base, dtype=np.uint64))))
        prefix = np.concatenate((np.zeros(1, dtype=np.uint64),
                                 np.cumsum(tokens.astype(np.uint64) * pw, dtype=np.uint64)))
        return (prefix[length:length + n] - prefix[:n]) * inv[:n]


def _documents(prefix: str, rows: Optional[np.ndarray]):
    tokens = np.memmap(prefix + ".bin", dtype=np.uint16, mode="r")
    mask = np.memmap(prefix + ".mask", dtype=np.uint8, mode="r")
    offsets = np.memmap(prefix + ".idx", dtype=np.uint64, mode="r")
    picks = range(len(offsets) - 1) if rows is None else rows
    for d in picks:
        a, b = int(offsets[d]), int(offsets[d + 1])
        yield np.asarray(tokens[a:b]), np.asarray(mask[a:b])


def overlap_share(clean_prefix: str, train_prefix: str, rows: int = 2000, seed: int = 0) -> dict:
    """Share of ``clean_prefix`` tokens covered by a verbatim run from supervised training tokens.

    Every 64-token run of the clean slice at stride 32 is hashed; every 64-token run at stride 1
    of ``train_prefix`` whose tokens are all mask 1 is looked up. A clean token counts as covered
    when it lies inside a matched run.

    Args:
        clean_prefix: split to test (``.bin``, ``.idx``).
        train_prefix: training split (``.bin``, ``.idx``, ``.mask``).
        rows: training rows to scan, 0 for all.
        seed: seeds the row sample.

    Returns:
        ``{"covered_share", "covered_tokens", "tokens", "rows_scanned", "rows_total"}``.
    """
    n_rows = len(np.memmap(train_prefix + ".idx", dtype=np.uint64, mode="r")) - 1
    picks = None
    if rows and rows < n_rows:
        picks = np.sort(np.random.default_rng(seed).choice(n_rows, size=rows, replace=False))
    clean_docs = [t for t, _ in _documents(clean_prefix, None)]
    table, owner, starts = [], [], []
    for i, t in enumerate(clean_docs):
        h = gram_hashes(t)[::STRIDE]
        table.append(h)
        owner.append(np.full(len(h), i, dtype=np.int64))
        starts.append(np.arange(0, len(h) * STRIDE, STRIDE, dtype=np.int64))
    table = np.concatenate(table) if table else np.zeros(0, dtype=np.uint64)
    owner = np.concatenate(owner) if owner else np.zeros(0, dtype=np.int64)
    starts = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
    order = np.argsort(table, kind="stable")
    table, owner, starts = table[order], owner[order], starts[order]
    hit = np.zeros(len(table), dtype=bool)
    scanned = 0
    for tokens, mask in _documents(train_prefix, picks):
        scanned += 1
        h = gram_hashes(tokens)
        if not len(h):
            continue
        run = np.concatenate(([0], np.cumsum(mask == 1)))
        ok = (run[GRAM:GRAM + len(h)] - run[:len(h)]) == GRAM
        h = h[ok]
        if not len(h):
            continue
        lo = np.searchsorted(table, h, side="left")
        hi = np.searchsorted(table, h, side="right")
        for a, b in zip(lo[hi > lo], hi[hi > lo]):
            hit[a:b] = True
    covered = 0
    total = sum(len(t) for t in clean_docs)
    for i, t in enumerate(clean_docs):
        m = np.zeros(len(t), dtype=bool)
        for st in starts[(owner == i) & hit]:
            m[st:st + GRAM] = True
        covered += int(m.sum())
    return {"covered_share": covered / max(1, total), "covered_tokens": covered, "tokens": total,
            "rows_scanned": scanned, "rows_total": n_rows}


# ------------------------------------------------------------------------------------------ main

def build(out_dir: str, filler: Sequence[Tuple[str, str]], phases: Sequence[str], window: int,
          seed: int, val_fraction: float, state_paths: Sequence[str], webtext_phase: str,
          name: str, target_tokens: Optional[int], bos: int, eos: int) -> dict:
    """Recompute the validation windows, drop the used documents and write the slice."""
    memmaps, _, val = load_filler_windows(filler, window, seed, val_fraction)
    offsets = [np.asarray(np.memmap(i, dtype=np.uint64, mode="r")) for _, i in filler]
    used = used_documents(state_paths, webtext_phase, phases)
    kept, _, stats = filter_windows(val, offsets, phases, used)
    if target_tokens is not None:
        cum = np.cumsum(kept[:, 2] + 2)
        kept = kept[:int(np.searchsorted(cum, target_tokens, side="right"))]
    os.makedirs(out_dir, exist_ok=True)
    writer = write_slice(out_dir, name, memmaps, kept, bos, eos)
    meta = {"name": name, "windows": int(writer.docs), "tokens": int(writer.tokens),
            "window": window, "seed": seed, "val_fraction": val_fraction,
            "windows_before_exclusion": int(len(val)), "target_tokens": target_tokens,
            "sources": stats, "evidence_states": list(state_paths), "webtext_phase": webtext_phase,
            "used_documents": {k: ("all" if v is None else v) for k, v in used.items()}}
    with open(os.path.join(out_dir, f"{name}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default=os.path.join(BASE_DIR, "data", "prepared_pilot"))
    p.add_argument("--filler-dir", default=os.path.join(BASE_DIR, "data", "prepared"))
    p.add_argument("--filler-phases", default="ir,phase1,phase2")
    p.add_argument("--seq-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.005)
    p.add_argument("--target-tokens", type=int, default=None)
    p.add_argument("--exclude-evidence-state", default=os.path.join(
        BASE_DIR, "data", "prepared", "_prepare_state_evidence_nomany.json"))
    p.add_argument("--webtext-phase", default="ir")
    p.add_argument("--name", default="inject_val_clean")
    p.add_argument("--check-against", default=None)
    p.add_argument("--check-rows", type=int, default=2000)
    p.add_argument("--check-baseline", default=None)
    args = p.parse_args()

    phases = [x for x in args.filler_phases.split(",") if x]
    filler = [(os.path.join(args.filler_dir, f"{x}.bin"), os.path.join(args.filler_dir, f"{x}.idx"))
              for x in phases]
    target = args.target_tokens
    if target is None:
        ref = os.path.join(args.out_dir, "inject_val.idx")
        if os.path.exists(ref):
            target = int(np.memmap(ref, dtype=np.uint64, mode="r")[-1])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    meta = build(args.out_dir, filler, phases, window_size(args.seq_length), args.seed,
                 args.val_fraction, args.exclude_evidence_state.split(","), args.webtext_phase,
                 args.name, target, tok.bos_token_id, tok.eos_token_id)
    logger.info(f"{args.name}: {meta['windows']:,} windows, {meta['tokens']:,} tokens "
                f"(of {meta['windows_before_exclusion']:,} windows before exclusion)")
    for s in meta["sources"]:
        logger.info(f"  {s}")
    if args.check_against:
        for label, prefix in (("clean", os.path.join(args.out_dir, args.name)),
                              ("baseline", args.check_baseline)):
            if prefix:
                r = overlap_share(prefix, args.check_against, args.check_rows)
                logger.info(f"overlap {label} {os.path.basename(prefix)} vs "
                            f"{os.path.basename(args.check_against)}: {r}")
                print(f"overlap {label}: {r['covered_share']:.4%} ({r})")


if __name__ == "__main__":
    main()

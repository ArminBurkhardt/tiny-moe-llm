"""Merge several evidence-format splits into one, at fixed token shares, from a seeded stream.

The inputs are splits written by the corpus builders (``prepare_evidence_data.py``,
``prepare_injection_data.py``, ``prepare_chain_data.py``). Each is one *slice* of the output with a
label, a share of the target prompt tokens and a pass cap. An input with only ``bin idx mask`` is a
plain slice: it contributes prompt tokens and no evidence. The output is one split the trainer reads
as it reads any other: ``EvidenceDataset`` when ``{split}.ev`` exists, ``SFTDataset`` otherwise.

Output files (the sidecars ``src``, ``factspan``, ``evhop`` and ``chains.jsonl`` are not merged):

    evidence form   bin idx mask ev evidx evchunk evkey evkeyidx evgold cond ans
    --no-evidence   bin idx mask
    always          slice (uint8 per document: the slice index in ``--slice`` order), merge.json

``evidx`` and ``evkeyidx`` are rebuilt cumulatively; ``evchunk`` is a chunk index within its
document and is copied unchanged unless ``--max-chunks-per-doc`` drops chunks. A plain slice's
documents get no evidence, condition ``none`` and answerable 1; so do the documents of an evidence
slice that lacks ``evgold``/``cond``/``ans`` (its chunks then get gold flag 0).

Selection and order are deterministic and built so slices are independent:

* Documents of a slice are taken from a seeded permutation that depends only on the seed, the
  slice label and the slice's document count, consumed in order until the slice's token target
  (``share * target_tokens``) would be crossed (the crossing document is not taken). Documents of
  zero prompt tokens are skipped. With ``max_passes`` above 1 the next permutation of the same
  generator follows the first; a slice never emits more than ``max_passes`` times its own tokens,
  so a short slice leaves a gap that the report prints.
* The output order comes from one generator seeded with ``seed``: at each step a slice is drawn
  with probability proportional to its remaining planned tokens among the slices with documents
  left, and that slice's next document is emitted.

So each slice's own subsequence of documents is the same in every run with the same seed, label and
document count, whatever the other slices are. A control merge and a warm-up merge that differ only
in the biography slice therefore hold byte identical question answering and chain subsequences;
``md5_by_slice`` in ``merge.json`` (md5 over the prompt token bytes and the evidence token bytes of
the slice's documents in output order) is the check. The interleave itself is not identical across
such merges: its weights are the slices' remaining tokens, and two forms of one slice differ in
token length. A run that is resumed by document position on a second merge needs the same order,
so ``--order-from`` takes the first merge's ``.slice`` file and reuses its order; every slice must
then select the same number of documents in both merges. Input files are memmapped and the output
is written one document at a time, so a slice never has to fit in memory.

Example:
    python scripts/merge_evidence_splits.py --out-dir data/prepared_pilot --split pilot_main_train \\
        --target-tokens 1000000000 --seed 42 \\
        --slice bios=data/prepared_pilot/inject_retrieval_train:0.60 \\
        --slice qa=data/prepared/evidence_nomany_train:0.25:2 \\
        --slice chains=data/prepared/chains_train:0.10:1
"""
import os
import sys
import json
import zlib
import hashlib
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.prepare_evidence_data import CONDITIONS, EMBED_DIM, EVIDENCE_SUFFIXES
from utils import logger

NONE_CONDITION = CONDITIONS.index("none")
MAX_CHUNKS_PER_SEGMENT = 256
PLAIN_SUFFIXES = ("bin", "idx", "mask")
SIDE_FILES = ("slice", "merge.json")


@dataclass
class SliceSpec:
    label: str
    prefix: str
    share: float
    max_passes: int = 1


def parse_slice(arg: str) -> SliceSpec:
    """Parses ``label=path_prefix:share[:max_passes]``."""
    label, sep, rest = arg.partition("=")
    if not sep or not label or not rest:
        raise argparse.ArgumentTypeError(f"--slice {arg!r}: expected label=path:share[:max_passes]")
    parts = rest.split(":")
    passes = 1
    if len(parts) >= 3 and parts[-1].isdigit() and _is_float(parts[-2]):
        passes = int(parts[-1])
        parts = parts[:-1]
    if len(parts) < 2 or not _is_float(parts[-1]):
        raise argparse.ArgumentTypeError(f"--slice {arg!r}: share is not a number")
    share = float(parts[-1])
    if share <= 0 or passes < 1:
        raise argparse.ArgumentTypeError(f"--slice {arg!r}: share must be positive and passes at least 1")
    return SliceSpec(label=label, prefix=":".join(parts[:-1]), share=share, max_passes=passes)


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def parse_chunk_caps(text: str, labels: Sequence[str]) -> Dict[str, int]:
    caps = {label: 0 for label in labels}
    if not text:
        return caps
    for item in text.split(","):
        label, sep, value = item.partition("=")
        if not sep or label not in caps or not value.isdigit():
            raise ValueError(f"--max-chunks-per-doc {item!r}: expected label=K with a --slice label")
        caps[label] = int(value)
    return caps


class SliceSource:
    """One input split, memmapped. Evidence parts are present iff ``{prefix}.ev`` exists."""

    def __init__(self, spec: SliceSpec):
        self.spec = spec
        path = lambda suffix: f"{spec.prefix}.{suffix}"
        for suffix in PLAIN_SUFFIXES:
            if not os.path.isfile(path(suffix)):
                raise FileNotFoundError(f"{path(suffix)} is missing (slice {spec.label})")
        self.bin = np.memmap(path("bin"), dtype=np.uint16, mode="r")
        self.mask = np.memmap(path("mask"), dtype=np.uint8, mode="r")
        self.idx = np.fromfile(path("idx"), dtype=np.uint64)
        self.n_docs = len(self.idx) - 1
        assert len(self.bin) == len(self.mask) == int(self.idx[-1]), f"{spec.prefix}: bin, mask, idx disagree"
        self.has_evidence = os.path.isfile(path("ev"))
        self.has_labels = False
        if self.has_evidence:
            self.ev = np.memmap(path("ev"), dtype=np.uint16, mode="r")
            self.evchunk = np.memmap(path("evchunk"), dtype=np.uint16, mode="r")
            self.evkey = np.memmap(path("evkey"), dtype=np.float16, mode="r").reshape(-1, EMBED_DIM)
            self.evidx = np.fromfile(path("evidx"), dtype=np.uint64)
            self.evkeyidx = np.fromfile(path("evkeyidx"), dtype=np.uint64)
            assert len(self.evidx) == len(self.evkeyidx) == len(self.idx), f"{spec.prefix}: indices disagree"
            assert len(self.ev) == len(self.evchunk) == int(self.evidx[-1]), f"{spec.prefix}: ev disagrees"
            assert len(self.evkey) == int(self.evkeyidx[-1]), f"{spec.prefix}: evkey disagrees"
            present = [os.path.isfile(path(s)) for s in ("evgold", "cond", "ans")]
            if any(present) and not all(present):
                raise ValueError(f"{spec.prefix}: evgold, cond and ans must all exist or all be absent")
            self.has_labels = all(present)
            if self.has_labels:
                self.evgold = np.memmap(path("evgold"), dtype=np.uint8, mode="r")
                self.cond = np.memmap(path("cond"), dtype=np.uint8, mode="r")
                self.ans = np.memmap(path("ans"), dtype=np.uint8, mode="r")
                assert len(self.evgold) == len(self.evkey) and len(self.cond) == len(self.ans) == self.n_docs
            else:
                logger.warning(f"slice {spec.label}: no evgold/cond/ans, using gold 0, condition none, "
                               f"answerable 1")

    @property
    def lengths(self) -> np.ndarray:
        return (self.idx[1:] - self.idx[:-1]).astype(np.int64)


def select_documents(lengths: np.ndarray, target_tokens: int, max_passes: int, seed: int,
                     label: str) -> np.ndarray:
    """Returns the slice's documents in consumption order.

    Args:
        lengths: prompt tokens per document of the slice.
        target_tokens: token target; a document that would cross it is not taken.
        max_passes: permutations of the slice that may be consumed.
        seed: run seed.
        label: slice label; with the seed and the document count it fixes the permutations.
    """
    rng = np.random.default_rng([seed, zlib.crc32(label.encode())])
    remaining = int(target_tokens)
    taken: List[np.ndarray] = []
    for _ in range(max_passes):
        perm = rng.permutation(len(lengths))
        perm = perm[lengths[perm] > 0]
        cum = np.cumsum(lengths[perm])
        k = int(np.searchsorted(cum, remaining, side="right"))
        taken.append(perm[:k])
        if k < len(perm):
            break
        remaining -= int(cum[-1]) if k else 0
    return np.concatenate(taken) if taken else np.zeros(0, dtype=np.int64)


def interleave(planned_tokens: List[List[int]], seed: int) -> np.ndarray:
    """Returns the slice index of each output document.

    Args:
        planned_tokens: per slice, the prompt tokens of its selected documents in order.
        seed: run seed for the order generator.
    """
    rng = np.random.default_rng(seed)
    remaining = [float(sum(t)) for t in planned_tokens]
    counts = [len(t) for t in planned_tokens]
    pos = [0] * len(counts)
    total = sum(counts)
    order = np.empty(total, dtype=np.uint8)
    uniforms = rng.random(total)
    for n in range(total):
        live = [i for i in range(len(counts)) if pos[i] < counts[i]]
        if len(live) == 1:
            pick = live[0]
        else:
            weight_sum = sum(remaining[i] for i in live)
            x = uniforms[n] * weight_sum
            pick = live[-1]
            for i in live:
                x -= remaining[i]
                if x < 0:
                    pick = i
                    break
        order[n] = pick
        remaining[pick] -= planned_tokens[pick][pos[pick]]
        pos[pick] += 1
    return order


def cap_chunks(gold: np.ndarray, cap: int) -> np.ndarray:
    """Boolean keep mask over a document's chunks: every gold chunk, then distractors up to ``cap``."""
    keep = gold > 0
    room = max(cap - int(keep.sum()), 0)
    distractors = np.flatnonzero(~keep)
    keep[distractors[:room]] = True
    return keep


def output_paths(out_dir: str, split: str) -> List[str]:
    return [os.path.join(out_dir, f"{split}.{s}") for s in set(EVIDENCE_SUFFIXES) | set(SIDE_FILES)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--target-tokens", type=int, required=True, help="merged prompt token target")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slice", dest="slices", action="append", type=parse_slice, required=True,
                   metavar="LABEL=PREFIX:SHARE[:PASSES]")
    p.add_argument("--no-evidence", action="store_true", help="write only bin idx mask")
    p.add_argument("--max-chunks-per-doc", default="", metavar="LABEL=K,...",
                   help="per slice cap on chunks kept per document (0 = none); gold always kept")
    p.add_argument("--max-evidence-tokens", type=int, default=4608,
                   help="the trainer drops documents above this; only counted in the report")
    p.add_argument("--order-from", default=None, metavar="SLICE_FILE",
                   help="reuse the document order of an earlier merge's .slice file instead of "
                        "drawing one; the per slice document counts must match")
    p.add_argument("--overwrite", action="store_true")
    return p


def load_order(path: str, counts: Sequence[int]) -> np.ndarray:
    """Reads a ``.slice`` file and checks it holds exactly ``counts[i]`` documents of each slice.

    Args:
        path: the earlier merge's ``{split}.slice`` file.
        counts: documents selected per slice in this merge, in ``--slice`` order.
    """
    order = np.fromfile(path, dtype=np.uint8)
    found = np.bincount(order, minlength=len(counts)).tolist()
    if found != list(counts):
        raise ValueError(f"{path} holds {found} documents per slice, this merge selects {list(counts)}")
    return order


def merge(args: argparse.Namespace) -> dict:
    labels = [s.label for s in args.slices]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate slice labels {labels}")
    if len(labels) > 255:
        raise ValueError("at most 255 slices")
    caps = parse_chunk_caps(args.max_chunks_per_doc, labels)
    existing = [p for p in output_paths(args.out_dir, args.split) if os.path.exists(p)]
    if existing and not args.overwrite:
        raise FileExistsError(f"{existing[0]} exists (and {len(existing) - 1} more); pass --overwrite")
    os.makedirs(args.out_dir, exist_ok=True)
    for p in existing:
        os.remove(p)

    sources = [SliceSource(s) for s in args.slices]
    selected = []
    for spec, src in zip(args.slices, sources):
        target = int(spec.share * args.target_tokens)
        selected.append(select_documents(src.lengths, target, spec.max_passes, args.seed, spec.label))
    lengths = [src.lengths for src in sources]
    planned = [lengths[i][selected[i]].tolist() for i in range(len(sources))]
    if args.order_from:
        order = load_order(args.order_from, [len(s) for s in selected])
    else:
        order = interleave(planned, args.seed)
    n_docs = len(order)

    evidence = not args.no_evidence
    suffixes = EVIDENCE_SUFFIXES if evidence else PLAIN_SUFFIXES
    stream_suffixes = [s for s in suffixes if s not in ("idx", "evidx", "evkeyidx", "cond", "ans")]
    files = {s: open(os.path.join(args.out_dir, f"{args.split}.{s}"), "wb", buffering=1 << 22)
             for s in stream_suffixes}
    idx_out = np.zeros(n_docs + 1, dtype=np.uint64)
    evidx_out = np.zeros(n_docs + 1, dtype=np.uint64)
    evkeyidx_out = np.zeros(n_docs + 1, dtype=np.uint64)
    cond_out = np.full(n_docs, NONE_CONDITION, dtype=np.uint8)
    ans_out = np.ones(n_docs, dtype=np.uint8)
    md5 = [hashlib.md5() for _ in sources]
    stats = [dict(documents=0, prompt_tokens=0, evidence_tokens=0, chunks=0,
                  over_evidence_cap=0, over_chunk_limit=0) for _ in sources]
    pos = [0] * len(sources)
    tokens = ev_tokens = chunks = 0

    for n in range(n_docs):
        s = int(order[n])
        src, st = sources[s], stats[s]
        d = int(selected[s][pos[s]])
        pos[s] += 1
        a, b = int(src.idx[d]), int(src.idx[d + 1])
        ids = src.bin[a:b]
        md5[s].update(ids)
        e_ids = e_chunk = e_keys = e_gold = None
        n_ev = n_chunks = 0
        if src.has_evidence:
            ea, eb = int(src.evidx[d]), int(src.evidx[d + 1])
            ka, kb = int(src.evkeyidx[d]), int(src.evkeyidx[d + 1])
            e_ids, e_chunk, e_keys = src.ev[ea:eb], src.evchunk[ea:eb], src.evkey[ka:kb]
            e_gold = src.evgold[ka:kb] if src.has_labels else np.zeros(kb - ka, dtype=np.uint8)
            if caps[src.spec.label] and kb - ka > caps[src.spec.label]:
                keep = cap_chunks(np.asarray(e_gold), caps[src.spec.label])
                new_id = np.cumsum(keep) - 1
                token_keep = keep[np.asarray(e_chunk)]
                e_ids = np.asarray(e_ids)[token_keep]
                e_chunk = new_id[np.asarray(e_chunk)][token_keep].astype(np.uint16)
                e_keys, e_gold = e_keys[keep], np.asarray(e_gold)[keep]
            n_ev, n_chunks = len(e_ids), len(e_gold)
            md5[s].update(e_ids)
            if src.has_labels:
                cond_out[n], ans_out[n] = src.cond[d], src.ans[d]
        tokens += b - a
        ev_tokens += n_ev
        chunks += n_chunks
        idx_out[n + 1], evidx_out[n + 1], evkeyidx_out[n + 1] = tokens, ev_tokens, chunks
        st["documents"] += 1
        st["prompt_tokens"] += b - a
        st["evidence_tokens"] += n_ev
        st["chunks"] += n_chunks
        st["over_evidence_cap"] += int(n_ev > args.max_evidence_tokens)
        st["over_chunk_limit"] += int(n_chunks > MAX_CHUNKS_PER_SEGMENT)
        files["bin"].write(ids)
        files["mask"].write(src.mask[a:b])
        if evidence and n_chunks + n_ev:
            files["ev"].write(e_ids)
            files["evchunk"].write(e_chunk)
            files["evkey"].write(np.ascontiguousarray(e_keys))
            files["evgold"].write(np.asarray(e_gold, dtype=np.uint8))

    for f in files.values():
        f.close()
    out = lambda suffix: os.path.join(args.out_dir, f"{args.split}.{suffix}")
    idx_out.tofile(out("idx"))
    if evidence:
        evidx_out.tofile(out("evidx"))
        evkeyidx_out.tofile(out("evkeyidx"))
        cond_out.tofile(out("cond"))
        ans_out.tofile(out("ans"))
    order.tofile(out("slice"))

    for spec, src, st, h in zip(args.slices, sources, stats, md5):
        st.update(label=spec.label, source_documents=src.n_docs, share_target=spec.share,
                  passes=st["documents"] / max(src.n_docs, 1), max_passes=spec.max_passes,
                  target_tokens=int(spec.share * args.target_tokens), max_chunks=caps[spec.label],
                  evidence_ratio=st["evidence_tokens"] / max(st["prompt_tokens"], 1),
                  share_realised=st["prompt_tokens"] / max(tokens, 1), md5=h.hexdigest())
    result = dict(
        args=dict(out_dir=args.out_dir, split=args.split, target_tokens=args.target_tokens,
                  seed=args.seed, slices=[f"{s.label}={s.prefix}:{s.share}:{s.max_passes}"
                                          for s in args.slices],
                  no_evidence=args.no_evidence, max_chunks_per_doc=args.max_chunks_per_doc,
                  max_evidence_tokens=args.max_evidence_tokens, order_from=args.order_from),
        slices=stats, md5_by_slice={st["label"]: st["md5"] for st in stats},
        documents=n_docs, prompt_tokens=tokens, evidence_tokens=ev_tokens, chunks=chunks,
        evidence_ratio=ev_tokens / max(tokens, 1), gap_to_target=args.target_tokens - tokens,
    )
    with open(out("merge.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def report(result: dict, args: argparse.Namespace) -> None:
    lines = [f"merged split {args.split} in {args.out_dir}"]
    for st in result["slices"]:
        lines.append(
            f"  {st['label']}: {st['documents']:,} docs ({st['passes']:.3f} passes of "
            f"{st['source_documents']:,}, cap {st['max_passes']}), {st['prompt_tokens']:,} prompt "
            f"tokens (target {st['target_tokens']:,}), {st['evidence_tokens']:,} evidence tokens, "
            f"{st['chunks']:,} chunks, ratio {st['evidence_ratio']:.3f}, share "
            f"{st['share_realised']:.3f} vs target {st['share_target']:.3f}, "
            f"{st['over_evidence_cap']:,} docs over {args.max_evidence_tokens} evidence tokens, "
            f"{st['over_chunk_limit']:,} docs over {MAX_CHUNKS_PER_SEGMENT} chunks "
            f"(MAX_CHUNKS_PER_SEGMENT)")
    lines.append(
        f"  total: {result['documents']:,} docs, {result['prompt_tokens']:,} prompt tokens, "
        f"{result['evidence_tokens']:,} evidence tokens, ratio {result['evidence_ratio']:.3f}, "
        f"gap to --target-tokens {result['gap_to_target']:,}")
    for p in sorted(output_paths(args.out_dir, args.split)):
        if os.path.exists(p):
            lines.append(f"  {os.path.basename(p)}: {os.path.getsize(p):,} bytes")
    for line in lines:
        logger.info(line)
        print(line)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    result = merge(args)
    report(result, args)
    return result


if __name__ == "__main__":
    main()

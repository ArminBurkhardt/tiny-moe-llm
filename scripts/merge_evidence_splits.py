"""Merge several evidence-format splits into one, at fixed token shares, from a seeded stream.

The inputs are splits written by the corpus builders (``prepare_evidence_data.py``,
``prepare_injection_data.py``, ``prepare_chain_data.py``). Each is one *slice* of the output with a
label, a share of the target prompt tokens and a pass cap. An input with only ``bin idx mask`` is a
plain slice: it contributes prompt tokens and no evidence. The output is one split the trainer reads
as it reads any other: ``EvidenceDataset`` when ``{split}.ev`` exists, ``SFTDataset`` otherwise.

Output files (the sidecars ``factspan``, ``evhop`` and ``chains.jsonl`` are not merged):

    evidence form   bin idx mask ev evidx evchunk evkey evkeyidx evgold cond ans
    --no-evidence   bin idx mask
    always          slice (uint8 per document: the slice index in ``--slice`` order), merge.json
    all or none     src (uint8 per document, the source index ``prepare_evidence_data`` writes)

``src`` is carried only when every input slice has ``{prefix}.src``, and then copied per document
unchanged. A merge where some slice lacks it writes no ``src`` at all, because the trainer reads the
file for every document or not at all, and a filler value for the slices without one would print a
per source line for a source that was never labelled. Today only the held-out evidence splits (and
the ``--sources-only`` backfill) write ``src``.

``merge.json`` also reports, per slice and in total, the documents the evidence reader drops as too
long for a row (``over_evidence_reader_row_limit``): the reader packs a row to ``seq_length - 1``
positions and a document costs its prompt tokens, one more when it does not start with BOS, plus
``num_mtp_tokens`` separator slots, so a document whose cost exceeds ``seq_length - 1`` is never
trained on by an evidence arm (the plain reader keeps it). ``--seq-length``, ``--num-mtp-tokens``
and ``--bos-id`` set that rule; nothing is dropped by the merge itself. ``md5_mask_by_slice`` is
the md5 of each slice's ``.mask`` bytes in output order, so a merge that differs from another only
in the loss mask is caught next to ``md5_by_slice``.

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
then select the same number of documents in both merges. Input files are memmapped and written in
blocks (a run of consecutive documents of one slice with consecutive source indices is copied as one
memmap slice), so a slice never has to fit in memory.

``--order-prefix-from SLICE_FILE`` builds a shorter merge whose documents are the start of a longer
merge's. It reads the longer merge's ``.slice`` order from the start and stops at the first document
boundary where this merge's prompt tokens reach ``--target-tokens``. Slice ``i``'s k-th document is
document ``k`` of its input in storage order (no permutation, no share or pass cap), so an input may
be a shorter build that is a prefix by index of the slice the longer merge used; a slice that runs out
of documents, or an order that ends, before the target is an error. The output ``.slice`` is the prefix
of the recorded order and ``merge.json`` gains ``num_docs`` (equal to ``documents``), ``order_prefix``
and ``order_prefix_from``. It works with ``--no-evidence`` and is exclusive with ``--order-from``.

Resume: every ``--checkpoint-seconds`` (default 300; ``--checkpoint-docs`` also by count) the merge
fsyncs its output files and writes ``{split}.merge_state.json`` (document cursor, byte offset of every
output file, a hash of the arguments that decide the bytes). Relaunched with the same arguments it
truncates each file to its offset, replays the finished documents to rebuild the per slice md5 digests
and counters (no writes), and continues; the files and ``merge.json`` are byte identical to an
uninterrupted merge. Other arguments are refused; ``--no-resume`` starts fresh (and overwrites). A
finished merge removes the state file. SIGTERM saves the state and exits 20.

Example:
    python scripts/merge_evidence_splits.py --out-dir data/prepared_pilot --split pilot_main_train \\
        --target-tokens 1000000000 --seed 42 \\
        --slice bios=data/prepared_pilot/inject_retrieval_train:0.60 \\
        --slice qa=data/prepared/evidence_nomany_train:0.25:2 \\
        --slice chains=data/prepared/chains_train:0.10:1
    # a shorter control that is the start of that merge, rerun with the same arguments after a kill
    python scripts/merge_evidence_splits.py --out-dir data/prepared_pilot --split pilot_control_train \\
        --target-tokens 100000000 --no-evidence \\
        --order-prefix-from data/prepared_pilot/pilot_main_train.slice \\
        --slice bios=... --slice qa=... --slice chains=...
"""
import os
import sys
import json
import zlib
import time
import signal
import hashlib
import argparse
import threading
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
SOURCE_FILE = "src"


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
        self.has_src = os.path.isfile(path(SOURCE_FILE))
        if self.has_src:
            self.src = np.memmap(path(SOURCE_FILE), dtype=np.uint8, mode="r")
            assert len(self.src) == self.n_docs, f"{spec.prefix}: src disagrees with idx"
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
    suffixes = set(EVIDENCE_SUFFIXES) | set(SIDE_FILES) | {SOURCE_FILE}
    return [os.path.join(out_dir, f"{split}.{s}") for s in suffixes]


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
    p.add_argument("--seq-length", type=int, default=4096,
                   help="the trainer's row length; only read for the evidence reader's row limit report")
    p.add_argument("--num-mtp-tokens", type=int, default=2,
                   help="separator slots the reader adds per document (mtp_num_extra_tokens); report only")
    p.add_argument("--bos-id", type=int, default=0,
                   help="the reader prepends this to a document that does not start with it; report only")
    order = p.add_mutually_exclusive_group()
    order.add_argument("--order-from", default=None, metavar="SLICE_FILE",
                       help="reuse the document order of an earlier merge's .slice file instead of "
                            "drawing one; the per slice document counts must match")
    order.add_argument("--order-prefix-from", default=None, metavar="SLICE_FILE",
                       help="write the document order of an earlier merge's .slice file from its "
                            "start and stop at the first document boundary where this merge's prompt "
                            "tokens reach --target-tokens; each slice's documents are taken by index "
                            "in sequence (no permutation, no share), so a slice input may be a "
                            "shorter build that is a prefix of the earlier merge's slice")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore a saved merge state and start fresh (implies --overwrite)")
    p.add_argument("--checkpoint-seconds", type=float, default=300.0,
                   help="seconds between merge state saves (0 = only at the end of a block)")
    p.add_argument("--checkpoint-docs", type=int, default=0,
                   help="also save the merge state every this many documents (0 = off)")
    p.add_argument("--stop-after-docs", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--no-block-copy", action="store_true", help=argparse.SUPPRESS)
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


def prefix_length(path: str, sources: Sequence["SliceSource"], target_tokens: int) -> int:
    """Documents of the recorded order that bring the merge's prompt tokens to ``target_tokens``.

    Slice ``i``'s k-th document in the order is document ``k`` of ``sources[i]``. Returns the first
    count whose token sum is at least the target.

    Args:
        path: the earlier merge's ``{split}.slice`` file.
        sources: the slice inputs in ``--slice`` order.
        target_tokens: merged prompt token target.

    Raises:
        ValueError: a slice runs out of documents, or the order ends, before the target.
    """
    order = np.memmap(path, dtype=np.uint8, mode="r")
    if len(order) and int(order.max()) >= len(sources):
        raise ValueError(f"{path} names slice {int(order.max())}, this merge has {len(sources)} slices")
    pos = [0] * len(sources)
    total = 0
    step = 1 << 23
    for a in range(0, len(order), step):
        sl = np.asarray(order[a:a + step])
        lim = len(sl)
        masks = [sl == s for s in range(len(sources))]
        short = None
        for s, src in enumerate(sources):
            k = int(masks[s].sum())
            if pos[s] + k > src.n_docs:
                cut = int(np.flatnonzero(masks[s])[src.n_docs - pos[s]])
                if cut < lim:
                    lim, short = cut, s
        lens = np.zeros(lim, dtype=np.int64)
        used = [0] * len(sources)
        for s, src in enumerate(sources):
            m = masks[s][:lim]
            k = int(m.sum())
            if k:
                lo = pos[s]
                lens[m] = (src.idx[lo + 1:lo + k + 1] - src.idx[lo:lo + k]).astype(np.int64)
            used[s] = k
        cum = total + np.cumsum(lens)
        hit = int(np.searchsorted(cum, target_tokens, side="left"))
        if hit < lim:
            return a + hit + 1
        if short is not None:
            raise ValueError(
                f"slice {sources[short].spec.label} has {sources[short].n_docs:,} documents and the order "
                f"in {path} needs more before the target of {target_tokens:,} prompt tokens is reached "
                f"(reached {int(cum[-1]) if lim else total:,} tokens at {a + lim:,} output documents)")
        total = int(cum[-1]) if lim else total
        for s in range(len(sources)):
            pos[s] += used[s]
    raise ValueError(f"the order in {path} holds {len(order):,} documents and {total:,} prompt tokens, "
                     f"below the target of {target_tokens:,}")


class MergeInterrupted(Exception):
    """The merge saved its state and stopped; the caller exits 20 so the job is rerun."""


STOP_REQUESTED = False
STATE_VERSION = 1


def _request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def state_path_for(out_dir: str, split: str) -> str:
    return os.path.join(out_dir, f"{split}.merge_state.json")


def order_file_fingerprint(path: Optional[str]):
    """Size plus a hash of the first and last MB of an order file, None for no file."""
    if not path:
        return None
    size = os.path.getsize(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 1 << 20:
            f.seek(max(size - (1 << 20), 1 << 20))
            h.update(f.read(1 << 20))
    return [size, h.hexdigest()]


def args_digest(args: argparse.Namespace) -> str:
    """Hash of everything that decides the output bytes (not the paths of the outputs).

    The order files count by size and a hash of their first and last MB, not by path.
    """
    sizes = []
    for s in args.slices:
        sizes.append([os.path.getsize(f"{s.prefix}.idx"), os.path.getsize(f"{s.prefix}.bin")])
    payload = dict(
        target_tokens=args.target_tokens, seed=args.seed,
        slices=[f"{s.label}={s.prefix}:{s.share}:{s.max_passes}" for s in args.slices], sizes=sizes,
        no_evidence=args.no_evidence, max_chunks_per_doc=args.max_chunks_per_doc,
        max_evidence_tokens=args.max_evidence_tokens, seq_length=args.seq_length,
        num_mtp_tokens=args.num_mtp_tokens, bos_id=args.bos_id,
        order_from_file=order_file_fingerprint(args.order_from),
        order_prefix_from_file=order_file_fingerprint(args.order_prefix_from))
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_state_atomic(state_path: str, state: dict) -> None:
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, state_path)


BLOCK_DOCS = 1 << 16


def _output_sizes(out_dir: str, split: str) -> Dict[str, int]:
    """Byte size of every output file of the split except ``merge.json`` itself."""
    sizes = {}
    for p in output_paths(out_dir, split):
        if os.path.exists(p) and not p.endswith(".merge.json"):
            sizes[os.path.basename(p)] = os.path.getsize(p)
    return sizes


def completed_result(args: argparse.Namespace) -> Optional[dict]:
    """The stored ``merge.json`` when it vouches for the files on disk, else None.

    It vouches when it carries this merge's argument digest and the recorded size of every output
    file matches. ``merge.json`` is written after every other output and before the state file is
    removed, so a merge killed between the two is recognised here.
    """
    path = os.path.join(args.out_dir, f"{args.split}.merge.json")
    try:
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, ValueError):
        return None
    if result.get("args_digest") != args_digest(args) or not result.get("output_bytes"):
        return None
    if result["output_bytes"] != _output_sizes(args.out_dir, args.split):
        return None
    return result


def merge(args: argparse.Namespace) -> dict:
    labels = [s.label for s in args.slices]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate slice labels {labels}")
    if len(labels) > 255:
        raise ValueError("at most 255 slices")
    caps = parse_chunk_caps(args.max_chunks_per_doc, labels)
    state_path = state_path_for(args.out_dir, args.split)
    if args.no_resume:
        args.overwrite = True
        if os.path.exists(state_path):
            os.remove(state_path)
    saved = None
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("version") != STATE_VERSION or saved.get("args_digest") != args_digest(args):
            raise ValueError(f"{state_path} was written by a merge with other arguments; "
                             f"pass --no-resume to start fresh")
    if not args.no_resume and (saved is not None or not args.overwrite):
        done_result = completed_result(args)
        if done_result is not None:
            if os.path.exists(state_path):
                os.remove(state_path)
            logger.info(f"already complete: {args.split} in {args.out_dir} matches its merge.json for "
                        f"these arguments; nothing touched")
            print(f"already complete: {args.split} in {args.out_dir}")
            return done_result
    existing = [p for p in output_paths(args.out_dir, args.split) if os.path.exists(p)]
    if existing and not args.overwrite and saved is None:
        raise FileExistsError(f"{existing[0]} exists (and {len(existing) - 1} more); pass --overwrite")
    os.makedirs(args.out_dir, exist_ok=True)
    if saved is None:
        for p in existing:
            os.remove(p)

    sources = [SliceSource(s) for s in args.slices]
    prefix_mode = bool(args.order_prefix_from)
    selected: List[Optional[np.ndarray]]
    if prefix_mode:
        n_docs = prefix_length(args.order_prefix_from, sources, args.target_tokens)
        order = np.memmap(args.order_prefix_from, dtype=np.uint8, mode="r")[:n_docs]
        selected = [None] * len(sources)
    else:
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
    carry_src = all(src.has_src for src in sources)
    stream_suffixes = list(suffixes)
    if carry_src:
        stream_suffixes.append(SOURCE_FILE)
    out_path = lambda suffix: os.path.join(args.out_dir, f"{args.split}.{suffix}")
    files = {}
    done = 0
    if saved is None:
        for s in stream_suffixes:
            files[s] = open(out_path(s), "wb", buffering=1 << 22)
        zero = np.zeros(1, dtype=np.uint64).tobytes()
        for s in ("idx", "evidx", "evkeyidx"):
            if s in files:
                files[s].write(zero)
    else:
        offsets = saved["offsets"]
        if sorted(offsets) != sorted(stream_suffixes):
            raise ValueError(f"{state_path} lists other output files than this merge writes")
        for s in stream_suffixes:
            p = out_path(s)
            size = os.path.getsize(p) if os.path.exists(p) else -1
            if size < offsets[s]:
                raise ValueError(f"{p} is {size} bytes, shorter than the saved {offsets[s]}")
            if size > offsets[s]:
                os.truncate(p, offsets[s])
                logger.warning(f"truncated {p} from {size} to {offsets[s]} bytes "
                               f"(dropping the uncommitted tail of an interrupted merge)")
            files[s] = open(p, "ab", buffering=1 << 22)
        done = int(saved["documents_done"])
        if done > n_docs:
            raise ValueError(f"{state_path} is past the end of this merge ({done} > {n_docs})")
        logger.info(f"resuming {args.split} at document {done:,} of {n_docs:,}; replaying the "
                    f"digests of the finished part")

    md5 = [hashlib.md5() for _ in sources]
    md5_mask = [hashlib.md5() for _ in sources]
    row_limit = args.seq_length - 1
    over_row = [dict(documents=0, tokens=0) for _ in sources]
    stats = [dict(documents=0, prompt_tokens=0, evidence_tokens=0, chunks=0,
                  over_evidence_cap=0, over_chunk_limit=0) for _ in sources]
    pos = [0] * len(sources)
    counters = dict(tokens=0, ev_tokens=0, chunks=0)
    block_copy = not args.no_block_copy
    min_block = 2

    def take(s: int, k: int) -> np.ndarray:
        lo = pos[s]
        pos[s] += k
        if selected[s] is None:
            return np.arange(lo, lo + k, dtype=np.int64)
        return np.asarray(selected[s][lo:lo + k], dtype=np.int64)

    def doc_path(s, d, i, bufs, write):
        src, st = sources[s], stats[s]
        a, b = int(src.idx[d]), int(src.idx[d + 1])
        ids = src.bin[a:b]
        md5[s].update(ids)
        mask = src.mask[a:b]
        md5_mask[s].update(mask)
        if b > a and (b - a) + int(int(ids[0]) != args.bos_id) + args.num_mtp_tokens > row_limit:
            over_row[s]["documents"] += 1
            over_row[s]["tokens"] += b - a
        if carry_src:
            bufs["src"][i] = src.src[d]
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
                bufs["cond"][i], bufs["ans"][i] = src.cond[d], src.ans[d]
        counters["tokens"] += b - a
        counters["ev_tokens"] += n_ev
        counters["chunks"] += n_chunks
        bufs["idx"][i] = counters["tokens"]
        bufs["evidx"][i] = counters["ev_tokens"]
        bufs["evkeyidx"][i] = counters["chunks"]
        st["documents"] += 1
        st["prompt_tokens"] += b - a
        st["evidence_tokens"] += n_ev
        st["chunks"] += n_chunks
        st["over_evidence_cap"] += int(n_ev > args.max_evidence_tokens)
        st["over_chunk_limit"] += int(n_chunks > MAX_CHUNKS_PER_SEGMENT)
        if write:
            files["bin"].write(ids)
            files["mask"].write(mask)
            if evidence and n_chunks + n_ev:
                files["ev"].write(e_ids)
                files["evchunk"].write(e_chunk)
                files["evkey"].write(np.ascontiguousarray(e_keys))
                files["evgold"].write(np.asarray(e_gold, dtype=np.uint8))

    def block_path(s, d0, count, i0, bufs, write):
        src, st = sources[s], stats[s]
        d1 = d0 + count
        a, b = int(src.idx[d0]), int(src.idx[d1])
        ids, mask = src.bin[a:b], src.mask[a:b]
        starts = src.idx[d0:d1 + 1].astype(np.int64)
        dl = np.diff(starts)
        md5_mask[s].update(mask)
        nonempty = dl > 0
        if nonempty.any():
            first = np.asarray(src.bin[starts[:-1][nonempty]]).astype(np.int64)
            cost = dl[nonempty] + (first != args.bos_id) + args.num_mtp_tokens
            over = cost > row_limit
            over_row[s]["documents"] += int(over.sum())
            over_row[s]["tokens"] += int(dl[nonempty][over].sum())
        if carry_src:
            bufs["src"][i0:i0 + count] = src.src[d0:d1]
        evl = np.zeros(count, dtype=np.int64)
        kl = np.zeros(count, dtype=np.int64)
        e_ids = e_chunk = e_keys = e_gold = None
        if src.has_evidence:
            ea, eb = int(src.evidx[d0]), int(src.evidx[d1])
            ka, kb = int(src.evkeyidx[d0]), int(src.evkeyidx[d1])
            e_ids, e_chunk, e_keys = src.ev[ea:eb], src.evchunk[ea:eb], src.evkey[ka:kb]
            evs = src.evidx[d0:d1 + 1].astype(np.int64)
            evl, kl = np.diff(evs), np.diff(src.evkeyidx[d0:d1 + 1].astype(np.int64))
            if src.has_labels:
                e_gold = src.evgold[ka:kb]
                bufs["cond"][i0:i0 + count] = src.cond[d0:d1]
                bufs["ans"][i0:i0 + count] = src.ans[d0:d1]
            else:
                e_gold = np.zeros(kb - ka, dtype=np.uint8)
            if eb > ea:
                so, eo = (starts - starts[0]).tolist(), (evs - evs[0]).tolist()
                h = md5[s]
                for k in range(count):
                    h.update(ids[so[k]:so[k + 1]])
                    if eo[k + 1] > eo[k]:
                        h.update(e_ids[eo[k]:eo[k + 1]])
            else:
                md5[s].update(ids)
        else:
            md5[s].update(ids)
        bufs["idx"][i0:i0 + count] = counters["tokens"] + np.cumsum(dl)
        bufs["evidx"][i0:i0 + count] = counters["ev_tokens"] + np.cumsum(evl)
        bufs["evkeyidx"][i0:i0 + count] = counters["chunks"] + np.cumsum(kl)
        n_tok, n_ev, n_ch = b - a, int(evl.sum()), int(kl.sum())
        counters["tokens"] += n_tok
        counters["ev_tokens"] += n_ev
        counters["chunks"] += n_ch
        st["documents"] += count
        st["prompt_tokens"] += n_tok
        st["evidence_tokens"] += n_ev
        st["chunks"] += n_ch
        st["over_evidence_cap"] += int((evl > args.max_evidence_tokens).sum())
        st["over_chunk_limit"] += int((kl > MAX_CHUNKS_PER_SEGMENT).sum())
        if write:
            files["bin"].write(ids)
            files["mask"].write(mask)
            if evidence and n_ch + n_ev:
                files["ev"].write(e_ids)
                files["evchunk"].write(e_chunk)
                files["evkey"].write(np.ascontiguousarray(e_keys))
                files["evgold"].write(np.asarray(e_gold, dtype=np.uint8))

    def process(a: int, b: int, write: bool) -> None:
        m = b - a
        sl = np.asarray(order[a:b])
        dd = np.empty(m, dtype=np.int64)
        for s in range(len(sources)):
            msk = sl == s
            k = int(msk.sum())
            if k:
                dd[msk] = take(s, k)
        bufs = dict(idx=np.zeros(m, dtype=np.uint64), evidx=np.zeros(m, dtype=np.uint64),
                    evkeyidx=np.zeros(m, dtype=np.uint64),
                    cond=np.full(m, NONE_CONDITION, dtype=np.uint8), ans=np.ones(m, dtype=np.uint8))
        if carry_src:
            bufs["src"] = np.zeros(m, dtype=np.uint8)
        if m > 1:
            brk = np.flatnonzero((sl[1:] != sl[:-1]) | (dd[1:] != dd[:-1] + 1)) + 1
        else:
            brk = np.zeros(0, dtype=np.int64)
        run_starts = np.concatenate([[0], brk]).tolist()
        run_ends = np.concatenate([brk, [m]]).tolist()
        for i0, i1 in zip(run_starts, run_ends):
            s = int(sl[i0])
            if block_copy and i1 - i0 >= min_block and not caps[sources[s].spec.label]:
                block_path(s, int(dd[i0]), i1 - i0, i0, bufs, write)
            else:
                for i in range(i0, i1):
                    doc_path(s, int(dd[i]), i, bufs, write)
        if write:
            bufs["idx"].tofile(files["idx"])
            if evidence:
                bufs["evidx"].tofile(files["evidx"])
                bufs["evkeyidx"].tofile(files["evkeyidx"])
                bufs["cond"].tofile(files["cond"])
                bufs["ans"].tofile(files["ans"])
            if carry_src:
                bufs["src"].tofile(files[SOURCE_FILE])

    def checkpoint(n_done: int) -> None:
        offs = {}
        for s, f in files.items():
            f.flush()
            os.fsync(f.fileno())
            offs[s] = f.tell()
        save_state_atomic(state_path, dict(
            version=STATE_VERSION, args_digest=args_digest(args), documents_done=n_done,
            offsets=offs, tokens=counters["tokens"], evidence_tokens=counters["ev_tokens"],
            chunks=counters["chunks"]))

    if saved is None:
        checkpoint(0)
    for a in range(0, done, BLOCK_DOCS):
        process(a, min(a + BLOCK_DOCS, done), write=False)
    if saved is not None:
        mine = dict(tokens=counters["tokens"], evidence_tokens=counters["ev_tokens"],
                    chunks=counters["chunks"])
        if mine != {k: saved[k] for k in mine}:
            raise ValueError(f"replay of the first {done:,} documents gives {mine}, the saved state "
                             f"says {({k: saved[k] for k in mine})}: the inputs changed")

    last_save = time.monotonic()
    n = done
    while n < n_docs:
        stop = min(n + BLOCK_DOCS, n_docs)
        if args.checkpoint_docs > 0:
            stop = min(stop, (n // args.checkpoint_docs + 1) * args.checkpoint_docs)
        if args.stop_after_docs > n:
            stop = min(stop, args.stop_after_docs)
        process(n, stop, write=True)
        n = stop
        if n == n_docs:
            break
        hook = args.stop_after_docs and n >= args.stop_after_docs
        due = args.checkpoint_docs > 0 and n % args.checkpoint_docs == 0
        if STOP_REQUESTED or hook or due or time.monotonic() - last_save >= args.checkpoint_seconds:
            checkpoint(n)
            last_save = time.monotonic()
            if STOP_REQUESTED or hook:
                for f in files.values():
                    f.close()
                raise MergeInterrupted(f"stopped at document {n:,} of {n_docs:,}; state in {state_path}")

    for f in files.values():
        f.close()
    order[:n_docs].tofile(out_path("slice"))
    tokens, ev_tokens, chunks = counters["tokens"], counters["ev_tokens"], counters["chunks"]

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
        md5_mask_by_slice={spec.label: h.hexdigest() for spec, h in zip(args.slices, md5_mask)},
        over_evidence_reader_row_limit=dict(
            seq_length=args.seq_length, num_mtp_tokens=args.num_mtp_tokens, bos_id=args.bos_id,
            row_limit=row_limit, documents=sum(o["documents"] for o in over_row),
            tokens=sum(o["tokens"] for o in over_row),
            by_slice={spec.label: o for spec, o in zip(args.slices, over_row)}),
        src_carried=carry_src,
    )
    if prefix_mode:
        result["args"]["order_prefix_from"] = args.order_prefix_from
        result["order_prefix_from"] = args.order_prefix_from
        result["num_docs"] = n_docs
        result["order_prefix"] = dict(documents=n_docs, prompt_tokens=tokens,
                                      documents_per_slice={st["label"]: st["documents"] for st in stats})
    result["args_digest"] = args_digest(args)
    result["output_bytes"] = _output_sizes(args.out_dir, args.split)
    tmp = out_path("merge.json") + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path("merge.json"))
    if os.path.exists(state_path):
        os.remove(state_path)
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
    over = result["over_evidence_reader_row_limit"]
    per_slice = ", ".join(f"{label} {o['documents']:,} docs / {o['tokens']:,} tokens"
                          for label, o in over["by_slice"].items())
    lines.append(
        f"  over the evidence reader's row limit (prompt + BOS if missing + {over['num_mtp_tokens']} "
        f"MTP slots > {over['row_limit']}, dropped by an evidence arm, kept by the plain reader): "
        f"{over['documents']:,} docs, {over['tokens']:,} prompt tokens ({per_slice})")
    lines.append(f"  src carried: {result['src_carried']} (only when every slice has {{prefix}}.src)")
    for p in sorted(output_paths(args.out_dir, args.split)):
        if os.path.exists(p):
            lines.append(f"  {os.path.basename(p)}: {os.path.getsize(p):,} bytes")
    for line in lines:
        logger.info(line)
        print(line)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    global STOP_REQUESTED
    args = build_parser().parse_args(argv)
    STOP_REQUESTED = False
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _request_stop)
    try:
        result = merge(args)
    except MergeInterrupted as stopped:
        logger.warning(str(stopped))
        print(str(stopped))
        sys.exit(20)
    report(result, args)
    return result


if __name__ == "__main__":
    main()

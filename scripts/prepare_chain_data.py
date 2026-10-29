"""Build the chain question splits in the evidence corpus format.

Every row is one chain question (``modules/data/chains.py``): the user turn is the evidence prompt
the port was trained with plus the question, the assistant turn is the answer, and the buffer is
the gold chain plus distractors, attached through the port with the standard eleven files. Rows are
written through ``EvidenceWriter`` exactly as ``prepare_evidence_data.build_heldout`` does, with
``.cond`` ``mixed`` and ``.ans`` 1, so ``EvidenceDataset`` and the evidence trainer read them
unchanged.

Splits, each named ``{prefix}_{name}``:

    train, val        hops 1 / 2 / 3 at 25 / 50 / 25, seen compositions
    eval              hops 1, 2, 3 balanced, seen compositions, fresh entities
    heldout_tmpl      2-hop, held out compositions only
    hop4              4-hop

Two sidecars ride along, both eval-only (no trainer reads them):

    {split}.evhop         uint8 per chunk (the ``.evkey`` / ``.evgold`` axis): 0 distractor, h the
                          gold chunk at hop h
    {split}.chains.jsonl  one line per document: qid, question, answer, answer_type, hops,
                          composition, held_out, candidates, chunk_kind, entities (and bridge,
                          the intermediates e1..e(k-1), under ``--bridge``; and, on the eval splits
                          only, chunk_facts, the [subject, relation, object] of every chunk, which
                          ``eval_chains.py`` reads for the shortcut baselines; a sidecar without it
                          stays readable and gets no baseline)

``--bridge`` writes the written bridge form: the assistant turn names the chain's intermediate
entities before the answer (``chains.bridge_answer``); the questions, chunks and sidecars are the
same as without it.

The train and val splits are streamed: questions are drawn from one seeded generator, encoded,
embedded and written ``--batch`` at a time, so memory does not grow with the split, and the bytes
equal those of drawing every question first. The streamed build is resumable: every
``--checkpoint-seconds`` (default 300) it fsyncs the files and writes
``{split}.build_state.json`` (questions written, the generator state, every file's byte offset, the
report counters, a signature of the arguments and the training hop mix). A relaunch with the same
arguments truncates the files to the offsets and continues; other arguments are refused; SIGTERM
saves the state and exits 20. A finished split removes the state and leaves ``{split}.build.json``
(the signature, the question count and every file's size); a relaunch without ``--overwrite`` that
finds it matching and the sizes intact says "already complete" and exits 0. ``--overwrite`` discards
a state and starts over. The ``.chains.jsonl`` sidecar of the train mix is about 0.58 KB per question (16 GB for 28M questions).

The 200k question train split is about 2.6M short chunks and wants the GPU for the embedder; the
eval splits embed on CPU in minutes (``--device cpu``).

Usage:

    python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains \\
        --train-questions 200000 --val-questions 2000 --eval-questions-per-hop 1000 --seed 42 \\
        --device cuda [--splits eval,heldout_tmpl,hop4] [--overwrite] [--bridge]

    python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains_bridge \\
        --splits train,val --train-questions 200000 --val-questions 2000 --seed 42 --device cuda \\
        --bridge
"""
import os
import sys
import json
import time
import random
import signal
import argparse
import itertools
from collections import Counter
from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.data import chains
from modules.data.chat import ChatTemplate
from scripts.prepare_evidence_data import (
    CONDITIONS, EVIDENCE_SUFFIXES, ChunkEmbedder, EvidenceWriter, evidence_prompt,
)
from utils import BASE_DIR, TOKENIZER_DIR, logger

SPLIT_NAMES = ("train", "val", "eval", "heldout_tmpl", "hop4")
SIDECARS = ("evhop", "chains.jsonl", "src", "build.json", "build_state.json")
STATE_VERSION = 1


class BuildInterrupted(Exception):
    """A stop was requested; the state at the last batch is on disk and a relaunch resumes."""


_STOP = {"requested": False}


def _request_stop(signum, frame) -> None:
    _STOP["requested"] = True


def split_files(data_dir: str, split: str) -> List[str]:
    """Every file a chain split can own, existing or not."""
    return [os.path.join(data_dir, f"{split}.{s}") for s in tuple(EVIDENCE_SUFFIXES) + SIDECARS]


def stream_questions(name: str, count: int, seed: int, held_out: Set[Tuple[str, ...]], per_hop: int = 0,
                     rng: Optional[random.Random] = None, start: int = 0) -> Iterator[chains.ChainQuestion]:
    """Yield the questions of ``make_split_questions`` one at a time.

    ``train`` and ``val`` are drawn lazily from ``rng`` (a resume passes the generator already
    restored to the state after ``start`` questions, and ``start`` only shortens the count); the
    small splits are drawn whole, then yielded from ``start``.

    Args:
        name: one of ``SPLIT_NAMES``.
        count: number of questions for train and val.
        seed: build seed.
        held_out: held out 2-tuples.
        per_hop: as in ``make_split_questions``.
        rng: the generator of the split, ``Random(f"chain_split:{seed}:{name}")`` when None.
        start: questions already written.
    """
    if name in ("train", "val"):
        rng = rng or random.Random(f"chain_split:{seed}:{name}")
        for _ in range(count - start):
            yield chains.sample_question(rng, held_out)
    else:
        yield from make_split_questions(name, count, seed, held_out, per_hop)[start:]


def make_split_questions(name: str, count: int, seed: int, held_out: Set[Tuple[str, ...]],
                         per_hop: int = 0) -> List[chains.ChainQuestion]:
    """Draw the questions of one split, deterministic in ``(seed, name)``.

    Args:
        name: one of ``SPLIT_NAMES``.
        count: number of questions for train and val.
        seed: build seed.
        held_out: held out 2-tuples from ``chains.split_compositions``.
        per_hop: questions per hop for ``eval`` (hops 1, 2, 3), per split for ``heldout_tmpl`` and
            ``hop4``.
    """
    rng = random.Random(f"chain_split:{seed}:{name}")
    out: List[chains.ChainQuestion] = []
    if name in ("train", "val"):
        for _ in range(count):
            out.append(chains.sample_question(rng, held_out))
    elif name == "eval":
        for hops in (1, 2, 3):
            out.extend(chains.sample_question(rng, held_out, hops=hops) for _ in range(per_hop))
        rng.shuffle(out)
    elif name == "heldout_tmpl":
        pairs = sorted(held_out)
        for i in range(per_hop):
            out.append(chains.sample_question(rng, held_out, composition=pairs[i % len(pairs)]))
        rng.shuffle(out)
    elif name == "hop4":
        tuples = chains.compositions(4)
        for _ in range(per_hop):
            out.append(chains.sample_question(rng, held_out, composition=rng.choice(tuples)))
    else:
        raise ValueError(f"unknown split {name!r}")
    return out


def conversation(q: chains.ChainQuestion, bridge: bool = False) -> List[dict]:
    """The rendered chat of one question: the evidence prompt, then the answer or the written bridge.

    Args:
        q: the question.
        bridge: write the intermediates before the answer (``chains.bridge_answer``).
    """
    answer = chains.bridge_answer(q) if bridge else q.answer
    return [{"role": "user", "content": evidence_prompt(q.question)},
            {"role": "assistant", "content": answer}]


def _new_report() -> dict:
    return {"questions": 0, "prompt_tokens": 0, "ev_tokens": 0, "chunks": 0,
            "by_hops": Counter(), "by_composition": Counter(), "kinds": Counter(),
            "candidates": {}, "distractors": Counter(), "shortcut_sums": {}}


def _report_to_json(report: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in report.items()}
    return json.loads(json.dumps(out))


def _report_from_json(data: dict) -> dict:
    report = _new_report()
    for key in ("questions", "prompt_tokens", "ev_tokens", "chunks"):
        report[key] = data[key]
    report["by_hops"] = Counter({int(k): v for k, v in data["by_hops"].items()})
    report["distractors"] = Counter({int(k): v for k, v in data["distractors"].items()})
    report["by_composition"] = Counter(data["by_composition"])
    report["kinds"] = Counter(data["kinds"])
    report["candidates"] = {int(k): list(v) for k, v in data["candidates"].items()}
    report["shortcut_sums"] = {int(k): {"n": v["n"], "totals": dict(v["totals"])}
                               for k, v in data["shortcut_sums"].items()}
    return report


def _split_sizes(data_dir: str, split: str) -> dict:
    return {os.path.basename(p): os.path.getsize(p) for p in split_files(data_dir, split)
            if os.path.exists(p) and not p.endswith((".build.json", ".build_state.json"))}


def build_signature(name: str, count: int, per_hop: int, seed: int, held_out: Set[Tuple[str, ...]],
                    bridge: bool, facts: bool) -> dict:
    """Everything that decides the bytes of a split: the draw, the hop mix, the held out set, the form."""
    return json.loads(json.dumps({
        "name": name, "count": count, "per_hop": per_hop, "seed": seed, "bridge": bridge, "facts": facts,
        "hop_weights": list(chains.sample_hops.__defaults__[0]),
        "held_out": [list(c) for c in sorted(held_out)], "state_version": STATE_VERSION}))


def completed_build(data_dir: str, split: str, signature: dict) -> bool:
    """True when ``{split}.build.json`` carries this signature and the recorded file sizes are intact."""
    try:
        with open(os.path.join(data_dir, f"{split}.build.json"), encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        return False
    return record.get("signature") == signature and record.get("files") == _split_sizes(data_dir, split)


def _write_json_atomic(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_split(data_dir: str, split: str, questions: Iterable[chains.ChainQuestion], template: ChatTemplate,
                embedder, held_out: Set[Tuple[str, ...]], batch: int = 1000, bridge: bool = False,
                facts: bool = False, *, rng: Optional[random.Random] = None,
                signature: Optional[dict] = None, resume: Optional[dict] = None,
                checkpoint_seconds: float = 300.0,
                stop_check: Optional[Callable[[], bool]] = None) -> dict:
    """Write one split and return its build report.

    The questions are consumed ``batch`` at a time, so a generator keeps memory flat. With a
    ``signature`` the split is resumable: ``{split}.build_state.json`` is written right after the
    files are opened and every ``checkpoint_seconds``, and a finished split leaves
    ``{split}.build.json`` and no state.

    Args:
        data_dir: output directory.
        split: split name including the prefix, for example ``chains_eval``.
        questions: the rows, in on-disk order; a list or an iterator (after a resume, the remaining
            ones).
        template: chat template, for the prompt ids and the answer mask.
        embedder: anything with ``encode(texts) -> [n, 384] float32`` (``ChunkEmbedder`` or a fake).
        held_out: held out 2-tuples, for the ``held_out`` flag in the metadata.
        batch: questions encoded and embedded per call.
        bridge: the assistant turn is the written bridge and the metadata carries ``bridge``, the
            intermediate entities e1..e(k-1).
        facts: the metadata also carries ``chunk_facts``, the ``[subject, relation, object]`` of every
            chunk, which ``eval_chains.py`` needs for the shortcut baselines. Only the eval splits
            set it; the training and validation sidecars stay as they were.
        rng: the generator ``questions`` draws from; its state goes into the resume state.
        signature: from ``build_signature``; None turns resuming and the completion record off.
        resume: the loaded state to continue from (the caller has already restored ``rng`` and
            shortened ``questions``).
        checkpoint_seconds: seconds between state saves.
        stop_check: called at every batch boundary; True saves the state and raises
            ``BuildInterrupted``. Defaults to the SIGTERM flag of ``main``.
    """
    stop_check = stop_check or (lambda: _STOP["requested"])
    state: dict = {} if resume is None else dict(resume["writer"])
    n_done = 0 if resume is None else resume["questions"]
    report = _new_report() if resume is None else _report_from_json(resume["report"])
    hop_path = os.path.join(data_dir, f"{split}.evhop")
    meta_path = os.path.join(data_dir, f"{split}.chains.jsonl")
    state_path = os.path.join(data_dir, f"{split}.build_state.json")
    if resume is not None:
        for path, size in ((hop_path, resume["evhop_bytes"]), (meta_path, resume["meta_bytes"])):
            have = os.path.getsize(path) if os.path.exists(path) else -1
            if have < size:
                raise SystemExit(f"{path} has {have} bytes, the saved state needs {size}; rerun with "
                                 f"--overwrite to start over")
            if have > size:
                with open(path, "r+b") as f:
                    f.truncate(size)
                logger.warning(f"truncated {os.path.basename(path)} from {have} to {size} bytes "
                               f"(the uncommitted tail of an interrupted build)")
    writer = EvidenceWriter(data_dir, split, state)
    hop_file = open(hop_path, "ab")
    meta_file = open(meta_path, "ab")
    t0 = time.monotonic()
    elapsed_before = 0.0 if resume is None else resume.get("elapsed", 0.0)

    def save_state() -> None:
        writer.sync()
        for f in (hop_file, meta_file):
            f.flush()
            os.fsync(f.fileno())
        if signature is None:
            return
        version, internal, gauss = rng.getstate() if rng is not None else (3, (), None)
        _write_json_atomic(state_path, {
            "signature": signature, "questions": n_done,
            "rng": [version, list(internal), gauss], "writer": dict(state),
            "evhop_bytes": hop_file.tell(), "meta_bytes": meta_file.tell(),
            "report": _report_to_json(report), "elapsed": elapsed_before + time.monotonic() - t0})

    try:
        if resume is None:
            save_state()
        last_save = time.monotonic()
        iterator = iter(questions)
        while True:
            if stop_check():
                save_state()
                raise BuildInterrupted(f"{split}: stopped after {n_done:,} questions; state in {state_path}")
            group = list(itertools.islice(iterator, batch))
            if not group:
                break
            start = n_done
            convs = [conversation(q, bridge) for q in group]
            encoded = template.encode_batch(convs)
            flat = [c for q in group for c in q.chunks]
            pieces = template.tokenizer(flat, add_special_tokens=False)["input_ids"]
            vectors = embedder.encode(flat)
            cursor = 0
            for offset, (q, enc) in enumerate(zip(group, encoded)):
                n = len(q.chunks)
                chunk_pieces, chunk_vectors = pieces[cursor:cursor + n], vectors[cursor:cursor + n]
                cursor += n
                assert enc is not None and all(chunk_pieces), f"unencodable question {q.question!r}"
                ids, mask = enc
                ev_ids, ev_chunk = [], []
                for c, piece in enumerate(chunk_pieces):
                    ev_ids.extend(piece)
                    ev_chunk.extend([c] * len(piece))
                gold = [1 if h else 0 for h in q.chunk_hop]
                writer.write(ids, mask, ev_ids, ev_chunk, np.asarray(chunk_vectors, dtype=np.float32),
                             gold, CONDITIONS.index("mixed"), 1)
                hop_file.write(np.asarray(q.chunk_hop, dtype=np.uint8).tobytes())
                meta = {
                    "qid": f"{split}-{start + offset}", "question": q.question, "answer": q.answer,
                    "answer_type": q.answer_type, "hops": q.hops, "composition": list(q.composition),
                    "held_out": chains.contains_held_out(q.composition, held_out),
                    "candidates": q.candidates, "chunk_kind": q.chunk_kind, "entities": q.entities,
                }
                if bridge:
                    meta["bridge"] = q.entities[1:-1]
                if facts:
                    meta["chunk_facts"] = [list(f) for f in q.chunk_facts]
                meta_file.write((json.dumps(meta) + "\n").encode("utf-8"))
                report["questions"] += 1
                report["prompt_tokens"] += len(ids)
                report["ev_tokens"] += len(ev_ids)
                report["chunks"] += n
                report["by_hops"][q.hops] += 1
                report["by_composition"]["+".join(q.composition)] += 1
                report["kinds"].update(q.chunk_kind)
                acc = report["candidates"].setdefault(q.hops, [0, 0, 0.0])
                acc[0] += 1
                acc[1] += len(q.candidates)
                acc[2] += 1.0 / len(q.candidates)
                report["distractors"][n - q.hops] += 1
            for hops in sorted({q.hops for q in group}):
                part = [q for q in group if q.hops == hops]
                base = chains.shortcut_baselines(part)
                acc = report["shortcut_sums"].setdefault(hops, {"n": 0, "totals": {}})
                acc["n"] += len(part)
                for k, v in base.items():
                    acc["totals"][k] = acc["totals"].get(k, 0.0) + v * len(part)
            n_done += len(group)
            if signature is not None and time.monotonic() - last_save >= checkpoint_seconds:
                save_state()
                last_save = time.monotonic()
        save_state()
    finally:
        writer.close()
        hop_file.close()
        meta_file.close()
    report["shortcuts"] = {
        hops: {k: v / max(acc["n"], 1) for k, v in acc["totals"].items()}
        for hops, acc in sorted(report["shortcut_sums"].items())
    }
    if signature is not None:
        _write_json_atomic(os.path.join(data_dir, f"{split}.build.json"), {
            "signature": signature, "questions": n_done, "files": _split_sizes(data_dir, split)})
        if os.path.exists(state_path):
            os.remove(state_path)
    return report


def build_split_streaming(data_dir: str, split: str, name: str, count: int, per_hop: int, seed: int,
                          held_out: Set[Tuple[str, ...]], template: ChatTemplate, embedder, *,
                          bridge: bool = False, facts: bool = False, batch: int = 1000,
                          overwrite: bool = False, checkpoint_seconds: float = 300.0,
                          stop_check: Optional[Callable[[], bool]] = None) -> Optional[dict]:
    """Stream, resume and complete-check one split; the report, or None when it was already complete.

    Args:
        data_dir: output directory.
        split: split name including the prefix.
        name: one of ``SPLIT_NAMES``.
        count: questions for train and val.
        per_hop: questions per hop for the small splits.
        seed: build seed.
        held_out: held out 2-tuples.
        template: chat template.
        embedder: ``encode(texts)`` object.
        bridge: the written bridge form.
        facts: carry ``chunk_facts`` in the sidecar.
        batch: questions per encode call.
        overwrite: discard a state and any files of the split, then start over.
        checkpoint_seconds: seconds between state saves.
        stop_check: see ``build_split``.

    Raises:
        FileExistsError: files exist with no matching state or completion record and no ``overwrite``.
        SystemExit: the state was written by a build with other arguments.
    """
    signature = build_signature(name, count, per_hop, seed, held_out, bridge, facts)
    state_path = os.path.join(data_dir, f"{split}.build_state.json")
    if overwrite:
        for path in split_files(data_dir, split):
            if os.path.exists(path):
                os.remove(path)
    resume = None
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            resume = json.load(f)
        if resume["signature"] != signature:
            differing = sorted(k for k in signature if resume["signature"].get(k) != signature[k])
            raise SystemExit(f"{state_path} was written by a build with other arguments ({differing}); "
                             f"relaunch with the same arguments or pass --overwrite to start over")
    if completed_build(data_dir, split, signature):
        if resume is not None:
            os.remove(state_path)
        logger.info(f"already complete: {split} matches its build.json for these arguments; nothing touched")
        print(f"already complete: {split}")
        return None
    if resume is not None:
        logger.info(f"resuming {split} from question {resume['questions']:,}")
    else:
        existing = [p for p in split_files(data_dir, split) if os.path.exists(p)]
        if existing:
            raise FileExistsError(f"{split} exists ({len(existing)} files); pass --overwrite")
    rng = random.Random(f"chain_split:{seed}:{name}")
    start = 0
    if resume is not None:
        version, internal, gauss = resume["rng"]
        if name in ("train", "val"):
            rng.setstate((version, tuple(internal), gauss))
        start = resume["questions"]
    questions = stream_questions(name, count, seed, held_out, per_hop, rng=rng, start=start)
    return build_split(data_dir, split, questions, template, embedder, held_out, batch=batch, bridge=bridge,
                       facts=facts, rng=rng, signature=signature, resume=resume,
                       checkpoint_seconds=checkpoint_seconds, stop_check=stop_check)


def log_report(split: str, report: dict) -> None:
    """Log the build counts of one split, the leak detectors included."""
    q = max(report["questions"], 1)
    logger.info(
        f"{split}: {report['questions']} questions, {report['prompt_tokens']} prompt tokens, "
        f"{report['ev_tokens']} evidence tokens in {report['chunks']} chunks, evidence to prompt "
        f"ratio {report['ev_tokens'] / max(report['prompt_tokens'], 1):.2f}, "
        f"{report['chunks'] / q:.1f} chunks per question"
    )
    logger.info(f"{split}: by hops {dict(sorted(report['by_hops'].items()))}")
    logger.info(f"{split}: chunk kinds {dict(report['kinds'])}")
    logger.info(f"{split}: distractors per question {dict(sorted(report['distractors'].items()))}")
    for hops, (n, total, inverse) in sorted(report["candidates"].items()):
        logger.info(f"{split}: {hops}-hop mean type matched candidates {total / n:.2f} "
                    f"(chance {inverse / n:.3f})")
    for hops, base in report["shortcuts"].items():
        logger.info(f"{split}: {hops}-hop shortcut baselines " +
                    ", ".join(f"{k} {v:.3f}" for k, v in base.items()))
        if hops >= 2:
            logger.info(f"{split}: {hops}-hop penultimate_fed {base['penultimate_fed']:.3f} against "
                        f"random {base['random']:.3f} is a known bias of the distractor recipe and "
                        f"the line an above chance latent read has to beat")
    for comp, n in report["by_composition"].most_common():
        logger.info(f"{split}: composition {comp}: {n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "data", "prepared"))
    parser.add_argument("--prefix", default="chains")
    parser.add_argument("--train-questions", type=int, default=200_000)
    parser.add_argument("--val-questions", type=int, default=2000)
    parser.add_argument("--eval-questions-per-hop", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", default=",".join(SPLIT_NAMES))
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an existing split of the same name (and its resume state) first")
    parser.add_argument("--checkpoint-seconds", type=float, default=300.0,
                        help="seconds between resume state saves")
    parser.add_argument("--bridge", action="store_true",
                        help="write the chain's intermediate entities before the answer")
    args = parser.parse_args()

    if args.prefix.startswith("evidence"):
        parser.error("prefix would collide with the evidence corpora; pick another")
    names = [s.strip() for s in args.splits.split(",") if s.strip()]
    for name in names:
        if name not in SPLIT_NAMES:
            parser.error(f"unknown split {name!r}; choose from {SPLIT_NAMES}")
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    template = ChatTemplate(AutoTokenizer.from_pretrained(TOKENIZER_DIR))
    embedder = ChunkEmbedder(device=args.device, cache_size=20_000)
    seen, held_out = chains.split_compositions(args.seed)
    logger.info(f"held out 2-hop compositions: {sorted(held_out)}")
    logger.info(f"note: the held out set is {len(held_out)} pairs with first relations "
                f"{sorted({c[0] for c in held_out})}; kept as drawn, since a redraw changes the "
                f"splits the pilot arms were read on")
    signal.signal(signal.SIGTERM, _request_stop)
    for name in names:
        split = f"{args.prefix}_{name}"
        count = args.train_questions if name == "train" else args.val_questions
        try:
            report = build_split_streaming(
                args.out_dir, split, name, count, args.eval_questions_per_hop, args.seed, held_out,
                template, embedder, bridge=args.bridge, facts=name not in ("train", "val"),
                overwrite=args.overwrite, checkpoint_seconds=args.checkpoint_seconds)
        except BuildInterrupted as stop:
            logger.warning(str(stop))
            sys.exit(20)
        except FileExistsError as exists:
            parser.error(str(exists))
        if report is not None:
            log_report(split, report)


if __name__ == "__main__":
    main()

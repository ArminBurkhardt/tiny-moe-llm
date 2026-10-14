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
                          composition, held_out, candidates, chunk_kind, entities

The 200k question train split is about 2.6M short chunks and wants the GPU for the embedder; the
eval splits embed on CPU in minutes (``--device cpu``).

Usage:

    python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains \\
        --train-questions 200000 --val-questions 2000 --eval-questions-per-hop 1000 --seed 42 \\
        --device cuda [--splits eval,heldout_tmpl,hop4] [--overwrite]
"""
import os
import sys
import json
import random
import argparse
from collections import Counter, defaultdict
from typing import List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.data import chains
from modules.data.chat import ChatTemplate
from scripts.prepare_evidence_data import (
    CONDITIONS, EVIDENCE_SUFFIXES, ChunkEmbedder, EvidenceWriter, evidence_prompt,
)
from utils import BASE_DIR, TOKENIZER_DIR, logger

SPLIT_NAMES = ("train", "val", "eval", "heldout_tmpl", "hop4")
SIDECARS = ("evhop", "chains.jsonl", "src")


def split_files(data_dir: str, split: str) -> List[str]:
    """Every file a chain split can own, existing or not."""
    return [os.path.join(data_dir, f"{split}.{s}") for s in tuple(EVIDENCE_SUFFIXES) + SIDECARS]


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


def build_split(data_dir: str, split: str, questions: List[chains.ChainQuestion], template: ChatTemplate,
                embedder, held_out: Set[Tuple[str, ...]], batch: int = 1000) -> dict:
    """Write one split and return its build report.

    Args:
        data_dir: output directory.
        split: split name including the prefix, for example ``chains_eval``.
        questions: the rows, in on-disk order.
        template: chat template, for the prompt ids and the answer mask.
        embedder: anything with ``encode(texts) -> [n, 384] float32`` (``ChunkEmbedder`` or a fake).
        held_out: held out 2-tuples, for the ``held_out`` flag in the metadata.
        batch: questions encoded and embedded per call.
    """
    state: dict = {}
    writer = EvidenceWriter(data_dir, split, state)
    hop_file = open(os.path.join(data_dir, f"{split}.evhop"), "ab")
    meta_file = open(os.path.join(data_dir, f"{split}.chains.jsonl"), "ab")
    report = {"questions": 0, "prompt_tokens": 0, "ev_tokens": 0, "chunks": 0,
              "by_hops": Counter(), "by_composition": Counter(), "kinds": Counter(),
              "candidates": defaultdict(list), "distractors": Counter()}
    try:
        for start in range(0, len(questions), batch):
            group = questions[start:start + batch]
            convs = [[{"role": "user", "content": evidence_prompt(q.question)},
                      {"role": "assistant", "content": q.answer}] for q in group]
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
                meta_file.write((json.dumps(meta) + "\n").encode("utf-8"))
                report["questions"] += 1
                report["prompt_tokens"] += len(ids)
                report["ev_tokens"] += len(ev_ids)
                report["chunks"] += n
                report["by_hops"][q.hops] += 1
                report["by_composition"]["+".join(q.composition)] += 1
                report["kinds"].update(q.chunk_kind)
                report["candidates"][q.hops].append(len(q.candidates))
                report["distractors"][n - q.hops] += 1
        writer.sync()
        hop_file.flush()
        os.fsync(hop_file.fileno())
        meta_file.flush()
        os.fsync(meta_file.fileno())
    finally:
        writer.close()
        hop_file.close()
        meta_file.close()
    report["shortcuts"] = {
        hops: chains.shortcut_baselines([q for q in questions if q.hops == hops])
        for hops in sorted({q.hops for q in questions})
    }
    return report


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
    for hops, sizes in sorted(report["candidates"].items()):
        logger.info(f"{split}: {hops}-hop mean type matched candidates {np.mean(sizes):.2f} "
                    f"(chance {np.mean([1.0 / s for s in sizes]):.3f})")
    for hops, base in report["shortcuts"].items():
        logger.info(f"{split}: {hops}-hop shortcut baselines " +
                    ", ".join(f"{k} {v:.3f}" for k, v in base.items()))
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
                        help="delete an existing split of the same name first")
    args = parser.parse_args()

    if args.prefix.startswith("evidence"):
        parser.error("prefix would collide with the evidence corpora; pick another")
    names = [s.strip() for s in args.splits.split(",") if s.strip()]
    for name in names:
        if name not in SPLIT_NAMES:
            parser.error(f"unknown split {name!r}; choose from {SPLIT_NAMES}")
    os.makedirs(args.out_dir, exist_ok=True)
    for name in names:
        existing = [p for p in split_files(args.out_dir, f"{args.prefix}_{name}") if os.path.exists(p)]
        if existing and not args.overwrite:
            parser.error(f"{args.prefix}_{name} exists ({len(existing)} files); pass --overwrite")
        for path in existing:
            os.remove(path)

    from transformers import AutoTokenizer
    template = ChatTemplate(AutoTokenizer.from_pretrained(TOKENIZER_DIR))
    embedder = ChunkEmbedder(device=args.device, cache_size=20_000)
    seen, held_out = chains.split_compositions(args.seed)
    logger.info(f"held out 2-hop compositions: {sorted(held_out)}")
    for name in names:
        split = f"{args.prefix}_{name}"
        count = args.train_questions if name == "train" else args.val_questions
        questions = make_split_questions(name, count, args.seed, held_out, args.eval_questions_per_hop)
        log_report(split, build_split(args.out_dir, split, questions, template, embedder, held_out))


if __name__ == "__main__":
    main()

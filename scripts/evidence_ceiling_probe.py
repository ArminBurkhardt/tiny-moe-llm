"""What in-context evidence is worth to this model, before any port is built.

Gate G3 asks for a **gold-vs-no-evidence CE gap of ~0.3 nats on the answer span**, read through a
new evidence port that costs 300-500M tokens to train. That gap has a ceiling that can be measured
today for free: hand the model the same gold passage through in-context attention -- the pathway it
already has and was finetuned on -- and read the same quantity. A port that routes evidence through
the IR/CrossAttention experts is trying to reach what in-context reading already delivers; if
in-context evidence is not worth 0.3 nats on this checkpoint, the port has no ceiling to reach and
the gate is measuring the wrong thing rather than the model failing.

Three conditions, one forward pass each, on the answerable half of the standard SQuAD v2 slice:

  * **gold** -- the row's own passage, i.e. exactly what ``eval_abstention.py`` scores.
  * **none** -- the passage block removed, instruction and question unchanged. The floor.
  * **distractor** -- another answerable row's passage in the same slot, so the prompt keeps its
    shape and its length distribution and only the *relevance* of the evidence changes. Gold minus
    distractor is the selection signal Phase 4's gold-among-distractors condition trains; gold minus
    none is the reading signal.

Everything is scored with ``eval_abstention.teacher_forced_calibration`` by import rather than a
second copy of the loop, so these CEs are the same quantity that script reports, restricted to the
answer span. Read-only.

``--fixed-split NAME`` measures the same ceiling on the rows of a held-out evidence split
(``prepare_evidence_data.py --heldout``) instead of a fresh SQuAD slice: the question, the real
answer and the four conditions' passages are read back from disk, so the number is the in-context
reading of exactly the questions the trainer's fixed-target pass scores through the port. Each
condition's passages are decoded from the split's own evidence buffer (``gold``, ``mixed``,
``distractors``, ``none``) and put in the prompt in buffer order. The result is split by source
when the split carries ``.src``, and its JSON is what ``sft.py --ceiling-json`` reads.
"""
import os
import sys
import json
import random
import argparse
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer

from config import ModelConfig, SFTConfig
from modules.data.chat import ChatTemplate
from utils import BASE_DIR, TOKENIZER_DIR, get_hf_token, logger
from scripts.prepare_sft_data import SQUAD_INSTRUCTION
from scripts.prepare_evidence_data import FIXED_CONDITIONS, SOURCE_KEYS

# the question follows this marker in the evidence prompt, whatever the instruction says
QUESTION_MARKER = "\n\nQuestion: "


def prompt_for(context: str, question: str) -> str:
    """The user turn. ``context=""`` drops the passage block entirely.

    ``SQUAD_INSTRUCTION`` is imported for the same reason ``eval_abstention.squad_prompt`` imports
    it: the instruction is what licenses abstention, and a restated copy that drifted by a word
    would score the model on a prompt it never saw.
    """
    if not context:
        return f"{SQUAD_INSTRUCTION}\n\nQuestion: {question}"
    return f"{SQUAD_INSTRUCTION}\n\nPassage:\n{context}\n\nQuestion: {question}"


def build_conditions(frame, template: ChatTemplate, *, max_examples, max_prompt_tokens, seed):
    """One record list per condition, over the same rows in the same order.

    The row set is fixed by the *gold* condition's length check and then reused verbatim: letting
    each condition drop its own over-long rows would score the three on different questions, and the
    no-evidence prompts are shorter, so it would systematically drop the hardest passages from one
    arm only.
    """
    from scripts.eval_abstention import squad_references

    rows = frame.to_dict("records")
    rng = random.Random(seed)
    rng.shuffle(rows)

    # answerable rows only: the gap is defined on an answer span, and an unanswerable row's forced
    # target is a fixed abstention phrasing whose CE says nothing about reading the passage
    usable = []
    for row in rows:
        context = str(row.get("context") or "").strip()
        question = str(row.get("question") or "").strip()
        references = squad_references(row)
        if not context or not question or not references:
            continue
        if len(template.encode_prompt([{"role": "user", "content": prompt_for(context, question)}])) > max_prompt_tokens:
            continue
        usable.append((context, question, references[0]))
        if max_examples is not None and len(usable) >= max_examples:
            break

    # a deterministic derangement: row i reads row (i + 1)'s passage, so no row ever draws its own
    # and the distractor pool is the same passage distribution as the gold one
    conditions = {"gold": [], "none": [], "distractor": []}
    for i, (context, question, answer) in enumerate(usable):
        passages = {
            "gold": context,
            "none": "",
            "distractor": usable[(i + 1) % len(usable)][0],
        }
        for name, passage in passages.items():
            messages = [
                {"role": "user", "content": prompt_for(passage, question)},
                {"role": "assistant", "content": answer},
            ]
            encoded = template.encode(messages)
            if encoded is None:
                continue
            ids, mask = encoded
            conditions[name].append({"forced_ids": ids, "forced_mask": mask})
    return conditions, len(usable)


def read_fixed_groups(data_dir: str, split: str, tokenizer, template: ChatTemplate,
                      max_questions: Optional[int]) -> List[dict]:
    """The questions of a fixed split, each with its four conditions' passages, in file order.

    The split is question-major: four consecutive documents per question, in ``FIXED_CONDITIONS``
    order, identical prompts. The question is the user turn after the last "Question: " marker, the
    answer is the supervised span minus its closing EOS, and a condition's passages are its chunks
    decoded in buffer order.

    Args:
        data_dir: directory holding the split.
        split: split name, e.g. ``evidence_fixed``.
        tokenizer: the model tokenizer.
        template: its chat template (for the control token ids).
        max_questions: stop after this many questions, or None for all. File order is a
            representative prefix (the build shuffles sources together).

    Returns:
        ``[{"question", "answer", "passages": {condition: text}, "source"}]``; ``source`` is the
        ``SOURCE_KEYS`` name from ``.src``, or "unknown" without it.
    """
    def path(suffix):
        return os.path.join(data_dir, f"{split}.{suffix}")

    ids_mm = np.memmap(path("bin"), dtype=np.uint16, mode="r")
    idx = np.memmap(path("idx"), dtype=np.uint64, mode="r")
    mask_mm = np.memmap(path("mask"), dtype=np.uint8, mode="r")
    ev_mm = np.memmap(path("ev"), dtype=np.uint16, mode="r")
    evidx = np.memmap(path("evidx"), dtype=np.uint64, mode="r")
    evchunk = np.memmap(path("evchunk"), dtype=np.uint16, mode="r")
    src = np.memmap(path("src"), dtype=np.uint8, mode="r") if os.path.isfile(path("src")) else None
    n_docs = idx.shape[0] - 1
    group = len(FIXED_CONDITIONS)
    if n_docs % group:
        raise SystemExit(f"{split} has {n_docs} documents, not a multiple of {group}: not a fixed split")

    def passage_text(doc: int) -> str:
        start, end = int(evidx[doc]), int(evidx[doc + 1])
        tokens, chunks = ev_mm[start:end], evchunk[start:end]
        texts = []
        for chunk in range(int(chunks.max()) + 1 if end > start else 0):
            texts.append(tokenizer.decode(tokens[chunks == chunk].tolist(), skip_special_tokens=True).strip())
        return "\n\n".join(t for t in texts if t)

    def source_name(doc: int) -> str:
        if src is None or int(src[doc]) >= len(SOURCE_KEYS):
            return "unknown"
        return SOURCE_KEYS[int(src[doc])]

    out = []
    for first in range(0, n_docs, group):
        if max_questions is not None and len(out) >= max_questions:
            break
        a, b = int(idx[first]), int(idx[first + 1])
        ids, mask = ids_mm[a:b].tolist(), mask_mm[a:b].tolist()
        try:
            user_at, assistant_at = ids.index(template.user_id), ids.index(template.assistant_id)
        except ValueError:
            raise SystemExit(f"{split} document {first} is not a chat row")
        prompt = tokenizer.decode(ids[user_at + 1:assistant_at], skip_special_tokens=True).strip()
        if QUESTION_MARKER not in prompt:
            raise SystemExit(f"{split} document {first} has no question marker in {prompt[:80]!r}")
        answer_ids = [t for t, m in zip(ids, mask) if m][:-1]
        out.append({
            "question": prompt.rsplit(QUESTION_MARKER, 1)[1].strip(),
            "answer": tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
            "passages": {c: passage_text(first + i) for i, c in enumerate(FIXED_CONDITIONS)},
            "source": source_name(first),
        })
    return out


def build_fixed_conditions(groups: List[dict], template: ChatTemplate, max_prompt_tokens: int):
    """One record list per fixed condition over the same questions, tagged by source.

    A question whose longest prompt exceeds ``max_prompt_tokens`` is dropped from every condition,
    so the four are always scored on the same rows.

    Args:
        groups: output of ``read_fixed_groups``.
        template: the chat template.
        max_prompt_tokens: longest prompt kept.

    Returns:
        ``({condition: [{"forced_ids", "forced_mask", "source"}]}, kept groups)``.
    """
    conditions = {c: [] for c in FIXED_CONDITIONS}
    kept = []
    for g in groups:
        prompts = {c: prompt_for(g["passages"][c], g["question"]) for c in FIXED_CONDITIONS}
        if max(len(template.encode_prompt([{"role": "user", "content": p}]))
               for p in prompts.values()) > max_prompt_tokens:
            continue
        encoded = {
            c: template.encode([{"role": "user", "content": prompts[c]},
                                {"role": "assistant", "content": g["answer"]}])
            for c in FIXED_CONDITIONS
        }
        if any(e is None for e in encoded.values()):
            continue
        for c in FIXED_CONDITIONS:
            conditions[c].append({"forced_ids": encoded[c][0], "forced_mask": encoded[c][1],
                                  "source": g["source"]})
        kept.append(g)
    return conditions, kept


def summarize_fixed(per_source: Dict[str, Dict[str, dict]], questions: Dict[str, int]) -> dict:
    """Per source and pooled ceiling readings from per (source, condition) CE and token counts.

    The pooled number is the token weighted mean of the per source ones, which is the CE a single
    pass over every row would have read. ``ceiling`` is CE(none) - CE(gold) and ``content`` is
    CE(distractors) - CE(gold), the same two quantities the trainer's fixed pass reports.

    Args:
        per_source: ``{source: {condition: {"ce", "tokens"}}}``.
        questions: ``{source: number of questions}``.
    """
    def entry(rows: Dict[str, dict], n_questions: int) -> dict:
        out = {c: rows[c]["ce"] for c in FIXED_CONDITIONS}
        out["ceiling"] = out["none"] - out["gold"]
        out["content"] = out["distractors"] - out["gold"]
        out["questions"] = n_questions
        out["tokens"] = rows["gold"]["tokens"]
        return out

    by_source = {name: entry(rows, questions[name]) for name, rows in per_source.items()}
    pooled = {}
    for c in FIXED_CONDITIONS:
        tokens = sum(rows[c]["tokens"] for rows in per_source.values())
        pooled[c] = {"ce": sum(rows[c]["ce"] * rows[c]["tokens"] for rows in per_source.values()) / tokens,
                     "tokens": tokens}
    return {"by_source": by_source, "all": entry(pooled, sum(questions.values()))}


def run_fixed_split(args, tokenizer, template: ChatTemplate) -> None:
    """``--fixed-split`` mode: the in-context ceiling on a held-out split's own rows."""
    from scripts.eval_abstention import load_model, teacher_forced_calibration

    groups = read_fixed_groups(args.data_dir, args.fixed_split, tokenizer, template, args.max_questions)
    conditions, kept = build_fixed_conditions(groups, template, args.max_prompt_tokens)
    if not kept:
        raise SystemExit("no question fits --max-prompt-tokens")
    logger.info(f"{len(kept):,} of {len(groups):,} questions kept under {args.max_prompt_tokens} "
                f"prompt tokens")
    questions: Dict[str, int] = {}
    for g in kept:
        questions[g["source"]] = questions.get(g["source"], 0) + 1

    model = load_model(args.checkpoint, args.device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    per_source: Dict[str, Dict[str, dict]] = {name: {} for name in questions}
    for condition in FIXED_CONDITIONS:
        for name in questions:
            records = [r for r in conditions[condition] if r["source"] == name]
            logger.info(f"scoring {len(records):,} forced answers, condition {condition}, source {name}")
            scored = teacher_forced_calibration(
                model, template, records, pad_id=pad_id, batch_size=args.batch_size,
                device=args.device, max_seq_len=ModelConfig.Params["max_seq_len"],
            )
            per_source[name][condition] = {"ce": scored["ce"], "tokens": scored["tokens"]}

    summary = summarize_fixed(per_source, questions)
    print(f"\n=== in-context answer span CE on {args.fixed_split}, {len(kept):,} questions ===")
    print(f"  checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"\n  {'source':<12}" + "".join(f"{c:>13}" for c in FIXED_CONDITIONS)
          + f"{'ceiling':>10}{'content':>10}{'questions':>11}")
    for name, row in list(summary["by_source"].items()) + [("all", summary["all"])]:
        print(f"  {name:<12}" + "".join(f"{row[c]:>13.4f}" for c in FIXED_CONDITIONS)
              + f"{row['ceiling']:>10.4f}{row['content']:>10.4f}{row['questions']:>11,}")
    print("\n  ceiling = CE(none) - CE(gold), the in-context reading of the number the fixed pass "
          "reaches through the port (read it with sft.py --ceiling-json)")

    if args.json_out:
        payload = {"checkpoint": args.checkpoint, "split": args.fixed_split, **summary}
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(f"wrote {args.json_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--squad-dir", default=None)
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="longest prompt kept (default 1024, or 3800 with --fixed-split, whose "
                             "mixed and distractor rows carry several passages)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SFTConfig.seed)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fixed-split", default=None,
                        help="score the rows of this held-out evidence split (e.g. evidence_fixed) "
                             "instead of a SQuAD slice, split by source; see the module docstring")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data", "prepared"),
                        help="--fixed-split: directory holding the split")
    parser.add_argument("--max-questions", type=int, default=2000,
                        help="--fixed-split: questions read, in file order")
    args = parser.parse_args()
    if args.max_prompt_tokens is None:
        args.max_prompt_tokens = 3800 if args.fixed_split else 1024

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = ChatTemplate(tokenizer)
    if args.fixed_split:
        run_fixed_split(args, tokenizer, template)
        return
    from scripts.eval_abstention import load_model, load_squad_split, teacher_forced_calibration

    scratch_dir = os.path.join(BASE_DIR, "data", "benchmarks", "squad_v2_validation")
    frame = load_squad_split(scratch_dir, args.hf_token or get_hf_token(), args.squad_dir)
    conditions, n_rows = build_conditions(
        frame, template, max_examples=args.max_examples,
        max_prompt_tokens=args.max_prompt_tokens, seed=args.seed,
    )
    if not n_rows:
        raise SystemExit("no usable answerable questions -- check --max-prompt-tokens / --squad-dir")

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    results = {}
    for name in ("gold", "none", "distractor"):
        logger.info(f"scoring {len(conditions[name]):,} forced answers, condition {name}")
        results[name] = teacher_forced_calibration(
            model, template, conditions[name], pad_id=pad_id, batch_size=args.batch_size,
            device=args.device, max_seq_len=ModelConfig.Params["max_seq_len"],
        )

    print(f"\n=== answer span CE by evidence condition, {n_rows:,} answerable questions ===")
    print(f"  checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"\n  {'condition':<14}{'CE':>9}{'ppl':>10}{'top1':>9}{'mean p_max':>13}{'tokens':>10}")
    for name in ("gold", "none", "distractor"):
        r = results[name]
        print(f"  {name:<14}{r['ce']:>9.4f}{r['ppl']:>10.2f}{r['top1_acc']:>9.4f}"
              f"{r['mean_p_max']:>13.4f}{r['tokens']:>10,}")

    gap_none = results["none"]["ce"] - results["gold"]["ce"]
    gap_distractor = results["distractor"]["ce"] - results["gold"]["ce"]
    print(f"\n  gold vs no evidence:  {gap_none:+.4f} nats   (G3 wants >= ~0.3 through the port)")
    print(f"  gold vs distractor:   {gap_distractor:+.4f} nats   (the selection signal)")
    print("\n  this is the IN CONTEXT ceiling, not the port -- the port is trying to reach it")

    if args.json_out:
        payload = {
            "checkpoint": args.checkpoint,
            "questions": n_rows,
            "conditions": results,
            "gap_gold_vs_none": gap_none,
            "gap_gold_vs_distractor": gap_distractor,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

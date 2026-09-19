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
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from config import ModelConfig, SFTConfig
from modules.data.chat import ChatTemplate
from utils import BASE_DIR, TOKENIZER_DIR, get_hf_token, logger
from scripts.prepare_sft_data import SQUAD_INSTRUCTION
from scripts.eval_abstention import (
    load_squad_split,
    squad_references,
    teacher_forced_calibration,
    load_model,
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--squad-dir", default=None)
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SFTConfig.seed)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = ChatTemplate(tokenizer)

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

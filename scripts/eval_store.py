"""Store-level retrieval, the open-book pathway and store edits, scored with the model's own query.

The evidence port's selector scores a retrieved chunk by ``normalize(down + loop_query_bias)``
against ``normalize(key_adapter(bge key))``. This script reads that query out of a forward pass
without touching the model (a forward pre-hook on every IR module), builds the adapted index once
per checkpoint, and ranks a store with it. Three questions, three subcommands:

``recall``
    Does the model's own query find the gold chunk as often as bge's? Recall@k per source and
    overall, per loop and per IR expert, against the bge row. HotpotQA also reports both supporting
    paragraphs in the top k. The headline is loop 1 at ``--gate-k``, the read before any evidence
    has entered the state, which is what a serving-time first retrieval uses. ``--bge-only`` runs
    the bge row alone on the CPU with no checkpoint.
``pathway``
    Does a buffer retrieved by the model answer as well as the oracle buffer? The top ``--buffer``
    chunks by the model's query at ``--loop`` go through the port; arms are ``model``, ``bge``,
    ``oracle`` (gold chunks only) and ``none`` (no evidence). Exact match, token F1 and abstention
    rate per source, binomial sigma on every rate.
``edit``
    Does an answer follow an edit to the store? Among questions the model answers correctly on the
    original store, re-retrieve on the edit store with the same query and classify the answer:
    ``follow`` (the substitute), ``stuck`` (the original), ``abstained`` or ``other``. A flip is
    ``follow`` for an edit and not ``stuck`` for a delete; the line passes at 0.70 per kind. The
    oracle row attaches the edited gold chunk directly and is a diagnostic, not a gate.

The query is read with ``evidence=None``: the first retrieval of a session happens before any
evidence is in the state. Scoring is exact brute force over the index. A model with no port
(``key_adapter`` is None) cannot be read this way.

```bash
python scripts/eval_store.py recall -c CKPT --store data/index/openqa --k 1,2,5,10,20 --json-out out.json
python scripts/eval_store.py pathway -c CKPT --store data/index/openqa --buffer 4 --loop 1 \\
  --max-questions 2000 --batch-size 16 --max-new-tokens 32 --json-out out.json
python scripts/eval_store.py edit -c CKPT --store data/index/openqa --edit-store data/index/openqa_edit \\
  --buffer 4 --loop 1 --json-out out.json
python scripts/eval_store.py recall --bge-only --store data/index/openqa
```
"""
import os
import sys
import json
import math
import random
import argparse
from typing import Callable, Dict, List, Optional, Sequence, Tuple

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modules.data import abstention
from modules.data.chat import ChatTemplate
from modules.data.store import Store, load_store
from scripts.prepare_evidence_data import evidence_prompt
from utils import TOKENIZER_DIR, logger

FLIP_BAR = 0.70


# --------------------------------------------------------------------------------- arithmetic


def topk_ids(queries, index, k: int, device: str = "cpu", block: int = 512) -> np.ndarray:
    """Exact top-k rows of ``index`` by dot product, for every query.

    Args:
        queries: ``[Q, D]`` array or tensor.
        index: ``[N, D]`` array or tensor; a tensor already on ``device`` is used without a copy.
        k: how many ids to return per query (capped at ``N``).
        device: where the matmul runs.
        block: queries scored per matmul.

    Returns:
        int64 ``[Q, min(k, N)]`` row ids, best first.
    """
    index = torch.as_tensor(index, dtype=torch.float32, device=device)
    queries = torch.as_tensor(queries, dtype=torch.float32)
    k = min(int(k), index.shape[0])
    out = []
    for start in range(0, queries.shape[0], block):
        scores = queries[start:start + block].to(device) @ index.t()
        out.append(scores.topk(k, dim=-1).indices.cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, k), dtype=np.int64)


def hit_vector(topk: np.ndarray, questions: Sequence[dict], k: int, both: bool = False) -> np.ndarray:
    """Per question: is a gold chunk in the top ``k`` (``both``: one from every gold group)?

    Args:
        topk: ``[Q, >=k]`` ranked chunk ids, row ``i`` for ``questions[i]``.
        questions: question records with ``gold_chunk_ids`` and ``gold_groups``.
        k: cut.
        both: require every gold group to be hit instead of any gold chunk.
    """
    hits = np.zeros(len(questions), dtype=bool)
    for i, q in enumerate(questions):
        top = set(topk[i, :k].tolist())
        if both:
            groups = q.get("gold_groups") or [q["gold_chunk_ids"]]
            hits[i] = all(any(c in top for c in g) for g in groups)
        else:
            hits[i] = any(c in top for c in q["gold_chunk_ids"])
    return hits


def recall_summary(topk: np.ndarray, questions: Sequence[dict], ks: Sequence[int]) -> dict:
    """Recall@k per source and overall, plus both-gold recall where a question has two groups.

    Args:
        topk: ranked ids per question.
        questions: aligned question records.
        ks: cuts to report.

    Returns:
        ``{"all" | source: {k: {"recall", "n", "both"}}}``; ``both`` is None for a source with no
        multi-group question.
    """
    sources = ["all"] + sorted({q["source"] for q in questions})
    out = {}
    for source in sources:
        index = [i for i, q in enumerate(questions) if source == "all" or q["source"] == source]
        subset = [questions[i] for i in index]
        multi = any(len(q.get("gold_groups") or []) > 1 for q in subset)
        out[source] = {}
        for k in ks:
            hits = hit_vector(topk[index], subset, k)
            both = hit_vector(topk[index], subset, k, both=True) if multi else None
            out[source][k] = {"recall": float(hits.mean()) if len(hits) else float("nan"),
                              "n": len(hits),
                              "both": float(both.mean()) if both is not None else None}
    return out


def paired_counts(hit_a: np.ndarray, hit_b: np.ndarray) -> dict:
    """McNemar-style comparison of two hit vectors over the same questions.

    Args:
        hit_a: bool per question for the first system.
        hit_b: bool per question for the second.

    Returns:
        ``{"a_only", "b_only", "n", "diff" (a - b), "sigma", "z"}``; sigma is the paired standard
        error of the difference in rates.
    """
    n = len(hit_a)
    a_only, b_only = int((hit_a & ~hit_b).sum()), int((~hit_a & hit_b).sum())
    diff = (a_only - b_only) / max(n, 1)
    variance = max(a_only + b_only - (a_only - b_only) ** 2 / max(n, 1), 0.0)
    sigma = math.sqrt(variance) / max(n, 1)
    return {"a_only": a_only, "b_only": b_only, "n": n, "diff": diff, "sigma": sigma,
            "z": diff / sigma if sigma > 0 else 0.0}


def binomial_sigma(p: float, n: int) -> float:
    """Standard error of a rate ``p`` measured on ``n`` questions."""
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def classify_edit(kind: str, prediction: str, substitute: Optional[str], answers: Sequence[str],
                  exact_match: Callable[[str, Sequence[str]], float]) -> str:
    """Name what an answer did after the store changed.

    Args:
        kind: ``"edit"`` or ``"delete"``.
        prediction: the generated answer.
        substitute: the edit's replacement string (None for a delete).
        answers: the original reference answers.
        exact_match: ``(prediction, references) -> 0 or 1``.

    Returns:
        ``follow`` / ``stuck`` / ``other`` for an edit; ``stuck`` / ``abstained`` / ``other`` for a
        delete.
    """
    if kind == "edit" and substitute is not None and exact_match(prediction, [substitute]):
        return "follow"
    if exact_match(prediction, answers):
        return "stuck"
    if kind == "delete" and abstention.is_abstention(prediction):
        return "abstained"
    return "other"


def flip_rates(labels: Sequence[Tuple[str, str]]) -> dict:
    """Per kind: counts per label, the flip rate and its PASS or FAIL against the bar.

    Args:
        labels: ``(kind, label)`` pairs from ``classify_edit``.
    """
    out = {}
    for kind in ("edit", "delete"):
        mine = [label for k, label in labels if k == kind]
        flipped = sum(1 for label in mine if (label == "follow" if kind == "edit" else label != "stuck"))
        rate = flipped / len(mine) if mine else float("nan")
        out[kind] = {"n": len(mine), "counts": {l: mine.count(l) for l in sorted(set(mine))},
                     "flip_rate": rate, "sigma": binomial_sigma(rate, len(mine)) if mine else float("nan"),
                     "pass": bool(mine) and rate >= FLIP_BAR}
    return out


# ------------------------------------------------------------------------------ query readout


class QueryReadout:
    """Reads the selector's query out of a forward pass with pre-hooks on every IR module.

    Use as a context manager around the forward. Set ``positions`` (flat indices into the
    ``[B * S]`` token axis) before each forward; afterwards ``captured[(ir_index, loop)]`` holds
    the float32 unit query rows at those positions, exactly the ``x_norm`` the module scores
    external chunks with. The hooks change nothing; they are removed on exit.

    Args:
        model: a ``TinyMoETransformer`` with at least one IR expert.
    """

    def __init__(self, model):
        self.model = model
        self.modules = list(model.moe.ir_modules)
        if not self.modules:
            raise ValueError("the model has no IR expert, there is no selector query to read")
        self.n_loops = int(model.moe.n_loops)
        self.positions: Optional[torch.Tensor] = None
        self.tokens: Optional[int] = None
        self.captured: Dict[Tuple[int, int], torch.Tensor] = {}
        self._handles = []

    def _hook(self, ir_index: int):
        def hook(module, args, kwargs):
            if self.positions is None:
                return None
            down = args[0]
            flat = down.reshape(-1, down.shape[-1])
            if self.tokens is not None and flat.shape[0] != self.tokens:
                raise RuntimeError(f"IR input has {flat.shape[0]} rows, expected {self.tokens}")
            loop = int(kwargs.get("loop_idx", 0))
            with torch.no_grad():
                rows = flat.index_select(0, self.positions)
                rows = rows + module._loop_query_bias(loop, rows.dtype)
                self.captured[(ir_index, loop)] = F.normalize(rows, p=2, dim=-1).float()
            return None
        return hook

    def __enter__(self):
        for i, module in enumerate(self.modules):
            self._handles.append(module.register_forward_pre_hook(self._hook(i), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False


@torch.inference_mode()
def extract_queries(model, prompts: List[List[int]], *, pad_id: int, device: str,
                    batch_size: int) -> Dict[Tuple[int, int], np.ndarray]:
    """The selector query at each prompt's last token, for every IR expert and loop.

    Args:
        model: the checkpoint's model, eval mode.
        prompts: token ids per question, ending at the answer-start position.
        pad_id: right padding id.
        device: model device.
        batch_size: prompts per forward.

    Returns:
        ``{(ir_index, loop_index): [len(prompts), 384] float32}``, rows in prompt order. The
        forward carries no evidence and skips the MTP head.
    """
    from modules.model.attention import cu_seqlens_from_doc_ids

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    pieces: Dict[Tuple[int, int], List[Tuple[List[int], torch.Tensor]]] = {}
    with QueryReadout(model) as readout:
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            width = max(len(prompts[i]) for i in index)
            packed = np.full((len(index), width), pad_id, dtype=np.int64)
            real = np.zeros((len(index), width), dtype=np.int64)
            for row, i in enumerate(index):
                packed[row, :len(prompts[i])] = prompts[i]
                real[row, :len(prompts[i])] = 1
            ids = torch.from_numpy(packed).to(device)
            doc = torch.from_numpy(real).to(device)
            last = np.array([len(prompts[i]) - 1 for i in index], dtype=np.int64)
            readout.positions = torch.from_numpy(np.arange(len(index), dtype=np.int64) * width + last).to(device)
            readout.tokens = len(index) * width
            readout.captured = {}
            cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(doc)
            model(input_ids=ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                  return_hidden=True, skip_mtp=True)
            for key, value in readout.captured.items():
                pieces.setdefault(key, []).append((index, value.cpu()))
    out = {}
    for key, parts in pieces.items():
        result = np.zeros((len(prompts), parts[0][1].shape[-1]), dtype=np.float32)
        for index, value in parts:
            result[index] = value.numpy()
        out[key] = result
    return out


@torch.inference_mode()
def adapted_index(model, keys: np.ndarray, ir_index: int, device: str, block: int = 16384) -> torch.Tensor:
    """``normalize(key_adapter(keys))`` for one IR expert, float32 ``[N, 384]`` on ``device``.

    Computed the way the module computes it per forward (keys cast to the adapter's dtype), once
    per checkpoint and store.

    Args:
        model: the checkpoint's model.
        keys: ``[N, 384]`` store keys.
        ir_index: which IR expert's adapter.
        device: where the index lives.
        block: rows adapted per call.
    """
    module = model.moe.ir_modules[ir_index]
    if module.key_adapter is None:
        raise ValueError("this checkpoint has no key adapter: the evidence port is missing")
    dtype = module.key_adapter.weight.dtype
    out = torch.zeros((keys.shape[0], module.latent_dim), dtype=torch.float32, device=device)
    for start in range(0, keys.shape[0], block):
        chunk = torch.from_numpy(keys[start:start + block].astype(np.float32)).to(device=device, dtype=dtype)
        out[start:start + block] = F.normalize(module.key_adapter(chunk), p=2, dim=-1).float()
    return out


# -------------------------------------------------------------------------- model + generation


def load_eval_model(args):
    """Load checkpoint, tokenizer and chat template; refuse a model without a selector.

    Args:
        args: parsed arguments with ``checkpoint``, ``tokenizer`` and ``device``.
    """
    from scripts.eval_abstention import load_model

    if not args.checkpoint:
        raise SystemExit("this subcommand needs -c CHECKPOINT")
    model = load_model(args.checkpoint, args.device)
    modules = model.moe.ir_modules
    if not modules or any(m.key_adapter is None for m in modules):
        raise SystemExit("the checkpoint has no evidence port, so it has no selector query to read")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    return model, tokenizer, ChatTemplate(tokenizer)


def question_prompt(template: ChatTemplate, question: str) -> List[int]:
    """The answer-start prompt for a question: the evidence-mode instruction as a user turn."""
    return template.encode_prompt([{"role": "user", "content": evidence_prompt(question)}])


def evidence_row(tokenizer, store: Store, chunk_ids: Sequence[int]) -> Optional[dict]:
    """An evidence row (token ids, per-token chunk index, key per chunk) for the given chunks.

    Args:
        tokenizer: the repo tokenizer.
        store: the store the ids index.
        chunk_ids: chunks in buffer order.

    Returns:
        None when there are no tokens to attach.
    """
    ids, chunk_index, kept = [], [], []
    for chunk_id in chunk_ids:
        tokens = tokenizer(store.chunks[int(chunk_id)]["text"], add_special_tokens=False)["input_ids"]
        if not tokens:
            continue
        ids.extend(tokens)
        chunk_index.extend([len(kept)] * len(tokens))
        kept.append(int(chunk_id))
    if not ids:
        return None
    return {"ids": ids, "chunk_ids": chunk_index, "keys": store.keys[kept].astype(np.float32)}


def generate_answers(model, tokenizer, template, prompts: List[List[int]],
                     rows: Optional[List[Optional[dict]]], args) -> List[str]:
    """Greedy answers for every prompt, in prompt order.

    Args:
        model: the checkpoint's model.
        tokenizer: the repo tokenizer.
        template: chat template (supplies the end-of-turn id).
        prompts: prompt ids.
        rows: one evidence row (or None) per prompt, or None for no evidence on any prompt.
        args: parsed arguments with ``batch_size``, ``max_new_tokens`` and ``device``.
    """
    from config import ModelConfig
    from scripts.eval_abstention import generate_batch

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    answers = [""] * len(prompts)
    for start in range(0, len(order), args.batch_size):
        index = order[start:start + args.batch_size]
        generated, _, _ = generate_batch(
            model, [prompts[i] for i in index], max_new_tokens=args.max_new_tokens, temperature=0.0,
            top_k=0, eos_id=template.eos_id, pad_id=tokenizer.pad_token_id, device=args.device,
            max_seq_len=ModelConfig.Params["max_seq_len"],
            evidence_rows=None if rows is None else [rows[i] for i in index],
        )
        for i, token_ids in zip(index, generated):
            answers[i] = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if (start // args.batch_size) % 20 == 0:
            logger.info(f"[eval_store] generated {min(start + args.batch_size, len(order)):,}/{len(order):,}")
    return answers


def sample_questions(store: Store, per_source: Optional[int], seed: int) -> List[int]:
    """Indices into ``store.questions``: a seeded sample of at most ``per_source`` per source.

    Args:
        store: a loaded store with questions.
        per_source: cap per source, or None for every question.
        seed: seeds the shuffle, so the same sample is drawn every run.
    """
    rng = random.Random(seed)
    chosen = []
    for source in sorted({q["source"] for q in store.questions}):
        mine = [i for i, q in enumerate(store.questions) if q["source"] == source]
        rng.shuffle(mine)
        chosen.extend(mine[:per_source] if per_source else mine)
    return sorted(chosen)


def headline_loop(args, model) -> int:
    """Zero-based index of the loop the headline numbers are read at, from ``--loop`` (one-based).

    Args:
        args: parsed arguments with ``loop``.
        model: the loaded model, whose ``moe.n_loops`` bounds the choice.
    """
    loop = int(args.loop) - 1
    if not 0 <= loop < int(model.moe.n_loops):
        raise SystemExit(f"--loop {args.loop} is outside 1..{model.moe.n_loops}")
    return loop


# -------------------------------------------------------------------------------- subcommands


def cmd_recall(args) -> dict:
    """Retrieval recall of the store's query keys, per source, by the bge cosine.

    Args:
        args: parsed arguments (``store``, ``k``, ``gate_k``, ``bge_only``, ``checkpoint`` and ``loop``).

    Returns:
        The summary dict that is also printed and optionally written as JSON.
    """
    store = load_store(args.store)
    if not store.questions or store.query_keys is None:
        raise SystemExit(f"{args.store} needs questions.jsonl and query_keys_bge.npy")
    ks = sorted(set(int(k) for k in args.k.split(",")) | {args.gate_k})
    kmax = max(ks)
    device = args.device
    bge_top = topk_ids(store.query_keys.astype(np.float32), store.keys.astype(np.float32), kmax, device)
    result = {"store": args.store, "ks": ks, "n_questions": len(store.questions),
              "n_chunks": len(store.chunks), "bge": recall_summary(bge_top, store.questions, ks), "model": {}}
    rows = [("bge", result["bge"])]

    if not args.bge_only:
        model, tokenizer, template = load_eval_model(args)
        prompts = [question_prompt(template, q["question"]) for q in store.questions]
        queries = extract_queries(model, prompts, pad_id=tokenizer.pad_token_id, device=args.device,
                                  batch_size=args.batch_size)
        for ir in range(len(model.moe.ir_modules)):
            index = adapted_index(model, store.keys, ir, device)
            for loop in range(int(model.moe.n_loops)):
                top = topk_ids(queries[(ir, loop)], index, kmax, device)
                name = f"ir{ir}_loop{loop + 1}"
                result["model"][name] = recall_summary(top, store.questions, ks)
                rows.append((name, result["model"][name]))
                if ir == 0 and loop == int(args.loop) - 1:
                    gate = paired_counts(hit_vector(top, store.questions, args.gate_k),
                                         hit_vector(bge_top, store.questions, args.gate_k))
                    result["gate"] = {"k": args.gate_k, "loop": args.loop, "ir": 0,
                                      "model": result["model"][name]["all"][args.gate_k]["recall"],
                                      "bge": result["bge"]["all"][args.gate_k]["recall"], **gate,
                                      "pass": gate["diff"] >= 0}
            del index

    print(f"\nrecall@k, {len(store.questions):,} questions over {len(store.chunks):,} chunks")
    for source in result["bge"]:
        print(f"  {source}")
        print("    " + "system".ljust(14) + "".join(f"@{k}".rjust(8) for k in ks) + "   both-gold")
        for name, summary in rows:
            cells = "".join(f"{summary[source][k]['recall']:8.3f}" for k in ks)
            both = summary[source][ks[-1]]["both"]
            print("    " + name.ljust(14) + cells + (f"   {both:.3f}@{ks[-1]}" if both is not None else ""))
    gate = result.get("gate")
    if gate:
        print(f"\n  GATE recall@{gate['k']} loop {gate['loop']}: model {gate['model']:.3f} vs bge "
              f"{gate['bge']:.3f}, diff {gate['diff']:+.4f} sigma {gate['sigma']:.4f} z {gate['z']:+.2f} "
              f"(model only {gate['a_only']}, bge only {gate['b_only']}, n {gate['n']}): "
              f"{'PASS' if gate['pass'] else 'FAIL'}")
    return result


def _arm_summary(records: List[dict], arm: str, source: Optional[str]) -> dict:
    mine = [r for r in records if source is None or r["source"] == source]
    n = len(mine)
    em = float(np.mean([r[arm]["em"] for r in mine])) if n else float("nan")
    return {"n": n, "em": em, "em_sigma": binomial_sigma(em, n),
            "f1": float(np.mean([r[arm]["f1"] for r in mine])) if n else float("nan"),
            "abstain": float(np.mean([r[arm]["abstained"] for r in mine])) if n else float("nan")}


def cmd_pathway(args) -> dict:
    """Answer questions through the model's own selector over the store and score the answers.

    Args:
        args: parsed arguments (``store``, ``checkpoint``, ``loop``, ``buffer``, ``max_questions``, ``seed``).

    Returns:
        The summary dict that is also printed and optionally written as JSON.
    """
    from scripts.eval_abstention import exact_match, token_f1

    store = load_store(args.store)
    if not store.questions or store.query_keys is None:
        raise SystemExit(f"{args.store} needs questions.jsonl and query_keys_bge.npy")
    model, tokenizer, template = load_eval_model(args)
    loop = headline_loop(args, model)
    chosen = sample_questions(store, args.max_questions, args.seed)
    questions = [store.questions[i] for i in chosen]
    prompts = [question_prompt(template, q["question"]) for q in questions]

    queries = extract_queries(model, prompts, pad_id=tokenizer.pad_token_id, device=args.device,
                              batch_size=args.batch_size)[(0, loop)]
    model_top = topk_ids(queries, adapted_index(model, store.keys, 0, args.device), args.buffer, args.device)
    bge_top = topk_ids(store.query_keys[chosen].astype(np.float32), store.keys.astype(np.float32),
                       args.buffer, args.device)
    buffers = {
        "model": [list(map(int, row)) for row in model_top],
        "bge": [list(map(int, row)) for row in bge_top],
        "oracle": [q["gold_chunk_ids"][:args.buffer] for q in questions],
        "none": [[] for _ in questions],
    }
    records = [{"qid": q["qid"], "source": q["source"]} for q in questions]
    for arm, buffer in buffers.items():
        logger.info(f"[pathway] arm {arm}")
        rows = None if arm == "none" else [evidence_row(tokenizer, store, ids) for ids in buffer]
        answers = generate_answers(model, tokenizer, template, prompts, rows, args)
        for record, q, answer in zip(records, questions, answers):
            record[arm] = {"completion": answer, "em": exact_match(answer, q["answers"]),
                           "f1": token_f1(answer, q["answers"]),
                           "abstained": float(abstention.is_abstention(answer))}

    arms = list(buffers)
    sources = sorted({r["source"] for r in records})
    result = {"store": args.store, "buffer": args.buffer, "loop": args.loop, "max_questions": args.max_questions,
              "arms": {arm: {"all": _arm_summary(records, arm, None),
                             **{s: _arm_summary(records, arm, s) for s in sources}} for arm in arms},
              "records": records}
    print(f"\npathway, buffer {args.buffer}, model query at loop {args.loop}")
    for source in ["all"] + sources:
        print(f"  {source}")
        for arm in arms:
            s = result["arms"][arm][source]
            print(f"    {arm:<8} n {s['n']:5d}  EM {s['em']:.3f} +/- {s['em_sigma']:.3f}  F1 {s['f1']:.3f}  "
                  f"abstain {s['abstain']:.3f}")
        model_s, oracle_s = result["arms"]["model"][source], result["arms"]["oracle"][source]
        gap = model_s["em"] - oracle_s["em"]
        sigma = math.sqrt(model_s["em_sigma"] ** 2 + oracle_s["em_sigma"] ** 2)
        print(f"    GATE pathway EM {model_s['em']:.3f} vs oracle EM {oracle_s['em']:.3f}: gap {gap:+.3f} "
              f"(sigma {sigma:.3f}, z {gap / sigma if sigma else 0.0:+.2f})")
    return result


def cmd_edit(args) -> dict:
    """Score the same questions on the original and the edit store and classify each answer change.

    Args:
        args: parsed arguments (``store``, ``edit_store``, ``checkpoint``, ``loop``, ``max_questions``, ``seed``).

    Returns:
        The summary dict that is also printed and optionally written as JSON.
    """
    from scripts.eval_abstention import exact_match

    original = load_store(args.store)
    edited = load_store(args.edit_store)
    if not edited.edits:
        raise SystemExit(f"{args.edit_store} has no edits.jsonl")
    by_qid = {q["qid"]: q for q in original.questions}
    edit_questions = {q["qid"]: q for q in edited.questions}
    edits = [e for e in edited.edits if e["qid"] in by_qid]
    if args.max_questions and len(edits) > args.max_questions:
        random.Random(args.seed).shuffle(edits)
        edits = edits[:args.max_questions]

    model, tokenizer, template = load_eval_model(args)
    loop = headline_loop(args, model)
    questions = [by_qid[e["qid"]] for e in edits]
    prompts = [question_prompt(template, q["question"]) for q in questions]
    queries = extract_queries(model, prompts, pad_id=tokenizer.pad_token_id, device=args.device,
                              batch_size=args.batch_size)[(0, loop)]

    top = topk_ids(queries, adapted_index(model, original.keys, 0, args.device), args.buffer, args.device)
    rows = [evidence_row(tokenizer, original, ids) for ids in top]
    before = generate_answers(model, tokenizer, template, prompts, rows, args)
    correct = [i for i, (q, a) in enumerate(zip(questions, before)) if exact_match(a, q["answers"])]
    logger.info(f"[edit] {len(correct):,}/{len(edits):,} answered correctly on the original store")

    sub_prompts = [prompts[i] for i in correct]
    top_after = topk_ids(queries[correct], adapted_index(model, edited.keys, 0, args.device),
                         args.buffer, args.device)
    after = generate_answers(model, tokenizer, template, sub_prompts,
                             [evidence_row(tokenizer, edited, ids) for ids in top_after], args)
    labels, per_item = [], []
    for i, answer in zip(correct, after):
        e, q = edits[i], questions[i]
        label = classify_edit(e["kind"], answer, e["substitute"], q["answers"], exact_match)
        labels.append((e["kind"], label))
        per_item.append({"qid": e["qid"], "kind": e["kind"], "label": label, "completion": answer})

    oracle_idx = [i for i in correct if edits[i]["kind"] == "edit" and edits[i]["gold_edited_chunk_ids"]]
    oracle_labels = []
    if oracle_idx:
        oracle_rows = [evidence_row(tokenizer, edited, edits[i]["gold_edited_chunk_ids"][:args.buffer])
                       for i in oracle_idx]
        oracle_answers = generate_answers(model, tokenizer, template, [prompts[i] for i in oracle_idx],
                                          oracle_rows, args)
        oracle_labels = [("edit", classify_edit("edit", a, edits[i]["substitute"], questions[i]["answers"],
                                                exact_match))
                         for i, a in zip(oracle_idx, oracle_answers)]

    result = {"store": args.store, "edit_store": args.edit_store, "buffer": args.buffer, "loop": args.loop,
              "n_edits": len(edits), "n_correct_before": len(correct),
              "flip": flip_rates(labels), "oracle_edit": flip_rates(oracle_labels)["edit"],
              "items": per_item, "n_questions_in_edit_store": len(edit_questions)}
    print(f"\nstore edit: {len(correct):,} of {len(edits):,} questions answered correctly before the edit")
    for kind, row in result["flip"].items():
        print(f"  {kind:<7} n {row['n']:5d}  counts {row['counts']}  flip {row['flip_rate']:.3f} "
              f"+/- {row['sigma']:.3f}  (bar {FLIP_BAR:.2f}): {'PASS' if row['pass'] else 'FAIL'}")
    oracle = result["oracle_edit"]
    print(f"  diagnostic, edited gold chunk attached directly: n {oracle['n']}  counts {oracle['counts']}  "
          f"follow {oracle['flip_rate']:.3f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="store recall, pathway and edit evals with the model's own query")
    parser.add_argument("mode", choices=("recall", "pathway", "edit"))
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument("--store", required=True)
    parser.add_argument("--edit-store", default=None)
    parser.add_argument("--bge-only", action="store_true", help="recall: the bge row only, no checkpoint")
    parser.add_argument("--k", default="1,2,5,10,20")
    parser.add_argument("--gate-k", type=int, default=5)
    parser.add_argument("--buffer", type=int, default=4)
    parser.add_argument("--loop", type=int, default=1, help="1-based loop whose query retrieves")
    parser.add_argument("--max-questions", type=int, default=2000, help="per source, seeded")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.device is None:
        args.device = "cpu" if args.bge_only else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.bge_only and args.mode != "recall":
        raise SystemExit("--bge-only only applies to recall")
    if args.mode == "edit" and not args.edit_store:
        raise SystemExit("edit needs --edit-store")

    result = {"recall": cmd_recall, "pathway": cmd_pathway, "edit": cmd_edit}[args.mode](args)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"mode": args.mode, "checkpoint": args.checkpoint,
                       "flags": {k: v for k, v in vars(args).items() if k != "hf_token"}, **result},
                      handle, indent=2)
        logger.info(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

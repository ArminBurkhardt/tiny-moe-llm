"""Composition by read sites: accuracy on chain questions as the evidence reads are cut away.

Each question has a buffer (``scripts/prepare_chain_data.py``) and a set of candidate
answers: the distinct objects of the chunks carrying the final relation, which the wh-frame names. For every candidate the script sums the log-probability of ``candidate tokens + EOS`` after
the training prompt with the question's buffer attached, and the prediction is the argmax. That is
repeated at fixed depth D (the model is run at ``n_loops=D``) for every number j of kept read sites
from 0 to D, through ``reader_sites_kept``: every site above j reads nothing, so j sequential reads
are all the model has to work with.

The validity rule for the instrument: a question needing k sequential reads cannot be answered with
fewer than k sites, so every cell with ``j < hops`` must sit at chance (one over the number of
candidates, averaged over the cell). The 1-hop curve at one or more sites must be at least three
sigma above chance; if it is not, the instrument is "not readable" and says so rather than passing.
A checkpoint that never trained on chains will usually give "not readable": the instrument is first
readable on a chain trained one.

Besides the grid the report carries, per split and depth, the gold answer's NLL against ln K (the
cost of a uniform pick among the K candidates), the per-token answer CE and the NLL at the first
token where the candidates diverge (the token that carries the choice); the depth readings (answer
cross entropy by kept sites on the 4-hop split, and by depth 3 against 4 at full sites); the held
out composition 2-hop accuracy beside the seen 2-hop accuracy; and a paired delta between the
fewest and the most kept sites with a paired bootstrap sigma. The JSON holds one record per
question and cell (candidate scores, gold rank, margin, NLLs) under ``records``, so any two cells
can be paired by position.

Usage:

    python scripts/eval_chains.py -c CKPT --data-dir data/prepared \\
        --splits chains_eval,chains_heldout_tmpl,chains_hop4 --depths 3,4 --sites all \\
        --max-questions 1000 --batch-size 16 --json-out ckpts/chains_eval.json
"""
import os
import sys
import json
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer

from modules.data import chains
from modules.data.chat import ChatTemplate
from modules.model.attention import _segment_ids, cu_seqlens_from_doc_ids
from scripts.eval_abstention import _pack_evidence_batch, load_model
from scripts.prepare_evidence_data import EMBED_DIM, evidence_prompt
from utils import TOKENIZER_DIR, logger


class ChainSplit:
    """One built chain split, read by memmap: questions, candidates and each buffer."""

    def __init__(self, data_dir: str, split: str):
        self.split = split

        def mm(suffix, dtype):
            return np.memmap(os.path.join(data_dir, f"{split}.{suffix}"), dtype=dtype, mode="r")

        self.ev, self.evidx, self.evchunk = mm("ev", np.uint16), mm("evidx", np.uint64), mm("evchunk", np.uint16)
        self.evkey, self.evkeyidx = mm("evkey", np.float16), mm("evkeyidx", np.uint64)
        self.evkey = self.evkey.reshape(-1, EMBED_DIM)
        with open(os.path.join(data_dir, f"{split}.chains.jsonl"), encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f]
        assert len(self.meta) == len(self.evidx) - 1, f"{split}: metadata and evidence disagree"

    def __len__(self) -> int:
        return len(self.meta)

    def evidence_row(self, i: int) -> dict:
        """The C4 evidence row of document ``i``."""
        a, b = int(self.evidx[i]), int(self.evidx[i + 1])
        ca, cb = int(self.evkeyidx[i]), int(self.evkeyidx[i + 1])
        return {"ids": self.ev[a:b].astype(np.int64).tolist(),
                "chunk_ids": self.evchunk[a:b].astype(np.int64).tolist(),
                "keys": np.asarray(self.evkey[ca:cb], dtype=np.float32)}

    def select(self, max_per_hop: int) -> List[int]:
        """Document indices, the first ``max_per_hop`` of each hop count, in file order."""
        taken, out = defaultdict(int), []
        for i, m in enumerate(self.meta):
            if taken[m["hops"]] < max_per_hop:
                taken[m["hops"]] += 1
                out.append(i)
        return out


def encode_items(split: ChainSplit, indices: Sequence[int], tokenizer, template: ChatTemplate) -> List[dict]:
    """Prompt ids, candidate ids (answer tokens plus EOS) and the buffer for each question."""
    items = []
    for i in indices:
        m = split.meta[i]
        prompt = template.encode_prompt([{"role": "user", "content": evidence_prompt(m["question"])}])
        cands = m["candidates"]
        pieces = tokenizer(cands, add_special_tokens=False)["input_ids"]
        items.append({
            "index": i, "hops": m["hops"], "answer": m["answer"], "candidates": cands,
            "gold": cands.index(m["answer"]), "prompt": prompt,
            "cand_ids": [p + [template.eos_id] for p in pieces], "evidence": split.evidence_row(i),
        })
    return items


@torch.no_grad()
def score_items(model, items: List[dict], *, depth: int, kept: Optional[int], pad_id: int,
                batch_size: int, device: str, max_seq_len: int,
                return_tokens: bool = False) -> Union[List[List[float]], Tuple[List[List[float]], List[List[List[float]]]]]:
    """Summed log-probability of every candidate of every item, at depth ``depth`` with ``kept`` sites.

    Rows are right padded with one extra pad column, which gives every row a second attention
    segment to dump the shared evidence width into (see ``_pack_evidence_batch``). The head is
    applied only at the positions that predict a candidate token.

    Args:
        items: from ``encode_items``.
        depth: ``n_loops`` for the forward.
        kept: ``reader_sites_kept`` for the forward; None keeps every site.
        batch_size: rows (question, candidate pairs) per forward.
        max_seq_len: longest row allowed.
        return_tokens: also return every candidate's per-token log-probabilities.

    Returns:
        Per item, one summed log-probability per candidate; with ``return_tokens`` a pair of that and
        the per-token log-probabilities (item, candidate, token, EOS last).
    """
    rows = [(n, c) for n, item in enumerate(items) for c in range(len(item["candidates"]))]
    scores = [[0.0] * len(item["candidates"]) for item in items]
    token_lps = [[[] for _ in item["candidates"]] for item in items]
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        seqs = [items[n]["prompt"] + items[n]["cand_ids"][c] for n, c in chunk]
        width = max(len(s) for s in seqs) + 1
        assert width <= max_seq_len, f"row of {width} tokens exceeds --max-seq-len {max_seq_len}"
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        doc = torch.zeros((len(chunk), width), dtype=torch.long)
        predict, target, owner = [], [], []
        for r, ((n, c), seq) in enumerate(zip(chunk, seqs)):
            ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            doc[r, :len(seq)] = 1
            p, k = len(items[n]["prompt"]), len(items[n]["cand_ids"][c])
            predict.extend(r * width + p - 1 + t for t in range(k))
            target.extend(seq[p:p + k])
            owner.extend([r] * k)
        ids, doc = ids.to(device), doc.to(device)
        cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(doc)
        seg = _segment_ids(cu_seqlens, len(chunk), width, device)
        evidence = _pack_evidence_batch(
            model, seg[:, 0], seg[:, -1], [items[n]["evidence"] for n, _ in chunk],
            num_segments=int(cu_seqlens.numel() - 1), device=device,
        )
        out = model(
            input_ids=ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, return_hidden=True,
            skip_mtp=True, evidence=evidence, n_loops=depth, reader_sites_kept=kept,
        )
        hidden = (out[0] if isinstance(out, tuple) else out)[-1]
        h = hidden.reshape(-1, hidden.size(-1))[torch.tensor(predict, device=device)]
        logp = model.lm_head(h).float().log_softmax(-1)
        token_lp = logp.gather(-1, torch.tensor(target, device=device).unsqueeze(-1)).squeeze(-1)
        per_row = torch.zeros(len(chunk), device=device).index_add_(0, torch.tensor(owner, device=device), token_lp)
        for (n, c), value in zip(chunk, per_row.tolist()):
            scores[n][c] = value
        flat, at = token_lp.tolist(), 0
        for n, c in chunk:
            k = len(items[n]["cand_ids"][c])
            token_lps[n][c] = flat[at:at + k]
            at += k
    return (scores, token_lps) if return_tokens else scores


def parse_sites(text: str, depth: int) -> List[int]:
    """``all`` is 0..depth; otherwise a comma list of kept-site counts, clipped to the depth.

    Args:
        text: ``all`` or a comma list such as ``0,1,3``.
        depth: the forward's ``n_loops``.
    """
    if text == "all":
        return list(range(depth + 1))
    wanted = sorted({min(int(s), depth) for s in text.split(",") if s.strip()})
    return wanted


def tally(items: List[dict], scores: List[List[float]]) -> Dict[int, dict]:
    """Per hop count: correctness, chance and the gold answer's log-probability and token count.

    Chance per question is one over its candidates, which are the final relation objects of the
    buffer, the only answers a reader can pick without composing.

    Args:
        items: from ``encode_items``.
        scores: from ``score_items``, one list of candidate log-probabilities per item.
    """
    by_hops: Dict[int, dict] = defaultdict(lambda: {"correct": [], "chance": [], "gold_lp": [], "gold_n": []})
    for item, s in zip(items, scores):
        cell = by_hops[item["hops"]]
        cell["correct"].append(int(np.argmax(s)) == item["gold"])
        cell["chance"].append(1.0 / len(item["candidates"]))
        cell["gold_lp"].append(s[item["gold"]])
        cell["gold_n"].append(len(item["cand_ids"][item["gold"]]))
    return by_hops


def answer_ce(cell: dict) -> float:
    """Mean cross entropy per gold answer token (EOS included) over a cell.

    Args:
        cell: one entry of ``tally``, with ``gold_lp`` and ``gold_n`` lists.
    """
    return -sum(cell["gold_lp"]) / max(sum(cell["gold_n"]), 1)


def divergent_position(cand_ids: Sequence[Sequence[int]]) -> int:
    """First token index where the candidates' id sequences are not all equal.

    Args:
        cand_ids: one id list per candidate (EOS included).

    Returns:
        The index, 0 when the first tokens differ or when no position within the shortest
        candidate separates them.
    """
    for t in range(min(len(c) for c in cand_ids)):
        if len({c[t] for c in cand_ids}) > 1:
            return t
    return 0


def build_records(items: List[dict], scores: List[List[float]], token_lps: List[List[List[float]]]) -> List[dict]:
    """One record per question: the candidate scores and what they say about the gold answer.

    Rank 0 is best; ties go to the lower candidate index, as ``argmax`` does, so ``correct`` is
    ``rank == 0``. The margin is the gold score minus the best other candidate (0 with one
    candidate). NLL per answer sums the gold's tokens, EOS included.

    Args:
        items: from ``encode_items``.
        scores: from ``score_items``.
        token_lps: per-token log-probabilities from ``score_items(return_tokens=True)``.
    """
    records = []
    for item, s, tl in zip(items, scores, token_lps):
        g, k = item["gold"], len(s)
        rank = sum(1 for c, v in enumerate(s) if v > s[g] or (v == s[g] and c < g))
        others = [v for c, v in enumerate(s) if c != g]
        pos = divergent_position(item["cand_ids"])
        n_gold = len(item["cand_ids"][g])
        records.append({
            "question": item["index"], "hops": item["hops"], "gold": g, "n_candidates": k,
            "scores": [round(v, 4) for v in s], "gold_rank": rank, "correct": rank == 0,
            "margin": round(s[g] - max(others), 4) if others else 0.0,
            "gold_tokens": n_gold, "gold_nll": round(-s[g], 4),
            "gold_nll_per_token": round(-s[g] / max(n_gold, 1), 4), "ln_k": round(math.log(k), 4),
            "divergent_pos": pos, "gold_nll_at_divergence": round(-tl[g][pos], 4),
        })
    return records


def cell_extras(records: List[dict]) -> dict:
    """Mean gold NLL per answer, mean ln K, CE per gold token and mean NLL at the divergent token.

    Args:
        records: ``build_records`` entries of one cell.
    """
    n = max(len(records), 1)
    return {
        "gold_nll_per_answer": sum(r["gold_nll"] for r in records) / n,
        "ln_k": sum(r["ln_k"] for r in records) / n,
        "ce_per_token": sum(r["gold_nll"] for r in records) / max(sum(r["gold_tokens"] for r in records), 1),
        "nll_at_divergence": sum(r["gold_nll_at_divergence"] for r in records) / n,
    }


def paired_delta(lo: Sequence[bool], hi: Sequence[bool], n_boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy of ``hi`` minus accuracy of ``lo`` on the same questions, with a paired bootstrap sigma.

    Args:
        lo: per question correct flags of the first cell.
        hi: per question correct flags of the second cell, same question order.
        n_boot: bootstrap resamples of the questions.
        seed: seed of the resampling generator.
    """
    assert len(lo) == len(hi), "paired cells must hold the same questions"
    n = len(lo)
    if n == 0:
        return {"delta": float("nan"), "sigma": float("nan"), "n": 0}
    d = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    return {"delta": float(d.mean()), "sigma": float(d[idx].mean(axis=1).std(ddof=1)), "n": n}


def paired_delta_by_hops(by_kept: Dict[int, List[dict]]) -> Dict[int, dict]:
    """Per hop count, accuracy at the most kept sites minus accuracy at one kept site, paired by question.

    Args:
        by_kept: records of one split and depth, keyed by kept-site count, in the same question order.
    """
    if 1 not in by_kept or max(by_kept) <= 1:
        return {}
    hi_kept = max(by_kept)
    out = {}
    for hops in sorted({r["hops"] for r in by_kept[1]}):
        lo = [r for r in by_kept[1] if r["hops"] == hops]
        hi = [r for r in by_kept[hi_kept] if r["hops"] == hops]
        assert [r["question"] for r in lo] == [r["question"] for r in hi], "cells are not paired"
        out[hops] = dict(paired_delta([r["correct"] for r in lo], [r["correct"] for r in hi]),
                         kept_lo=1, kept_hi=hi_kept)
    return out


ACC_WIDTH = 29
NLL_WIDTH = 34


def accuracy_table(by_kept: Dict[int, Dict[int, dict]]) -> List[str]:
    """Header and one row per kept count; a two character marker column precedes each cell.

    Args:
        by_kept: grid cells of one split and depth, ``by_kept[kept][hops]``.
    """
    hops_seen = sorted({h for k in by_kept.values() for h in k})
    lines = ["kept  " + "".join(f"{f'hops {h}':>{ACC_WIDTH + 2}}" for h in hops_seen)]
    for kept, by_hops in by_kept.items():
        line = f"{kept:>4}  "
        for h in hops_seen:
            s = by_hops.get(h)
            if s is None:
                line += " " * (ACC_WIDTH + 2)
                continue
            mark = "*" if kept < h else " "
            body = f"{s['acc']:.3f}+-{s['sigma']:.3f} | {s['chance']:.3f} | {s['z_vs_chance']:+6.1f}"
            line += f" {mark}{body:>{ACC_WIDTH}}"
        lines.append(line)
    return lines


def nll_table(by_kept: Dict[int, Dict[int, dict]]) -> List[str]:
    """Gold NLL per answer, mean ln K, CE per token and NLL at the divergent token, per kept count and hops.

    Args:
        by_kept: grid cells of one split and depth carrying the ``cell_extras`` keys.
    """
    hops_seen = sorted({h for k in by_kept.values() for h in k})
    lines = ["kept  " + "".join(f"{f'hops {h}':>{NLL_WIDTH}}" for h in hops_seen)]
    for kept, by_hops in by_kept.items():
        line = f"{kept:>4}  "
        for h in hops_seen:
            s = by_hops.get(h)
            if s is None or "gold_nll_per_answer" not in s:
                line += " " * NLL_WIDTH
                continue
            body = (f"{s['gold_nll_per_answer']:6.3f} /{s['ln_k']:6.3f} | {s['ce_per_token']:5.3f}"
                    f" | {s['nll_at_divergence']:6.3f}")
            line += f"{body:>{NLL_WIDTH}}"
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/prepared")
    parser.add_argument("--splits", default="chains_eval,chains_heldout_tmpl,chains_hop4")
    parser.add_argument("--depths", default="3,4")
    parser.add_argument("--sites", default="all", help="'all' or a comma list of kept-site counts")
    parser.add_argument("--max-questions", type=int, default=1000, help="per split and hop count")
    parser.add_argument("--batch-size", type=int, default=16, help="candidate rows per forward")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    template = ChatTemplate(tokenizer)
    model = load_model(args.checkpoint, device)
    assert model.moe.evidence_port, "this checkpoint has no evidence port"
    depths = [int(d) for d in args.depths.split(",")]
    pad_id = template.eos_id
    common = dict(pad_id=pad_id, batch_size=args.batch_size, device=device, max_seq_len=args.max_seq_len)

    grid: Dict[str, dict] = {}
    cells: List[dict] = []
    ce_by_split: Dict[str, dict] = {}
    records: Dict[str, dict] = {}
    for split_name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split = ChainSplit(args.data_dir, split_name)
        items = encode_items(split, split.select(args.max_questions), tokenizer, template)
        logger.info(f"{split_name}: {len(items)} questions")
        grid[split_name], ce_by_split[split_name], records[split_name] = {}, {}, {}
        for depth in depths:
            if split_name == args.splits.split(",")[0].strip() and depth == depths[0]:
                head = items[:max(1, args.batch_size // 4)]
                a = score_items(model, head, depth=depth, kept=None, **common)
                b = score_items(model, head, depth=depth, kept=depth, **common)
                drift = max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))
                assert drift < 1e-4, f"kept == depth differs from reader_sites_kept=None by {drift}"
            grid[split_name][depth], ce_by_split[split_name][depth] = {}, {}
            records[split_name][depth] = {}
            for kept in parse_sites(args.sites, depth):
                scores, token_lps = score_items(model, items, depth=depth, kept=None if kept >= depth else kept,
                                                return_tokens=True, **common)
                by_hops = tally(items, scores)
                records[split_name][depth][kept] = build_records(items, scores, token_lps)
                grid[split_name][depth][kept] = {}
                ce_by_split[split_name][depth][kept] = {}
                for hops, cell in sorted(by_hops.items()):
                    stats = chains.cell_stats(cell["correct"], cell["chance"])
                    extras = cell_extras([r for r in records[split_name][depth][kept] if r["hops"] == hops])
                    grid[split_name][depth][kept][hops] = dict(stats, **extras)
                    ce_by_split[split_name][depth][kept][hops] = answer_ce(cell)
                    cells.append(dict(stats, split=split_name, depth=depth, hops=hops, kept=kept))
                    logger.info(f"{split_name} D={depth} kept={kept} hops={hops}: acc {stats['acc']:.3f} "
                                f"chance {stats['chance']:.3f} z {stats['z_vs_chance']:+.1f} n {stats['n']}")

    verdict = chains.validity_report(cells)
    print("\n=== chain composition by read sites ===")
    for split_name, by_depth in grid.items():
        for depth, by_kept in by_depth.items():
            print(f"\n{split_name}  depth {depth}   (acc +- sigma | chance | z vs chance)")
            for line in accuracy_table(by_kept):
                print(line)
    print("\n(* marks a cell with fewer kept sites than hops: it must sit at chance)")
    print(f"sub-hop cells outside 3 sigma of chance: {len(verdict['failing'])}")
    for c in verdict["failing"]:
        print(f"  FAIL {c['split']} D={c['depth']} hops={c['hops']} kept={c['kept']}: "
              f"acc {c['acc']:.3f} chance {c['chance']:.3f}")
    if not verdict["readable"]:
        print(f"1-hop curve not readable: {len(verdict['unreadable'])} cells under 3 sigma above chance")
    print(f"instrument valid: {'yes' if verdict['valid'] else 'no'}")

    print("\n=== gold NLL per answer / mean ln K | CE per token | NLL at the divergent token ===")
    for split_name, by_depth in grid.items():
        for depth, by_kept in by_depth.items():
            print(f"\n{split_name}  depth {depth}")
            for line in nll_table(by_kept):
                print(line)

    readings: dict = {"paired_delta": {}}
    for split_name, by_depth in records.items():
        readings["paired_delta"][split_name] = {
            str(d): {str(h): v for h, v in paired_delta_by_hops(by_kept).items()} for d, by_kept in by_depth.items()}
    if "chains_hop4" in ce_by_split:
        print("\nanswer CE by kept sites on the 4-hop split")
        for depth, by_kept in ce_by_split["chains_hop4"].items():
            line = ", ".join(f"j={k}: {v[4]:.3f}" for k, v in by_kept.items() if 4 in v)
            print(f"  depth {depth}: {line}")
        readings["hop4_ce_by_kept"] = {str(d): {str(k): v.get(4) for k, v in bk.items()}
                                       for d, bk in ce_by_split["chains_hop4"].items()}
        full = {d: bk[max(bk)][4] for d, bk in ce_by_split["chains_hop4"].items() if 4 in bk[max(bk)]}
        if len(full) > 1:
            print("  full sites by depth: " + ", ".join(f"D={d}: {v:.3f}" for d, v in sorted(full.items())))
            readings["hop4_ce_full_by_depth"] = {str(d): v for d, v in full.items()}
    if "chains_heldout_tmpl" in grid and 3 in grid["chains_heldout_tmpl"] and "chains_eval" in grid:
        print("\nheld out composition 2-hop at depth 3 (acc, chance)")
        held, seen = grid["chains_heldout_tmpl"][3], grid.get("chains_eval", {}).get(3, {})
        for kept in sorted(held):
            h = held[kept].get(2)
            s = seen.get(kept, {}).get(2)
            if h is not None and kept in (1, 3):
                seen_text = f"{s['acc']:.3f} (chance {s['chance']:.3f})" if s else "n/a"
                print(f"  kept {kept}: held out {h['acc']:.3f} (chance {h['chance']:.3f}) | seen {seen_text}")
        for name, split_name in (("held out", "chains_heldout_tmpl"), ("seen", "chains_eval")):
            pd = readings["paired_delta"].get(split_name, {}).get("3", {}).get("2")
            if pd:
                print(f"  paired Delta acc (kept {pd['kept_hi']} minus kept {pd['kept_lo']}), {name}: "
                      f"{pd['delta']:+.3f} +- {pd['sigma']:.3f} (n {pd['n']})")
        readings["heldout_vs_seen_depth3"] = {
            str(k): {"held_out": held[k].get(2), "seen": seen.get(k, {}).get(2)} for k in sorted(held)}

    if args.json_out:
        payload = {
            "checkpoint": args.checkpoint, "args": vars(args),
            "grid": {s: {str(d): {str(k): {str(h): v for h, v in bh.items()} for k, bh in bk.items()}
                         for d, bk in bd.items()} for s, bd in grid.items()},
            "validity": {"valid": verdict["valid"], "readable": verdict["readable"],
                         "failing": verdict["failing"], "unreadable": verdict["unreadable"]},
            "records": {s: {str(d): {str(k): recs for k, recs in bk.items()} for d, bk in bd.items()}
                        for s, bd in records.items()},
            "readings": readings,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        logger.info(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

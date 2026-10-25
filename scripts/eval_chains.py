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

``--generate-bridge`` reads a model trained on the written bridge form (``prepare_chain_data.py
--bridge``): at each depth and kept-site count it first greedy decodes from the prompt, with the
buffer attached and the same cut, until the text holds the answer marker (or
``--bridge-max-tokens`` runs out), then scores every candidate after that generated prefix, so
the grid and the NLL tables mean the same thing one step later. Each record gains ``generated``
(the decoded text), ``bridge_marker`` (whether the marker came out), ``bridge_hits`` (per
intermediate hop, whether its entity was written as whole words) and ``bridge_all``. A row with
``bridge_marker`` false is still scored, after the whole generated text and a space, but that
prefix is off the trained distribution, so such rows are excluded from accuracy claims; the extra
table gives the marker rate, the hit rate per intermediate hop, the accuracy among marker rows
only, and the accuracy among questions with every intermediate written against the rest. The
candidate tail carries the closing "." token here, so ``gold_nll`` and ``ce_per_token`` are not
comparable to a plain run; ``ln_k`` is. In this mode ``acc`` in the grid and the log counts every
row, marker or not; quote ``acc_marker``. The latent read's sub-hop rule and instrument verdict are
not printed (the JSON ``validity`` carries ``mode: bridge, valid: null``), and the kept 3 minus
kept 1 line is labelled bridge-internal.

Every cell also reads against the shortcut line: ``chains.shortcut_baselines`` over the cell's
questions, rebuilt from the eval sidecar's ``chunk_facts``. ``penultimate_fed`` (about 0.256
against 1 / K 0.185 on 2-hop) is what a picker that never composes scores, so ``z_vs_shortcut``,
not ``z_vs_chance``, is the z a composition claim has to clear. A sidecar without ``chunk_facts``
(built before 2026-10-25) prints n/a.

Usage:

    python scripts/eval_chains.py -c CKPT --data-dir data/prepared \\
        --splits chains_eval,chains_heldout_tmpl,chains_hop4 --depths 3,4 --sites all \\
        --max-questions 1000 --batch-size 16 --json-out ckpts/chains_eval.json \\
        [--generate-bridge --bridge-max-tokens 48]
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


def split_shortcuts(split: ChainSplit, indices: Sequence[int]) -> Dict[int, Optional[Dict[str, float]]]:
    """The shortcut baselines per hop count over the selected questions.

    The cell's questions are the same at every depth and kept count, so this runs once per split.
    A hop count with any question lacking ``chunk_facts`` in the sidecar maps to None.

    Args:
        split: the split read.
        indices: the selected question indices, as ``ChainSplit.select`` returns.
    """
    by_hops: Dict[int, list] = defaultdict(list)
    for i in indices:
        by_hops[split.meta[i]["hops"]].append(chains.question_from_meta(split.meta[i]))
    return {h: (None if any(q is None for q in qs) else chains.shortcut_baselines(qs))
            for h, qs in sorted(by_hops.items())}


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
            "entities": m["entities"], "gold": cands.index(m["answer"]), "prompt": prompt,
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
    applied only at the positions that predict a candidate token. An item carrying ``prefix``
    (generated bridge ids, from ``bridge_items``) places it between the prompt and each candidate,
    and the scored positions move past it (``chains.scored_row``).

    Args:
        items: from ``encode_items`` or ``bridge_items``.
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
        built = [chains.scored_row(items[n]["prompt"], items[n].get("prefix", []), items[n]["cand_ids"][c])
                 for n, c in chunk]
        width = max(len(seq) for seq, _, _ in built) + 1
        assert width <= max_seq_len, f"row of {width} tokens exceeds --max-seq-len {max_seq_len}"
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        doc = torch.zeros((len(chunk), width), dtype=torch.long)
        predict, target, owner = [], [], []
        for r, (seq, positions, targets) in enumerate(built):
            ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            doc[r, :len(seq)] = 1
            predict.extend(r * width + p for p in positions)
            target.extend(targets)
            owner.extend([r] * len(targets))
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


@torch.no_grad()
def generate_bridges(model, tokenizer, items: List[dict], *, depth: int, kept: Optional[int], pad_id: int,
                     eos_id: int, batch_size: int, device: str, max_seq_len: int,
                     max_new_tokens: int) -> List[List[int]]:
    """Greedy decode from each item's prompt with its buffer attached, until the answer marker.

    Batched over questions and left padded, with no KV cache (the reader has no cache slot), the
    way ``eval_abstention.generate_batch`` decodes: the pad run is each row's first segment and
    takes the shared evidence width, the real run its last, and the evidence batch is built once
    per batch of questions (the segment layout does not change as the real runs grow). A row stops at EOS or once its decoded text contains
    ``chains.BRIDGE_ANSWER_MARKER``; a stopped row emits pad inside its own segment so the batch
    stays rectangular.

    Args:
        items: from ``encode_items``.
        depth: ``n_loops`` for every forward.
        kept: ``reader_sites_kept`` for every forward; None keeps every site.
        eos_id: ends a row; never kept in the returned ids.
        batch_size: questions per forward.
        max_seq_len: longest row allowed, prompt plus generated tokens.
        max_new_tokens: decode budget per row.

    Returns:
        Per item, the generated ids, EOS excluded.
    """
    out: List[List[int]] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        b = len(chunk)
        width = max(len(it["prompt"]) for it in chunk) + 1
        assert width + max_new_tokens <= max_seq_len, \
            f"prompt of {width} plus {max_new_tokens} generated tokens exceeds --max-seq-len {max_seq_len}"
        ids = torch.full((b, width), pad_id, dtype=torch.long, device=device)
        doc = torch.zeros((b, width), dtype=torch.long, device=device)
        for r, it in enumerate(chunk):
            p = len(it["prompt"])
            ids[r, width - p:] = torch.tensor(it["prompt"], dtype=torch.long, device=device)
            doc[r, width - p:] = 1
        generated: List[List[int]] = [[] for _ in chunk]
        finished = [False] * b
        rows = [it["evidence"] for it in chunk]
        evidence = None
        for _ in range(max_new_tokens):
            cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(doc)
            if evidence is None:
                # the segment layout (one pad run, one real run per row) never changes while the
                # real runs grow, so the evidence batch of step 0 is every step's
                seg = _segment_ids(cu_seqlens, b, doc.shape[1], device)
                evidence = _pack_evidence_batch(model, seg[:, -1], seg[:, 0], rows,
                                                num_segments=int(cu_seqlens.numel() - 1), device=device)
            res = model(
                input_ids=ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, return_hidden=True,
                skip_mtp=True, evidence=evidence, n_loops=depth, reader_sites_kept=kept,
            )
            hidden = (res[0] if isinstance(res, tuple) else res)[-1]
            nxt = model.lm_head(hidden[:, -1, :]).float().argmax(-1).tolist()
            for r, token in enumerate(nxt):
                # a finished row's last token (EOS or the one completing the marker) is its end;
                # from then on it emits pad inside its own segment, which no other row reads
                if finished[r]:
                    nxt[r] = pad_id
                    continue
                if token == eos_id:
                    finished[r] = True
                    continue
                generated[r].append(token)
                text = tokenizer.decode(generated[r], skip_special_tokens=True)
                if chains.BRIDGE_ANSWER_MARKER in text:
                    finished[r] = True
            if all(finished):
                break
            step = torch.tensor(nxt, dtype=torch.long, device=device).unsqueeze(-1)
            ids = torch.cat([ids, step], dim=-1)
            doc = torch.cat([doc, torch.ones_like(step)], dim=-1)
        out.extend(generated)
    return out


def bridge_items(items: List[dict], generated: List[List[int]], tokenizer, eos_id: int) -> List[dict]:
    """Copies of the items whose candidates are scored after the generated bridge.

    The decoded text is cut after the answer marker (``chains.split_bridge``); without a marker the
    whole text is the prefix and the row is flagged ``bridge_marker`` false: it is scored after an
    off-distribution prefix and is left out of accuracy claims. The prefix and each candidate (with its
    leading space, closing period and EOS) are tokenized as training tokenized the assistant turn
    (``chains.bridge_candidate_ids``), and the prefix ids go under ``prefix`` for ``score_items``.

    Args:
        items: from ``encode_items``.
        generated: from ``generate_bridges``, one id list per item.
        eos_id: closes every candidate.
    """
    out = []
    for item, gen in zip(items, generated):
        text = tokenizer.decode(gen, skip_special_tokens=True)
        prefix_text, rest = chains.split_bridge(text)
        marker = rest is not None
        body = prefix_text[:-1] if marker else text.rstrip()
        prefix, cand_ids = None, []
        for cand in item["candidates"]:
            p, c = chains.bridge_candidate_ids(tokenizer, body, cand, eos_id)
            prefix = p if prefix is None else prefix
            cand_ids.append(c)
        hits = chains.bridge_hits(text, item["entities"])
        out.append(dict(item, prefix=prefix, cand_ids=cand_ids, generated=text, bridge_marker=marker,
                        bridge_hits=hits, bridge_all=all(hits)))
    return out


def bridge_stats(records: List[dict]) -> dict:
    """Marker rate, hit rate per intermediate hop, accuracy among marker rows, and accuracy split by
    whether every hop was written.

    Rows without the marker were scored after an off-distribution prefix; ``acc_marker`` leaves
    them out and is the accuracy to quote.

    Args:
        records: ``build_records`` entries of one cell carrying the bridge fields.
    """
    n = len(records)
    hops = max((len(r["bridge_hits"]) for r in records), default=0)
    marked = [r for r in records if r["bridge_marker"]]
    full = [r for r in records if r["bridge_all"]]
    part = [r for r in records if not r["bridge_all"]]

    def acc(rs):
        return sum(1 for r in rs if r["correct"]) / len(rs) if rs else float("nan")

    return {
        "bridge_marker_rate": sum(1 for r in records if r["bridge_marker"]) / n if n else float("nan"),
        "bridge_hit_rate": [sum(1 for r in records if r["bridge_hits"][h]) / n for h in range(hops)],
        "acc_marker": acc(marked), "n_marker": len(marked),
        "acc_bridge_all": acc(full), "n_bridge_all": len(full),
        "acc_bridge_not_all": acc(part), "n_bridge_not_all": len(part),
    }


def bridge_table(by_kept: Dict[int, Dict[int, dict]]) -> List[str]:
    """One line per kept count and hops: marker rate, hit rate by hop, accuracy among marker rows
    and accuracy by full bridge.

    Args:
        by_kept: grid cells of one split and depth carrying the ``bridge_stats`` keys.
    """
    lines = [f"{'kept':>4} {'hops':>4} {'n':>5} {'marker':>7}  {'hits by hop':<20} "
             f"{'acc | marker (n)':>18} {'acc | all written (n)':>22} {'acc | not (n)':>18}"]
    for kept, by_hops in by_kept.items():
        for h, s in sorted(by_hops.items()):
            if "bridge_marker_rate" not in s:
                continue
            hits = "/".join(f"{v:.2f}" for v in s["bridge_hit_rate"]) or "-"
            lines.append(f"{kept:>4} {h:>4} {s['n']:>5} {s['bridge_marker_rate']:>7.3f}  {hits:<20} "
                         f"{s['acc_marker']:>10.3f} ({s['n_marker']:>5}) "
                         f"{s['acc_bridge_all']:>14.3f} ({s['n_bridge_all']:>5}) "
                         f"{s['acc_bridge_not_all']:>10.3f} ({s['n_bridge_not_all']:>5})")
    return lines


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
SHORTCUT_WIDTH = 17
NLL_WIDTH = 34


def accuracy_table(by_kept: Dict[int, Dict[int, dict]], mark_subhop: bool = True) -> List[str]:
    """Header and one row per kept count; a two character marker column precedes each cell.

    Cells that carry the shortcut line (``shortcut`` and ``z_vs_shortcut``, from ``shortcut_stats``)
    get two more columns, the line and the z against it.

    Args:
        by_kept: grid cells of one split and depth, ``by_kept[kept][hops]``.
        mark_subhop: put ``*`` on cells with fewer kept sites than hops (the latent read's rule; off
            for the written bridge read, which that rule does not describe).
    """
    hops_seen = sorted({h for k in by_kept.values() for h in k})
    with_line = any("shortcut" in s for k in by_kept.values() for s in k.values())
    width = ACC_WIDTH + (SHORTCUT_WIDTH if with_line else 0)
    lines = ["kept  " + "".join(f"{f'hops {h}':>{width + 2}}" for h in hops_seen)]
    for kept, by_hops in by_kept.items():
        line = f"{kept:>4}  "
        for h in hops_seen:
            s = by_hops.get(h)
            if s is None:
                line += " " * (width + 2)
                continue
            mark = "*" if mark_subhop and kept < h else " "
            body = f"{s['acc']:.3f}+-{s['sigma']:.3f} | {s['chance']:.3f} | {s['z_vs_chance']:+6.1f}"
            if with_line:
                if s.get("shortcut") is None:
                    body += " | " + "n/a".rjust(SHORTCUT_WIDTH - 3)
                else:
                    body += f" | {s['shortcut']:.3f} | {s['z_vs_shortcut']:+6.1f}"
            line += f" {mark}{body:>{width}}"
        lines.append(line)
    return lines


def shortcut_stats(correct: Sequence[bool], baselines: Optional[Dict[str, float]]) -> dict:
    """The shortcut line of one cell and the z of its accuracy against it.

    The line is ``penultimate_fed``, the picker that never composes and sits above 1 / K on two or
    more hops (on one hop it equals chance). Cells without baselines (a sidecar written before the
    eval splits carried ``chunk_facts``) get ``shortcut`` None.

    Args:
        correct: per question, whether the prediction was the answer.
        baselines: ``chains.shortcut_baselines`` of the cell's questions, or None.
    """
    if baselines is None:
        return {"shortcut": None, "z_vs_shortcut": float("nan"), "shortcut_baselines": None}
    n = len(correct)
    acc = sum(1 for c in correct if c) / n if n else float("nan")
    line = baselines["penultimate_fed"]
    return {"shortcut": line, "z_vs_shortcut": chains.z_vs_line(acc, line, n), "shortcut_baselines": baselines}


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
    parser.add_argument("--generate-bridge", action="store_true",
                        help="greedy decode the written bridge first, then score the candidates after it")
    parser.add_argument("--bridge-max-tokens", type=int, default=48,
                        help="decode budget per question under --generate-bridge")
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
    shortcut_lines: Dict[str, dict] = {}
    for split_name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split = ChainSplit(args.data_dir, split_name)
        selected = split.select(args.max_questions)
        items = encode_items(split, selected, tokenizer, template)
        logger.info(f"{split_name}: {len(items)} questions")
        shortcuts = split_shortcuts(split, selected)
        shortcut_lines[split_name] = {h: b for h, b in shortcuts.items()}
        for h, b in shortcuts.items():
            if b is None:
                logger.info(f"{split_name}: {h}-hop shortcut baselines n/a (the sidecar has no chunk_facts)")
            else:
                logger.info(f"{split_name}: {h}-hop shortcut baselines " +
                            ", ".join(f"{k} {v:.3f}" for k, v in b.items()))
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
                kept_arg = None if kept >= depth else kept
                cell_items = items
                if args.generate_bridge:
                    generated = generate_bridges(
                        model, tokenizer, items, depth=depth, kept=kept_arg, pad_id=pad_id,
                        eos_id=template.eos_id, batch_size=args.batch_size, device=device,
                        max_seq_len=args.max_seq_len, max_new_tokens=args.bridge_max_tokens)
                    cell_items = bridge_items(items, generated, tokenizer, template.eos_id)
                    logger.info(f"{split_name} D={depth} kept={kept}: generated {len(cell_items)} bridges")
                scores, token_lps = score_items(model, cell_items, depth=depth, kept=kept_arg,
                                                return_tokens=True, **common)
                by_hops = tally(cell_items, scores)
                records[split_name][depth][kept] = build_records(cell_items, scores, token_lps)
                if args.generate_bridge:
                    for rec, it in zip(records[split_name][depth][kept], cell_items):
                        rec.update({k: it[k] for k in ("generated", "bridge_marker", "bridge_hits", "bridge_all")})
                grid[split_name][depth][kept] = {}
                ce_by_split[split_name][depth][kept] = {}
                for hops, cell in sorted(by_hops.items()):
                    stats = chains.cell_stats(cell["correct"], cell["chance"])
                    cell_records = [r for r in records[split_name][depth][kept] if r["hops"] == hops]
                    extras = cell_extras(cell_records)
                    if args.generate_bridge:
                        extras.update(bridge_stats(cell_records))
                    line = shortcut_stats(cell["correct"], shortcut_lines[split_name].get(hops))
                    grid[split_name][depth][kept][hops] = dict(stats, **extras, **line)
                    ce_by_split[split_name][depth][kept][hops] = answer_ce(cell)
                    cells.append(dict(stats, split=split_name, depth=depth, hops=hops, kept=kept))
                    shortcut_text = ("" if line["shortcut"] is None else
                                     f" shortcut {line['shortcut']:.3f} z {line['z_vs_shortcut']:+.1f}")
                    if args.generate_bridge:
                        acc_text = (f"acc_all_rows {stats['acc']:.3f} acc_marker {extras['acc_marker']:.3f} "
                                    f"(n_marker {extras['n_marker']})")
                    else:
                        acc_text = f"acc {stats['acc']:.3f}"
                    logger.info(f"{split_name} D={depth} kept={kept} hops={hops}: {acc_text} "
                                f"chance {stats['chance']:.3f} z {stats['z_vs_chance']:+.1f}{shortcut_text} "
                                f"n {stats['n']}")

    verdict = chains.validity_report(cells)
    print("\n=== chain composition by read sites ===")
    for split_name, by_depth in grid.items():
        for depth, by_kept in by_depth.items():
            if args.generate_bridge:
                head = "acc over all rows, marker or not, +- sigma | chance | z vs chance"
            else:
                head = "acc +- sigma | chance | z vs chance"
            if any("shortcut" in s for bk in by_kept.values() for s in bk.values()):
                head += " | shortcut line (penultimate_fed) | z vs shortcut"
            print(f"\n{split_name}  depth {depth}   ({head})")
            for line in accuracy_table(by_kept, mark_subhop=not args.generate_bridge):
                print(line)
    if args.generate_bridge:
        print("\nbridge mode: acc above counts rows without the answer marker; quote acc_marker from the "
              "written bridge table below.")
        print("the latent read's sub-hop verdict and instrument validity do not describe a written "
              "bridge and are not printed (validity mode bridge, valid null in the JSON).")
    else:
        print("\n(* marks a cell with fewer kept sites than hops: it must sit at chance)")
        print(f"sub-hop cells outside 3 sigma of chance: {len(verdict['failing'])}")
        for c in verdict["failing"]:
            print(f"  FAIL {c['split']} D={c['depth']} hops={c['hops']} kept={c['kept']}: "
                  f"acc {c['acc']:.3f} chance {c['chance']:.3f}")
        if not verdict["readable"]:
            print(f"1-hop curve not readable: {len(verdict['unreadable'])} cells under 3 sigma above chance")
        print(f"instrument valid: {'yes' if verdict['valid'] else 'no'}")
    print("\nshortcut line per split and hop (penultimate_fed; chance is 1/K):")
    for split_name, by_hops in shortcut_lines.items():
        for h, b in by_hops.items():
            print(f"  {split_name} hops {h}: " + ("n/a (no chunk_facts in the sidecar)" if b is None else
                  f"penultimate_fed {b['penultimate_fed']:.3f} against random {b['random']:.3f}"))

    print("\n=== gold NLL per answer / mean ln K | CE per token | NLL at the divergent token ===")
    for split_name, by_depth in grid.items():
        for depth, by_kept in by_depth.items():
            print(f"\n{split_name}  depth {depth}")
            for line in nll_table(by_kept):
                print(line)

    if args.generate_bridge:
        print("\n=== written bridge: marker rate | hit rate by intermediate hop | acc by full bridge ===")
        for split_name, by_depth in grid.items():
            for depth, by_kept in by_depth.items():
                print(f"\n{split_name}  depth {depth}")
                for line in bridge_table(by_kept):
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
                label = "bridge-internal " if args.generate_bridge else ""
                print(f"  {label}paired Delta acc (kept {pd['kept_hi']} minus kept {pd['kept_lo']}), {name}: "
                      f"{pd['delta']:+.3f} +- {pd['sigma']:.3f} (n {pd['n']})")
        readings["heldout_vs_seen_depth3"] = {
            str(k): {"held_out": held[k].get(2), "seen": seen.get(k, {}).get(2)} for k in sorted(held)}

    if args.json_out:
        payload = {
            "checkpoint": args.checkpoint,
            "args": {k: v for k, v in vars(args).items()
                     if args.generate_bridge or k not in ("generate_bridge", "bridge_max_tokens")},
            "grid": {s: {str(d): {str(k): {str(h): v for h, v in bh.items()} for k, bh in bk.items()}
                         for d, bk in bd.items()} for s, bd in grid.items()},
            "validity": ({"mode": "bridge", "valid": None, "readable": None, "failing": [],
                          "unreadable": [], "note": "the latent read's sub-hop rule does not apply to "
                          "the written bridge read"}
                         if args.generate_bridge else
                         {"mode": "latent", "valid": verdict["valid"], "readable": verdict["readable"],
                          "failing": verdict["failing"], "unreadable": verdict["unreadable"]}),
            "shortcut_baselines": {s: {str(h): b for h, b in by_hops.items()}
                                   for s, by_hops in shortcut_lines.items()},
            "records": {s: {str(d): {str(k): recs for k, recs in bk.items()} for d, bk in bd.items()}
                        for s, bd in records.items()},
            "readings": readings,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        logger.info(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

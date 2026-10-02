"""Closed-book likelihood rank: does the model know a fact it was never shown at read time?

An item is a context, a gold continuation and a candidate set that holds the gold exactly once. Every
candidate is scored as ``" " + candidate + terminator`` given ``[BOS] + context`` by summed
log-probability, and the gold is ranked among them: ``norm_rank = (rank - 1) / (n - 1)``, lower is
better, ties count half.

Chance is read against a control, not assumed to be 0.5. Each person's value is drawn uniformly from
its pool, but the corpus frequency of a value is exposure weighted (a person in tier 1000 writes
their value a thousand more times), so a model that learns only the value marginals and never binds
a value to a name ranks the gold of a heavily exposed person above 0.5 and can "climb" with nothing
stored. So every item has a paired prior control: the same candidates and the same context template
with the person's name replaced by a fresh name that occurs nowhere in the corpus
(``biographies.FreshNames``), scored in the same call and carrying ``item_id + "|prior"``. The
reading is ``delta = norm_rank_prior - norm_rank`` per item, positive when the name helps, with its
bootstrap sigma over items paired by id. A tier is "above the prior" when ``delta > 3 sigma`` and
"at the prior" when ``|delta| <= 3 sigma``. The 0.5 line stays in the summary as a reference only.

The library half (``RankItem``, ``rank_items``, ``summarize``, ``paired_compare``) takes a duck typed
backend (``encode_many``, ``score``, ``bos_id``, and ``score_with_evidence`` only when items carry
evidence), so a test can drive it with a scripted scorer and another script can drive it with its own
backend. It imports no model code at module level.

The CLI half builds items from the biography facts and scores a checkpoint through the port:

    TINY_LLM_CONFIG=config_micro.yaml python scripts/closed_book_rank.py bios -c CKPT \\
        --facts data/prepared_inject/inject_facts.jsonl --pools data/prepared_inject/inject_pools.json \\
        --form both --evidence none --json-out ckpts/inject/rank_full.json
    python scripts/closed_book_rank.py compare full.json masked.json retrieval.json

``--evidence gold`` attaches the person's own card: the open-book precondition (a model that cannot
read the card it is handed has an uninformative closed-book reading). ``--evidence swapped`` attaches
the card with that item's attribute replaced by another pool value and records whether the model
follows the card or its memory.

Args:
    (cli) bios: score one checkpoint on the biography items.
    (cli) compare: paired comparison of two result files, or the three arm verdict for three.
"""
import os
import sys
import json
import math
import hashlib
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.data import biographies as bio
from utils import BASE_DIR, TOKENIZER_DIR, logger


PRIOR_SUFFIX = "|prior"


@dataclass
class RankItem:
    """One closed-book item.

    Attributes:
        item_id: stable across runs and checkpoints; pairs two runs.
        context: prompt text, no trailing space.
        gold: one of ``candidates``.
        candidates: the gold exactly once, all distinct after normalization.
        terminator: appended to every candidate, e.g. ``"\\n"`` or ``"."``.
        tier: ``"1"``, ``"10"``, ``"100"``, ``"1000"`` or ``"head"``, ``"mid"``, ``"tail"``.
        group: attribute or relation name.
        evidence: an evidence row ``{"ids", "chunk_ids", "keys"}`` or None; needs a backend with
            ``score_with_evidence``.
    """
    item_id: str
    context: str
    gold: str
    candidates: List[str]
    terminator: str = ""
    tier: str = ""
    group: str = ""
    evidence: Optional[dict] = None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _rank_of(scores: np.ndarray, index: int) -> float:
    """1-based rank of ``scores[index]`` among all scores, higher is better, ties count half."""
    others = np.delete(scores, index)
    return float(1 + (others > scores[index]).sum() + 0.5 * (others == scores[index]).sum())


def rank_items(backend, items: List[RankItem], *, batch_size: int = 32, max_len: int = 1024,
               prepend_bos: bool = True, chunk_items: int = 256, keep_scores: bool = False) -> List[dict]:
    """Score every candidate of every item and rank the gold.

    Args:
        backend: ``encode_many(texts) -> ids``, ``score(batch) -> [(logprob_sum, greedy_exact)]``,
            ``bos_id``; plus ``score_with_evidence(batch, evidence_rows)`` when an item has evidence.
        items: the items; order does not change any result.
        batch_size: sequences per scoring call.
        max_len: a context plus continuation longer than this loses context from the left.
        prepend_bos: start every context with the backend's BOS.
        chunk_items: items tokenized and scored together, which bounds memory.
        keep_scores: also return each item's per candidate scores under ``"scores"``.

    Returns:
        One dict per item, in input order: ``item_id``, ``tier``, ``group``, ``n``, ``rank``
        (1-based, ties count half), ``norm_rank``, ``top1``, ``gold_logprob`` and ``norm_rank_bytes``
        (the same rank on log-probability per utf-8 byte of the candidate).
    """
    results: List[dict] = []
    for lo in range(0, len(items), chunk_items):
        chunk = items[lo:lo + chunk_items]
        for it in chunk:
            assert it.gold in it.candidates and it.candidates.count(it.gold) == 1, it.item_id
            normalized = [_normalize(c) for c in it.candidates]
            assert len(set(normalized)) == len(normalized), f"{it.item_id}: candidates collide"
            assert len(it.candidates) >= 2, f"{it.item_id}: needs at least two candidates"
        contexts = backend.encode_many([it.context for it in chunk])
        texts = [" " + c + it.terminator for it in chunk for c in it.candidates]
        continuations = backend.encode_many(texts)

        sequences, owner = [], []
        cursor = 0
        for i, (it, ctx) in enumerate(zip(chunk, contexts)):
            prefix = ([backend.bos_id] if prepend_bos else []) + list(ctx)
            for j in range(len(it.candidates)):
                cont = list(continuations[cursor])
                cursor += 1
                if len(prefix) + len(cont) > max_len:
                    keep = max(1, max_len - len(cont))
                    prefix_now = prefix[-keep:]
                    if prepend_bos and prefix_now[0] != backend.bos_id:
                        prefix_now = [backend.bos_id] + prefix_now[1:]
                else:
                    prefix_now = prefix
                sequences.append((prefix_now, cont))
                owner.append((i, j))

        flat = np.full(len(sequences), np.nan, dtype=np.float64)
        plain = [k for k, (i, _) in enumerate(owner) if chunk[i].evidence is None]
        with_evidence = [k for k, (i, _) in enumerate(owner) if chunk[i].evidence is not None]
        if with_evidence and not hasattr(backend, "score_with_evidence"):
            raise ValueError("an item carries evidence but the backend has no score_with_evidence")
        for group, uses_evidence in ((plain, False), (with_evidence, True)):
            group = sorted(group, key=lambda k: len(sequences[k][0]) + len(sequences[k][1]))
            for start in range(0, len(group), batch_size):
                ks = group[start:start + batch_size]
                batch = [sequences[k] for k in ks]
                if uses_evidence:
                    scored = backend.score_with_evidence(batch, [chunk[owner[k][0]].evidence for k in ks])
                else:
                    scored = backend.score(batch)
                for k, (logprob, _) in zip(ks, scored):
                    flat[k] = logprob
        assert not np.isnan(flat).any(), "a sequence was never scored"

        cursor = 0
        for it in chunk:
            n = len(it.candidates)
            scores = flat[cursor:cursor + n]
            cursor += n
            gold_at = it.candidates.index(it.gold)
            n_bytes = np.array([max(1, len(c.encode("utf-8"))) for c in it.candidates], dtype=np.float64)
            rank = _rank_of(scores, gold_at)
            rank_bytes = _rank_of(scores / n_bytes, gold_at)
            result = {
                "item_id": it.item_id, "tier": it.tier, "group": it.group, "n": n, "rank": rank,
                "norm_rank": (rank - 1) / (n - 1), "top1": bool(rank == 1.0),
                "gold_logprob": float(scores[gold_at]),
                "norm_rank_bytes": (rank_bytes - 1) / (n - 1),
            }
            if keep_scores:
                result["scores"] = [float(s) for s in scores]
            results.append(result)
    return results


def bootstrap_sigma(values: Sequence[float], n_boot: int = 1000, seed: int = 0) -> float:
    """Standard deviation of the mean of ``values`` over bootstrap resamples of the items.

    Args:
        values: one number per item.
        n_boot: resamples.
        seed: the resampling seed.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    means = []
    block = max(1, min(n_boot, 2_000_000 // n))
    for start in range(0, n_boot, block):
        count = min(block, n_boot - start)
        means.append(values[rng.integers(0, n, size=(count, n))].mean(axis=1))
    return float(np.concatenate(means).std(ddof=1))


def _group_key(result: dict, by: Sequence[str]) -> str:
    return "|".join(str(result[k]) for k in by)


def _z(numerator: float, sigma: float) -> float:
    return float(numerator / sigma) if sigma > 0 else 0.0


def summarize(results: List[dict], *, by: Sequence[str] = ("tier",), n_boot: int = 1000,
              seed: int = 0) -> dict:
    """Mean ``norm_rank`` and ``top1`` with bootstrap sigmas, per group and overall.

    Args:
        results: the output of ``rank_items`` (extra keys such as ``cls`` can be grouped by).
        by: result keys that define a group; the key of a group is their values joined by ``|``.
        n_boot: resamples for each sigma.
        seed: the resampling seed.

    Returns:
        ``{group_key: {"n", "norm_rank", "norm_rank_sigma", "norm_rank_bytes", "top1", "top1_sigma",
        "chance_norm_rank", "chance_top1", "z_vs_chance"}}`` plus an ``"all"`` entry. ``z_vs_chance``
        is ``(0.5 - norm_rank) / sigma``, positive when better than 0.5. That line is a reference
        only; the reading that counts is ``prior_deltas``. Control items (ids ending in
        ``PRIOR_SUFFIX``) must be left out of ``results`` by the caller.
    """
    ordered = sorted(results, key=lambda r: r["item_id"])
    groups: Dict[str, List[dict]] = {}
    for r in ordered:
        groups.setdefault(_group_key(r, by), []).append(r)
    groups["all"] = ordered
    out = {}
    for key, rows in groups.items():
        nr = np.array([r["norm_rank"] for r in rows])
        t1 = np.array([float(r["top1"]) for r in rows])
        sigma = bootstrap_sigma(nr, n_boot, seed)
        out[key] = {
            "n": len(rows), "norm_rank": float(nr.mean()), "norm_rank_sigma": sigma,
            "norm_rank_bytes": float(np.mean([r["norm_rank_bytes"] for r in rows])),
            "top1": float(t1.mean()), "top1_sigma": bootstrap_sigma(t1, n_boot, seed),
            "chance_norm_rank": 0.5, "chance_top1": float(np.mean([1.0 / r["n"] for r in rows])),
            "z_vs_chance": _z(0.5 - float(nr.mean()), sigma),
        }
    return out


def is_prior(result: dict) -> bool:
    """True for a prior control result (its id ends in ``PRIOR_SUFFIX``).

    Args:
        result: one ``rank_items`` result.
    """
    return result["item_id"].endswith(PRIOR_SUFFIX)


def real_only(results: List[dict]) -> List[dict]:
    """The results without the prior control items.

    Args:
        results: ``rank_items`` output.
    """
    return [r for r in results if not is_prior(r)]


def prior_deltas(results: List[dict], *, by: Sequence[str] = ("tier",), n_boot: int = 1000,
                 seed: int = 0) -> dict:
    """Real against fresh-name prior, paired by item: ``delta = norm_rank_prior - norm_rank``.

    Args:
        results: ``rank_items`` output holding real items and their ``|prior`` controls.
        by: result keys that define a group.
        n_boot: resamples for each sigma.
        seed: the resampling seed.

    Returns:
        ``{group_key: {"n", "norm_rank", "norm_rank_prior", "delta", "delta_sigma", "z_delta"}}`` plus
        an ``"all"`` entry; empty when no item has a control. ``delta`` is positive when the name
        helps and ``delta_sigma`` is the bootstrap sigma of the mean paired difference (a nonzero
        delta with zero sigma reads as z of plus or minus 1e6).
    """
    by_id = {r["item_id"]: r for r in results}
    groups: Dict[str, List[Tuple[float, float]]] = {"all": []}
    for item_id in sorted(by_id):
        control = by_id.get(item_id + PRIOR_SUFFIX)
        if control is None or is_prior(by_id[item_id]):
            continue
        pair = (by_id[item_id]["norm_rank"], control["norm_rank"])
        groups.setdefault(_group_key(by_id[item_id], by), []).append(pair)
        groups["all"].append(pair)
    if not groups["all"]:
        return {}
    out = {}
    for key, pairs in groups.items():
        real = np.array([p[0] for p in pairs])
        prior = np.array([p[1] for p in pairs])
        sigma = bootstrap_sigma(prior - real, n_boot, seed)
        delta = float((prior - real).mean())
        out[key] = {"n": len(pairs), "norm_rank": float(real.mean()), "norm_rank_prior": float(prior.mean()),
                    "delta": delta, "delta_sigma": sigma,
                    "z_delta": _z(delta, sigma) if sigma > 0 or delta == 0 else math.copysign(1e6, delta)}
    return out


def paired_compare(results_a: List[dict], results_b: List[dict], *, by: Sequence[str] = ("tier",),
                   n_boot: int = 1000, seed: int = 0) -> dict:
    """Difference of mean ``norm_rank`` (a minus b) over the items both runs scored.

    Items are resampled jointly, so a hard item is hard for both runs and the sigma is the sigma of
    the paired difference. A negative difference means ``a`` ranks the gold better.

    Args:
        results_a: one run's ``rank_items`` output.
        results_b: another run's, over the same item ids (extras on either side are ignored).
        by: result keys that define a group.
        n_boot: resamples.
        seed: the resampling seed.

    Returns:
        ``{group_key: {"n", "diff_norm_rank", "sigma", "z"}}`` plus an ``"all"`` entry.
    """
    b_by_id = {r["item_id"]: r for r in results_b}
    ids = sorted(r["item_id"] for r in results_a if r["item_id"] in b_by_id)
    a_by_id = {r["item_id"]: r for r in results_a}
    groups: Dict[str, List[float]] = {"all": []}
    for item_id in ids:
        diff = a_by_id[item_id]["norm_rank"] - b_by_id[item_id]["norm_rank"]
        groups.setdefault(_group_key(a_by_id[item_id], by), []).append(diff)
        groups["all"].append(diff)
    out = {}
    for key, diffs in groups.items():
        mean = float(np.mean(diffs)) if diffs else 0.0
        sigma = bootstrap_sigma(diffs, n_boot, seed)
        out[key] = {"n": len(diffs), "diff_norm_rank": mean, "sigma": sigma, "z": _z(mean, sigma)}
    return out


# --------------------------------------------------------------------------------------- backend

class TinyEvidenceBackend:
    """This repo's model behind the rank library, with an optional evidence row per sequence.

    Rows are right padded plus one extra pad column, so every row has a trailing pad segment of its
    own to take the extra width of the rectangular evidence tensor (see ``_pack_evidence_batch``).
    The final loop's hidden state is gathered at the positions that predict the continuation before
    the LM head, never materializing ``[B, S, vocab]``.

    Args:
        model: a model from ``scripts.eval_abstention.load_model``.
        tokenizer: the pruned tokenizer.
        device: the model's device.
        max_seq_len: the model's context length.
    """

    def __init__(self, model, tokenizer, device: str, max_seq_len: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        self.pad_id = tokenizer.pad_token_id

    def encode_many(self, texts: Sequence[str]) -> List[List[int]]:
        """Token ids of each text, no special tokens.

        Args:
            texts: the strings to tokenize.
        """
        return self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]

    def score(self, batch):
        """Summed log-probability and greedy match of each continuation, no evidence.

        Args:
            batch: ``(context_ids, continuation_ids)`` pairs.
        """
        return self.score_with_evidence(batch, [None] * len(batch))

    def score_with_evidence(self, batch, evidence_rows):
        import torch
        from scripts.eval_abstention import _pack_evidence_batch
        from modules.model.attention import _segment_ids, cu_seqlens_from_doc_ids

        with torch.inference_mode():
            lengths = np.array([len(c) + len(k) for c, k in batch], dtype=np.int64)
            context_lengths = np.array([len(c) for c, _ in batch], dtype=np.int64)
            continuation_lengths = np.array([len(k) for _, k in batch], dtype=np.int64)
            width = int(lengths.max()) + 1
            assert width <= self.max_seq_len, f"row of {width} tokens (with its pad column) exceeds the context"
            packed = np.full((len(batch), width), self.pad_id, dtype=np.int64)
            for row, (context, continuation) in enumerate(batch):
                packed[row, :lengths[row]] = list(context) + list(continuation)
            base = np.repeat(np.arange(len(batch), dtype=np.int64) * width + context_lengths,
                             continuation_lengths)
            within = np.concatenate([np.arange(n, dtype=np.int64) for n in continuation_lengths])

            ids = torch.from_numpy(packed).to(self.device)
            real = (torch.arange(width, device=self.device)[None, :]
                    < torch.from_numpy(lengths).to(self.device)[:, None]).long()
            predict_at = torch.from_numpy(base + within - 1).to(self.device)
            target_at = torch.from_numpy(base + within).to(self.device)

            cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(real)
            evidence = None
            if any(row for row in evidence_rows):
                seg = _segment_ids(cu_seqlens, len(batch), width, self.device)
                evidence = _pack_evidence_batch(
                    self.model, seg[:, 0], seg[:, -1], list(evidence_rows),
                    num_segments=int(cu_seqlens.numel() - 1), device=self.device,
                )
            out = self.model(input_ids=ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                             return_hidden=True, skip_mtp=True, evidence=evidence)
            hidden = (out[0] if isinstance(out, tuple) else out)[-1]
            gathered = hidden.reshape(-1, hidden.size(-1)).index_select(0, predict_at)
            logprobs = self.model.lm_head(gathered).float().log_softmax(-1)
            targets = ids.reshape(-1).index_select(0, target_at)
            token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            greedy = logprobs.argmax(-1) == targets
            scored, at = [], 0
            token_logprobs, greedy = token_logprobs.cpu(), greedy.cpu()
            for n in continuation_lengths:
                span = slice(at, at + int(n))
                scored.append((float(token_logprobs[span].sum()), bool(greedy[span].all())))
                at += int(n)
            return scored


# ------------------------------------------------------------------------------- biography items

def _seeded(text: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big"))


def select_people(people: List[bio.Person], max_per_tier: int) -> List[bio.Person]:
    """Up to ``max_per_tier`` people of each tier, a fixed sample, in person id order.

    Args:
        people: every person of the facts file.
        max_per_tier: the most people kept from one tier.
    """
    chosen = []
    for tier in sorted({p.tier for p in people}):
        group = [p for p in people if p.tier == tier]
        if len(group) > max_per_tier:
            group = random.Random(0).sample(group, max_per_tier)
        chosen.extend(group)
    return sorted(chosen, key=lambda p: p.person_id)


def make_bio_items(people: List[bio.Person], pools: Dict[str, List[str]], *, forms: Sequence[str],
                   n_candidates: int = 100, evidence: str = "none", card_evidence=None,
                   fresh_names: Optional[bio.FreshNames] = None
                   ) -> Tuple[List[RankItem], Dict[str, dict]]:
    """Closed-book items for the given people, one per person, attribute and probe form.

    The candidates are the gold plus values drawn without replacement from the same pool by a
    generator seeded from the item id (sha1, never the builtin hash), so every run and every
    checkpoint scores the same candidates. ``major`` has 100 values, so at 100 candidates its set is
    the whole pool.

    Args:
        people: whom to probe.
        pools: ``{attribute: [values]}``.
        forms: any of ``"indist"``, ``"heldout"``.
        n_candidates: candidates per item, gold included.
        evidence: ``"none"``, ``"gold"`` (the person's card) or ``"swapped"`` (the card with the
            probed attribute replaced by a substitute, which is made one of the candidates).
        card_evidence: ``fn(person, text) -> evidence row``; required unless ``evidence`` is
            ``"none"``.
        fresh_names: when given, every item gets a paired prior control: the same candidates, tier,
            group and context template with the name replaced by ``fresh_names.name(item_id)``, id
            ``item_id + PRIOR_SUFFIX``, placed right after its item. Only for ``evidence="none"``.

    Returns:
        ``(items, swaps)`` where ``swaps`` maps an item id to ``{"substitute"}`` in swapped mode.
    """
    assert evidence in ("none", "gold", "swapped"), evidence
    if evidence != "none":
        assert card_evidence is not None, "evidence modes need card_evidence"
        assert fresh_names is None, "the prior control is a closed-book reading, evidence must be none"
    items, swaps = [], {}
    for person in people:
        for attribute in bio.ATTRIBUTES:
            gold = person.attributes[attribute]
            others = [v for v in pools[attribute] if v != gold]
            for form in forms:
                item_id = f"{person.person_id}:{attribute}:{form}"
                rng = _seeded(item_id)
                context, terminator = bio.probe_prompt(attribute, person.name, form)
                row, picks = None, None
                if evidence == "swapped":
                    substitute = _seeded(item_id + ":swap").choice(others)
                    rest = [v for v in others if v != substitute]
                    picks = [substitute] + rng.sample(rest, min(n_candidates - 2, len(rest)))
                    card = bio.render_store_chunk(person, {attribute: substitute})
                    row = card_evidence(person, card)
                    swaps[item_id] = {"substitute": substitute}
                else:
                    picks = rng.sample(others, min(n_candidates - 1, len(others)))
                    if evidence == "gold":
                        row = card_evidence(person, person.store_chunk)
                candidates = picks + [gold]
                rng.shuffle(candidates)
                items.append(RankItem(item_id, context, gold, candidates, terminator,
                                      str(person.tier), attribute, row))
                if fresh_names is not None:
                    fresh = fresh_names.name(item_id)
                    control_context, _ = bio.probe_prompt(attribute, fresh, form)
                    items.append(RankItem(item_id + PRIOR_SUFFIX, control_context, gold, list(candidates),
                                          terminator, str(person.tier), attribute, None))
    return items, swaps


def swap_readings(items: List[RankItem], results: List[dict], swaps: Dict[str, dict]) -> None:
    """Add the counterfactual readings to swapped-evidence results, in place, and drop the scores.

    ``ll_orig`` and ``ll_swap`` are the log-probabilities of the original value and the substitute,
    ``swap_norm_rank`` the substitute's normalized rank, ``follow`` whether the substitute ranked
    first (the model took the card over its memory) and ``mr_ll = sigmoid(ll_orig - ll_swap)``,
    near 1 when the model prefers its memory and near 0 when it prefers the card.

    Args:
        items: the items the results came from.
        results: ``rank_items(..., keep_scores=True)`` output, in the same order.
        swaps: the second return of ``make_bio_items``.
    """
    for item, result in zip(items, results):
        scores = np.asarray(result.pop("scores"), dtype=np.float64)
        substitute = swaps[item.item_id]["substitute"]
        gold_at, sub_at = item.candidates.index(item.gold), item.candidates.index(substitute)
        rank = _rank_of(scores, sub_at)
        delta = float(scores[gold_at] - scores[sub_at])
        result.update({
            "swap_norm_rank": (rank - 1) / (len(scores) - 1), "follow": bool(rank == 1.0),
            "ll_orig": float(scores[gold_at]), "ll_swap": float(scores[sub_at]),
            "mr_ll": 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, delta)))),
        })


def annotate(results: List[dict]) -> None:
    """Add ``cls`` (entity, date, noun), ``form`` and ``person_id`` parsed from the item id.

    Args:
        results: ``rank_items`` output, edited in place; prior control ids parse like their item.
    """
    for r in results:
        person_id, attribute, form = r["item_id"].removesuffix(PRIOR_SUFFIX).split(":")
        r["cls"], r["form"], r["person_id"] = bio.attribute_class(attribute), form, int(person_id)


# ------------------------------------------------------------------------------------- reporting

TIER_ORDER = ("1", "10", "100", "1000")


def _tier_rows(summary: dict) -> List[Tuple[str, dict]]:
    return [(k, summary[k]) for k in TIER_ORDER if k in summary]


def print_summary(title: str, results: List[dict], n_boot: int = 1000) -> dict:
    """Print norm_rank and top-1 per tier for each attribute class, the fresh-name prior control and
    the swap readings when present.

    Args:
        title: heading of the block.
        results: annotated ``rank_items`` output, prior control items included when scored.
        n_boot: resamples for each sigma.

    Returns:
        ``{class: summarize output}`` per class, plus ``"prior"`` holding ``prior_deltas`` by tier and by
        tier and group (``norm_rank``, ``norm_rank_prior``, ``delta``, ``delta_sigma``, ``z_delta``)
        when the results carry controls. The 0.5 columns are a reference only.
    """
    out = {}
    print(f"\n=== {title} ===")
    print(f"{'class':<8} {'tier':>5} {'n':>6} {'norm_rank':>10} {'sigma':>7} {'ref z0.5':>9} "
          f"{'top1':>7} {'chance':>7} {'prior':>7} {'delta':>7} {'d sigma':>8} {'d z':>6}")
    has_prior = any(is_prior(r) for r in results)
    prior_tier, prior_group = {}, {}
    for cls in ("entity", "date", "noun"):
        class_rows = [r for r in results if r["cls"] == cls]
        rows = real_only(class_rows)
        if not rows:
            continue
        summary = summarize(rows, by=("tier",), n_boot=n_boot)
        out[cls] = summary
        deltas = prior_deltas(class_rows, by=("tier",), n_boot=n_boot) if has_prior else {}
        prior_tier[cls] = deltas
        if has_prior:
            prior_group[cls] = prior_deltas(class_rows, by=("tier", "group"), n_boot=n_boot)
        for tier, s in _tier_rows(summary) + [("all", summary["all"])]:
            d = deltas.get(tier)
            tail = (f" {d['norm_rank_prior']:>7.4f} {d['delta']:>7.4f} {d['delta_sigma']:>8.4f} "
                    f"{d['z_delta']:>6.1f}") if d else ""
            print(f"{cls:<8} {tier:>5} {s['n']:>6} {s['norm_rank']:>10.4f} {s['norm_rank_sigma']:>7.4f} "
                  f"{s['z_vs_chance']:>9.1f} {s['top1']:>7.3f} {s['chance_top1']:>7.3f}{tail}")
    if has_prior:
        out["prior"] = {"by_tier": prior_tier, "by_tier_group": prior_group}
        print("prior control, per tier and attribute (delta = prior - real, positive means the name helps)")
        print(f"{'group':<12} {'tier':>5} {'n':>6} {'norm_rank':>10} {'prior':>7} {'delta':>7} {'d sigma':>8} {'d z':>6}")
        for cls, table in prior_group.items():
            for key in sorted(k for k in table if k != "all"):
                tier, group = key.split("|")
                d = table[key]
                print(f"{group:<12} {tier:>5} {d['n']:>6} {d['norm_rank']:>10.4f} {d['norm_rank_prior']:>7.4f} "
                      f"{d['delta']:>7.4f} {d['delta_sigma']:>8.4f} {d['z_delta']:>6.1f}")
    followed = [r for r in real_only(results) if "follow" in r]
    if followed:
        print(f"{'tier':>5} {'n':>6} {'follow':>8} {'mean mr_ll':>11}")
        for tier in TIER_ORDER:
            rows = [r for r in followed if r["tier"] == tier]
            if rows:
                print(f"{tier:>5} {len(rows):>6} {np.mean([r['follow'] for r in rows]):>8.3f} "
                      f"{np.mean([r['mr_ll'] for r in rows]):>11.3f}")
    return out


def micro_verdict(arms: Dict[str, List[dict]], n_boot: int = 1000) -> Tuple[str, dict]:
    """Read the three arm results of one probe form for whether a store can carry the facts.

    Chance is the fresh-name prior control, not 0.5. Passes when the arms trained on masked spans and
    on retrieval have a paired delta (prior minus real) within 3 sigma of 0 at tiers 1, 10 and 100,
    two-sided, while the arm trained on full cross entropy has a delta more than 3 sigma above 0 at
    tier 100 or 1000. All read on entity attributes. Results of more than one probe form must not be
    pooled; use ``micro_verdicts``.

    Args:
        arms: ``{"full", "masked", "retrieval"}`` to ``rank_items`` output annotated by ``annotate``,
            real items and their prior controls, one form.
        n_boot: resamples for each sigma.

    Returns:
        ``(verdict line, details)``; ``details`` holds ``full_climbs``, ``leaks`` and the per tier deltas.
    """
    tiers = {}
    for name, results in arms.items():
        entity = [r for r in results if r["cls"] == "entity"]
        tiers[name] = prior_deltas(entity, by=("tier",), n_boot=n_boot)
        if not tiers[name]:
            raise ValueError(f"arm {name!r} has no prior control items; score with the fresh-name control")
    climbs = any(tiers["full"].get(t, {}).get("z_delta", 0.0) > 3.0 for t in ("100", "1000"))
    leaks = {}
    for name in ("masked", "retrieval"):
        leaks[name] = [t for t in ("1", "10", "100") if abs(tiers[name].get(t, {}).get("z_delta", 0.0)) > 3.0]
    details = {"full_climbs": climbs, "leaks": leaks, "deltas": tiers}
    if not climbs:
        return ("UNINFORMATIVE: the full cross entropy arm is not 3 sigma above its fresh-name prior at "
                "tier 100 or 1000, so there is nothing for the other arms to be compared against"), details
    if not any(leaks.values()):
        return ("HOLDS: masked and retrieval are within 3 sigma of their fresh-name prior through tier 100 "
                "while full climbs"), details
    named = ", ".join(f"{n} off its prior at tiers {t}" for n, t in leaks.items() if t)
    return f"FAILS: {named} while full climbs", details


def micro_verdicts(arms: Dict[str, List[dict]], n_boot: int = 1000) -> Dict[str, Tuple[str, dict]]:
    """``micro_verdict`` once per probe form present, never pooling forms.

    Pooling the in-distribution and held-out items of one person and attribute would count the same
    fact twice and understate sigma.

    Args:
        arms: as for ``micro_verdict``, results of any forms.
        n_boot: resamples for each sigma.

    Returns:
        ``{form: (verdict line, details)}`` in form order.
    """
    forms = sorted({r["form"] for results in arms.values() for r in results})
    return {form: micro_verdict({name: [r for r in results if r["form"] == form]
                                 for name, results in arms.items()}, n_boot)
            for form in forms}


# ------------------------------------------------------------------------------------------ main

def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def run_bios(args) -> None:
    import torch
    from transformers import AutoTokenizer
    from scripts.eval_abstention import load_model

    people = bio.load_facts(_resolve(args.facts))
    pools = bio.load_pools(_resolve(args.pools))
    chosen = select_people(people, args.max_persons_per_tier)
    forms = ["indist", "heldout"] if args.form == "both" else [args.form]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    card_evidence = None
    if args.evidence != "none":
        from modules.data.store import load_store
        store = load_store(_resolve(args.store))
        assert len(store.chunks) == len(people), "the store and the facts disagree on the people"

        def card_evidence(person, text):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            return {"ids": ids, "chunk_ids": [0] * len(ids),
                    "keys": np.asarray(store.keys[person.person_id:person.person_id + 1], dtype=np.float32)}

    fresh_names = bio.FreshNames(people, pools) if args.evidence == "none" else None
    items, swaps = make_bio_items(chosen, pools, forms=forms, n_candidates=args.candidates,
                                  evidence=args.evidence, card_evidence=card_evidence,
                                  fresh_names=fresh_names)
    logger.info(f"{len(items):,} items (prior controls included) from {len(chosen):,} people, forms {forms}, "
                f"evidence {args.evidence}, {args.candidates} candidates")

    model = load_model(_resolve(args.checkpoint), args.device)
    from config import ModelConfig
    backend = TinyEvidenceBackend(model, tokenizer, args.device, ModelConfig.Params["max_seq_len"])
    results = rank_items(backend, items, batch_size=args.batch_size, max_len=ModelConfig.Params["max_seq_len"] - 1,
                         keep_scores=args.evidence == "swapped")
    if args.evidence == "swapped":
        swap_readings(items, results, swaps)
    annotate(results)

    summaries = {}
    for form in forms:
        subset = [r for r in results if r["form"] == form]
        summaries[form] = print_summary(f"{os.path.basename(args.checkpoint)} {form} "
                                        f"evidence={args.evidence}", subset, args.n_boot)
    if args.json_out:
        path = _resolve(args.json_out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"flags": vars(args), "results": results, "summary": summaries}, f)
        logger.info(f"wrote {path}")


def run_compare(args) -> None:
    runs = []
    for path in args.files:
        with open(_resolve(path)) as f:
            runs.append(json.load(f))
    by = tuple(args.by.split(","))
    names = ("full", "masked", "retrieval") if len(runs) == 3 else ("a", "b")
    for name, run in zip(names, runs):
        print_summary(f"{name}: {run['flags'].get('checkpoint', '?')}", run["results"], args.n_boot)
    print("\n=== paired differences (norm_rank, first minus second; negative favours the first) ===")
    pairs = [(0, 1)] if len(runs) == 2 else [(0, 1), (0, 2)]
    for i, j in pairs:
        print(f"{names[i]} minus {names[j]}")
        cmp = paired_compare(real_only(runs[i]["results"]), real_only(runs[j]["results"]), by=by,
                             n_boot=args.n_boot)
        for key, s in sorted(cmp.items()):
            print(f"  {key:<24} n {s['n']:>6}  diff {s['diff_norm_rank']:>8.4f}  "
                  f"sigma {s['sigma']:.4f}  z {s['z']:>6.1f}")
    if len(runs) == 3:
        for form, (line, _) in micro_verdicts(dict(zip(names, (r["results"] for r in runs))),
                                              args.n_boot).items():
            print(f"\nverdict ({form} form): {line}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bios", help="score a checkpoint on the biography items")
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--facts", default="data/prepared_inject/inject_facts.jsonl")
    p.add_argument("--pools", default="data/prepared_inject/inject_pools.json")
    p.add_argument("--store", default="data/index/inject_bios",
                   help="the biography store, read for the card keys when evidence is attached")
    p.add_argument("--form", choices=("indist", "heldout", "both"), default="both")
    p.add_argument("--evidence", choices=("none", "gold", "swapped"), default="none")
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--max-persons-per-tier", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer", default=TOKENIZER_DIR)
    p.add_argument("--json-out", default=None)

    c = sub.add_parser("compare", help="paired comparison of result files, or the three arm verdict")
    c.add_argument("files", nargs="+", help="two result files, or three in the order full masked "
                                            "retrieval for the verdict")
    c.add_argument("--by", default="tier", help="comma separated result keys, e.g. tier,group")
    c.add_argument("--n-boot", type=int, default=1000)

    args = parser.parse_args()
    if args.command == "bios":
        run_bios(args)
    else:
        if len(args.files) not in (2, 3):
            raise SystemExit("compare takes two or three result files")
        run_compare(args)


if __name__ == "__main__":
    main()

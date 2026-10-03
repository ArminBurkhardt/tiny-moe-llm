"""The closed-book rank library, driven by a scripted backend (no model, no tokenizer).

1. rank arithmetic: ties count half, ``norm_rank`` is ``(rank - 1) / (n - 1)`` inside [0, 1], the
   gold at the top ranks 0, at the bottom 1, all ties 0.5; ``top1`` only for a clean first place;
2. a random scorer over 5,000 items lands within 3 sigma of 0.5 and its top-1 near 1 / n, and a
   perfect scorer gives ``norm_rank`` 0 and top-1 1;
3. ``bootstrap_sigma`` is close to the analytic standard error of a mean;
4. ``paired_compare`` on identical inputs gives a zero difference, on shifted inputs the shift;
5. item order changes no result and no summary;
6. evidence rows reach the backend aligned with their sequences, an item with evidence on a backend
   without ``score_with_evidence`` fails loudly, and over-long contexts are cut from the left with
   the BOS kept;
7. ``make_bio_items``: gold once among 100 distinct candidates drawn by sha1 seeded generators (the
   same on every call), ``major`` takes its whole pool, the swapped mode makes the substitute a
   candidate and ``swap_readings`` reads follow and ``mr_ll`` off the scores;
8. the three arm verdict, read against the paired fresh-name prior (two sided for the arms that must
   sit at the prior, one verdict per probe form): HOLDS, FAILS and UNINFORMATIVE on constructed results;
9. the prior control: a scorer that knows only value frequencies ranks heavy tiers well above 0.5 yet
   has delta 0, a scorer that binds names to values has a large delta;
10. ``compare`` on three files prints the verdict.

GPU free; imports no model code.
"""
import os, sys, math, random, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modules.data import biographies as bio
from scripts.closed_book_rank import (
    RankItem, annotate, bootstrap_sigma, make_bio_items, micro_verdict, micro_verdicts, paired_compare,
    prior_deltas, rank_items, real_only, summarize, swap_readings,
)


def encode(text):
    return [ord(c) for c in text]


def decode(ids):
    return "".join(chr(i) for i in ids)


class FakeBackend:
    """Characters are token ids. ``scorer(context_text, continuation_text, evidence) -> float``."""
    bos_id = 1

    def __init__(self, scorer):
        self.scorer = scorer
        self.seen = []

    def encode_many(self, texts):
        return [encode(t) for t in texts]

    def _score(self, batch, rows):
        out = []
        for (ctx, cont), row in zip(batch, rows):
            assert ctx[0] == self.bos_id
            self.seen.append((len(ctx), len(cont), row))
            out.append((float(self.scorer(decode(ctx[1:]), decode(cont), row)), False))
        return out

    def score(self, batch):
        return self._score(batch, [None] * len(batch))

    def score_with_evidence(self, batch, evidence_rows):
        return self._score(batch, evidence_rows)


class PlainBackend(FakeBackend):
    """A backend that cannot read evidence: the method does not exist at all."""

    def __getattribute__(self, name):
        if name == "score_with_evidence":
            raise AttributeError(name)
        return super().__getattribute__(name)


def hashed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) / 2 ** 32


def item(item_id, candidates, gold, **kw):
    return RankItem(item_id, "ctx " + item_id, gold, candidates, **kw)


def main():
    # 1. rank arithmetic
    def table(scores):
        return lambda ctx, cont, row: scores[cont.strip()]

    cands = ["a", "b", "c", "d"]
    cases = {
        "tie": ({"a": 1.0, "b": 1.0, "c": 2.0, "d": 0.0}, 2.5, 0.5, False),
        "best": ({"a": 3.0, "b": 1.0, "c": 2.0, "d": 0.0}, 1.0, 0.0, True),
        "worst": ({"a": -1.0, "b": 1.0, "c": 2.0, "d": 0.0}, 4.0, 1.0, False),
        "alltied": ({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}, 2.5, 0.5, False),
        "tiefirst": ({"a": 2.0, "b": 2.0, "c": 0.0, "d": 0.0}, 1.5, 1 / 6, False),
    }
    for name, (scores, rank, norm, top1) in cases.items():
        (res,) = rank_items(FakeBackend(table(scores)), [item(name, cands, "a", tier="10", group="g")])
        assert res["rank"] == rank and abs(res["norm_rank"] - norm) < 1e-12 and res["top1"] == top1, (name, res)
        assert res["n"] == 4 and res["tier"] == "10" and res["group"] == "g"
        assert res["gold_logprob"] == scores["a"] and 0.0 <= res["norm_rank_bytes"] <= 1.0
    # per byte ranking can differ from the summed ranking: a long candidate loses on bytes only
    long_cands = ["aa", "bbbbbbbb"]
    scores = {"aa": -2.0, "bbbbbbbb": -4.0}
    (res,) = rank_items(FakeBackend(table(scores)), [item("bytes", long_cands, "aa")])
    assert res["norm_rank"] == 0.0 and res["norm_rank_bytes"] == 1.0, res
    (res,) = rank_items(FakeBackend(table(scores)), [item("bytes2", long_cands, "bbbbbbbb")])
    assert res["norm_rank"] == 1.0 and res["norm_rank_bytes"] == 0.0, res
    # the terminator is part of every continuation
    seen = []
    rank_items(FakeBackend(lambda c, k, r: seen.append(k) or 0.0), [item("t", ["x", "y"], "x", terminator=".")])
    assert seen == [" x.", " y."], seen
    print("1. rank arithmetic: ties, bounds, bytes, terminator                      PASS")

    # 2. random and perfect scorers
    rng = random.Random(0)
    pool = [f"v{i}" for i in range(300)]
    items = []
    for i in range(5000):
        gold = rng.choice(pool)
        others = rng.sample([v for v in pool if v != gold], 99)
        items.append(item(f"i{i}", others + [gold], gold, tier=str(rng.choice([1, 10, 100])), group="g"))
    random_results = rank_items(FakeBackend(lambda c, k, r: hashed(c, k)), items, batch_size=128)
    summary = summarize(random_results, by=("tier",), n_boot=200)
    assert all(0.0 <= r["norm_rank"] <= 1.0 for r in random_results)
    allrow = summary["all"]
    assert abs(allrow["norm_rank"] - 0.5) <= 3 * allrow["norm_rank_sigma"], allrow
    assert abs(allrow["top1"] - 0.01) < 0.01 and abs(allrow["chance_top1"] - 0.01) < 1e-12
    for tier in ("1", "10", "100"):
        s = summary[tier]
        assert abs(s["norm_rank"] - 0.5) <= 4 * s["norm_rank_sigma"], (tier, s)
    assert abs(allrow["z_vs_chance"]) < 3
    perfect = rank_items(FakeBackend(lambda c, k, r: 1.0 if k.strip() == "GOLD" else 0.0),
                         [item(f"p{i}", [f"x{i}", "GOLD", f"y{i}"], "GOLD") for i in range(50)])
    ps = summarize(perfect, n_boot=50)["all"]
    assert ps["norm_rank"] == 0.0 and ps["top1"] == 1.0 and ps["norm_rank_sigma"] == 0.0
    print(f"2. random scorer norm_rank {allrow['norm_rank']:.4f} +- {allrow['norm_rank_sigma']:.4f}, "
          f"perfect 0                  PASS")

    # 3. bootstrap sigma against the analytic standard error
    values = np.random.RandomState(1).randn(400) * 0.3 + 0.5
    analytic = values.std(ddof=1) / math.sqrt(len(values))
    sigma = bootstrap_sigma(values, n_boot=2000, seed=0)
    assert abs(sigma - analytic) / analytic < 0.1, (sigma, analytic)
    assert bootstrap_sigma([0.4], 100) == 0.0 and bootstrap_sigma([0.5] * 10, 100) == 0.0
    assert bootstrap_sigma(values, 500, 3) == bootstrap_sigma(values, 500, 3)
    print(f"3. bootstrap sigma {sigma:.4f} vs analytic {analytic:.4f}                       PASS")

    # 4. paired compare
    same = paired_compare(random_results, random_results, n_boot=100)
    assert all(v["diff_norm_rank"] == 0.0 and v["z"] == 0.0 for v in same.values()), same
    shifted = [dict(r, norm_rank=min(1.0, r["norm_rank"] + 0.1)) for r in random_results]
    diff = paired_compare(random_results, shifted, n_boot=100)["all"]
    assert diff["diff_norm_rank"] < -0.05 and diff["z"] < -10 and diff["n"] == 5000, diff
    partial = paired_compare(random_results[:100], random_results[50:], n_boot=50)["all"]
    assert partial["n"] == 50
    print("4. paired compare: identical is zero, a shift is read, partial overlap   PASS")

    # 5. order invariance
    shuffled = list(items)
    random.Random(5).shuffle(shuffled)
    back = {r["item_id"]: r for r in rank_items(FakeBackend(lambda c, k, r: hashed(c, k)), shuffled, batch_size=7)}
    assert all(back[r["item_id"]] == r for r in random_results)
    assert summarize(list(back.values()), n_boot=100) == summarize(random_results, n_boot=100)
    print("5. item order and batch size change no result                           PASS")

    # 6. evidence plumbing and truncation
    card = {"ids": [1], "chunk_ids": [0], "keys": np.zeros((1, 384), dtype=np.float32)}
    with_row = RankItem("e1", "ctx", "x", ["x", "y", "z"], "", "1", "g",
                        dict(card, marker="x"))
    without_row = RankItem("e2", "ctx", "x", ["x", "y", "z"], "", "1", "g")
    backend = FakeBackend(lambda c, k, row: 5.0 if row is not None and row["marker"] == k.strip() else 0.0)
    out = rank_items(backend, [with_row, without_row])
    assert out[0]["top1"]
    assert out[1]["rank"] == 2.0                                  # three ties
    assert sum(1 for _, _, row in backend.seen if row is not None) == 3
    try:
        rank_items(PlainBackend(lambda c, k, r: 0.0), [with_row])
        raise SystemExit("an evidence item scored on a backend without evidence support")
    except ValueError:
        pass
    long_item = RankItem("long", "w" * 500, "x", ["x", "y"])
    probe = FakeBackend(lambda c, k, r: 0.0)
    rank_items(probe, [long_item], max_len=64)
    assert all(a + b <= 64 for a, b, _ in probe.seen), probe.seen
    print("6. evidence rows aligned, no-evidence backend refused, long context cut  PASS")

    # 7. biography items
    pools = bio.make_pools(0)
    people = bio.make_people(pools, {1000: 3, 10: 4, 1: 5}, 1)
    items_a, swaps = make_bio_items(people, pools, forms=["indist", "heldout"])
    items_b, _ = make_bio_items(people, pools, forms=["indist", "heldout"])
    assert [i.candidates for i in items_a] == [i.candidates for i in items_b]
    assert len(items_a) == len(people) * 5 * 2 and not swaps
    assert len({i.item_id for i in items_a}) == len(items_a)
    for i in items_a:
        attribute = i.group
        assert i.candidates.count(i.gold) == 1 and len(set(i.candidates)) == len(i.candidates) == 100
        assert set(i.candidates) <= set(pools[attribute]) and i.evidence is None
        if attribute == "major":
            assert set(i.candidates) == set(pools["major"])
        assert i.terminator == ("." if i.item_id.endswith("indist") else "\n")
        assert i.tier in ("1", "10", "1000")
    assert any(a.candidates != b.candidates for a, b in zip(items_a, items_a[1:]) if a.group == b.group)

    def card_evidence(person, text):
        return {"ids": encode(text), "chunk_ids": [0] * len(text), "keys": None, "text": text}

    gold_items, _ = make_bio_items(people, pools, forms=["indist"], evidence="gold", card_evidence=card_evidence)
    assert all(i.evidence["text"] == people[int(i.item_id.split(":")[0])].store_chunk for i in gold_items)
    swapped_items, swaps = make_bio_items(people, pools, forms=["indist"], evidence="swapped",
                                          card_evidence=card_evidence)
    assert len(swaps) == len(swapped_items)
    for i in swapped_items:
        sub = swaps[i.item_id]["substitute"]
        assert sub in i.candidates and sub != i.gold and len(i.candidates) == 100
        assert sub in i.evidence["text"] and i.gold not in i.evidence["text"]
    # a model that always reads the card follows it; one that always uses memory does not
    reads_card = FakeBackend(lambda c, k, row: 5.0 if k.strip().rstrip(".") in row["text"] else 0.0)
    follower = rank_items(reads_card, swapped_items[:20], keep_scores=True)
    swap_readings(swapped_items[:20], follower, swaps)
    assert all(r["follow"] and r["mr_ll"] < 0.01 and "scores" not in r for r in follower), follower[0]
    assert all(r["swap_norm_rank"] < 0.05 for r in follower)
    gold_of = {i.context: i.gold for i in swapped_items[:20]}
    memory = FakeBackend(lambda c, k, row: 5.0 if k.strip().rstrip(".") == gold_of[c] else 0.0)
    rememberer = rank_items(memory, swapped_items[:20], keep_scores=True)
    swap_readings(swapped_items[:20], rememberer, swaps)
    assert not any(r["follow"] for r in rememberer) and all(r["mr_ll"] > 0.99 for r in rememberer)
    annotate(follower)
    assert follower[0]["cls"] in ("entity", "date", "noun") and follower[0]["form"] == "indist"
    # the prompt mode: the card text precedes the probe, nothing rides the port, same candidates
    prompt_items, _ = make_bio_items(people, pools, forms=["indist", "heldout"], evidence="prompt")
    assert [i.candidates for i in prompt_items] == [i.candidates for i in items_a]
    for i, plain in zip(prompt_items, items_a):
        card = people[int(i.item_id.split(":")[0])].store_chunk
        assert i.evidence is None and i.context == card + "\n" + plain.context
    refused = False
    try:
        make_bio_items(people, pools, forms=["indist"], evidence="prompt", fresh_names=object())
    except AssertionError:
        refused = True
    assert refused, "a prior control was built for an open-book reading"
    print("7. bio items: candidates, sha1 seeds, major pool, swapped and prompt      PASS")

    # 8. the three arm verdict, read against the paired prior control
    def synthetic(mean_by_tier, seed, prior_by_tier=None, n_per_tier=300, form="indist"):
        rng = np.random.RandomState(seed)
        prior_by_tier = prior_by_tier or {t: 0.5 for t in mean_by_tier}
        results = []

        def row(item_id, tier, mean):
            nr = float(np.clip(rng.beta(mean * 4, (1 - mean) * 4), 0, 1))
            rank = 1 + nr * 99
            return {"item_id": item_id, "tier": tier, "group": "birth_city", "cls": "entity", "form": form,
                    "n": 100, "rank": rank, "norm_rank": nr, "top1": bool(rank == 1.0),
                    "norm_rank_bytes": nr}

        for tier, mean in mean_by_tier.items():
            for k in range(n_per_tier):
                item_id = f"{seed}:{tier}:{k}"
                results.append(row(item_id, tier, mean))
                results.append(row(item_id + "|prior", tier, prior_by_tier[tier]))
        return results

    chance = {"1": 0.5, "10": 0.5, "100": 0.5, "1000": 0.5}
    climbing = {"1": 0.5, "10": 0.47, "100": 0.3, "1000": 0.1}
    arms = {"full": synthetic(climbing, 0), "masked": synthetic(chance, 1), "retrieval": synthetic(chance, 2)}
    line, details = micro_verdict(arms, n_boot=100)
    assert line.startswith("HOLDS"), line
    leaky = dict(arms, retrieval=synthetic({"1": 0.5, "10": 0.4, "100": 0.3, "1000": 0.1}, 3))
    line, details = micro_verdict(leaky, n_boot=100)
    assert line.startswith("FAILS") and "retrieval" in line and details["leaks"]["retrieval"], line
    flat = dict(arms, full=synthetic(chance, 4))
    line, _ = micro_verdict(flat, n_boot=100)
    assert line.startswith("UNINFORMATIVE"), line
    # a value marginal learner sits below 0.5 at the heavy tiers but equally so with a fresh name:
    # the delta is 0 and the old 0.5 reading would have called it a leak
    marginal = {"1": 0.5, "10": 0.48, "100": 0.35, "1000": 0.2}
    line, _ = micro_verdict(dict(arms, masked=synthetic(marginal, 5, marginal)), n_boot=100)
    assert line.startswith("HOLDS"), line
    # the other side is read too: a real rank worse than the prior is not "at chance"
    worse = {"1": 0.5, "10": 0.5, "100": 0.7, "1000": 0.5}
    line, details = micro_verdict(dict(arms, masked=synthetic(worse, 6)), n_boot=100)
    assert line.startswith("FAILS") and details["leaks"]["masked"] == ["100"], line
    # a verdict per form, never pooled
    both = {name: results + [dict(r, form="heldout", item_id="h" + r["item_id"]) for r in results]
            for name, results in arms.items()}
    verdicts = micro_verdicts(both, n_boot=100)
    assert sorted(verdicts) == ["heldout", "indist"] and all(v[0].startswith("HOLDS") for v in verdicts.values())
    heldout_flat = {name: [r for r in results if r["form"] != "heldout"]
                    + [dict(r, form="heldout", item_id="h" + r["item_id"]) for r in
                       (synthetic(chance, 40 + i, form="heldout") if name == "full" else results)]
                    for i, (name, results) in enumerate(arms.items())}
    verdicts = micro_verdicts(heldout_flat, n_boot=100)
    assert verdicts["indist"][0].startswith("HOLDS") and verdicts["heldout"][0].startswith("UNINFORMATIVE")
    print("8. verdict vs prior: HOLDS, FAILS two sided, UNINFORMATIVE, per form       PASS")

    # 9. the prior control through make_bio_items and scripted scorers
    pools9 = bio.make_pools(0)
    people9 = bio.make_people(pools9, {1000: 40, 100: 40, 10: 40, 1: 40}, 2)
    fresh = bio.FreshNames(people9, pools9)
    controlled, _ = make_bio_items(people9, pools9, forms=["indist", "heldout"], fresh_names=fresh)
    plain, _ = make_bio_items(people9, pools9, forms=["indist", "heldout"])
    assert len(controlled) == 2 * len(plain)
    by_id = {i.item_id: i for i in controlled}
    for i in plain:
        c = by_id[i.item_id + "|prior"]
        assert c.candidates == i.candidates and c.gold == i.gold and c.tier == i.tier and c.group == i.group
        assert c.terminator == i.terminator and c.evidence is None and c.context != i.context
        person = people9[int(i.item_id.split(":")[0])]
        assert person.name not in c.context
        assert c.context == bio.probe_prompt(i.group, fresh.name(i.item_id), i.item_id.split(":")[2])[0]
    controlled_indist = [i for i in controlled if ":indist" in i.item_id]
    names_of = {p.name: p for p in people9}

    count = {}
    for p in people9:
        for v in p.attributes.values():
            count[v] = count.get(v, 0) + p.tier

    def value_of(cont):
        return cont.strip().rstrip(".")

    marginal_only = FakeBackend(lambda ctx, cont, row: math.log1p(count.get(value_of(cont), 0)))
    results = rank_items(marginal_only, controlled_indist, batch_size=64)
    annotate(results)
    d = prior_deltas(results, by=("tier",), n_boot=200)
    heavy = summarize(real_only(results), by=("tier",), n_boot=200)["1000"]
    assert heavy["norm_rank"] < 0.35, heavy                      # the old chance line would call this knowledge
    for key, s in d.items():
        assert abs(s["delta"]) <= 3 * s["delta_sigma"] + 1e-9, (key, s)
    assert d["all"]["delta"] == 0.0

    def binder(ctx, cont, row):
        person = names_of.get(" ".join(ctx.split()[:3]))
        known = person is not None and value_of(cont) in person.attributes.values()
        return (5.0 if known else 0.0) + 0.01 * hashed(ctx, cont)

    results = rank_items(FakeBackend(binder), controlled_indist, batch_size=64)
    annotate(results)
    d = prior_deltas(results, by=("tier",), n_boot=200)
    for tier in ("1", "10", "100", "1000"):
        assert d[tier]["delta"] > 0.3 and d[tier]["z_delta"] > 10, (tier, d[tier])
    assert abs(d["all"]["norm_rank_prior"] - 0.5) < 0.05, d["all"]
    by_group = prior_deltas(results, by=("tier", "group"), n_boot=50)
    assert "1000|birth_city" in by_group and by_group["all"]["n"] == len(controlled_indist) // 2
    print(f"9. prior control: marginal learner delta 0, name binder delta {d['all']['delta']:.2f}        PASS")

    # 10. compare on three files prints a verdict per form
    import io, json, tempfile, shutil
    from argparse import Namespace
    from contextlib import redirect_stdout
    from scripts.closed_book_rank import run_compare
    tmp = tempfile.mkdtemp(prefix="rank_")
    try:
        paths = []
        for name, results in arms.items():
            path = os.path.join(tmp, f"{name}.json")
            with open(path, "w") as f:
                json.dump({"flags": {"checkpoint": name}, "results": results, "summary": {}}, f)
            paths.append(path)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            run_compare(Namespace(files=paths, by="tier", n_boot=50))
        text = buffer.getvalue()
        assert "verdict (indist form): HOLDS" in text and "prior" in text, text
        assert "full minus masked" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("10. compare on three files prints the verdict                            PASS")
    print("all closed-book rank checks passed")


if __name__ == "__main__":
    main()

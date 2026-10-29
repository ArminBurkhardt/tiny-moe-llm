"""Pure-function checks for scripts/eval_chains.py: records, tables, paired delta, answer CE."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_chains import (accuracy_table, answer_ce, build_records, cell_extras, divergent_position,
                                 nll_table, paired_delta, paired_delta_by_hops, tally)


def item(index, hops, gold, cand_ids):
    k = len(cand_ids)
    return {"index": index, "hops": hops, "answer": f"c{gold}", "candidates": [f"c{i}" for i in range(k)],
            "gold": gold, "prompt": [1, 2, 3], "cand_ids": cand_ids, "evidence": None}


def test_divergent_position():
    assert divergent_position([[5, 6, 9], [7, 6, 9]]) == 0
    assert divergent_position([[5, 6, 9], [5, 7, 9], [5, 6, 8]]) == 1
    assert divergent_position([[5, 6, 9], [5, 6, 8]]) == 2
    assert divergent_position([[5, 6, 9]]) == 0
    assert divergent_position([[5, 6], [5, 6]]) == 0


def test_records():
    items = [
        item(10, 2, 1, [[5, 6, 0], [5, 7, 0], [5, 8, 0]]),
        item(11, 1, 0, [[4, 0], [9, 0]]),
    ]
    scores = [[-3.0, -1.0, -5.0], [-4.0, -2.0]]
    lps = [[[-1.0, -1.5, -0.5], [-0.25, -0.5, -0.25], [-2.0, -2.0, -1.0]], [[-3.0, -1.0], [-1.5, -0.5]]]
    recs = build_records(items, scores, lps)
    a, b = recs
    assert a["question"] == 10 and a["hops"] == 2 and a["gold"] == 1 and a["n_candidates"] == 3
    assert a["gold_rank"] == 0 and a["correct"] is True
    assert abs(a["margin"] - 2.0) < 1e-9
    assert a["gold_tokens"] == 3 and abs(a["gold_nll"] - 1.0) < 1e-9
    assert abs(a["gold_nll_per_token"] - 1.0 / 3) < 1e-3
    assert abs(a["ln_k"] - math.log(3)) < 1e-3
    assert a["divergent_pos"] == 1 and abs(a["gold_nll_at_divergence"] - 0.5) < 1e-9
    assert b["gold_rank"] == 1 and b["correct"] is False
    assert abs(b["margin"] + 2.0) < 1e-9
    assert b["divergent_pos"] == 0 and abs(b["gold_nll_at_divergence"] - 3.0) < 1e-9
    assert b["scores"] == [-4.0, -2.0]
    # a tie goes to the lower index, as argmax does
    tie = build_records([item(0, 1, 1, [[4, 0], [9, 0]])], [[-1.0, -1.0]], [[[-0.5, -0.5], [-0.5, -0.5]]])[0]
    assert tie["gold_rank"] == 1 and tie["correct"] is False
    one = build_records([item(0, 1, 0, [[4, 0]])], [[-1.0]], [[[-0.5, -0.5]]])[0]
    assert one["margin"] == 0.0 and one["ln_k"] == 0.0 and one["gold_rank"] == 0

    ex = cell_extras(recs)
    assert abs(ex["gold_nll_per_answer"] - (1.0 + 4.0) / 2) < 1e-9
    assert abs(ex["ce_per_token"] - 5.0 / (3 + 2)) < 1e-9
    assert abs(ex["nll_at_divergence"] - (0.5 + 3.0) / 2) < 1e-9
    assert abs(ex["ln_k"] - (math.log(3) + math.log(2)) / 2) < 1e-3


def test_paired_delta():
    same = [True, False, True, True] * 5
    d = paired_delta(same, same)
    assert d["delta"] == 0.0 and d["sigma"] == 0.0 and d["n"] == 20
    lo = [False] * 20
    hi = [True] + [False] * 19
    d = paired_delta(lo, hi)
    assert abs(d["delta"] - 1 / 20) < 1e-12 and d["n"] == 20 and d["sigma"] > 0
    assert paired_delta(lo, hi) == d
    assert math.isnan(paired_delta([], [])["delta"])

    def rec(q, hops, ok):
        return {"question": q, "hops": hops, "correct": ok}

    by_kept = {0: [rec(0, 1, False)], 1: [rec(0, 1, False), rec(1, 2, False)],
               3: [rec(0, 1, True), rec(1, 2, False)]}
    by_kept[0] = [rec(0, 1, False), rec(1, 2, False)]
    out = paired_delta_by_hops(by_kept)
    assert out[1]["delta"] == 1.0 and out[1]["kept_lo"] == 1 and out[1]["kept_hi"] == 3 and out[1]["n"] == 1
    assert out[2]["delta"] == 0.0
    assert paired_delta_by_hops({0: by_kept[0]}) == {}
    assert paired_delta_by_hops({1: by_kept[1]}) == {}


def test_tables():
    stats = {"acc": 0.178, "sigma": 0.012, "chance": 0.185, "z_vs_chance": -0.5, "n": 10}
    by_kept = {0: {1: dict(stats), 2: dict(stats)}, 2: {1: dict(stats), 2: dict(stats)}}
    lines = accuracy_table(by_kept)
    assert len(lines) == 3
    assert "  *0.178+-0.012 | 0.185 |   -0.5" in lines[1]
    assert lines[1].count("*") == 2 and lines[2].count("*") == 0
    # the marker sits in its own column, aligned across rows and under the header
    col = lines[1].index("*")
    assert lines[1][col - 1] == " " and lines[2][col] == " "
    assert lines[0].rstrip().endswith("hops 2") and len(lines[0]) == len(lines[1])

    extras = dict(stats, gold_nll_per_answer=1.234, ln_k=1.5, ce_per_token=0.5, nll_at_divergence=0.75)
    lines = nll_table({1: {1: extras}})
    assert "1.234" in lines[1] and "1.500" in lines[1] and "0.500" in lines[1] and "0.750" in lines[1]


def test_answer_ce_unchanged():
    items = [item(0, 2, 0, [[5, 0], [6, 0]]), item(1, 2, 1, [[5, 0], [6, 0], [7, 0]])]
    by_hops = tally(items, [[-1.0, -3.0], [-2.0, -2.5, -4.0]])
    cell = by_hops[2]
    assert cell["correct"] == [True, False]
    assert abs(answer_ce(cell) - (1.0 + 2.5) / 4) < 1e-12
    assert answer_ce({"gold_lp": [], "gold_n": []}) == 0.0


if __name__ == "__main__":
    test_divergent_position()
    test_records()
    test_paired_delta()
    test_tables()
    test_answer_ce_unchanged()
    print("test_eval_chains: ok")

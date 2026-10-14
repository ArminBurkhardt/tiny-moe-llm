"""The exit read statistics on hand built tensors (no model, no CUDA, no tokenizer).

1. per exit mean CE, top-1 accuracy and mean p_max;
2. paired gains: consecutive and first to last, mean, iid and cluster standard errors and z against
   hand values, and the no spread cases (z None for a non-zero mean);
3. oracle: per token minimum, argmin shares with ties going to the earliest exit, headroom;
4. entropy-regularized optimum: shares sum to 1, uniform at equal CE, a T=2 hand value, and the
   small beta limit approaching the argmin;
5. confidence rule: exit index, mean passes and CE per threshold, the neighbour interpolation and the
   lower hull baselines (they differ on a non convex CE), a p_max exactly at the threshold counting
   as reached, and a threshold nothing reaches (everything leaves at the last exit);
6. decile table: counts, bin membership by ascending p_max at exit 1, gains and the ratio, with an
   uneven N;
7. nothing is hardcoded to three exits: T=2 and T=1 run through ``analyze``, which also refuses an
   empty token set;
8. ``exit_readouts`` against ``F.cross_entropy`` per exit on CPU with an ``nn.Linear`` head, runs of
   -100, several rows and chunks smaller than the supervised count.
"""
import os
import sys
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from scripts.eval_exit import (
    analyze, beta_optimum, confidence_rule, decile_table, exit_readouts, hull_ce, interpolated_ce,
    oracle_stats, paired_gains, per_exit_stats,
)


def close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b}"


def close_list(a, b, tol=1e-6):
    assert len(a) == len(b), (a, b)
    for x, y in zip(a, b):
        close(x, y, tol)


# four tokens, three exits; columns are tokens
CE = torch.tensor([[3.0, 2.0, 1.0, 4.0],
                   [2.0, 2.0, 2.0, 1.0],
                   [1.0, 3.0, 1.0, 2.0]])
PMAX = torch.tensor([[0.4, 0.6, 0.95, 0.3],
                     [0.8, 0.5, 0.9, 0.99],
                     [0.9, 0.9, 0.9, 0.9]])
OK = torch.tensor([[0.0, 0.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 1.0],
                   [1.0, 1.0, 1.0, 1.0]])


def check_per_exit():
    s = per_exit_stats(CE, PMAX, OK)
    close_list(s["ce"], [2.5, 1.75, 1.75])
    close_list(s["top1"], [0.25, 0.75, 1.0])
    close_list(s["p_max"], [0.5625, 0.7975, 0.9])


def check_gains():
    g = paired_gains(CE)
    assert [(c["from"], c["to"]) for c in g["consecutive"]] == [(1, 2), (2, 3)]
    first = g["consecutive"][0]
    close(first["mean"], 0.75)
    # diffs 1, 0, -1, 3: sample variance 8.75 / 3
    close(first["se"], math.sqrt(8.75 / 3) / 2)
    close(first["z"], 0.75 / first["se"])
    close(first["se_iid"], first["se"])
    assert first["se_cluster"] is None and first["clusters"] is None
    # second pair diffs 1, -1, 1, -1: mean 0
    second = g["consecutive"][1]
    close(second["mean"], 0.0)
    close(second["z"], 0.0)
    ftl = g["first_to_last"]
    assert (ftl["from"], ftl["to"]) == (1, 3)
    # diffs 2, -1, 0, 2
    close(ftl["mean"], 0.75)
    close(ftl["se"], math.sqrt(((1.25 ** 2) + (1.75 ** 2) + (0.75 ** 2) + (1.25 ** 2)) / 3) / 2)
    # a constant non-zero difference has no spread: no finite z, reported None
    flat = paired_gains(torch.tensor([[2.0, 3.0], [1.0, 2.0]]))
    close(flat["first_to_last"]["mean"], 1.0)
    close(flat["first_to_last"]["se"], 0.0)
    assert flat["first_to_last"]["z"] is None
    # no spread and a zero mean: z is 0
    close(paired_gains(torch.ones(2, 3))["first_to_last"]["z"], 0.0)
    # clusters {0, 1}: diffs 1, 0 | -1, 3 give sums 1 and 2, counts 2 and 2, mean 0.75, residuals
    # -0.5 and 0.5, se = sqrt(2 / 1 * 0.5) / 4 = 0.25, and z comes from the cluster se
    cg = paired_gains(CE, torch.tensor([0, 0, 1, 1]))["consecutive"][0]
    close(cg["se_cluster"], 0.25)
    close(cg["se"], 0.25)
    close(cg["z"], 3.0)
    close(cg["se_iid"], math.sqrt(8.75 / 3) / 2)
    assert cg["clusters"] == 2
    # one cluster per token: sqrt(N / (N - 1)) * sqrt(sum (x - m)^2) / N equals the iid se
    per_token = paired_gains(CE, torch.arange(4))["consecutive"][0]
    close(per_token["se_cluster"], per_token["se_iid"])
    # a single cluster has no spread estimate: falls back to the iid se
    one = paired_gains(CE, torch.zeros(4, dtype=torch.long))["consecutive"][0]
    assert one["se_cluster"] is None
    close(one["se"], one["se_iid"])


def check_oracle():
    o = oracle_stats(CE)
    # per token minimum 1, 2, 1, 1; argmin 2, 0 (tie with 1), 0 (tie with 2), 1
    close(o["oracle_ce"], 1.25)
    close_list(o["argmin_share"], [0.5, 0.25, 0.25])
    close(o["headroom"], 1.75 - 1.25)
    # all equal: everything goes to exit 1
    flat = oracle_stats(torch.ones(3, 5))
    close_list(flat["argmin_share"], [1.0, 0.0, 0.0])
    close(flat["headroom"], 0.0)


def check_beta():
    for beta in (0.05, 0.1, 1.0):
        b = beta_optimum(CE, beta)
        close(sum(b["exit_share"]), 1.0)
        assert 0.0 <= b["entropy_over_ln_t"] <= 1.0 + 1e-12
        assert b["expected_ce"] >= 1.25 - 1e-9
        assert 1.0 <= b["expected_depth"] <= 3.0
    # equal CE at every exit: uniform, full entropy, mean depth 2, no confident token
    u = beta_optimum(torch.ones(3, 4), 0.1)
    close_list(u["exit_share"], [1 / 3] * 3)
    close(u["entropy_over_ln_t"], 1.0)
    close(u["expected_depth"], 2.0)
    close(u["share_max_p_above_0.99"], 0.0)
    # T=2, one token with ce [0, beta ln 3]: p = [0.75, 0.25]
    beta = 0.2
    ce = torch.tensor([[0.0], [beta * math.log(3.0)]])
    h = beta_optimum(ce, beta)
    close_list(h["exit_share"], [0.75, 0.25])
    close(h["expected_ce"], 0.25 * beta * math.log(3.0))
    close(h["expected_depth"], 1.25)
    close(h["entropy_over_ln_t"], -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(2.0))
    close(h["share_max_p_above_0.99"], 0.0)
    # a tiny beta puts all mass on the lower CE exit of each token (no ties here)
    sharp = beta_optimum(torch.tensor([[1.0, 3.0], [2.0, 1.0]]), 1e-3)
    close_list(sharp["exit_share"], [0.5, 0.5], 1e-6)
    close(sharp["share_max_p_above_0.99"], 1.0)
    close(sharp["expected_ce"], 1.0, 1e-6)


def check_confidence():
    c = confidence_rule(CE, PMAX, 0.9)
    # exits 3, 3, 1, 2 (1-based): CE 1, 3, 1, 1
    close(c["ce"], 1.5)
    close(c["passes"], 2.25)
    close_list(c["exit_share"], [0.25, 0.25, 0.5])
    # mix of depth 2 (1.75) and depth 3 (1.75) at 2.25 passes
    close(c["mix_ce"], 1.75)
    close(c["neighbour_mix_ce"], 1.75)
    c5 = confidence_rule(CE, PMAX, 0.5)
    # exits 2, 1, 1, 2: CE 2, 2, 1, 1
    close(c5["ce"], 1.5)
    close(c5["passes"], 1.5)
    close(c5["mix_ce"], 0.5 * 2.5 + 0.5 * 1.75)
    # nothing reaches 0.999 before the last exit
    top = confidence_rule(CE, PMAX, 0.999)
    close(top["passes"], 3.0)
    close(top["ce"], 1.75)
    close(top["mix_ce"], 1.75)
    # a p_max exactly at tau at a middle exit is reached (>=): token 0 leaves at exit 2 with CE 1,
    # where a strict > would send it to exit 3 with CE 2 and change every number below
    ge = confidence_rule(torch.tensor([[3.0, 3.0], [1.0, 1.0], [2.0, 9.0]]),
                         torch.tensor([[0.2, 0.2], [0.5, 0.2], [0.9, 0.9]]), 0.5)
    close(ge["ce"], 5.0)
    close(ge["passes"], 2.5)
    close_list(ge["exit_share"], [0.0, 0.5, 0.5])
    # not convex in depth: mean CE 1, 5, 2; leaving at exits 2 and 3 gives 2.5 passes and CE 3.5,
    # the neighbour mix of depths 2 and 3 is 3.5 but the hull (depths 1 and 3) is 1.75
    nc = confidence_rule(torch.tensor([[1.0, 1.0], [5.0, 5.0], [2.0, 2.0]]),
                         torch.tensor([[0.1, 0.1], [0.95, 0.1], [0.1, 0.1]]), 0.9)
    close(nc["ce"], 3.5)
    close(nc["passes"], 2.5)
    close(nc["neighbour_mix_ce"], 0.5 * 5.0 + 0.5 * 2.0)
    close(nc["mix_ce"], 1.75)
    assert nc["mix_ce"] < nc["neighbour_mix_ce"]
    close(hull_ce([1.0, 5.0, 2.0], 2.0), 1.5)
    close(hull_ce([1.0, 5.0, 2.0], 1.0), 1.0)
    close(hull_ce([1.0, 5.0, 2.0], 3.0), 2.0)
    # convex CE: the hull is the neighbour interpolation
    close(hull_ce([4.0, 2.0, 1.0], 1.5), interpolated_ce([4.0, 2.0, 1.0], 1.5))
    close(hull_ce([4.0, 2.0, 1.0], 2.75), interpolated_ce([4.0, 2.0, 1.0], 2.75))
    close(hull_ce([3.0], 1.0), 3.0)
    close(interpolated_ce([3.0], 1.0), 3.0)
    # interpolation: integer and fractional pass counts
    close(interpolated_ce([4.0, 2.0, 1.0], 1.0), 4.0)
    close(interpolated_ce([4.0, 2.0, 1.0], 1.5), 3.0)
    close(interpolated_ce([4.0, 2.0, 1.0], 2.75), 2.0 * 0.25 + 1.0 * 0.75)
    close(interpolated_ce([4.0, 2.0, 1.0], 3.0), 1.0)


def check_deciles():
    n = 20
    rank = torch.tensor([(7 * j) % n for j in range(n)])
    p0 = (rank.double() + 1) / n
    ce0 = rank.double()
    ce = torch.stack([ce0, ce0 / 2])
    pmax = torch.stack([p0, torch.ones(n, dtype=torch.double)])
    d = decile_table(ce, pmax)
    assert len(d["bins"]) == 10
    assert all(b["count"] == 2 for b in d["bins"])
    close(d["bins"][0]["p_max_first"], 0.075)
    close(d["bins"][9]["p_max_first"], 0.975)
    close_list(d["bins"][0]["ce"], [0.5, 0.25])
    close_list(d["bins"][9]["ce"], [18.5, 9.25])
    for b, bin_ in enumerate(d["bins"]):
        close(bin_["gain_first_last"], b + 0.25)
        close(bin_["gain_last_step"], b + 0.25)
    close(d["gain_lowest"], 0.25)
    close(d["gain_highest"], 9.25)
    close(d["gain_ratio"], 0.25 / 9.25)
    # an uneven N: the first bins take the extra tokens
    uneven = decile_table(torch.rand(3, 23).double(), torch.rand(3, 23).double())
    assert [b["count"] for b in uneven["bins"]] == [3, 3, 3] + [2] * 7
    # last step gain is the second to last exit minus the last
    ce3 = torch.tensor([[5.0, 6.0], [3.0, 3.0], [2.0, 1.0]]).double()
    d3 = decile_table(ce3, torch.tensor([[0.1, 0.9]] * 3).double(), n_bins=2)
    close(d3["bins"][0]["gain_first_last"], 3.0)
    close(d3["bins"][0]["gain_last_step"], 1.0)
    close(d3["bins"][1]["gain_first_last"], 5.0)
    close(d3["bins"][1]["gain_last_step"], 2.0)
    close(d3["gain_ratio"], 3.0 / 5.0)
    # a zero or negative gain in the top decile leaves the ratio undefined
    zero = decile_table(torch.ones(2, 10).double(), torch.rand(2, 10).double())
    assert zero["gain_ratio"] is None
    neg = decile_table(torch.tensor([[1.0, 1.0], [2.0, 2.0]]).double(), torch.rand(2, 2).double(), n_bins=2)
    assert neg["gain_ratio"] is None and neg["gain_highest"] < 0


def check_other_depths():
    # T=2: the same functions with nothing tied to three exits
    ce = torch.tensor([[2.0, 1.0, 4.0, 3.0], [1.0, 1.0, 2.0, 5.0]])
    pm = torch.tensor([[0.2, 0.95, 0.4, 0.99], [0.7, 0.8, 0.9, 0.6]])
    ok = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    r = analyze(ce, pm, ok, [0.1], [0.9])
    assert r["n_exits"] == 2 and r["n_tokens"] == 4 and r["n_clusters"] is None
    close_list(r["per_exit"]["ce"], [2.5, 2.25])
    assert len(r["gains"]["consecutive"]) == 1
    close(r["gains"]["first_to_last"]["mean"], 0.25)
    # per token minimum 1, 1, 2, 3; argmin 1, 0 (tie), 1, 0
    close(r["oracle"]["oracle_ce"], 1.75)
    close_list(r["oracle"]["argmin_share"], [0.5, 0.5])
    close(r["oracle"]["headroom"], 0.5)
    # tau 0.9: tokens 2 and 4 leave at exit 1, tokens 1 and 3 at exit 2
    c = r["confidence"][0]
    close(c["ce"], (1.0 + 1.0 + 2.0 + 3.0) / 4)
    close(c["passes"], 1.5)
    close(c["mix_ce"], 0.5 * 2.5 + 0.5 * 2.25)
    assert len(r["beta"][0]["exit_share"]) == 2
    assert len(r["deciles"]["bins"]) == 4
    rc = analyze(ce, pm, ok, [0.1], [0.9], torch.tensor([5, 5, 9, 9]))
    assert rc["n_clusters"] == 2
    # no tokens: a clear refusal, not a NaN report
    try:
        analyze(torch.zeros(3, 0), torch.zeros(3, 0), torch.zeros(3, 0), [0.1], [0.9])
    except AssertionError as err:
        assert "no supervised tokens" in str(err)
    else:
        raise AssertionError("an empty token set must be refused")
    # T=1: a single exit has no gain to report and nothing divides by ln 1
    one = analyze(ce[:1], pm[:1], ok[:1], [0.1], [0.9])
    assert one["gains"]["consecutive"] == []
    close(one["beta"][0]["entropy_over_ln_t"], 0.0)
    close(one["confidence"][0]["passes"], 1.0)
    # T=4: shares and passes scale with the number of exits
    ce4 = torch.tensor([[4.0, 4.0], [3.0, 4.0], [2.0, 4.0], [1.0, 4.0]])
    r4 = analyze(ce4, torch.full((4, 2), 0.5), torch.zeros(4, 2), [0.05], [0.9])
    close_list(r4["oracle"]["argmin_share"], [0.5, 0.0, 0.0, 0.5])
    close(r4["confidence"][0]["passes"], 4.0)
    close(sum(r4["beta"][0]["exit_share"]), 1.0)


def check_exit_readouts():
    gen = torch.Generator().manual_seed(0)
    T, B, S, H, V = 3, 3, 9, 6, 11
    hidden = torch.randn(T, B, S, H, generator=gen)
    head = torch.nn.Linear(H, V)
    torch.nn.init.normal_(head.weight, generator=gen)
    labels = torch.randint(0, V, (B, S), generator=gen)
    labels[0, :4] = -100
    labels[1, 2:5] = -100
    labels[2, 6:] = -100
    labels[1, -1] = -100
    supervised = labels[:, 1:] != -100
    n = int(supervised.sum())
    assert n > 8
    with torch.no_grad():
        for chunk in (4, 7, 4096):
            ce, pm, ok, index = exit_readouts(hidden, labels, head, chunk)
            assert ce.shape == (T, n) and pm.shape == (T, n) and ok.shape == (T, n)
            assert torch.equal(index, supervised.nonzero())
            for t in range(T):
                flat = head(hidden[t])[:, :-1].reshape(-1, V)
                want = F.cross_entropy(flat, labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none")
                want = want[supervised.reshape(-1)]
                assert torch.allclose(ce[t], want, atol=1e-5), (chunk, t)
                probs = flat.softmax(-1)[supervised.reshape(-1)]
                assert torch.allclose(pm[t], probs.max(-1).values, atol=1e-5)
                hit = (probs.argmax(-1) == labels[:, 1:][supervised]).float()
                assert torch.equal(ok[t], hit)
        # a batch with nothing supervised returns empty readouts
        e_ce, _, _, e_idx = exit_readouts(hidden, torch.full((B, S), -100), head, 4)
        assert e_ce.shape == (T, 0) and e_idx.shape[0] == 0


check_per_exit()
check_gains()
check_oracle()
check_beta()
check_confidence()
check_deciles()
check_other_depths()
check_exit_readouts()
print("all eval exit checks passed")

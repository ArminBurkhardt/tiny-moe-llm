"""Per-exit readout of the looped trunk: how much would choosing a depth per token buy?

The model applies one block ``n_loops`` times. A forward with ``return_hidden=True`` returns the
normed hidden state after every pass, and the LM head applied to pass ``t + 1``'s state is the
readout of an exit after ``t + 1`` passes. This script runs one full depth forward per batch on a
data split with no evidence attached (the closed-book read), applies the head at every exit to the
supervised positions only, and keeps per-token cross entropy, max probability and top-1 correctness
for every exit. All statistics are pure functions of three ``[T, N]`` tensors (exits by tokens):

1. per exit mean CE, top-1 accuracy and mean ``p_max``;
2. paired gain between consecutive exits and first to last, with the iid standard error and the
   cluster robust one (one cluster per document), z from the cluster one;
3. the oracle (per token minimum over exits) and the argmin share per exit (ties go to the earliest
   exit);
4. the entropy-regularized optimum per ``beta``: ``p = softmax_t(-ce / beta)`` per token;
5. the confidence exit rule per threshold: exit at the first pass whose ``p_max`` reaches the
   threshold, else at the last, against the best random mix of fixed depths with the same mean
   passes (the lower convex hull of mean CE over depth, evaluated at the rule's mean passes);
6. a decile table on ``p_max`` at the first exit: does low confidence early predict a large gain
   from the later passes?

**The exit is readout-only.** Every token still runs all passes and feeds the later passes of the
tokens after it through attention, so the CE under the confidence rule or the beta optimum is
optimistic against a real per-token exit, where a token that stopped would leave no later state for
its successors. The oracle and the beta optimum choose the exit with the true label, so they are
bounds, not policies (that is the main reason they are optimistic; the minimum over noisy values is
a second, smaller one).

    TINY_LLM_CONFIG=config_micro.yaml python scripts/eval_exit.py -c CKPT --split inject_val \\
        --data-dir data/prepared_inject --json-out ckpts/inject/exit_full.json

The CE here is the plain per token mean over supervised tokens of the next token loss. It matches the
trainer's ``[eval]`` CE (final exit) only when the split has no ``.ev`` file (this read attaches no
evidence and the packing of an evidence split differs, so such a split is refused), conversation
weighting is off, and the profile's batch size and batch count are used. The head here runs in fp32
on the logits where the trainer takes the CE on bf16 logits, so expect small differences.

Args:
    -c/--checkpoint: checkpoint to read; the shape comes from the checkpoint.
    --data-dir: directory holding the split (default: the SFT config's).
    --split: split name, for example ``inject_val``; a split with a ``.ev`` file is refused.
    --batch-size: rows per forward (default: the profile's batch size).
    --max-batches: batches to read; a batch with no supervised token does not count.
    --betas: comma list of entropy regularization strengths.
    --thresholds: comma list of ``p_max`` thresholds for the confidence exit rule.
    --json-out: write the report here.
    --device: torch device.
"""
import os
import sys
import json
import math
import argparse
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils import BASE_DIR, TOKENIZER_DIR, logger

HEAD_CHUNK = 4096
N_DECILES = 10
CLUSTER_STRIDE = 1 << 20


def _f(x) -> float:
    return float(x)


def mean_se(values: torch.Tensor, clusters: Optional[torch.Tensor] = None) -> dict:
    """Mean of ``values`` with its iid standard error and, given cluster ids, the cluster robust one.

    The cluster standard error is the linearized one for a mean: with per cluster sum ``S_c`` and
    count ``n_c``, ``m = sum S_c / N`` and
    ``SE = sqrt(G / (G - 1) * sum_c (S_c - m * n_c)^2) / N`` over ``G`` clusters.

    Args:
        values: ``[N]`` per token values.
        clusters: ``[N]`` integer cluster ids, or None for the iid reading only.

    Returns:
        ``mean``, ``se_iid``, ``se_cluster`` (None without clusters or with a single one), ``clusters``
        (count, None without ids).
    """
    values = values.double()
    n = values.numel()
    mean = values.mean()
    se_iid = _f(values.std(correction=1) / math.sqrt(n)) if n > 1 else 0.0
    se_cluster, groups = None, None
    if clusters is not None:
        _, inverse = torch.unique(clusters, return_inverse=True)
        groups = int(inverse.max().item()) + 1
        if groups > 1:
            sums = torch.zeros(groups, dtype=torch.double).index_add_(0, inverse, values)
            counts = torch.bincount(inverse, minlength=groups).double()
            resid = sums - mean * counts
            se_cluster = _f(torch.sqrt(groups / (groups - 1) * (resid ** 2).sum()) / n)
    return {"mean": _f(mean), "se_iid": se_iid, "se_cluster": se_cluster, "clusters": groups}


def per_exit_stats(ce: torch.Tensor, p_max: torch.Tensor, correct: torch.Tensor) -> dict:
    """Mean CE, top-1 accuracy and mean ``p_max`` at every exit.

    Args:
        ce: ``[T, N]`` per token cross entropy at each exit.
        p_max: ``[T, N]`` per token max probability.
        correct: ``[T, N]`` top-1 correctness (0/1).
    """
    ce, p_max, correct = ce.double(), p_max.double(), correct.double()
    return {
        "ce": [_f(v) for v in ce.mean(1)],
        "top1": [_f(v) for v in correct.mean(1)],
        "p_max": [_f(v) for v in p_max.mean(1)],
    }


def _paired(diff: torch.Tensor, clusters: Optional[torch.Tensor]) -> dict:
    stats = mean_se(diff, clusters)
    se = stats["se_cluster"] if stats["se_cluster"] is not None else stats["se_iid"]
    if se > 0:
        z = stats["mean"] / se
    else:
        # no spread: a zero mean has nothing to test, a non-zero mean has no finite z
        z = 0.0 if stats["mean"] == 0 else None
    return {**stats, "se": se, "z": z}


def paired_gains(ce: torch.Tensor, clusters: Optional[torch.Tensor] = None) -> dict:
    """Paired CE gain between consecutive exits and from the first exit to the last.

    Args:
        ce: ``[T, N]`` per token cross entropy at each exit.
        clusters: ``[N]`` cluster ids (one per document). The z uses the cluster robust standard
            error; None falls back to the iid one.

    Returns:
        ``{"consecutive": [{"from": 1, "to": 2, mean, se_iid, se_cluster, se, z, clusters}, ...],
        "first_to_last": {...}}`` with exits 1-based and the gain defined as ``ce[from] - ce[to]``;
        ``se`` is the one z uses.
    """
    ce = ce.double()
    T = ce.size(0)
    consecutive = []
    for t in range(T - 1):
        consecutive.append({"from": t + 1, "to": t + 2, **_paired(ce[t] - ce[t + 1], clusters)})
    first_to_last = {"from": 1, "to": T, **_paired(ce[0] - ce[-1], clusters)}
    return {"consecutive": consecutive, "first_to_last": first_to_last}


def oracle_stats(ce: torch.Tensor) -> dict:
    """Per token minimum over exits, argmin shares and headroom over the last exit.

    A bound, not a policy: the exit is chosen with the true label, which is the main reason it is
    optimistic (the minimum over noisy values is a smaller second one). Ties go to the earliest exit.

    Args:
        ce: ``[T, N]`` per token cross entropy at each exit.
    """
    ce = ce.double()
    T, N = ce.shape
    lowest = ce.min(0).values
    # first index of the minimum: count the exits before the first one equal to it
    is_min = ce == lowest.unsqueeze(0)
    argmin = (is_min.cumsum(0) == 0).sum(0)
    share = torch.bincount(argmin, minlength=T).double() / N
    oracle = _f(lowest.mean())
    return {
        "oracle_ce": oracle,
        "argmin_share": [_f(v) for v in share],
        "headroom": _f(ce[-1].mean()) - oracle,
    }


def beta_optimum(ce: torch.Tensor, beta: float) -> dict:
    """The entropy-regularized exit distribution ``p = softmax_t(-ce / beta)`` per token.

    It chooses with the true label, so it is a bound, not a policy.

    Args:
        ce: ``[T, N]`` per token cross entropy at each exit.
        beta: regularization strength; large is uniform over exits, small is the argmin.

    Returns:
        mean exit share per exit, mean entropy over ``ln T``, expected CE, the share of tokens
        whose largest exit probability exceeds 0.99, and the mean expected exit depth (1-based).
    """
    ce = ce.double()
    T, N = ce.shape
    p = torch.softmax(-ce / beta, dim=0)
    entropy = -(p * torch.log(p.clamp_min(1e-300))).sum(0)
    norm = math.log(T) if T > 1 else 1.0
    depth = torch.arange(1, T + 1, dtype=torch.double).unsqueeze(1)
    return {
        "beta": beta,
        "exit_share": [_f(v) for v in p.mean(1)],
        "entropy_over_ln_t": _f(entropy.mean()) / norm,
        "expected_ce": _f((p * ce).sum(0).mean()),
        "share_max_p_above_0.99": _f((p.max(0).values > 0.99).double().mean()),
        "expected_depth": _f((p * depth).sum(0).mean()),
    }


def interpolated_ce(exit_ce: Sequence[float], passes: float) -> float:
    """CE of a random mix of the two fixed depths neighbouring a mean pass count.

    Args:
        exit_ce: mean CE per exit, index 0 is one pass.
        passes: mean passes (1-based), clamped to ``[1, T]``.
    """
    T = len(exit_ce)
    if T == 1:
        return exit_ce[0]
    passes = min(max(passes, 1.0), float(T))
    lo = min(int(math.floor(passes)), T - 1)
    frac = passes - lo
    return exit_ce[lo - 1] * (1.0 - frac) + exit_ce[lo] * frac


def hull_ce(exit_ce: Sequence[float], passes: float) -> float:
    """Lowest CE a random mix of fixed depths can reach at a mean pass count.

    Mixing depths 1 and 3 can beat mixing neighbours when CE is not convex in depth, so the
    baseline is the lower convex hull of the points ``(t, mean CE_t)`` evaluated at ``passes``.

    Args:
        exit_ce: mean CE per exit, index 0 is one pass.
        passes: mean passes (1-based), clamped to ``[1, T]``.
    """
    T = len(exit_ce)
    if T == 1:
        return exit_ce[0]
    passes = min(max(passes, 1.0), float(T))
    hull: List[Tuple[float, float]] = []
    for point in ((float(t + 1), exit_ce[t]) for t in range(T)):
        while len(hull) >= 2:
            (ax, ay), (bx, by) = hull[-2], hull[-1]
            if (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax) <= 0:
                hull.pop()
            else:
                break
        hull.append(point)
    for (ax, ay), (bx, by) in zip(hull, hull[1:]):
        if passes <= bx:
            return ay + (by - ay) * (passes - ax) / (bx - ax)
    return hull[-1][1]


def confidence_rule(ce: torch.Tensor, p_max: torch.Tensor, tau: float) -> dict:
    """Exit at the first exit whose ``p_max`` is at least ``tau``, else at the last.

    Readout-only: every token still runs all passes, so this CE is optimistic against a real exit.

    Args:
        ce: ``[T, N]`` per token cross entropy at each exit.
        p_max: ``[T, N]`` per token max probability.
        tau: confidence threshold.

    Returns:
        mean CE under the rule, mean passes (1-based), the share of tokens leaving at each exit,
        ``mix_ce`` (the lower hull of fixed depth CE at the same mean passes) and
        ``neighbour_mix_ce`` (the mix of the two neighbouring integer depths).
    """
    ce, p_max = ce.double(), p_max.double()
    T, N = ce.shape
    reached = p_max >= tau
    reached[T - 1] = True
    idx = (reached.cumsum(0) == 0).sum(0)
    chosen = ce.gather(0, idx.unsqueeze(0)).squeeze(0)
    passes = _f((idx + 1).double().mean())
    exit_ce = [_f(v) for v in ce.mean(1)]
    return {
        "tau": tau,
        "ce": _f(chosen.mean()),
        "passes": passes,
        "exit_share": [_f(v) for v in torch.bincount(idx, minlength=T).double() / N],
        "mix_ce": hull_ce(exit_ce, passes),
        "neighbour_mix_ce": interpolated_ce(exit_ce, passes),
    }


def decile_table(ce: torch.Tensor, p_max: torch.Tensor, n_bins: int = N_DECILES) -> dict:
    """Equal count bins by ascending ``p_max`` at the first exit.

    Ties at a bin edge split by token order (a stable sort), so equal ``p_max`` values can land in
    adjacent bins.

    Args:
        ce: ``[T, N]`` per token cross entropy at each exit.
        p_max: ``[T, N]`` per token max probability.
        n_bins: bins (10 for deciles); the first ``N % n_bins`` bins take one extra token.

    Returns:
        ``bins`` (count, mean ``p_max`` at exit 1, mean CE per exit, mean gain first to last, mean
        gain from the second to last exit to the last), ``gain_lowest`` and ``gain_highest`` (the
        first to last gain of the lowest and highest confidence bins) and ``gain_ratio``, their
        ratio, None unless the highest confidence gain is positive.
    """
    ce, p_max = ce.double(), p_max.double()
    T, N = ce.shape
    n_bins = min(n_bins, N)
    order = torch.argsort(p_max[0], stable=True)
    base, extra = divmod(N, n_bins)
    bins, at = [], 0
    for b in range(n_bins):
        size = base + (1 if b < extra else 0)
        rows = order[at:at + size]
        at += size
        c = ce[:, rows]
        bins.append({
            "count": int(size),
            "p_max_first": _f(p_max[0, rows].mean()),
            "ce": [_f(v) for v in c.mean(1)],
            "gain_first_last": _f((c[0] - c[-1]).mean()),
            "gain_last_step": _f((c[T - 2] - c[T - 1]).mean()) if T > 1 else 0.0,
        })
    low, high = bins[0]["gain_first_last"], bins[-1]["gain_first_last"]
    return {"bins": bins, "gain_lowest": low, "gain_highest": high,
            "gain_ratio": low / high if high > 0 else None}


def analyze(ce: torch.Tensor, p_max: torch.Tensor, correct: torch.Tensor,
            betas: Sequence[float], thresholds: Sequence[float],
            clusters: Optional[torch.Tensor] = None) -> dict:
    """Every statistic of the report from the three ``[T, N]`` tensors.

    Args:
        ce: per token cross entropy at each exit.
        p_max: per token max probability at each exit.
        correct: per token top-1 correctness at each exit.
        betas: entropy regularization strengths.
        thresholds: confidence thresholds.
        clusters: ``[N]`` cluster ids (one per document) for the cluster robust standard errors;
            None reads iid only.
    """
    assert ce.dim() == 2 and ce.size(1) > 0, (
        f"no supervised tokens to read (ce shape {tuple(ce.shape)}): check the split and its mask")
    return {
        "n_exits": int(ce.size(0)),
        "n_tokens": int(ce.size(1)),
        "n_clusters": None if clusters is None else int(torch.unique(clusters).numel()),
        "per_exit": per_exit_stats(ce, p_max, correct),
        "gains": paired_gains(ce, clusters),
        "oracle": oracle_stats(ce),
        "beta": [beta_optimum(ce, b) for b in betas],
        "confidence": [confidence_rule(ce, p_max, t) for t in thresholds],
        "deciles": decile_table(ce, p_max),
    }


def _row(values: Sequence[float], fmt: str = "{:.4f}") -> str:
    return " ".join(fmt.format(v) for v in values)


def print_report(report: dict, title: str = "") -> None:
    """Print the report blocks.

    Args:
        report: the dict returned by ``analyze``.
        title: heading for the block.
    """
    T = report["n_exits"]
    clusters = report["n_clusters"]
    print(f"=== exit read {title}: {report['n_tokens']:,} supervised tokens, {T} exits, "
          f"{'no cluster ids' if clusters is None else f'{clusters:,} clusters'} ===")
    pe = report["per_exit"]
    print("per exit (1-based passes)")
    for t in range(T):
        print(f"  exit {t + 1}: ce {pe['ce'][t]:.4f}  top1 {pe['top1'][t]:.4f}  p_max {pe['p_max'][t]:.4f}")
    print("paired gain ce[from] - ce[to] (mean, iid se, cluster se, z from the cluster se when present)")
    for g in report["gains"]["consecutive"] + [report["gains"]["first_to_last"]]:
        cluster_se = "n/a" if g["se_cluster"] is None else f"{g['se_cluster']:.4f}"
        z = "n/a" if g["z"] is None else f"{g['z']:.1f}"
        print(f"  {g['from']} -> {g['to']}: {g['mean']:+.4f}  se iid {g['se_iid']:.4f}  "
              f"se cluster {cluster_se}  z {z}")
    o = report["oracle"]
    print("oracle, per token minimum over exits (BOUND: it picks the exit with the true label, "
          "not a policy)")
    print(f"  oracle ce {o['oracle_ce']:.4f}  headroom over last exit {o['headroom']:.4f}")
    print(f"  argmin share by exit (ties to the earliest): {_row(o['argmin_share'], '{:.3f}')}")
    for b in report["beta"]:
        print(f"entropy regularized optimum, beta {b['beta']:g} (BOUND: chooses with the true label; "
              f"readout-only, every token still runs all passes)")
        print(f"  exit share {_row(b['exit_share'], '{:.3f}')}  entropy/ln T {b['entropy_over_ln_t']:.3f}")
        print(f"  expected ce {b['expected_ce']:.4f}  max p > 0.99 on {b['share_max_p_above_0.99']:.3f} of tokens"
              f"  expected depth {b['expected_depth']:.3f}")
    for c in report["confidence"]:
        print(f"confidence exit rule, tau {c['tau']:g} (readout-only: every token still runs all "
              f"passes, so the ce is optimistic against a real exit)")
        print(f"  ce {c['ce']:.4f}  passes {c['passes']:.3f}  exit share {_row(c['exit_share'], '{:.3f}')}")
        print(f"  fixed depth mix at the same passes: hull {c['mix_ce']:.4f}  "
              f"neighbour depths {c['neighbour_mix_ce']:.4f}")
    d = report["deciles"]
    print("deciles by p_max at exit 1 (ascending): count, p_max1, ce per exit, gain 1->T, gain T-1->T")
    for i, b in enumerate(d["bins"]):
        print(f"  d{i + 1:<2} {b['count']:>8,}  {b['p_max_first']:.4f}  {_row(b['ce'])}"
              f"  {b['gain_first_last']:+.4f}  {b['gain_last_step']:+.4f}")
    ratio = d["gain_ratio"]
    print(f"  gain 1->T lowest confidence decile {d['gain_lowest']:+.4f}, highest {d['gain_highest']:+.4f}, "
          f"ratio {'n/a' if ratio is None else f'{ratio:.2f}'}")


def exit_readouts(hidden: torch.Tensor, labels: torch.Tensor, lm_head, chunk: int = HEAD_CHUNK):
    """The head at every exit on the supervised positions of one batch.

    Position ``i`` predicts ``labels[:, i + 1]``, so the hidden states drop their last position and
    the labels their first, exactly as ``compute_mtp_loss`` shifts them. Positions are taken in
    row-major order and the head runs in chunks so a ``[chunk, vocab]`` fp32 logit tensor is the
    most that is ever live.

    Args:
        hidden: ``[T, B, S, H]`` per exit normed hidden states.
        labels: ``[B, S]`` labels with -100 where unsupervised.
        lm_head: module mapping ``[n, H]`` to ``[n, vocab]`` logits.
        chunk: supervised positions per head call.

    Returns:
        ``ce``, ``p_max``, ``correct`` as fp32 CPU ``[T, n]`` tensors and ``index`` ``[n, 2]``, the
        (row, position) of each token in the batch (position is the input position, not the label's).
    """
    target = labels[:, 1:]
    supervised = target != -100
    index = supervised.nonzero().cpu()
    T = hidden.size(0)
    n = int(supervised.sum().item())
    if n == 0:
        empty = torch.zeros(T, 0)
        return empty, empty.clone(), empty.clone(), index
    h_sel = hidden[:, :, :-1][:, supervised]
    t_sel = target[supervised]
    ce_parts, pm_parts, ok_parts = [], [], []
    for start in range(0, n, chunk):
        tgt = t_sel[start:start + chunk]
        ce_t, pm_t, ok_t = [], [], []
        for t in range(T):
            logits = lm_head(h_sel[t, start:start + chunk]).float()
            lse = torch.logsumexp(logits, dim=-1)
            top, arg = logits.max(-1)
            ce_t.append(lse - logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
            pm_t.append(torch.exp(top - lse))
            ok_t.append((arg == tgt).float())
            del logits
        ce_parts.append(torch.stack(ce_t).cpu())
        pm_parts.append(torch.stack(pm_t).cpu())
        ok_parts.append(torch.stack(ok_t).cpu())
    return torch.cat(ce_parts, 1), torch.cat(pm_parts, 1), torch.cat(ok_parts, 1), index


def collect_exits(model, dataset, device: str, pad_token_id: int, max_batches: int):
    """One full depth forward per batch, the head applied at every exit to the supervised positions.

    Args:
        model: a model from ``scripts.eval_abstention.load_model``.
        dataset: an ``SFTDataset`` (or any reader yielding ``input_ids``/``labels``/``document_ids``).
        device: the model's device.
        pad_token_id: id of the padding token.
        max_batches: batches to read; a batch with no supervised token does not count.

    Returns:
        ``ce``, ``p_max`` and ``correct`` as fp32 CPU ``[T, N]`` tensors and ``clusters`` ``[N]``,
        one id per (row of the run, document) pair.
    """
    import transformer_engine.pytorch as te
    from modules.model.attention import cu_seqlens_from_doc_ids
    from scripts.pretrain import USE_LOW_PRECISION, chosen_recipe

    ce_parts, pm_parts, ok_parts, cluster_parts = [], [], [], []
    n_batches, rows_seen = 0, 0
    with torch.no_grad():
        for batch in dataset:
            if n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            document_ids = batch["document_ids"].to(device)
            if not (labels[:, 1:] != -100).any():
                continue
            cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
            pad_mask = input_ids == pad_token_id
            with te.autocast(enabled=USE_LOW_PRECISION, recipe=chosen_recipe):
                out = model(input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                            return_hidden=True, skip_mtp=True, token_mask=~pad_mask)
                hidden = out[0] if isinstance(out, tuple) else out
                ce, pm, ok, index = exit_readouts(hidden, labels, model.lm_head)
            docs = document_ids.cpu()[index[:, 0], index[:, 1]].long()
            cluster_parts.append((rows_seen + index[:, 0]).long() * CLUSTER_STRIDE + docs)
            ce_parts.append(ce)
            pm_parts.append(pm)
            ok_parts.append(ok)
            rows_seen += labels.size(0)
            n_batches += 1
            del out, hidden, labels, input_ids, document_ids, cu_seqlens, pad_mask
    assert ce_parts, "no supervised tokens were read"
    return (torch.cat(ce_parts, 1), torch.cat(pm_parts, 1), torch.cat(ok_parts, 1),
            torch.cat(cluster_parts))


def _floats(text: str) -> List[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=40)
    parser.add_argument("--betas", default="0.05,0.1")
    parser.add_argument("--thresholds", default="0.5,0.7,0.9,0.99")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from config import SFTConfig

    data_dir = os.path.join(BASE_DIR, args.data_dir or SFTConfig.data_dir)
    if os.path.isfile(os.path.join(data_dir, f"{args.split}.ev")):
        raise SystemExit(
            f"{args.split} carries evidence ({args.split}.ev): this read attaches none and the "
            f"evidence packing differs from the SFT packing, so its CE would not match the trainer's "
            f"[eval]. Read a split without evidence."
        )

    from transformers import AutoTokenizer
    from scripts.eval_abstention import load_model
    from scripts.sft import make_dataset

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    cfg = type("ExitConfig", (SFTConfig,), {"Batch_size": args.batch_size or SFTConfig.Batch_size})
    dataset = make_dataset(data_dir, args.split, tokenizer, cfg, shuffle=False)

    path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(BASE_DIR, args.checkpoint)
    model = load_model(path, args.device)
    ce, p_max, correct, clusters = collect_exits(model, dataset, args.device, tokenizer.pad_token_id,
                                                 args.max_batches)
    logger.info(f"{ce.size(1):,} supervised tokens, {ce.size(0)} exits")

    report = analyze(ce, p_max, correct, _floats(args.betas), _floats(args.thresholds), clusters)
    print_report(report, f"{os.path.basename(args.checkpoint)} {args.split}")
    if args.json_out:
        out = args.json_out if os.path.isabs(args.json_out) else os.path.join(BASE_DIR, args.json_out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"flags": vars(args), "report": report}, f)
        logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()

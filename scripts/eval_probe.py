"""The answerability probe: is "this question is unanswerable" in the trunk's representation at all?

Two checkpoints' worth of abstention work has established that the *policy* cannot tell answerable
from unanswerable questions -- the corpus ratio slides the operating point along a fixed curve, and
``p_max`` carries no usable discrimination (AUROC 0.462-0.478 across three checkpoints, and its sign
is not even stable across a 2,000-question slice). What that record does not say is *why*, and the
two possible reasons ask completely different things of the phases that follow:

  * **the trunk does not represent it** -- then only new information can create the signal, and no
    readout trick, threshold or loss reweighting will. Retrieval-grounded evidence is the only
    principled candidate.
  * **the trunk represents it but the policy cannot read it out** -- then a thresholded probe is a
    legitimate shipping mechanism on its own (tunable operating point, minutes to refit), and a
    preference pass has something real to amplify rather than something to invent.

A linear probe separates the two. Fit logistic regression on the final loop's last-position hidden
state (plus the free scalars that come with it) over SQuAD v2 **train** questions, and read AUROC on
the same standard 2,000-question **validation** slice every abstention number in this project is
quoted against. A linear probe is a *lower* bound on what the representation carries -- it finds
only what is linearly decodable -- which is the right direction for this question: a pass says the
information is there, and says it cheaply.

What is measured
----------------

The features are read at the **last prompt position**, which is the state the model would decode its
first answer token from -- not an average over the passage. Three feature sets are fitted separately
so the answer is attributable:

  * ``scalars`` -- ``p_max``, the predictive entropy and the top-1/top-2 log-probability margin.
    Three numbers that are free at decode time; if these alone score, the policy fix is trivial.
  * ``hidden`` -- the 768-d hidden state on its own.
  * ``hidden+scalars`` -- both, which is the plan's stated feature set.

The L2 strength is chosen on a held-out slice of the **train** questions, never on the evaluation
slice, and the chosen model is then refit on all of them. Tuning on the eval slice would report a
number that the next checkpoint cannot reproduce.

Caveats that belong next to the number
--------------------------------------

  * ``squad_v2/train`` is what ``prepare_sft_data.py`` builds the SFT and repair corpora from, so
    the checkpoint has *trained on* the questions the probe is fitted on. That does not leak into
    the reported AUROC -- it is read on held-out validation questions -- but it does mean the probe
    is fitted on a distribution the model has seen, which if anything flatters the fit and makes a
    low AUROC the stronger of the two results.
  * Batch size is part of the measurement, as everywhere else here: ``ParallelSparseMoELayer`` tiles
    its grouped GEMM by per-expert row counts computed over every token in the batch, padding
    included, so a different ``--batch-size`` perturbs the features at the ~0.5-1% level.

Run from the repo root:

```bash
python scripts/eval_probe.py -c ckpts/repair/checkpoint_repair_final.pt \\
  --json-out docs/measurements/probe_repair_055.json
```
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import numpy as np
from transformers import AutoTokenizer

from modules.data.chat import ChatTemplate
from config import ModelConfig, SFTConfig
from scripts.eval_calibration import roc_auc
from scripts.eval_abstention import (
    SFT_CHECKPOINT_DIR, build_records, find_latest_checkpoint, load_model, load_squad_split,
    _final_hidden,
)
from utils import BASE_DIR, TOKENIZER_DIR, get_hf_token, logger

# decades: with standardized features the useful range spans several orders of magnitude and the
# dev split picks within it, so a coarse grid costs nothing and avoids pretending to a precision
# 2,000 questions cannot support
L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
FEATURE_SETS = ("scalars", "hidden", "hidden+scalars")
# measured eval-sampling spread on an AUROC over a 2,000-question slice (docs/measurements/
# noise_floor.md). Two AUROCs closer than this are the same reading.
AUROC_NOISE = 0.030


# --------------------------------------------------------------------------- features


@torch.inference_mode()
def collect_features(model, records: List[dict], *, batch_size: int, pad_id: int, device: str,
                     max_seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final-loop hidden state and readout scalars at each prompt's last position.

    Left-padded with the pad run given its own ``document_ids`` segment, exactly as
    ``eval_abstention.generate_batch`` decodes: left padding puts every row's last real token in the
    same column (so one slice reads them all) and the separate segment is what stops a real token
    ever attending to a pad. Rows are length-sorted into batches and written back through the
    returned index, so padding stays small without reordering the output.

    Args:
        model: a loaded ``TinyMoETransformer`` in eval mode.
        records: ``eval_abstention.build_records`` output; only ``prompt_ids`` is read.
        batch_size: rows per forward. Part of the measurement -- see this module's docstring.
        pad_id: the tokenizer's pad id (== eos id for this tokenizer).
        device: torch device string.
        max_seq_len: the model's context; prompts are already capped below it by build_records.

    Returns:
        ``(hidden, scalars, labels)`` -- ``[N, H]`` float32, ``[N, 3]`` float32 (``p_max``,
        entropy, top-1/top-2 log-probability margin) and ``[N]`` float64 with 1.0 = unanswerable.
    """
    n = len(records)
    hidden_dim = ModelConfig.Params["hidden_size"]
    hidden = np.zeros((n, hidden_dim), dtype=np.float32)
    scalars = np.zeros((n, 3), dtype=np.float32)

    order = sorted(range(n), key=lambda i: len(records[i]["prompt_ids"]))
    done = 0
    for start in range(0, n, batch_size):
        rows = order[start:start + batch_size]
        prompts = [records[i]["prompt_ids"][-max_seq_len:] for i in rows]
        width = max(len(p) for p in prompts)
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
        doc = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        for r, prompt in enumerate(prompts):
            ids[r, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
            doc[r, width - len(prompt):] = 1

        h_last = _final_hidden(model, ids, doc)[:, -1, :]        # [B, H]
        logits = model.lm_head(h_last).float()                   # [B, vocab], one position per row
        logprobs = logits.log_softmax(-1)
        top2 = logprobs.topk(2, dim=-1).values
        # torch.special.entr is -x log x in one kernel, the same form the training-time retrieval
        # entropy tracker uses
        entropy = torch.special.entr(logprobs.exp()).sum(-1)

        hidden[rows] = h_last.float().cpu().numpy()
        scalars[rows, 0] = top2[:, 0].exp().cpu().numpy()        # p_max
        scalars[rows, 1] = entropy.cpu().numpy()
        scalars[rows, 2] = (top2[:, 0] - top2[:, 1]).cpu().numpy()

        done += len(rows)
        if done % (batch_size * 25) < batch_size:
            logger.info(f"[eval_probe] featurized {done:,}/{n:,} questions")

    labels = np.array([float(r["unanswerable"]) for r in records], dtype=np.float64)
    return hidden, scalars, labels


def assemble(hidden: np.ndarray, scalars: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set == "scalars":
        return scalars
    if feature_set == "hidden":
        return hidden
    return np.concatenate([hidden, scalars], axis=1)


# ---------------------------------------------------------------------------- probe


def fit_logistic(x: torch.Tensor, y: torch.Tensor, l2: float, iters: int = 300
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """L2-regularized logistic regression by full-batch LBFGS.

    Full batch rather than SGD because the whole design matrix is a few tens of MB and a
    deterministic optimum is what makes two feature sets comparable -- an SGD run would add a seed
    to a measurement whose entire purpose is to say what is *in* the features. The bias is
    unpenalized, so the fit can still match the class prior at any regularization strength.

    Args:
        x: ``[N, D]`` float32 design matrix, already standardized.
        y: ``[N]`` float32 targets in {0, 1}.
        l2: penalty on the weight vector's squared L2 norm, added to the mean BCE.
        iters: LBFGS iterations (strong-Wolfe line search; convergence is well inside this).

    Returns:
        ``(w, b)`` -- ``[D]`` weights and a scalar bias.
    """
    w = torch.zeros(x.size(1), dtype=torch.float32, device=x.device, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=iters, history_size=20,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(x @ w + b, y) + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach(), b.detach()


def standardize(fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Column mean/std from the fit set only. A zero-variance column gets std 1 (it becomes a
    constant 0 feature the bias absorbs) rather than a division by zero."""
    mean = fit.mean(axis=0)
    std = fit.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def score(x: np.ndarray, mean: np.ndarray, std: np.ndarray, w: torch.Tensor, b: torch.Tensor,
          device: str) -> np.ndarray:
    z = torch.from_numpy(((x - mean) / std).astype(np.float32)).to(device)
    return (z @ w + b).sigmoid().double().cpu().numpy()


def fit_and_score(train_x: np.ndarray, train_y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray,
                  eval_x: np.ndarray, device: str) -> dict:
    """Pick L2 on the dev split, refit on train+dev, score the eval slice.

    The refit matters: the L2 chosen on a 80% fit would be mistuned for the 100% one only mildly,
    but throwing away a fifth of the fit data after using it purely to choose one scalar is waste,
    and the eval slice never enters either step.
    """
    mean, std = standardize(train_x)
    z_fit = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(device)
    y_fit = torch.from_numpy(train_y.astype(np.float32)).to(device)

    best = None
    for l2 in L2_GRID:
        w, b = fit_logistic(z_fit, y_fit, l2)
        dev_auroc = roc_auc(score(dev_x, mean, std, w, b, device), dev_y)
        if best is None or dev_auroc > best["dev_auroc"]:
            best = {"l2": l2, "dev_auroc": dev_auroc}

    full_x = np.concatenate([train_x, dev_x], axis=0)
    full_y = np.concatenate([train_y, dev_y], axis=0)
    mean, std = standardize(full_x)
    z_full = torch.from_numpy(((full_x - mean) / std).astype(np.float32)).to(device)
    y_full = torch.from_numpy(full_y.astype(np.float32)).to(device)
    w, b = fit_logistic(z_full, y_full, best["l2"])

    return {
        "l2": best["l2"],
        "dev_auroc": best["dev_auroc"],
        "fit_auroc": roc_auc(score(full_x, mean, std, w, b, device), full_y),
        "eval_scores": score(eval_x, mean, std, w, b, device),
        "n_features": train_x.shape[1],
    }


def operating_points(scores: np.ndarray, unanswerable: np.ndarray) -> List[dict]:
    """Precision/recall/false-abstention of "probe says unanswerable" over a threshold sweep.

    The same three numbers ``eval_abstention.py`` reports for the model's own abstention decision,
    so a probe threshold and the shipped policy are read on one scale. Thresholds are the score
    quantiles rather than a fixed grid -- a probe whose scores all sit in a narrow band would
    otherwise show one populated row and eleven empty ones.
    """
    positives = unanswerable.astype(bool)
    n_answerable = float((~positives).sum())
    rows = []
    for quantile in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        threshold = float(np.quantile(scores, quantile))
        flagged = scores >= threshold
        tp = float((flagged & positives).sum())
        fp = float((flagged & ~positives).sum())
        fn = float((~flagged & positives).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        rows.append({
            "threshold": threshold,
            "flag_rate": float(flagged.mean()),
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision + recall else float("nan")),
            "false_abstention_rate": fp / n_answerable if n_answerable else float("nan"),
        })
    return rows


# ----------------------------------------------------------------------------- report


def report(results: Dict[str, dict], eval_labels: np.ndarray, eval_scalars: np.ndarray,
           n_train: int, n_dev: int, offset: int, checkpoint: str) -> None:
    n_unanswerable = int(eval_labels.sum())
    print("\n=== Answerability probe (positive class: unanswerable) ===")
    print(f"  checkpoint: {os.path.basename(checkpoint)}")
    print(f"  fitted on {n_train:,} squad_v2/train questions "
          f"({n_dev:,} of them held out to choose the L2 penalty)")
    print(f"  read on the standard slice: validation questions {offset:,}-"
          f"{offset + len(eval_labels):,}  ({n_unanswerable:,} unanswerable, "
          f"{len(eval_labels) - n_unanswerable:,} answerable)")

    print(f"\n  {'feature set':<16} {'n_feat':>7} {'L2':>8} {'fit':>8} {'dev':>8} {'SLICE':>8}")
    for name in FEATURE_SETS:
        r = results[name]
        print(f"  {name:<16} {r['n_features']:>7} {r['l2']:>8.0e} {r['fit_auroc']:>8.4f} "
              f"{r['dev_auroc']:>8.4f} {r['eval_auroc']:>8.4f}")

    # the unfitted references on the identical slice: what the shipped policy already has
    p_max = eval_scalars[:, 0].astype(np.float64)
    entropy = eval_scalars[:, 1].astype(np.float64)
    margin = eval_scalars[:, 2].astype(np.float64)
    print(f"\n  {'unfitted reference':<16} {'':>7} {'':>8} {'':>8} {'':>8} {'SLICE':>8}")
    unfitted = []
    for label, signal in (("1 - p_max", -p_max), ("entropy", entropy), ("-margin", -margin)):
        auroc = roc_auc(signal, eval_labels)
        unfitted.append(auroc)
        print(f"  {label:<16} {'':>7} {'':>8} {'':>8} {'':>8} {auroc:>8.4f}")

    best_name = max(FEATURE_SETS, key=lambda k: results[k]["eval_auroc"])
    best = results[best_name]
    print(f"\n=== Operating points, {best_name} probe, on the same slice ===")
    print(f"  {'flag rate':>10} {'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>8} "
          f"{'false abst.':>12}")
    for row in best["operating_points"]:
        print(f"  {row['flag_rate']:>10.3f} {row['threshold']:>10.4f} {row['precision']:>10.4f} "
              f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['false_abstention_rate']:>12.4f}")

    auroc = best["eval_auroc"]
    reference = max(unfitted)
    print("\n=== Verdict ===")
    if auroc >= 0.70:
        verdict = ("READOUT PROBLEM -- the trunk carries the signal and the policy does not read it. "
                   "A thresholded probe is a legitimate shipping mechanism on its own, and a "
                   "preference pass has something real to amplify.")
    elif auroc >= 0.60:
        verdict = ("PARTIAL -- the representation carries real signal but not enough to ship as the "
                   "whole abstention story. Both a readout mechanism and new information stay in "
                   "play.")
    elif auroc - reference >= AUROC_NOISE:
        # AUROC_NOISE is the measured slice-to-slice spread; a gap inside it is not a reading
        verdict = ("WEAK -- a linear read of the trunk beats every free confidence scalar by more "
                   "than the measured slice noise, so the signal exists and the policy is not using "
                   "it, but it is far below a shippable readout. New information stays the main "
                   "lever; a threshold is worth quoting only against the shipped operating point.")
    else:
        verdict = ("NOT IN THE REPRESENTATION -- a linear read of the trunk is no better than the "
                   "confidence scalars it already has. Only new information can create this signal; "
                   "no readout trick will.")
    print(f"  best AUROC {auroc:.4f} ({best_name}) vs {reference:.4f} for the best unfitted "
          f"scalar, noise +-{AUROC_NOISE:.3f}")
    print(f"  {verdict}")


def main():
    parser = argparse.ArgumentParser(
        description="linear answerability probe on SQuAD v2 (does the trunk represent it?)")
    parser.add_argument("--checkpoint", "-c", default=find_latest_checkpoint(SFT_CHECKPOINT_DIR),
                        help="checkpoint to probe (default: newest in ckpts/sft)")
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--squad-train-dir", default=None,
                        help="directory of local squad_v2 TRAIN parquet shards (skips the Hub)")
    parser.add_argument("--squad-dir", default=None,
                        help="directory of local squad_v2 VALIDATION parquet shards (skips the Hub)")
    parser.add_argument("--train-examples", type=int, default=10000,
                        help="train questions to fit on (seeded subsample)")
    parser.add_argument("--eval-examples", type=int, default=2000,
                        help="validation questions to read AUROC on -- the standard slice")
    parser.add_argument("--example-offset", type=int, default=0,
                        help="skip this many usable validation questions first, as in eval_abstention")
    parser.add_argument("--dev-fraction", type=float, default=0.2,
                        help="share of the train questions held out to choose the L2 penalty")
    parser.add_argument("--max-prompt-tokens", type=int, default=1024,
                        help="drop questions whose rendered prompt exceeds this (never truncated)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="part of the measurement -- keep it fixed across runs you compare")
    parser.add_argument("--seed", type=int, default=SFTConfig.seed,
                        help="seeds both subsample shuffles and the train/dev split")
    parser.add_argument("--json-out", default=None, help="write the full result payload here")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint is None:
        raise SystemExit(f"No checkpoint found in {SFT_CHECKPOINT_DIR} and none passed via --checkpoint")

    logger.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = ChatTemplate(tokenizer)
    hf_token = args.hf_token or get_hf_token()
    benchmarks = os.path.join(BASE_DIR, "data", "benchmarks")

    # the eval slice must be the one every other abstention number is quoted against: same loader,
    # same renderer, same seeded shuffle, same flags
    logger.info("Rendering the evaluation slice (squad_v2/validation)")
    eval_frame = load_squad_split(os.path.join(benchmarks, "squad_v2_validation"), hf_token,
                                  args.squad_dir, split="validation")
    eval_records = build_records(eval_frame, template, max_examples=args.eval_examples,
                                 max_prompt_tokens=args.max_prompt_tokens, seed=args.seed,
                                 with_forced=False, offset=args.example_offset)
    logger.info("Rendering the fit set (squad_v2/train)")
    train_frame = load_squad_split(os.path.join(benchmarks, "squad_v2_train"), hf_token,
                                   args.squad_train_dir, split="train")
    train_records = build_records(train_frame, template, max_examples=args.train_examples,
                                  max_prompt_tokens=args.max_prompt_tokens, seed=args.seed,
                                  with_forced=False)
    if not eval_records or not train_records:
        raise SystemExit("no usable questions -- check --max-prompt-tokens / --squad-dir")

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)
    max_seq_len = ModelConfig.Params["max_seq_len"]

    logger.info(f"Featurizing {len(train_records):,} train questions")
    train_hidden, train_scalars, train_labels = collect_features(
        model, train_records, batch_size=args.batch_size, pad_id=tokenizer.pad_token_id,
        device=args.device, max_seq_len=max_seq_len)
    logger.info(f"Featurizing {len(eval_records):,} evaluation questions")
    eval_hidden, eval_scalars, eval_labels = collect_features(
        model, eval_records, batch_size=args.batch_size, pad_id=tokenizer.pad_token_id,
        device=args.device, max_seq_len=max_seq_len)

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(train_records))
    n_dev = int(round(len(train_records) * args.dev_fraction))
    dev_idx, fit_idx = permutation[:n_dev], permutation[n_dev:]

    results = {}
    for name in FEATURE_SETS:
        x_train = assemble(train_hidden, train_scalars, name)
        x_eval = assemble(eval_hidden, eval_scalars, name)
        outcome = fit_and_score(x_train[fit_idx], train_labels[fit_idx],
                                x_train[dev_idx], train_labels[dev_idx], x_eval, args.device)
        outcome["eval_auroc"] = roc_auc(outcome["eval_scores"], eval_labels)
        outcome["operating_points"] = operating_points(outcome["eval_scores"], eval_labels)
        results[name] = outcome
        logger.info(f"[eval_probe] {name}: dev AUROC {outcome['dev_auroc']:.4f}, "
                    f"slice AUROC {outcome['eval_auroc']:.4f} (L2 {outcome['l2']:.0e})")

    report(results, eval_labels, eval_scalars, len(train_records), n_dev,
           args.example_offset, args.checkpoint)

    if args.json_out:
        payload = {
            "checkpoint": os.path.relpath(args.checkpoint, BASE_DIR),
            "flags": {
                "train_examples": len(train_records), "eval_examples": len(eval_records),
                "example_offset": args.example_offset, "dev_fraction": args.dev_fraction,
                "batch_size": args.batch_size, "max_prompt_tokens": args.max_prompt_tokens,
                "seed": args.seed, "l2_grid": list(L2_GRID),
            },
            "eval_unanswerable": int(eval_labels.sum()),
            "unfitted": {
                "one_minus_p_max": roc_auc(-eval_scalars[:, 0].astype(np.float64), eval_labels),
                "entropy": roc_auc(eval_scalars[:, 1].astype(np.float64), eval_labels),
                "neg_margin": roc_auc(-eval_scalars[:, 2].astype(np.float64), eval_labels),
            },
            "probes": {
                name: {k: v for k, v in r.items() if k != "eval_scores"}
                for name, r in results.items()
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

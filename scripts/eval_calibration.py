"""Gate 5 calibration sanity check (PLAN.md's "Local validation gates").

Runs a checkpoint once (full configured loop depth, no gradient) over a held-out slice of the
training corpus and reports the numbers Gate 5 asks for:

  - ECE of ``p_correct`` -- the load-bearing number for the whole abstention thesis (< 0.15 to pass).
  - ``p_correct`` vs ``p_max``: ECE and abstention AUROC side by side, same slice. If ``p_correct``
    doesn't beat ``p_max`` on both, PLAN.md Step 4b says revert (drop the head/loss term/lambda_conf)
    and use ``p_max`` everywhere downstream instead.
  - Early-exit degradation curve: perplexity read out at every loop depth. A single forward pass at
    the configured (full) ``n_loops`` already contains this -- ``forward_step``'s per-loop update
    only depends on the *absolute* loop index and the previous loop's hidden state (see
    ``LoopMixtureOfExperts.forward``: ``for loop in range(n_loops): forward_step(..., loop_idx=loop)``),
    so loop ``k``'s hidden state in a full run is bit-identical to what an ``n_loops=k+1`` override
    would produce. No need to re-run the model once per depth.
  - ``p_halt``/``p_correct`` correlation on the final loop (expect weak; >~0.8 means one head
    collapsed into the other).

Held-out slice: documents from the checkpoint's own ``global_offset`` onward in the same
``{phase}.bin``/``.idx`` pair it trained on. The dataset is read once, sequentially, with no
shuffling (PLAN.md Step 9) and ``global_offset`` is exactly "the smallest next document any worker
still wanted" at checkpoint time, so this slice is (up to the same few-document resume slop the
training loop itself already accepts) never seen during training -- no separate held-out file needed.
"""
import os
import sys
import math
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.data.dataset import Dataset
from config import ModelConfig, TrainingConfig
from utils import BASE_DIR, BF16, logger, TOKENIZER_DIR

CE_CHUNK_SIZE = 2048


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    best_ts, best_path = 0, None
    if not os.path.isdir(checkpoint_dir):
        return None
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("checkpoint") and fname.endswith(".pt"):
            fpath = os.path.join(checkpoint_dir, fname)
            ts = os.path.getmtime(fpath)
            if ts > best_ts:
                best_ts, best_path = ts, fpath
    return best_path


def load_model(checkpoint_path: str, device: str):
    model = TinyMoETransformer(**ModelConfig.Params).to(device).to(BF16)
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # legacy (pre Step 9) checkpoints have no global_offset -- see utils.load_checkpoint
    global_offset = ckpt.get("global_offset", 0)
    return model, global_offset


def _average_ranks(sorted_scores: np.ndarray) -> np.ndarray:
    """average rank (1-indexed) for each position in an already-sorted array, tied values sharing
    the mean of their rank range -- required for a correct Mann-Whitney U/AUROC under ties."""
    n = len(sorted_scores)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = ranks[i:j + 1].mean()
        i = j + 1
    return ranks


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank-sum (Mann-Whitney U) identity -- no sklearn/scipy dependency, neither of
    which is otherwise used in this repo. ``labels`` in {0, 1}; higher ``scores`` should mean more
    likely label==1 (here: more likely correct)."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = _average_ranks(scores[order])
    sum_ranks_pos = ranks[labels == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 15) -> float:
    """standard binned ECE: mean absolute gap between confidence and accuracy, weighted by bin
    occupancy. ``correctness`` is 1.0 where the prediction was right, 0.0 otherwise."""
    n = len(confidences)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correctness[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _chunked_eval(lm_head, hidden: torch.Tensor, labels: torch.Tensor, correct_proj=None, chunk_size: int = CE_CHUNK_SIZE):
    """no_grad chunked CE + (on the final loop) per-token diagnostics.

    Bounds logit memory to ``chunk_size * vocab`` like ``mtp.py``'s training-time chunked CE, but
    -- unlike that one -- also collects the raw per-token arrays onto the host: ECE/AUROC need the
    full distribution, not a running sum, and this runs once per eval rather than once per training
    step, so materializing a few million floats on CPU is cheap.
    """
    T = hidden.size(0)
    ce_sum, n_valid = 0.0, 0
    p_correct_chunks, p_max_chunks, is_correct_chunks, valid_idx_chunks = [], [], [], []
    for start in range(0, T, chunk_size):
        h = hidden[start:start + chunk_size]
        l = labels[start:start + chunk_size]
        valid = l != -100
        if not valid.any():
            continue
        logits = lm_head(h).float()
        ce_sum += F.cross_entropy(logits[valid], l[valid], reduction="sum").item()
        n_valid += int(valid.sum().item())
        if correct_proj is not None:
            argmax = logits.argmax(-1)
            is_correct = (argmax == l).float()
            p_max = logits.softmax(-1).max(-1).values
            p_correct = torch.sigmoid(correct_proj(h).float().squeeze(-1))
            p_correct_chunks.append(p_correct[valid].cpu().numpy())
            p_max_chunks.append(p_max[valid].cpu().numpy())
            is_correct_chunks.append(is_correct[valid].cpu().numpy())
            valid_idx_chunks.append(torch.nonzero(valid, as_tuple=False).squeeze(-1) + start)
    extra = None
    if correct_proj is not None:
        extra = {
            "p_correct": np.concatenate(p_correct_chunks) if p_correct_chunks else np.zeros(0),
            "p_max": np.concatenate(p_max_chunks) if p_max_chunks else np.zeros(0),
            "is_correct": np.concatenate(is_correct_chunks) if is_correct_chunks else np.zeros(0),
            "valid_idx": torch.cat(valid_idx_chunks) if valid_idx_chunks else torch.zeros(0, dtype=torch.long, device=hidden.device),
        }
    return ce_sum, n_valid, extra


@torch.no_grad()
def collect_stats(model: TinyMoETransformer, dataset: Dataset, tokenizer, device: str, max_batches: int | None):
    n_loops = ModelConfig.Params["n_loops"]
    per_loop_ce_sum = [0.0] * n_loops
    per_loop_count = [0] * n_loops
    final_p_correct, final_p_max, final_is_correct, final_p_halt = [], [], [], []

    n_batches = 0
    for batch in dataset:
        if max_batches is not None and n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        document_ids = batch["document_ids"].to(device)
        labels = batch["labels"].to(device)
        cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)

        out = model(
            input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            return_aux_loss=True, return_hidden=True,
        )
        # return_aux_loss=True always yields (hidden, aux_loss, p_halt, ...) -- a 4th MTP element
        # only when has_mtp, which we don't need here.
        hidden_all, aux_loss, p_halt = out[0], out[1], out[2]
        # hidden_all: [n_loops, B, S, H] post-norm; p_halt: [n_loops, B, S]

        main_labels = labels[:, 1:].contiguous().view(-1)
        for loop in range(hidden_all.size(0)):
            h = hidden_all[loop, :, :-1, :].contiguous().view(-1, hidden_all.size(-1))
            is_final = loop == hidden_all.size(0) - 1
            ce_sum, n_valid, extra = _chunked_eval(
                model.lm_head, h, main_labels,
                correct_proj=model.correct_proj if is_final else None,
            )
            per_loop_ce_sum[loop] += ce_sum
            per_loop_count[loop] += n_valid
            if is_final and extra is not None and extra["valid_idx"].numel() > 0:
                final_p_correct.append(extra["p_correct"])
                final_p_max.append(extra["p_max"])
                final_is_correct.append(extra["is_correct"])
                p_halt_final_flat = p_halt[-1, :, :-1].contiguous().view(-1)
                final_p_halt.append(p_halt_final_flat[extra["valid_idx"]].float().cpu().numpy())

        n_batches += 1
        if n_batches % 20 == 0:
            logger.info(f"[eval_calibration] processed {n_batches} batches ({sum(per_loop_count) // n_loops:,} tokens)")

    per_loop_ppl = [math.exp(s / max(c, 1)) for s, c in zip(per_loop_ce_sum, per_loop_count)]
    per_loop_ce = [s / max(c, 1) for s, c in zip(per_loop_ce_sum, per_loop_count)]
    final = {
        "p_correct": np.concatenate(final_p_correct) if final_p_correct else np.zeros(0),
        "p_max": np.concatenate(final_p_max) if final_p_max else np.zeros(0),
        "is_correct": np.concatenate(final_is_correct) if final_is_correct else np.zeros(0),
        "p_halt": np.concatenate(final_p_halt) if final_p_halt else np.zeros(0),
    }
    return per_loop_ce, per_loop_ppl, final, n_batches


def main():
    parser = argparse.ArgumentParser(description="PLAN.md Gate 5 -- calibration sanity check")
    parser.add_argument("--checkpoint", "-c", default=find_latest_checkpoint(os.path.join(BASE_DIR, "ckpts", "training")))
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, TrainingConfig.data_dir))
    parser.add_argument("--phase", default=TrainingConfig.phase)
    parser.add_argument("--start-doc-idx", type=int, default=None, help="override the checkpoint's own global_offset (held-out start point)")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.Batch_size)
    parser.add_argument("--max-batches", type=int, default=None, help="cap eval to this many batches (default: whole held-out remainder)")
    parser.add_argument("--n-bins", type=int, default=15, help="ECE histogram bins")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint is None:
        raise SystemExit("No checkpoint found in ckpts/training and none passed via --checkpoint")

    logger.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model, checkpoint_offset = load_model(args.checkpoint, args.device)
    start_doc_idx = args.start_doc_idx if args.start_doc_idx is not None else checkpoint_offset
    logger.info(f"Held-out slice starts at doc {start_doc_idx:,} (checkpoint global_offset={checkpoint_offset:,})")

    dataset = Dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=ModelConfig.Params["max_seq_len"],
        split=args.phase,
        num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
        start_doc_idx=start_doc_idx,
    )
    if start_doc_idx >= dataset.num_docs:
        raise SystemExit(
            f"start_doc_idx ({start_doc_idx:,}) >= num_docs ({dataset.num_docs:,}) in {args.phase} -- "
            "nothing held out. Train less, prepare more data, or point --phase at phase2."
        )

    per_loop_ce, per_loop_ppl, final, n_batches = collect_stats(model, dataset, tokenizer, args.device, args.max_batches)
    n_tokens = len(final["p_correct"])
    if n_tokens == 0:
        raise SystemExit("No supervised tokens collected on the held-out slice -- check start_doc_idx/max_batches.")

    logger.info(f"Evaluated {n_batches} batches, {n_tokens:,} final-loop supervised tokens")

    print("\n=== Early-exit degradation curve (PLAN.md Gate 5) ===")
    for loop, (ce, ppl) in enumerate(zip(per_loop_ce, per_loop_ppl)):
        print(f"  loop {loop + 1}/{len(per_loop_ce)}: CE={ce:.4f}  perplexity={ppl:.3f}")
    monotone = all(per_loop_ppl[i] > per_loop_ppl[i + 1] for i in range(len(per_loop_ppl) - 1))
    print(f"  monotone improvement across loops: {'PASS' if monotone else 'FAIL'}"
          f"{'' if len(per_loop_ppl) > 1 else ' (n_loops=1, nothing to compare)'}")

    ece_correct = expected_calibration_error(final["p_correct"], final["is_correct"], args.n_bins)
    ece_max = expected_calibration_error(final["p_max"], final["is_correct"], args.n_bins)
    auroc_correct = roc_auc(final["p_correct"], final["is_correct"])
    auroc_max = roc_auc(final["p_max"], final["is_correct"])
    top1_acc = final["is_correct"].mean()

    print("\n=== p_correct vs p_max (final loop) ===")
    print(f"  batch top-1 accuracy: {top1_acc:.4f}")
    print(f"  {'signal':<10} {'mean':>8} {'ECE':>8} {'AUROC':>8}")
    print(f"  {'p_correct':<10} {final['p_correct'].mean():>8.4f} {ece_correct:>8.4f} {auroc_correct:>8.4f}")
    print(f"  {'p_max':<10} {final['p_max'].mean():>8.4f} {ece_max:>8.4f} {auroc_max:>8.4f}")

    ece_pass = ece_correct < 0.15
    beats_max = (ece_correct < ece_max) and (auroc_correct > auroc_max)
    print(f"\n  ECE(p_correct) < 0.15: {'PASS' if ece_pass else 'FAIL'} ({ece_correct:.4f})")
    print(f"  p_correct beats p_max on both ECE and AUROC: {'PASS' if beats_max else 'FAIL (revert Step 4b, use p_max)'}")

    corr = float(np.corrcoef(final["p_halt"], final["p_correct"])[0, 1]) if n_tokens > 1 else float("nan")
    print(f"\n=== p_halt / p_correct correlation (final loop) ===")
    print(f"  mean p_halt: {final['p_halt'].mean():.4f}")
    print(f"  correlation: {corr:.4f}  ({'PASS (weak, as expected)' if abs(corr) < 0.8 else 'WARN: heads may have collapsed together'})")

    print("\n=== Gate 5 summary ===")
    print(f"  ECE(p_correct) < 0.15:              {'PASS' if ece_pass else 'FAIL'}")
    print(f"  p_correct beats p_max (ECE & AUROC): {'PASS' if beats_max else 'FAIL'}")
    print(f"  early-exit perplexity monotone:      {'PASS' if monotone else 'FAIL'}")
    print(f"  p_halt/p_correct weakly correlated:  {'PASS' if abs(corr) < 0.8 else 'WARN'}")


if __name__ == "__main__":
    main()

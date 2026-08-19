"""The abstention acceptance metric: SQuAD v2 abstention precision/recall + calibration.

Two numbers decide it:

  * **abstention precision and recall on the unanswerable split**, both reported. Measured by
    actually *generating* an answer for every held-out question and classifying it with
    ``modules.data.abstention.is_abstention`` -- which is an exact check rather than a
    classification problem precisely because the abstention phrasings are a small closed set (see
    that module's docstring).
  * **ECE of the abstention signal doesn't degrade relative to the pretrained checkpoint.**

``rajpurkar/squad_v2``'s **validation** split is the eval set here, and
``scripts/prepare_sft_data.py`` deliberately never consumes it (only ``squad_v2/train``) -- that
exclusion exists for this script.

Two calibration passes, because "the abstention signal" means two different things depending on
what you are willing to spend:

  * **answer-level** (from the generation pass, free): ``p_max`` averaged over the tokens the
    model actually generated, scored against whether the generated answer was right. This is the
    number that matches the user-facing claim -- "when it says it knows, does it?"
  * **token-level, teacher-forced** (``--baseline-checkpoint``): the same per-token quantity
    ``scripts/eval_calibration.py`` reports, computed on *these* prompts with the reference answer
    forced. This exists only to make the "doesn't degrade" half of the criterion an actual
    comparison: the generation pass cannot be run meaningfully on the pretrained checkpoint (it was
    never taught the chat format, so it does not produce answers to classify), whereas a
    teacher-forced pass over identical inputs can. **Caveat, stated in the printed report too:** the
    pretrained checkpoint is out of distribution on the chat control tokens, so its number is a
    conservative baseline -- SFT beating it is weaker evidence than SFT losing to it is.

Both ECE and AUROC come from ``scripts.eval_calibration`` by import, so Gate 5's numbers and these
are computed by the same code.

Generation here has no KV cache, unlike ``scripts/inference.py``: ``modules/model/kv_cache.py`` is
single-sequence, and this script's whole point is batched decoding over left-padded, varlen-segmented
rows. So every decode step re-runs the full prefix and cost is quadratic in the answer length. It is
tolerable because SQuAD answers are short -- ``--max-new-tokens`` defaults to 32 -- and because
prompts are length-sorted into batches so padding stays small. Use ``--max-examples`` to trade
precision for time.

**Results are not bit-reproducible across a change of ``--batch-size``**, and that is the model, not
this script. Left padding is genuinely invisible to the real tokens -- the dense decoder's output for
them is *bit-identical* when the pad region's contents change, because every attention path here is
varlen-segmented. But ``ParallelSparseMoELayer`` tiles its grouped GEMM by ``m_splits``, the
per-expert row counts, which are computed over every token in the batch including the pads; a
different batch composition therefore changes the bf16 accumulation order for the real tokens' rows
too (~0.5-1% of hidden-state magnitude on an untrained model). Greedy decoding is robust to that
once the logits have real margins, but keep ``--batch-size`` and ``--max-examples`` fixed across runs
you intend to compare.

Run from the repo root: `python scripts/eval_abstention.py -c ckpts/sft/checkpoint_sft_final.pt`.
"""
import os
import sys
import json
import math
import random
import string
import argparse
import collections
from typing import List, Optional, Sequence

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.data import abstention
from modules.data.chat import ChatTemplate
from config import ModelConfig, SFTConfig
from scripts.eval_calibration import expected_calibration_error, roc_auc
from scripts.prepare_sft_data import SQUAD_INSTRUCTION
from utils import BASE_DIR, BF16, TOKENIZER_DIR, get_hf_token, logger

SQUAD_REPO = "rajpurkar/squad_v2"
SQUAD_VAL_PREFIX = "squad_v2/validation"
SFT_CHECKPOINT_DIR = os.path.join(BASE_DIR, "ckpts", "sft")
CE_CHUNK_SIZE = 2048


# ---------------------------------------------------------------------------- data


def load_squad_validation(scratch_dir: str, hf_token: Optional[str], local_dir: Optional[str]) -> pd.DataFrame:
    """Read every ``squad_v2/validation`` parquet shard into one frame.

    ``local_dir`` short-circuits the Hub entirely (any directory of validation parquet files), which
    is what makes this runnable on a box that already has the shards or has no network. The Hub path
    pins the dataset revision for the same reason ``prepare_sft_data.py`` does: a repo that updated
    between the SFT corpus build and this eval would change what "held out" means.
    """
    if local_dir:
        files = sorted(
            os.path.join(local_dir, f) for f in os.listdir(local_dir) if f.endswith(".parquet")
        )
        if not files:
            raise SystemExit(f"no .parquet files in {local_dir}")
        logger.info(f"reading {len(files)} local validation shard(s) from {local_dir}")
        return pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files], ignore_index=True)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=hf_token)
    info = api.dataset_info(SQUAD_REPO)
    names = sorted(
        f for f in api.list_repo_files(SQUAD_REPO, repo_type="dataset", revision=info.sha)
        if f.startswith(SQUAD_VAL_PREFIX) and f.endswith(".parquet")
    )
    if not names:
        raise SystemExit(
            f"no files under {SQUAD_VAL_PREFIX!r} in {SQUAD_REPO} -- the Hub layout may have "
            "changed; pass --squad-dir with local parquet shards instead"
        )
    logger.info(f"downloading {len(names)} validation shard(s) from {SQUAD_REPO} @ {info.sha[:10]}")
    os.makedirs(scratch_dir, exist_ok=True)
    frames = []
    for name in names:
        path = hf_hub_download(
            repo_id=SQUAD_REPO, filename=name, repo_type="dataset",
            local_dir=scratch_dir, token=hf_token, revision=info.sha,
        )
        frames.append(pd.read_parquet(path, engine="pyarrow"))
    return pd.concat(frames, ignore_index=True)


def squad_references(row: dict) -> List[str]:
    """Reference answers for one row; empty list means the question is unanswerable."""
    answers = row.get("answers") or {}
    texts = answers.get("text") if isinstance(answers, dict) else None
    # pandas hands back a numpy array here, whose truthiness is ambiguous -- length-check it, same
    # as prepare_sft_data.render_squad_v2 does
    return [str(t).strip() for t in (texts if texts is not None else []) if str(t).strip()]


def squad_prompt(row: dict) -> Optional[str]:
    """The user turn for one row, byte-identical to what SFT trained on.

    ``SQUAD_INSTRUCTION`` is *imported* rather than restated: the instruction explicitly licenses
    abstention ("If the passage does not contain the answer, say so"), so a copy of it that drifted
    by a word would be measuring the model on a prompt it never saw, and the abstention rate is
    exactly the thing most sensitive to that.
    """
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    if not context or not question:
        return None
    return f"{SQUAD_INSTRUCTION}\n\nPassage:\n{context}\n\nQuestion: {question}"


# ------------------------------------------------------------------------- scoring


def normalize_answer(text: str) -> str:
    """SQuAD's official normalization: lowercase, drop articles/punctuation, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    tokens = [t for t in text.split() if t not in ("a", "an", "the")]
    return " ".join(tokens)


def exact_match(prediction: str, references: Sequence[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(r) for r in references))


def token_f1(prediction: str, references: Sequence[str]) -> float:
    """Max token-overlap F1 against any reference -- SQuAD's second official metric.

    Reported alongside EM because a generative model rarely reproduces a span verbatim; EM alone
    would understate answer quality and therefore overstate how often a confident answer was wrong,
    which biases the calibration numbers below.
    """
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for reference in references:
        ref_tokens = normalize_answer(reference).split()
        if not pred_tokens or not ref_tokens:
            best = max(best, float(pred_tokens == ref_tokens))
            continue
        common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def abstention_scores(abstained: np.ndarray, unanswerable: np.ndarray) -> dict:
    """Precision/recall of "the model abstained" as a detector of "the question is unanswerable".

    Positive class = unanswerable, prediction = abstained. The false-abstention rate on the
    answerable half is reported separately because it is the failure mode precision alone hides:
    a model that abstains on everything scores recall 1.0 and precision at the base rate, which
    looks unremarkable rather than degenerate.
    """
    tp = float((abstained & unanswerable).sum())
    fp = float((abstained & ~unanswerable).sum())
    fn = float((~abstained & unanswerable).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    n_answerable = float((~unanswerable).sum())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "abstention_rate": float(abstained.mean()) if len(abstained) else float("nan"),
        "false_abstention_rate": fp / n_answerable if n_answerable else float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------- model


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Newest ``.pt`` by mtime, or None. A final checkpoint wins ties by being written last."""
    best_ts, best_path = 0.0, None
    if not os.path.isdir(checkpoint_dir):
        return None
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("checkpoint") and fname.endswith(".pt"):
            fpath = os.path.join(checkpoint_dir, fname)
            ts = os.path.getmtime(fpath)
            if ts > best_ts:
                best_ts, best_path = ts, fpath
    return best_path


def load_model(checkpoint_path: str, device: str) -> TinyMoETransformer:
    """Load a checkpoint for eval. Accepts an SFT checkpoint or a pretraining one.

    ``sft.save_sft_checkpoint`` writes a strict superset of the pretraining payload, so one reader
    covers both. ``delayed_mtp_loss(True)`` keeps the MTP head returning hidden states rather than
    ``[B, S, vocab]`` logits per extra token -- nothing here reads them, and materializing them
    would dominate the decode step's memory.
    """
    model = TinyMoETransformer(**ModelConfig.Params).to(device).to(BF16)
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def _final_hidden(model: TinyMoETransformer, input_ids: torch.Tensor, document_ids: torch.Tensor) -> torch.Tensor:
    """One forward pass, returning the final loop's post-norm hidden states ``[B, S, H]``.

    ``return_hidden=True`` is what keeps this affordable: the alternative returns
    ``[B, S, vocab]`` logits (1GB at B=16/S=512/vocab=65536 in bf16), where every caller here needs
    the head applied to a handful of positions at most.
    """
    cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
    out = model(
        input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, return_hidden=True,
    )
    hidden_all = out[0] if isinstance(out, tuple) else out
    return hidden_all[-1]


# ------------------------------------------------------------------------ generation


@torch.inference_mode()
def generate_batch(model, prompt_ids: List[List[int]], *, max_new_tokens: int, temperature: float,
                   top_k: int, eos_id: int, pad_id: int, device: str, max_seq_len: int):
    """Greedy/top-k decode for a batch of variable-length prompts.

    **Left-padded and varlen-segmented.** Left padding puts every row's last real token at the same
    index, so one append extends every row at once -- with right padding the write position differs
    per row and drifts as rows finish. The padding is made harmless by giving it its own segment in
    ``document_ids`` (pad run = 0, real run = 1): flash's block-diagonal causal mask then keeps real
    tokens from ever attending to a pad, exactly as it keeps packed documents apart during training.
    RoPE positions are offset by the pad length, which is fine because the attention score depends
    only on the *relative* offset within a segment.

    That isolation is exact through the decoder and every attention expert (verified: the decoder's
    output for the real tokens is bit-identical when the pad region's contents change), but *not*
    bit-exact through the MoE -- see this module's docstring on ``m_splits``.

    Returns:
        ``(texts, p_max_mean, n_generated)`` -- the decoded completions plus, per row, ``p_max``
        averaged over the tokens actually generated (the terminating EOS included; padding after a
        finished row excluded).
    """
    batch = len(prompt_ids)
    width = max(len(p) for p in prompt_ids)
    ids = torch.full((batch, width), pad_id, dtype=torch.long, device=device)
    doc = torch.zeros((batch, width), dtype=torch.long, device=device)
    for i, prompt in enumerate(prompt_ids):
        ids[i, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
        doc[i, width - len(prompt):] = 1

    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    generated = [[] for _ in range(batch)]
    p_max_sum = torch.zeros(batch, dtype=torch.float32, device=device)
    counts = torch.zeros(batch, dtype=torch.float32, device=device)

    for _ in range(max_new_tokens):
        window_ids = ids[:, -max_seq_len:]
        window_doc = doc[:, -max_seq_len:]
        hidden = _final_hidden(model, window_ids, window_doc)
        h_last = hidden[:, -1, :]                       # [B, H]
        logits = model.lm_head(h_last).float()          # [B, vocab] -- one position, not the row

        live = (~finished).float()
        p_max_sum += logits.softmax(-1).max(-1).values * live
        counts += live

        if temperature > 0:
            scaled = logits / temperature
            if top_k > 0:
                kth = torch.topk(scaled, min(top_k, scaled.size(-1))).values[:, -1:]
                scaled = scaled.masked_fill(scaled < kth, float("-inf"))
            next_token = torch.multinomial(scaled.softmax(-1), num_samples=1)
        else:
            next_token = logits.argmax(-1, keepdim=True)
        # a finished row keeps emitting pad so the tensor stays rectangular; it is inside that
        # row's own segment and cannot affect any other row
        next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, pad_id), next_token)

        flat = next_token.squeeze(-1).tolist()
        already = finished.tolist()
        for i, token in enumerate(flat):
            if not already[i]:
                generated[i].append(token)
        finished |= next_token.squeeze(-1) == eos_id
        if bool(finished.all()):
            break

        ids = torch.cat([ids, next_token], dim=-1)
        doc = torch.cat([doc, torch.ones_like(next_token)], dim=-1)

    denominator = counts.clamp(min=1.0)
    return generated, (p_max_sum / denominator).cpu().numpy(), counts.cpu().numpy()


def run_generation(model, tokenizer, template: ChatTemplate, records: List[dict], *, batch_size: int,
                   max_new_tokens: int, temperature: float, top_k: int, device: str) -> None:
    """Fill in ``completion``/``p_max`` on every record, in place.

    Records are length-sorted into batches (and restored to their original order by writing back
    through the record objects): padding is what a batched, cache-free decoder wastes most compute
    on, and SQuAD passages vary by an order of magnitude in length.
    """
    max_seq_len = ModelConfig.Params["max_seq_len"]
    order = sorted(range(len(records)), key=lambda i: len(records[i]["prompt_ids"]))
    done = 0
    for start in range(0, len(order), batch_size):
        chunk = [records[i] for i in order[start:start + batch_size]]
        texts, p_max, counts = generate_batch(
            model, [r["prompt_ids"] for r in chunk],
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            eos_id=template.eos_id, pad_id=tokenizer.pad_token_id, device=device,
            max_seq_len=max_seq_len,
        )
        for record, token_ids, pm, n in zip(chunk, texts, p_max, counts):
            record["completion"] = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            record["p_max"] = float(pm)
            record["n_generated"] = int(n)
        done += len(chunk)
        if done % (batch_size * 10) < batch_size:
            logger.info(f"[eval_abstention] generated {done:,}/{len(records):,} answers")


# -------------------------------------------------------------------- teacher forcing


@torch.inference_mode()
def teacher_forced_calibration(model, template: ChatTemplate, records: List[dict], *,
                               pad_id: int, batch_size: int, device: str, max_seq_len: int,
                               n_bins: int = 15) -> dict:
    """Per-token CE and confidence over the reference answers, forced.

    This is ``scripts/eval_calibration.py``'s measurement (same ECE/AUROC functions, same ``p_max``
    signal) restricted to the supervised tokens of these SQuAD prompts, which is what makes it
    comparable across two checkpoints that cannot both be *generated* from. Unanswerable rows are
    forced onto the same fixed abstention phrasing the SFT corpus used, so "was the model confident
    about the abstention" is part of the number rather than excluded from it.
    """
    p_max_parts, is_correct_parts = [], []
    ce_sum, n_tokens = 0.0, 0

    order = sorted(range(len(records)), key=lambda i: len(records[i]["forced_ids"]))
    for start in range(0, len(order), batch_size):
        chunk = [records[i] for i in order[start:start + batch_size]]
        width = max(len(r["forced_ids"]) for r in chunk)
        if width > max_seq_len:
            width = max_seq_len
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
        doc = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long, device=device)
        for i, record in enumerate(chunk):
            row = torch.tensor(record["forced_ids"][:width], dtype=torch.long, device=device)
            supervised = torch.tensor(record["forced_mask"][:width], dtype=torch.bool, device=device)
            n = row.numel()
            ids[i, :n] = row
            doc[i, :n] = 1
            labels[i, :n] = torch.where(supervised, row, torch.full_like(row, -100))

        hidden = _final_hidden(model, ids, doc)
        # position t predicts token t+1, so the supervised label tensor shifts left against hidden
        h = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
        target = labels[:, 1:].reshape(-1)

        for chunk_start in range(0, h.size(0), CE_CHUNK_SIZE):
            h_part = h[chunk_start:chunk_start + CE_CHUNK_SIZE]
            t_part = target[chunk_start:chunk_start + CE_CHUNK_SIZE]
            valid = t_part != -100
            if not valid.any():
                continue
            logits = model.lm_head(h_part).float()
            ce_sum += F.cross_entropy(logits[valid], t_part[valid], reduction="sum").item()
            n_tokens += int(valid.sum().item())
            is_correct = (logits.argmax(-1) == t_part).float()
            p_max = logits.softmax(-1).max(-1).values
            p_max_parts.append(p_max[valid].cpu().numpy())
            is_correct_parts.append(is_correct[valid].cpu().numpy())

    if n_tokens == 0:
        return {}
    p_max_all = np.concatenate(p_max_parts)
    is_correct_all = np.concatenate(is_correct_parts)
    return {
        "tokens": n_tokens,
        "ce": ce_sum / n_tokens,
        "ppl": math.exp(min(ce_sum / n_tokens, 20.0)),
        "top1_acc": float(is_correct_all.mean()),
        "ece_p_max": expected_calibration_error(p_max_all, is_correct_all, n_bins),
        "auroc_p_max": roc_auc(p_max_all, is_correct_all),
        "mean_p_max": float(p_max_all.mean()),
    }


# ------------------------------------------------------------------------------ main


def build_records(frame: pd.DataFrame, template: ChatTemplate, *, max_examples: Optional[int],
                  max_prompt_tokens: int, seed: int, with_forced: bool = True) -> List[dict]:
    """Render, tokenize and (optionally) subsample the validation split.

    Subsampling shuffles before truncating so a capped run keeps the split's answerable/unanswerable
    balance in expectation; the shuffle is seeded so two runs of this script compare like for like.
    Over-long rows are **dropped, not truncated** -- truncating a passage can remove the very span
    that makes a question answerable, silently relabelling it.

    ``with_forced=False`` (``--skip-forced``) skips the second full-corpus tokenizer pass that the
    teacher-forced targets need; the passages dominate that cost and they are already encoded.
    """
    rows = frame.to_dict("records")
    rng = random.Random(seed)
    rng.shuffle(rows)

    records, forced_conversations, dropped_long, dropped_bad = [], [], 0, 0
    for row in rows:
        if max_examples is not None and len(records) >= max_examples:
            break
        prompt = squad_prompt(row)
        if prompt is None:
            dropped_bad += 1
            continue
        references = squad_references(row)
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = template.encode_prompt(messages)
        if len(prompt_ids) > max_prompt_tokens:
            dropped_long += 1
            continue
        if with_forced:
            # the forced target is exactly what prepare_sft_data.render_squad_v2 would have written
            # for this row: the first reference, or one of the same fixed abstention phrasings
            answer = references[0] if references else abstention.pick(abstention.ABSTENTIONS_PASSAGE, rng)
            forced_conversations.append(messages + [{"role": "assistant", "content": answer}])
        records.append({
            "id": str(row.get("id", "")),
            "question": str(row.get("question") or ""),
            "references": references,
            "unanswerable": not references,
            "prompt_ids": prompt_ids,
        })

    if with_forced:
        kept = []
        for record, pair in zip(records, template.encode_batch(forced_conversations)):
            if pair is None:
                dropped_bad += 1
                continue
            record["forced_ids"], record["forced_mask"] = pair
            kept.append(record)
    else:
        kept = records

    logger.info(
        f"{len(kept):,} validation questions "
        f"({sum(r['unanswerable'] for r in kept):,} unanswerable), "
        f"dropped {dropped_long:,} over {max_prompt_tokens} prompt tokens / {dropped_bad:,} unusable"
    )
    return kept


def report(records: List[dict], forced: dict, baseline: Optional[dict], n_bins: int) -> None:
    abstained = np.array([r["abstained"] for r in records], dtype=bool)
    unanswerable = np.array([r["unanswerable"] for r in records], dtype=bool)
    em = np.array([r["em"] for r in records], dtype=np.float64)
    f1 = np.array([r["f1"] for r in records], dtype=np.float64)
    is_correct = np.array([r["is_correct"] for r in records], dtype=np.float64)
    p_max = np.array([r["p_max"] for r in records], dtype=np.float64)

    scores = abstention_scores(abstained, unanswerable)
    answerable = ~unanswerable

    print("\n=== SQuAD v2 validation, generated answers ===")
    print(f"  questions: {len(records):,}  ({int(unanswerable.sum()):,} unanswerable, "
          f"{int(answerable.sum()):,} answerable)")
    print(f"  abstention precision: {scores['precision']:.4f}   "
          f"({scores['tp']} correct abstentions / {scores['tp'] + scores['fp']} total)")
    print(f"  abstention recall:    {scores['recall']:.4f}   "
          f"({scores['tp']} / {scores['tp'] + scores['fn']} unanswerable)")
    print(f"  abstention F1:        {scores['f1']:.4f}")
    print(f"  overall abstention rate:            {scores['abstention_rate']:.4f}")
    print(f"  false abstention rate (answerable): {scores['false_abstention_rate']:.4f}"
          "   <- the degenerate 'refuse everything' tell")

    print("\n=== Answer quality (answerable half only) ===")
    if answerable.any():
        print(f"  exact match: {em[answerable].mean():.4f}   token F1: {f1[answerable].mean():.4f}")
    else:
        print("  no answerable questions in this sample")
    print(f"  overall correctness (EM on answerable, abstention on unanswerable): {is_correct.mean():.4f}")

    print("\n=== Answer-level calibration (p_max over generated tokens) ===")
    print(f"  {'signal':<10} {'mean':>8} {'ECE':>8} {'AUROC':>8}")
    ece = expected_calibration_error(p_max, is_correct, n_bins)
    print(f"  {'p_max':<10} {p_max.mean():>8.4f} {ece:>8.4f} {roc_auc(p_max, is_correct):>8.4f}")
    # the literal "abstention signal": does low confidence predict that the question is unanswerable
    print(f"  AUROC of (1 - p_max) for detecting unanswerable: "
          f"{roc_auc(-p_max, unanswerable.astype(np.float64)):.4f}")

    if not forced:
        return
    print("\n=== Token-level calibration, teacher-forced on the same prompts ===")
    print("  (the quantity scripts/eval_calibration.py reports, restricted to these supervised tokens)")
    header = f"  {'checkpoint':<12} {'CE':>8} {'top-1':>8} {'ECE(pmax)':>10} {'AUROC(pmax)':>12}"
    print(header)
    print(f"  {'sft':<12} {forced['ce']:>8.4f} {forced['top1_acc']:>8.4f} "
          f"{forced['ece_p_max']:>10.4f} {forced['auroc_p_max']:>12.4f}")
    if baseline:
        print(f"  {'pretrained':<12} {baseline['ce']:>8.4f} {baseline['top1_acc']:>8.4f} "
              f"{baseline['ece_p_max']:>10.4f} {baseline['auroc_p_max']:>12.4f}")
        delta = forced["ece_p_max"] - baseline["ece_p_max"]
        print(f"\n  ECE(p_max) change vs pretrained: {delta:+.4f} "
              f"({'PASS -- no degradation' if delta <= 0 else 'FAIL -- calibration degraded'})")
        print("  Caveat: the pretrained checkpoint never saw the chat control tokens, so it is out of")
        print("  distribution on these inputs. Read a PASS here as weak evidence and a FAIL as strong.")
    else:
        print("\n  pass --baseline-checkpoint <pretrained .pt> for the 'doesn't degrade' comparison")


def main():
    parser = argparse.ArgumentParser(
        description="SQuAD v2 abstention precision/recall + calibration")
    parser.add_argument("--checkpoint", "-c", default=find_latest_checkpoint(SFT_CHECKPOINT_DIR),
                        help="SFT checkpoint to evaluate (default: newest in ckpts/sft)")
    parser.add_argument("--baseline-checkpoint", default=None,
                        help="pretrained checkpoint for the 'ECE doesn't degrade' comparison "
                             "(teacher-forced pass only -- see this module's docstring)")
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--squad-dir", default=None,
                        help="directory of local squad_v2 validation parquet shards (skips the Hub)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="cap the eval to this many questions (seeded subsample; default: all)")
    parser.add_argument("--max-prompt-tokens", type=int, default=1024,
                        help="drop questions whose rendered prompt exceeds this (never truncated)")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="decode budget per answer; SQuAD answers and the fixed abstentions are short")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (default). The acceptance number should be deterministic; "
                             "temperature sampling is Step 13's job")
    parser.add_argument("--top-k", type=int, default=50, help="ignored when --temperature is 0")
    parser.add_argument("--seed", type=int, default=SFTConfig.seed,
                        help="seeds the subsample shuffle and the forced abstention phrasings")
    parser.add_argument("--n-bins", type=int, default=15, help="ECE histogram bins")
    parser.add_argument("--skip-forced", action="store_true",
                        help="generation metrics only; skip the teacher-forced calibration pass")
    parser.add_argument("--json-out", default=None,
                        help="write per-question records here (Step 13 reads this shape)")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint is None:
        raise SystemExit(f"No checkpoint found in {SFT_CHECKPOINT_DIR} and none passed via --checkpoint")

    logger.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = ChatTemplate(tokenizer)

    scratch_dir = os.path.join(BASE_DIR, "data", "prepared", "_squad_val_scratch")
    frame = load_squad_validation(scratch_dir, args.hf_token or get_hf_token(), args.squad_dir)
    records = build_records(
        frame, template, max_examples=args.max_examples,
        max_prompt_tokens=args.max_prompt_tokens, seed=args.seed,
        with_forced=not args.skip_forced,
    )
    if not records:
        raise SystemExit("no usable validation questions -- check --max-prompt-tokens / --squad-dir")

    logger.info(f"Loading SFT checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    logger.info(f"Generating {len(records):,} answers "
                f"(batch {args.batch_size}, <= {args.max_new_tokens} new tokens, "
                f"{'greedy' if args.temperature <= 0 else f'T={args.temperature}'})")
    run_generation(
        model, tokenizer, template, records, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
        device=args.device,
    )

    for record in records:
        completion = record["completion"]
        record["abstained"] = abstention.is_abstention(completion)
        record["em"] = exact_match(completion, record["references"]) if record["references"] else 0.0
        record["f1"] = token_f1(completion, record["references"]) if record["references"] else 0.0
        # an unanswerable question is answered correctly by abstaining; an answerable one by
        # producing the span -- and abstaining on it is wrong however well phrased
        record["is_correct"] = (
            float(record["abstained"]) if record["unanswerable"]
            else (0.0 if record["abstained"] else record["em"])
        )

    forced, baseline = {}, None
    if not args.skip_forced:
        logger.info("Teacher-forced calibration pass (SFT checkpoint)")
        forced = teacher_forced_calibration(
            model, template, records, pad_id=tokenizer.pad_token_id,
            batch_size=args.batch_size, device=args.device,
            max_seq_len=ModelConfig.Params["max_seq_len"], n_bins=args.n_bins,
        )
        if args.baseline_checkpoint:
            # both checkpoints are the same architecture at ~660MB in bf16, but the second one is
            # loaded onto the same device -- drop the first rather than hold two
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            logger.info(f"Teacher-forced calibration pass (baseline {args.baseline_checkpoint})")
            baseline_model = load_model(args.baseline_checkpoint, args.device)
            baseline = teacher_forced_calibration(
                baseline_model, template, records, pad_id=tokenizer.pad_token_id,
                batch_size=args.batch_size, device=args.device,
                max_seq_len=ModelConfig.Params["max_seq_len"], n_bins=args.n_bins,
            )

    report(records, forced, baseline, args.n_bins)

    if args.json_out:
        payload = {
            "checkpoint": args.checkpoint,
            "baseline_checkpoint": args.baseline_checkpoint,
            "temperature": args.temperature,
            "seed": args.seed,
            "forced": forced,
            "baseline_forced": baseline,
            "records": [
                {k: v for k, v in r.items() if k not in ("prompt_ids", "forced_ids", "forced_mask")}
                for r in records
            ],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"wrote {len(records):,} per-question records to {args.json_out}")


if __name__ == "__main__":
    main()

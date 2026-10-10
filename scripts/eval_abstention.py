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
import hashlib
import random
import string
import argparse
import collections
from typing import Dict, List, Optional, Sequence, Tuple

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids, _segment_ids
from modules.data import abstention
from modules.data.chat import ChatTemplate
from modules.data.entity_swap import (
    Gazetteer, counterfactual_readings, frequency_stratum, prepare_counterfactual, record_for_json,
    sigmoid_ratio,
)
from modules.data.evidence_dataset import EMBED_DIM
from config import ModelConfig, SFTConfig
from scripts.eval_calibration import expected_calibration_error, roc_auc
from scripts.prepare_sft_data import SQUAD_INSTRUCTION
from scripts.prepare_evidence_data import ChunkEmbedder, EVIDENCE_SQUAD_INSTRUCTION, evidence_prompt
from utils import BASE_DIR, BF16, TOKENIZER_DIR, get_hf_token, load_model_state, logger, model_params_for_state_dict

SQUAD_REPO = "rajpurkar/squad_v2"
SFT_CHECKPOINT_DIR = os.path.join(BASE_DIR, "ckpts", "sft")
CE_CHUNK_SIZE = 2048
# the port's conditions this script can attach on the eval side ("many" is corpus-only -- it
# exists to put a large buffer in TRAINING somewhere, which an eval slice this small has no use
# for). "counterfactual" is gold evidence with the answer swapped for another of the same type: it
# scores whether the model repeats what the evidence says or what it remembers, and needs "gold" in
# the same run to know which items the model answers correctly with the unswapped chunk
EVIDENCE_CONDITIONS = ("gold", "none", "distractors", "mixed", "counterfactual")
DEFAULT_FREQUENCY_BIN = os.path.join(BASE_DIR, "data", "prepared", "ir.bin")
FREQUENCY_CACHE_DIR = os.path.join(BASE_DIR, "data", "benchmarks", "entity_freq")
DEFAULT_FACTS = os.path.join(BASE_DIR, "data", "prepared_inject", "inject_facts.jsonl")
DEFAULT_POOLS = os.path.join(BASE_DIR, "data", "prepared_inject", "inject_pools.json")
DEFAULT_STORE_DIR = os.path.join(BASE_DIR, "data", "index", "inject_bios")
# which record field holds the text of each answer whose log-probability is scored
ANSWER_TEXT_FIELDS = {"orig": "cf_original", "swap": "cf_substitute"}


# ---------------------------------------------------------------------------- data


def load_squad_split(scratch_dir: str, hf_token: Optional[str], local_dir: Optional[str],
                     split: str = "validation") -> pd.DataFrame:
    """Read every ``squad_v2/{split}`` parquet shard into one frame.

    ``local_dir`` short-circuits the Hub entirely (any directory of parquet files for that split),
    which is what makes this runnable on a box that already has the shards or has no network. The
    Hub path pins the dataset revision for the same reason ``prepare_sft_data.py`` does: a repo that
    updated between the SFT corpus build and this eval would change what "held out" means.

    ``split`` is a parameter because ``scripts/eval_probe.py`` fits on the *train* split while this
    script scores the *validation* one -- one reader, so neither can drift into a different revision
    or a different column convention than the other.
    """
    prefix = f"squad_v2/{split}"
    if local_dir:
        files = sorted(
            os.path.join(local_dir, f) for f in os.listdir(local_dir) if f.endswith(".parquet")
        )
        if not files:
            raise SystemExit(f"no .parquet files in {local_dir}")
        logger.info(f"reading {len(files)} local {split} shard(s) from {local_dir}")
        return pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files], ignore_index=True)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=hf_token)
    info = api.dataset_info(SQUAD_REPO)
    names = sorted(
        f for f in api.list_repo_files(SQUAD_REPO, repo_type="dataset", revision=info.sha)
        if f.startswith(prefix) and f.endswith(".parquet")
    )
    if not names:
        raise SystemExit(
            f"no files under {prefix!r} in {SQUAD_REPO} -- the Hub layout may have "
            "changed; pass --squad-dir with local parquet shards instead"
        )
    logger.info(f"downloading {len(names)} {split} shard(s) from {SQUAD_REPO} @ {info.sha[:10]}")
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # shape from the checkpoint, not config.yaml -- the pre-reshape checkpoints stay measurable
    params = model_params_for_state_dict(state_dict, ModelConfig.Params)
    model = TinyMoETransformer(**params).to(device).to(BF16)
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    load_model_state(model, state_dict)
    model.eval()
    return model


def _final_hidden(model: TinyMoETransformer, input_ids: torch.Tensor, document_ids: torch.Tensor,
                  evidence=None) -> torch.Tensor:
    """One forward pass, returning the final loop's post-norm hidden states ``[B, S, H]``.

    ``return_hidden=True`` is what keeps this affordable: the alternative returns
    ``[B, S, vocab]`` logits (1GB at B=16/S=512/vocab=65536 in bf16), where every caller here needs
    the head applied to a handful of positions at most.

    ``skip_mtp=True`` for the same reason one step further out: nothing here reads the drafted
    tokens, and the head would otherwise run over the whole prefix on every decode step of a
    cache-free generation loop.

    ``evidence`` (optional) is an ``EvidenceBatch`` built by ``_pack_evidence_batch``, read by the
    port at every loop exactly as training reads it. ``None`` (the default, and every call site that
    predates ``--evidence-port``) reproduces the forward this function always ran, bit for bit.
    """
    cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
    out = model(
        input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, return_hidden=True,
        skip_mtp=True, evidence=evidence,
    )
    hidden_all = out[0] if isinstance(out, tuple) else out
    return hidden_all[-1]


# ------------------------------------------------------------------------ evidence attachment


def _pack_evidence_batch(model, real_segment: torch.Tensor, dump_segment: torch.Tensor,
                         evidence_rows: List[Optional[dict]], num_segments: int, device: str):
    """Build the ``EvidenceBatch`` for one forward call from per-row evidence dicts.

    ``real_segment``/``dump_segment`` are THIS call's global segment ids (from this call's own
    ``cu_seqlens``) for, respectively, the row's real query content and a segment whose output
    nothing reads. A row's own evidence tokens are attributed to its real segment; whatever extra
    width the shared rectangular ``[B, S_ev]`` tensor needs beyond that row's own evidence is
    attributed to the dump segment instead -- the same trick ``modules/data/evidence_dataset.py``
    uses for its trailing pad segment, generalized because the caller decides which of a row's two
    segments plays which role: ``generate_batch`` pads on the left (so the dump is the row's FIRST
    segment) and ``teacher_forced_calibration`` pads on the right (so the dump is its LAST), and both
    need a genuine second segment to exist -- see each caller's ``extra_pad`` reservation.

    Args:
        evidence_rows: one entry per row, ``{"ids": [...], "chunk_ids": [...], "keys": [C, 384] or
            None}``, or ``None``/``{}`` for a row with nothing attached.
        num_segments: the WHOLE batch's segment count (``len(cu_seqlens) - 1``), passed rather than
            derived from ``real_segment``/``dump_segment`` alone -- both are per-row [B] slices and
            cannot see segments neither of them names.

    Returns:
        An ``EvidenceBatch``, or ``None`` when no row has anything attached (the bit-identical
        no-evidence forward).
    """
    B = len(evidence_rows)
    widths = [len(r["ids"]) if r and r.get("ids") else 0 for r in evidence_rows]
    chunk_counts = [(r["keys"].shape[0] if r and r.get("keys") is not None else 0) for r in evidence_rows]
    S_ev, C = max(widths, default=0), max(chunk_counts, default=0)
    if S_ev == 0 or C == 0:
        return None

    real_list, dump_list = real_segment.tolist(), dump_segment.tolist()
    ids = torch.zeros((B, S_ev), dtype=torch.long)
    chunk_ids = torch.full((B, S_ev), -1, dtype=torch.long)
    ev_segments = torch.tensor([[dump_list[i]] * S_ev for i in range(B)], dtype=torch.long)
    keys = torch.zeros((B, C, EMBED_DIM), dtype=torch.float32)
    chunk_segments_local = torch.full((B, C), -1, dtype=torch.long)

    for i, row in enumerate(evidence_rows):
        if not row or not row.get("ids"):
            continue
        n = len(row["ids"])
        ids[i, :n] = torch.tensor(row["ids"], dtype=torch.long)
        chunk_ids[i, :n] = torch.tensor(row["chunk_ids"], dtype=torch.long)
        ev_segments[i, :n] = real_list[i]
        keys_row = row.get("keys")
        k = keys_row.shape[0] if keys_row is not None else 0
        if k:
            keys[i, :k] = torch.as_tensor(keys_row, dtype=torch.float32)
            chunk_segments_local[i, :k] = real_list[i]

    valid = chunk_segments_local >= 0
    chunk_segments = chunk_segments_local[valid].to(device)
    chunk_keys = keys[valid].to(device)
    return model.build_evidence(
        ids.to(device), chunk_ids.to(device), ev_segments.to(device), num_segments,
        chunk_keys=chunk_keys, chunk_segments=chunk_segments,
    )


def _capture_memory_mass(model, batch: int, seq_len: int, mass_out: List[Optional[dict]]) -> None:
    """Fill ``mass_out`` in place with the IR table's external-mass fraction at each row's LAST
    position of the just-completed forward -- read right after it, since the module overwrites
    ``last_memory_mass`` on every call.

    ``None`` per row where the checkpoint carries no IR module, or the forward carried no evidence
    at all (the module clears its own mass then, rather than leaving a stale value from an earlier
    batch, which is exactly what lets this tell the two cases apart).

    ``"last"`` and ``"by_loop"`` read IR expert 0, as they always have, so older outputs stay
    comparable. ``"by_expert"`` is ``{expert index: last}`` over every IR expert that carried mass.
    """
    ir_modules = getattr(model.moe, "ir_modules", [])
    ir_module = ir_modules[0] if ir_modules else None
    if ir_module is None or ir_module.last_memory_mass is None:
        for i in range(batch):
            mass_out[i] = None
        return
    last = ir_module.last_memory_mass.view(batch, seq_len)[:, -1].float().cpu().tolist()
    by_loop = getattr(ir_module, "memory_mass_by_loop", None) or {}
    loop_last = {
        loop_idx: tensor.view(batch, seq_len)[:, -1].float().cpu().tolist()
        for loop_idx, tensor in by_loop.items()
    }
    expert_last = {
        idx: module.last_memory_mass.view(batch, seq_len)[:, -1].float().cpu().tolist()
        for idx, module in enumerate(ir_modules) if module.last_memory_mass is not None
    }
    for i in range(batch):
        mass_out[i] = {"last": last[i], "by_loop": {idx: vals[i] for idx, vals in loop_last.items()},
                       "by_expert": {idx: vals[i] for idx, vals in expert_last.items()}}


# ------------------------------------------------------------------------ generation


@torch.inference_mode()
def generate_batch(model, prompt_ids: List[List[int]], *, max_new_tokens: int, temperature: float,
                   top_k: int, eos_id: int, pad_id: int, device: str, max_seq_len: int,
                   evidence_rows: Optional[List[Optional[dict]]] = None,
                   mass_out: Optional[List[Optional[dict]]] = None):
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

    ``evidence_rows`` (optional) attaches retrieved evidence through the port, one entry per row
    (see ``_pack_evidence_batch``). There is no KV cache on this path at all, so a decode step
    already re-runs the whole prefix; attaching evidence adds one more thing rebuilt from scratch
    every step, because the evidence batch's segment ids are only valid against THIS step's
    ``cu_seqlens`` -- the query side grows by one token every step, which renumbers every segment
    past the first. The evidence CONTENT never changes across steps, only its packaging, so this is
    pure waste rather than a correctness requirement, and is the reason this mode should be run with
    a small ``--max-new-tokens``. When given, an extra pad column is reserved on the left even for
    the longest prompt, so every row keeps a real (if length-1) pad segment distinct from its query
    content -- the segment ``_pack_evidence_batch`` dumps another row's extra evidence width into.

    ``mass_out`` (optional, only meaningful with ``evidence_rows``): filled in place, one entry per
    row, with the IR table's external-memory-mass fraction at the row's LAST PROMPT position --
    read at the very first step, before any token has been generated, which left padding is what
    makes a plain ``[:, -1]`` slice.

    Returns:
        ``(texts, p_max_mean, n_generated)`` -- the decoded completions plus, per row, ``p_max``
        averaged over the tokens actually generated (the terminating EOS included; padding after a
        finished row excluded). Unchanged in shape and meaning from before evidence support existed;
        ``mass_out`` is filled as a side effect rather than added to this tuple, so every existing
        caller (this module's own default path, ``eval_benchmarks.py``) is untouched.
    """
    batch = len(prompt_ids)
    extra_pad = 1 if evidence_rows is not None else 0
    width = max(len(p) for p in prompt_ids) + extra_pad
    ids = torch.full((batch, width), pad_id, dtype=torch.long, device=device)
    doc = torch.zeros((batch, width), dtype=torch.long, device=device)
    for i, prompt in enumerate(prompt_ids):
        ids[i, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
        doc[i, width - len(prompt):] = 1

    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    generated = [[] for _ in range(batch)]
    p_max_sum = torch.zeros(batch, dtype=torch.float32, device=device)
    counts = torch.zeros(batch, dtype=torch.float32, device=device)

    for step in range(max_new_tokens):
        window_ids = ids[:, -max_seq_len:]
        window_doc = doc[:, -max_seq_len:]
        evidence_batch = None
        if evidence_rows is not None:
            cu_seqlens, _ = cu_seqlens_from_doc_ids(window_doc)
            seg = _segment_ids(cu_seqlens, batch, window_doc.shape[1], device)
            # left padded: the row's real content is its LAST segment, the pad run (guaranteed to
            # exist by extra_pad above) is its FIRST -- see _pack_evidence_batch's docstring
            evidence_batch = _pack_evidence_batch(
                model, seg[:, -1], seg[:, 0], evidence_rows,
                num_segments=int(cu_seqlens.numel() - 1), device=device,
            )
        hidden = _final_hidden(model, window_ids, window_doc, evidence=evidence_batch)
        if step == 0 and mass_out is not None:
            _capture_memory_mass(model, batch, window_doc.shape[1], mass_out)
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
                   max_new_tokens: int, temperature: float, top_k: int, device: str,
                   attach_evidence: bool = False) -> None:
    """Fill in ``completion``/``p_max`` on every record, in place.

    Records are length-sorted into batches (and restored to their original order by writing back
    through the record objects): padding is what a batched, cache-free decoder wastes most compute
    on, and SQuAD passages vary by an order of magnitude in length.

    ``attach_evidence`` (``--evidence-port`` mode) reads each record's ``evidence_row`` (set by
    ``attach_condition`` for whichever condition is currently being scored) and also fills in
    ``memory_mass``/``memory_mass_by_loop``/``memory_mass_by_expert`` -- the external store's share
    of the read at the row's last prompt position, read by ``generate_batch``'s ``mass_out``.
    """
    max_seq_len = ModelConfig.Params["max_seq_len"]
    order = sorted(range(len(records)), key=lambda i: len(records[i]["prompt_ids"]))
    done = 0
    for start in range(0, len(order), batch_size):
        chunk = [records[i] for i in order[start:start + batch_size]]
        evidence_rows = [r.get("evidence_row") for r in chunk] if attach_evidence else None
        mass_out = [None] * len(chunk) if attach_evidence else None
        texts, p_max, counts = generate_batch(
            model, [r["prompt_ids"] for r in chunk],
            max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            eos_id=template.eos_id, pad_id=tokenizer.pad_token_id, device=device,
            max_seq_len=max_seq_len, evidence_rows=evidence_rows, mass_out=mass_out,
        )
        for record, token_ids, pm, n in zip(chunk, texts, p_max, counts):
            record["completion"] = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            record["p_max"] = float(pm)
            record["n_generated"] = int(n)
        if mass_out is not None:
            for record, mass in zip(chunk, mass_out):
                record["memory_mass"] = mass["last"] if mass else None
                record["memory_mass_by_loop"] = mass["by_loop"] if mass else {}
                record["memory_mass_by_expert"] = mass.get("by_expert", {}) if mass else {}
        done += len(chunk)
        if done % (batch_size * 10) < batch_size:
            logger.info(f"[eval_abstention] generated {done:,}/{len(records):,} answers")


# -------------------------------------------------------------------- teacher forcing


@torch.inference_mode()
def teacher_forced_calibration(model, template: ChatTemplate, records: List[dict], *,
                               pad_id: int, batch_size: int, device: str, max_seq_len: int,
                               n_bins: int = 15, use_evidence: bool = False) -> dict:
    """Per-token CE and confidence over the reference answers, forced.

    This is ``scripts/eval_calibration.py``'s measurement (same ECE/AUROC functions, same ``p_max``
    signal) restricted to the supervised tokens of these SQuAD prompts, which is what makes it
    comparable across two checkpoints that cannot both be *generated* from. Unanswerable rows are
    forced onto the same fixed abstention phrasing the SFT corpus used, so "was the model confident
    about the abstention" is part of the number rather than excluded from it.

    ``use_evidence`` (``--evidence-port`` mode) reads each record's ``evidence_row`` (set by
    ``attach_condition``) and attaches it through the port instead of leaving the prompt as is --
    the caller is responsible for having rendered ``forced_ids``/``forced_mask`` from a prompt that
    never had a passage in it (see ``attach_condition``/``build_evidence_records``). Off by default,
    which reproduces every number this function reported before evidence support existed.
    """
    p_max_parts, is_correct_parts = [], []
    ce_sum, n_tokens = 0.0, 0

    order = sorted(range(len(records)), key=lambda i: len(records[i]["forced_ids"]))
    for start in range(0, len(order), batch_size):
        chunk = [records[i] for i in order[start:start + batch_size]]
        width = max(len(r["forced_ids"]) for r in chunk)
        # one guaranteed trailing pad column so the evidence packer always has a real segment to
        # dump another row's extra evidence width into (see _pack_evidence_batch) -- harmless when
        # width is already below max_seq_len, and in the rare case a forced sequence fills the whole
        # context this simply reverts to the pre-evidence truncation, same as the line below
        if use_evidence:
            width += 1
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

        evidence_batch = None
        if use_evidence:
            evidence_rows = [r.get("evidence_row") for r in chunk]
            cu_seqlens, _ = cu_seqlens_from_doc_ids(doc)
            seg = _segment_ids(cu_seqlens, len(chunk), width, device)
            # right padded here (unlike generate_batch): the row's real content is its FIRST
            # segment, the trailing pad run is its LAST -- see _pack_evidence_batch's docstring
            evidence_batch = _pack_evidence_batch(
                model, seg[:, 0], seg[:, -1], evidence_rows,
                num_segments=int(cu_seqlens.numel() - 1), device=device,
            )
        hidden = _final_hidden(model, ids, doc, evidence=evidence_batch)
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
                  max_prompt_tokens: int, seed: int, with_forced: bool = True,
                  offset: int = 0) -> List[dict]:
    """Render, tokenize and (optionally) subsample the validation split.

    Subsampling shuffles before truncating so a capped run keeps the split's answerable/unanswerable
    balance in expectation; the shuffle is seeded so two runs of this script compare like for like.
    Over-long rows are **dropped, not truncated** -- truncating a passage can remove the very span
    that makes a question answerable, silently relabelling it.

    ``offset`` skips that many *usable* questions before collecting any, which is how a disjoint
    slice of the same seeded shuffle is taken: ``--example-offset 2000 --max-examples 2000`` is
    questions 2000-4000 of exactly the ordering ``--max-examples 2000`` reads the first 2000 of.
    Same split, same shuffle, same flags -- so the spread between the two is eval sampling noise and
    nothing else.

    ``with_forced=False`` (``--skip-forced``) skips the second full-corpus tokenizer pass that the
    teacher-forced targets need; the passages dominate that cost and they are already encoded.
    """
    rows = frame.to_dict("records")
    rng = random.Random(seed)
    rng.shuffle(rows)

    records, forced_conversations, dropped_long, dropped_bad = [], [], 0, 0
    skipped = 0
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
        if skipped < offset:
            # counted only once it is known to be usable, so the offset indexes the same sequence
            # the un-offset run collects -- an offset over raw rows would land somewhere else
            skipped += 1
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
        f"{len(kept):,} questions "
        f"({sum(r['unanswerable'] for r in kept):,} unanswerable), "
        f"skipped {skipped:,} by --example-offset, "
        f"dropped {dropped_long:,} over {max_prompt_tokens} prompt tokens / {dropped_bad:,} unusable"
    )
    return kept


# ------------------------------------------------------------------------ evidence-port records


def chunk_passage_ids(tokenizer, text: str, chunk_tokens: int) -> List[List[int]]:
    """Turn a passage into evidence chunks.

    ``chunk_tokens <= 0`` (the default reading) keeps the whole passage as ONE chunk, exactly as
    ``prepare_evidence_data.py`` writes a SQuAD passage however long. A positive value splits it
    into fixed-size token chunks instead: a secondary robustness reading that exercises the
    selector's split over several candidates per question, closer to a real multi-chunk retrieval
    buffer than the single always-picked chunk the training corpus gives this source.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if chunk_tokens <= 0:
        return [ids] if ids else []
    chunks = [ids[i:i + chunk_tokens] for i in range(0, len(ids), chunk_tokens)]
    return [c for c in chunks if c]


def build_evidence_records(frame: pd.DataFrame, template: ChatTemplate, tokenizer, *,
                           max_examples: Optional[int], max_prompt_tokens: int, chunk_tokens: int,
                           max_evidence_tokens: int, seed: int, offset: int = 0) -> List[dict]:
    """Render the validation split for ``--evidence-port`` mode.

    Same seeded shuffle as ``build_records`` (same ``seed``, same ``random.Random`` call), so the
    two modes draw from the same ordering -- but kept as its own function rather than a branch
    inside ``build_records``, because the two render genuinely different prompts
    (``EVIDENCE_SQUAD_INSTRUCTION`` with no passage vs ``SQUAD_INSTRUCTION`` with one inline) and
    keep genuinely different per-row state (chunked passage token ids here, nothing there); a single
    function branching on a flag would make every reader work out which fields exist under which
    mode. The forced target and the evidence actually attached are NOT built here -- they depend on
    the condition, which is only known once ``attach_condition`` is called per condition.
    """
    rows = frame.to_dict("records")
    rng = random.Random(seed)
    rng.shuffle(rows)

    records, dropped_long, dropped_bad, dropped_evidence, skipped = [], 0, 0, 0, 0
    for row in rows:
        if max_examples is not None and len(records) >= max_examples:
            break
        context = str(row.get("context") or "").strip()
        question = str(row.get("question") or "").strip()
        if not context or not question:
            dropped_bad += 1
            continue
        prompt_text = evidence_prompt(question)
        prompt_ids = template.encode_prompt([{"role": "user", "content": prompt_text}])
        if len(prompt_ids) > max_prompt_tokens:
            dropped_long += 1
            continue
        if skipped < offset:
            skipped += 1
            continue
        gold_ids = chunk_passage_ids(tokenizer, context, chunk_tokens)
        if sum(len(c) for c in gold_ids) > max_evidence_tokens:
            # dropped, not truncated -- truncating removes whichever chunk landed last, which under
            # `mixed` is the gold one some of the time, mislabelling the condition rather than
            # shortening it (the same reason prepare_evidence_data.py drops instead of truncating)
            dropped_evidence += 1
            continue
        gold_texts = [tokenizer.decode(c, skip_special_tokens=True) for c in gold_ids]
        references = squad_references(row)
        records.append({
            "id": str(row.get("id", "")),
            "question": question,
            "references": references,
            "unanswerable": not references,
            "prompt_ids": prompt_ids,
            "evidence_prompt_text": prompt_text,
            "gold_chunks": list(zip(gold_texts, gold_ids)),
            "context_key": hashlib.sha1(context.encode("utf-8")).hexdigest(),
        })

    for i, record in enumerate(records):
        record["index"] = i  # this slice's position

    logger.info(
        f"{len(records):,} questions for --evidence-port "
        f"({sum(r['unanswerable'] for r in records):,} unanswerable), "
        f"skipped {skipped:,} by --example-offset, dropped {dropped_long:,} over "
        f"{max_prompt_tokens} prompt tokens, {dropped_evidence:,} over {max_evidence_tokens} "
        f"evidence tokens, {dropped_bad:,} unusable"
    )
    return records


def _build_distractor_pool(records: List[dict]) -> List[Tuple[str, str, List[int]]]:
    """Every distinct passage's gold chunks, flattened and tagged with their source passage.

    SQuAD v2 validation has several questions per passage, so the pool is deduplicated by passage:
    a passage shared by five questions contributes its chunks once, not five times as likely to be
    drawn. The tag is the passage identity (``context_key``), not the record index, so another
    question on the SAME passage cannot hand the gold passage to a record as a distractor (see
    ``_sample_distractors``). Built once and shared by every condition.
    """
    pool, seen = [], set()
    for r in records:
        if r["context_key"] in seen:
            continue
        seen.add(r["context_key"])
        pool.extend((r["context_key"], text, ids) for text, ids in r["gold_chunks"])
    return pool


def _sample_distractors(pool: List[Tuple[str, str, List[int]]], own_key: str, k: int,
                        rng: random.Random) -> List[Tuple[str, List[int]]]:
    """``k`` chunks drawn from other passages' gold chunks -- the eval-side reservoir.

    Every chunk of the record's own passage is excluded. Scans the whole pool per call rather than
    indexing into it, which costs O(pool size) per record; fine at the slice sizes this script runs
    at (about a thousand passages), and simpler than a live exclusion index for a one-shot pass.
    """
    candidates = [(text, ids) for key, text, ids in pool if key != own_key]
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates
    return rng.sample(candidates, k)


def attach_condition(records: List[dict], condition: str, template: ChatTemplate, embedder,
                     pool: List[Tuple[str, str, List[int]]], *, num_distractors: int,
                     rng: random.Random) -> None:
    """Fill in, per record, the evidence this CONDITION attaches and the target it forces.

    Mutates records in place rather than returning a copy: scoring a second condition over the same
    slice only needs ``evidence_row``/``forced_ids``/``forced_mask``/``condition_unanswerable`` to
    change, not the (unchanged) prompt side, so re-tokenizing the question every condition is the
    only repeated cost.

    Mirrors ``prepare_evidence_data.apply_condition``'s target rule for a QA row: ``gold``/``mixed``
    target the real answer unless the row is natively unanswerable (its passage is gold-shaped and
    still does not answer -- the one case that forces abstention under every condition, and the only
    supervision that separates "retrieved something relevant" from "can answer"); ``distractors``/
    ``none`` force abstention even when the row is answerable, because the point of those two
    conditions is that the BUFFER, not the question, decides whether an answer exists.
    """
    conversations = []
    for record in records:
        gold = record["gold_chunks"]
        if condition == "gold":
            chosen = [(t, i, True) for t, i in gold]
        elif condition == "none":
            chosen = []
        elif condition == "distractors":
            chosen = [
                (t, i, False)
                for t, i in _sample_distractors(pool, record["context_key"], num_distractors, rng)
            ]
        elif condition == "counterfactual":
            # the gold passage with the answer swapped, built once by prepare_counterfactual
            chosen = [(t, i, True) for t, i in record["cf"]["chunks"]]
        else:  # mixed
            distract = _sample_distractors(pool, record["context_key"], num_distractors, rng)
            chosen = [(t, i, True) for t, i in gold] + [(t, i, False) for t, i in distract]
            rng.shuffle(chosen)

        record["condition_unanswerable"] = record["unanswerable"] or condition in ("distractors", "none")
        use_real_answer = condition in ("gold", "mixed") and not record["condition_unanswerable"]
        if condition == "counterfactual":
            # the swapped evidence supports the substitute, so that is the span a faithful reader
            # produces; nothing downstream reads the teacher forced pass for this condition, but
            # the forced pair must not be left over from the previous condition
            target = record["cf"]["substitute"]
        else:
            target = (
                record["references"][0] if use_real_answer
                else abstention.pick(abstention.ABSTENTIONS_PASSAGE, rng)
            )

        ev_ids: List[int] = []
        ev_chunk_ids: List[int] = []
        for local_idx, (_, ids, _is_gold) in enumerate(chosen):
            ev_ids.extend(ids)
            ev_chunk_ids.extend([local_idx] * len(ids))
        keys = embedder.encode([t for t, _, _ in chosen]) if chosen else None
        record["evidence_row"] = (
            {"ids": ev_ids, "chunk_ids": ev_chunk_ids, "keys": keys} if ev_ids else None
        )
        conversations.append([
            {"role": "user", "content": record["evidence_prompt_text"]},
            {"role": "assistant", "content": target},
        ])

    # the SAME target under every condition, and always the real answer: the CE gap between gold and
    # none is only a reading of what the evidence is worth if both sides score the identical span.
    # The forced target above deliberately changes with the condition (that is what makes the
    # abstention numbers mean something), so a gap built on it compares "say the answer" against
    # "say a refusal" -- two different, differently-priced targets, and the refusal wins on CE
    # whatever the port does. Natively unanswerable rows have no answer span to score and are left
    # without one; the gap is read over the rest.
    answer_conversations, answer_records = [], []
    for record in records:
        if record["unanswerable"] or not record["references"]:
            record["answer_forced"] = None
            continue
        answer_records.append(record)
        answer_conversations.append([
            {"role": "user", "content": record["evidence_prompt_text"]},
            {"role": "assistant", "content": record["references"][0]},
        ])
    for record, pair in zip(answer_records, template.encode_batch(answer_conversations)):
        record["answer_forced"] = pair

    for record, pair in zip(records, template.encode_batch(conversations)):
        if pair is None:
            # practically unreachable here -- the user turn already tokenized cleanly when
            # prompt_ids was built, and the target is either a known-good reference or a fixed
            # abstention phrasing -- but a record has to end up with SOMETHING forced rather than a
            # stale value from the previous condition, so fall back to an empty (unsupervised) turn
            record["forced_ids"], record["forced_mask"] = [template.bos_id, template.eos_id], [0, 0]
            continue
        record["forced_ids"], record["forced_mask"] = pair


def report_evidence_condition(condition: str, records: List[dict], forced: dict, n_bins: int) -> dict:
    """Print and return one condition's numbers -- the per-condition analogue of ``report``.

    ``unanswerable_effective`` (``condition_unanswerable``) is what abstention correctness is scored
    against: under ``distractors``/``none`` the model is right to abstain even on a natively
    answerable question, because the buffer -- not the question -- decides whether an answer exists
    under those two conditions. The external-mass AUROC is scored against the NATIVE label instead
    (whether SQuAD calls the question unanswerable), because that is the one signal the port's
    selector could plausibly carry regardless of which condition supplied the evidence.
    """
    abstained = np.array([r["abstained"] for r in records], dtype=bool)
    unanswerable_native = np.array([r["unanswerable"] for r in records], dtype=bool)
    unanswerable_effective = np.array([r["condition_unanswerable"] for r in records], dtype=bool)
    em = np.array([r["em"] for r in records], dtype=np.float64)
    f1 = np.array([r["f1"] for r in records], dtype=np.float64)
    is_correct = np.array([r["is_correct"] for r in records], dtype=np.float64)
    p_max = np.array([r["p_max"] for r in records], dtype=np.float64)

    scores = abstention_scores(abstained, unanswerable_effective)
    answerable = ~unanswerable_effective

    print(f"\n=== SQuAD v2 validation, evidence port, condition '{condition}' ===")
    print(f"  questions: {len(records):,}  ({int(unanswerable_effective.sum()):,} unanswerable "
          f"under this condition, {int(unanswerable_native.sum()):,} natively)")
    print(f"  abstention precision: {scores['precision']:.4f}   "
          f"({scores['tp']} correct abstentions / {scores['tp'] + scores['fp']} total)")
    print(f"  abstention recall:    {scores['recall']:.4f}")
    print(f"  false abstention rate (answerable under this condition): "
          f"{scores['false_abstention_rate']:.4f}")
    if answerable.any():
        print(f"  exact match (answerable under this condition): {em[answerable].mean():.4f}   "
              f"token F1: {f1[answerable].mean():.4f}")
    else:
        print("  no questions answerable under this condition in this sample")
    print(f"  overall correctness: {is_correct.mean():.4f}")
    print(f"  answer-level AUROC of p_max vs correctness: {roc_auc(p_max, is_correct):.4f}")

    result = {
        "abstention": scores,
        "em_answerable": float(em[answerable].mean()) if answerable.any() else None,
        "f1_answerable": float(f1[answerable].mean()) if answerable.any() else None,
    }

    if condition in ("distractors", "none"):
        # here an abstention is the right answer, so the readings that matter are the ones that
        # say how often the model answered anyway and whether that answer was the real one, which
        # on a buffer without the answer can only have come from its weights
        native_answerable = ~unanswerable_native
        non_abstain = 1.0 - scores["abstention_rate"]
        em_real = float(em[native_answerable].mean()) if native_answerable.any() else None
        print(f"  non-abstain rate: {non_abstain:.4f}   EM against the real answer on natively "
              f"answerable rows (an answer from memory): "
              + (f"{em_real:.4f}" if em_real is not None else "n/a"))
        result["non_abstain_rate"] = non_abstain
        result["em_real_answer"] = em_real

    if forced:
        print(f"  teacher forced, answer span: CE {forced['ce']:.4f}  top-1 {forced['top1_acc']:.4f}  "
              f"ECE(pmax) {forced['ece_p_max']:.4f}  AUROC(pmax) {forced['auroc_p_max']:.4f}")
        result["ce"] = forced["ce"]
        result["top1_acc"] = forced["top1_acc"]

    mass_by_record = [r.get("memory_mass") for r in records]
    have_mass = np.array([m is not None for m in mass_by_record])
    if have_mass.any():
        mass_vals = np.array([m if m is not None else np.nan for m in mass_by_record], dtype=np.float64)
        auroc_mass = roc_auc(mass_vals[have_mass], unanswerable_native[have_mass].astype(np.float64))
        print(f"  external memory mass @ last prompt position: mean {mass_vals[have_mass].mean():.4f}"
              f"   AUROC (native unanswerable): {auroc_mass:.4f}   "
              f"({int(have_mass.sum())}/{len(records)} scored)")
        result["memory_mass_auroc"] = auroc_mass
        result["memory_mass_mean"] = float(mass_vals[have_mass].mean())

        loop_idxs = sorted({idx for r in records for idx in (r.get("memory_mass_by_loop") or {})})
        per_loop = {}
        for loop_idx in loop_idxs:
            vals = np.array(
                [(r.get("memory_mass_by_loop") or {}).get(loop_idx, np.nan) for r in records],
                dtype=np.float64,
            )
            mask = ~np.isnan(vals)
            if not mask.any():
                continue
            auroc_loop = roc_auc(vals[mask], unanswerable_native[mask].astype(np.float64))
            print(f"    loop {loop_idx + 1}: mean {vals[mask].mean():.4f}   AUROC {auroc_loop:.4f}")
            per_loop[loop_idx] = {"mean": float(vals[mask].mean()), "auroc": auroc_loop}
        if per_loop:
            result["memory_mass_by_loop"] = per_loop
        expert_idxs = sorted({idx for r in records for idx in (r.get("memory_mass_by_expert") or {})})
        per_expert = {}
        for expert_idx in expert_idxs:
            vals = np.array(
                [(r.get("memory_mass_by_expert") or {}).get(expert_idx, np.nan) for r in records],
                dtype=np.float64,
            )
            if np.isnan(vals).all():
                continue
            per_expert[expert_idx] = float(np.nanmean(vals))
        if len(per_expert) > 1:
            print("    per IR expert mean mass: "
                  + ", ".join(f"expert {i}: {v:.4f}" for i, v in per_expert.items()))
        if per_expert:
            result["memory_mass_by_expert"] = per_expert
    else:
        print("  external memory mass: not available (no IR module, or this condition attaches "
              "no evidence at all)")

    return result


# -------------------------------------------------------------------- counterfactual condition


@torch.inference_mode()
def answer_logprobs(model, template: ChatTemplate, records: List[dict], *, pad_id: int,
                    batch_size: int, device: str, max_seq_len: int, use_evidence: bool,
                    answer_key: str) -> None:
    """Fill ``record["ll_<answer_key>"]`` with the log-probability of one answer, teacher forced.

    The score is the sum over the answer tokens plus the closing EOS (the supervised span of the
    chat template), given the evidence prompt and, with ``use_evidence``, each record's current
    ``evidence_row``. A sum rather than a mean, because the two answers being compared differ in
    length and the question is which sequence the model would emit.

    Args:
        records: records with ``evidence_prompt_text``, the answer text field named by
            ``answer_key`` and, with ``use_evidence``, ``evidence_row``.
        use_evidence: attach each record's evidence row through the port (False scores the answer
            with no evidence at all).
        answer_key: ``"orig"`` scores ``cf_original``, ``"swap"`` scores ``cf_substitute``.
    """
    field = ANSWER_TEXT_FIELDS.get(answer_key)
    if field is None:
        raise ValueError(f"answer_key {answer_key!r} is not one of {sorted(ANSWER_TEXT_FIELDS)}")
    key = f"ll_{answer_key}"
    for record in records:
        record[key] = float("nan")
    pairs = template.encode_batch([
        [{"role": "user", "content": r["evidence_prompt_text"]},
         {"role": "assistant", "content": r[field]}] for r in records
    ])
    order = sorted((i for i, pair in enumerate(pairs) if pair is not None),
                   key=lambda i: len(pairs[i][0]))
    for start in range(0, len(order), batch_size):
        chosen = order[start:start + batch_size]
        width = max(len(pairs[i][0]) for i in chosen)
        if use_evidence:
            width += 1
        width = min(width, max_seq_len)
        ids = torch.full((len(chosen), width), pad_id, dtype=torch.long, device=device)
        doc = torch.zeros((len(chosen), width), dtype=torch.long, device=device)
        labels = torch.full((len(chosen), width), -100, dtype=torch.long, device=device)
        for row_idx, i in enumerate(chosen):
            row = torch.tensor(pairs[i][0][:width], dtype=torch.long, device=device)
            supervised = torch.tensor(pairs[i][1][:width], dtype=torch.bool, device=device)
            ids[row_idx, :row.numel()] = row
            doc[row_idx, :row.numel()] = 1
            labels[row_idx, :row.numel()] = torch.where(supervised, row, torch.full_like(row, -100))
        evidence_batch = None
        if use_evidence:
            cu_seqlens, _ = cu_seqlens_from_doc_ids(doc)
            seg = _segment_ids(cu_seqlens, len(chosen), width, device)
            evidence_batch = _pack_evidence_batch(
                model, seg[:, 0], seg[:, -1], [records[i].get("evidence_row") for i in chosen],
                num_segments=int(cu_seqlens.numel() - 1), device=device,
            )
        hidden = _final_hidden(model, ids, doc, evidence=evidence_batch)
        target = labels[:, 1:]
        rows, cols = (target != -100).nonzero(as_tuple=True)
        logits = model.lm_head(hidden[:, :-1][rows, cols]).float()
        logp = torch.log_softmax(logits, dim=-1).gather(-1, target[rows, cols].unsqueeze(-1)).squeeze(-1)
        sums = torch.zeros(len(chosen), dtype=torch.float32, device=device).index_add_(0, rows, logp)
        for row_idx, i in enumerate(chosen):
            records[i][key] = float(sums[row_idx])


def assign_frequency_strata(records: List[dict], tokenizer, bin_path: str) -> None:
    """Set ``stratum`` (and ``cf_freq``) on every eligible record from the original answer's count.

    The count is the answer's token sequence in a reference corpus (see ``entity_frequency.py``).
    A missing corpus leaves every record in one stratum, "all", and says so.
    """
    eligible = [r for r in records if r.get("cf")]
    if not os.path.isfile(bin_path):
        logger.warning(f"{bin_path} does not exist: no frequency strata, one pooled stratum")
        for record in eligible:
            record["stratum"] = "all"
        return
    from scripts.entity_frequency import answer_frequencies

    counts = answer_frequencies(bin_path, [r["cf_original"] for r in eligible], tokenizer,
                                cache_dir=FREQUENCY_CACHE_DIR)
    for record in eligible:
        record["cf_freq"] = counts[record["cf_original"]]
        record["stratum"] = frequency_stratum(counts[record["cf_original"]])


class LookupEmbedder:
    """Chunk keys from a table of known texts, for an evidence source whose keys already exist.

    An injected-fact store ships one canonical key per card, and its swapped card is read through
    the same key (the selector chooses by person, the reader reads the swapped text), so nothing is
    re-embedded.
    """

    def __init__(self, keys_by_text: dict):
        self.keys_by_text = keys_by_text

    def encode(self, texts: List[str]) -> np.ndarray:
        """[len(texts), 384] float32; a text that is not in the table is an error."""
        return np.stack([self.keys_by_text[t] for t in texts]).astype(np.float32)


def build_biography_records(args, template: ChatTemplate, tokenizer, *, max_examples: Optional[int],
                            seed: int) -> Tuple[List[dict], "LookupEmbedder"]:
    """Records for the counterfactual condition over injected biographies (the facts-file source).

    One record per (person, entity-valued attribute): the question, the person's store card as the
    gold chunk, and the card re-rendered with that attribute swapped for another pool value as the
    counterfactual chunk. The stratum is the person's exposure tier. Meaningful on a checkpoint that
    was trained to answer in the chat format.

    Args:
        args: needs ``facts``, ``pools`` and ``store_dir``.
        max_examples: cap on records, or None for every (person, attribute).
        seed: seeds the sample and the substitute draw.
    """
    try:
        from modules.data import biographies as bio
        from modules.data.store import load_store
    except ImportError as e:
        raise SystemExit(f"--counterfactual-source bios needs the biography and store modules: {e}")
    people = bio.load_facts(args.facts)
    pools = bio.load_pools(args.pools)
    store = load_store(args.store_dir)
    rng = random.Random(seed)
    pairs = [(p, a) for p in people for a in bio.ENTITY_ATTRIBUTES]
    rng.shuffle(pairs)
    if max_examples is not None:
        pairs = pairs[:max_examples]

    keys_by_text, records = {}, []
    for person, attribute in pairs:
        value = person.attributes[attribute]
        candidates = [v for v in pools[attribute] if v != value]
        substitute = candidates[rng.randrange(len(candidates))]
        card = store.chunks[person.person_id]["text"]
        swapped = bio.render_store_chunk(person, {attribute: substitute})
        key = np.asarray(store.keys[person.person_id], dtype=np.float32)
        keys_by_text[card] = keys_by_text[swapped] = key
        question = bio.question_for(attribute, person.name)
        prompt_text = evidence_prompt(question)
        card_ids = tokenizer(card, add_special_tokens=False)["input_ids"]
        swapped_ids = tokenizer(swapped, add_special_tokens=False)["input_ids"]
        records.append({
            "id": f"bio{person.person_id}:{attribute}", "question": question,
            "references": [value], "unanswerable": False,
            "prompt_ids": template.encode_prompt([{"role": "user", "content": prompt_text}]),
            "evidence_prompt_text": prompt_text, "gold_chunks": [(card, card_ids)],
            "context_key": str(person.person_id), "stratum": str(person.tier),
            "cf": {"original": value, "substitute": substitute, "type": attribute,
                   "chunks": [(swapped, swapped_ids)], "count": 1},
            "cf_original": value, "cf_substitute": substitute, "cf_type": attribute,
        })
    for i, record in enumerate(records):
        record["index"] = i
    # distractor chunks come from other people's cards, which a --max-examples slice leaves out
    for text, key in zip((c["text"] for c in store.chunks), store.keys):
        keys_by_text.setdefault(text, np.asarray(key, dtype=np.float32))
    logger.info(f"{len(records):,} biography records over {len(people):,} people")
    return records, LookupEmbedder(keys_by_text)


def report_counterfactual(records: List[dict], ineligible: dict, n_total: int) -> dict:
    """Print and return the counterfactual condition's readings.

    One line per stratum, over the items answered correctly with the unswapped chunk (the
    population the memorization ratio is defined on) and again over every eligible item.

    Args:
        records: the eligible records, after generation and ``answer_logprobs``.
        ineligible: ``{reason: count}`` for the records that took no part.
        n_total: how many records the slice had.
    """
    readings = counterfactual_readings(records)
    print("\n=== counterfactual evidence: gold chunk with the answer swapped for another of its type ===")
    print(f"  {len(records):,} of {n_total:,} questions eligible; skipped by reason: "
          + (", ".join(f"{k} {v:,}" for k, v in sorted(ineligible.items())) or "none"))

    def fmt(value, spec):
        return "n/a" if value is None else format(value, spec)

    for label, title in (("correct_with_gold", "items answered correctly with the unswapped chunk"),
                         ("all_eligible", "every eligible item")):
        print(f"  --- {title} ---")
        print(f"  {'stratum':<10} {'n':>6} {'follow':>8} {'mr_gen':>8} {'mr_ll':>8} {'other':>8}  gate")
        for name, r in readings[label].items():
            gate = f"{r['gate']} (n={r['n']})" if label == "correct_with_gold" else ""
            print(f"  {name:<10} {r['n']:>6} {fmt(r['follow_rate'], '.3f'):>8} "
                  f"{fmt(r['mr_gen'], '.3f'):>8} {fmt(r['mr_ll'], '.3f'):>8} "
                  f"{fmt(r['other_rate'], '.3f'):>8}  {gate}")
    print("  follow = the completion is the substitute; mr_gen = stuck / (stuck + follow), stuck = "
          "the original answer; mr_ll = sigmoid(ll_orig - ll_swap); gate PASS = mr_gen <= 0.05 and "
          "follow >= 0.90")
    return {"eligible": len(records), "total": n_total, "ineligible": dict(ineligible),
            "readings": readings}


def report_counterfactual_likelihood(records: List[dict], ineligible: dict, n_total: int) -> dict:
    """Print and return the counterfactual readings that need no generation.

    Every eligible item, by stratum: ``mr_ll`` (mean of ``sigmoid(ll_orig - ll_swap)``, near 1 when
    the model keeps its memory, near 0 when it takes the swapped chunk) and ``prefers_swap``, the
    share of items whose substitute outscores the original. ``mr_gen`` and the gate are generation
    readings and are absent.

    Args:
        records: the eligible records after ``answer_logprobs`` for both answer fields.
        ineligible: ``{reason: count}`` for the records that took no part.
        n_total: how many records the slice had.
    """
    pooled = counterfactual_readings(records)["all_eligible"]
    by_stratum: Dict[str, List[dict]] = {}
    for r in records:
        by_stratum.setdefault(str(r.get("stratum", "all")), []).append(r)
    by_stratum["all"] = list(records)
    print("\n=== counterfactual evidence, likelihood only: gold chunk with the answer swapped ===")
    print(f"  {len(records):,} of {n_total:,} questions eligible; skipped by reason: "
          + (", ".join(f"{k} {v:,}" for k, v in sorted(ineligible.items())) or "none"))
    print(f"  {'stratum':<10} {'n':>6} {'mr_ll':>8} {'prefers_swap':>13}")
    readings = {}
    for name in pooled:
        rows = [r for r in by_stratum.get(name, []) if r.get("mr_ll") is not None]
        prefers = (sum(1 for r in rows if r["ll_swap"] > r["ll_orig"]) / len(rows)) if rows else None
        mr_ll = pooled[name]["mr_ll"]
        readings[name] = {"n": pooled[name]["n"], "mr_ll": mr_ll, "prefers_swap": prefers}
        print(f"  {name:<10} {pooled[name]['n']:>6} "
              f"{'n/a' if mr_ll is None else format(mr_ll, '.3f'):>8} "
              f"{'n/a' if prefers is None else format(prefers, '.3f'):>13}")
    print("  mr_ll = sigmoid(ll_orig - ll_swap), 1 keeps memory, 0 follows the chunk; no generation ran")
    return {"eligible": len(records), "total": n_total, "ineligible": dict(ineligible),
            "likelihood_only": True, "readings": {"all_eligible": readings}}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value)} is not JSON serializable")


def run_evidence_port_eval(args, tokenizer, template: ChatTemplate, frame: pd.DataFrame) -> None:
    """``--evidence-port`` mode: the passage leaves the prompt and enters the port instead.

    Every requested condition is scored over the SAME slice in one pass -- ``build_evidence_records``
    runs once, and ``attach_condition`` only rewrites the per-condition fields -- so the gold-vs-none
    CE gap (the number this mode exists to produce) comes from two passes over identical questions
    rather than two separate invocations that could have drawn a different sample.
    """
    conditions = [c.strip() for c in args.evidence_condition.split(",") if c.strip()]
    bad = [c for c in conditions if c not in EVIDENCE_CONDITIONS]
    if bad:
        raise SystemExit(f"unknown --evidence-condition {bad} -- choose from {EVIDENCE_CONDITIONS}")
    if not conditions:
        raise SystemExit("--evidence-condition resolved to an empty list")
    likelihood_only = getattr(args, "counterfactual_likelihood_only", False)
    if likelihood_only and "counterfactual" not in conditions:
        raise SystemExit("--counterfactual-likelihood-only needs the counterfactual condition")
    if "counterfactual" in conditions:
        if "gold" not in conditions and not likelihood_only:
            raise SystemExit("counterfactual needs gold in the same run: the memorization ratio is "
                             "read over the items answered correctly with the unswapped chunk "
                             "(--counterfactual-likelihood-only reads mr_ll without it)")
        # last, so every record's gold-condition correctness is recorded before it is needed
        conditions = [c for c in conditions if c != "counterfactual"] + ["counterfactual"]

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)
    if not model.moe.evidence_port:
        raise SystemExit(
            f"{args.checkpoint} has no evidence port -- build one with "
            f"scripts/migrate_evidence_port.py before running --evidence-port"
        )

    biographies = args.counterfactual_source == "bios"
    if biographies:
        if "counterfactual" not in conditions:
            raise SystemExit("--counterfactual-source bios only makes sense with the counterfactual "
                             "condition")
        records, embedder = build_biography_records(
            args, template, tokenizer, max_examples=args.max_examples, seed=args.seed,
        )
    else:
        records = build_evidence_records(
            frame, template, tokenizer, max_examples=args.max_examples,
            max_prompt_tokens=args.max_prompt_tokens, chunk_tokens=args.chunk_tokens,
            max_evidence_tokens=args.max_evidence_tokens, seed=args.seed, offset=args.example_offset,
        )
    if not records:
        raise SystemExit("no usable validation questions for --evidence-port -- check "
                         "--max-prompt-tokens / --max-evidence-tokens / --squad-dir")

    if not biographies:
        logger.info(f"loading the external embedder ({ChunkEmbedder.REPO}) for chunk keys")
        embedder = ChunkEmbedder(device=args.device)
    pool = _build_distractor_pool(records)
    # one rng shared across conditions (like build_records', but seeded off it rather than reused)
    # so the distractor draw and the abstention-phrase draw are reproducible run to run without
    # colliding with the shuffle seed the records themselves were drawn with
    rng = random.Random(args.seed + 7)

    ineligible = {}
    if "counterfactual" in conditions:
        # decided once, before any condition runs: eligibility must not depend on what a condition
        # did to the record
        gazetteer = Gazetteer.from_pairs(
            (r["question"], r["references"][0]) for r in records if r["references"]
        )
        counts = prepare_counterfactual(records, gazetteer, random.Random(args.seed + 11), tokenizer)
        ineligible = {k: v for k, v in counts.items() if k != "eligible"}
        logger.info(f"counterfactual: {counts['eligible']:,} of {len(records):,} records eligible "
                    f"({ineligible})")
        if counts["eligible"] == 0:
            raise SystemExit("no record is eligible for the counterfactual condition")
        if not biographies:
            # an injected-fact source stratifies by exposure tier, set when its records were built
            assign_frequency_strata(records, tokenizer, args.counterfactual_freq_bin)

    results, records_by_condition = {}, {}
    for condition in conditions:
        logger.info(f"=== --evidence-port condition: {condition} ===")
        scored = [r for r in records if r.get("cf")] if condition == "counterfactual" else records
        attach_condition(scored, condition, template, embedder, pool,
                         num_distractors=args.num_distractors, rng=rng)

        if condition == "counterfactual" and likelihood_only:
            for answer_key in ANSWER_TEXT_FIELDS:
                answer_logprobs(
                    model, template, scored, pad_id=tokenizer.pad_token_id,
                    batch_size=args.batch_size, device=args.device,
                    max_seq_len=ModelConfig.Params["max_seq_len"], use_evidence=True,
                    answer_key=answer_key,
                )
            for record in scored:
                both = (record["ll_orig"], record["ll_swap"])
                record["mr_ll"] = None if any(math.isnan(v) for v in both) else sigmoid_ratio(*both)
            results[condition] = report_counterfactual_likelihood(scored, ineligible, len(records))
            records_by_condition[condition] = [record_for_json(r) for r in scored]
            continue

        logger.info(f"Generating {len(scored):,} answers under condition {condition!r} "
                    f"(batch {args.batch_size}, <= {args.max_new_tokens} new tokens)")
        run_generation(
            model, tokenizer, template, scored, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
            device=args.device, attach_evidence=True,
        )
        for record in scored:
            completion = record["completion"]
            record["abstained"] = abstention.is_abstention(completion)
            record["em"] = exact_match(completion, record["references"]) if record["references"] else 0.0
            record["f1"] = token_f1(completion, record["references"]) if record["references"] else 0.0
            record["is_correct"] = (
                float(record["abstained"]) if record["condition_unanswerable"]
                else (0.0 if record["abstained"] else record["em"])
            )
            if condition == "gold":
                record["gold_em"] = record["em"]

        if condition == "counterfactual":
            for record in scored:
                record["cf_follow"] = exact_match(record["completion"], [record["cf_substitute"]])
                record["cf_stuck"] = exact_match(record["completion"], record["references"])
                record["cf_other"] = 1.0 - record["cf_follow"] - record["cf_stuck"]
            for answer_key in ANSWER_TEXT_FIELDS:
                answer_logprobs(
                    model, template, scored, pad_id=tokenizer.pad_token_id,
                    batch_size=args.batch_size, device=args.device,
                    max_seq_len=ModelConfig.Params["max_seq_len"], use_evidence=True,
                    answer_key=answer_key,
                )
            for record in scored:
                both = (record["ll_orig"], record["ll_swap"])
                record["mr_ll"] = None if any(math.isnan(v) for v in both) else sigmoid_ratio(*both)
            results[condition] = report_counterfactual(scored, ineligible, len(records))
            records_by_condition[condition] = [record_for_json(r) for r in scored]
            continue

        forced = {}
        if not args.skip_forced:
            logger.info(f"Teacher-forced calibration pass, condition {condition!r}")
            forced = teacher_forced_calibration(
                model, template, records, pad_id=tokenizer.pad_token_id,
                batch_size=args.batch_size, device=args.device,
                max_seq_len=ModelConfig.Params["max_seq_len"], n_bins=args.n_bins,
                use_evidence=True,
            )

        results[condition] = report_evidence_condition(condition, records, forced, args.n_bins)
        # copied now: the next condition rewrites the same record objects
        records_by_condition[condition] = [record_for_json(r) for r in records]

        # the fixed-target pass: the real answer span, scored under this condition's evidence. This
        # is the quantity the gold-vs-none gap is read on -- see attach_condition for why the
        # condition-dependent target above cannot carry it. Scored on shallow copies so the records
        # keep the forced pair their own condition's numbers were computed from.
        if not args.skip_forced:
            answerable = [
                dict(r, forced_ids=r["answer_forced"][0], forced_mask=r["answer_forced"][1])
                for r in records if r.get("answer_forced") is not None
            ]
            if answerable:
                logger.info(f"Teacher-forced answer-span pass, condition {condition!r} "
                            f"({len(answerable):,} answerable rows, real answer as the target)")
                fixed = teacher_forced_calibration(
                    model, template, answerable, pad_id=tokenizer.pad_token_id,
                    batch_size=args.batch_size, device=args.device,
                    max_seq_len=ModelConfig.Params["max_seq_len"], n_bins=args.n_bins,
                    use_evidence=True,
                )
                if "ce" in fixed:
                    results[condition]["answer_span_ce"] = fixed["ce"]
                    print(f"  answer span CE, real answer as target under this condition: "
                          f"{fixed['ce']:.4f}   ({len(answerable):,} rows)")

    if all("answer_span_ce" in results.get(c, {}) for c in ("gold", "none")):
        gap = results["none"]["answer_span_ce"] - results["gold"]["answer_span_ce"]
        print("\n=== gold vs none, answer-span CE gap ===")
        print(f"  {gap:+.4f} nats -- one target, evidence present vs absent; this is the number "
              f"--evidence-port exists to read")

    if args.json_out:
        payload = {"checkpoint": args.checkpoint, "conditions": conditions, "results": results,
                   "records": records_by_condition}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        logger.info(f"wrote per-condition results and {sum(map(len, records_by_condition.values())):,} "
                    f"per-record entries to {args.json_out}")


def report(records: List[dict], forced: dict, baseline: Optional[dict], n_bins: int,
           offset: int = 0) -> None:
    abstained = np.array([r["abstained"] for r in records], dtype=bool)
    unanswerable = np.array([r["unanswerable"] for r in records], dtype=bool)
    em = np.array([r["em"] for r in records], dtype=np.float64)
    f1 = np.array([r["f1"] for r in records], dtype=np.float64)
    is_correct = np.array([r["is_correct"] for r in records], dtype=np.float64)
    p_max = np.array([r["p_max"] for r in records], dtype=np.float64)

    scores = abstention_scores(abstained, unanswerable)
    answerable = ~unanswerable

    print("\n=== SQuAD v2 validation, generated answers ===")
    print(f"  slice: questions {offset:,}-{offset + len(records):,} of the seeded shuffle")
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
    parser.add_argument("--example-offset", type=int, default=0,
                        help="skip this many usable questions first, taking a disjoint slice of the "
                             "same seeded shuffle -- the eval-sampling noise measurement")
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
    parser.add_argument("--evidence-port", action="store_true",
                        help="the passage leaves the prompt and enters the evidence port instead "
                             "(needs a checkpoint built by scripts/migrate_evidence_port.py). "
                             "Every other flag above still applies; --evidence-condition and the "
                             "flags below only matter when this is set")
    parser.add_argument("--evidence-condition", default="gold,none",
                        help=f"comma separated subset of {EVIDENCE_CONDITIONS} to score in one pass "
                             "(default: the gold-vs-none acceptance gap)")
    parser.add_argument("--chunk-tokens", type=int, default=0,
                        help="evidence chunk size in tokens. 0 (default) keeps each passage as one "
                             "chunk, matching how the training corpus writes a SQuAD passage; a "
                             "positive value (e.g. 128) re-chunks as a robustness reading")
    parser.add_argument("--num-distractors", type=int, default=3,
                        help="distractor chunks drawn per row for distractors/mixed, matching "
                             "prepare_evidence_data.py's default")
    parser.add_argument("--max-evidence-tokens", type=int, default=2048,
                        help="drop a question if its own gold chunks exceed this many tokens")
    parser.add_argument("--counterfactual-source", choices=("squad", "bios"), default="squad",
                        help="records for the counterfactual condition: SQuAD questions whose answer "
                             "is swapped inside the passage (default), or injected biographies from "
                             "--facts / --pools / --store-dir (meaningful on a chat trained "
                             "checkpoint only)")
    parser.add_argument("--counterfactual-likelihood-only", action="store_true",
                        help="score the counterfactual condition by likelihood alone: no generation "
                             "for it, no gold condition required, mr_ll by stratum over every "
                             "eligible item. For checkpoints whose generations read nothing (they "
                             "abstain, or were never chat trained)")
    parser.add_argument("--counterfactual-freq-bin", default=DEFAULT_FREQUENCY_BIN,
                        help="token corpus the SQuAD answers are counted in to stratify the "
                             "counterfactual readings by how common the answer is (a missing file "
                             "means one pooled stratum)")
    parser.add_argument("--facts", default=DEFAULT_FACTS,
                        help="bios source: the facts file written by prepare_injection_data.py")
    parser.add_argument("--pools", default=DEFAULT_POOLS,
                        help="bios source: the value pools written next to the facts file")
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR,
                        help="bios source: the biography store (one card and key per person)")
    args = parser.parse_args()

    if args.checkpoint is None:
        raise SystemExit(f"No checkpoint found in {SFT_CHECKPOINT_DIR} and none passed via --checkpoint")

    logger.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = ChatTemplate(tokenizer)

    # the eval split lives with the other benchmark downloads, not under data/prepared: the corpus
    # builders delete shards from there and archive_corpus.py packs it wholesale
    scratch_dir = os.path.join(BASE_DIR, "data", "benchmarks", "squad_v2_validation")
    # the injected-fact source never reads SQuAD, so it needs no download
    frame = None
    if not (args.evidence_port and args.counterfactual_source == "bios"):
        frame = load_squad_split(scratch_dir, args.hf_token or get_hf_token(), args.squad_dir)

    if args.evidence_port:
        run_evidence_port_eval(args, tokenizer, template, frame)
        return

    records = build_records(
        frame, template, max_examples=args.max_examples,
        max_prompt_tokens=args.max_prompt_tokens, seed=args.seed,
        with_forced=not args.skip_forced, offset=args.example_offset,
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

    report(records, forced, baseline, args.n_bins, args.example_offset)

    if args.json_out:
        payload = {
            "checkpoint": args.checkpoint,
            "baseline_checkpoint": args.baseline_checkpoint,
            "temperature": args.temperature,
            "seed": args.seed,
            "example_offset": args.example_offset,
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

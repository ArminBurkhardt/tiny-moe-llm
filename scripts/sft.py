"""Supervised fine-tuning, and NEXT.md Phase 2's abstention repair pass on top of it.

Written for a **local** run -- the pretrained checkpoint and ``manifest.json`` come down from the
Hub once pretraining finishes (``--from-hub`` does that), and the fine-tune itself is a couple of
hours on the dev GPU. It will however run unattended on a rented box: it honours the same
``modules/runtime/control`` stop contract as pretraining (SIGTERM -> checkpoint and exit 20, STOP
sentinel -> exit 10, SIGUSR1 -> checkpoint and keep going) and returns those exit codes, so a
trivial restart wrapper is enough on an interruptible instance. What it deliberately does *not*
have is a phase supervisor: there are no phases here, only epochs, and epoch position is already
part of the checkpoint.

What it deliberately *reuses* rather than reimplements:

  * ``pretrain.train_step`` verbatim. The cheapest way to guarantee every loss term stays
    *identical* between pretraining and SFT -- per-loop CE weights, aux loss, loop-count sampling --
    is to have exactly one copy of it. Prompt masking needs no changes at all:
    the dataset emits ``-100`` labels over prompt tokens and every loss term already routes through
    ``ignore_index=-100``, including the MTP heads (they read the same ``labels`` tensor).
  * The model's **global token counter**, continued rather than reset. The router-noise anneal is
    driven from it, and it has long since finished at ~16B tokens. SFT progress is tracked
    separately as ``token_count - start_token_count``.

What is genuinely different:

  * **fp32 master weights for every parameter, not just the undecayed ones** -- see
    ``build_sft_param_groups``. This is a correctness requirement at SFT's learning rate, not a
    refinement.
  * **A masked, shuffled, non-splitting dataset** (``modules/data/sft_dataset.py``).
  * **A validation pass** on ``sft_val`` at checkpoint cadence, reporting the calibration signals
    (``p_max``/top-1) the abstention acceptance criterion is about.

**``--repair`` runs NEXT.md Phase 2** through this same function: the abstention repair finetune is
the SFT run with a repaired corpus (``repair_train``/``repair_val``, from
``prepare_sft_data.py --profile repair``), ``lr=1e-5``, one epoch, and per-conversation loss
weighting. It reads ``RepairConfig`` instead of ``SFTConfig``, writes into ``ckpts/repair`` under
phase ``"repair"``, and is seeded with ``-c <the SFT checkpoint>``. One code path rather than a
second script, for the same reason ``train_step`` is shared: what has to change between the two runs
is the data and three numbers, and anything else that drifts makes the comparison meaningless.

Run from the repo root:

    python scripts/sft.py --from-hub          # pull the pretrained checkpoint, then train
    python scripts/sft.py -c ckpts/training/checkpoint_phase2_final.pt
    python scripts/sft.py                     # resume from the newest checkpoint in ckpts/sft
    python scripts/sft.py --repair -c ckpts/trained/checkpoint_sft_final_phase0.pt

**Data parallel training (``--evidence`` on a multi GPU node).** The same file runs under
``accelerate launch`` or ``torchrun``; one process per GPU, no DDP wrap. On the cluster
the launch is by hand and detached, one of

    TINY_LLM_CONFIG=config_run2.yaml setsid nohup accelerate launch --num_processes 8         --mixed_precision no scripts/sft.py --evidence -c SEED --run-name NAME ...         > run.log 2>&1 < /dev/null & disown
    TINY_LLM_CONFIG=config_run2.yaml setsid nohup torchrun --nproc_per_node 8         scripts/sft.py --evidence -c SEED --run-name NAME ... > run.log 2>&1 < /dev/null & disown

(``--mixed_precision no``: the model is already bf16 and the trainer keeps its own fp32 masters).
**``batch_size`` is per GPU**, so the global batch is ``batch_size * num_processes *
grad_accumulation_steps`` rows (8 x 4096 x 2 x 8 = 524,288 tokens per update on 8 GPUs); the
cosine spans the same tokens as a one process run because ``total_steps`` counts per rank rows.
With one process nothing below applies and the run is exactly what it was. With more:

  * every rank streams its own interleaved share of the document positions
    (``SFTDataset(rank, world_size)``; the resume cursor stays one scalar and a relaunch on
    another number of ranks resumes at a valid position); the accelerate loader is NOT used (its
    iterable dataset sharding would shard the stream a second time), the rank loaders are kept in
    lockstep by ``lockstep_batches`` (a MIN over the ranks every ``LOG_INTERVAL`` batches), so
    every rank takes the same number of micro steps and a few trailing batches of an epoch are
    dropped, like ``drop_last``;
  * the model is NOT wrapped in DDP (the LM heads run outside the model forward, which a DDP
    reducer would never see): ``pretrain.all_reduce_grads`` averages every gradient in fp32
    buckets inside ``train_step``, a rank whose slice has no evidence, gold flag or answer start
    contributing zeros; only the optimizer goes through ``accelerator.prepare``. Nothing
    broadcasts weights or buffers, so ``check_ranks_identical`` verifies at startup that every
    rank holds the same parameters, buffers and fp32 masters, and the balance bias moves from
    counts summed over the ranks;
  * at the log cadence one float64 all-reduce carries the token count, the evidence count, the
    stop and checkpoint requests of any rank, and the means of the logged scalars; eval batches
    are sharded by batch index and their sums gathered (``merge_eval_states``); the loss-free
    balancer's counts are summed before every bias update; after an IR cluster refresh rank 0's
    IR state is broadcast;
  * saves, ``run_state.json``, graphs, uploads and retention are rank 0's, between two barriers.

``--grad-accumulation-steps N`` overrides the profile's ``grad_accumulation_steps`` so a job script
can hold the global batch fixed across node sizes (8 x 4096 x N x GPUs: N = 2 on 8 GPUs, 1 on 16).
``total_steps`` and the cosine follow it; the plan (``total_steps``, ``rows_per_epoch``,
``world_size``, ``grad_accumulation_steps``) is saved in every checkpoint and a resume under
another value warns and keeps the new ``total_steps``.

**Head-first epoch 0.** When ``{data_dir}/{split}.order.json`` holds ``epoch0_head_docs`` (H), the
training split's epoch 0 order is the permutation of its first H documents that a split of just
those H documents would use, then a permutation of the rest from a distinct stream (see
``modules/data/sft_dataset.py``). A warm-up split built as the first H documents of the main split
then lets the ``--switch-split`` relaunch at position p < H visit the head documents it has not
seen before the rest. Epochs from 1 on and splits without the sidecar are unchanged. H is saved in
the checkpoint (``epoch0_head_docs``) and a resume on the same split under another H is refused.

**Warm-up to main switch (run notes).** ``--plan-split SPLIT`` (same data dir) makes the packed row
estimate, ``total_steps`` and so the LR schedule (warmup length and cosine) come from SPLIT instead
of the training split; the job scripts pass the main split on every launch of the main run, warm
and main, so the schedule spans the whole run from step 0 and the switch changes nothing in the
plan. It is saved in the checkpoint (``plan_split``) and a resume under another one warns.
``--max-epochs-this-split N`` (warm launches pass 1): on reaching the end of epoch N - 1 the run
saves at that epoch, logs "warm split exhausted" and exits 10, never entering the next epoch of the
warm split. ``--switch-split`` is then checked against the checkpoint and refused with exit 30
unless the saved epoch is 0, the saved position lies within the new split's head (its
``order.json`` H, else its document count), the warm run left no ``_final.pt``, and the warm
split's saved document count equals the new split's H or the new split's own count (the
checkpoint's ``num_docs``). The packed row estimate is cached in ``{data_dir}/{plan split}.
rows_estimate.json`` under a key of the index files' sizes and content fingerprints and the packing and order
parameters; ``--estimate-rows-only`` (CPU, exit 0, ``--estimate-world-sizes 8,16``) writes it, and
a launch of more than one rank with no valid cache for its rank count exits 2 instead of computing
it. The kill check (``kill_tokens``) may run during the warm-up and
``kill_checked`` then carries across the switch, by design. Every exit path of a multi process run
destroys the process group.
"""
import os
import re
import sys
import json
import time
import math
import random
import argparse
import hashlib
from collections import deque
from datetime import timedelta

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import torch
from torch import optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs

import transformer_engine.pytorch as te

from modules.data.dataset import Dataset
from modules.data.evidence_dataset import EvidenceDataset, evidence_from_batch
from modules.data.sft_dataset import SFTDataset
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.information_retrieval import is_rebuilt_ir_param
from modules.model.moe import is_fresh_loop_param
from modules.model.mtp import compute_mtp_loss
from modules.model.optim import build_optimizer, check_resume_kind, partition_summary
from modules.model.transformer import TinyMoETransformer
from modules.runtime import checkpoints as ckpt_lib
from modules.runtime.control import EXIT_OK, EXIT_PREEMPTED, EXIT_RESUME_FAILED, EXIT_USER_STOP, RunControl
from modules.runtime.hf_sync import HFSync
from modules.runtime.status import eta_seconds, format_duration, write_status
from config import EvidenceConfig, IRConfig, ModelConfig, RepairConfig, SFTConfig, TrainingConfig
from scripts.pretrain import (
    USE_LOW_PRECISION, answer_start_positions, chosen_recipe, log_precision_mode, next_grid_point,
    predicting_positions, sample_n_loops, save_expert_selection_graph, save_loss_graph, train_step,
)
# the same ranking metric eval_abstention.py reports G3b with, imported rather than restated so the
# validation pass and the acceptance script cannot disagree about what AUROC means here
from scripts.eval_calibration import roc_auc
# imported rather than restated, so the condition index -> name mapping used for validation and
# training logs can never drift from what the corpus builder actually wrote into `.cond`
from scripts.prepare_evidence_data import CONDITIONS, SOURCE_KEYS
from utils import (BASE_DIR, BF16, HF_UPLOAD_REPO, TOKENIZER_DIR, get_hf_token, load_model_state, with_balance_default,
                   logger, model_params_for_state_dict)

# the phase label baked into checkpoint filenames and the run-state sidecar. Distinct from
# ("phase1", "phase2") so ckpt_lib's newest-that-loads search can never pick up a pretraining
# checkpoint out of a shared directory, and so a downstream consumer can tell them apart by name.
SFT_PHASE = "sft"
# checkpoints live in their own directory: ckpts/training belongs to the pretraining run (its
# run_state.json, STOP sentinel and retention policy all assume that run), and mixing SFT files in
# would confuse checkpoints.resume_phase_index if the supervisor ever ran against the same box.
SFT_CHECKPOINT_DIR = os.path.join(BASE_DIR, "ckpts", "sft")
# --repair's counterparts. A distinct phase label AND a distinct directory, for the same two
# reasons: load_sft_checkpoint refuses to adopt another run's optimizer state by name, and a repair
# checkpoint must never be picked up as the resume point of an interrupted SFT run (its LR schedule,
# epoch count and objective are all different).
REPAIR_PHASE = "repair"
REPAIR_CHECKPOINT_DIR = os.path.join(BASE_DIR, "ckpts", "repair")
# --ir's counterparts, same contract again: the IR sharpening pass carries a second LR group and a
# temperature anneal, so its optimizer state means nothing to either other profile.
IR_PHASE = "ir"
IR_CHECKPOINT_DIR = os.path.join(BASE_DIR, "ckpts", "ir")
# --evidence's counterparts. Same contract a fourth time, and here the separation matters most: this
# profile's corpus carries a second token stream, so a checkpoint from it describes a model with
# tensors the other three do not have.
EVIDENCE_PHASE = "evidence"
EVIDENCE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "ckpts", "evidence")
# --from-hub lands the pretrained checkpoint HERE, deliberately not in SFT_CHECKPOINT_DIR: it is
# named checkpoint_phase2_final.pt, which matches ckpt_lib's "checkpoint_*.pt" resume scan, so a
# second launch would offer pretraining's own optimizer/scheduler state to load_sft_checkpoint as
# if it were a resumable SFT run.
PRETRAINED_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained")
DEFAULT_HUB_CHECKPOINT = "checkpoints/final/checkpoint_phase2_final.pt"
NUM_DATA_WORKERS = 4
LOG_INTERVAL = 10
# fraction of the exact top-k the two stage candidate set must contain, measured at every cluster
# refresh. Below this the centroid stage is dropping entries the read wanted, and the fix is more
# probed clusters, not more training (docs/plans/NEXT.md Phase 3).
IR_MIN_CANDIDATE_RECALL = 0.9
# answer-span CE gained by putting the gold passage in the prompt instead of nothing, measured on
# this checkpoint family (docs/measurements/evidence_ceiling.md). A reading through the port is
# reported as a fraction of it.
EVIDENCE_CEILING_NATS = 3.2306
# optimizer steps at which the selector's input-side gradients are logged once each: early enough to
# catch a tensor that never receives gradient, late enough to see whether one started to
GRAD_PROBE_STEPS = (10, 100)
GRAD_PROBE_SUFFIXES = ("key_adapter.weight", "value_adapter.weight", "down_proj.weight",
                       "loop_query_bias.weight")
# every checkpoint carries each parameter's fp32 master under "fp32_masters" (about 4 bytes per
# parameter on disk: 0.4 GB at 99M), so a resume continues the masters instead of reseeding them
# from bf16. Off writes checkpoints exactly as before; a resume then reseeds as it always did
SAVE_FP32_MASTERS = True
# the data fields a checkpoint records, compared on resume (see split_changes)
DATA_KEYS = ("data_dir", "train_split", "val_split")


def make_dataset(data_dir: str, split: str, tokenizer, cfg, shuffle: bool = True,
                 rank: int = 0, world_size: int = 1):
    """``SFTDataset`` when the split has a loss mask, the pretraining ``Dataset`` when it does not.

    The IR sharpening pass trains on a general LM mix with chat replay, which
    ``scripts/prepare_data.py`` writes as a plain ``{split}.bin``/``.idx`` pair with no mask -- and
    it should not have one: every token of a web document is supervised, which is exactly what the
    pretraining dataset's labels already mean. The two readers also pack differently for good
    reasons (SFT never splits a conversation across rows because a split tail loses its prompt and
    its supervised EOS; a web document has neither problem and splitting it wastes nothing), so
    picking the reader by whether a mask exists picks the right packing at the same time.

    Everything downstream is unchanged: both yield batch-aligned ``input_ids``/``labels``/
    ``document_ids``/``doc_idx``/``worker_id``, and ``loss_weights`` -- the only key the LM reader
    omits -- is read only under ``conversation_loss_weighting``, which is off for this profile.

    Args:
        shuffle: ignored for the LM reader, which reads in on-disk order on purpose (the corpus
            builder already baked the source mix into that order).
        rank, world_size: the data parallel shard the reader streams (the training split only;
            validation readers keep the full stream and are sharded by batch index in
            ``evaluate``). The LM reader has no sharding and refuses a world above one.
    """
    if os.path.isfile(os.path.join(data_dir, f"{split}.ev")):
        # a third reader, picked by the same rule: the evidence corpus carries a second token stream
        # per row, so the presence of the file IS the statement that this split has evidence
        return EvidenceDataset(
            data_dir=data_dir, tokenizer=tokenizer, batch_size=cfg.Batch_size,
            max_length=cfg.Seq_length, split=split,
            num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
            seed=cfg.seed, shuffle=shuffle,
            max_evidence_tokens=getattr(cfg, "max_evidence_tokens", 12288),
            loss_weight_floor_tokens=getattr(cfg, "loss_weight_floor_tokens", 64),
            rank=rank, world_size=world_size,
        )
    if os.path.isfile(os.path.join(data_dir, f"{split}.mask")):
        return SFTDataset(
            data_dir=data_dir, tokenizer=tokenizer, batch_size=cfg.Batch_size,
            max_length=cfg.Seq_length, split=split,
            num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
            seed=cfg.seed, shuffle=shuffle, rank=rank, world_size=world_size,
        )
    if world_size > 1:
        raise SystemExit(
            f"{split} has no loss mask, so it is read by the pretraining Dataset, which does not "
            f"shard across data parallel ranks; multi GPU runs need an SFT or evidence split"
        )
    return Dataset(
        data_dir=data_dir, tokenizer=tokenizer, batch_size=cfg.Batch_size,
        max_length=cfg.Seq_length, split=split,
        num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
    )


def _fresh_family(name: str) -> str:
    """collapse a fresh parameter's name to its family for a short log line -- e.g. two IR experts'
    ``z_keys`` collapse to one entry instead of printing once per expert index."""
    return re.sub(r"\.\d+\.", ".N.", name)


def build_sft_param_groups(model: TinyMoETransformer, weight_decay: float, fresh_lr: float = None,
                           is_fresh_param=None):
    """Split parameters into decayed / undecayed groups, **all** shadowed by fp32 masters.

    The decay split is the same one ``pretrain.build_param_groups`` makes and for the same reasons
    (``moe.loop_scale``, ``layer_scalar`` and the RMSNorm gains all have a degenerate zero, and
    every one of them is ndim <= 1).

    The difference is which tensors get an fp32 master. Pretraining shadows only the undecayed
    group, on the argument that ordinary 2D weights are safe because "their values and needed steps
    both scale with their own init std". **That argument does not survive SFT's learning rate.**
    Redo the arithmetic at lr=3e-5: a hidden_size=768 weight sits around its init std ~0.02-0.03,
    where bf16's ulp is ~0.4% of magnitude, i.e. ~1e-4. A steady-state AdamW step has magnitude
    ~lr = 3e-5. That is three times *below* the ulp, so ``param -= lr * update`` rounds to exactly
    the original bf16 value -- forever, no matter how much momentum accumulates. At pretraining's
    4e-4 the same step is ~4x *above* the ulp and lands fine, which is why the narrower fix was
    correct there and is not correct here.

    So: AdamW steps fp32 masters for everything, and the bf16 parameters the forward pass actually
    reads are refreshed from their masters after every real optimizer step, via the same
    ``sync_master_grads_``/``sync_master_values_`` pair pretraining already uses.

    Cost at 332M params: ~4.0GB of optimizer state (1.3GB masters + 2.7GB fp32 Adam moments, since
    ``torch.zeros_like(p)`` gives a master's moments fp32 where a bf16 parameter's would be bf16)
    against ~1.4GB for the pretraining arrangement -- ~2.6GB more, which a 32GB local card running
    a halved SFT batch has to spare. The masters are checkpointed under ``fp32_masters`` while
    ``SAVE_FP32_MASTERS`` is on, and a resume restores them (``restore_fp32_masters``); a
    checkpoint without them reseeds the masters from the bf16 weights, which discards up to half
    an ulp of sub-ulp progress per weight at every relaunch.

    Args:
        model: the (already bf16) model.
        weight_decay: applied to the ndim >= 2 groups only.
        fresh_lr: when given, whatever ``is_fresh_param`` matches goes into its own groups at this
            learning rate instead of sharing the run's. Ignored (nothing is fresh) when ``None``.
        is_fresh_param: which tensors ``fresh_lr`` applies to, **per profile, not a fixed union**.
            The IR profile rebuilt the whole ``ir_module`` subtree from scratch and needs all of it
            at the fresh rate (``is_rebuilt_ir_param(name) or is_fresh_loop_param(name)``). The
            evidence profile grafts its reader onto a table that already carries a full sharpening
            run, so only the port's own zero-init tensors qualify (``is_fresh_loop_param`` alone) --
            the table stays at the trunk's rate like every other converged tensor. Passing the wrong
            one silently retrains a converged table from scratch, or leaves a genuinely fresh tensor
            stuck at its init for the whole run.

    Returns:
        ``(param_groups, master_pairs)`` where ``master_pairs`` is ``[(bf16_param, fp32_master)]``
        covering every trainable parameter.
    """
    buckets = {("trunk", True): [], ("trunk", False): [], ("fresh", True): [], ("fresh", False): []}
    fresh_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_fresh = fresh_lr is not None and is_fresh_param is not None and is_fresh_param(name)
        if is_fresh:
            fresh_names.append(name)
        buckets[("fresh" if is_fresh else "trunk", param.ndim >= 2)].append(param)

    param_groups, master_pairs, summary = [], [], []
    for (origin, decayed), params in buckets.items():
        if not params:
            continue
        masters = [p.detach().clone().float().requires_grad_(True) for p in params]
        group = {"params": masters, "weight_decay": weight_decay if decayed else 0.0}
        if origin == "fresh":
            group["lr"] = fresh_lr
        param_groups.append(group)
        master_pairs.extend(zip(params, masters))
        summary.append(f"{len(params)} {origin}/{'decayed' if decayed else 'undecayed'}")
    logger.info(
        f"SFT optimizer param groups: {', '.join(summary)} (wd={weight_decay})"
        + (f", fresh lr={fresh_lr:.1e}" if fresh_lr is not None else "")
        + f"; all {len(master_pairs)} stepped via fp32 masters"
    )
    if fresh_names:
        # names, not just a count: a table's tensors and a reader's tensors can sum to the same
        # total either way, so the count line above cannot tell a correct split from the union bug
        # it replaces -- only naming what actually landed in the fresh group can
        logger.info(f"fresh group tensors: {sorted(set(_fresh_family(n) for n in fresh_names))}")
    return param_groups, master_pairs


def estimate_packed_rows(idx_path: str, max_length: int, num_mtp_tokens: int,
                         split_documents: bool = False, evidx_path: str = None,
                         max_evidence_tokens: int = 0, order=None, num_workers: int = 1) -> int:
    """How many packed rows the corpus yields, by replaying the packing rule over the index.

    The LR schedule needs a total step count up front, and "corpus tokens / (batch * seq)" is a bad
    estimate for the SFT reader: ``SFTDataset`` never splits a conversation across rows, so every
    row carries some trailing padding, and each conversation also costs ``num_mtp_tokens``
    separator slots. On a corpus of short conversations that gap is easily 10%, which would end the
    cosine well before the data does and leave the tail of training at the LR floor.

    The replay is over the on-disk order rather than the epoch's permutation -- the row count barely
    moves between orderings (it depends on the length *distribution*, not the sequence), and doing
    it exactly per epoch would mean materializing every permutation before training starts.

    Args:
        idx_path: ``{split}.idx``, uint64 document-end offsets with a leading 0.
        max_length: row length. The evidence reader (``evidx_path`` given) packs to ``max_length - 1``
            and skips a conversation with ``len + num_mtp_tokens`` over that; the SFT reader packs
            to ``max_length`` and skips one over it. Both rules are replayed here.
        num_mtp_tokens: separator slots appended after each conversation.
        split_documents: True for the pretraining ``Dataset``, which DOES split a document across
            rows and therefore drops nothing and wastes only the separator slots. That reader also
            keeps documents longer than ``max_length`` (it splits them), so the length filter below
            would throw away most of a web corpus rather than a handful of over-long conversations.
        evidx_path: ``{split}.evidx`` for the evidence reader, whose rows close on **either** budget.
            Replaying only the token budget undercounts them, because a row that filled its evidence
            first is shorter than the packing rule alone predicts -- which is the same failure this
            function exists to avoid, one budget further in.
        max_evidence_tokens: the evidence cap those rows close against. Ignored without ``evidx_path``.
        order: the epoch's document permutation, when the reader shuffles. With ONE budget the row
            count really does depend only on the length distribution, which is why the note above
            says the on-disk order is good enough. With two it does not: the corpus is written
            source by source under a weighted round robin, so on-disk order clusters documents whose
            prompt and evidence lengths are correlated, and replaying that order closes rows on
            evidence far more often than the shuffled stream does.
        num_workers: how many workers share the stream. Each keeps its OWN partial row and flushes it
            at the end of the epoch, so the count is per worker and the tail rows are real.

    Returns:
        Estimated number of packed rows for one epoch.
    """
    offsets = np.fromfile(idx_path, dtype=np.uint64)
    doc_lengths = np.diff(offsets).astype(np.int64)
    del offsets
    if split_documents:
        return max(1, int((doc_lengths + num_mtp_tokens).sum() // max_length) + 1)
    # int32 from here: a document is far under 2**31 tokens, and at hundreds of millions of
    # documents every int64 copy of the index is several GB (the counts are the same)
    doc_lengths = doc_lengths.astype(np.int32)
    lengths = doc_lengths + np.int32(num_mtp_tokens)

    # the evidence reader packs to one position under the row, so a row ends with a pad segment;
    # the SFT reader packs to the full row. lengths assume the BOS the builders always write
    row_limit = max_length - 1 if evidx_path else max_length

    ev_lengths = np.zeros_like(lengths)
    if evidx_path and max_evidence_tokens:
        ev_offsets = np.fromfile(evidx_path, dtype=np.uint64)
        ev_lengths = np.diff(ev_offsets).astype(np.int32)
        del ev_offsets

    # the reader's own stream order, when it has one. Only the two budget case actually needs this
    # (see the Args note), but running it for every profile keeps one code path
    if order is not None:
        lengths, ev_lengths = lengths[order], ev_lengths[order]

    keep = lengths <= row_limit
    # both readers skip empty documents
    keep &= (doc_lengths[order] if order is not None else doc_lengths) > 0
    del doc_lengths
    if max_evidence_tokens:
        keep &= ev_lengths <= max_evidence_tokens

    rows = 0
    for shard in range(max(1, num_workers)):
        # each worker packs its own stream and flushes its partial row at the end of the epoch, so
        # the tail rows are real rows and the count is per worker rather than over the whole corpus
        shard_keep = keep[shard::num_workers]
        shard_len = lengths[shard::num_workers][shard_keep].tolist()
        shard_ev = ev_lengths[shard::num_workers][shard_keep].tolist()
        used, used_ev, open_row = 0, 0, False
        for length, ev in zip(shard_len, shard_ev):
            if open_row and (used + length > row_limit
                             or (max_evidence_tokens and used_ev + ev > max_evidence_tokens)):
                rows += 1
                used, used_ev = 0, 0
            used, used_ev, open_row = used + length, used_ev + ev, True
        rows += int(open_row)
    return max(1, rows)


ROWS_CACHE_SUFFIX = ".rows_estimate.json"
ROWS_CACHE_VERSION = 2
EXIT_NO_ROWS_CACHE = 2
FINGERPRINT_BYTES = 1 << 20


def _file_stamp(path):
    """Size and a sha1 of the first and last megabyte of ``path`` for a cache key, None for no file.

    No mtime: a copy of the corpus to another disk or a restage does not preserve it, and the
    cache must stay valid for identical bytes.
    """
    if not path or not os.path.isfile(path):
        return None
    size = os.path.getsize(path)
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        digest.update(f.read(FINGERPRINT_BYTES))
        if size > FINGERPRINT_BYTES:
            f.seek(max(FINGERPRINT_BYTES, size - FINGERPRINT_BYTES))
            digest.update(f.read(FINGERPRINT_BYTES))
    return [size, digest.hexdigest()]


def rows_cache_key(dataset, max_length: int, num_mtp_tokens: int) -> dict:
    """Everything ``estimate_packed_rows``'s result depends on except the number of workers.

    Args:
        dataset: the plan split's dataset (``idx_path``, optional ``evidx_path``, ``seed``,
            ``shuffle``, ``epoch0_head_docs``, ``max_evidence_tokens``).
        max_length: row length.
        num_mtp_tokens: separator slots per document.

    Returns:
        A JSON-comparable dict: the index files' sizes and content fingerprints (first and last MB), the packing parameters, the
        reader rule and the epoch 0 order's seed and head.
    """
    return {
        "version": ROWS_CACHE_VERSION,
        "num_docs": getattr(dataset, "num_docs", None),
        "idx": _file_stamp(dataset.idx_path),
        "evidx": _file_stamp(getattr(dataset, "evidx_path", None)),
        "max_length": int(max_length),
        "num_mtp_tokens": int(num_mtp_tokens),
        "max_evidence_tokens": int(getattr(dataset, "max_evidence_tokens", 0) or 0),
        "split_documents": isinstance(dataset, Dataset),
        "shuffle": bool(getattr(dataset, "shuffle", False)),
        "seed": getattr(dataset, "seed", None),
        "epoch0_head_docs": getattr(dataset, "epoch0_head_docs", None),
    }


def rows_cache_path(data_dir: str, split: str) -> str:
    return os.path.join(data_dir, f"{split}{ROWS_CACHE_SUFFIX}")


def read_rows_cache(path: str, key: dict, num_workers: int):
    """The cached row estimate for ``num_workers`` streams, or None when absent, unreadable, for
    other inputs (``key`` differs) or for another worker count."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("key") != key:
            return None
        rows = saved["rows"].get(str(num_workers))
        return int(rows) if rows is not None else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write_rows_cache(path: str, key: dict, num_workers: int, rows: int) -> None:
    """Record an estimate atomically, keeping the other worker counts cached under the same key."""
    entries = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("key") == key:
            entries = dict(saved.get("rows") or {})
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    entries[str(num_workers)] = int(rows)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"key": key, "rows": entries}, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def resolve_rows_estimate(cache_path: str, key: dict, world_size: int, compute,
                          write_on_compute: bool = True):
    """The plan's row estimate: from the cache, computed (one process only), or refused.

    Args:
        cache_path: the plan split's ``rows_estimate.json``.
        key: ``rows_cache_key``'s dict.
        world_size: ranks of this launch; the estimate is per ``NUM_DATA_WORKERS * world_size``.
        compute: ``workers -> rows``, called only for a single process launch with no valid cache.
        write_on_compute: write the cache after computing (rank 0).

    Returns:
        ``(rows, "cache" | "computed")``, or ``(None, "refused")`` for several ranks with no valid
        cache: computing there would hold the other ranks at a collective for minutes.
    """
    workers = NUM_DATA_WORKERS * world_size
    rows = read_rows_cache(cache_path, key, workers)
    if rows is not None:
        return rows, "cache"
    if world_size > 1:
        return None, "refused"
    rows = compute(workers)
    if write_on_compute:
        try:
            write_rows_cache(cache_path, key, workers, rows)
        except OSError as e:
            logger.warning(f"could not write the row estimate cache {cache_path}: {e!r}")
    return rows, "computed"


def compute_rows_estimate(dataset, max_length: int, num_mtp_tokens: int, num_workers: int) -> int:
    """``estimate_packed_rows`` over a dataset's own files and epoch 0 order."""
    return estimate_packed_rows(
        dataset.idx_path, max_length, num_mtp_tokens,
        split_documents=isinstance(dataset, Dataset),
        evidx_path=getattr(dataset, "evidx_path", None),
        max_evidence_tokens=getattr(dataset, "max_evidence_tokens", 0),
        order=(dataset.document_order(dataset.num_docs)
               if getattr(dataset, "shuffle", False) else None),
        num_workers=num_workers,
    )


@torch.no_grad()
def apply_ir_refresh(model, optimizer, master_pairs, stats):
    """Re-cluster the IR tables and repair the optimizer state the recycling invalidated.

    Runs under ``no_grad`` because the fp32 masters are leaf tensors that require grad -- AdamW
    steps them -- and an in-place write to one of those raises rather than quietly detaching.

    Recycling rewrites individual rows of ``z_keys`` and ``y_values`` *outside* the optimizer. Two
    things then have to be fixed or the recycle silently does nothing:

    - **The fp32 masters.** Every parameter here is stepped through a master and refreshed from it
      after each step, so a master still holding the dead key would overwrite the new one on the
      very next optimizer step.
    - **AdamW's moments for those rows.** They describe a parameter that no longer exists; leaving
      them means a recycled entry starts with the momentum of the entry it replaced, in a direction
      that has nothing to do with its new position.

    Both are per-row, not per-tensor: the surviving 98% of the table keeps its moments, which is the
    whole reason the recycle is cheap.

    Args:
        model: the unwrapped ``TinyMoETransformer``.
        optimizer: the AdamW whose state indexes the fp32 masters.
        master_pairs: ``[(bf16_param, fp32_master)]`` from ``build_sft_param_groups``.
        stats: what ``moe.refresh_ir_clusters`` returned, one dict per IR table.

    Returns:
        The same stats, with the (device-side) id tensors dropped so they are loggable.
    """
    masters = {id(p): m for p, m in master_pairs}
    clean = []
    for module, table_stats in zip(model.moe.ir_modules, stats):
        ids = table_stats.pop("recycled_ids", None)
        if ids is not None:
            for param in (module.z_keys, module.y_values):
                master = masters.get(id(param))
                if master is None:
                    continue
                master.index_copy_(0, ids, param.detach().index_select(0, ids).float())
                state = optimizer.state.get(master)
                if state:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key].index_fill_(0, ids, 0.0)
        recall = table_stats.get("recall")
        if recall is not None and recall < IR_MIN_CANDIDATE_RECALL:
            # said loudly because the symptom otherwise looks like the anneal failing: if the
            # centroid stage misses the entries the read wanted, sharpening the temperature just
            # concentrates mass on the wrong candidates, and more training cannot fix it
            logger.warning(
                f"IR candidate recall@{module.read_top_k} = {recall:.3f}, below "
                f"{IR_MIN_CANDIDATE_RECALL} -- raise model.ir_probe_clusters (currently "
                f"{module.probe_clusters} of {module.num_clusters}) rather than reading the "
                f"retrieval entropy as a training result"
            )
        clean.append(table_stats)
    return clean


def build_sft_scheduler(optimizer: optim.Optimizer, total_steps: int, config=SFTConfig):
    """Linear warmup -> cosine decay to ``lr * lr_min_factor``, anchored to this run's own steps.

    Not shared with ``pretrain.build_scheduler``: that one is anchored to
    ``TrainingConfig.total_steps`` (the combined pretraining budget) because phase 2 has to
    continue phase 1's decay. SFT is a fresh schedule over a fresh optimizer.

    One MULTIPLICATIVE shape applied to every param group, rather than a warmup plus a
    ``CosineAnnealingLR``. The two are the same curve for a single group, but the IR profile runs
    two base rates (a from-scratch table at 3e-4, a 16B-token trunk at 1e-5) and
    ``CosineAnnealingLR``'s ``eta_min`` is one absolute floor shared by all groups -- so the fresh
    group would decay by 600x while the trunk decayed by 20x. A factor decays both by the same
    ratio, which is what "same schedule, different rates" has to mean.

    Args:
        config: ``SFTConfig``, ``RepairConfig`` or ``IRConfig`` -- the profile whose warmup and
            floor this follows.
    """
    warmup_steps = max(1, min(int(total_steps * config.warmup_fraction), total_steps - 1))
    decay_steps = max(total_steps - warmup_steps, 1)
    floor = config.lr_min_factor

    def shape(step: int) -> float:
        if step < warmup_steps:
            return 0.01 + (1.0 - 0.01) * step / warmup_steps
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=shape)


def pull_from_hub(repo_id: str, filename: str, dest_dir: str, token: str = None) -> str:
    """Download one file from the training mirror repo into ``dest_dir``.

    The pretraining run pushes checkpoints, graphs and ``manifest.json`` to
    ``TrainingConfig.hf_upload_repo`` precisely so a reclaimed instance doesn't take them with it
    (see ``modules/runtime/hf_sync.py``); this is the other end of that.
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"downloading {repo_id}/{filename} -> {dest_dir}")
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir, token=token)


def split_changes(saved: dict, current: dict, base_dir: str = BASE_DIR) -> list:
    """The data fields a resume would change, as ``(name, saved value, current value)`` triples.

    A field the checkpoint does not record (every checkpoint written before the fields existed)
    is never a change. ``data_dir`` is compared as a path resolved against ``base_dir``, so a
    relative and an absolute spelling of one directory agree.

    Args:
        saved: the checkpoint's ``sft`` fields (``data_dir``, ``train_split``, ``val_split``).
        current: this launch's effective values, after any command line override.
        base_dir: the root a relative ``data_dir`` resolves against.

    Returns:
        The differing fields, empty when the launch matches the checkpoint.
    """
    changes = []
    for key in DATA_KEYS:
        old = saved.get(key)
        if old is None:
            continue
        new = current.get(key)
        if key == "data_dir":
            same = (new is not None and os.path.realpath(os.path.join(base_dir, old))
                    == os.path.realpath(os.path.join(base_dir, new)))
        else:
            same = old == new
        if not same:
            changes.append((key, old, new))
    return changes


def check_split_switch(saved: dict, current: dict, allow_switch: bool,
                       base_dir: str = BASE_DIR) -> list:
    """Refuse a resume on other data than the checkpoint's unless the switch is asked for.

    The pilot launchers default to the warm-up split, so a relaunch after the switch that forgets
    the split would otherwise resume silently on the warm-up data.

    Args:
        saved: the checkpoint's ``sft`` fields.
        current: this launch's effective ``data_dir``, ``train_split`` and ``val_split``.
        allow_switch: ``--switch-split`` was passed.
        base_dir: the root a relative ``data_dir`` resolves against.

    Returns:
        The changes (see ``split_changes``), when there are none or the switch is allowed.

    Raises:
        SystemExit: the data differ and the switch was not asked for.
    """
    changes = split_changes(saved, current, base_dir)
    # a corpus moved to another directory is not other data: the document count check in the
    # caller guards it
    data_changes = [c for c in changes if c[0] != "data_dir"]
    if data_changes and not allow_switch:
        listed = "; ".join(f"{key}: checkpoint {old!r}, this launch {new!r}"
                           for key, old, new in data_changes)
        raise SystemExit(
            f"this launch resumes on other data than the checkpoint was trained on ({listed}). "
            f"Relaunch with the checkpoint's values, or pass --switch-split to change the data "
            f"on purpose."
        )
    return changes


def check_head_docs(saved_head, current_head, same_data: bool) -> None:
    """Refuse a resume on the same split under another head-first epoch 0 order.

    The resume position indexes into epoch 0's permutation, and the head size ``H`` of the split's
    ``order.json`` sidecar (None without one) is part of it. A switch to other data reads the new
    split's own ``H`` and is the normal case.

    Args:
        saved_head: ``epoch0_head_docs`` the checkpoint recorded (None when none or not recorded).
        current_head: the value this launch's training dataset read.
        same_data: the data fields match the checkpoint's.

    Raises:
        SystemExit: same data, different ``H``.
    """
    if same_data and saved_head != current_head:
        raise SystemExit(
            f"checkpoint was trained with epoch0_head_docs={saved_head!r} but this launch's "
            f"split reads {current_head!r}. The resume position indexes into epoch 0's order, "
            f"which that value sets -- restore the split's order.json or start a fresh run directory."
        )


def check_switch_state(saved_epoch, saved_offset, saved_num_docs, new_num_docs, new_head):
    """Whether a ``--switch-split`` resume lands on a position the new split can continue from.

    The warm-up split is the first H documents of the main split, visited in epoch 0. A warm run
    that crossed its own epoch boundary saved epoch 1 (the main launch would then skip main's
    epoch 0), and a warm split that is not the head of the new split (a stale or missing
    ``order.json``) would continue at a position that means something else.

    Args:
        saved_epoch: the checkpoint's epoch.
        saved_offset: the checkpoint's resume position (``global_offset``).
        saved_num_docs: documents of the split the checkpoint trained on, None when not recorded.
        new_num_docs: documents of the new training split.
        new_head: the new split's ``epoch0_head_docs`` (None without a sidecar).

    Returns:
        None when the switch is sound, else the message to log before exiting 30.
    """
    if saved_epoch != 0:
        return (f"--switch-split refused: the checkpoint is at epoch {saved_epoch}, not 0. The warm "
                f"split crossed its own epoch boundary, and resuming the new split at that epoch "
                f"would skip its epoch 0.")
    limit = new_head if new_head is not None else new_num_docs
    if saved_offset > limit:
        return (f"--switch-split refused: the checkpoint's position {saved_offset:,} is past the new "
                f"split's head of {limit:,} documents.")
    if saved_num_docs is None:
        return (f"--switch-split refused: the checkpoint records no document count for its split, so "
                f"it cannot be checked against the new split (head {new_head!r}, "
                f"{new_num_docs:,} documents).")
    if (new_head is not None and new_head == saved_num_docs) or new_num_docs == saved_num_docs:
        return None
    return (f"--switch-split refused: the checkpoint's split has {saved_num_docs:,} documents but the "
            f"new split's order.json head is {'absent' if new_head is None else f'{new_head:,}'} and it "
            f"has {new_num_docs:,} documents; the "
            f"warm split is not a head of the new one (missing or stale sidecar).")


def final_checkpoint_data(path: str) -> dict:
    """The data fields a finished run's final checkpoint recorded, read without loading tensors.

    Args:
        path: the ``*_final.pt`` path.

    Returns:
        ``data_dir``, ``train_split`` and ``val_split`` as saved (empty when not recorded).
    """
    saved = (torch.load(path, map_location="cpu", mmap=True).get("sft") or {})
    return {key: saved[key] for key in DATA_KEYS if saved.get(key) is not None}


def select_epoch(dataloader, dataset, epoch: int) -> None:
    """Point the next pass over ``dataloader`` at ``epoch``'s document permutation.

    accelerate's prepared loader calls ``set_epoch(self.iteration)`` on its dataset at the start
    of every pass, and ``iteration`` counts the passes of this process from zero, so setting the
    epoch on the dataset alone is overridden on a resume past epoch 0: the rest of that epoch
    read epoch 0's order again and the pass after it epoch 1's. Setting it through the loader
    keeps its counter and the dataset in step. A fresh run is unchanged (the counter and the
    epoch agree on every pass), and a loader without ``set_epoch`` falls back to the dataset's.

    Args:
        dataloader: the loader the epoch loop iterates, prepared by accelerate or plain.
        dataset: the dataset behind it; the LM reader has no ``set_epoch`` and reads in on-disk
            order.
        epoch: the epoch whose permutation the next pass uses.
    """
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(epoch)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def stop_needs_save(optimizer_steps: int, steps_at_last_save, slack_steps: int) -> bool:
    """Whether a stop request should write a checkpoint before exiting.

    ``False`` when the last save is within ``slack_steps`` optimizer steps: the stop is polled at
    the log cadence only, so a STOP dropped right after a save (``stop_after_save.sh``) is seen
    up to one log interval later, and a second file a few steps past the first is a near
    duplicate. The save on disk is then the resume point and the steps since it are retrained.

    Args:
        optimizer_steps: optimizer steps taken so far (the scheduler's ``last_epoch``).
        steps_at_last_save: the same count at the last save or at the resumed checkpoint, or
            ``None`` when this process has neither (a fresh run always saves).
        slack_steps: optimizer steps in one log interval.

    Returns:
        True when the stop must save.
    """
    if steps_at_last_save is None:
        return True
    return optimizer_steps - steps_at_last_save > slack_steps


def plan_total_steps(rows_per_epoch: int, num_epochs: int, batch_size: int,
                     grad_accumulation_steps: int, world_size: int = 1) -> int:
    """Optimizer steps the LR schedule is fitted to.

    ``rows_per_epoch`` counts the rows of every rank together (the estimator replays all the
    interleaved streams), and each rank trains ``batch_size`` rows per micro step, so a step
    consumes ``batch_size * grad_accumulation_steps * world_size`` rows and the cosine spans the
    same tokens whatever the number of ranks. One rank reproduces the old expression exactly.

    Args:
        rows_per_epoch: ``estimate_packed_rows``'s estimate over all ranks.
        num_epochs: epochs of the profile.
        batch_size: rows per micro step PER GPU.
        grad_accumulation_steps: micro steps per optimizer step.
        world_size: number of data parallel ranks.

    Returns:
        The per rank optimizer step count, at least 1.
    """
    micro_steps = rows_per_epoch * num_epochs / (batch_size * world_size)
    return max(1, int(micro_steps / grad_accumulation_steps))


def select_retention_deletions(paths, keep_recent: int, keep_every_tokens: int) -> list:
    """Rolling checkpoints to delete under the long-term keep rule.

    Keeps the newest ``keep_recent`` rolling saves, every ``*_final.pt``, and the long-term keeps:
    a save is one when it is the first at or past a multiple of ``keep_every_tokens`` (names carry
    the token count truncated to millions and a save lands a log step past its grid point, so "a
    multiple" is read as "first save at or past one"). Everything else rolling is returned.

    Args:
        paths: checkpoint paths of one run directory.
        keep_recent: ``keep_local_checkpoints``.
        keep_every_tokens: spacing of the long-term keeps, 0 for none (nothing is selected, the
            old behaviour: only upload-confirmed pruning applies).

    Returns:
        The paths to delete, oldest first.
    """
    if keep_every_tokens <= 0:
        return []
    rolling = []
    for path in paths:
        name = os.path.basename(path)
        match = re.search(r"_tok(\d+)M_", name)
        if ckpt_lib.is_final(name) or match is None:
            continue
        rolling.append((int(match.group(1)) * 1_000_000, path))
    rolling.sort()
    older = rolling[:max(len(rolling) - keep_recent, 0)]
    doomed, previous = [], 0
    for tokens, path in rolling:
        milestone = tokens // keep_every_tokens > previous // keep_every_tokens
        previous = tokens
        if (tokens, path) in older and not milestone:
            doomed.append(path)
    return doomed


def plan_changed(saved: dict, current: dict) -> list:
    """The plan fields a resume would change, as ``(name, saved, current)`` triples.

    ``total_steps`` is rebuilt from the corpus estimate on every launch and the new value is the
    one the schedule then follows, so a split switch, an estimator change or another number of
    ranks moves the cosine under a run that already began it. A field the checkpoint does not
    record is never a change.

    Args:
        saved: ``total_steps``, ``rows_per_epoch``, ``world_size`` and
            ``grad_accumulation_steps`` from the checkpoint.
        current: the same fields for this launch.

    Returns:
        The differing fields, empty when the plan is the checkpoint's.
    """
    return [(key, saved[key], current.get(key)) for key in ("total_steps", "rows_per_epoch",
                                                              "world_size",
                                                              "grad_accumulation_steps",
                                                              "plan_split")
            if saved.get(key) is not None and saved[key] != current.get(key)]


def check_run_state(run_state, phase: str, resumed_tokens, slack_tokens: int):
    """Whether a launch is consistent with the ``run_state.json`` the last process left.

    A run directory that lost its checkpoints (or a ``-c`` relaunch into a wiped directory that
    kept ``run_state.json``) would otherwise train from the seed again behind a plausible loss
    curve. A state recorded under another phase, or with no tokens, compares to nothing, and a
    directory with no ``run_state.json`` is a fresh run.

    Args:
        run_state: ``ckpt_lib.read_run_state``'s output, or None.
        phase: this launch's phase label.
        resumed_tokens: the token count of the checkpoint that loaded, or None when none did.
        slack_tokens: how far behind the recorded count a resume may be, ``2 *
            checkpoint_every_tokens`` (the fall back to an older checkpoint).

    Returns:
        None when the launch is consistent, else the message to log before exiting 30.
    """
    if run_state is None or run_state.get("phase") != phase:
        return None
    recorded = int(run_state.get("token_count", 0))
    if recorded <= 0:
        return None
    if resumed_tokens is None:
        return (f"resume verification failed for {phase}: run_state.json records {recorded:,} "
                f"tokens (checkpoint {run_state.get('checkpoint')}) but no checkpoint loads from "
                f"this directory, so this launch would start again from the seed. Restore the "
                f"checkpoints, or remove run_state.json to start fresh on purpose.")
    if resumed_tokens >= recorded - slack_tokens:
        return None
    return (f"resume verification failed for {phase}: run_state.json records {recorded:,} tokens "
            f"but the newest checkpoint that loads is at {resumed_tokens:,} (slack "
            f"{slack_tokens:,}). Refusing to retrain ground already covered; inspect the run "
            f"directory before restarting.")


# ---------------------------------------------------------------------------------------------
# data parallel helpers. Everything here is pure or takes its collective as an argument, so the
# logic is tested with fake ranks; sft() passes torch.distributed's.
# ---------------------------------------------------------------------------------------------

class RankLog:
    """``utils.logger`` for a multi process run: info and warnings from rank 0 only, errors from
    every rank with the rank in the message, so eight processes do not print every line eight
    times and a failure on rank 3 is still seen."""

    def __init__(self, base, rank: int = 0):
        self.base, self.rank = base, rank

    def info(self, msg, *args, **kwargs):
        if self.rank == 0:
            self.base.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        # every rank reaches the same warnings (plan changed, switch notes): rank 0 prints them once
        if self.rank == 0:
            self.base.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.base.error(msg if self.rank == 0 else f"[rank {self.rank}] {msg}", *args, **kwargs)


def dist_env() -> tuple:
    """``(rank, world_size)`` from the launcher's environment (accelerate and torchrun both set
    ``RANK`` and ``WORLD_SIZE``); ``(0, 1)`` for a plain ``python scripts/sft.py``."""
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def accelerator_kwargs(world_size: int, grad_accumulation_steps: int) -> dict:
    """The ``Accelerator`` arguments for a run of ``world_size`` processes.

    One process keeps the arguments every earlier run used. More processes get DDP with unused
    parameters allowed and no buffer broadcast (kept for any DDP wrap; the trainer reduces the
    gradients itself, see ``pretrain.all_reduce_grads``), each rank feeding its
    own loader (``split_batches`` and ``dispatch_batches`` off), and a 30 minute collective
    timeout, since rank 0's save (a multi GB write) keeps the others at a barrier.

    Args:
        world_size: the number of processes.
        grad_accumulation_steps: micro steps per optimizer step.

    Returns:
        Keyword arguments for ``Accelerator``.
    """
    if world_size <= 1:
        return dict(device_placement=True, split_batches=True,
                    gradient_accumulation_steps=grad_accumulation_steps)
    return dict(
        device_placement=True,
        gradient_accumulation_steps=grad_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(split_batches=False, dispatch_batches=False),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False),
            InitProcessGroupKwargs(timeout=timedelta(minutes=30)),
        ],
    )


def lockstep_batches(batches, chunk: int, reduce_min):
    """Yield from ``batches`` so that every rank yields exactly the same number of batches.

    The ranks' shards pack to different row counts, and an all-reduce with one rank out of
    batches is a hang. Every ``chunk`` batches each rank reports how many it has buffered, the
    MINIMUM over the ranks is the number all of them take, and the first chunk in which any rank
    has none ends the epoch on every rank at once; the buffered batches beyond it are dropped
    (a few rows of an epoch, like ``drop_last``). A batch counts as consumed when it is yielded, so
    a stop between chunks leaves the unconsumed ones to be read again after a resume.

    One host sync per chunk (the reduce), the same cadence as the log block's.

    Args:
        batches: this rank's batch iterable.
        chunk: batches buffered per reduce, ``LOG_INTERVAL``.
        reduce_min: ``int -> int`` MIN of the argument over the ranks.

    Yields:
        This rank's batches, as many as every rank has.
    """
    iterator = iter(batches)
    buffered = deque()
    while True:
        while len(buffered) < chunk:
            try:
                buffered.append(next(iterator))
            except StopIteration:
                break
        take = int(reduce_min(len(buffered)))
        if take <= 0:
            return
        for _ in range(take):
            yield buffered.popleft()


# one float64 vector, summed over the ranks, per log step: counts first, then (value, valid) pairs
# for each scalar, then the per loop CE
COUNT_KEYS = ("tokens", "evidence", "stop_user", "stop_preempt", "checkpoint")
LOG_SCALAR_KEYS = ("loss", "ce", "aux", "p_max", "top1", "selection", "contrastive", "grounded")
OPTIONAL_LOG_KEYS = ("selection", "contrastive", "grounded")


def pack_log_vector(counts: dict, scalars: dict, per_loop, device) -> torch.Tensor:
    """The log step's all-reduce payload.

    Args:
        counts: ``COUNT_KEYS`` -> number (this rank's token and evidence deltas since the last
            log, and 0 or 1 for a stop or checkpoint request); missing keys are 0.
        scalars: ``LOG_SCALAR_KEYS`` -> 0-dim tensor, or None / non finite when the rank has no
            value this step (a slice with no evidence has no selection loss); such a rank adds
            nothing to the mean.
        per_loop: the per loop CE tensors.
        device: where the vector is built.

    Returns:
        ``[len(COUNT_KEYS) + 2 * len(LOG_SCALAR_KEYS) + len(per_loop)]`` float64.
    """
    head = torch.tensor([float(counts.get(k, 0)) for k in COUNT_KEYS], dtype=torch.float64,
                        device=device)
    zero = torch.zeros((), dtype=torch.float64, device=device)
    cols = []
    for key in LOG_SCALAR_KEYS:
        value = scalars.get(key)
        if value is None:
            cols += [zero, zero]
            continue
        v = value.detach().to(device=device, dtype=torch.float64).reshape(())
        # only the terms a slice may lack are left out when not finite; a diverged loss, CE or aux
        # on any rank must propagate as NaN, not vanish from the mean
        ok = torch.isfinite(v) if key in OPTIONAL_LOG_KEYS else torch.ones_like(v)
        cols += [torch.where(ok.bool(), v, zero) if key in OPTIONAL_LOG_KEYS else v, ok.to(torch.float64)]
    loops = [ce.detach().to(device=device, dtype=torch.float64).reshape(()) for ce in per_loop]
    return torch.cat([head, torch.stack(cols), torch.stack(loops) if loops else zero.new_empty(0)])


def unpack_log_vector(values, world_size: int) -> dict:
    """Invert ``pack_log_vector`` on the summed vector (``.tolist()``): the summed counts, the mean
    of each scalar over the ranks that had one (NaN when none did), and the mean per loop CE."""
    counts = dict(zip(COUNT_KEYS, values[:len(COUNT_KEYS)]))
    scalars, at = {}, len(COUNT_KEYS)
    for key in LOG_SCALAR_KEYS:
        total, n = values[at], values[at + 1]
        scalars[key] = total / n if n > 0 else float("nan")
        at += 2
    return {"counts": counts, "scalars": scalars,
            "per_loop": [v / world_size for v in values[at:]]}


def stop_exit_code(counts: dict):
    """The exit code a reduced log vector asks for: a restartable stop (SIGTERM) wins over a user
    stop (STOP file), None when nobody asked. Identical on every rank, the vector being a sum."""
    if counts["stop_preempt"] > 0:
        return EXIT_PREEMPTED
    if counts["stop_user"] > 0:
        return EXIT_USER_STOP
    return None


def control_counts(control, include_user_stop: bool = True) -> dict:
    """This rank's stop and checkpoint requests as ``pack_log_vector`` counts.

    Args:
        control: the rank's ``RunControl``.
        include_user_stop: False on ranks that must not act on the STOP file (only rank 0 polls
            it; every rank's signal handlers still set the flags).
    """
    stopped = control.stop_requested
    return {
        "stop_preempt": int(stopped and control.exit_code == EXIT_PREEMPTED),
        "stop_user": int(stopped and control.exit_code != EXIT_PREEMPTED and include_user_stop),
        "checkpoint": int(control.take_checkpoint_request()),
    }


def tensors_checksum(tensors) -> torch.Tensor:
    """One float64 sum per tensor, to compare across ranks at startup."""
    sums = [t.detach().double().sum() for t in tensors]
    return torch.stack(sums) if sums else torch.zeros(0, dtype=torch.float64)


def check_ranks_identical(named, reduce_min, reduce_max) -> None:
    """Raise when a tensor's checksum differs between ranks.

    Run once after the model, the fp32 masters and DDP are in place: ranks that began from
    different weights (a seed that left a tensor at a random init, a checkpoint read short on one
    node) would train eight models that only average their gradients.

    Args:
        named: ``[(name, tensor)]``.
        reduce_min, reduce_max: callables returning the MIN / MAX of a tensor over the ranks.

    Raises:
        RuntimeError: naming the first few tensors that differ.
    """
    names = [n for n, _ in named]
    local = tensors_checksum([t for _, t in named])
    low, high = reduce_min(local.clone()), reduce_max(local.clone())
    differs = (low != high).cpu().tolist()
    bad = [name for name, flag in zip(names, differs) if flag]
    if bad:
        raise RuntimeError(
            f"{len(bad)} of {len(names)} tensors differ between the data parallel ranks at "
            f"startup (first: {bad[:5]}): the ranks did not start from the same weights"
        )


def reseed_cuda_for_rank(cuda_state: torch.Tensor, rank: int):
    """The CUDA RNG state a rank resumes from.

    Rank 0 takes the saved state exactly (a one process resume is bit for bit what it was). Another
    rank derives a seed from the same saved state and its own rank, so the ranks' dropout masks
    and loss subsamples stay different from each other and a function of the checkpoint.

    Args:
        cuda_state: the checkpoint's ``cuda_rng_state`` (rank 0's).
        rank: this process's data parallel rank.

    Returns:
        ``(state, seed)``: rank 0 gets ``(cuda_state, None)``; another rank ``(None, seed)`` to
        pass to ``torch.cuda.manual_seed``.
    """
    if rank == 0:
        return cuda_state, None
    digest = hashlib.sha256(bytes(cuda_state.cpu().tolist()) + rank.to_bytes(4, "little"))
    return None, int(digest.hexdigest()[:15], 16)


def ir_sync_tensors(moe, optimizer, master_pairs) -> list:
    """Every tensor an IR cluster refresh may have changed: the IR modules' parameters and buffers,
    the fp32 masters of those parameters and the optimizer's state for them.

    The refresh clusters and recycles from each rank's own query reservoir and usage counts, so
    after it the ranks' tables, centroids and moments differ; rank 0's are broadcast to the rest.
    """
    masters = {id(p): m for p, m in master_pairs}
    tensors = []
    for module in moe.ir_modules:
        for param in module.parameters():
            tensors.append(param.data)
            master = masters.get(id(param))
            if master is not None:
                tensors.append(master.data)
                state = optimizer.state.get(master)
                if state:
                    tensors.extend(v for v in state.values() if torch.is_tensor(v))
        tensors.extend(module.buffers())
    return tensors


def sync_ir_from_main(moe, optimizer, master_pairs, broadcast) -> int:
    """Broadcast rank 0's IR state to every rank after a cluster refresh; returns the tensor count.

    Args:
        broadcast: callable that overwrites a tensor with rank 0's copy in place
            (``lambda t: torch.distributed.broadcast(t, src=0)``).
    """
    tensors = ir_sync_tensors(moe, optimizer, master_pairs)
    for tensor in tensors:
        broadcast(tensor)
    return len(tensors)


def worker_offset(rows: list, stride: int, fallback: int) -> int:
    """The resume position from every stream's last consumed position: the minimum, plus the
    stride (the next position that stream wants), so no stream's unconsumed work is skipped.

    Args:
        rows: ``worker_state`` as a list, -1 for a stream that produced nothing yet. Under data
            parallel training these are the streams of every rank (the ranks' states are merged
            with a MAX first: a stream is only ever written by its own rank).
        stride: ``NUM_DATA_WORKERS * world_size``.
        fallback: the position the run started from, used when no stream has produced anything.
    """
    seen = [r for r in rows if r >= 0]
    return min(seen) + stride if seen else fallback


def _merge_numeric(a: dict, b: dict) -> dict:
    """Sum two nested dicts of numbers key by key; a key only one side has is kept."""
    out = dict(a)
    for key, value in b.items():
        if key not in out:
            out[key] = value
        elif isinstance(value, dict):
            out[key] = _merge_numeric(out[key], value)
        else:
            out[key] = out[key] + value
    return out


EVAL_SCALARS = ("ce_sum", "ce_weight_sum", "token_sum", "n_batches", "selection_sum",
                "selection_batches", "contrastive_sum", "contrastive_batches", "grounded_sum",
                "grounded_batches")
EVAL_DICTS = ("signal_sums", "cond_ce_sum", "cond_weight_sum", "cond_ce_by_loop_sum",
              "source_cond_sum", "source_cond_weight", "source_token_sum")
EVAL_LISTS = ("grounded_scores", "grounded_labels", "grounded_has_evidence")


def merge_eval_states(states: list) -> dict:
    """Combine the per rank accumulators of ``evaluate`` into the one a single process would hold.

    Sums add, the nested per condition, per loop and per source sums add key by key, and the
    per token score lists and the selector sink concatenate in rank order (the AUROCs and means
    they feed do not depend on order).

    Args:
        states: one state dict per rank, in rank order.

    Returns:
        The merged state, same keys.
    """
    out = {k: sum(s[k] for s in states) for k in EVAL_SCALARS}
    for key in EVAL_DICTS:
        merged = {}
        for state in states:
            merged = _merge_numeric(merged, state[key])
        out[key] = merged
    for key in EVAL_LISTS:
        out[key] = [item for state in states for item in state[key]]
    sink = {}
    for state in states:
        for loop_idx, entry in state["selector_sink"].items():
            into = sink.setdefault(loop_idx, {name: [] for name in entry})
            for name, items in entry.items():
                into[name].extend(items)
    out["selector_sink"] = sink
    return out


def all_gather_states(world_size: int):
    """``gather`` for ``evaluate``: ``torch.distributed.all_gather_object`` over the ranks."""
    def gather(state):
        out = [None] * world_size
        torch.distributed.all_gather_object(out, state)
        return out
    return gather


def fp32_master_state(named_masters) -> dict:
    """The fp32 masters as a CPU dict keyed like the model's state dict.

    Args:
        named_masters: ``[(name, bf16_param, fp32_master)]``.

    Returns:
        ``{name: fp32 CPU tensor}``, copies that a later step cannot change.
    """
    return {name: master.detach().to(device="cpu", dtype=torch.float32, copy=True)
            for name, _, master in named_masters}


def restore_fp32_masters(named_masters, saved) -> str:
    """Set the masters from a checkpoint's ``fp32_masters``, or reseed them from bf16.

    The saved values are copied into the existing master tensors in place, so the optimizer's
    param groups and its state (both keyed by those tensor objects) stay valid. The saved masters
    are used only when every master has an entry of its shape and each rounds to the bf16
    parameter just loaded (which is what the last step left behind); anything else reseeds all
    masters from the bf16 weights, which is what every resume did before the masters were saved.

    Args:
        named_masters: ``[(name, bf16_param, fp32_master)]``, the bf16 weights already loaded.
        saved: the checkpoint's ``fp32_masters`` dict, or None.

    Returns:
        A short description of the path taken, for the log.
    """
    reason = None
    if saved is None:
        reason = "the checkpoint carries none"
    else:
        for name, param, master in named_masters:
            value = saved.get(name)
            if value is None or tuple(value.shape) != tuple(master.shape):
                reason = f"{name} is missing or has another shape"
                break
            if not torch.equal(value.to(device=param.device, dtype=param.dtype), param.detach()):
                reason = f"{name} does not round to the loaded bf16 weight"
                break
    with torch.no_grad():
        if reason is None:
            for name, _, master in named_masters:
                master.copy_(saved[name].to(device=master.device, dtype=master.dtype))
            return f"restored {len(named_masters)} fp32 masters from the checkpoint"
        for _, param, master in named_masters:
            master.copy_(param.detach().float())
    return f"reseeded the fp32 masters from bf16 ({reason})"


def restore_rng_states(loop_rng: random.Random, sft_extra: dict, rank: int = 0) -> list:
    """Restore the depth sampling RNG and the CUDA RNG when the checkpoint recorded them.

    Args:
        loop_rng: the trainer's depth sampling RNG, seeded from the config.
        sft_extra: the checkpoint's ``sft`` fields.
        rank: this process's data parallel rank. The depth sampling RNG is restored identically on
            every rank (it is consumed in lockstep, so rank 0's saved state is every rank's); the
            CUDA RNG is rank 0's saved state, which another rank turns into its own seed
            (``reseed_cuda_for_rank``).

    Returns:
        The names of the states restored; empty for a checkpoint that recorded neither.
    """
    restored = []
    state = sft_extra.get("loop_rng_state")
    if state is not None:
        version, internal, gauss_next = state
        loop_rng.setstate((version, tuple(internal), gauss_next))
        restored.append("loop_rng")
    cuda_state = sft_extra.get("cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        state, seed = reseed_cuda_for_rank(cuda_state, rank)
        if state is not None:
            torch.cuda.set_rng_state(state.cpu())
        else:
            torch.cuda.manual_seed(seed)
        restored.append("cuda")
    return restored


def run_already_complete(checkpoint_dir: str, phase: str) -> bool:
    """True when the run in ``checkpoint_dir`` finished: its newest checkpoint is the final one
    and ``run_state.json`` names it.

    The final checkpoint is written only at the end of the last epoch, and ``run_state.json``
    records the basename of every save, so the pair marks completion. A relaunch would otherwise
    resume from the final checkpoint and overwrite it.

    Args:
        checkpoint_dir: the run directory.
        phase: the profile's phase label.

    Returns:
        Whether the run is complete.
    """
    if not os.path.isdir(checkpoint_dir):
        return False
    names = [n for n in os.listdir(checkpoint_dir)
             if n.startswith(ckpt_lib.CKPT_PREFIX) and n.endswith(".pt")]
    if not names:
        return False
    newest = max(names, key=lambda n: os.path.getmtime(os.path.join(checkpoint_dir, n)))
    if newest != ckpt_lib.final_name(phase):
        return False
    state = ckpt_lib.read_run_state(os.path.join(checkpoint_dir, "run_state.json"))
    return bool(state) and state.get("phase") == phase and state.get("checkpoint") == newest


def save_sft_checkpoint(model, optimizer, scheduler, path, *, epoch, step, token_count,
                        start_token_count, global_offset, losses, seed, phase=SFT_PHASE,
                        kill_checked=False, sft_fields=None, fp32_masters=None):
    """Write an SFT checkpoint atomically.

    Deliberately its own function rather than an extension of ``utils.save_checkpoint``: SFT needs
    two fields pretraining has no concept of (``start_token_count``, so SFT progress can be
    recovered from the continued global counter, and ``seed``, which selects the document
    permutation a ``global_offset`` indexes into), and the pretraining run is *live on rented
    hardware right now* -- changing ``utils.load_checkpoint``'s tuple arity would break the running
    job on its next preemption relaunch, since onstart.sh re-clones the branch.

    The payload is a strict **superset** of what ``utils.load_checkpoint`` expects, so
    ``scripts/inference.py`` and ``scripts/eval_calibration.py`` read an SFT checkpoint unchanged.

    Args:
        sft_fields: extra entries of the ``sft`` dict (the data the run trains on and the RNG
            states), read back with ``.get`` so older checkpoints load without them.
        fp32_masters: ``fp32_master_state``'s dict, stored as ``fp32_masters`` when given.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "dataset_idx": step,
        "token_count": token_count,
        "global_offset": global_offset,
        "phase": phase,
        "losses": losses,
        # SFT-only extras, ignored by utils.load_checkpoint's .get()-based reader
        # kill_checked: the one automatic kill decision was taken, so a resume must not take it again
        "sft": {"start_token_count": start_token_count, "seed": seed,
                "kill_checked": kill_checked, **(sft_fields or {})},
    }
    if fp32_masters is not None:
        payload["fp32_masters"] = fp32_masters
    # write-then-rename, same reasoning as utils.save_checkpoint: a crash mid-write must not leave
    # a truncated .pt that is also the newest file by mtime, i.e. the one a resume would pick
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    logger.info(f"Checkpoint saved at {path}")


def load_sft_checkpoint(model, optimizer, scheduler, path, expected_phase=SFT_PHASE,
                        checkpoint_dir=SFT_CHECKPOINT_DIR):
    """Restore a full SFT (or repair) run from its own checkpoint. Returns the resume state dict.

    Raises on anything that is not a checkpoint of ``expected_phase`` -- and checks that *before*
    touching the model, so a rejected file leaves no partial state behind. A pretraining checkpoint
    dropped into ``ckpts/sft`` by hand would otherwise load cleanly: its optimizer state has the
    same two param groups with the same shapes, so AdamW's moments from a 4e-4 run would be silently
    adopted as this fine-tune's, along with a scheduler anchored to the 29.9B-token cosine. The same
    argument covers SFT vs. repair, which differ by an order of magnitude in LR and by a whole
    objective.
    """
    checkpoint = torch.load(path, map_location="cpu")
    phase = checkpoint.get("phase")
    if phase != expected_phase:
        raise ValueError(
            f"{os.path.basename(path)} was written during phase={phase!r}, not {expected_phase!r}. "
            f"Pass it with -c to initialize FROM it instead of resuming it, and keep "
            f"{checkpoint_dir} for {expected_phase} checkpoints only."
        )
    check_resume_kind(optimizer, checkpoint["optimizer_state_dict"], os.path.basename(path))
    model.load_state_dict(with_balance_default(model, checkpoint["model_state_dict"]))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    sft_extra = checkpoint.get("sft", {}) or {}
    logger.info(f"Checkpoint loaded from {path}")
    return {
        # read with .get: every checkpoint before these fields existed resumes as before
        "data": {key: sft_extra[key] for key in DATA_KEYS if sft_extra.get(key) is not None},
        "rng": {key: sft_extra[key] for key in ("loop_rng_state", "cuda_rng_state")
                if sft_extra.get(key) is not None},
        "fp32_masters": checkpoint.get("fp32_masters"),
        "optimizer_steps": int(getattr(scheduler, "last_epoch", 0)),
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("dataset_idx", 0),
        "token_count": checkpoint.get("token_count", 0),
        "start_token_count": sft_extra.get("start_token_count", checkpoint.get("token_count", 0)),
        "global_offset": checkpoint.get("global_offset", 0),
        "losses": checkpoint.get("losses", None) or [],
        "seed": sft_extra.get("seed", SFTConfig.seed),
        "kill_checked": sft_extra.get("kill_checked", False),
        # the plan the run's cosine was fitted to, None for a checkpoint written before they were saved
        "total_steps": sft_extra.get("total_steps"),
        "rows_per_epoch": sft_extra.get("rows_per_epoch"),
        "world_size": sft_extra.get("world_size"),
        "grad_accumulation_steps": sft_extra.get("grad_accumulation_steps"),
        "epoch0_head_docs": sft_extra.get("epoch0_head_docs"),
        "num_docs": sft_extra.get("num_docs"),
        "plan_split": sft_extra.get("plan_split"),
    }


# the cross attention reader's own tensors, which a key/value reader model does not have
LEGACY_READER_PREFIXES = ("moe.shared_evidence.", "moe.evidence_query_bias.")


def load_pretrained_weights(model, path: str):
    """Seed SFT from a pretraining checkpoint: weights and bookkeeping, no optimizer state.

    The optimizer is deliberately *not* restored. Pretraining's AdamW moments were accumulated at
    lr=4e-4 against a different objective; carrying them into a 3e-5 fine-tune would spend the
    first few hundred steps unwinding momentum that no longer describes the loss surface.

    Returns:
        The pretraining token count, carried forward on purpose -- see this module's docstring.
    """
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    if getattr(model.moe, "reader_kv", False):
        # a key/value reader seeded from a cross reader seed: the old reader's tensors have no
        # place in this model. Named and logged rather than a non strict load
        dropped = [k for k in state if k.startswith(LEGACY_READER_PREFIXES)]
        if dropped:
            state = {k: v for k, v in state.items() if k not in set(dropped)}
            logger.info(f"key/value reader: dropped {len(dropped)} cross reader tensors from the seed")
    # through load_model_state, not load_state_dict: a seed built before the retrieval temperature
    # became learned carries neither temperature tensor, and their inits ARE the 1.0 it was hardcoded
    # to. Still strict about everything else -- a trunk tensor left random here would train anyway
    load_model_state(model, state)
    token_count = checkpoint.get("token_count", 0)
    logger.info(
        f"Initialized from pretrained checkpoint {os.path.basename(path)} "
        f"({token_count / 1e9:.3f}B pretraining tokens, phase={checkpoint.get('phase')})"
    )
    return token_count


def source_name(index: int) -> str:
    """``SOURCE_KEYS`` name of a ``.src`` byte, ``"unknown"`` for 255 or anything past the table."""
    return SOURCE_KEYS[index] if 0 <= index < len(SOURCE_KEYS) else "unknown"


def _within_buffer_auroc(share: torch.Tensor, gold: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """``[P]`` AUROC of gold against distractor chunks inside each token's own visible buffer.

    Pairwise over (gold, distractor) pairs of one row: 1 when the gold chunk's share is higher, 0.5
    on a tie, 0 otherwise. A uniform selector reads exactly 0.5 here. Pooling the shares of every
    token into one ranking instead does not: it compares shares across buffers of different sizes,
    and a buffer of more chunks gives each a smaller share, so the pooled reading of a uniform
    selector on the fixed split was 0.421, not 0.5.

    Args:
        share: ``[P, M]`` each chunk's share of the token's external mass.
        gold: ``[P, M]`` bool, visible gold chunks.
        vis: ``[P, M]`` bool, the token's visible chunks. Rows need at least one of each kind.
    """
    distractor = vis & ~gold
    pairs = gold.unsqueeze(-1) & distractor.unsqueeze(-2)                # [P, M gold, M distractor]
    diff = share.unsqueeze(-1) - share.unsqueeze(-2)
    score = (diff > 0).float() + 0.5 * (diff == 0).float()
    return (score * pairs).sum(dim=(-2, -1)) / pairs.sum(dim=(-2, -1)).clamp_min(1)


def _selector_readings_by_loop(model, chunk_gold: torch.Tensor, positions: torch.Tensor,
                               condition_ids: torch.Tensor, sink: dict) -> None:
    """Append this batch's per loop selector readings at the answer start positions to ``sink``.

    Three readings per loop, all at the last prompt token of each answer span (the state that has
    seen the question and the buffer and none of the answer):

    - ``mass``: the external share of the read, bucketed by condition, raw.
    - ``mass_per_chunk``: that mass over the number of visible chunks. The union softmax gives a
      buffer of more chunks more external mass whatever it holds, and ``mixed`` rows carry more
      chunks than ``distractors`` ones, so the gold-present against distractors-only AUROC is scored
      on this one; ``none`` rows are left out because their mass is zero by construction.
    - ``chunk_auroc``: per token, the within-buffer AUROC of gold against distractor shares (see
      ``_within_buffer_auroc``), over tokens whose buffer holds both kinds. Chance is 0.5.

    IR experts are averaged, as ``LoopMixtureOfExperts._selector_chunk_mass`` does. Eval only: this
    moves tensors to the host.
    """
    visible = model.moe.last_memory_visible
    modules = [m for m in model.moe.ir_modules if m.memory_weights_by_loop]
    if visible is None or not modules:
        return
    flat = positions.reshape(-1).nonzero().squeeze(-1)
    if flat.numel() == 0:
        return
    cond = condition_ids.reshape(-1)[flat]
    vis = visible[flat]                                          # [P, M]
    gold = chunk_gold.to(torch.bool).unsqueeze(0).expand_as(vis) & vis
    has_both = (gold.any(dim=-1) & (vis & ~gold).any(dim=-1))
    n_visible = vis.sum(dim=-1).clamp_min(1)
    for loop_idx in sorted(modules[0].memory_weights_by_loop):
        weights = torch.stack([
            m.memory_weights_by_loop[loop_idx][flat].float() for m in modules
            if m.memory_weights_by_loop.get(loop_idx) is not None
        ]).mean(dim=0)                                           # [P, M]
        weights = weights * vis
        mass = weights.sum(dim=-1)
        entry = sink.setdefault(loop_idx, {"mass": [], "mass_per_chunk": [], "cond": [],
                                           "chunk_auroc": [], "gold_share": []})
        entry["mass"].append(mass.cpu())
        entry["mass_per_chunk"].append((mass / n_visible).cpu())
        entry["cond"].append(cond.cpu())
        if has_both.any():
            share = weights[has_both] / mass[has_both].clamp_min(1e-12).unsqueeze(-1)
            entry["chunk_auroc"].append(
                _within_buffer_auroc(share, gold[has_both], vis[has_both]).cpu()
            )
            entry["gold_share"].append((share * gold[has_both]).sum(dim=-1).cpu())


def _summarize_selector_readings(sink: dict) -> dict:
    """``_selector_readings_by_loop``'s sink -> ``{loop: {mass_by_condition,
    mass_per_chunk_by_condition, mass_auroc, chunk_auroc, gold_share}}``, each entry present only
    when its classes are. ``chunk_auroc`` is the mean of the per token within-buffer AUROC."""
    out = {}
    for loop_idx, entry in sorted(sink.items()):
        mass = torch.cat(entry["mass"]).numpy()
        per_chunk = torch.cat(entry["mass_per_chunk"]).numpy()
        cond = torch.cat(entry["cond"]).numpy()
        reading = {
            "mass_by_condition": {
                c: float(mass[cond == i].mean()) for i, c in enumerate(CONDITIONS)
                if (cond == i).any()
            },
            "mass_per_chunk_by_condition": {
                c: float(per_chunk[cond == i].mean()) for i, c in enumerate(CONDITIONS)
                if (cond == i).any()
            },
        }
        gold_rows = np.isin(cond, [CONDITIONS.index("gold"), CONDITIONS.index("mixed")])
        distract_rows = cond == CONDITIONS.index("distractors")
        if gold_rows.any() and distract_rows.any():
            rows = gold_rows | distract_rows
            reading["mass_auroc"] = float(
                roc_auc(per_chunk[rows], gold_rows[rows].astype(np.float64))
            )
        if entry["chunk_auroc"]:
            reading["chunk_auroc"] = float(torch.cat(entry["chunk_auroc"]).mean())
            reading["gold_share"] = float(torch.cat(entry["gold_share"]).mean())
        out[loop_idx] = reading
    return out


@torch.no_grad()
def evaluate(model, dataset: SFTDataset, device: str, pad_token_id: int, max_batches: int,
             conversation_weighting: bool = False, selector_by_loop: bool = False,
             rank: int = 0, world_size: int = 1, gather=None):
    """Validation pass over the val split: CE on supervised tokens plus the calibration signals.

    Reports ``p_max``/top-1 accuracy because the acceptance criterion is about the abstention
    signal's calibration, not about val loss, and a fixed held-out slice shows drift in it far
    earlier than the noisy training log does.

    Runs at the full configured loop depth (no ``n_loops`` override, no loop-count sampling) and
    with subsampling off, so successive eval numbers are read at one fixed operating point.

    Args:
        conversation_weighting: mirror the training objective's per-conversation weighting into the
            reported CE. On means val CE tracks what is actually being minimized; it also means the
            number is not comparable to a run with it off. ``p_max``/top-1 stay token-level either
            way (``_chunked_linear_ce`` never weights them), so those two remain comparable across
            every checkpoint this repo has measured.

    When a batch carries ``condition_ids`` (the evidence corpus, if it was built with the ``.cond``
    sidecar), the returned dict also has ``per_condition_ce``: the same final-loop CE term the
    overall ``ce`` is built from, restricted to one condition's tokens at a time, computed without
    a second forward (see the call site below). On a split whose targets depend on the condition
    it compares an answer against a refusal; only on the fixed-target split, where every condition
    forces the real answer, is CE(none) - CE(gold) what the evidence is worth.
    ``per_condition_ce_by_loop`` is ``{loop: {condition: CE}}`` off the same call, so its final loop
    equals ``per_condition_ce``.

    When the corpus also carries the per chunk gold flag, ``selection`` is the held-out reading of
    the supervised selection term the objective adds (``LoopMixtureOfExperts.evidence_selection_term``).
    With a groundedness head, ``grounded_auroc`` is read over every answer start and
    ``grounded_auroc_evidence`` over those whose buffer holds at least one visible chunk.

    When a batch also carries ``source_ids`` (the held-out splits with a ``.src`` sidecar), the
    per condition pass is repeated per source: ``per_source_condition_ce`` is
    ``{source: {condition: CE}}`` over the same tokens and weights, whose token weighted average over
    sources equals ``per_condition_ce``, and ``per_source_tokens`` is ``{source: supervised tokens}``
    summed over every row of the source (a fixed split counts a question's answer once per condition).

    Args:
        selector_by_loop: also collect the selector's per loop readings at the answer start
            positions (see ``_selector_readings_by_loop``), returned as ``selector_by_loop``. Needs
            the gold flag and ``condition_ids``; silently absent without them.
        rank, world_size, gather: data parallel evaluation. Every rank iterates the same full
            stream and evaluates the batches whose index is ``rank`` modulo ``world_size``, among
            the first ``max_batches`` (a batch with no supervised token costs its slot, where one
            process would read on), then ``gather`` (``state -> [state per rank]``, rank order)
            hands every rank all the accumulators and each computes the same result through
            ``merge_eval_states``. One process (the default) is the old pass, byte for byte.
    """
    was_training = model.training
    model.eval()
    # the eval forwards would otherwise inflate the trained-token counter, which drives the
    # router-noise anneal, the checkpoint cadence and the reported progress (same guard as
    # pretrain.dry_run)
    token_count_before = model._token_tracker.num_tokens

    ce_sum, ce_weight_sum, token_sum = 0.0, 0.0, 0
    signal_sums = {"p_max": 0.0, "top1_acc": 0.0}
    n_batches = 0
    cond_ce_sum = {c: 0.0 for c in CONDITIONS}
    cond_weight_sum = {c: 0.0 for c in CONDITIONS}
    cond_ce_by_loop_sum = {}
    source_cond_sum, source_cond_weight, source_token_sum = {}, {}, {}
    # plain per-batch mean: the selection loss is already a mean over a batch's supervised,
    # evidence-bearing positions, and weighting it by supervised TOKENS would weight it by answer
    # length, which has nothing to do with how many chunks were ranked
    selection_sum, selection_batches = 0.0, 0
    contrastive_sum, contrastive_batches = 0.0, 0
    grounded_sum, grounded_batches = 0.0, 0
    grounded_scores, grounded_labels, grounded_has_evidence = [], [], []
    selector_sink = {}

    for batch_index, batch in enumerate(dataset):
        if world_size > 1:
            if batch_index >= max_batches:
                break
            if batch_index % world_size != rank:
                continue
        elif n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        document_ids = batch["document_ids"].to(device)
        cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
        pad_mask = input_ids == pad_token_id
        loss_weights = batch["loss_weights"].to(device) if conversation_weighting else None

        # the val split carries evidence too, and reading it WITHOUT is a different task: val CE
        # would then be measuring the model answering from memory, which is not what is being
        # trained and would drift away from the training curve for the wrong reason
        evidence = evidence_from_batch(model, batch, cu_seqlens)

        with te.autocast(enabled=USE_LOW_PRECISION, recipe=chosen_recipe):
            out = model(
                input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                return_aux_loss=True, return_hidden=True,
                evidence=evidence,
                token_mask=~pad_mask,
            )
            # held-out reading of the same term the objective adds (see pretrain.train_step): the
            # training log's copy is one batch of whatever the selector was just pushed toward, so
            # a selector that is memorizing the training slice's chunk order separates here first
            if evidence is not None and evidence.chunk_gold is not None:
                selection = model.moe.evidence_selection_term(
                    evidence.chunk_gold, supervised=predicting_positions(labels).reshape(-1)
                )
                if selection is not None:
                    selection_sum += selection.item()
                    selection_batches += 1
                # the retriever's term, held out, read whenever the objective carries it
                if TrainingConfig.evidence_contrastive_weight > 0.0:
                    contrastive = model.moe.evidence_contrastive_term(
                        evidence.chunk_gold, answer_start_positions(labels).reshape(-1),
                        temperature=TrainingConfig.evidence_contrastive_temperature,
                        loops=TrainingConfig.evidence_contrastive_loops,
                    )
                    if contrastive is not None:
                        contrastive_sum += contrastive.item()
                        contrastive_batches += 1
                if selector_by_loop and batch.get("condition_ids") is not None:
                    _selector_readings_by_loop(
                        model, evidence.chunk_gold, answer_start_positions(labels),
                        batch["condition_ids"].to(device), selector_sink,
                    )

                # the groundedness readout, held out. The BCE says it is training; the AUROC over
                # the same positions is the number the gate is actually read at, and the two can
                # disagree -- a head that predicts the corpus's base rate has a falling BCE and an
                # AUROC of 0.5, which is the failure this pass exists to make visible early.
                answerable = batch.get("answerable_ids")
                if answerable is not None and model.groundedness_head is not None:
                    answerable = answerable.to(device)
                    positions = answer_start_positions(labels)
                    grounded = model.groundedness_term(
                        evidence.chunk_gold, answerable, positions
                    )
                    if grounded is not None:
                        grounded_sum += grounded.item()
                        grounded_batches += 1
                        flat = positions.reshape(-1)
                        logits = model.groundedness_head(
                            model.moe.last_reader_output
                        ).reshape(-1)[flat]
                        gold_present = (
                            model.moe.last_memory_visible.to(logits.dtype)
                            @ evidence.chunk_gold.to(logits.dtype)
                        )[flat] > 0
                        target = gold_present & (answerable.reshape(-1)[flat] > 0)
                        grounded_scores.append(logits.float().cpu())
                        grounded_labels.append(target.cpu())
                        # replay rows carry no buffer and are always negatives, so the all rows
                        # AUROC is partly "is there evidence at all"; the evidence rows one is not
                        grounded_has_evidence.append(
                            model.moe.last_memory_visible[flat].any(dim=-1).cpu()
                        )
            hidden = out[0]
            extra_token_outputs = out[2] if model.has_mtp else None
            _, loss_ce, metrics = compute_mtp_loss(
                hidden, labels,
                mtp_outputs=extra_token_outputs,
                lm_head=model.mtp_head.lm_head if extra_token_outputs is not None else None,
                lambda_mtp=TrainingConfig.lambda_mtp,
                main_lm_head=model.lm_head,
                pad_mask=pad_mask,
                loop_ce_weights=TrainingConfig.loop_ce_weights,
                loop_ce_subsample=1.0,
                return_metrics=True,
                loss_weights=loss_weights,
            )

            condition_ids = batch.get("condition_ids")
            if condition_ids is not None:
                condition_ids = condition_ids.to(device)
                # the same per-token weight the overall CE uses (the conversation weight when that
                # objective is on, else a plain supervised mask), so restricting it to one
                # condition's tokens and re-running ONLY the lm_head + chunked CE (no second model
                # forward -- `hidden` is already computed above) gives the exact quantity `loss_ce`
                # would read if the batch had contained only that condition's tokens. mtp_outputs is
                # deliberately omitted here: loss_ce never depends on the MTP term (see
                # compute_mtp_loss), so skipping it halves the cost of this per-condition pass for
                # nothing lost.
                base_weight = loss_weights if loss_weights is not None else (labels != -100).float()
                source_ids = batch.get("source_ids")
                present_sources = []
                if source_ids is not None:
                    source_ids = source_ids.to(device)
                    present_sources = [s for s in torch.unique(source_ids).tolist() if s >= 0]
                    supervised = labels[:, 1:] != -100
                    for src in present_sources:
                        name = source_name(src)
                        source_token_sum[name] = source_token_sum.get(name, 0) + int(
                            (supervised & (source_ids[:, 1:] == src)).sum().item()
                        )
                for idx, cond in enumerate(CONDITIONS):
                    cond_weight = base_weight * (condition_ids == idx).float()
                    weight_total = float(cond_weight[:, 1:].sum().item())
                    if weight_total <= 0.0:
                        continue
                    # per loop readings off the same call: with subsampling off every loop is
                    # scored on the same tokens and weights, and the final loop's entry is the very
                    # tensor returned as cond_ce
                    _, cond_ce, cond_metrics = compute_mtp_loss(
                        hidden, labels,
                        lambda_mtp=TrainingConfig.lambda_mtp,
                        main_lm_head=model.lm_head,
                        pad_mask=pad_mask,
                        loop_ce_weights=TrainingConfig.loop_ce_weights,
                        loop_ce_subsample=1.0,
                        return_metrics=True,
                        loss_weights=cond_weight,
                    )
                    cond_ce_sum[cond] += cond_ce.item() * weight_total
                    cond_weight_sum[cond] += weight_total
                    for loop_idx, loop_ce in enumerate(cond_metrics["per_loop_ce"]):
                        loop_sums = cond_ce_by_loop_sum.setdefault(loop_idx, {})
                        loop_sums[cond] = loop_sums.get(cond, 0.0) + loop_ce.item() * weight_total

                    # the same condition's tokens split by the source of their conversation. The
                    # weights partition cond_weight exactly, so the token weighted average over
                    # sources reproduces the pooled number above
                    if source_ids is not None:
                        for src in present_sources:
                            src_weight = cond_weight * (source_ids == src).float()
                            src_total = float(src_weight[:, 1:].sum().item())
                            if src_total <= 0.0:
                                continue
                            _, src_ce = compute_mtp_loss(
                                hidden, labels,
                                lambda_mtp=TrainingConfig.lambda_mtp,
                                main_lm_head=model.lm_head,
                                pad_mask=pad_mask,
                                loop_ce_weights=TrainingConfig.loop_ce_weights,
                                loop_ce_subsample=1.0,
                                loss_weights=src_weight,
                            )
                            name = source_name(src)
                            sums = source_cond_sum.setdefault(name, {})
                            weights_ = source_cond_weight.setdefault(name, {})
                            sums[cond] = sums.get(cond, 0.0) + src_ce.item() * src_total
                            weights_[cond] = weights_.get(cond, 0.0) + src_total

        # weight each batch by its supervised token count: rows differ a lot in how much of them
        # is prompt, so an unweighted mean over batches is not the corpus mean. Under
        # conversation weighting the batch's CE is a per-conversation mean, so its denominator is
        # the batch's total weight instead -- mixing the two would over-count long-answer batches
        # in exactly the direction this phase is trying to remove.
        n_supervised = int((labels[:, 1:] != -100).sum().item())
        if n_supervised == 0:
            continue
        ce_weight = (
            float(loss_weights[:, 1:].sum().item()) if loss_weights is not None else n_supervised
        )
        ce_sum += loss_ce.item() * ce_weight
        ce_weight_sum += ce_weight
        token_sum += n_supervised
        for key in ("p_max", "top1_acc"):
            value = metrics.get(key)
            signal_sums[key] += (value.item() if value is not None else float("nan")) * n_supervised
        n_batches += 1

    model._token_tracker.num_tokens = token_count_before
    if was_training:
        model.train()

    if world_size > 1:
        merged = merge_eval_states(gather({
            "ce_sum": ce_sum, "ce_weight_sum": ce_weight_sum, "token_sum": token_sum,
            "n_batches": n_batches, "selection_sum": selection_sum,
            "selection_batches": selection_batches, "contrastive_sum": contrastive_sum,
            "contrastive_batches": contrastive_batches, "grounded_sum": grounded_sum,
            "grounded_batches": grounded_batches, "signal_sums": signal_sums,
            "cond_ce_sum": cond_ce_sum, "cond_weight_sum": cond_weight_sum,
            "cond_ce_by_loop_sum": cond_ce_by_loop_sum, "source_cond_sum": source_cond_sum,
            "source_cond_weight": source_cond_weight, "source_token_sum": source_token_sum,
            "grounded_scores": grounded_scores, "grounded_labels": grounded_labels,
            "grounded_has_evidence": grounded_has_evidence, "selector_sink": selector_sink,
        }))
        ce_sum, ce_weight_sum, token_sum = (merged["ce_sum"], merged["ce_weight_sum"],
                                            merged["token_sum"])
        n_batches = merged["n_batches"]
        selection_sum, selection_batches = merged["selection_sum"], merged["selection_batches"]
        contrastive_sum, contrastive_batches = (merged["contrastive_sum"],
                                                merged["contrastive_batches"])
        grounded_sum, grounded_batches = merged["grounded_sum"], merged["grounded_batches"]
        signal_sums, cond_ce_sum, cond_weight_sum = (
            merged["signal_sums"], {c: merged["cond_ce_sum"].get(c, 0.0) for c in CONDITIONS},
            {c: merged["cond_weight_sum"].get(c, 0.0) for c in CONDITIONS})
        cond_ce_by_loop_sum = merged["cond_ce_by_loop_sum"]
        source_cond_sum, source_cond_weight = merged["source_cond_sum"], merged["source_cond_weight"]
        source_token_sum = merged["source_token_sum"]
        grounded_scores, grounded_labels = merged["grounded_scores"], merged["grounded_labels"]
        grounded_has_evidence = merged["grounded_has_evidence"]
        selector_sink = merged["selector_sink"]

    if token_sum == 0 or ce_weight_sum == 0:
        return None
    result = {"ce": ce_sum / ce_weight_sum, "tokens": token_sum, "batches": n_batches}
    result.update({key: total / token_sum for key, total in signal_sums.items()})
    result["ppl"] = math.exp(min(result["ce"], 20.0))
    per_condition = {
        cond: cond_ce_sum[cond] / cond_weight_sum[cond]
        for cond in CONDITIONS if cond_weight_sum[cond] > 0.0
    }
    if per_condition:
        result["per_condition_ce"] = per_condition
        result["per_condition_ce_by_loop"] = {
            loop_idx: {cond: total / cond_weight_sum[cond] for cond, total in sums.items()}
            for loop_idx, sums in sorted(cond_ce_by_loop_sum.items())
        }
    if source_cond_sum:
        result["per_source_condition_ce"] = {
            name: {cond: total / source_cond_weight[name][cond] for cond, total in sums.items()}
            for name, sums in source_cond_sum.items()
        }
        result["per_source_tokens"] = dict(source_token_sum)
    if selection_batches:
        result["selection"] = selection_sum / selection_batches
    if contrastive_batches:
        result["contrastive"] = contrastive_sum / contrastive_batches
    if grounded_batches:
        result["grounded"] = grounded_sum / grounded_batches
        scores = torch.cat(grounded_scores).numpy()
        labels_cat = torch.cat(grounded_labels).numpy()
        has_evidence = torch.cat(grounded_has_evidence).numpy()
        # only defined with both classes present on the slice; a slice that happens to be all
        # grounded (or all not) gets no number rather than a misleading 0.5
        if 0 < labels_cat.sum() < labels_cat.size:
            result["grounded_auroc"] = float(roc_auc(scores, labels_cat))
        evidence_labels = labels_cat[has_evidence]
        if 0 < evidence_labels.sum() < evidence_labels.size:
            result["grounded_auroc_evidence"] = float(
                roc_auc(scores[has_evidence], evidence_labels)
            )
    if selector_sink:
        result["selector_by_loop"] = _summarize_selector_readings(selector_sink)
    return result


def cli_overrides(args) -> dict:
    """The profile overrides the command line asks for.

    A flag left unset (None) or empty is no override; a ``--grad-accumulation-steps 0`` is kept so
    the ``< 1`` check refuses it instead of the run silently using the profile's value.

    Args:
        args: the parsed arguments.

    Returns:
        ``data_dir``, ``train_split``, ``val_split`` and ``grad_accumulation_steps`` that were given.

    Raises:
        SystemExit: ``--grad-accumulation-steps`` below 1.
    """
    overrides = {name: value for name, value in (
        ("data_dir", getattr(args, "data_dir", None)),
        ("train_split", getattr(args, "train_split", None)),
        ("val_split", getattr(args, "val_split", None)),
        ("grad_accumulation_steps", getattr(args, "grad_accumulation_steps", None)),
    ) if value is not None and value != ""}
    if overrides.get("grad_accumulation_steps", 1) < 1:
        raise SystemExit("--grad-accumulation-steps must be at least 1")
    return overrides


def estimate_rows_only(args, cfg, log) -> int:
    """Compute the plan split's row estimate for the given rank counts, write the cache, exit 0.

    CPU only: no accelerator, no model. The corpus driver runs it after the merges so a training
    launch reads the cache instead of replaying hundreds of millions of documents.

    Args:
        args: parsed arguments (``plan_split``, ``estimate_world_sizes``).
        cfg: the profile class after the overrides.
        log: the rank log.

    Returns:
        ``EXIT_OK``.
    """
    data_dir = os.path.join(BASE_DIR, cfg.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    split = getattr(args, "plan_split", None) or cfg.train_split
    dataset = make_dataset(data_dir, split, tokenizer, cfg)
    mtp_tokens = ModelConfig.Params["mtp_num_extra_tokens"]
    key = rows_cache_key(dataset, cfg.Seq_length, mtp_tokens)
    path = rows_cache_path(data_dir, split)
    for world in sorted({int(w) for w in str(args.estimate_world_sizes).split(",") if w.strip()}):
        workers = NUM_DATA_WORKERS * world
        cached = read_rows_cache(path, key, workers)
        if cached is not None:
            log.info(f"row estimate for {split!r} at {world} ranks already cached: {cached:,}")
            continue
        started = time.time()
        rows = compute_rows_estimate(dataset, cfg.Seq_length, mtp_tokens, workers)
        write_rows_cache(path, key, workers, rows)
        log.info(f"row estimate for {split!r} at {world} ranks: {rows:,} packed rows per epoch "
                 f"in {time.time() - started:.0f}s -> {path}")
    return EXIT_OK


def destroy_process_group_quietly() -> None:
    """Tear down the process group at exit, so torch does not warn that it was never destroyed
    (that line lands after the last step line). A failure here never changes the exit code."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as e:
        logger.warning(f"could not destroy the process group: {e!r}")


def sft(args):
    """Run the profile and return its exit code; a multi process run destroys its process group on
    every way out (returns of 0, 10, 20 and 30 and an exception)."""
    _, world_size = dist_env()
    try:
        return _sft_impl(args)
    finally:
        if world_size > 1:
            destroy_process_group_quietly()


def _sft_impl(args):
    rank, world_size = dist_env()
    log = RankLog(logger, rank)
    # one function, three profiles. --repair / --ir swap the config class, the phase label and the
    # checkpoint directory and nothing else: see this module's docstring for why the repair pass is
    # not a second script.
    chosen = [n for n, on in (("--repair", args.repair), ("--ir", args.ir),
                              ("--evidence", args.evidence)) if on]
    if len(chosen) > 1:
        raise SystemExit(f"{' and '.join(chosen)} are different profiles; pick one")
    if args.evidence:
        cfg, phase, checkpoint_dir = EvidenceConfig, EVIDENCE_PHASE, EVIDENCE_CHECKPOINT_DIR
    elif args.ir:
        cfg, phase, checkpoint_dir = IRConfig, IR_PHASE, IR_CHECKPOINT_DIR
    elif args.repair:
        cfg, phase, checkpoint_dir = RepairConfig, REPAIR_PHASE, REPAIR_CHECKPOINT_DIR
    else:
        cfg, phase, checkpoint_dir = SFTConfig, SFT_PHASE, SFT_CHECKPOINT_DIR
    # a subclass rather than a mutation, so every other attribute keeps inheriting and the config
    # class other code imports is untouched. The effective values are saved in the checkpoint and
    # a resume on other ones is refused without --switch-split, so relaunch with the same flags
    overrides = cli_overrides(args)
    if overrides:
        cfg = type(cfg.__name__, (cfg,), overrides)
        log.info(f"data overrides: {overrides}")
    if getattr(args, "estimate_rows_only", False):
        return estimate_rows_only(args, cfg, log)
    ceiling_by_source, ceiling_all = {}, None
    if getattr(args, "ceiling_json", None):
        with open(args.ceiling_json, "r", encoding="utf-8") as f:
            ceiling = json.load(f)
        ceiling_all = (ceiling.get("all") or {}).get("ceiling")
        ceiling_by_source = {name: entry["ceiling"] for name, entry in
                             (ceiling.get("by_source") or {}).items() if "ceiling" in entry}
        log.info(f"in-context ceiling read from {args.ceiling_json}: pooled {ceiling_all}, "
                    f"by source {ceiling_by_source}")
    # two variants of the same profile (different seeds, identical everything else) would otherwise
    # share a directory, and the second would silently RESUME the first instead of starting from
    # its own seed -- the resume path only checks the phase label, which is the same for both
    if args.run_name:
        checkpoint_dir = f"{checkpoint_dir}_{args.run_name}"
    if run_already_complete(checkpoint_dir, phase):
        if getattr(args, "switch_split", False):
            # a warm-up that finished every epoch of its split left a final checkpoint: the main
            # launch would read it as a complete run and exit 0 without training
            try:
                saved_data = final_checkpoint_data(
                    os.path.join(checkpoint_dir, ckpt_lib.final_name(phase)))
                moved = [c for c in split_changes(saved_data, {"data_dir": cfg.data_dir,
                                                               "train_split": cfg.train_split,
                                                               "val_split": cfg.val_split})
                         if c[0] != "data_dir"]
            except Exception as e:
                log.error(f"--switch-split: could not read the final checkpoint's data fields: {e!r}")
                return EXIT_RESUME_FAILED
            if moved:
                log.error(
                    f"--switch-split refused: {ckpt_lib.final_name(phase)} exists in {checkpoint_dir}, "
                    f"so the run on {saved_data.get('train_split')!r} finished all its epochs and the "
                    f"switch would read it as a complete run. The warm launch must stop at its split's "
                    f"epoch end (--max-epochs-this-split 1)."
                )
                return EXIT_RESUME_FAILED
        log.warning(
            f"run already complete: {ckpt_lib.final_name(phase)} is the newest checkpoint in "
            f"{checkpoint_dir} and run_state.json names it; nothing to train and exiting 0. A "
            f"longer run on the same data needs a new run directory seeded from it with -c (and "
            f"--switch-split if the split changes); a raised epoch count or token target on this "
            f"directory is not picked up."
        )
        return EXIT_OK

    # built before the model and the datasets: under several processes every rank must put its model
    # on accelerator.device (cuda:local_rank), not on cuda:0, and the shard of the data it reads
    # is its process index
    accelerator = Accelerator(**accelerator_kwargs(world_size, cfg.grad_accumulation_steps))
    if accelerator.num_processes != world_size or accelerator.process_index != rank:
        raise SystemExit(
            f"the launcher environment says rank {rank} of {world_size} but accelerate sees "
            f"{accelerator.process_index} of {accelerator.num_processes}"
        )
    is_main = accelerator.is_main_process
    device = accelerator.device
    if world_size > 1:
        log.info(
            f"data parallel: {world_size} processes, batch {cfg.Batch_size} per GPU x accum "
            f"{cfg.grad_accumulation_steps} -> {cfg.Batch_size * world_size * cfg.grad_accumulation_steps} "
            f"rows per optimizer step; logged loss, CE, aux, p_max, top1, selection, contrastive, "
            f"grounded and per loop CE are means over the ranks, the IR entropy and the external mass "
            f"are rank 0's slice"
        )

    data_dir = os.path.join(BASE_DIR, cfg.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    log.info(f"Tokenizer loaded from {TOKENIZER_DIR} with vocab size {tokenizer.vocab_size}")

    log.info(f"Using device: {device}")
    log_precision_mode()
    log.info(
        f"Profile: {phase} (lr={cfg.lr:.1e}, {cfg.num_epochs} epoch(s), "
        f"splits {cfg.train_split}/{cfg.val_split}, per-conversation loss weighting "
        f"{'ON' if cfg.conversation_loss_weighting else 'off'}) -> {checkpoint_dir}"
    )

    train_dataset = make_dataset(data_dir, cfg.train_split, tokenizer, cfg,
                                 rank=rank, world_size=world_size)
    # a stable order makes successive eval numbers comparable
    val_dataset = make_dataset(data_dir, cfg.val_split, tokenizer, cfg, shuffle=False)
    fixed_dataset = None
    fixed_split = getattr(cfg, "fixed_split", "")
    if fixed_split:
        if not os.path.isfile(os.path.join(data_dir, f"{fixed_split}.ev")):
            raise SystemExit(
                f"{fixed_split} is missing from {data_dir}: build it with "
                f"`python scripts/prepare_evidence_data.py --heldout --max-evidence-tokens 4608`, "
                f"or set evidence.fixed_split: \"\" to run without the kill number"
            )
        # on-disk order, never shuffled: the split is written question-major, so the first N
        # batches hold every condition of the same questions and the gap is paired
        fixed_dataset = make_dataset(data_dir, fixed_split, tokenizer, cfg, shuffle=False)
    dataloader = DataLoader(train_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS,
                            prefetch_factor=2)

    # the schedule is fitted to the plan split when one is named (the main split on every launch of
    # the main run, warm-up included), else to the training split as before
    plan_split = getattr(args, "plan_split", None) or None
    plan_dataset = train_dataset
    if plan_split and plan_split != cfg.train_split:
        plan_dataset = make_dataset(data_dir, plan_split, tokenizer, cfg,
                                    rank=rank, world_size=world_size)
        log.info(f"schedule planned on {plan_split!r} ({plan_dataset.num_docs:,} documents), "
                 f"training on {cfg.train_split!r}")

    # one stream per (rank, worker) pair, so the estimate replays NUM_DATA_WORKERS * world_size
    # streams and counts the rows of every rank together. At hundreds of millions of documents the
    # replay takes minutes and gigabytes, so a multi rank job reads it from the cache that
    # --estimate-rows-only wrote and never computes it
    mtp_tokens = ModelConfig.Params["mtp_num_extra_tokens"]
    cache_path = rows_cache_path(data_dir, plan_split or cfg.train_split)
    cache_key = rows_cache_key(plan_dataset, cfg.Seq_length, mtp_tokens)
    rows_per_epoch, how = resolve_rows_estimate(
        cache_path, cache_key, world_size,
        lambda workers: compute_rows_estimate(plan_dataset, cfg.Seq_length, mtp_tokens, workers),
        write_on_compute=is_main)
    if how == "cache":
        log.info(f"row estimate from cache: {rows_per_epoch:,} ({cache_path})")
    elif how == "refused":
        log.error(
            f"no valid row estimate for {plan_split or cfg.train_split!r} at {world_size} ranks in "
            f"{cache_path} (missing, an older cache format, written for other inputs, or for another rank count). Computing "
            f"it inside the multi rank job would hold every rank at a collective: run `python "
            f"scripts/sft.py --evidence --estimate-rows-only --data-dir {cfg.data_dir} --train-split "
            f"{cfg.train_split}" + (f" --plan-split {plan_split}" if plan_split else "")
            + f" --estimate-world-sizes {world_size}` first (CPU only)."
        )
        return EXIT_NO_ROWS_CACHE
    total_steps = plan_total_steps(rows_per_epoch, cfg.num_epochs, cfg.Batch_size,
                                   cfg.grad_accumulation_steps, world_size)
    log.info(
        f"{phase} plan: ~{rows_per_epoch:,} packed rows/epoch x {cfg.num_epochs} epochs "
        f"-> ~{total_steps:,} optimizer steps at batch {cfg.Batch_size} x accum "
        f"{cfg.grad_accumulation_steps}" + (f" x {world_size} ranks" if world_size > 1 else "")
    )

    # dropout override only; every other model hyperparameter must match the pretrained checkpoint.
    # The shape-bearing ones are read off the SEED rather than the yaml, same rule the eval scripts
    # follow: a seed is produced by a migration, and the yaml flip that matches it is a separate
    # manual step that can be forgotten or done twice. Getting it from the file makes an arm and its
    # control differ by what their seeds differ by, which is the comparison the run exists to make.
    model_params = cfg.model_params()
    if args.checkpoint:
        seed_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(BASE_DIR, args.checkpoint)
        # mmap: only the tensors' metadata is read here, and the same file is loaded again in full
        # by load_pretrained_weights below -- this must not cost a second 3.8GB read
        seed_keys = torch.load(seed_path, map_location="cpu", mmap=True)["model_state_dict"]
        model_params = model_params_for_state_dict(seed_keys, model_params)
        del seed_keys
    if args.reader_no_rotary:
        if not args.evidence:
            raise SystemExit("--reader-no-rotary only applies to --evidence")
        # the seed decides every other mode; this one is chosen here because no seed carries it yet.
        # The run's own checkpoints do, so its resumes must pass the flag again or fail to load
        model_params["evidence_reader_rotary"] = False
        log.info("evidence reader: no rotary (query and evidence keys unrotated)")
    if args.reader_kv:
        if not args.evidence:
            raise SystemExit("--reader-kv only applies to --evidence")
        if args.reader_no_rotary:
            raise SystemExit("--reader-kv places evidence by rotation and has no no-rotary mode")
        if not model_params.get("evidence_port"):
            raise SystemExit("--reader-kv needs a seed with the evidence port")
        model_params["evidence_reader"] = "kv"
        log.info("evidence reader: key/value (evidence as leading keys of shared_attn, every loop)")
    model = TinyMoETransformer(**model_params).to(device).to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    model.moe.set_contrastive_query_stash(
        TrainingConfig.evidence_contrastive_weight > 0.0
        and TrainingConfig.evidence_contrastive_trunk_grad < 1.0
    )
    model._token_tracker.pad_token_id = tokenizer.pad_token_id
    # router exploration noise is fully annealed by ~1B pretraining tokens; SFT is not exploration
    model.moe.set_router_noise(0.0)
    # counted in every mode so the step line reads the load under the aux term too; the rate
    # train_step passes is zero outside loss-free, so the bias never moves there
    model.moe.set_balance_tracking(True)
    log.info(f"moe balancing: {TrainingConfig.moe_balancing}" + (
        f" (bias rate {TrainingConfig.moe_balance_bias_rate})"
        if TrainingConfig.moe_balancing == "loss_free" else f" (weight {TrainingConfig.aux_loss_weight})"
    ))
    # before the param groups are built, so the gate lands in none of them and gets no fp32 master
    frozen_gate = None
    if args.evidence and getattr(cfg, "freeze_evidence_gate", False):
        frozen_gate = model.moe.evidence_gate_scale
        if frozen_gate is not None:
            frozen_gate.requires_grad_(False)

    # the two profiles that set fresh_lr disagree on what "fresh" means: the IR profile rebuilt the
    # whole ir_module subtree from scratch, but the evidence profile's table is that same subtree
    # carrying a full sharpening run, not a rebuild -- only its own new reader/adapter tensors are
    # fresh there. Neither of the other two profiles sets fresh_lr, so the predicate is never read.
    if args.ir:
        fresh_predicate = lambda name: is_rebuilt_ir_param(name) or is_fresh_loop_param(name)
    elif args.evidence:
        fresh_predicate = is_fresh_loop_param
    else:
        fresh_predicate = None
    param_groups, master_pairs = build_sft_param_groups(
        model, cfg.weight_decay, fresh_lr=getattr(cfg, "fresh_lr", None),
        is_fresh_param=fresh_predicate,
    )
    # the masters are what the optimizer steps, so the partition rule sees them under the model
    # parameter's name
    names = {id(p): n for n, p in model.named_parameters()}
    names_by_master = {id(master): names[id(param)] for param, master in master_pairs}
    optimizer = build_optimizer(TrainingConfig.optimizer, param_groups, names_by_master, lr=cfg.lr)
    log.info(f"optimizer {TrainingConfig.optimizer}: {partition_summary(optimizer)}")
    scheduler = build_sft_scheduler(optimizer, total_steps, cfg)

    os.makedirs(checkpoint_dir, exist_ok=True)
    if is_main:
        ckpt_lib.cleanup_stale_files(checkpoint_dir)
    run_state_path = os.path.join(checkpoint_dir, "run_state.json")
    run_state = ckpt_lib.read_run_state(run_state_path)
    resume_slack = 2 * cfg.checkpoint_every_tokens

    start_epoch, step_offset, start_doc_idx = 0, 0, 0
    losses, resumed, resumed_kill_checked = [], False, False
    start_token_count, token_count = 0, 0
    # what this launch trains on, after the command line overrides; saved in every checkpoint
    data_fields = {"data_dir": cfg.data_dir, "train_split": cfg.train_split,
                   "val_split": cfg.val_split}
    resumed_masters, resumed_rng, steps_at_last_save = None, {}, None

    try:
        found = ckpt_lib.find_resume_checkpoint(
            checkpoint_dir,
            lambda path: load_sft_checkpoint(model, optimizer, scheduler, path, phase, checkpoint_dir),
        )
    except RuntimeError as e:
        # checkpoints exist and none loads: with a run_state.json behind them that is a lost run, not
        # a fresh start, and the supervisor must not restart it
        if check_run_state(run_state, phase, None, resume_slack) is None:
            raise
        log.error(str(e))
        log.error(check_run_state(run_state, phase, None, resume_slack))
        return EXIT_RESUME_FAILED
    # the check the pretraining trainer makes: a directory that lost its checkpoints, or whose newest
    # loadable one is far behind what run_state.json recorded, would otherwise train from the seed
    # (or from old ground) again behind a plausible loss curve
    problem = check_run_state(run_state, phase,
                              found[1]["token_count"] if found is not None else None, resume_slack)
    if problem is not None:
        log.error(problem)
        return EXIT_RESUME_FAILED
    plan = {"total_steps": total_steps, "rows_per_epoch": rows_per_epoch, "world_size": world_size,
            "grad_accumulation_steps": cfg.grad_accumulation_steps, "plan_split": plan_split}
    head_docs = getattr(train_dataset, "epoch0_head_docs", None)
    data_fields_saved = {"epoch0_head_docs": head_docs,
                         "num_docs": getattr(train_dataset, "num_docs", None)}
    if head_docs is not None:
        log.info(f"epoch 0 head-first order: {head_docs:,} head documents "
                 f"({train_dataset.order_path})")
    if found is not None:
        _, state = found
        moved = plan_changed(state, plan)
        if moved:
            log.warning(
                "the plan differs from the one this run's cosine was fitted to: "
                + "; ".join(f"{key} {old:,} -> {new:,}" if isinstance(old, int) and isinstance(new, int)
                            else f"{key} {old!r} -> {new!r}" for key, old, new in moved)
                + f". Keeping the new total_steps ({total_steps:,}); the schedule's shape changes "
                f"under the resumed run (a split switch, another number of ranks or an estimator "
                f"change)"
            )
        start_epoch = state["epoch"]
        step_offset = state["step"]
        token_count = state["token_count"]
        start_token_count = state["start_token_count"]
        start_doc_idx = state["global_offset"]
        losses = state["losses"]
        resumed_kill_checked = state["kill_checked"]
        if state["seed"] != cfg.seed:
            # the resume position indexes into a permutation generated from the seed; reading it
            # back under a different seed silently reshuffles which conversations were "already
            # seen", so refuse rather than half-repeat and half-skip an epoch
            raise SystemExit(
                f"checkpoint was written with seed={state['seed']} but config.yaml now says "
                f"{cfg.seed}. The document order (and therefore the resume position) is a "
                f"function of the seed -- restore the old seed or start a fresh run directory."
            )
        if not state["data"]:
            log.info(
                f"the checkpoint does not record its data (written before the field existed); "
                f"resuming on {data_fields}"
            )
        else:
            all_changes = check_split_switch(state["data"], data_fields,
                                             getattr(args, "switch_split", False))
            moved_dir = [c for c in all_changes if c[0] == "data_dir"]
            changes = [c for c in all_changes if c[0] != "data_dir"]
            if moved_dir and "train_split" not in {c[0] for c in changes}:
                saved_docs = state.get("num_docs")
                now_docs = getattr(train_dataset, "num_docs", None)
                if saved_docs is not None and now_docs is not None and saved_docs != now_docs:
                    log.error(
                        f"the corpus moved ({moved_dir[0][1]!r} -> {moved_dir[0][2]!r}) and the split "
                        f"{cfg.train_split!r} has {now_docs:,} documents where the checkpoint trained "
                        f"on {saved_docs:,}: this is not the same corpus, so the resume position "
                        f"means something else. Restore the corpus or start a fresh run directory."
                    )
                    return EXIT_RESUME_FAILED
                log.warning(
                    f"data_dir moved: {moved_dir[0][1]!r} -> {moved_dir[0][2]!r}; same split "
                    f"{cfg.train_split!r}, resuming as a plain resume (document count "
                    f"{now_docs if now_docs is None else format(now_docs, ',')} "
                    f"{'matches the checkpoint' if saved_docs is not None else 'not recorded'})"
                )
            if changes:
                refusal = None if "train_split" not in {c[0] for c in changes} else (
                    check_switch_state(state["epoch"], state["global_offset"], state.get("num_docs"),
                                       getattr(train_dataset, "num_docs", 0), head_docs))
                if refusal is not None:
                    log.error(refusal)
                    return EXIT_RESUME_FAILED
                log.warning(
                    "--switch-split: resuming on other data than the checkpoint's: "
                    + "; ".join(f"{key} {old!r} -> {new!r}" for key, old, new in changes)
                    + ". The document position carries over unchanged."
                )
            elif getattr(args, "switch_split", False):
                log.info("--switch-split passed but the data match the checkpoint's")
            check_head_docs(state.get("epoch0_head_docs"), head_docs, not changes)
        resumed_masters, resumed_rng = state.pop("fp32_masters"), state["rng"]
        steps_at_last_save = state["optimizer_steps"]
        model._token_tracker.num_tokens = token_count
        resumed = True
        log.info(
            f"Resumed {phase} at epoch {start_epoch}, position {start_doc_idx:,}, "
            f"{(token_count - start_token_count) / 1e6:.1f}M {phase} tokens"
        )
    else:
        init_path = args.checkpoint
        if args.from_hub:
            repo = args.hub_repo or TrainingConfig.upload_repo(HF_UPLOAD_REPO)
            if not repo:
                raise SystemExit(
                    "--from-hub needs a repo: set training.hf_upload_repo in config.yaml or pass "
                    "--hub-repo"
                )
            token = get_hf_token()
            if not is_main:
                accelerator.wait_for_everyone()      # rank 0 downloads first, the rest hit the cache
            init_path = pull_from_hub(repo, args.hub_file, PRETRAINED_DIR, token)
            try:
                pull_from_hub(repo, "manifest.json", BASE_DIR, token)
            except Exception as e:
                # the manifest matters for prepare_sft_data.py (holdout hashes), not for training
                log.warning(f"could not pull manifest.json from {repo}: {e}")
            if is_main and world_size > 1:
                accelerator.wait_for_everyone()
        if not init_path:
            raise SystemExit(
                f"no {phase} checkpoint to resume and no checkpoint to initialize from -- pass "
                + ("-c <path to an SFT checkpoint, e.g. "
                   "ckpts/trained/checkpoint_sft_final_phase0.pt>" if args.repair
                   else "--from-hub, or -c <path to checkpoint_phase2_final.pt>")
            )
        start_token_count = load_pretrained_weights(model, init_path)
        token_count = start_token_count
        model._token_tracker.num_tokens = token_count

    # the masters were cloned from the random init at optimizer-construction time, before either
    # branch above loaded weights into the model -- set them or the first step would undo the
    # entire pretrained state. A resume restores the saved masters when the checkpoint has them,
    # else reseeds from bf16 like a fresh run. Adam's moments came back by param-group position on
    # a resume, which an in place copy does not disturb.
    named_masters = [(names[id(param)], param, master) for param, master in master_pairs]
    if resumed:
        log.info(restore_fp32_masters(named_masters, resumed_masters))
    else:
        with torch.no_grad():
            for bf16_param, master in master_pairs:
                master.data.copy_(bf16_param.data.float())
    resumed_masters = None

    if frozen_gate is not None:
        gate_value = float(frozen_gate.detach().float().item())
        if gate_value != 0.0:
            log.warning(
                f"evidence_gate_scale loaded as {gate_value:.4e}; zeroing it, since the gate is "
                f"frozen and only a zero scale makes it exactly 1.0"
            )
            with torch.no_grad():
                frozen_gate.zero_()
        log.info("evidence_gate_scale frozen at 0 (ungated reader, no gradient)")

    if args.evidence:
        # read once: two zero matrices in series (g_proj inside the table module, direct_gate on the
        # expert) each get a gradient proportional to the other, so both zero means the IR value
        # path stays exactly zero for the whole run and the selector trains only through the
        # selection loss
        for expert in model.moe.experts:
            if not hasattr(expert, "ir_module"):
                continue
            g_rms = expert.ir_module.g_proj.weight.detach().float().pow(2).mean().sqrt().item()
            gate = getattr(expert, "direct_gate", None)
            d_rms = gate.weight.detach().float().pow(2).mean().sqrt().item() if gate is not None else None
            log.info(
                f"seed IR value path: |g_proj|rms {g_rms:.3e}"
                + (f", |direct_gate|rms {d_rms:.3e}" if d_rms is not None else "")
            )
            if g_rms == 0.0 and d_rms == 0.0:
                log.warning(
                    "g_proj and direct_gate are both exactly zero: they hold each other at zero, so "
                    "the IR expert's output is zero for this run and the selector learns only from "
                    "the selection loss (the frozen gate removes its other path through the reader)"
                )

    # same stop contract as pretraining, so an unattended/interruptible box gets a checkpoint out
    # of a SIGTERM instead of losing everything since the last save. clear_sentinel() first: a STOP
    # left over from a previous run would otherwise kill every relaunch before it trains a step.
    control = RunControl(checkpoint_dir)
    keep_stop = bool(getattr(args, "keep_stop", False))
    if is_main and not keep_stop:
        # only rank 0 touches the STOP file; every rank's signal handlers set their own flags and the
        # flags meet in the log step's reduction. --keep-stop leaves it for the check before the
        # first step, so a STOP dropped while a container was starting is not lost
        control.clear_sentinel()
    control.install()

    upload_repo = args.upload_repo if args.upload_repo is not None else cfg.upload_repo(HF_UPLOAD_REPO)
    if not upload_repo and is_main:
        log.warning(
            f"{phase} uploads are OFF ({phase}.hf_upload_repo is empty). Fine locally; on a rented "
            "box the checkpoints die with the instance -- pass --upload-repo <repo> there."
        )
    hf = HFSync(upload_repo if is_main else "", token=get_hf_token())
    loss_png = os.path.join(checkpoint_dir, "loss_graph.png")
    experts_png = os.path.join(checkpoint_dir, "expert_selection.png")
    status_path = os.path.join(checkpoint_dir, "status.json")

    def broadcast_from_main(tensor):
        # NCCL cannot broadcast a CPU tensor (AdamW's step counter is one)
        if tensor.is_cuda:
            torch.distributed.broadcast(tensor, src=0)
        else:
            moved = tensor.to(device)
            torch.distributed.broadcast(moved, src=0)
            tensor.copy_(moved.cpu())

    def reduce_op(op):
        def reduce(tensor):
            tensor = tensor.to(device)
            torch.distributed.all_reduce(tensor, op=op)
            return tensor
        return reduce

    if world_size > 1:
        # DDP broadcasts rank 0's parameters and buffers when it wraps the model, which would hide
        # a rank that began from other weights while its masters and moments kept the local ones
        check_ranks_identical(
            [(f"param {n}", p) for n, p in model.named_parameters()]
            + [(f"buffer {n}", b) for n, b in model.named_buffers()]
            + [(f"master {n}", m) for n, _, m in named_masters],
            reduce_op(torch.distributed.ReduceOp.MIN), reduce_op(torch.distributed.ReduceOp.MAX),
        )
        # the accelerate loader is not used: with an iterable dataset it would shard the stream a
        # second time (IterableDatasetShard), and the datasets already read their own share. The
        # model is NOT wrapped in DDP either: the LM heads run outside its forward, so its reducer
        # would never see their gradients; train_step averages every gradient in fp32 instead
        optimizer = accelerator.prepare(optimizer)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    unwrapped_model = accelerator.unwrap_model(model)

    full_n_loops = ModelConfig.Params["n_loops"]
    # seeded identically on every rank on purpose: the sampled depth must match across ranks (the
    # collectives and the balancer's counts assume it). The CUDA RNG (the MTP subsample's randperm,
    # dropout) is per rank and stays that way, which is fine: those only need to be random
    loop_rng = random.Random(cfg.seed)
    if rank > 0 and torch.cuda.is_available():
        # a different dropout and subsample stream per rank on a fresh run (a resume reseeds below)
        torch.cuda.manual_seed(cfg.seed + 1000 * rank)
    if resumed:
        restored = restore_rng_states(loop_rng, resumed_rng, rank)
        log.info(f"RNG states restored: {restored}" if restored else
                    "the checkpoint records no RNG state; depth sampling restarts from the seed")
    # one entry per (rank, worker) stream, indexed by the worker_id the dataset reports
    stream_stride = max(NUM_DATA_WORKERS, 1) * world_size
    worker_state = torch.full((stream_stride,), -1, dtype=torch.long, device=device)

    # the ranks' worker positions as MAX-merged at the latest log step, for the interrupt save
    last_merged_state = None

    def snapshot_global_offset(fallback, collective=True):
        # same conservative min-across-workers rule as pretrain.py: a worker at position p next
        # wants p + the stride, so the smallest such value skips nobody's unconsumed work. Under data
        # parallel training each rank only wrote its own streams' entries, so the ranks' states are
        # merged with a MAX first (every rank calls this, being a collective)
        state = worker_state
        if world_size > 1 and collective:
            state = worker_state.clone()
            torch.distributed.all_reduce(state, op=torch.distributed.ReduceOp.MAX)
        return worker_offset(state.cpu().tolist(), stream_stride, fallback)

    timer = time.time()
    last_log_time = timer
    last_token_count = unwrapped_model._token_tracker.sync()
    # this rank's own tracker total as of the last log: the reduction sums the ranks' deltas, so the
    # global count is the resumed or seeded count plus what every rank has trained since
    last_local_total = last_token_count
    gather_eval = all_gather_states(world_size) if world_size > 1 else None
    max_epochs_this_split = int(getattr(args, "max_epochs_this_split", 0) or 0)

    def synced_token_count(count, local_total):
        # the end of epoch count, as the final save reads it: the ranks left their last chunk
        # together, so the reduction is matched on every rank
        if world_size == 1:
            return unwrapped_model._token_tracker.sync()
        delta = torch.tensor([float(unwrapped_model._token_tracker.sync() - local_total)],
                             dtype=torch.float64, device=device)
        torch.distributed.all_reduce(delta)
        return count + int(round(delta.item()))
    target_tokens = None  # filled in after the first log interval, once tokens/row is known

    def save_and_sync(epoch, step, loss_value, tokens, final=False, collective=True):
        """Write a checkpoint. Every rank calls it (the save is a collective: barriers around it and
        the merge of the ranks' stream positions), rank 0 alone writes. ``collective=False`` is the
        interrupt path, where only the interrupted rank may be alive: rank 0 writes with its own
        stream positions and no barrier."""
        nonlocal steps_at_last_save
        use_collectives = collective and world_size > 1
        if use_collectives:
            accelerator.wait_for_everyone()
        if world_size > 1 and not collective:
            # only this rank's streams are known, and the ranks' streams drift apart by thousands
            # of documents within an epoch: save the last position every rank agreed on at a log
            # step. The resume then repeats at most a log interval of batches and skips nobody's
            global_offset = (worker_offset(last_merged_state.cpu().tolist(), stream_stride,
                                           start_doc_idx)
                             if last_merged_state is not None else start_doc_idx)
        else:
            global_offset = snapshot_global_offset(start_doc_idx, collective=collective)
        if is_main:
            name = (ckpt_lib.final_name(phase) if final
                    else ckpt_lib.rolling_name(phase, tokens - start_token_count, loss_value))
            path = os.path.join(checkpoint_dir, name)
            sft_fields = dict(data_fields)
            sft_fields["loop_rng_state"] = loop_rng.getstate()
            # rank 0's CUDA state; a resumed rank derives its own from it (restore_rng_states)
            sft_fields["cuda_rng_state"] = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            # the plan the cosine was fitted to, compared on the next resume
            sft_fields.update(plan)
            sft_fields.update(data_fields_saved)
            # the masters and the weights are identical on every rank after a reduced step
            save_sft_checkpoint(
                unwrapped_model, optimizer, scheduler, path,
                epoch=epoch, step=step, token_count=tokens, start_token_count=start_token_count,
                global_offset=global_offset, losses=losses,
                seed=cfg.seed, phase=phase, kill_checked=kill_checked, sft_fields=sft_fields,
                fp32_masters=fp32_master_state(named_masters) if SAVE_FP32_MASTERS else None,
            )
            ckpt_lib.write_run_state(run_state_path, phase, tokens, name)
            try:
                save_loss_graph(losses, loss_png)
                save_expert_selection_graph(unwrapped_model.moe.expert_tracker.get_stats(), experts_png)
            except Exception as e:
                log.error(f"Error occurred while saving graphs: {e}")

            repo_dir = f"{phase}/final" if final else phase
            hf.upload(path, f"{repo_dir}/{name}", droppable=not final)
            for local, remote in ((status_path, f"{phase}/status.json"),
                                  (loss_png, f"{phase}/graphs/loss_graph.png")):
                if os.path.isfile(local):
                    hf.upload(local, remote)
            # retention: only deletes what is both outside the window AND confirmed uploaded. With
            # uploads off (the local default) is_uploaded is never true, so nothing is pruned and the
            # local run simply keeps every checkpoint -- which is the right default when the disk is
            # the only copy.
            for deleted in ckpt_lib.prune_checkpoints(
                checkpoint_dir, cfg.keep_local_checkpoints, hf.is_uploaded
            ):
                hf.delete(f"{phase}/{os.path.basename(deleted)}")
            # long-term retention for runs with uploads off: delete old rolling saves that are not
            # milestones, whatever the uploader says (keep_every_tokens 0 keeps today's behaviour)
            keep_every = int(getattr(cfg, "keep_every_tokens", 0) or 0)
            if keep_every:
                names = [os.path.join(checkpoint_dir, n) for n in os.listdir(checkpoint_dir)
                         if n.startswith(ckpt_lib.CKPT_PREFIX) and n.endswith(".pt")]
                for doomed in select_retention_deletions(names, cfg.keep_local_checkpoints, keep_every):
                    try:
                        os.remove(doomed)
                        logger.info(f"retention: deleted {os.path.basename(doomed)} (not a "
                                    f"{keep_every:,} token milestone, outside the newest "
                                    f"{cfg.keep_local_checkpoints})")
                    except OSError as e:
                        logger.warning(f"retention: could not delete {doomed}: {e}")
        # every rank, so the stop and resume decisions that read it stay identical across ranks
        steps_at_last_save = int(scheduler.last_epoch)
        if use_collectives:
            accelerator.wait_for_everyone()

    def grounded_str(stats):
        if "grounded_auroc" not in stats and "grounded_auroc_evidence" not in stats:
            return ""
        all_rows, evidence_rows = (
            f"{stats[key]:.4f}" if key in stats else "n/a"
            for key in ("grounded_auroc", "grounded_auroc_evidence")
        )
        return f"grounded AUROC all rows {all_rows} / evidence rows {evidence_rows}"

    def run_fixed_validation(epoch, step):
        """The fixed-target pass: the real answer forced under every condition. Returns
        ``(gold_gain, content_gain)`` in nats: CE(none) - CE(gold), and CE(distractors) - CE(gold),
        which is what the gold chunk's content is worth over a buffer of the same shape without it.
        Either is None when it could not be read."""
        stats = evaluate(unwrapped_model, fixed_dataset, device, tokenizer.pad_token_id,
                         cfg.fixed_eval_max_batches, conversation_weighting=False,
                         selector_by_loop=True, rank=rank, world_size=world_size,
                         gather=gather_eval)
        per_condition = (stats or {}).get("per_condition_ce") or {}
        if "none" not in per_condition or "gold" not in per_condition:
            log.warning(f"fixed-target pass read no gold/none tokens from {fixed_split}")
            return None, None
        # token-level mean over the answer tokens, the unit the in-context ceiling was measured in
        none_ce = per_condition["none"]
        gains = {c: none_ce - v for c, v in per_condition.items() if c != "none"}
        gold_gain = gains["gold"]
        content_gain = (
            per_condition["distractors"] - per_condition["gold"]
            if "distractors" in per_condition else None
        )
        parts = [
            f"[eval fixed] epoch {epoch} step {step} | answer CE: {{"
            + ", ".join(f"{c}: {v:.4f}" for c, v in per_condition.items()) + "}",
            "gain vs none: {" + ", ".join(f"{c}: {v:+.4f}" for c, v in gains.items()) + "}",
            "content gain (gold minus distractors) "
            + (f"{content_gain:+.4f}" if content_gain is not None else "n/a"),
            f"gold gain {gold_gain / EVIDENCE_CEILING_NATS:.1%} of the {EVIDENCE_CEILING_NATS:.2f}"
            f" nat SQuAD in-context ceiling (different rows)",
        ]
        if ceiling_all:
            parts.append(f"gold gain {gold_gain / ceiling_all:.1%} of the {ceiling_all:.2f} nat "
                         f"in-context ceiling (same split)")
        if grounded_str(stats):
            parts.append(grounded_str(stats))
        parts.append(f"{stats['tokens']:,} answer tokens over {stats['batches']} batches")
        log.info(" | ".join(parts))
        by_loop_ce = stats.get("per_condition_ce_by_loop") or {}
        selector = stats.get("selector_by_loop") or {}
        for loop_idx in sorted(set(by_loop_ce) | set(selector)):
            loop_ce = by_loop_ce.get(loop_idx)
            if loop_ce:
                line = (f"[eval fixed] loop {loop_idx + 1} | answer CE: {{"
                        + ", ".join(f"{c}: {v:.4f}" for c, v in loop_ce.items()) + "}")
                if "none" in loop_ce:
                    line += " | gain vs none: {" + ", ".join(
                        f"{c}: {loop_ce['none'] - v:+.4f}" for c, v in loop_ce.items()
                        if c != "none") + "}"
                log.info(line)
            reading = selector.get(loop_idx)
            if reading is None:
                continue
            fields = [
                "external mass {" + ", ".join(
                    f"{c}: {v:.3f}" for c, v in reading["mass_by_condition"].items()) + "}",
                "mass per chunk {" + ", ".join(
                    f"{c}: {v:.4f}" for c, v in reading["mass_per_chunk_by_condition"].items())
                + "}",
            ]
            for key, label in (("mass_auroc", "mass/chunk AUROC gold-present vs distractors"),
                               ("chunk_auroc", "chunk AUROC (per token, chance 0.5)"),
                               ("gold_share", "gold share")):
                if key in reading:
                    fields.append(f"{label} {reading[key]:.4f}")
            log.info(f"[eval fixed] loop {loop_idx + 1} | " + " | ".join(fields))
        # the pooled numbers above are what the decision reads; these only say which source moves them
        by_source = stats.get("per_source_condition_ce") or {}
        for name, source_ce in by_source.items():
            if "none" not in source_ce:
                continue
            source_gains = {c: source_ce["none"] - v for c, v in source_ce.items() if c != "none"}
            source_content = (source_ce["distractors"] - source_ce["gold"]
                              if "distractors" in source_ce and "gold" in source_ce else None)
            fields = [
                f"[eval fixed] source {name} | answer CE {{"
                + ", ".join(f"{c}: {v:.4f}" for c, v in source_ce.items()) + "}",
                "gain vs none {" + ", ".join(f"{c}: {v:+.4f}" for c, v in source_gains.items()) + "}",
                "content gain " + (f"{source_content:+.4f}" if source_content is not None else "n/a"),
            ]
            source_ceiling = ceiling_by_source.get(name)
            if source_ceiling and "gold" in source_gains:
                fields.append(f"gold gain {source_gains['gold'] / source_ceiling:.1%} of the "
                              f"{source_ceiling:.2f} nat in-context ceiling (same split)")
            fields.append(f"{stats['per_source_tokens'].get(name, 0):,} answer tokens")
            log.info(" | ".join(fields))
        return gold_gain, content_gain

    def run_validation(epoch, step):
        """The held-out objective pass, then the fixed-target pass. Returns
        ``(gold_gain, content_gain)``, both None without a fixed split."""
        stats = evaluate(unwrapped_model, val_dataset, device, tokenizer.pad_token_id,
                         cfg.eval_max_batches,
                         conversation_weighting=cfg.conversation_loss_weighting,
                         rank=rank, world_size=world_size, gather=gather_eval)
        if stats is None:
            log.warning(
                f"validation pass produced no supervised tokens -- is {cfg.val_split} empty?"
            )
        else:
            log_validation(epoch, step, stats)
        return run_fixed_validation(epoch, step) if fixed_dataset is not None else (None, None)

    def log_validation(epoch, step, stats):
        per_condition = stats.get("per_condition_ce")
        # each condition scored on its OWN target (an answer under gold, a refusal under none), so
        # the gap between them is not what the evidence is worth: that is the fixed-target line
        cond_str = (
            " | per-condition CE: {" + ", ".join(f"{c}: {v:.4f}" for c, v in per_condition.items())
            + "}" if per_condition else ""
        )
        sel_str = (
            f" | selection: {stats['selection']:.4f}" if "selection" in stats else ""
        )
        if "contrastive" in stats:
            sel_str += f" | contrastive: {stats['contrastive']:.4f}"
        if "grounded" in stats:
            sel_str += f" | grounded: {stats['grounded']:.4f}"
            if grounded_str(stats):
                sel_str += f" ({grounded_str(stats)})"
        log.info(
            f"[eval] epoch {epoch} step {step} | CE: {stats['ce']:.4f} | ppl: {stats['ppl']:.3f} | "
            f"p_max: {stats['p_max']:.4f} | top1_acc: {stats['top1_acc']:.4f} | "
            f"{stats['tokens']:,} supervised tokens over {stats['batches']} batches"
            f"{cond_str}{sel_str}"
        )

    sft_tokens = token_count - start_token_count
    # on the fixed grid, so a stop and relaunch does not shift every later save and eval
    next_checkpoint = next_grid_point(sft_tokens, cfg.checkpoint_every_tokens)
    next_eval = next_grid_point(sft_tokens, cfg.eval_every_tokens)
    # a stop polled within one log interval of the last save exits on that save
    stop_slack_steps = math.ceil(LOG_INTERVAL / max(cfg.grad_accumulation_steps, 1))
    # the IR and evidence profiles' extra schedule (the temperature anneal is --ir only, the refresh
    # is whichever profile's config sets cluster_refresh_tokens). Both are driven from the log block,
    # which already syncs the token counter -- neither adds a host sync of its own, and LOG_INTERVAL
    # is ~160k tokens here, far finer than either cadence needs.
    total_micro_steps = max(1, total_steps * cfg.grad_accumulation_steps)
    next_refresh = sft_tokens + getattr(cfg, "cluster_refresh_tokens", 0)
    ir_refresh_stats = []
    # the evidence stream's own tally, kept separate from _token_tracker's prompt-token count (which
    # anchors the LR schedule / token target / checkpoint cadence and must not change meaning). Same
    # on-device-accumulate-then-drain-at-log-cadence shape as TokenTracker, for the same reason: this
    # runs in the per-micro-step path and must not force a sync there.
    evidence_token_count, evidence_token_pending = 0, None
    # bound before the try: the interrupt handler saves a checkpoint using both, and a Ctrl-C
    # during the very first batch must not turn into a NameError that loses the save
    step, epoch = step_offset, start_epoch
    stop_training, exit_code = False, EXIT_OK
    kill_tokens = getattr(cfg, "kill_tokens", 0) if fixed_dataset is not None else 0
    kill_checked = resumed_kill_checked
    # the selector's own tensors only: the IR experts' down_proj shares its name with every decoder
    # and MLP expert projection
    ir_prefixes = tuple(
        f"moe.experts.{i}." for i, expert in enumerate(unwrapped_model.moe.experts)
        if hasattr(expert, "ir_module")
    )
    grad_probe_names = [
        name for name, p in unwrapped_model.named_parameters()
        if p.requires_grad and name.startswith(ir_prefixes) and name.endswith(GRAD_PROBE_SUFFIXES)
    ] if args.evidence and ir_prefixes else []

    def probe_gradients(optimizer_step):
        def probe():
            params = dict(unwrapped_model.named_parameters())
            norms = []
            for name in grad_probe_names:
                grad = params[name].grad
                norms.append(f"{_fresh_family(name)}: "
                             + (f"{grad.detach().float().norm().item():.3e}" if grad is not None
                                else "None"))
            log.info(f"grad norms at optimizer step {optimizer_step} (pre-clip): "
                        + ", ".join(norms))
        return probe

    if keep_stop:
        stop_at_start = torch.tensor([1.0 if (is_main and os.path.exists(control.sentinel_path))
                                      else 0.0], device=device)
        if world_size > 1:
            torch.distributed.all_reduce(stop_at_start)
        if stop_at_start.item() > 0:
            log.info(f"--keep-stop: a STOP sentinel is present at {control.sentinel_path}; exiting "
                     f"{EXIT_USER_STOP} before the first step, nothing saved, the sentinel kept")
            hf.drain(timeout=600)
            hf.close()
            return EXIT_USER_STOP

    if not resumed:
        # the baseline every later reading is a difference from; the seed's reader is dead, so the
        # fixed-target gain should read about zero here
        run_validation(start_epoch, step_offset)

    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            resume_epoch = resumed and epoch == start_epoch
            # the LM reader has no per-epoch permutation to set: it reads in on-disk order, which
            # is where the corpus builder already put the source mix
            select_epoch(dataloader, train_dataset, epoch)
            train_dataset.start_doc_idx = start_doc_idx if resume_epoch else 0
            last_merged_state = None
            if not resume_epoch:
                worker_state.fill_(-1)
                step = 0

            batches = dataloader
            if world_size > 1:
                def min_over_ranks(n):
                    t = torch.tensor([n], dtype=torch.long, device=device)
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
                    return int(t.item())
                # every rank takes the same number of micro steps, or a collective hangs
                batches = lockstep_batches(dataloader, LOG_INTERVAL, min_over_ranks)

            for local_step, batch in enumerate(batches):
                step = local_step + (step_offset if resume_epoch else 0)
                worker_state[batch["worker_id"][0].to(worker_state.device)] = (
                    batch["doc_idx"][0].to(worker_state.device)
                )

                input_ids = batch["input_ids"].to(device)
                document_ids = batch["document_ids"].to(device)
                cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
                pad_mask = input_ids == tokenizer.pad_token_id

                if "evidence_chunk_ids" in batch:
                    # a real evidence token has a chunk id >= 0; the batch's padding out to its
                    # widest row does not, which mirrors _token_tracker's own non-pad rule without a
                    # second definition of "padding" -- kept on-device and only summed into the
                    # pending scalar, never .item()-ed here
                    ev_count = (batch["evidence_chunk_ids"].to(device) >= 0).sum()
                    if evidence_token_pending is None or evidence_token_pending.device != ev_count.device:
                        evidence_token_pending = torch.zeros((), dtype=torch.long, device=ev_count.device)
                    evidence_token_pending += ev_count

                # log steps pinned to full depth so the recorded loss curve is always read at one
                # operating point (same reasoning as pretrain.py)
                # and on a micro step that completes an accumulation window (accelerate's counter
                # restarts at zero in every process, so the step index alone cannot say): a stop or
                # a save then never lands with half a window's rows consumed and never stepped.
                # At accumulation 1 this is exactly step % LOG_INTERVAL == 0
                ends_window = (accelerator.step + 1) % max(cfg.grad_accumulation_steps, 1) == 0
                is_log_step = ends_window and step % LOG_INTERVAL < max(cfg.grad_accumulation_steps, 1)
                step_n_loops = full_n_loops if is_log_step else sample_n_loops(
                    loop_rng, full_n_loops, TrainingConfig.loop_count_sampling
                )

                loss, loss_ce, aux_loss, metrics = train_step(
                    model,
                    input_ids,
                    cu_seqlens,
                    max_seqlen,
                    batch["labels"].to(device),
                    pad_mask,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    # every parameter is fp32-shadowed here, not just the undecayed ones
                    no_decay_master_pairs=master_pairs,
                    collect_metrics=is_log_step,
                    n_loops=step_n_loops,
                    # per-conversation weighting, off for plain SFT. The tensor is in every batch;
                    # not passing it is exactly the plain per-token objective.
                    loss_weights=(batch["loss_weights"].to(device)
                                  if cfg.conversation_loss_weighting else None),
                    # built here rather than in the worker: numbering the evidence segments needs
                    # the query side's cu_seqlens, which is itself built in-thread because it is
                    # ragged and accelerate truncates a ragged dim 0 to the batch size. Returns None
                    # for a batch that retrieved nothing, which is the bit-identical forward.
                    evidence=evidence_from_batch(
                        accelerator.unwrap_model(model), batch, cu_seqlens
                    ),
                    # keep the load balancing loss off the padding. It matters more here than in
                    # pretraining: a conversation is never split across rows and an evidence row can
                    # close on its evidence budget with the prompt axis half empty, so row fill
                    # varies batch to batch and the aux loss moves with it rather than with routing.
                    token_mask=~pad_mask,
                    # the groundedness label's second axis; absent from every corpus built before
                    # the .ans sidecar, which drops the term rather than guessing it
                    answerable=(batch["answerable_ids"].to(device)
                                if "answerable_ids" in batch else None),
                    grad_probe=(
                        probe_gradients((step + 1) // cfg.grad_accumulation_steps)
                        if grad_probe_names and (step + 1) % cfg.grad_accumulation_steps == 0
                        and (step + 1) // cfg.grad_accumulation_steps in GRAD_PROBE_STEPS
                        else None
                    ),
                )

                if not is_log_step:
                    continue

                # everything below pulls to the host; throttled to LOG_INTERVAL like pretrain.py
                if world_size == 1:
                    reduced = None
                    val_loss = loss.item()
                    losses.append(val_loss)
                    token_count = unwrapped_model._token_tracker.sync()
                    if evidence_token_pending is not None:
                        evidence_token_count += int(evidence_token_pending.item())
                        evidence_token_pending.zero_()
                else:
                    # the one reduction of the log step: token and evidence deltas, any rank's stop
                    # and checkpoint request, the means of the logged scalars. Every decision below
                    # reads this vector's sum, so every rank takes the same branch
                    local_total = unwrapped_model._token_tracker.sync()
                    local_evidence = 0
                    if evidence_token_pending is not None:
                        local_evidence = int(evidence_token_pending.item())
                        evidence_token_pending.zero_()
                    if is_main:
                        control.poll()
                    counts = control_counts(control, include_user_stop=is_main)
                    counts.update(tokens=local_total - last_local_total, evidence=local_evidence)
                    last_local_total = local_total
                    vec = pack_log_vector(
                        counts,
                        {"loss": loss, "ce": loss_ce, "aux": aux_loss,
                         "p_max": metrics.get("p_max"), "top1": metrics.get("top1_acc"),
                         "selection": metrics.get("selection_loss"),
                         "contrastive": metrics.get("contrastive_loss"),
                         "grounded": metrics.get("groundedness_loss")},
                        metrics["per_loop_ce"], device,
                    )
                    torch.distributed.all_reduce(vec)
                    last_merged_state = worker_state.clone()
                    torch.distributed.all_reduce(last_merged_state,
                                                 op=torch.distributed.ReduceOp.MAX)
                    reduced = unpack_log_vector(vec.tolist(), world_size)
                    val_loss = reduced["scalars"]["loss"]
                    losses.append(val_loss)
                    token_count += int(round(reduced["counts"]["tokens"]))
                    evidence_token_count += int(round(reduced["counts"]["evidence"]))
                sft_tokens = token_count - start_token_count
                now = time.time()
                interval_s = max(now - last_log_time, 1e-6)
                tokens_per_sec = (token_count - last_token_count) / interval_s
                last_token_count, last_log_time = token_count, now

                if target_tokens is None and step > 0:
                    # tokens/step is only knowable once training has run: packing density depends
                    # on the corpus, not the config. Anchors the ETA, never the LR schedule.
                    target_tokens = int(sft_tokens / max(step, 1) * total_steps
                                        * cfg.grad_accumulation_steps)

                if args.ir:
                    # anneal the retrieval temperature on MICRO-step progress, not on tokens: the
                    # token target is only estimable once training has run, and the anneal has to
                    # be a known function of position from step 0 to be reproducible. Evidence trains
                    # the table at the trunk's rate rather than annealing it -- the values were never
                    # rebuilt, so there is no near-uniform read to sharpen out of.
                    scale = cfg.temperature_scale(step / total_micro_steps)
                    unwrapped_model.moe.set_ir_temperature_scale(scale)

                # gated on the CONFIG carrying a refresh cadence, not on --ir: any profile that
                # trains the IR keys needs its centroids to keep tracking them, and the evidence
                # profile now does (at the trunk's rate, not the fresh one, but "slow" is not
                # "frozen") -- SFTConfig/RepairConfig have no cluster_refresh_tokens, so this is a
                # no-op there exactly as before.
                refresh_tokens = getattr(cfg, "cluster_refresh_tokens", 0)
                if refresh_tokens and sft_tokens >= next_refresh:
                    ir_refresh_stats = apply_ir_refresh(
                        unwrapped_model, optimizer, master_pairs,
                        unwrapped_model.moe.refresh_ir_clusters(
                            dead_quantile=getattr(cfg, "dead_quantile", 0.0)
                        ),
                    )
                    next_refresh = sft_tokens + refresh_tokens
                    if world_size > 1:
                        # each rank clustered and recycled from its own reservoir; rank 0's result wins
                        sync_ir_from_main(
                            unwrapped_model.moe, optimizer, master_pairs,
                            broadcast_from_main,
                        )
                    # logged as its own line so a loss step at a refresh boundary is attributable to
                    # the refresh rather than to the data; candidate recall (the number that says
                    # whether the partition is still covering what the read wants) is already inside
                    # ir_refresh_stats, reported identically regardless of which profile triggered it
                    log.info(
                        f"IR cluster refresh at {sft_tokens / 1e6:.1f}M {phase} tokens"
                        + (f" (temperature scale {scale:.4f})" if args.ir else "")
                        + f": {ir_refresh_stats}"
                    )

                per_loop_ce = ", ".join(
                    f"{v:.4f}" for v in (reduced["per_loop"] if reduced is not None
                                         else [ce.item() for ce in metrics["per_loop_ce"]])
                )
                loop_scale = ", ".join(f"{s:.4f}" for s in unwrapped_model.moe.loop_scale.tolist())
                # per loop IR retrieval entropy over ln(num_ir_entries), same field pretrain.py logs
                ir_entropy = (
                    unwrapped_model.moe.ir_tracker.get_stats()
                    if unwrapped_model.moe.ir_tracker is not None else []
                )
                ir_entropy_str = (
                    f"IR E/ln{unwrapped_model.moe.ir_tracker.num_entries}"
                    f"{' (rank 0 slice)' if world_size > 1 else ''}: ["
                    + ", ".join(f"{e:.4f}" for e in ir_entropy) + "] | "
                    if ir_entropy else ""
                )
                # per pass, the most loaded MLP expert over the mean (1.0 even); counted in
                # every balancing mode, the same field pretrain.py logs
                balance, dense_share = unwrapped_model.moe.balance_load_stats()
                if balance:
                    ir_entropy_str += (
                        "load max/mean: [" + ", ".join(f"{r:.2f}" for r in balance) + "] | "
                        + "dense share: [" + ", ".join(f"{d:.3f}" for d in dense_share) + "] | "
                    )
                if args.ir:
                    ir_module = unwrapped_model.moe.ir_modules[0]
                    # g_proj carries the migration's neutrality zero, so it is the tensor the whole
                    # read is waiting on: dL/dy_values flows through it and is exactly zero until it
                    # moves. Logged against the value rows' own norm so a run that sharpens without
                    # ever leaving zero is visible in the first few log lines rather than at the
                    # ablation. Both are host syncs, hence the log cadence.
                    g_rms = ir_module.g_proj.weight.detach().float().pow(2).mean().sqrt().item()
                    y_norm = ir_module.y_values.detach().float().norm(dim=-1).mean().item()
                    ir_entropy_str += (
                        f"IR temp: {ir_module.temperature.detach().item():.4f} | "
                        f"|g_proj|rms: {g_rms:.2e} | |y|row: {y_norm:.4f} | "
                    )
                if unwrapped_model.moe.inject is not None:
                    # same reason as |g_proj|rms above: the injection is migrated in at zero, so
                    # "did it leave zero" is a fact about the run that has to be readable while the
                    # run is still going, not reconstructed from the final checkpoint
                    inj_rms = unwrapped_model.moe.inject.weight.detach().float().pow(2).mean().sqrt().item()
                    ir_entropy_str += f"|inject|rms: {inj_rms:.2e} | "
                if args.evidence:
                    # the supervised selection term itself (NaN on a step whose batch retrieved
                    # nothing, or on a corpus with no gold flag -- both are "no label", not zero
                    # loss). Falling and the external mass separating by condition is the pair that
                    # says the selector is ranking; either alone can move for the other's reason.
                    sel = metrics.get("selection_loss")
                    if sel is not None:
                        ir_entropy_str += (f"selection: {reduced['scalars']['selection']:.4f} | "
                                           if reduced is not None else f"selection: {sel.item():.4f} | ")
                    # the retriever's term; NaN on a run that does not carry it
                    con = metrics.get("contrastive_loss")
                    if con is not None and TrainingConfig.evidence_contrastive_weight > 0.0:
                        ir_entropy_str += (f"contrastive: {reduced['scalars']['contrastive']:.4f} | "
                                           if reduced is not None else f"contrastive: {con.item():.4f} | ")
                    # the other supervised head on the port. NaN whenever the checkpoint carries no
                    # head or the corpus no .ans, which is a missing label rather than a zero loss
                    grounded = metrics.get("groundedness_loss")
                    if grounded is not None:
                        ir_entropy_str += (f"grounded: {reduced['scalars']['grounded']:.4f} | "
                                           if reduced is not None else f"grounded: {grounded.item():.4f} | ")
                    evidence_module = unwrapped_model.moe.shared_evidence
                    if unwrapped_model.moe.reader_kv:
                        # no zero-init tensor to watch in this mode: the read shares shared_attn's
                        # softmax from step 0. The per loop value gains are what it owns
                        gains = unwrapped_model.moe.evidence_loop_scale.detach().float().tolist()
                        ir_entropy_str += "evidence gain: [" + ", ".join(f"{g:.3f}" for g in gains) + "] | "
                    if evidence_module is not None:
                        # the reader's own neutrality zero, same reason as |g_proj|rms above: if
                        # this never leaves zero the port never wrote anything to the residual
                        # whatever the selector's mass split says
                        o_rms = (
                            evidence_module.attn.o_proj.weight.detach().float().pow(2).mean()
                            .sqrt().item()
                        )
                        ir_entropy_str += f"|shared_evidence.o_proj|rms: {o_rms:.2e} | "
                    # the G3b signal, bucketed by the condition that produced the query: how much of
                    # the read mass the external store won, on this step's (full-depth, since this
                    # is a log step) forward. Read from the IR module's own instrumentation rather
                    # than recomputed -- last_memory_mass is None whenever this step's batch carried
                    # no evidence at all, or the module has no external store attached, so both are
                    # guarded with getattr/None checks rather than assumed present.
                    ir_modules = unwrapped_model.moe.ir_modules
                    mass = getattr(ir_modules[0], "last_memory_mass", None) if ir_modules else None
                    if mass is not None and "condition_ids" in batch:
                        cond_ids_flat = batch["condition_ids"].to(mass.device).reshape(-1)
                        mass_by_cond = [
                            f"{cond}: {mass[cond_ids_flat == idx].mean().item():.3f}"
                            for idx, cond in enumerate(CONDITIONS)
                            if bool((cond_ids_flat == idx).any())
                        ]
                        if mass_by_cond:
                            ir_entropy_str += (
                                f"external mass{' (rank 0 slice)' if world_size > 1 else ''}: "
                                f"{{{', '.join(mass_by_cond)}}} | "
                            )

                def _metric(key):
                    if reduced is not None:
                        return reduced["scalars"][{"top1_acc": "top1"}.get(key, key)]
                    value = metrics.get(key)
                    return value.item() if value is not None else float("nan")

                ce_val = reduced["scalars"]["ce"] if reduced is not None else loss_ce.item()
                aux_val = reduced["scalars"]["aux"] if reduced is not None else aux_loss.item()

                eta = eta_seconds(sft_tokens, target_tokens or 0, tokens_per_sec) if target_tokens else None
                # a separate field from "{phase} tokens", never folded into it: that counter anchors
                # the LR schedule, the token target and the checkpoint cadence, and the evidence
                # stream (~3x its size on the smoke corpus) is not part of what any of those read.
                # Without its own field the log makes an evidence-conditioned step look as cheap as
                # a plain SFT one at the same "{phase} tokens" reading.
                evidence_field = (
                    f" | evidence stream: {evidence_token_count / 1e6:.2f}M" if args.evidence else ""
                )
                log.info(
                    f"Epoch {epoch} | Step {step} | Loss: {val_loss:.4f} | Loss (CE): {ce_val:.4f} | "
                    f"Aux: {aux_val:.4f} | loop_scale: [{loop_scale}] | "
                    f"p_max: {_metric('p_max'):.4f} | top1_acc: {_metric('top1_acc'):.4f} | "
                    f"per-loop CE: [{per_loop_ce}] | {ir_entropy_str}"
                    f"LR: {scheduler.get_last_lr()[0]:.3e} | {phase} tokens: {sft_tokens / 1e6:.2f}M"
                    f"{evidence_field} | "
                    f"Tokens/sec: {tokens_per_sec:.0f} | "
                    f"Peak Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | "
                    f"Time: {(now - timer) / 60:.2f} min"
                    + (f" | ETA: {format_duration(eta)}" if eta is not None else "")
                )

                if is_main:
                    write_status(
                        status_path, phase=phase, tokens=sft_tokens,
                        phase_target=target_tokens or 0, run_target=target_tokens or 0,
                        tokens_per_sec=tokens_per_sec, loss=val_loss,
                        eta_phase=format_duration(eta) if eta is not None else "n/a",
                        eta_run=format_duration(eta) if eta is not None else "n/a",
                        step=step, epoch=epoch,
                    )

                if sft_tokens >= next_eval:
                    gold_gain, content_gain = run_validation(epoch, step)
                    # on the grid, not one cadence past a log step that overshot it
                    next_eval = next_grid_point(sft_tokens, cfg.eval_every_tokens)
                    if kill_tokens and not kill_checked and sft_tokens >= kill_tokens:
                        min_gain = cfg.kill_min_gain
                        if gold_gain is None or content_gain is None:
                            # left armed: a pass that read nothing is not a decision
                            log.info(
                                "kill check skipped: the fixed-target pass read no gold, none or "
                                "distractors tokens; it runs again at the next eval"
                            )
                        else:
                            # one decision point, not a running threshold: a run that clears it
                            # keeps going and is judged on the pass bar from then on
                            kill_checked = True
                            readings = (f"gold gain {gold_gain:+.4f} nats, content gain "
                                        f"{content_gain:+.4f} nats")
                            if gold_gain < min_gain or content_gain < min_gain:
                                log.warning(
                                    f"KILL: fixed-target {readings}; one is under {min_gain} at "
                                    f"{sft_tokens / 1e6:.2f}M {phase} tokens. Saving and stopping "
                                    f"(exit {EXIT_USER_STOP}, not restarted by a wrapper)."
                                )
                                save_and_sync(epoch, step, val_loss, token_count)
                                exit_code = EXIT_USER_STOP
                                stop_training = True
                                break
                            log.info(
                                f"kill check passed at {sft_tokens / 1e6:.2f}M {phase} tokens: "
                                f"{readings}, both >= {min_gain}"
                            )

                # polled at the log cadence: a stat every few seconds, no GPU sync, and well
                # inside vast's SIGTERM grace period
                if reduced is None:
                    control.poll()
                    stop_now, stop_code, stop_reason = (control.stop_requested, control.exit_code,
                                                        control.reason)
                    checkpoint_request = None
                else:
                    # the requests of every rank, summed at the top of this log step (rank 0's STOP
                    # file and any rank's SIGTERM or SIGUSR1): the same answer on every rank
                    stop_code = stop_exit_code(reduced["counts"])
                    stop_now = stop_code is not None
                    stop_reason = control.reason or "a stop request on another rank"
                    checkpoint_request = reduced["counts"]["checkpoint"] > 0
                if stop_now:
                    if stop_needs_save(int(scheduler.last_epoch), steps_at_last_save,
                                       stop_slack_steps):
                        log.info(f"Stopping: {stop_reason}. Saving checkpoint...")
                        save_and_sync(epoch, step, val_loss, token_count)
                    else:
                        log.info(
                            f"Stopping: {stop_reason}. The last checkpoint is "
                            f"{int(scheduler.last_epoch) - steps_at_last_save} optimizer step(s) "
                            f"back, within one log interval: exiting on it without another save"
                        )
                    exit_code = stop_code
                    stop_training = True
                    break

                if sft_tokens >= next_checkpoint or (
                    checkpoint_request if reduced is not None else control.take_checkpoint_request()
                ):
                    save_and_sync(epoch, step, val_loss, token_count)
                    # the next grid point past where this save landed (a SIGUSR1 save included),
                    # not one cadence past it: a coarse log step must not walk the grid
                    next_checkpoint = next_grid_point(sft_tokens, cfg.checkpoint_every_tokens)

            if stop_training:
                break

            if max_epochs_this_split and epoch + 1 >= max_epochs_this_split:
                # a warm split never wraps into its next epoch: the save keeps this epoch and the
                # position at the end of it, which is what the switch to the main split continues
                # from (check_switch_state refuses a saved epoch past 0)
                token_count = synced_token_count(token_count, last_local_total)
                log.info(f"warm split exhausted: epoch {epoch} of {cfg.train_split!r} finished at "
                         f"{(token_count - start_token_count) / 1e6:.2f}M {phase} tokens; saving and "
                         f"exiting {EXIT_USER_STOP} before epoch {epoch + 1}")
                save_and_sync(epoch, step, losses[-1] if losses else float("nan"), token_count)
                return EXIT_USER_STOP

            # end of epoch: the next one starts its own permutation from position 0
            start_doc_idx, step_offset = 0, 0
            resumed = False
            worker_state.fill_(-1)
            log.info(f"Epoch {epoch} finished at {sft_tokens / 1e6:.2f}M {phase} tokens")

        if stop_training:
            # a stop is not a finished run: no final checkpoint, and a restartable exit code so a
            # wrapper knows to relaunch (the rolling checkpoint just written is the resume point)
            return exit_code

        if world_size == 1:
            token_count = unwrapped_model._token_tracker.sync()
        else:
            # the ranks left their last chunk together, so this reduction is matched on every rank
            final_delta = torch.tensor(
                [float(unwrapped_model._token_tracker.sync() - last_local_total)],
                dtype=torch.float64, device=device)
            torch.distributed.all_reduce(final_delta)
            token_count += int(round(final_delta.item()))
        run_validation(cfg.num_epochs - 1, step)
        save_and_sync(cfg.num_epochs - 1, step, losses[-1] if losses else float("nan"),
                      token_count, final=True)
        log.info(
            f"{phase} complete: {(token_count - start_token_count) / 1e6:.2f}M tokens over "
            f"{cfg.num_epochs} epochs in {(time.time() - timer) / 60:.1f} min"
        )
    except KeyboardInterrupt:
        log.info("Interrupted. Saving checkpoint...")
        exit_code = EXIT_USER_STOP
        try:
            # no collectives here: only the interrupted process may still be alive
            save_and_sync(epoch, step, losses[-1] if losses else float("nan"),
                          unwrapped_model._token_tracker.sync() if world_size == 1 else token_count,
                          collective=False)
        except Exception as e:
            log.error(f"Failed to save the interrupt checkpoint: {e}")
    finally:
        # never let a stop race the uploader: drain before the process goes away
        hf.drain(timeout=600)
        hf.close()

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="supervised fine-tuning (and Phase 2's repair pass)")
    parser.add_argument("--repair", action="store_true",
                        help="run NEXT.md Phase 2's abstention repair finetune instead: "
                             "config.yaml's repair: block, the repair_train/repair_val splits, and "
                             "ckpts/repair. Seed it with -c <an SFT checkpoint>")
    parser.add_argument("--evidence", action="store_true",
                        help="the evidence conditioned finetune: reads EvidenceConfig, writes into "
                             "ckpts/evidence under phase 'evidence'. Needs a corpus from "
                             "scripts/prepare_evidence_data.py and a seed checkpoint carrying the "
                             "port (scripts/migrate_evidence_port.py)")
    parser.add_argument("--ir", action="store_true",
                        help="run the IR table's sharpening finetune instead: config.yaml's ir: "
                             "block, the ir_train/ir_val splits, ckpts/ir, a second learning rate "
                             "for the rebuilt table and a retrieval temperature anneal. Seed it "
                             "with -c <a scripts/migrate_ir_reshape.py output>")
    parser.add_argument("--reader-no-rotary", action="store_true",
                        help="--evidence only: build the evidence reader without rotary on its "
                             "query and keys (the encoder already positions chunk tokens). The "
                             "mode is saved in the run's checkpoints, so pass it on every launch of "
                             "that run, and give the run its own --run-name")
    parser.add_argument("--reader-kv", action="store_true",
                        help="--evidence only: read the evidence as leading keys and values of the "
                             "shared self attention in every loop instead of through the separate "
                             "cross attention reader. A cross reader seed loads with its reader "
                             "tensors dropped. The mode is saved in the run's checkpoints, so pass "
                             "it on every launch of that run, and give the run its own --run-name")
    parser.add_argument("--run-name", default=None,
                        help="suffix the profile's checkpoint directory, e.g. --run-name random "
                             "writes ckpts/ir_random. Required to run two seeds of one profile "
                             "against each other: a shared directory means the second run resumes "
                             "the first instead of starting from its own seed")
    parser.add_argument("--data-dir", default=None,
                        help="override the profile's data_dir (relative paths resolve against the "
                             "repo root). Saved in the checkpoint and checked on resume: pass it "
                             "on every launch")
    parser.add_argument("--train-split", default=None,
                        help="override the profile's train split. Saved in the checkpoint and "
                             "checked on resume: pass it on every launch")
    parser.add_argument("--val-split", default=None,
                        help="override the profile's validation split. Saved in the checkpoint "
                             "and checked on resume: pass it on every launch")
    parser.add_argument("--switch-split", action="store_true",
                        help="allow a resume on other data than the checkpoint recorded (data_dir, "
                             "train or val split), e.g. the switch from the warm-up split: "
                             "SPLIT=pilot_main_train bash ckpts/inject/launch_pilot_X.sh with "
                             "--switch-split. Without it such a resume is refused; pass it only on "
                             "the one launch that switches")
    parser.add_argument("--plan-split", default=None,
                        help="fit the packed row estimate, total_steps and so the LR schedule to "
                             "this split of the same data dir instead of the training split. The "
                             "warm-up to main run passes the main split on every launch, so the "
                             "schedule spans the whole run from step 0 and the switch changes "
                             "nothing in the plan. Saved in the checkpoint")
    parser.add_argument("--keep-stop", action="store_true",
                        help="honour a STOP sentinel already present at startup: after the setup "
                             "every rank exits 10 before the first step, saving nothing and leaving "
                             "the file. Default off: a launch deletes a leftover STOP")
    parser.add_argument("--estimate-rows-only", action="store_true",
                        help="CPU only: compute the plan split's packed row estimate for each rank "
                             "count in --estimate-world-sizes, write {data_dir}/{split}."
                             "rows_estimate.json and exit 0 without building the model. A launch "
                             "of more than one rank refuses to start without a valid cache")
    parser.add_argument("--estimate-world-sizes", default="8,16",
                        help="with --estimate-rows-only: comma separated rank counts to cache the "
                             "estimate for (it depends on NUM_DATA_WORKERS x ranks)")
    parser.add_argument("--max-epochs-this-split", type=int, default=0,
                        help="stop at the end of epoch N - 1 of the training split: save at that "
                             "epoch, log 'warm split exhausted' and exit 10 instead of entering "
                             "epoch N. The warm launch passes 1; 0 (default) is off")
    parser.add_argument("--grad-accumulation-steps", type=int, default=None,
                        help="override the profile's grad_accumulation_steps (keep the global batch "
                             "fixed: per GPU batch x accumulation x GPUs). total_steps and the "
                             "cosine follow it; a resume under another value warns that the plan "
                             "moved")
    parser.add_argument("--ceiling-json", default=None,
                        help="--evidence only: the in-context ceiling file written by "
                             "scripts/evidence_ceiling_probe.py --fixed-split; each source's gold "
                             "gain in [eval fixed] is then also printed as a share of its own "
                             "ceiling on the same split")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="checkpoint to initialize from (ignored when resuming a run from this "
                             "profile's own checkpoint directory)")
    parser.add_argument("--from-hub", action="store_true",
                        help="download the pretrained checkpoint and manifest.json from the "
                             "training mirror repo before starting")
    parser.add_argument("--hub-repo", default=None,
                        help="repo to pull from (default: config.yaml's training.hf_upload_repo)")
    parser.add_argument("--hub-file", default=DEFAULT_HUB_CHECKPOINT,
                        help=f"path within the repo (default: {DEFAULT_HUB_CHECKPOINT})")
    parser.add_argument("--upload-repo", default=None,
                        help="mirror checkpoints to this repo ('' disables; default: config.yaml's "
                             "sft.hf_upload_repo / repair.hf_upload_repo)")
    return sft(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

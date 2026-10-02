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
"""
import os
import re
import sys
import json
import time
import math
import random
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import torch
from torch import optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

import transformer_engine.pytorch as te

from modules.data.dataset import Dataset
from modules.data.evidence_dataset import EvidenceDataset, evidence_from_batch
from modules.data.sft_dataset import SFTDataset
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.information_retrieval import is_rebuilt_ir_param
from modules.model.moe import is_fresh_loop_param
from modules.model.mtp import compute_mtp_loss
from modules.model.transformer import TinyMoETransformer
from modules.runtime import checkpoints as ckpt_lib
from modules.runtime.control import EXIT_OK, EXIT_USER_STOP, RunControl
from modules.runtime.hf_sync import HFSync
from modules.runtime.status import eta_seconds, format_duration, write_status
from config import EvidenceConfig, IRConfig, ModelConfig, RepairConfig, SFTConfig, TrainingConfig
from scripts.pretrain import (
    USE_LOW_PRECISION, answer_start_positions, chosen_recipe, log_precision_mode,
    predicting_positions, sample_n_loops, save_expert_selection_graph, save_loss_graph, train_step,
)
# the same ranking metric eval_abstention.py reports G3b with, imported rather than restated so the
# validation pass and the acceptance script cannot disagree about what AUROC means here
from scripts.eval_calibration import roc_auc
# imported rather than restated, so the condition index -> name mapping used for validation and
# training logs can never drift from what the corpus builder actually wrote into `.cond`
from scripts.prepare_evidence_data import CONDITIONS, SOURCE_KEYS
from utils import (BASE_DIR, BF16, HF_UPLOAD_REPO, TOKENIZER_DIR, get_hf_token, load_model_state,
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


def make_dataset(data_dir: str, split: str, tokenizer, cfg, shuffle: bool = True):
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
        )
    if os.path.isfile(os.path.join(data_dir, f"{split}.mask")):
        return SFTDataset(
            data_dir=data_dir, tokenizer=tokenizer, batch_size=cfg.Batch_size,
            max_length=cfg.Seq_length, split=split,
            num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
            seed=cfg.seed, shuffle=shuffle,
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
    a halved SFT batch has to spare. Note the masters are NOT checkpointed: a resume reseeds them
    from the bf16 weights, so sub-ulp progress accumulated since the last save is discarded. That
    is inherent -- the saved model is bf16 either way -- and bounded by one checkpoint interval.

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
        max_length: row length.
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
    lengths = np.diff(offsets).astype(np.int64) + num_mtp_tokens
    if split_documents:
        return max(1, int(lengths.sum() // max_length) + 1)

    ev_lengths = np.zeros_like(lengths)
    if evidx_path and max_evidence_tokens:
        ev_offsets = np.fromfile(evidx_path, dtype=np.uint64)
        ev_lengths = np.diff(ev_offsets).astype(np.int64)

    # the reader's own stream order, when it has one. Only the two budget case actually needs this
    # (see the Args note), but running it for every profile keeps one code path
    if order is not None:
        lengths, ev_lengths = lengths[order], ev_lengths[order]

    keep = lengths <= max_length
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
            if open_row and (used + length > max_length
                             or (max_evidence_tokens and used_ev + ev > max_evidence_tokens)):
                rows += 1
                used, used_ev = 0, 0
            used, used_ev, open_row = used + length, used_ev + ev, True
        rows += int(open_row)
    return max(1, rows)


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


def save_sft_checkpoint(model, optimizer, scheduler, path, *, epoch, step, token_count,
                        start_token_count, global_offset, losses, seed, phase=SFT_PHASE,
                        kill_checked=False):
    """Write an SFT checkpoint atomically.

    Deliberately its own function rather than an extension of ``utils.save_checkpoint``: SFT needs
    two fields pretraining has no concept of (``start_token_count``, so SFT progress can be
    recovered from the continued global counter, and ``seed``, which selects the document
    permutation a ``global_offset`` indexes into), and the pretraining run is *live on rented
    hardware right now* -- changing ``utils.load_checkpoint``'s tuple arity would break the running
    job on its next preemption relaunch, since onstart.sh re-clones the branch.

    The payload is a strict **superset** of what ``utils.load_checkpoint`` expects, so
    ``scripts/inference.py`` and ``scripts/eval_calibration.py`` read an SFT checkpoint unchanged.
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
                "kill_checked": kill_checked},
    }
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
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    sft_extra = checkpoint.get("sft", {}) or {}
    logger.info(f"Checkpoint loaded from {path}")
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("dataset_idx", 0),
        "token_count": checkpoint.get("token_count", 0),
        "start_token_count": sft_extra.get("start_token_count", checkpoint.get("token_count", 0)),
        "global_offset": checkpoint.get("global_offset", 0),
        "losses": checkpoint.get("losses", None) or [],
        "seed": sft_extra.get("seed", SFTConfig.seed),
        "kill_checked": sft_extra.get("kill_checked", False),
    }


def load_pretrained_weights(model, path: str):
    """Seed SFT from a pretraining checkpoint: weights and bookkeeping, no optimizer state.

    The optimizer is deliberately *not* restored. Pretraining's AdamW moments were accumulated at
    lr=4e-4 against a different objective; carrying them into a 3e-5 fine-tune would spend the
    first few hundred steps unwinding momentum that no longer describes the loss surface.

    Returns:
        The pretraining token count, carried forward on purpose -- see this module's docstring.
    """
    checkpoint = torch.load(path, map_location="cpu")
    # through load_model_state, not load_state_dict: a seed built before the retrieval temperature
    # became learned carries neither temperature tensor, and their inits ARE the 1.0 it was hardcoded
    # to. Still strict about everything else -- a trunk tensor left random here would train anyway
    load_model_state(model, checkpoint["model_state_dict"])
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
             conversation_weighting: bool = False, selector_by_loop: bool = False):
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
    grounded_sum, grounded_batches = 0.0, 0
    grounded_scores, grounded_labels, grounded_has_evidence = [], [], []
    selector_sink = {}

    for batch in dataset:
        if n_batches >= max_batches:
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


def sft(args):
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
    # class other code imports is untouched. Not saved in the checkpoint: a resumed run is
    # relaunched with the same flags, like --reader-no-rotary
    overrides = {name: value for name, value in (
        ("data_dir", getattr(args, "data_dir", None)),
        ("train_split", getattr(args, "train_split", None)),
        ("val_split", getattr(args, "val_split", None)),
    ) if value}
    if overrides:
        cfg = type(cfg.__name__, (cfg,), overrides)
        logger.info(f"data overrides: {overrides}")
    ceiling_by_source, ceiling_all = {}, None
    if getattr(args, "ceiling_json", None):
        with open(args.ceiling_json, "r", encoding="utf-8") as f:
            ceiling = json.load(f)
        ceiling_all = (ceiling.get("all") or {}).get("ceiling")
        ceiling_by_source = {name: entry["ceiling"] for name, entry in
                             (ceiling.get("by_source") or {}).items() if "ceiling" in entry}
        logger.info(f"in-context ceiling read from {args.ceiling_json}: pooled {ceiling_all}, "
                    f"by source {ceiling_by_source}")
    # two variants of the same profile (different seeds, identical everything else) would otherwise
    # share a directory, and the second would silently RESUME the first instead of starting from
    # its own seed -- the resume path only checks the phase label, which is the same for both
    if args.run_name:
        checkpoint_dir = f"{checkpoint_dir}_{args.run_name}"

    data_dir = os.path.join(BASE_DIR, cfg.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    logger.info(f"Tokenizer loaded from {TOKENIZER_DIR} with vocab size {tokenizer.vocab_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    log_precision_mode()
    logger.info(
        f"Profile: {phase} (lr={cfg.lr:.1e}, {cfg.num_epochs} epoch(s), "
        f"splits {cfg.train_split}/{cfg.val_split}, per-conversation loss weighting "
        f"{'ON' if cfg.conversation_loss_weighting else 'off'}) -> {checkpoint_dir}"
    )

    train_dataset = make_dataset(data_dir, cfg.train_split, tokenizer, cfg)
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

    rows_per_epoch = estimate_packed_rows(
        train_dataset.idx_path, cfg.Seq_length, ModelConfig.Params["mtp_num_extra_tokens"],
        split_documents=isinstance(train_dataset, Dataset),
        evidx_path=getattr(train_dataset, "evidx_path", None),
        max_evidence_tokens=getattr(train_dataset, "max_evidence_tokens", 0),
        order=(train_dataset.document_order(train_dataset.num_docs)
               if getattr(train_dataset, "shuffle", False) else None),
        num_workers=NUM_DATA_WORKERS,
    )
    micro_steps = rows_per_epoch * cfg.num_epochs / cfg.Batch_size
    total_steps = max(1, int(micro_steps / cfg.grad_accumulation_steps))
    logger.info(
        f"{phase} plan: ~{rows_per_epoch:,} packed rows/epoch x {cfg.num_epochs} epochs "
        f"-> ~{total_steps:,} optimizer steps at batch {cfg.Batch_size} x accum "
        f"{cfg.grad_accumulation_steps}"
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
        logger.info("evidence reader: no rotary (query and evidence keys unrotated)")
    model = TinyMoETransformer(**model_params).to(device).to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    model._token_tracker.pad_token_id = tokenizer.pad_token_id
    # router exploration noise is fully annealed by ~1B pretraining tokens; SFT is not exploration
    model.moe.set_router_noise(0.0)
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
    optimizer = optim.AdamW(param_groups, lr=cfg.lr)
    scheduler = build_sft_scheduler(optimizer, total_steps, cfg)

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_lib.cleanup_stale_files(checkpoint_dir)
    run_state_path = os.path.join(checkpoint_dir, "run_state.json")

    start_epoch, step_offset, start_doc_idx = 0, 0, 0
    losses, resumed, resumed_kill_checked = [], False, False
    start_token_count, token_count = 0, 0

    found = ckpt_lib.find_resume_checkpoint(
        checkpoint_dir,
        lambda path: load_sft_checkpoint(model, optimizer, scheduler, path, phase, checkpoint_dir),
    )
    if found is not None:
        _, state = found
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
        model._token_tracker.num_tokens = token_count
        resumed = True
        logger.info(
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
            init_path = pull_from_hub(repo, args.hub_file, PRETRAINED_DIR, token)
            try:
                pull_from_hub(repo, "manifest.json", BASE_DIR, token)
            except Exception as e:
                # the manifest matters for prepare_sft_data.py (holdout hashes), not for training
                logger.warning(f"could not pull manifest.json from {repo}: {e}")
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
    # branch above loaded weights into the model -- reseed them or the first step would undo the
    # entire pretrained state. Adam's moments came back by param-group position on a resume, which
    # this copy does not disturb.
    with torch.no_grad():
        for bf16_param, master in master_pairs:
            master.data.copy_(bf16_param.data.float())

    if frozen_gate is not None:
        gate_value = float(frozen_gate.detach().float().item())
        if gate_value != 0.0:
            logger.warning(
                f"evidence_gate_scale loaded as {gate_value:.4e}; zeroing it, since the gate is "
                f"frozen and only a zero scale makes it exactly 1.0"
            )
            with torch.no_grad():
                frozen_gate.zero_()
        logger.info("evidence_gate_scale frozen at 0 (ungated reader, no gradient)")

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
            logger.info(
                f"seed IR value path: |g_proj|rms {g_rms:.3e}"
                + (f", |direct_gate|rms {d_rms:.3e}" if d_rms is not None else "")
            )
            if g_rms == 0.0 and d_rms == 0.0:
                logger.warning(
                    "g_proj and direct_gate are both exactly zero: they hold each other at zero, so "
                    "the IR expert's output is zero for this run and the selector learns only from "
                    "the selection loss (the frozen gate removes its other path through the reader)"
                )

    # same stop contract as pretraining, so an unattended/interruptible box gets a checkpoint out
    # of a SIGTERM instead of losing everything since the last save. clear_sentinel() first: a STOP
    # left over from a previous run would otherwise kill every relaunch before it trains a step.
    control = RunControl(checkpoint_dir)
    control.clear_sentinel()
    control.install()

    upload_repo = args.upload_repo if args.upload_repo is not None else cfg.upload_repo(HF_UPLOAD_REPO)
    if not upload_repo:
        logger.warning(
            f"{phase} uploads are OFF ({phase}.hf_upload_repo is empty). Fine locally; on a rented "
            "box the checkpoints die with the instance -- pass --upload-repo <repo> there."
        )
    hf = HFSync(upload_repo, token=get_hf_token())
    loss_png = os.path.join(checkpoint_dir, "loss_graph.png")
    experts_png = os.path.join(checkpoint_dir, "expert_selection.png")
    status_path = os.path.join(checkpoint_dir, "status.json")

    accelerator = Accelerator(
        device_placement=True,
        split_batches=True,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    unwrapped_model = accelerator.unwrap_model(model)

    full_n_loops = ModelConfig.Params["n_loops"]
    loop_rng = random.Random(cfg.seed)
    worker_state = torch.full((max(NUM_DATA_WORKERS, 1),), -1, dtype=torch.long, device=device)

    def snapshot_global_offset(fallback):
        # same conservative min-across-workers rule as pretrain.py: a worker at position p next
        # wants p + NUM_DATA_WORKERS, so the smallest such value skips nobody's unconsumed work
        rows = worker_state.cpu().tolist()
        seen = [r for r in rows if r >= 0]
        return min(seen) + NUM_DATA_WORKERS if seen else fallback

    timer = time.time()
    last_log_time = timer
    last_token_count = unwrapped_model._token_tracker.sync()
    target_tokens = None  # filled in after the first log interval, once tokens/row is known

    def save_and_sync(epoch, step, loss_value, tokens, final=False):
        name = (ckpt_lib.final_name(phase) if final
                else ckpt_lib.rolling_name(phase, tokens - start_token_count, loss_value))
        path = os.path.join(checkpoint_dir, name)
        save_sft_checkpoint(
            unwrapped_model, optimizer, scheduler, path,
            epoch=epoch, step=step, token_count=tokens, start_token_count=start_token_count,
            global_offset=snapshot_global_offset(start_doc_idx), losses=losses,
            seed=cfg.seed, phase=phase, kill_checked=kill_checked,
        )
        ckpt_lib.write_run_state(run_state_path, phase, tokens, name)
        try:
            save_loss_graph(losses, loss_png)
            save_expert_selection_graph(unwrapped_model.moe.expert_tracker.get_stats(), experts_png)
        except Exception as e:
            logger.error(f"Error occurred while saving graphs: {e}")

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
                         selector_by_loop=True)
        per_condition = (stats or {}).get("per_condition_ce") or {}
        if "none" not in per_condition or "gold" not in per_condition:
            logger.warning(f"fixed-target pass read no gold/none tokens from {fixed_split}")
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
        logger.info(" | ".join(parts))
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
                logger.info(line)
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
            logger.info(f"[eval fixed] loop {loop_idx + 1} | " + " | ".join(fields))
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
            logger.info(" | ".join(fields))
        return gold_gain, content_gain

    def run_validation(epoch, step):
        """The held-out objective pass, then the fixed-target pass. Returns
        ``(gold_gain, content_gain)``, both None without a fixed split."""
        stats = evaluate(unwrapped_model, val_dataset, device, tokenizer.pad_token_id,
                         cfg.eval_max_batches,
                         conversation_weighting=cfg.conversation_loss_weighting)
        if stats is None:
            logger.warning(
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
        if "grounded" in stats:
            sel_str += f" | grounded: {stats['grounded']:.4f}"
            if grounded_str(stats):
                sel_str += f" ({grounded_str(stats)})"
        logger.info(
            f"[eval] epoch {epoch} step {step} | CE: {stats['ce']:.4f} | ppl: {stats['ppl']:.3f} | "
            f"p_max: {stats['p_max']:.4f} | top1_acc: {stats['top1_acc']:.4f} | "
            f"{stats['tokens']:,} supervised tokens over {stats['batches']} batches"
            f"{cond_str}{sel_str}"
        )

    sft_tokens = token_count - start_token_count
    next_checkpoint = sft_tokens + cfg.checkpoint_every_tokens
    next_eval = sft_tokens + cfg.eval_every_tokens
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
            logger.info(f"grad norms at optimizer step {optimizer_step} (pre-clip): "
                        + ", ".join(norms))
        return probe

    if not resumed:
        # the baseline every later reading is a difference from; the seed's reader is dead, so the
        # fixed-target gain should read about zero here
        run_validation(start_epoch, step_offset)

    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            resume_epoch = resumed and epoch == start_epoch
            # the LM reader has no per-epoch permutation to set: it reads in on-disk order, which
            # is where the corpus builder already put the source mix
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
            train_dataset.start_doc_idx = start_doc_idx if resume_epoch else 0
            if not resume_epoch:
                worker_state.fill_(-1)
                step = 0

            for local_step, batch in enumerate(dataloader):
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
                is_log_step = step % LOG_INTERVAL == 0
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
                val_loss = loss.item()
                losses.append(val_loss)
                token_count = unwrapped_model._token_tracker.sync()
                sft_tokens = token_count - start_token_count
                if evidence_token_pending is not None:
                    evidence_token_count += int(evidence_token_pending.item())
                    evidence_token_pending.zero_()
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
                    # logged as its own line so a loss step at a refresh boundary is attributable to
                    # the refresh rather than to the data; candidate recall (the number that says
                    # whether the partition is still covering what the read wants) is already inside
                    # ir_refresh_stats, reported identically regardless of which profile triggered it
                    logger.info(
                        f"IR cluster refresh at {sft_tokens / 1e6:.1f}M {phase} tokens"
                        + (f" (temperature scale {scale:.4f})" if args.ir else "")
                        + f": {ir_refresh_stats}"
                    )

                per_loop_ce = ", ".join(f"{ce.item():.4f}" for ce in metrics["per_loop_ce"])
                loop_scale = ", ".join(f"{s:.4f}" for s in unwrapped_model.moe.loop_scale.tolist())
                # per loop IR retrieval entropy over ln(num_ir_entries), same field pretrain.py logs
                ir_entropy = (
                    unwrapped_model.moe.ir_tracker.get_stats()
                    if unwrapped_model.moe.ir_tracker is not None else []
                )
                ir_entropy_str = (
                    f"IR E/ln{unwrapped_model.moe.ir_tracker.num_entries}: ["
                    + ", ".join(f"{e:.4f}" for e in ir_entropy) + "] | "
                    if ir_entropy else ""
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
                        ir_entropy_str += f"selection: {sel.item():.4f} | "
                    # the other supervised head on the port. NaN whenever the checkpoint carries no
                    # head or the corpus no .ans, which is a missing label rather than a zero loss
                    grounded = metrics.get("groundedness_loss")
                    if grounded is not None:
                        ir_entropy_str += f"grounded: {grounded.item():.4f} | "
                    evidence_module = unwrapped_model.moe.shared_evidence
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
                            ir_entropy_str += f"external mass: {{{', '.join(mass_by_cond)}}} | "

                def _metric(key):
                    value = metrics.get(key)
                    return value.item() if value is not None else float("nan")

                eta = eta_seconds(sft_tokens, target_tokens or 0, tokens_per_sec) if target_tokens else None
                # a separate field from "{phase} tokens", never folded into it: that counter anchors
                # the LR schedule, the token target and the checkpoint cadence, and the evidence
                # stream (~3x its size on the smoke corpus) is not part of what any of those read.
                # Without its own field the log makes an evidence-conditioned step look as cheap as
                # a plain SFT one at the same "{phase} tokens" reading.
                evidence_field = (
                    f" | evidence stream: {evidence_token_count / 1e6:.2f}M" if args.evidence else ""
                )
                logger.info(
                    f"Epoch {epoch} | Step {step} | Loss: {val_loss:.4f} | Loss (CE): {loss_ce.item():.4f} | "
                    f"Aux: {aux_loss.item():.4f} | loop_scale: [{loop_scale}] | "
                    f"p_max: {_metric('p_max'):.4f} | top1_acc: {_metric('top1_acc'):.4f} | "
                    f"per-loop CE: [{per_loop_ce}] | {ir_entropy_str}"
                    f"LR: {scheduler.get_last_lr()[0]:.3e} | {phase} tokens: {sft_tokens / 1e6:.2f}M"
                    f"{evidence_field} | "
                    f"Tokens/sec: {tokens_per_sec:.0f} | "
                    f"Peak Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | "
                    f"Time: {(now - timer) / 60:.2f} min"
                    + (f" | ETA: {format_duration(eta)}" if eta is not None else "")
                )

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
                    next_eval = sft_tokens + cfg.eval_every_tokens
                    if kill_tokens and not kill_checked and sft_tokens >= kill_tokens:
                        min_gain = cfg.kill_min_gain
                        if gold_gain is None or content_gain is None:
                            # left armed: a pass that read nothing is not a decision
                            logger.info(
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
                                logger.warning(
                                    f"KILL: fixed-target {readings}; one is under {min_gain} at "
                                    f"{sft_tokens / 1e6:.2f}M {phase} tokens. Saving and stopping "
                                    f"(exit {EXIT_USER_STOP}, not restarted by a wrapper)."
                                )
                                save_and_sync(epoch, step, val_loss, token_count)
                                exit_code = EXIT_USER_STOP
                                stop_training = True
                                break
                            logger.info(
                                f"kill check passed at {sft_tokens / 1e6:.2f}M {phase} tokens: "
                                f"{readings}, both >= {min_gain}"
                            )

                # polled at the log cadence: a stat every few seconds, no GPU sync, and well
                # inside vast's SIGTERM grace period
                control.poll()
                if control.stop_requested:
                    logger.info(f"Stopping: {control.reason}. Saving checkpoint...")
                    save_and_sync(epoch, step, val_loss, token_count)
                    exit_code = control.exit_code
                    stop_training = True
                    break

                if sft_tokens >= next_checkpoint or control.take_checkpoint_request():
                    save_and_sync(epoch, step, val_loss, token_count)
                    next_checkpoint = sft_tokens + cfg.checkpoint_every_tokens

            if stop_training:
                break

            # end of epoch: the next one starts its own permutation from position 0
            start_doc_idx, step_offset = 0, 0
            resumed = False
            worker_state.fill_(-1)
            logger.info(f"Epoch {epoch} finished at {sft_tokens / 1e6:.2f}M {phase} tokens")

        if stop_training:
            # a stop is not a finished run: no final checkpoint, and a restartable exit code so a
            # wrapper knows to relaunch (the rolling checkpoint just written is the resume point)
            return exit_code

        token_count = unwrapped_model._token_tracker.sync()
        run_validation(cfg.num_epochs - 1, step)
        save_and_sync(cfg.num_epochs - 1, step, losses[-1] if losses else float("nan"),
                      token_count, final=True)
        logger.info(
            f"{phase} complete: {(token_count - start_token_count) / 1e6:.2f}M tokens over "
            f"{cfg.num_epochs} epochs in {(time.time() - timer) / 60:.1f} min"
        )
    except KeyboardInterrupt:
        logger.info("Interrupted. Saving checkpoint...")
        exit_code = EXIT_USER_STOP
        try:
            save_and_sync(epoch, step, losses[-1] if losses else float("nan"),
                          unwrapped_model._token_tracker.sync())
        except Exception as e:
            logger.error(f"Failed to save the interrupt checkpoint: {e}")
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
    parser.add_argument("--run-name", default=None,
                        help="suffix the profile's checkpoint directory, e.g. --run-name random "
                             "writes ckpts/ir_random. Required to run two seeds of one profile "
                             "against each other: a shared directory means the second run resumes "
                             "the first instead of starting from its own seed")
    parser.add_argument("--data-dir", default=None,
                        help="override the profile's data_dir (relative paths resolve against the "
                             "repo root). Not saved in the checkpoint: pass it on every launch")
    parser.add_argument("--train-split", default=None,
                        help="override the profile's train split. Pass it on every launch")
    parser.add_argument("--val-split", default=None,
                        help="override the profile's validation split. Pass it on every launch")
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

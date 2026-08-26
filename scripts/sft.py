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
import sys
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
from modules.data.sft_dataset import SFTDataset
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.information_retrieval import is_rebuilt_ir_param
from modules.model.mtp import compute_mtp_loss
from modules.model.transformer import TinyMoETransformer
from modules.runtime import checkpoints as ckpt_lib
from modules.runtime.control import EXIT_OK, EXIT_USER_STOP, RunControl
from modules.runtime.hf_sync import HFSync
from modules.runtime.status import eta_seconds, format_duration, write_status
from config import IRConfig, ModelConfig, RepairConfig, SFTConfig, TrainingConfig
from scripts.pretrain import (
    USE_LOW_PRECISION, chosen_recipe, log_precision_mode, sample_n_loops,
    save_expert_selection_graph, save_loss_graph, train_step,
)
from utils import BASE_DIR, BF16, HF_UPLOAD_REPO, TOKENIZER_DIR, get_hf_token, logger

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


def build_sft_param_groups(model: TinyMoETransformer, weight_decay: float, fresh_lr: float = None):
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
        fresh_lr: when given, the rebuilt IR tensors (``is_rebuilt_ir_param``) go into their own
            groups at this learning rate instead of sharing the run's. They are the only tensors in
            the model with no training behind them: at the trunk's 1e-5 a from-scratch key table
            against an otherwise converged model never gets anywhere, and at a rate that would
            train it the 16B-token trunk moves too. The LR schedule still scales every group by the
            same factor, so this sets two *base* rates, not two shapes.

    Returns:
        ``(param_groups, master_pairs)`` where ``master_pairs`` is ``[(bf16_param, fp32_master)]``
        covering every trainable parameter.
    """
    buckets = {("trunk", True): [], ("trunk", False): [], ("fresh", True): [], ("fresh", False): []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        origin = "fresh" if (fresh_lr is not None and is_rebuilt_ir_param(name)) else "trunk"
        buckets[(origin, param.ndim >= 2)].append(param)

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
    return param_groups, master_pairs


def estimate_packed_rows(idx_path: str, max_length: int, num_mtp_tokens: int,
                         split_documents: bool = False) -> int:
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

    Returns:
        Estimated number of packed rows for one epoch.
    """
    offsets = np.fromfile(idx_path, dtype=np.uint64)
    lengths = np.diff(offsets).astype(np.int64) + num_mtp_tokens
    if split_documents:
        return max(1, int(lengths.sum() // max_length) + 1)
    lengths = lengths[lengths <= max_length]

    rows, used = 1, 0
    for length in lengths.tolist():
        if used + length > max_length:
            rows += 1
            used = 0
        used += length
    return rows


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
                        start_token_count, global_offset, losses, seed, phase=SFT_PHASE):
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
        "sft": {"start_token_count": start_token_count, "seed": seed},
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
    model.load_state_dict(checkpoint["model_state_dict"])
    token_count = checkpoint.get("token_count", 0)
    logger.info(
        f"Initialized from pretrained checkpoint {os.path.basename(path)} "
        f"({token_count / 1e9:.3f}B pretraining tokens, phase={checkpoint.get('phase')})"
    )
    return token_count


@torch.no_grad()
def evaluate(model, dataset: SFTDataset, device: str, pad_token_id: int, max_batches: int,
             conversation_weighting: bool = False):
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

    for batch in dataset:
        if n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        document_ids = batch["document_ids"].to(device)
        cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
        pad_mask = input_ids == pad_token_id
        loss_weights = batch["loss_weights"].to(device) if conversation_weighting else None

        with te.autocast(enabled=USE_LOW_PRECISION, recipe=chosen_recipe):
            out = model(
                input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                return_aux_loss=True, return_hidden=True,
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
    return result


def sft(args):
    # one function, two profiles. --repair swaps the config class, the phase label and the
    # checkpoint directory and nothing else: see this module's docstring for why the repair pass is
    # not a second script.
    if args.repair and args.ir:
        raise SystemExit("--repair and --ir are different profiles; pick one")
    if args.ir:
        cfg, phase, checkpoint_dir = IRConfig, IR_PHASE, IR_CHECKPOINT_DIR
    elif args.repair:
        cfg, phase, checkpoint_dir = RepairConfig, REPAIR_PHASE, REPAIR_CHECKPOINT_DIR
    else:
        cfg, phase, checkpoint_dir = SFTConfig, SFT_PHASE, SFT_CHECKPOINT_DIR

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
    dataloader = DataLoader(train_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS,
                            prefetch_factor=2)

    rows_per_epoch = estimate_packed_rows(
        train_dataset.idx_path, cfg.Seq_length, ModelConfig.Params["mtp_num_extra_tokens"],
        split_documents=isinstance(train_dataset, Dataset),
    )
    micro_steps = rows_per_epoch * cfg.num_epochs / cfg.Batch_size
    total_steps = max(1, int(micro_steps / cfg.grad_accumulation_steps))
    logger.info(
        f"{phase} plan: ~{rows_per_epoch:,} packed rows/epoch x {cfg.num_epochs} epochs "
        f"-> ~{total_steps:,} optimizer steps at batch {cfg.Batch_size} x accum "
        f"{cfg.grad_accumulation_steps}"
    )

    # dropout override only; every other model hyperparameter must match the pretrained checkpoint
    model = TinyMoETransformer(**cfg.model_params()).to(device).to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    model._token_tracker.pad_token_id = tokenizer.pad_token_id
    # router exploration noise is fully annealed by ~1B pretraining tokens; SFT is not exploration
    model.moe.set_router_noise(0.0)

    param_groups, master_pairs = build_sft_param_groups(
        model, cfg.weight_decay, fresh_lr=getattr(cfg, "fresh_lr", None),
    )
    optimizer = optim.AdamW(param_groups, lr=cfg.lr)
    scheduler = build_sft_scheduler(optimizer, total_steps, cfg)

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_lib.cleanup_stale_files(checkpoint_dir)
    run_state_path = os.path.join(checkpoint_dir, "run_state.json")

    start_epoch, step_offset, start_doc_idx = 0, 0, 0
    losses, resumed = [], False
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
            seed=cfg.seed, phase=phase,
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

    def run_validation(epoch, step):
        stats = evaluate(unwrapped_model, val_dataset, device, tokenizer.pad_token_id,
                         cfg.eval_max_batches,
                         conversation_weighting=cfg.conversation_loss_weighting)
        if stats is None:
            logger.warning(
                f"validation pass produced no supervised tokens -- is {cfg.val_split} empty?"
            )
            return
        logger.info(
            f"[eval] epoch {epoch} step {step} | CE: {stats['ce']:.4f} | ppl: {stats['ppl']:.3f} | "
            f"p_max: {stats['p_max']:.4f} | top1_acc: {stats['top1_acc']:.4f} | "
            f"{stats['tokens']:,} supervised tokens over {stats['batches']} batches"
        )

    sft_tokens = token_count - start_token_count
    next_checkpoint = sft_tokens + cfg.checkpoint_every_tokens
    next_eval = sft_tokens + cfg.eval_every_tokens
    # the IR profile's two extra schedules. Both are driven from the log block, which already
    # syncs the token counter -- neither adds a host sync of its own, and LOG_INTERVAL is ~160k
    # tokens here, far finer than either cadence needs.
    total_micro_steps = max(1, total_steps * cfg.grad_accumulation_steps)
    next_refresh = sft_tokens + getattr(cfg, "cluster_refresh_tokens", 0)
    ir_refresh_stats = []
    # bound before the try: the interrupt handler saves a checkpoint using both, and a Ctrl-C
    # during the very first batch must not turn into a NameError that loses the save
    step, epoch = step_offset, start_epoch
    stop_training, exit_code = False, EXIT_OK

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
                )

                if not is_log_step:
                    continue

                # everything below pulls to the host; throttled to LOG_INTERVAL like pretrain.py
                val_loss = loss.item()
                losses.append(val_loss)
                token_count = unwrapped_model._token_tracker.sync()
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
                    # be a known function of position from step 0 to be reproducible.
                    scale = cfg.temperature_scale(step / total_micro_steps)
                    unwrapped_model.moe.set_ir_temperature_scale(scale)
                    if sft_tokens >= next_refresh:
                        ir_refresh_stats = apply_ir_refresh(
                            unwrapped_model, optimizer, master_pairs,
                            unwrapped_model.moe.refresh_ir_clusters(dead_quantile=cfg.dead_quantile),
                        )
                        next_refresh = sft_tokens + cfg.cluster_refresh_tokens
                        # logged as its own line so a loss step at a refresh boundary is
                        # attributable to the refresh rather than to the data
                        logger.info(
                            f"IR cluster refresh at {sft_tokens / 1e6:.1f}M {phase} tokens "
                            f"(temperature scale {scale:.4f}): {ir_refresh_stats}"
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
                    ir_entropy_str += f"IR temp: {ir_module.temperature.detach().item():.4f} | "

                def _metric(key):
                    value = metrics.get(key)
                    return value.item() if value is not None else float("nan")

                eta = eta_seconds(sft_tokens, target_tokens or 0, tokens_per_sec) if target_tokens else None
                logger.info(
                    f"Epoch {epoch} | Step {step} | Loss: {val_loss:.4f} | Loss (CE): {loss_ce.item():.4f} | "
                    f"Aux: {aux_loss.item():.4f} | loop_scale: [{loop_scale}] | "
                    f"p_max: {_metric('p_max'):.4f} | top1_acc: {_metric('top1_acc'):.4f} | "
                    f"per-loop CE: [{per_loop_ce}] | {ir_entropy_str}"
                    f"LR: {scheduler.get_last_lr()[0]:.3e} | {phase} tokens: {sft_tokens / 1e6:.2f}M | "
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
                    run_validation(epoch, step)
                    next_eval = sft_tokens + cfg.eval_every_tokens

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
    parser.add_argument("--ir", action="store_true",
                        help="run the IR table's sharpening finetune instead: config.yaml's ir: "
                             "block, the ir_train/ir_val splits, ckpts/ir, a second learning rate "
                             "for the rebuilt table and a retrieval temperature anneal. Seed it "
                             "with -c <a scripts/migrate_ir_reshape.py output>")
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

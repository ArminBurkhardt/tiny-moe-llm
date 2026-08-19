"""Supervised fine-tuning (PLAN.md Step 12).

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
    *identical* between pretraining and SFT -- per-loop CE weights, aux loss, ponder ramp,
    loop-count sampling -- is to have exactly one copy of it. Prompt masking needs no changes at all:
    the dataset emits ``-100`` labels over prompt tokens and every loss term already routes through
    ``ignore_index=-100``, including the MTP heads (they read the same ``labels`` tensor).
  * The model's **global token counter**, continued rather than reset. The ponder ramp and the
    router-noise anneal are both driven from it, and both are functions that have long since
    finished at ~30B tokens. Resetting it to zero would silently restart the ponder warmup, i.e.
    turn the ponder loss off for the whole of SFT and let ``p_halt`` drift wherever CE pushes it.
    SFT progress is tracked separately as ``token_count - start_token_count``.

What is genuinely different:

  * **fp32 master weights for every parameter, not just the undecayed ones** -- see
    ``build_sft_param_groups``. This is a correctness requirement at SFT's learning rate, not a
    refinement.
  * **A masked, shuffled, non-splitting dataset** (``modules/data/sft_dataset.py``).
  * **A validation pass** on ``sft_val`` at checkpoint cadence, reporting the calibration signals
    (``p_max``/top-1) the abstention acceptance criterion is about.

Run from the repo root:

    python scripts/sft.py --from-hub          # pull the pretrained checkpoint, then train
    python scripts/sft.py -c ckpts/training/checkpoint_phase2_final.pt
    python scripts/sft.py                     # resume from the newest checkpoint in ckpts/sft
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

from modules.data.sft_dataset import SFTDataset
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.mtp import compute_mtp_loss
from modules.model.transformer import TinyMoETransformer
from modules.runtime import checkpoints as ckpt_lib
from modules.runtime.control import EXIT_OK, EXIT_USER_STOP, RunControl
from modules.runtime.hf_sync import HFSync
from modules.runtime.ponder import PonderController
from modules.runtime.status import eta_seconds, format_duration, write_status
from config import ModelConfig, SFTConfig, TrainingConfig
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
# --from-hub lands the pretrained checkpoint HERE, deliberately not in SFT_CHECKPOINT_DIR: it is
# named checkpoint_phase2_final.pt, which matches ckpt_lib's "checkpoint_*.pt" resume scan, so a
# second launch would offer pretraining's own optimizer/scheduler state to load_sft_checkpoint as
# if it were a resumable SFT run.
PRETRAINED_DIR = os.path.join(BASE_DIR, "ckpts", "pretrained")
DEFAULT_HUB_CHECKPOINT = "checkpoints/final/checkpoint_phase2_final.pt"
NUM_DATA_WORKERS = 4
LOG_INTERVAL = 10


def build_sft_param_groups(model: TinyMoETransformer, weight_decay: float):
    """Split parameters into decayed / undecayed groups, **all** shadowed by fp32 masters.

    The decay split is the same one ``pretrain.build_param_groups`` makes and for the same reasons
    (``moe.loop_scale``, ``layer_scalar``, RMSNorm gains and ``halt_proj.bias`` all have a
    degenerate zero, and every one of them is ndim <= 1).

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
        weight_decay: applied to the ndim >= 2 group only.

    Returns:
        ``(param_groups, master_pairs)`` where ``master_pairs`` is ``[(bf16_param, fp32_master)]``
        covering every trainable parameter.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)

    decay_masters = [p.detach().clone().float().requires_grad_(True) for p in decay]
    no_decay_masters = [p.detach().clone().float().requires_grad_(True) for p in no_decay]
    logger.info(
        f"SFT optimizer param groups: {len(decay)} decayed (wd={weight_decay}), {len(no_decay)} "
        f"undecayed (norms/biases/gates); all {len(decay) + len(no_decay)} stepped via fp32 masters"
    )
    param_groups = [
        {"params": decay_masters, "weight_decay": weight_decay},
        {"params": no_decay_masters, "weight_decay": 0.0},
    ]
    master_pairs = list(zip(decay, decay_masters)) + list(zip(no_decay, no_decay_masters))
    return param_groups, master_pairs


def estimate_packed_rows(idx_path: str, max_length: int, num_mtp_tokens: int) -> int:
    """How many packed rows the SFT corpus yields, by replaying the packing rule over the index.

    The LR schedule needs a total step count up front, and "corpus tokens / (batch * seq)" is a bad
    estimate here: SFTDataset never splits a conversation across rows, so every row carries some
    trailing padding, and each conversation also costs ``num_mtp_tokens`` separator slots. On a
    corpus of short conversations that gap is easily 10%, which would end the cosine well before
    the data does and leave the tail of training at the LR floor.

    The replay is over the on-disk order rather than the epoch's permutation -- the row count barely
    moves between orderings (it depends on the length *distribution*, not the sequence), and doing
    it exactly per epoch would mean materializing every permutation before training starts.

    Args:
        idx_path: ``{split}.idx``, uint64 document-end offsets with a leading 0.
        max_length: row length.
        num_mtp_tokens: separator slots appended after each conversation.

    Returns:
        Estimated number of packed rows for one epoch.
    """
    offsets = np.fromfile(idx_path, dtype=np.uint64)
    lengths = np.diff(offsets).astype(np.int64) + num_mtp_tokens
    lengths = lengths[lengths <= max_length]

    rows, used = 1, 0
    for length in lengths.tolist():
        if used + length > max_length:
            rows += 1
            used = 0
        used += length
    return rows


def build_sft_scheduler(optimizer: optim.Optimizer, total_steps: int):
    """Linear warmup -> cosine decay to ``lr * lr_min_factor``, anchored to this run's own steps.

    Not shared with ``pretrain.build_scheduler``: that one is anchored to
    ``TrainingConfig.total_steps`` (the combined pretraining budget) because phase 2 has to
    continue phase 1's decay. SFT is a fresh schedule over a fresh optimizer.
    """
    warmup_steps = max(1, min(int(total_steps * SFTConfig.warmup_fraction), total_steps - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=SFTConfig.lr * SFTConfig.lr_min_factor,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )


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
                        start_token_count, global_offset, losses, ponder_state, seed):
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
        "phase": SFT_PHASE,
        "losses": losses,
        "ponder_state": ponder_state,
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


def load_sft_checkpoint(model, optimizer, scheduler, path):
    """Restore a full SFT run from its own checkpoint. Returns the resume state dict.

    Raises on anything that is not an SFT checkpoint -- and checks that *before* touching the
    model, so a rejected file leaves no partial state behind. A pretraining checkpoint dropped into
    ``ckpts/sft`` by hand would otherwise load cleanly: its optimizer state has the same two param
    groups with the same shapes, so AdamW's moments from a 4e-4 run would be silently adopted as
    this fine-tune's, along with a scheduler anchored to the 29.9B-token cosine.
    """
    checkpoint = torch.load(path, map_location="cpu")
    phase = checkpoint.get("phase")
    if phase != SFT_PHASE:
        raise ValueError(
            f"{os.path.basename(path)} was written during phase={phase!r}, not {SFT_PHASE!r}. "
            f"Pass it with -c to initialize FROM it instead of resuming it, and keep "
            f"{SFT_CHECKPOINT_DIR} for SFT checkpoints only."
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
        "ponder_state": checkpoint.get("ponder_state", None),
        "seed": sft_extra.get("seed", SFTConfig.seed),
    }


def load_pretrained_weights(model, path: str):
    """Seed SFT from a pretraining checkpoint: weights and bookkeeping, no optimizer state.

    The optimizer is deliberately *not* restored. Pretraining's AdamW moments were accumulated at
    lr=4e-4 against a different objective; carrying them into a 3e-5 fine-tune would spend the
    first few hundred steps unwinding momentum that no longer describes the loss surface.

    Returns:
        ``(token_count, ponder_state)``. The token count is carried forward on purpose -- see this
        module's docstring for why resetting it would silently disable the ponder loss.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    token_count = checkpoint.get("token_count", 0)
    ponder_state = checkpoint.get("ponder_state", None)
    logger.info(
        f"Initialized from pretrained checkpoint {os.path.basename(path)} "
        f"({token_count / 1e9:.3f}B pretraining tokens, phase={checkpoint.get('phase')})"
    )
    return token_count, ponder_state


@torch.no_grad()
def evaluate(model, dataset: SFTDataset, device: str, pad_token_id: int, max_batches: int):
    """Validation pass over ``sft_val``: CE on supervised tokens plus the calibration signals.

    Reports ``p_max``/top-1 accuracy because the acceptance criterion is about the abstention
    signal's calibration, not about val loss, and a fixed held-out slice shows drift in it far
    earlier than the noisy training log does.

    Runs at the full configured loop depth (no ``n_loops`` override, no loop-count sampling) and
    with subsampling off, so successive eval numbers are read at one fixed operating point.
    """
    was_training = model.training
    model.eval()
    # the eval forwards would otherwise inflate the trained-token counter, which drives the ponder
    # ramp, the checkpoint cadence and the reported progress (same guard as pretrain.dry_run)
    token_count_before = model._token_tracker.num_tokens

    ce_sum, token_sum = 0.0, 0
    signal_sums = {"p_max": 0.0, "top1_acc": 0.0, "p_halt": 0.0}
    n_batches = 0

    for batch in dataset:
        if n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        document_ids = batch["document_ids"].to(device)
        cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
        pad_mask = input_ids == pad_token_id

        with te.autocast(enabled=USE_LOW_PRECISION, recipe=chosen_recipe):
            out = model(
                input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                return_aux_loss=True, return_hidden=True,
            )
            hidden, p_halt = out[0], out[2]
            extra_token_outputs = out[3] if model.has_mtp else None
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
            )

        # weight each batch by its supervised token count: rows differ a lot in how much of them
        # is prompt, so an unweighted mean over batches is not the corpus mean
        n_supervised = int((labels[:, 1:] != -100).sum().item())
        if n_supervised == 0:
            continue
        ce_sum += loss_ce.item() * n_supervised
        token_sum += n_supervised
        for key in ("p_max", "top1_acc"):
            value = metrics.get(key)
            signal_sums[key] += (value.item() if value is not None else float("nan")) * n_supervised
        valid = (~pad_mask).to(p_halt.dtype)
        signal_sums["p_halt"] += (
            ((p_halt * valid).sum() / (valid.sum().clamp(min=1) * p_halt.size(0))).item()
            * n_supervised
        )
        n_batches += 1

    model._token_tracker.num_tokens = token_count_before
    if was_training:
        model.train()

    if token_sum == 0:
        return None
    result = {"ce": ce_sum / token_sum, "tokens": token_sum, "batches": n_batches}
    result.update({key: total / token_sum for key, total in signal_sums.items()})
    result["ppl"] = math.exp(min(result["ce"], 20.0))
    return result


def sft(args):
    data_dir = os.path.join(BASE_DIR, SFTConfig.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    logger.info(f"Tokenizer loaded from {TOKENIZER_DIR} with vocab size {tokenizer.vocab_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    log_precision_mode()

    train_dataset = SFTDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=SFTConfig.Batch_size,
        max_length=SFTConfig.Seq_length,
        split=SFTConfig.train_split,
        num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
        seed=SFTConfig.seed,
    )
    val_dataset = SFTDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=SFTConfig.Batch_size,
        max_length=SFTConfig.Seq_length,
        split=SFTConfig.val_split,
        num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
        seed=SFTConfig.seed,
        shuffle=False,  # a stable order makes successive eval numbers comparable
    )
    dataloader = DataLoader(train_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS,
                            prefetch_factor=2)

    rows_per_epoch = estimate_packed_rows(
        train_dataset.idx_path, SFTConfig.Seq_length, ModelConfig.Params["mtp_num_extra_tokens"],
    )
    micro_steps = rows_per_epoch * SFTConfig.num_epochs / SFTConfig.Batch_size
    total_steps = max(1, int(micro_steps / SFTConfig.grad_accumulation_steps))
    logger.info(
        f"SFT plan: ~{rows_per_epoch:,} packed rows/epoch x {SFTConfig.num_epochs} epochs "
        f"-> ~{total_steps:,} optimizer steps at batch {SFTConfig.Batch_size} x accum "
        f"{SFTConfig.grad_accumulation_steps}"
    )

    # dropout override only; every other model hyperparameter must match the pretrained checkpoint
    model = TinyMoETransformer(**SFTConfig.model_params()).to(device).to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    model._token_tracker.pad_token_id = tokenizer.pad_token_id
    # router exploration noise is fully annealed by ~1B pretraining tokens; SFT is not exploration
    model.moe.set_router_noise(0.0)

    param_groups, master_pairs = build_sft_param_groups(model, SFTConfig.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=SFTConfig.lr)
    scheduler = build_sft_scheduler(optimizer, total_steps)

    ponder_controller = PonderController(
        TrainingConfig.lambda_ponder,
        target=TrainingConfig.ponder_target_p_halt,
        band=TrainingConfig.ponder_p_halt_band,
        factor=TrainingConfig.ponder_adjust_factor,
        cooldown_tokens=TrainingConfig.ponder_adjust_cooldown_tokens,
        lambda_min=TrainingConfig.ponder_lambda_min,
        lambda_max=TrainingConfig.ponder_lambda_max,
        enabled=TrainingConfig.ponder_auto_adjust,
    )

    os.makedirs(SFT_CHECKPOINT_DIR, exist_ok=True)
    ckpt_lib.cleanup_stale_files(SFT_CHECKPOINT_DIR)
    run_state_path = os.path.join(SFT_CHECKPOINT_DIR, "run_state.json")

    start_epoch, step_offset, start_doc_idx = 0, 0, 0
    losses, resumed = [], False
    start_token_count, token_count = 0, 0

    found = ckpt_lib.find_resume_checkpoint(
        SFT_CHECKPOINT_DIR, lambda path: load_sft_checkpoint(model, optimizer, scheduler, path)
    )
    if found is not None:
        _, state = found
        start_epoch = state["epoch"]
        step_offset = state["step"]
        token_count = state["token_count"]
        start_token_count = state["start_token_count"]
        start_doc_idx = state["global_offset"]
        losses = state["losses"]
        ponder_controller.load_state_dict(state["ponder_state"])
        if state["seed"] != SFTConfig.seed:
            # the resume position indexes into a permutation generated from the seed; reading it
            # back under a different seed silently reshuffles which conversations were "already
            # seen", so refuse rather than half-repeat and half-skip an epoch
            raise SystemExit(
                f"checkpoint was written with sft.seed={state['seed']} but config.yaml now says "
                f"{SFTConfig.seed}. The document order (and therefore the resume position) is a "
                f"function of the seed -- restore the old seed or start a fresh run directory."
            )
        model._token_tracker.num_tokens = token_count
        resumed = True
        logger.info(
            f"Resumed SFT at epoch {start_epoch}, position {start_doc_idx:,}, "
            f"{(token_count - start_token_count) / 1e6:.1f}M SFT tokens"
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
                "no SFT checkpoint to resume and no pretrained checkpoint given -- pass "
                "--from-hub, or -c <path to checkpoint_phase2_final.pt>"
            )
        start_token_count, ponder_state = load_pretrained_weights(model, init_path)
        token_count = start_token_count
        ponder_controller.load_state_dict(ponder_state)
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
    control = RunControl(SFT_CHECKPOINT_DIR)
    control.clear_sentinel()
    control.install()

    upload_repo = args.upload_repo if args.upload_repo is not None else SFTConfig.upload_repo(HF_UPLOAD_REPO)
    if not upload_repo:
        logger.warning(
            "SFT uploads are OFF (sft.hf_upload_repo is empty). Fine locally; on a rented box the "
            "checkpoints die with the instance -- pass --upload-repo <repo> there."
        )
    hf = HFSync(upload_repo, token=get_hf_token())
    loss_png = os.path.join(SFT_CHECKPOINT_DIR, "loss_graph.png")
    experts_png = os.path.join(SFT_CHECKPOINT_DIR, "expert_selection.png")
    status_path = os.path.join(SFT_CHECKPOINT_DIR, "status.json")

    accelerator = Accelerator(
        device_placement=True,
        split_batches=True,
        gradient_accumulation_steps=SFTConfig.grad_accumulation_steps,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    unwrapped_model = accelerator.unwrap_model(model)

    full_n_loops = ModelConfig.Params["n_loops"]
    loop_rng = random.Random(SFTConfig.seed)
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
        name = (ckpt_lib.final_name(SFT_PHASE) if final
                else ckpt_lib.rolling_name(SFT_PHASE, tokens - start_token_count, loss_value))
        path = os.path.join(SFT_CHECKPOINT_DIR, name)
        save_sft_checkpoint(
            unwrapped_model, optimizer, scheduler, path,
            epoch=epoch, step=step, token_count=tokens, start_token_count=start_token_count,
            global_offset=snapshot_global_offset(start_doc_idx), losses=losses,
            ponder_state=ponder_controller.state_dict(), seed=SFTConfig.seed,
        )
        ckpt_lib.write_run_state(run_state_path, SFT_PHASE, tokens, name)
        try:
            save_loss_graph(losses, loss_png)
            save_expert_selection_graph(unwrapped_model.moe.expert_tracker.get_stats(), experts_png)
        except Exception as e:
            logger.error(f"Error occurred while saving graphs: {e}")

        repo_dir = "sft/final" if final else "sft"
        hf.upload(path, f"{repo_dir}/{name}", droppable=not final)
        for local, remote in ((status_path, "sft/status.json"),
                              (loss_png, "sft/graphs/loss_graph.png")):
            if os.path.isfile(local):
                hf.upload(local, remote)
        # retention: only deletes what is both outside the window AND confirmed uploaded. With
        # uploads off (the local default) is_uploaded is never true, so nothing is pruned and the
        # local run simply keeps every checkpoint -- which is the right default when the disk is
        # the only copy.
        for deleted in ckpt_lib.prune_checkpoints(
            SFT_CHECKPOINT_DIR, SFTConfig.keep_local_checkpoints, hf.is_uploaded
        ):
            hf.delete(f"sft/{os.path.basename(deleted)}")

    def run_validation(epoch, step):
        stats = evaluate(unwrapped_model, val_dataset, device, tokenizer.pad_token_id,
                         SFTConfig.eval_max_batches)
        if stats is None:
            logger.warning("validation pass produced no supervised tokens -- is sft_val empty?")
            return
        logger.info(
            f"[eval] epoch {epoch} step {step} | CE: {stats['ce']:.4f} | ppl: {stats['ppl']:.3f} | "
            f"p_max: {stats['p_max']:.4f} | top1_acc: {stats['top1_acc']:.4f} | "
            f"p_halt: {stats['p_halt']:.4f} | "
            f"{stats['tokens']:,} supervised tokens over {stats['batches']} batches"
        )

    sft_tokens = token_count - start_token_count
    next_checkpoint = sft_tokens + SFTConfig.checkpoint_every_tokens
    next_eval = sft_tokens + SFTConfig.eval_every_tokens
    # bound before the try: the interrupt handler saves a checkpoint using both, and a Ctrl-C
    # during the very first batch must not turn into a NameError that loses the save
    step, epoch = step_offset, start_epoch
    stop_training, exit_code = False, EXIT_OK

    try:
        for epoch in range(start_epoch, SFTConfig.num_epochs):
            resume_epoch = resumed and epoch == start_epoch
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
                    lambda_ponder=ponder_controller.lambda_ponder,
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
                                        * SFTConfig.grad_accumulation_steps)

                per_loop_ce = ", ".join(f"{ce.item():.4f}" for ce in metrics["per_loop_ce"])
                loop_scale = ", ".join(f"{s:.4f}" for s in unwrapped_model.moe.loop_scale.tolist())
                p_halt_mean = metrics["p_halt_mean"].item()
                ponder_controller.observe(p_halt_mean, token_count, ramp_complete=True)

                def _metric(key):
                    value = metrics.get(key)
                    return value.item() if value is not None else float("nan")

                eta = eta_seconds(sft_tokens, target_tokens or 0, tokens_per_sec) if target_tokens else None
                logger.info(
                    f"Epoch {epoch} | Step {step} | Loss: {val_loss:.4f} | Loss (CE): {loss_ce.item():.4f} | "
                    f"Aux: {aux_loss.item():.4f} | Ponder: {metrics['ponder'].item():.4f} "
                    f"(lambda={metrics['lambda_ponder_now']:.2e}) | "
                    f"loop_scale: [{loop_scale}] | p_halt: {p_halt_mean:.4f} | "
                    f"p_max: {_metric('p_max'):.4f} | top1_acc: {_metric('top1_acc'):.4f} | "
                    f"per-loop CE: [{per_loop_ce}] | "
                    f"LR: {scheduler.get_last_lr()[0]:.3e} | SFT tokens: {sft_tokens / 1e6:.2f}M | "
                    f"Tokens/sec: {tokens_per_sec:.0f} | "
                    f"Peak Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | "
                    f"Time: {(now - timer) / 60:.2f} min"
                    + (f" | ETA: {format_duration(eta)}" if eta is not None else "")
                )

                write_status(
                    status_path, phase=SFT_PHASE, tokens=sft_tokens,
                    phase_target=target_tokens or 0, run_target=target_tokens or 0,
                    tokens_per_sec=tokens_per_sec, loss=val_loss,
                    eta_phase=format_duration(eta) if eta is not None else "n/a",
                    eta_run=format_duration(eta) if eta is not None else "n/a",
                    step=step, epoch=epoch,
                )

                if sft_tokens >= next_eval:
                    run_validation(epoch, step)
                    next_eval = sft_tokens + SFTConfig.eval_every_tokens

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
                    next_checkpoint = sft_tokens + SFTConfig.checkpoint_every_tokens

            if stop_training:
                break

            # end of epoch: the next one starts its own permutation from position 0
            start_doc_idx, step_offset = 0, 0
            resumed = False
            worker_state.fill_(-1)
            logger.info(f"Epoch {epoch} finished at {sft_tokens / 1e6:.2f}M SFT tokens")

        if stop_training:
            # a stop is not a finished run: no final checkpoint, and a restartable exit code so a
            # wrapper knows to relaunch (the rolling checkpoint just written is the resume point)
            return exit_code

        token_count = unwrapped_model._token_tracker.sync()
        run_validation(SFTConfig.num_epochs - 1, step)
        save_and_sync(SFTConfig.num_epochs - 1, step, losses[-1] if losses else float("nan"),
                      token_count, final=True)
        logger.info(
            f"SFT complete: {(token_count - start_token_count) / 1e6:.2f}M tokens over "
            f"{SFTConfig.num_epochs} epochs in {(time.time() - timer) / 60:.1f} min"
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
    parser = argparse.ArgumentParser(description="supervised fine-tuning (PLAN.md Step 12)")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="pretrained checkpoint to initialize from (ignored when resuming an "
                             "SFT run from ckpts/sft)")
    parser.add_argument("--from-hub", action="store_true",
                        help="download the pretrained checkpoint and manifest.json from the "
                             "training mirror repo before starting")
    parser.add_argument("--hub-repo", default=None,
                        help="repo to pull from (default: config.yaml's training.hf_upload_repo)")
    parser.add_argument("--hub-file", default=DEFAULT_HUB_CHECKPOINT,
                        help=f"path within the repo (default: {DEFAULT_HUB_CHECKPOINT})")
    parser.add_argument("--upload-repo", default=None,
                        help="mirror SFT checkpoints to this repo ('' disables; default: "
                             "config.yaml's sft.hf_upload_repo)")
    return sft(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

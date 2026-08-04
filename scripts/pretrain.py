import os
import sys
import time
import math
import random
import warnings
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
from torch import optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from modules.model.transformer import TinyMoETransformer
from modules.data.dataset import Dataset
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.mtp import compute_mtp_loss
from config import ModelConfig, TrainingConfig
from utils import save_checkpoint, load_checkpoint, BASE_DIR, logger, BF16, TOKENIZER_DIR


from transformer_engine.common.recipe import Format, DelayedScaling, MXFP8BlockScaling, NVFP4BlockScaling
import transformer_engine.pytorch as te

fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
mxfp8_format = Format.E4M3  # E4M3 used everywhere
mxfp8_recipe = MXFP8BlockScaling(fp8_format=mxfp8_format)
nvfp4_recipe = NVFP4BlockScaling(
    disable_rht=False,
    disable_stochastic_rounding=False,
)

# Note: using NVFP4 for the router and MTP heads leads to issues with the backward pass (mainly the requirement of divisability)

# off on the 5090 (BF16); set USE_FP8=1 on the H100 rental to switch chosen_recipe to fp8_recipe
USE_LOW_PRECISION = os.environ.get("USE_FP8", "0") == "1"
chosen_recipe = fp8_recipe if USE_LOW_PRECISION else None

# dense (non-sparse) BF16 tensor-core peak TFLOPS, for the MFU estimate in the training log
# (PLAN.md's Step 11 budget decision). NVIDIA's marketing figures are 2:4-sparse (2x dense) --
# 209.5 TFLOPS here is the 5090's sparse 419 halved; 990 TFLOPS matches PLAN.md's own H100 SXM
# assumption. Matched by substring against torch.cuda.get_device_name(0); add entries here as
# needed rather than guessing a number for an unrecognized GPU.
GPU_BF16_PEAK_TFLOPS = {
    "RTX 5090": 209.5e12,
    "H100": 990e12,
}


def detect_gpu_peak_flops():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for key, peak in GPU_BF16_PEAK_TFLOPS.items():
        if key in name:
            return peak
    return None

def train_step(
    model: TinyMoETransformer,
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    labels: torch.Tensor,
    pad_mask: torch.Tensor,
    accelerator: Accelerator = None,
    optimizer: optim.Optimizer = None,
    scheduler: optim.lr_scheduler._LRScheduler = None,
    no_decay_master_pairs: list = None,
    collect_metrics: bool = False,
    n_loops: int = None,
):
    """One micro-batch: forward, loss, backward, and (on a sync step) clip + optimizer step.

    Args:
        collect_metrics: whether to gather the PLAN.md Step 7 instrumentation dict. The trainer
            only reads it inside its LOG_INTERVAL block, and the p_max/p_correct reductions run
            over every chunk's live logits (and again on the checkpoint recompute), so gathering
            them on the other 9 steps out of 10 is pure waste. ``metrics`` is None when False.
        n_loops: loop depth for this step (see sample_n_loops). None runs the configured depth.
            loop_ce_weights is truncated/rescaled to match, so the deepest loop actually run is
            always the one carrying weight 1.0 and holding the correctness head.
    """
    loop_ce_weights = (
        TrainingConfig.loop_ce_weights if n_loops is None else loop_ce_weights_for(n_loops)
    )
    # attribute access (has_mtp, lm heads) must go through the unwrapped module so this works
    # under DDP, where ``model`` is a wrapper without those attributes. The forward call still
    # goes through ``model`` so accelerate handles gradient sync.
    unwrapped = accelerator.unwrap_model(model)
    with accelerator.accumulate(model):
        with te.autocast(enabled=USE_LOW_PRECISION, recipe=chosen_recipe):
            if unwrapped.has_mtp:
                logits, aux_loss, p_halt, extra_token_outputs = model(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                **ModelConfig.Forward,
                return_aux_loss=True,
                return_hidden=True,
                n_loops=n_loops,
            )
            else:
                logits, aux_loss, p_halt = model(
                    input_ids=input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    **ModelConfig.Forward,
                    return_aux_loss=True,
                    return_hidden=True,
                    n_loops=n_loops,
                )
                extra_token_outputs = None

            out = compute_mtp_loss(
                logits,
                labels,
                mtp_outputs=extra_token_outputs,
                lm_head=unwrapped.mtp_head.lm_head if extra_token_outputs is not None else None,
                lambda_mtp=TrainingConfig.lambda_mtp,
                main_lm_head=unwrapped.lm_head,
                pad_mask=pad_mask,
                loop_ce_weights=loop_ce_weights,
                loop_ce_subsample=TrainingConfig.loop_ce_subsample,
                correct_proj=unwrapped.correct_proj,
                lambda_conf=TrainingConfig.lambda_conf,
                return_metrics=collect_metrics,
            )
            loss, loss_ce, metrics = out if collect_metrics else (out[0], out[1], None)

            loss = loss + TrainingConfig.aux_loss_weight * aux_loss

            # ponder loss: penalize not-halting on real tokens, ramped from 0 so it can't deadlock
            # loop_scale before the loop has learned to do anything (see loop_scale's docstring).
            tokens = unwrapped.token_count
            warm, ramp = TrainingConfig.ponder_warmup_tokens, TrainingConfig.ponder_ramp_tokens
            lambda_ponder_now = TrainingConfig.lambda_ponder * min(1.0, max(0.0, (tokens - warm) / ramp))
            # p_halt is [n_loops, B, S] but valid_mask is [B, S], so the denominator has to carry
            # the loop axis too -- normalizing by valid_mask.sum() alone made both the ponder term
            # and the logged p_halt exactly n_loops times too large (p_halt could never read below
            # n_loops * sigmoid(halt_bias), and lambda_ponder was silently n_loops x its config
            # value).
            valid_mask = (~pad_mask).to(p_halt.dtype)
            valid_count = valid_mask.sum().clamp(min=1) * p_halt.size(0)
            ponder = ((1.0 - p_halt) * valid_mask).sum() / valid_count
            loss = loss + lambda_ponder_now * ponder

            # instrumentation (PLAN.md Step 7): all tensors, no host sync here -- the caller only
            # pulls these to the host inside its own LOG_INTERVAL-throttled block.
            if collect_metrics:
                metrics["p_halt_mean"] = ((p_halt * valid_mask).sum() / valid_count).detach()
                metrics["ponder"] = ponder.detach()
                metrics["lambda_ponder_now"] = lambda_ponder_now

            accelerator.backward(loss)
            # clip on the real update step only (matters once gradient accumulation > 1).
            # accelerator.clip_grad_norm_ unscales/handles the wrapped params correctly.
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
                # see build_param_groups: the no_decay group is optimized via fp32 masters
                # because bf16-native AdamW steps on these silently round to zero otherwise.
                if no_decay_master_pairs is not None:
                    sync_master_grads_(no_decay_master_pairs)
            optimizer.step()
            if accelerator.sync_gradients and no_decay_master_pairs is not None:
                sync_master_values_(no_decay_master_pairs)
            optimizer.zero_grad(set_to_none=True)

    if scheduler is not None and accelerator.sync_gradients:
        scheduler.step()

    return loss, loss_ce, aux_loss.detach(), metrics

def sample_n_loops(rng: random.Random, full_n_loops: int, prob: float) -> int:
    """Pick this step's loop depth (PLAN.md Step 3's "loops refine, bounded" goal).

    With probability ``prob`` the step runs a uniformly random *reduced* depth in
    ``1..full_n_loops-1``; otherwise the full depth. Keeping most steps at full depth matters --
    that's the primary operating point, and a plain uniform draw would give it only
    ``1/full_n_loops`` of the training signal.

    Why this rather than a ponder/halt penalty: p_halt gates the loop's *output* while every expert
    still runs, so penalizing it buys no compute back and only pushes the loop toward being a no-op.
    Sampling the depth instead makes each depth a genuine operating point, which is what an
    inference-time loop-count override actually needs, and it *reduces* mean training compute.
    """
    if prob <= 0.0 or full_n_loops <= 1 or rng.random() >= prob:
        return full_n_loops
    return rng.randint(1, full_n_loops - 1)


def loop_ce_weights_for(n_loops: int):
    """``loop_ce_weights`` truncated to ``n_loops``, rescaled so the deepest loop run carries 1.0.

    Truncating alone would shrink the CE term (running 1 of 3 loops would weight the whole main
    loss by 0.2), which is an unintended per-step learning-rate change on shallow steps. Rescaling
    keeps the deepest readout at full weight and preserves the relative ordering of the rest.
    """
    weights = TrainingConfig.loop_ce_weights[:n_loops]
    last = weights[-1]
    return [w / last for w in weights] if last > 0 else weights


def build_param_groups(model: TinyMoETransformer, weight_decay: float):
    """Split parameters into decayed / non-decayed groups.

    Decaying every parameter uniformly is actively harmful here, not just untidy: the architecture
    leans on several learned scalars whose *zero* is a degenerate state.
      * ``moe.loop_scale`` gates how much each loop contributes -- decaying it toward 0 decays the
        whole MoE block toward "off", which is the exact failure mode the ponder warmup exists to
        avoid.
      * ``Gemma4TextDecoderLayer.layer_scalar`` is a gain on the *whole* residual stream, so its
        decay compounds across depth: at lr=4e-4/wd=0.02 over ~9.5k steps it is a ~0.93x pull per
        layer, i.e. ~0.5x across 8 layers, before any gradient signal.
      * RMSNorm gains and biases (incl. ``halt_proj.bias``, which sets the p_halt operating point)
        have the same problem in milder form.
    The usual convention -- decay only tensors with ndim >= 2 -- covers all of these, since every
    one of them is a scalar or a vector.

    The no_decay group additionally needs fp32 shadow "master" weights, for a reason unrelated to
    weight decay: the model trains in native bf16 (``model.to(BF16)``, no fp32 master copy), and a
    steady-state AdamW step has magnitude ~lr (~1e-4-1e-3 here). Every no_decay tensor sits at O(0.1-2)
    magnitude (loop_scale ~1/sqrt(n_loops), layer_scalar/RMSNorm gains ~1, halt_proj.bias -2.0), where
    bf16's ulp (~0.4% of magnitude) is 10-40x bigger than that step -- so `param -= lr * update`
    rounds to EXACTLY the original bf16 value, every single step, forever, regardless of how much
    Adam momentum accumulates. Confirmed on a real checkpoint: after 344 optimizer steps loop_scale's
    exp_avg was large and nonzero while the bf16 value hadn't moved a single representable increment
    (matches Gate 4 failing -- p_halt pinned, loop_scale stuck at init). decay tensors don't have this
    problem in practice: their values and needed steps both scale with their own (much smaller) init
    std, so they stay inside bf16's relative resolution.
    The fix: AdamW steps the fp32 masters (never sub-ulp at fp32 precision), and the bf16 model
    parameter actually read by the forward pass is refreshed from its master after every real step
    (see sync_master_grads_/sync_master_values_ below). Returns the master pairs so the caller can
    drive that sync and re-seed the masters from a resumed checkpoint's bf16 weights.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)
    no_decay_masters = [p.detach().clone().float().requires_grad_(False) for p in no_decay]
    logger.info(
        f"Optimizer param groups: {len(decay)} decayed tensors (wd={weight_decay}), "
        f"{len(no_decay)} undecayed (norms/biases/gates) stepped via fp32 masters"
    )
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay_masters, "weight_decay": 0.0},
    ]
    return param_groups, list(zip(no_decay, no_decay_masters))


def sync_master_grads_(master_pairs):
    """Copy each bf16 model param's (post-clip) .grad into its fp32 master's .grad, in place.

    Call once, right before the real ``optimizer.step()`` (i.e. gated on
    ``accelerator.sync_gradients``, matching where grad clipping already runs) -- the bf16 param's
    .grad is the fully accumulated, clipped gradient at that point; the master needs exactly that,
    not a running sum of its own, so this overwrites rather than accumulates.
    """
    with torch.no_grad():
        for bf16_param, master in master_pairs:
            if bf16_param.grad is None:
                continue
            if master.grad is None:
                master.grad = bf16_param.grad.detach().float()
            else:
                master.grad.copy_(bf16_param.grad.detach())


def sync_master_values_(master_pairs):
    """Refresh each bf16 model param from its just-stepped fp32 master, in place.

    Call once, right after the real ``optimizer.step()`` (same gating as sync_master_grads_) --
    this is what makes the fp32 update actually visible to the forward pass, at bf16 precision
    (an ordinary, correctly-rounded fp32->bf16 cast, not the sub-ulp-discarding in-place bf16 add
    this whole mechanism exists to avoid).
    """
    with torch.no_grad():
        for bf16_param, master in master_pairs:
            bf16_param.data.copy_(master.data.to(bf16_param.dtype))


def build_scheduler(optimizer: optim.Optimizer):
    # linear warmup -> cosine decay, adjusted for the current total steps
    warmup_steps = min(TrainingConfig.warmup_steps, max(TrainingConfig.total_steps - 1, 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(TrainingConfig.total_steps - warmup_steps, 1),
        eta_min=TrainingConfig.lr * 0.1,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )

def get_latest_checkpoint_epoch(checkpoint_dir: str):
    last_timestamp = 0
    cur_file = None
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint") and filename.endswith(".pt"):
            os_timestamp = os.path.getmtime(os.path.join(checkpoint_dir, filename))
            if os_timestamp > last_timestamp:
                last_timestamp = os_timestamp
                cur_file = filename
    return cur_file

def checkpoint_name(epoch: int, dataset_idx: int, loss: float, interrupted=False):
    if interrupted:
        return f"checkpoint_epoch{epoch}_idx{dataset_idx}_loss{loss:.4f}_interrupted.pt"
    return f"checkpoint_epoch{epoch}_idx{dataset_idx}_loss{loss:.4f}.pt"


def dry_run(model: TinyMoETransformer, device="cuda", dtype=BF16, config=ModelConfig):
    model.train().to(device).to(dtype)
    # the dry run must not pollute the (possibly checkpoint-restored) token counter
    token_count_before = model._token_tracker.num_tokens
    B, S = TrainingConfig.Batch_size, TrainingConfig.Seq_length
    input_ids = torch.randint(0, config.Params["vocab_size"], (B, S)).to(device)

    # exercise the packed path exactly as training does: build document_ids (two docs per sample,
    # the second ending in a few trailing-pad length-1 segments) and derive cu_seqlens from them.
    pad_id = config.Params.get("pad_token_id", 0)
    input_ids[:, S - 4:] = pad_id
    half = S // 2
    row = [0] * half + [1] * (half - 4) + [2, 3, 4, 5]  # two blocks + 4 trailing length-1 segments
    document_ids = torch.tensor([row] * B, dtype=torch.long, device=device)
    cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
    pad_mask = (input_ids == pad_id)

    with te.autocast(enabled=USE_LOW_PRECISION, recipe=chosen_recipe):
        logits, aux_loss, p_halt, extra_token_outputs = model(
            input_ids=input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **ModelConfig.Forward,
            return_aux_loss=True,
            return_hidden=True,
        )
        loss, loss_ce = compute_mtp_loss(
            logits,
            input_ids,
            mtp_outputs=extra_token_outputs,
            lm_head=model.mtp_head.lm_head if extra_token_outputs is not None else None,
            lambda_mtp=TrainingConfig.lambda_mtp,
            main_lm_head=model.lm_head,
            pad_mask=pad_mask,
            loop_ce_weights=TrainingConfig.loop_ce_weights,
            loop_ce_subsample=TrainingConfig.loop_ce_subsample,
            correct_proj=model.correct_proj,
            lambda_conf=TrainingConfig.lambda_conf,
        )
        
        loss = loss + TrainingConfig.aux_loss_weight * aux_loss

        loss.backward()
        model.zero_grad(set_to_none=True)

    model._token_tracker.num_tokens = token_count_before
    if not torch.isfinite(loss):
        raise RuntimeError(f"dry_run produced non-finite loss: {loss.item()}")

def save_expert_selection_graph(stats, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    expert_counts = stats["choices"]
    expert_probs = stats["prob_dist"]
    post_skew_probs = stats["post_skew_dist"]

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    # tracker stats are per-token EMAs now: fractions/mean weights in [0, 1]
    axs[0].bar(range(len(expert_counts)), expert_counts)
    axs[0].set_title("Expert Selection Fraction (EMA)")
    axs[0].set_xlabel("Expert Index")
    axs[0].set_ylabel("Fraction of Tokens")

    axs[1].bar(range(len(expert_probs)), expert_probs)
    axs[1].set_title("Mean Routed Weight per Token (EMA)")
    axs[1].set_xlabel("Expert Index")
    axs[1].set_ylabel("Mean Weight")

    axs[2].bar(range(len(post_skew_probs)), post_skew_probs)
    axs[2].set_title("Mean Post-Skew Probability When Selected (EMA)")
    axs[2].set_xlabel("Expert Index")
    axs[2].set_ylabel("Mean Probability")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    del fig, axs, plt


def save_loss_graph(losses, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()

def pretrain():
    # TOKENIZER_DIR is the pruned 65536 tokenizer (PLAN.md Step 8) -- fits uint16, matches
    # config.yaml's vocab_size. shared with every other entry point via utils, and fetched onto a
    # fresh box by scripts/fetch_tokenizer.py since ckpts/ is gitignored.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    logger.info(f"Tokenizer loaded from {TOKENIZER_DIR} with vocab size {tokenizer.vocab_size}")
    
    dataset = Dataset(
        data_dir=os.path.join(BASE_DIR, TrainingConfig.data_dir),
        tokenizer=tokenizer,
        batch_size=TrainingConfig.Batch_size,
        max_length=TrainingConfig.Seq_length,
        split=TrainingConfig.phase,
        num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
    )
    # must stay the same across checkpoints
    NUM_DATA_WORKERS = 4
    dataloader = DataLoader(dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, prefetch_factor=2)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    gpu_peak_flops = detect_gpu_peak_flops()
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        if gpu_peak_flops is not None:
            logger.info(f"GPU: {gpu_name} | assumed BF16 dense peak: {gpu_peak_flops / 1e12:.1f} TFLOPS (MFU will be logged)")
        else:
            logger.warning(f"GPU: {gpu_name} not in GPU_BF16_PEAK_TFLOPS -- MFU will be logged as n/a, add an entry to enable it")

    model = TinyMoETransformer(**ModelConfig.Params).to(device).to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    # count only real (non-pad) tokens towards the trained-token total
    model._token_tracker.pad_token_id = tokenizer.pad_token_id
    
    # model = torch.compile(model)
    # from bitsandbytes.optim import AdamW8bit
    
    param_groups, no_decay_master_pairs = build_param_groups(model, TrainingConfig.weight_decay)
    optimizer = optim.AdamW(
        param_groups, lr=TrainingConfig.lr,
    )
    scheduler = build_scheduler(optimizer)

    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Scheduler total steps: {TrainingConfig.total_steps:,}")
    
    start_epoch, dataset_idx, start_doc_idx = 0, 0, 0
    resumed = False
    checkpoint_dir = os.path.join(BASE_DIR, "ckpts", "training")
    losses = []
    # "no checkpoint exists" and "a checkpoint exists but would not load" are NOT the same event.
    # Swallowing both into one warning is how an unattended, interruptible run silently restarts
    # from token 0 after e.g. a torch.save truncated by a preemption or a config change that
    # reshapes a weight -- the loss curve looks plausible and days of compute are already gone.
    # Only the first case is recoverable; the second re-raises.
    latest = get_latest_checkpoint_epoch(checkpoint_dir) if os.path.isdir(checkpoint_dir) else None
    if latest is None:
        logger.warning("No checkpoint found in ckpts/training. Starting training from scratch")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, latest)
        try:
            start_epoch, dataset_idx, resume_token_count, start_doc_idx, losses = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        except Exception as e:
            logger.error(
                f"Found checkpoint {checkpoint_path} but failed to load it: {type(e).__name__}: {e}. "
                f"Refusing to silently restart from scratch -- move or delete the file to start over."
            )
            raise
        model._token_tracker.num_tokens = resume_token_count
        resumed = True

        # the fp32 masters aren't part of model_state_dict (they're optimizer-only shadows built
        # at optimizer-construction time, before this resume) -- reseed them from the just-loaded
        # bf16 weights so the no_decay group keeps moving from where this checkpoint left off,
        # rather than from the pre-resume random init they were cloned from. Adam's exp_avg/
        # exp_avg_sq for the masters already came back correctly via optimizer.load_state_dict
        # above (restored by param-group position, not by identity, so unaffected by this copy).
        with torch.no_grad():
            for bf16_param, master in no_decay_master_pairs:
                master.data.copy_(bf16_param.data.float())

        # total_steps moves with batch size / grad accum, so reanchor the LR schedule by tokens
        tokens_per_step = TrainingConfig.Batch_size * TrainingConfig.Seq_length * TrainingConfig.grad_accumulation_steps
        resumed_sched_step = min(resume_token_count // tokens_per_step, TrainingConfig.total_steps)
        scheduler = build_scheduler(optimizer)
        with warnings.catch_warnings():
            # silence the "step() called before optimizer.step()" warning during fast forward
            warnings.simplefilter("ignore")
            for _ in range(resumed_sched_step):
                scheduler.step()
        logger.info(
            f"Reanchored LR scheduler to step {resumed_sched_step:,}/{TrainingConfig.total_steps:,} "
            f"({resume_token_count / 1e9:.3f}B tokens trained, current LR {scheduler.get_last_lr()[0]:.3e})"
        )

    logger.info(f"Starting training from epoch {start_epoch}, document index {start_doc_idx}")
    
    dry_run(model, device=device, dtype=BF16, config=ModelConfig)
    logger.info("Dry run successful. Starting training loop...")
    
    # setup accelerator
    accelerator = Accelerator(
        device_placement=True, 
        split_batches=True, 
        gradient_accumulation_steps=TrainingConfig.grad_accumulation_steps,
    )
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    unwrapped_model = accelerator.unwrap_model(model)

    timer = time.time()
    last_token_count = unwrapped_model._token_tracker.sync()
    last_log_time = timer

    # MFU accounting (PLAN.md Step 11's budget number). Separate accumulators because the pieces
    # scale differently, and none of them is just "tokens":
    #   * positions -- FLOPs are spent on padding too, and the token counter deliberately excludes
    #     pads. A plain python int from .numel(), so no sync.
    #   * sum(segment_len^2) -- attention cost scales with this, not with token count, and under
    #     document packing it is far below B*S^2. Accumulated on device, drained once per log
    #     interval alongside the existing syncs.
    #   * the *_loops variants weight each step by the depth it actually ran, since loop-count
    #     sampling means that is no longer the constant n_loops.
    positions_since_log = 0
    loop_positions_since_log = 0
    lm_head_passes_since_log = 0.0
    seg_sq_since_log = torch.zeros((), dtype=torch.long, device=device)
    loop_seg_sq_since_log = torch.zeros((), dtype=torch.long, device=device)

    # stochastic loop depth (see sample_n_loops). Its own RNG so it can't perturb the model's
    # init/dropout/router-noise stream, and so a config change here doesn't reshuffle those.
    full_n_loops = ModelConfig.Params["n_loops"]
    loop_rng = random.Random(TrainingConfig.seed)
    if TrainingConfig.loop_count_sampling > 0:
        logger.info(
            f"Loop-count sampling on: p={TrainingConfig.loop_count_sampling} of steps run a random "
            f"depth in 1..{full_n_loops - 1}, the rest run the full {full_n_loops}"
        )

    # log/sync cadence: pulling loss/aux/token-count to the host forces CPU/GPU syncs, so do it
    # every LOG_INTERVAL steps instead of every step (the model is small -> per-step syncs dominate)
    LOG_INTERVAL = 20

    # per worker, last doc_idx that worker had reached, kept on device; -1 = nothing seen yet
    worker_state = torch.full((max(NUM_DATA_WORKERS, 1),), -1, dtype=torch.long, device=device)

    def snapshot_global_offset():
        # only host sync of this state, done at checkpoint time. conservative min across workers:
        # a worker at doc_idx d will next want d + NUM_DATA_WORKERS (PLAN.md Step 9 sharding), so
        # taking the min "next wanted" doc guarantees no worker's unconsumed docs are skipped --
        # workers further ahead just redo a few already-seen docs, which is harmless.
        rows = worker_state.cpu().tolist()
        seen = [r for r in rows if r >= 0]
        if not seen:
            return start_doc_idx
        return min(seen) + NUM_DATA_WORKERS

    stop_training = False
    try:
        for epoch in range(start_epoch, TrainingConfig.num_epochs):
            resume_epoch = resumed and epoch == start_epoch
            dataset.start_doc_idx = start_doc_idx if resume_epoch else 0
            if not resume_epoch:
                worker_state.fill_(-1)
            # continue the step counter from the checkpointed step on the resumed epoch
            step_offset = dataset_idx if resume_epoch else 0

            for local_step, batch in enumerate(dataloader):
                step = local_step + step_offset
                # record this workers progress, no host sync
                w_id = batch["worker_id"][0].to(worker_state.device)
                worker_state[w_id] = batch["doc_idx"][0].to(worker_state.device)

                input_ids = batch["input_ids"].to(device)
                # document_ids: batch-aligned [B, S] segment ids for the packed documents. cu_seqlens
                # is built from them HERE (in-thread) and never carried in the batch: a ragged
                # cu_seqlens (dim0 = num_segments+1) gets truncated to the batch size by accelerates
                # split_batches, which silently corrupts the attention segmentation.
                document_ids = batch["document_ids"].to(device) if batch["document_ids"] is not None else None
                # this step's loop depth. Log steps are pinned to the full depth so the recorded
                # loss curve, per-loop CE and p_halt are always read at the same operating point --
                # otherwise `losses` (plotted and checkpointed) would jump whenever a logged step
                # happened to draw a shallower depth. Costs ~1/LOG_INTERVAL of the sampling rate.
                is_log_step = step % LOG_INTERVAL == 0
                step_n_loops = full_n_loops if is_log_step else sample_n_loops(
                    loop_rng, full_n_loops, TrainingConfig.loop_count_sampling
                )

                if document_ids is not None:
                    cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
                    # attention FLOPs scale with sum(segment_len^2); accumulate on device (no sync)
                    seg = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
                    step_seg_sq = (seg * seg).sum()
                else:
                    cu_seqlens, max_seqlen = None, None
                    # unpacked path: one full-length causal segment per sample
                    step_seg_sq = torch.as_tensor(
                        input_ids.size(0) * input_ids.size(1) ** 2, dtype=torch.long, device=device
                    )
                seg_sq_since_log += step_seg_sq
                loop_seg_sq_since_log += step_seg_sq * step_n_loops
                positions_since_log += input_ids.numel()
                loop_positions_since_log += input_ids.numel() * step_n_loops
                # lm_head runs once per supervised loop; non-final loops are token-subsampled
                # (Step 4a + loop_ce_subsample), so they cost a fraction of a full pass each
                lm_head_passes_since_log += input_ids.numel() * (
                    1.0 + (step_n_loops - 1) * TrainingConfig.loop_ce_subsample
                )

                if (device == "cuda") and (step % 20 == 0):
                    torch.cuda.reset_peak_memory_stats()

                pad_mask = (input_ids == tokenizer.pad_token_id)

                # anneal router exploration noise 1 -> 0 over the first noise_anneal_tokens tokens
                if TrainingConfig.noise_anneal_tokens > 0:
                    noise_factor = max(0.0, 1.0 - unwrapped_model.token_count / TrainingConfig.noise_anneal_tokens)
                    unwrapped_model.moe.set_router_noise(noise_factor)

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
                    no_decay_master_pairs=no_decay_master_pairs,
                    # gathered only on the steps the log block below actually reads them
                    collect_metrics=is_log_step,
                    n_loops=step_n_loops,
                )


                dataset_idx = step

                # logging pulls loss/aux/token-count/metrics to the host (syncs). Throttle to
                # LOG_INTERVAL -- everything read here (metrics dict, loop_scale) stayed a tensor
                # until this point, so this is the only sync, same cadence as the existing ones.
                if step % LOG_INTERVAL == 0:
                    val_loss = loss.item()
                    losses.append(val_loss)
                    n_tokens = unwrapped_model._token_tracker.sync()
                    now = time.time()
                    interval_s = max(now - last_log_time, 1e-6)
                    tokens_per_sec = (n_tokens - last_token_count) / interval_s
                    last_token_count = n_tokens
                    last_log_time = now

                    # MFU over the interval just elapsed. Built from three separately-scaled
                    # pieces (see TinyMoETransformer's FLOP block) rather than one per-token
                    # constant:
                    #   * body matmuls: 3x (fwd + bwd; activation checkpointing is off)
                    #   * heads: 4x -- compute_mtp_loss chunk-checkpoints every LM-head projection,
                    #     so each also pays a recompute, and lm_head runs once per supervised loop
                    #   * attention: keyed to the sum(segment_len^2) actually seen, since document
                    #     packing puts real attention cost well below the dense B*S^2
                    # n/a on an unrecognized GPU rather than a silently wrong number.
                    # one sync for both depth-weighted attention accumulators
                    seg_sq, loop_seg_sq = torch.stack(
                        [seg_sq_since_log, loop_seg_sq_since_log]
                    ).tolist()
                    seg_sq_since_log.zero_()
                    loop_seg_sq_since_log.zero_()
                    if gpu_peak_flops is not None and positions_since_log > 0:
                        m = unwrapped_model
                        body_flops = (
                            m.dense_flops_per_token * positions_since_log
                            + m.loop_flops_per_token * loop_positions_since_log
                            + m.dense_attn_flops_per_seqsq * seg_sq
                            + m.loop_attn_flops_per_seqsq * loop_seg_sq
                        )
                        head_flops = (
                            m.lm_head_flops_per_token * lm_head_passes_since_log
                            + m.mtp_flops_per_token * positions_since_log
                        )
                        interval_flops = 3 * body_flops + 4 * head_flops
                        mfu_str = f"{100 * interval_flops / interval_s / gpu_peak_flops:.1f}%"
                        mean_loops = loop_positions_since_log / positions_since_log
                    else:
                        mfu_str, mean_loops = "n/a", float("nan")
                    positions_since_log = 0
                    loop_positions_since_log = 0
                    lm_head_passes_since_log = 0.0

                    # PLAN.md Step 7 instrumentation: loop_scale, halt/correctness/confidence
                    # signals, and per-loop CE, all through unwrap_model per the training-loop rule.
                    # loop_scale is per-loop now (one gate per loop), so log the whole vector --
                    # "loop 3 grew, loops 1-2 collapsed" is exactly the failure a single mean hides
                    loop_scale_str = ", ".join(f"{s:.4f}" for s in unwrapped_model.moe.loop_scale.tolist())
                    per_loop_ce_str = ", ".join(f"{ce.item():.4f}" for ce in metrics["per_loop_ce"])
                    p_halt_mean = metrics["p_halt_mean"].item()
                    ponder_val = metrics["ponder"].item()
                    conf_loss_val = metrics["conf_loss"].item() if metrics["conf_loss"] is not None else float("nan")
                    p_correct_val = metrics["p_correct"].item() if metrics["p_correct"] is not None else float("nan")
                    p_max_val = metrics["p_max"].item() if metrics["p_max"] is not None else float("nan")
                    top1_val = metrics["top1_acc"].item() if metrics["top1_acc"] is not None else float("nan")

                    logger.info(
                        f"Epoch {epoch} | Step {step} | Loss: {val_loss:.4f} | Loss (CE): {loss_ce.item():.4f} | "
                        f"Aux Loss: {aux_loss.item():.4f} | Ponder: {ponder_val:.4f} (lambda={metrics['lambda_ponder_now']:.2e}) | "
                        f"Conf Loss: {conf_loss_val:.4f} | loop_scale: [{loop_scale_str}] | p_halt: {p_halt_mean:.4f} | "
                        f"p_correct: {p_correct_val:.4f} | p_max: {p_max_val:.4f} | top1_acc: {top1_val:.4f} | "
                        f"per-loop CE: [{per_loop_ce_str}] | mean loops: {mean_loops:.2f} | "
                        f"Tokens: {n_tokens / 1e6:.2f}M | Tokens/sec: {tokens_per_sec:.2f} | "
                        f"MFU: {mfu_str} | Peak Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | Time: {(now - timer) / 60:.2f} min"
                    )

                    # stop at the token budget the LR schedule is anchored to (PLAN.md Step 6) --
                    # checked at this same sync-free-until-now cadence, not per step. Without this,
                    # a run with more data than target_tokens keeps training past the schedule at
                    # eta_min instead of stopping where the cosine decay was anchored.
                    if n_tokens >= TrainingConfig.target_tokens:
                        logger.info(f"Reached target_tokens ({TrainingConfig.target_tokens:,}); saving final checkpoint and stopping.")
                        save_checkpoint(
                            unwrapped_model,
                            optimizer,
                            scheduler,
                            epoch,
                            dataset_idx,
                            path=os.path.join(checkpoint_dir, checkpoint_name(epoch, dataset_idx, loss)),
                            token_count=n_tokens,
                            global_offset=snapshot_global_offset(),
                            losses=losses,
                        )
                        stop_training = True
                        break

                if step % unwrapped_model.moe.expert_tracker.sliding_window_size == 0:
                    stats = unwrapped_model.moe.expert_tracker.get_stats()
                    try:
                        save_expert_selection_graph(stats, os.path.join(BASE_DIR, "ckpts", "training", f"expert_selection_epoch{epoch}_step{step}.png"))
                    except Exception as e:
                        logger.error(f"Error occurred while saving expert selection graph: {e}")
                    try:
                        save_loss_graph(losses, os.path.join(BASE_DIR, "ckpts", "training", f"loss_graph_epoch{epoch}_step{step}.png"))
                    except Exception as e:
                        logger.error(f"Error occurred while saving loss graph: {e}")

                # save checkpoint every 1500 steps (was 5000, PLAN.md Step 6 -- makes the run
                # interruptible at a granularity that matches the vast.ai preemptible-instance flow)
                if (step % 1500 == 0) and (local_step > 0):
                    save_checkpoint(
                        unwrapped_model,
                        optimizer,
                        scheduler,
                        epoch,
                        dataset_idx,
                        path=os.path.join(checkpoint_dir, checkpoint_name(epoch, dataset_idx, loss)),
                        token_count=unwrapped_model._token_tracker.sync(),
                        global_offset=snapshot_global_offset(),
                        losses=losses,
                    )
            dataset_idx = 0
            if stop_training:
                break
    except KeyboardInterrupt:
        try:
            input("Training interrupted. Press Enter to save checkpoint and exit...")
            logger.info("Training interrupted. Saving checkpoint...")
            save_checkpoint(
                unwrapped_model,
                optimizer,
                scheduler,
                epoch,
                dataset_idx,
                path=os.path.join(checkpoint_dir, checkpoint_name(epoch, dataset_idx, loss, interrupted=True)),
                token_count=unwrapped_model._token_tracker.sync(),
                global_offset=snapshot_global_offset(),
                losses=losses,
            )
        except Exception as e:
            logger.error(f"Exiting with: {e}")



if __name__ == "__main__":
    pretrain()




import os
import sys
import time
import math
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
from utils import save_checkpoint, load_checkpoint, BASE_DIR, logger, BF16


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
):
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
            )
            else:
                logits, aux_loss, p_halt = model(
                    input_ids=input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    **ModelConfig.Forward,
                    return_aux_loss=True,
                    return_hidden=True,
                )
                extra_token_outputs = None

            loss, loss_ce, metrics = compute_mtp_loss(
                logits,
                labels,
                mtp_outputs=extra_token_outputs,
                lm_head=unwrapped.mtp_head.lm_head if extra_token_outputs is not None else None,
                lambda_mtp=TrainingConfig.lambda_mtp,
                main_lm_head=unwrapped.lm_head,
                pad_mask=pad_mask,
                loop_ce_weights=TrainingConfig.loop_ce_weights,
                correct_proj=unwrapped.correct_proj,
                lambda_conf=TrainingConfig.lambda_conf,
                return_metrics=True,
            )

            loss = loss + TrainingConfig.aux_loss_weight * aux_loss

            # ponder loss: penalize not-halting on real tokens, ramped from 0 so it can't deadlock
            # loop_scale before the loop has learned to do anything (see loop_scale's docstring).
            tokens = unwrapped.token_count
            warm, ramp = TrainingConfig.ponder_warmup_tokens, TrainingConfig.ponder_ramp_tokens
            lambda_ponder_now = TrainingConfig.lambda_ponder * min(1.0, max(0.0, (tokens - warm) / ramp))
            valid_mask = (~pad_mask).to(p_halt.dtype)
            ponder = ((1.0 - p_halt) * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            loss = loss + lambda_ponder_now * ponder

            # instrumentation (PLAN.md Step 7): all tensors, no host sync here -- the caller only
            # pulls these to the host inside its own LOG_INTERVAL-throttled block.
            metrics["p_halt_mean"] = ((p_halt * valid_mask).sum() / valid_mask.sum().clamp(min=1)).detach()
            metrics["ponder"] = ponder.detach()
            metrics["lambda_ponder_now"] = lambda_ponder_now

            accelerator.backward(loss)
            # clip on the real update step only (matters once gradient accumulation > 1).
            # accelerator.clip_grad_norm_ unscales/handles the wrapped params correctly.
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if scheduler is not None and accelerator.sync_gradients:
        scheduler.step()

    return loss, loss_ce, aux_loss.detach(), metrics

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
    GEMMA4_TOKENIZER_PATH = os.path.join(BASE_DIR, "ckpts", "pretrained", "DeepSeek-V4-Pro-tokenizer-65536") # pruned to fit uint16 (PLAN.md Step 8), matches config.yaml's vocab_size: 65536
    tokenizer = AutoTokenizer.from_pretrained(GEMMA4_TOKENIZER_PATH)
    
    logger.info(f"Tokenizer loaded from {GEMMA4_TOKENIZER_PATH} with vocab size {tokenizer.vocab_size}")
    
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
    
    model = TinyMoETransformer(**ModelConfig.Params).to(device).to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    # count only real (non-pad) tokens towards the trained-token total
    model._token_tracker.pad_token_id = tokenizer.pad_token_id
    
    # model = torch.compile(model)
    # from bitsandbytes.optim import AdamW8bit
    
    optimizer = optim.AdamW(model.parameters(), lr=TrainingConfig.lr, weight_decay=TrainingConfig.weight_decay)
    scheduler = build_scheduler(optimizer)

    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Scheduler total steps: {TrainingConfig.total_steps:,}")
    
    start_epoch, dataset_idx, start_doc_idx = 0, 0, 0
    resumed = False
    checkpoint_dir = os.path.join(BASE_DIR, "ckpts", "training")
    losses = []
    try:
        checkpoint_path = os.path.join(checkpoint_dir, get_latest_checkpoint_epoch(checkpoint_dir))
        start_epoch, dataset_idx, resume_token_count, start_doc_idx, losses = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        model._token_tracker.num_tokens = resume_token_count
        resumed = True

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
    except Exception as e:
        logger.warning(f"No checkpoint found. Starting training from scratch")

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

    # log/sync cadence: pulling loss/aux/token-count to the host forces CPU/GPU syncs, so do it
    # every LOG_INTERVAL steps instead of every step (the model is small -> per-step syncs dominate)
    LOG_INTERVAL = 10

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
                if document_ids is not None:
                    cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
                else:
                    cu_seqlens, max_seqlen = None, None

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
                    tokens_per_sec = (n_tokens - last_token_count) / max(now - last_log_time, 1e-6)
                    last_token_count = n_tokens
                    last_log_time = now

                    # PLAN.md Step 7 instrumentation: loop_scale, halt/correctness/confidence
                    # signals, and per-loop CE, all through unwrap_model per the training-loop rule.
                    loop_scale = unwrapped_model.moe.loop_scale.item()
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
                        f"Conf Loss: {conf_loss_val:.4f} | loop_scale: {loop_scale:.4f} | p_halt: {p_halt_mean:.4f} | "
                        f"p_correct: {p_correct_val:.4f} | p_max: {p_max_val:.4f} | top1_acc: {top1_val:.4f} | "
                        f"per-loop CE: [{per_loop_ce_str}] | Tokens: {n_tokens / 1e6:.2f}M | Tokens/sec: {tokens_per_sec:.2f} | "
                        f"Peak Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | Time: {(now - timer) / 60:.2f} min"
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




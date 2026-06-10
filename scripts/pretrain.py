import os
import sys
import time
import math
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
    disable_rht=True, 
    disable_stochastic_rounding=True
)

chosen_recipe = nvfp4_recipe
# Note: using NVFP4 for the router and MTP heads leads to issues with the backward pass (mainly the requirement of divisability)

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
        with te.autocast(enabled=True, recipe=chosen_recipe):
            if unwrapped.has_mtp:
                logits, aux_loss, extra_token_outputs = model(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                **ModelConfig.Forward,
                return_aux_loss=True,
                return_hidden=True,
            )
            else:
                logits, aux_loss = model(
                    input_ids=input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    **ModelConfig.Forward,
                    return_aux_loss=True,
                    return_hidden=True,
                )
                extra_token_outputs = None

            loss = compute_mtp_loss(
                logits,
                labels,
                mtp_outputs=extra_token_outputs,
                lm_head=unwrapped.mtp_head.lm_head if extra_token_outputs is not None else None,
                lambda_mtp=TrainingConfig.lambda_mtp,
                main_lm_head=unwrapped.lm_head,
                pad_mask=pad_mask,
            ) + aux_loss

            accelerator.backward(loss)
            # clip on the real update step only (matters once gradient accumulation > 1).
            # accelerator.clip_grad_norm_ unscales/handles the wrapped params correctly.
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if scheduler is not None and accelerator.sync_gradients:
        scheduler.step()

    return loss

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

    with te.autocast(enabled=True, recipe=chosen_recipe):
        logits, aux_loss, extra_token_outputs = model(
            input_ids=input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **ModelConfig.Forward,
            return_aux_loss=True,
            return_hidden=True,
        )
        loss = compute_mtp_loss(
            logits,
            input_ids,
            mtp_outputs=extra_token_outputs,
            lm_head=model.mtp_head.lm_head if extra_token_outputs is not None else None,
            lambda_mtp=TrainingConfig.lambda_mtp,
            main_lm_head=model.lm_head,
            pad_mask=pad_mask,
        ) + aux_loss

        loss.backward()
        model.zero_grad(set_to_none=True)

    if not torch.isfinite(loss):
        raise RuntimeError(f"dry_run produced non-finite loss: {loss.item()}")

def save_expert_selection_graph(stats, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    expert_counts = stats["choices"]
    expert_probs = stats["prob_dist"]
    post_skew_probs = stats["post_skew_dist"]
    id_idx = stats["id_idx"]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    
    axs[0].bar(range(len(expert_counts)), expert_counts)
    axs[0].set_title("Expert Selection Counts")
    axs[0].set_xlabel("Expert Index")
    axs[0].set_ylabel("Count")
    
    axs[0].axvline(x=id_idx, color="red", linestyle="--", label="Identity Expert")
    axs[0].legend()
    
    axs[1].bar(range(len(expert_probs)), expert_probs)
    axs[1].set_title("Average Expert Selection Probabilities")
    axs[1].set_xlabel("Expert Index")
    axs[1].set_ylabel("Average Probability")
    
    axs[1].axvline(x=id_idx, color="red", linestyle="--", label="Identity Expert")
    axs[1].legend()
    
    axs[2].bar(range(len(post_skew_probs)), post_skew_probs)
    axs[2].set_title("Average Expert Probabilities After Skewing")
    axs[2].set_xlabel("Expert Index")
    axs[2].set_ylabel("Average Probability")
    
    axs[2].axvline(x=id_idx, color="red", linestyle="--", label="Identity Expert")
    axs[2].legend()
    
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
    GEMMA4_TOKENIZER_PATH = os.path.join(BASE_DIR, "ckpts", "pretrained", "DeepSeek-V4-Pro-tokenizer") # "gemma4-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(GEMMA4_TOKENIZER_PATH)
    
    logger.info(f"Tokenizer loaded from {GEMMA4_TOKENIZER_PATH} with vocab size {tokenizer.vocab_size}")
    
    dataset = Dataset(
        tokenizer=tokenizer,
        batch_size=TrainingConfig.Batch_size,
        max_length=TrainingConfig.Seq_length,
        mode="pretrain",
        config_path=os.path.join(BASE_DIR, "data_config.json"),
        num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=2, prefetch_factor=2)
    
    logger.info(f"Initialized dataset with {len(dataset.sources)} sources")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = TinyMoETransformer(**ModelConfig.Params).to(device).to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    
    # model = torch.compile(model)
    # from bitsandbytes.optim import AdamW8bit
    
    optimizer = optim.AdamW(model.parameters(), lr=TrainingConfig.lr, weight_decay=TrainingConfig.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainingConfig.total_steps - TrainingConfig.warmup_steps)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Scheduler total steps: {TrainingConfig.total_steps:,}, warmup steps: {TrainingConfig.warmup_steps}")
    
    start_epoch, dataset_idx = 0, 0
    checkpoint_dir = os.path.join(BASE_DIR, "ckpts", "training")
    try:
        checkpoint_path = os.path.join(checkpoint_dir, get_latest_checkpoint_epoch(checkpoint_dir))
        start_epoch, dataset_idx, resume_token_count = load_checkpoint(model, optimizer, checkpoint_path)
        model._token_tracker.num_tokens = resume_token_count
    except Exception as e:
        logger.warning(f"No checkpoint found. Starting training from scratch")

    logger.info(f"Starting training from epoch {start_epoch}, dataset index {dataset_idx}")
    
    dry_run(model, device=device, dtype=BF16, config=ModelConfig)
    logger.info("Dry run successful. Starting training loop...")
    
    # setup accelerator
    accelerator = Accelerator(
        device_placement=True, 
        split_batches=True, 
    )
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    unwrapped_model = accelerator.unwrap_model(model)

    timer = time.time()
    losses = []
    last_token_count = unwrapped_model.token_count
    
    try:
        for epoch in range(start_epoch, TrainingConfig.num_epochs):
            for step, batch in enumerate(dataloader):
                # skip steps until dataset_idx from checkpoint is reached
                if epoch == start_epoch and step < dataset_idx:
                    continue
                # dataset yields dummy batches (labels=None) during its fast-skip phase
                # guard here in case dataset and dataloader step counters drift
                if batch["labels"] is None:
                    continue

                input_ids = batch["input_ids"].to(device)
                # document_ids: batch-aligned [B, S] segment ids for the packed documents
                # build flash varlen cu_seqlens from them. None: plain causal
                document_ids = batch["document_ids"].to(device) if batch["document_ids"] is not None else None
                if document_ids is not None:
                    cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
                else:
                    cu_seqlens, max_seqlen = None, None

                if (device == "cuda") and (step % 20 == 0):
                    torch.cuda.reset_peak_memory_stats()

                pad_mask = (input_ids == tokenizer.pad_token_id)

                loss = train_step(
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
                    
                
                val_loss = loss.item()
                losses.append(val_loss)
                n_tokens = unwrapped_model.token_count
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {val_loss:.4f} | Tokens: {n_tokens / 1e6:.2f}M | Tokens/sec: {(n_tokens - last_token_count) / (time.time() - timer):.2f} | Peak Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | Time: {(time.time() - timer) / 60:.2f} min")
                
                dataset_idx = step
                
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

                # save checkpoint every 5000 steps 
                if (step % 5000 == 0) and (step > 0):
                    save_checkpoint(unwrapped_model, optimizer, epoch, dataset_idx, path=os.path.join(checkpoint_dir, checkpoint_name(epoch, dataset_idx, loss)), token_count=unwrapped_model.token_count)
            dataset_idx = 0
    except KeyboardInterrupt:
        try:
            input("Training interrupted. Press Enter to save checkpoint and exit...")
            logger.info("Training interrupted. Saving checkpoint...")
            save_checkpoint(unwrapped_model, optimizer, epoch, dataset_idx, path=os.path.join(checkpoint_dir, checkpoint_name(epoch, dataset_idx, loss, interrupted=True)), token_count=unwrapped_model.token_count)
        except Exception as e:
            logger.error(f"Exiting with: {e}")



if __name__ == "__main__":
    pretrain()




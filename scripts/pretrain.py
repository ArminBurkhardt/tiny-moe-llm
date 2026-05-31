import os
import sys
import time
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
from torch import optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator

from modules.model.transformer import TinyMoETransformer
from modules.data.dataset import Dataset
from modules.model.utils import create_causal_attention_mask
from modules.model.mtp import compute_mtp_loss
from config import ModelConfig, TrainingConfig
from utils import save_checkpoint, load_checkpoint, BASE_DIR, logger, BF16


def train_step(
    model: TinyMoETransformer, 
    optimizer: optim.Optimizer, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor,
):
    model.set_checkpointing(True, True)
    model.delayed_mtp_loss(True)
    
    if len(attention_mask.shape) == 2:
        attention_mask = attention_mask[:, None, None, :]
    
    if model.has_mtp:
        logits, aux_loss, extra_token_outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask.to(torch.bool), 
        **ModelConfig.Forward,
        return_aux_loss=True,
        return_hidden=True,
    )
    else:
        logits, aux_loss = model(
            input_ids=input_ids, 
            attention_mask=attention_mask.to(torch.bool), 
            **ModelConfig.Forward,
            return_aux_loss=True,
            return_hidden=True,
        )
        extra_token_outputs = None
        
    loss = compute_mtp_loss(
        logits, 
        input_ids, 
        mtp_outputs=extra_token_outputs, 
        lm_head=model.mtp_head.lm_head if extra_token_outputs is not None else None, 
        lambda_mtp=TrainingConfig.lambda_mtp,
        main_lm_head=model.lm_head
    ) + aux_loss
    
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    
    val_loss = loss.item()
    
    return val_loss

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
    input_ids = torch.randint(0, config.Params["vocab_size"], (TrainingConfig.Batch_size, TrainingConfig.Seq_length)).to(device)
    attention_mask = create_causal_attention_mask(input_ids.size(1), dtype=torch.bool, device=device)
    
    logits, aux_loss, extra_token_outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
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
        main_lm_head=model.lm_head
    ) + aux_loss
    
    loss.backward()
    model.zero_grad(set_to_none=True)

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

def pretrain():
    GEMMA4_TOKENIZER_PATH = os.path.join(BASE_DIR, "ckpts", "pretrained", "DeepSeek-V4-Pro-tokenizer") # "gemma4-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(GEMMA4_TOKENIZER_PATH)
    
    logger.info(f"Tokenizer loaded from {GEMMA4_TOKENIZER_PATH} with vocab size {tokenizer.vocab_size}")
    
    dataset = Dataset(
        tokenizer=tokenizer,
        batch_size=TrainingConfig.Batch_size,
        max_length=TrainingConfig.Seq_length,
        mode="pretrain",
        config_path=os.path.join(BASE_DIR, "data_config.json")
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=2, prefetch_factor=2)
    
    logger.info(f"Initialized dataset with {len(dataset.sources)} sources")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = TinyMoETransformer(**ModelConfig.Params).to(device).to(BF16).train()
    optimizer = optim.AdamW(model.parameters(), lr=TrainingConfig.lr, weight_decay=TrainingConfig.weight_decay)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    start_epoch, dataset_idx = 0, 0
    checkpoint_dir = os.path.join(BASE_DIR, "ckpts", "training")
    try:
        checkpoint_path = os.path.join(checkpoint_dir, get_latest_checkpoint_epoch(checkpoint_dir))
        start_epoch, dataset_idx = load_checkpoint(model, optimizer, checkpoint_path)
    except Exception as e:
        logger.warning(f"No checkpoint found. Starting training from scratch")
    
    logger.info(f"Starting training from epoch {start_epoch}, dataset index {dataset_idx}")
    if start_epoch != 0 or dataset_idx != 0:
        dataset.start_step = dataset_idx
    
    dry_run(model, device=device, dtype=BF16, config=ModelConfig)
    logger.info("Dry run successful. Starting training loop...")
    
    # setup accelerator
    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    timer = time.time()
    
    try:
        for epoch in range(start_epoch, TrainingConfig.num_epochs):
            for step, batch in enumerate(dataloader):
                # skip steps until we reach dataset_idx from checkpoint
                if epoch == start_epoch and step < dataset_idx:
                    continue
                    
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device) if batch["attention_mask"] is not None else create_causal_attention_mask(input_ids.size(1), dtype=torch.bool, device=device)
                               
                if (device == "cuda") and (step % 20 == 0):
                    torch.cuda.reset_peak_memory_stats()
                
                with accelerator.accumulate(model):
                    loss = train_step(model, optimizer, input_ids, attention_mask)
                n_tokens = model.token_count
                logger.info(f"Epoch {epoch} | Step {step} | Loss: {loss:.4f} | Tokens: {n_tokens / 1e6:.2f}M | Tokens/sec: {n_tokens / (time.time() - timer):.2f} | Peak Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                
                dataset_idx = step
                
                if step % model.moe.expert_tracker.sliding_window_size == 0:
                    stats = model.moe.expert_tracker.get_stats()
                    model._token_tracker.reset()
                    timer = time.time()
                    try:
                        save_expert_selection_graph(stats, os.path.join(BASE_DIR, "ckpts", "training", f"expert_selection_epoch{epoch}_step{step}.png"))
                    except Exception as e:
                        logger.error(f"Error occurred while saving expert selection graph: {e}")

            # save checkpoint at the end of each epoch
            save_checkpoint(model, optimizer, epoch, dataset_idx, path=os.path.join(checkpoint_dir, checkpoint_name(epoch, dataset_idx, loss)))
            dataset_idx = 0
    except KeyboardInterrupt:
        try:
            input("Training interrupted. Press Enter to save checkpoint and exit...")
            logger.info("Training interrupted. Saving checkpoint...")
            save_checkpoint(model, optimizer, epoch, dataset_idx, path=os.path.join(checkpoint_dir, checkpoint_name(epoch, dataset_idx, loss, interrupted=True)))
        except Exception as e:
            logger.error(f"Exiting with: {e}")



if __name__ == "__main__":
    pretrain()




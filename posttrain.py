#!/usr/bin/env python3
"""posttrain.py — Supervised fine-tuning (SFT) script for tiny-moe-llm.

Datasets
--------
* **KIMI-K2.5-550000x** – instruction-following conversations stored in a
  ``messages`` column containing a list of ``{"role": …, "content": …}`` dicts.
* **Reasoning datasets** (Claude-Opus-4.6, Claude-Sonnet-4.6) – reasoning
  conversations with the same ``messages`` schema, optionally accompanied by a
  ``metadata`` column (ignored during training).

Any additional dataset directory can be registered at the bottom of the
``SFT_SOURCES`` list (see *Expandability* section below), making it trivial to
incorporate synthetic data generated with e.g. Gemma 4.

Chat template
-------------
The Gemma 3 tokeniser's built-in chat template is applied to every conversation
via :class:`modules.data.chat_template.Chat`.  Labels are masked with ``-100``
for all non-assistant tokens so the gradient only propagates through the
model's own generated turns.

Forward pass
------------
SFT uses :meth:`FinalTransformer.sft_forward`, which routes the latent through
the existing experts without triggering the expert-addition / pruning lifecycle
used during pretraining.  All parameters (encoder, router, experts, decoder)
receive gradient updates at each step.

Usage
-----
    python posttrain.py \\
        --model_dir  ckpts/pretrained/gemma-3-1b-it \\
        --checkpoint ckpts/trained/pretrain/final.pt \\
        --output_dir ckpts/trained/sft \\
        --num_epochs 3

Expandability (synthetic data)
-------------------------------
To add an extra data source append a tuple ``(root_dir, "messages")`` to the
``SFT_SOURCES`` list in :func:`build_datasets`, or pass one or more
``--extra_data`` paths on the command line.
"""

import argparse
import json
import logging
import os
from typing import Iterator

import pandas as pd
import torch
from torch.utils.data import IterableDataset
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from modules.data.chat_template import Chat
from modules.data.dataloader import FileLoader
from modules.model.mtp import compute_mtp_loss
from modules.model.transformer import FinalTransformer
from utils import DIR, logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning (SFT) of tiny-moe-llm on chat datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        default=DIR.GEMMA_3_DIR,
        help="Path to the Gemma 3 checkpoint (tokeniser + encoder weights).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a pretrained FinalTransformer checkpoint to fine-tune.  "
            "When omitted a freshly initialised model is used."
        ),
    )
    parser.add_argument(
        "--data_config",
        default="data_config.json",
        help="Path to JSON file specifying datasets mapping.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(DIR.BASE_DIR, "ckpts", "trained", "sft"),
        help="Directory where SFT checkpoints are saved.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Latent dimension.  Inferred from the encoder config when not given.",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight-decay."
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient-norm clipping threshold (0 disables clipping).",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of passes over the dataset."
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Log metrics every N steps."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1_000,
        help="Save a checkpoint every N steps.",
    )
    parser.add_argument(
        "--mtp_steps",
        type=int,
        default=1,
        help="Number of future-token offsets used for the MTP objective.",
    )
    parser.add_argument(
        "--mtp_lambda",
        type=float,
        default=1.0,
        help="Geometric weighting factor for farther MTP offsets.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------

class SFTDataset(IterableDataset):
    """Iterable dataset that tokenises chat conversations for SFT.

    Each parquet row is expected to contain a ``messages`` column with a list of
    ``{"role": str, "content": str}`` dicts (the KIMI / reasoning schema).
    The Gemma chat template is applied before tokenisation, and labels are masked
    with ``-100`` for all non-assistant tokens so only the model's generated
    responses drive the gradient.

    Args:
        sources: Sequence of ``(root_dir, text_column)`` tuples.  Each
            *root_dir* is recursively walked by :class:`FileLoader`.  The
            *text_column* is always ``"messages"`` for the supported datasets but
            is kept configurable for forward compatibility.
        tokenizer: Hugging Face tokeniser (Gemma 3).
        max_length: Token budget per sample; sequences are truncated to this
            length.
        batch_size: Number of samples per yielded batch.
    """

    def __init__(
        self,
        sources: list[tuple[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        batch_size: int = 2,
    ) -> None:
        self.sources = sources
        self.tokenizer = tokenizer
        self.chat = Chat(tokenizer)
        self.max_length = max_length
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[dict]:
        return self._batch_iterator()

    def _batch_iterator(self) -> Iterator[dict]:
        """Yield batches of tokenised SFT samples."""
        buffer_ids: list[torch.Tensor] = []
        buffer_masks: list[torch.Tensor] = []
        buffer_labels: list[torch.Tensor] = []

        for input_ids, attention_mask, labels in self._sample_iterator():
            buffer_ids.append(input_ids)
            buffer_masks.append(attention_mask)
            buffer_labels.append(labels)

            if len(buffer_ids) >= self.batch_size:
                yield {
                    "input_ids": torch.stack(buffer_ids),
                    "attention_mask": torch.stack(buffer_masks),
                    "labels": torch.stack(buffer_labels),
                }
                buffer_ids, buffer_masks, buffer_labels = [], [], []

        # Yield any remaining samples as a partial batch
        if buffer_ids:
            yield {
                "input_ids": torch.stack(buffer_ids),
                "attention_mask": torch.stack(buffer_masks),
                "labels": torch.stack(buffer_labels),
            }

    def _sample_iterator(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield individual tokenised ``(input_ids, attention_mask, labels)`` triples."""
        for root_dir, col in self.sources:
            if not os.path.isdir(root_dir):
                logger.warning("SFT data root not found, skipping: %s", root_dir)
                continue

            for df in FileLoader(root_dir):
                if col not in df.columns:
                    continue

                for messages in df[col]:
                    try:
                        sample = self._tokenize_conversation(messages)
                    except Exception as exc:
                        logger.warning("Skipping malformed conversation: %s", exc)
                        continue
                    if sample is not None:
                        yield sample

    def _tokenize_conversation(
        self, messages: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Tokenise a conversation and build assistant-only labels.

        Strategy
        --------
        1. Format the full conversation with the chat template.
        2. Also format the conversation **up to and including each non-assistant
           turn** (with ``add_generation_prompt=True``) to determine the token
           boundary where assistant content begins.
        3. Set ``labels[start:end] = input_ids[start:end]`` for each assistant
           turn; all other positions remain ``-100``.

        Args:
            messages: List of ``{"role": str, "content": str}`` dicts.

        Returns:
            ``(input_ids, attention_mask, labels)`` each of shape
            ``[max_length]``, or ``None`` when the conversation is empty or
            unparseable.
        """
        if not messages:
            return None

        # Full conversation text
        full_text: str = self.chat.format_chat(messages)
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids: torch.Tensor = encoded["input_ids"][0]
        attention_mask: torch.Tensor = encoded["attention_mask"][0]

        # Start with everything masked
        labels = torch.full_like(input_ids, -100)

        # Find assistant-response token spans by diffing prefix lengths.
        # We compare the cumulative token count *up to but not including* the
        # assistant turn with the count *up to and including* it.
        cursor = 0
        for i, msg in enumerate(messages):
            # Text up to and including this message (= prefix after this turn)
            prefix_text: str = self.chat.format_chat(messages[: i + 1])
            prefix_ids = self.tokenizer(
                prefix_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"][0]
            end_pos = min(len(prefix_ids), self.max_length)

            if msg.get("role") == "assistant":
                # Label assistant tokens so they contribute to the loss
                labels[cursor:end_pos] = input_ids[cursor:end_pos]

            cursor = end_pos

        # Ensure padded positions are always masked
        labels = labels.masked_fill(attention_mask == 0, -100)

        return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(
    args: argparse.Namespace, vocab_size: int, latent_dim: int
) -> FinalTransformer:
    """Build a :class:`FinalTransformer` for SFT.

    Uses minimal expert count (2) and a high ``steps_per_expert_add`` so that
    no new experts are added during the SFT phase (the lifecycle is driven by
    :meth:`sft_forward`, which bypasses the MoE cycle entirely).

    Args:
        args: Parsed command-line arguments.
        vocab_size: Vocabulary size from the tokeniser.
        latent_dim: Latent space dimension.

    Returns:
        :class:`FinalTransformer` ready for SFT.
    """
    from modules.model.expert import ExpertModule

    expert_template = ExpertModule(latent_dim, latent_dim)
    model = FinalTransformer(
        model_dir=args.model_dir,
        latent_dim=latent_dim,
        vocab_size=vocab_size,
        num_initial_experts=2,
        # Large value so the MoE cycle never triggers expert-addition during SFT
        steps_per_expert_add=10_000_000,
        prune_step_interval=10_000_000,
        expert_template=expert_template,
    )
    return model


# ---------------------------------------------------------------------------
# Dataset construction (expandable)
# ---------------------------------------------------------------------------

def build_datasets(args: argparse.Namespace) -> list[tuple[str, str]]:
    with open(args.data_config, "r") as f:
        data_config = json.load(f)
    
    sources = [(item["root"], item["column"]) for item in data_config.get("sft", [])]
    if not sources:
        logger.warning(f"No SFT sources found in {args.data_config}")
    return sources


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def sft_step(
    model: FinalTransformer,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    vocab_size: int,
    device: str,
) -> dict:
    """Execute a single SFT step.

    Uses :meth:`FinalTransformer.sft_forward` for a gradient-enabled forward
    pass that routes through existing experts without triggering the expert
    lifecycle. Multi-token prediction (MTP) loss is computed only on assistant
    tokens (where ``labels != -100``).

    Args:
        model: :class:`FinalTransformer` in training mode.
        batch: Dict with ``input_ids``, ``attention_mask``, and ``labels``.
        optimizer: Active optimiser.
        args: Parsed command-line arguments.
        vocab_size: Vocabulary size.
        device: Target device string.

    Returns:
        Dict with scalar metrics: ``loss``.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = batch["labels"].to(device)

    # SFT forward: encoder → MoE (routing only) → decoder
    logits = model.sft_forward(input_ids, attention_mask=attention_mask)

    # Multi-token prediction loss (assistant tokens only)
    loss = compute_mtp_loss(
        logits=logits,
        labels=labels,
        mtp_steps=args.mtp_steps,
        mtp_lambda=args.mtp_lambda,
        ignore_index=-100,
    )

    optimizer.zero_grad()
    loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
    optimizer.step()

    return {"loss": loss.item()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: FinalTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    """Save model + optimiser state to *path*.

    Args:
        path: Destination file path.
        model: :class:`FinalTransformer` instance.
        optimizer: Active optimiser.
        step: Current global training step.
        epoch: Current epoch index.
        args: Parsed command-line arguments.
    """
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )
    logger.info("SFT checkpoint saved → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Tokeniser -----
    logger.info("Loading tokeniser from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    vocab_size = len(tokenizer)
    logger.info("Vocabulary size: %d", vocab_size)

    # ----- Latent dimension -----
    if args.latent_dim is None:
        config = AutoConfig.from_pretrained(args.model_dir)
        latent_dim = config.hidden_size
        logger.info("Inferred latent_dim=%d from encoder config.", latent_dim)
    else:
        latent_dim = args.latent_dim

    # ----- Model -----
    logger.info("Building FinalTransformer for SFT…")
    model = build_model(args, vocab_size, latent_dim)

    if args.checkpoint:
        logger.info("Loading pretrained weights from %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.train()
    # Enable dropout in the underlying Gemma model
    model.encoder.model.train()

    # ----- Optimiser (all parameters fine-tuned) -----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ----- Dataset sources -----
    sources = build_datasets(args)
    dataset = SFTDataset(
        sources=sources,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # ----- SFT training loop -----
    global_step = 0
    for epoch in range(args.num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.num_epochs)
        for batch in dataset:
            metrics = sft_step(model, batch, optimizer, args, vocab_size, device)
            global_step += 1

            if global_step % args.log_interval == 0:
                logger.info(
                    "epoch %d | step %6d | loss=%.4f",
                    epoch + 1,
                    global_step,
                    metrics["loss"],
                )

            if global_step % args.save_interval == 0:
                ckpt_path = os.path.join(
                    args.output_dir, f"sft_epoch{epoch + 1}_step{global_step}.pt"
                )
                save_checkpoint(ckpt_path, model, optimizer, global_step, epoch, args)

    # ----- Final checkpoint -----
    final_path = os.path.join(args.output_dir, "sft_final.pt")
    save_checkpoint(final_path, model, optimizer, global_step, args.num_epochs - 1, args)
    logger.info("SFT complete.  Final model → %s", final_path)


if __name__ == "__main__":
    main()

"""Post-training (SFT) script for tiny-moe-llm.

Datasets
--------
* **KIMI-K2.5-550000x** — ``messages`` column, each record is a list of
  ``{"role": "user"|"assistant", "content": "..."}`` dicts.
* **Claude-Opus-4.6** (reasoning) — ``messages`` column with an optional
  leading ``{"role": "system", ...}`` message.
* **Claude-Sonnet-4.6** (reasoning) — same ``messages`` schema as KIMI.
* **Synthetic data** — any additional parquet directory passed via
  ``--synthetic_data_dirs``; each parquet file must contain a ``messages``
  column in the same format (see *Expandability* below).

Training objective
------------------
Supervised fine-tuning (SFT) in representation space: the model is asked to
reproduce the Gemma 3 encoder's hidden states, but the loss is *masked* so
that only the token positions belonging to the **assistant's response** are
penalised.  User, system, and padding positions are ignored.

    target_vectors = encoder(input_ids)   # no_grad — frozen supervision signal
    output, router_loss = model(input_ids, target_vectors)
    task_loss = masked_MSE(output, target_vectors, assistant_mask)
    total_loss = task_loss + router_loss_weight * router_loss

All model parameters (encoder, MoE experts, decoder, router) are updated.

Chat template
-------------
The Gemma 3 tokenizer's built-in chat template is applied via
``modules/data/chat_template.Chat``.  The formatted string is then tokenized
with ``add_special_tokens=False`` so that special tokens introduced by
``apply_chat_template`` are preserved verbatim.

The assistant mask is derived by comparing the full-conversation token count to
the prompt-only token count: positions beyond the prompt prefix (up to the
``<end_of_turn>`` closing token) are treated as assistant tokens.

Expandability
-------------
Synthetic datasets (see ``docs/todo.md``) can be plugged in without modifying
this script by passing one or more directories to ``--synthetic_data_dirs``.
Each directory must follow the same parquet layout (subdirectory per dump,
files per shard) and expose a ``messages`` column.  The script interleaves
records from all sources in a round-robin fashion.

Usage
-----
    # Default paths
    python posttrain.py

    # Custom paths and extra synthetic data
    python posttrain.py \\
        --model_dir            ckpts/pretrained/gemma-3-1b-it \\
        --pretrained_ckpt      ckpts/pretrained/tiny-moe-pretrained/step-50000 \\
        --output_dir           ckpts/finetuned/tiny-moe-sft \\
        --batch_size           2 \\
        --max_steps            20000 \\
        --lr                   5e-5 \\
        --synthetic_data_dirs  data/datasets/parquet/my_synthetic_v1 \\
                               data/datasets/parquet/my_synthetic_v2
"""

import argparse
import ast
import itertools
import logging
import os
from typing import Generator, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modules.data.chat_template import Chat
from modules.data.dataloader import FileLoader
from modules.model.transformer import FinalTransformer
from modules.model.expert import ExpertModule
from utils import DIR, router_loss_scalar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT post-training of tiny-moe-llm on conversation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        default=DIR.GEMMA_3_DIR,
        help="Path to the Gemma 3 checkpoint directory (tokenizer + weights).",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        default=None,
        help=(
            "Path to a pre-trained tiny-moe-llm checkpoint directory produced "
            "by pretrain.py.  If omitted, the model is initialised from scratch."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(DIR.BASE_DIR, "ckpts", "finetuned", "tiny-moe-sft"),
        help="Where to save SFT checkpoints.",
    )
    # ---------------------------------------------------------------- datasets
    parser.add_argument(
        "--kimi_dir",
        default=DIR.KIMI_DIR,
        help="Root directory of the KIMI-K2.5-550000x parquet shards.",
    )
    parser.add_argument(
        "--reasoning_dir",
        default=DIR.REASONING_DIR,
        help="Root directory of the reasoning-dataset parquet shards "
             "(contains opus-4.6/ and sonnet-4.6/ sub-directories).",
    )
    parser.add_argument(
        "--synthetic_data_dirs",
        nargs="*",
        default=[],
        help=(
            "Zero or more additional parquet root directories containing "
            "synthetic conversation data.  Each must expose a ``messages`` "
            "column in the same format as KIMI/reasoning datasets.  "
            "Pass multiple paths separated by spaces."
        ),
    )
    # --------------------------------------------------------------- training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20_000,
        help="Total number of gradient update steps.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help=(
            "Latent dimension of the MoE / decoder.  Defaults to the encoder's "
            "hidden size."
        ),
    )
    parser.add_argument(
        "--num_initial_experts",
        type=int,
        default=4,
        help="Number of experts (only used when training from scratch).",
    )
    parser.add_argument(
        "--steps_per_expert_add",
        type=int,
        default=200,
        help="MoE routing steps between each new-expert addition.",
    )
    parser.add_argument(
        "--prune_step_interval",
        type=int,
        default=1000,
        help="Global steps between least-used-expert pruning.",
    )
    parser.add_argument(
        "--max_experts",
        type=int,
        default=16,
        help="Hard cap on the number of live experts.",
    )
    parser.add_argument(
        "--max_recurrence",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--router_loss_weight",
        type=float,
        default=0.1,
        help="Coefficient for the router auxiliary loss.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Global gradient norm clip value (0 to disable).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _parse_messages(raw) -> Optional[List[dict]]:
    """Convert a raw ``messages`` cell to a list of role/content dicts.

    Parquet loaders may materialise the stored list as an actual Python list
    (arrow → pandas) or as a string representation.  Both cases are handled.

    Returns ``None`` when the cell cannot be parsed or contains no messages.
    """
    if isinstance(raw, list):
        return raw if raw else None
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            return parsed if isinstance(parsed, list) and parsed else None
        except (ValueError, SyntaxError):
            return None
    return None


def _iter_conversations(data_dirs: List[str]) -> Generator[List[dict], None, None]:
    """Yield individual conversation ``messages`` lists from a set of roots.

    Each root is walked by :class:`~modules.data.dataloader.FileLoader`.
    Multiple roots are interleaved record-by-record in a round-robin fashion
    so that no single dataset dominates early training.

    Args:
        data_dirs: List of parquet root directories.  Each directory must
            contain sub-directories whose files are parquet shards with a
            ``messages`` column.

    Yields:
        A single conversation as a list of ``{"role": …, "content": …}``
        dicts.
    """
    loaders = []
    for root in data_dirs:
        if os.path.isdir(root):
            loaders.append(FileLoader(root))
        else:
            logger.warning("Data directory not found, skipping: %s", root)

    if not loaders:
        return

    # Round-robin across active loaders using itertools.cycle.
    all_iters = [_df_record_iter(loader) for loader in loaders]

    for record_iter in itertools.cycle(all_iters):
        try:
            messages = next(record_iter)
            if messages is not None:
                yield messages
        except StopIteration:
            break


def _df_record_iter(loader: FileLoader) -> Generator[Optional[List[dict]], None, None]:
    """Yield parsed ``messages`` values from every parquet shard in *loader*."""
    for df in loader:
        if "messages" not in df.columns:
            continue
        for raw in df["messages"]:
            yield _parse_messages(raw)


def _build_all_data_dirs(args: argparse.Namespace) -> List[str]:
    """Collect all dataset root directories in a deterministic order.

    Order: KIMI → reasoning (opus / sonnet sub-dirs) → synthetic (user-supplied).
    Each reasoning sub-directory is added individually so FileLoader can walk
    the correct level.
    """
    dirs: List[str] = []

    # KIMI
    if os.path.isdir(args.kimi_dir):
        dirs.append(args.kimi_dir)
    else:
        logger.warning("KIMI directory not found: %s", args.kimi_dir)

    # Reasoning: each sub-directory is a separate dataset shard root.
    if os.path.isdir(args.reasoning_dir):
        for sub in sorted(os.listdir(args.reasoning_dir)):
            sub_path = os.path.join(args.reasoning_dir, sub)
            if os.path.isdir(sub_path):
                dirs.append(sub_path)
    else:
        logger.warning("Reasoning directory not found: %s", args.reasoning_dir)

    # Synthetic data (user-provided, enables easy dataset expansion).
    for sd in args.synthetic_data_dirs:
        if os.path.isdir(sd):
            dirs.append(sd)
        else:
            logger.warning("Synthetic data directory not found: %s", sd)

    return dirs


# ---------------------------------------------------------------------------
# Assistant-token masking
# ---------------------------------------------------------------------------

def _build_assistant_mask(
    full_ids: torch.Tensor,
    messages: List[dict],
    chat: Chat,
    tokenizer,
    max_length: int,
) -> torch.Tensor:
    """Return a boolean mask of shape ``[seq_len]``.

    Positions set to ``True`` correspond to the assistant's response tokens in
    the final turn.  All other positions (user, system, padding) are ``False``.

    The mask is computed by comparing the length of the *prompt-only*
    tokenization (everything up to but not including the last assistant reply)
    with the full tokenized conversation.

    Args:
        full_ids: Tensor of shape ``[seq_len]`` containing the tokenized full
            conversation (already padded / truncated to ``max_length``).
        messages: The raw list of role/content dicts for this conversation.
        chat: :class:`~modules.data.chat_template.Chat` instance.
        tokenizer: The Gemma 3 tokenizer.
        max_length: The maximum sequence length used for tokenization.

    Returns:
        Boolean tensor of shape ``[seq_len]``.
    """
    seq_len = full_ids.size(0)
    mask = torch.zeros(seq_len, dtype=torch.bool)

    # Find the last assistant message so we can identify the prefix.
    last_asst_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") in ("assistant", "model"):
            last_asst_idx = i
            break

    if last_asst_idx is None:
        # No assistant turn found — mask is all zeros (no loss contribution).
        return mask

    # Prompt: everything up to (not including) the last assistant turn.
    prompt_messages = messages[:last_asst_idx]
    # Append an empty assistant placeholder so the template adds the turn header.
    prompt_with_header = prompt_messages + [
        {"role": "assistant", "content": ""}
    ]
    try:
        prompt_text = chat.format_chat(prompt_with_header)
    except Exception:
        # If the template rejects the modified messages list, fall back to an
        # empty prompt (entire sequence is marked as assistant tokens).
        prompt_text = ""

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"][0]

    start = min(len(prompt_ids), seq_len)
    mask[start:] = True
    return mask


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step: int, output_dir: str) -> None:
    ckpt_dir = os.path.join(output_dir, f"step-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    logger.info("Checkpoint saved to %s", ckpt_dir)


def load_checkpoint(model, optimizer, resume_from: str) -> int:
    model.load_state_dict(
        torch.load(os.path.join(resume_from, "model.pt"), map_location="cpu")
    )
    optimizer.load_state_dict(
        torch.load(os.path.join(resume_from, "optimizer.pt"), map_location="cpu")
    )
    try:
        step = int(os.path.basename(resume_from.rstrip("/")).split("-")[-1])
    except ValueError:
        step = 0
    logger.info("Loaded checkpoint from %s (step %d)", resume_from, step)
    return step


# ---------------------------------------------------------------------------
# Main SFT loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------- model
    logger.info("Loading tokenizer from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chat = Chat(tokenizer)

    if args.latent_dim is None:
        from transformers import AutoConfig
        enc_cfg = AutoConfig.from_pretrained(args.model_dir)
        args.latent_dim = enc_cfg.hidden_size
        logger.info("latent_dim inferred from encoder config: %d", args.latent_dim)

    logger.info("Building FinalTransformer (latent_dim=%d)", args.latent_dim)
    model = FinalTransformer(
        model_dir=args.model_dir,
        latent_dim=args.latent_dim,
        output_dim=args.latent_dim,
        num_initial_experts=args.num_initial_experts,
        steps_per_expert_add=args.steps_per_expert_add,
        prune_step_interval=args.prune_step_interval,
        max_recurrence=args.max_recurrence,
        expert_template=ExpertModule(args.latent_dim, args.latent_dim),
    )

    # All parameters are fine-tuned during SFT.
    model.encoder.train()
    for param in model.parameters():
        param.requires_grad_(True)
    # Enable grad on initial SolvableLinear experts.
    for expert in model.moe.experts:
        if hasattr(expert, 'consolidate'):
            expert.consolidate(disable_grad=False, dtype=torch.float32)
        elif hasattr(expert, 'enable_grad'):
            expert.enable_grad(True)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_step = 0
    if args.pretrained_ckpt:
        start_step = load_checkpoint(model, optimizer, args.pretrained_ckpt)

    # ------------------------------------------------------------ data
    all_dirs = _build_all_data_dirs(args)
    logger.info("SFT data sources (%d directories):", len(all_dirs))
    for d in all_dirs:
        logger.info("  %s", d)

    conv_iter = _iter_conversations(all_dirs)

    # -------------------------------------------------------- training loop
    step = start_step
    running_task_loss = 0.0
    running_router_loss = 0.0

    logger.info(
        "Starting SFT  (steps %d → %d, device=%s)",
        start_step, args.max_steps, device,
    )

    while step < args.max_steps:
        # ------------------------------------------------ accumulate a batch
        batch_messages: List[List[dict]] = []
        batch_texts: List[str] = []

        while len(batch_texts) < args.batch_size:
            try:
                messages = next(conv_iter)
            except StopIteration:
                # Restart all loaders when all sources are exhausted.
                logger.info("All SFT data exhausted; restarting data iterators.")
                all_dirs = _build_all_data_dirs(args)
                conv_iter = _iter_conversations(all_dirs)
                try:
                    messages = next(conv_iter)
                except StopIteration:
                    logger.warning("No SFT data available after restart; stopping.")
                    break

            if messages is None:
                continue

            try:
                text = chat.format_chat(messages)
            except Exception as exc:
                logger.debug("Skipping malformed conversation: %s", exc)
                continue

            batch_messages.append(messages)
            batch_texts.append(text)

        if not batch_texts:
            break

        # ------------------------------------------------- tokenise
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # ------------------------------------------ assistant masks
        # Shape: [batch_size, seq_len]
        asst_masks = torch.stack(
            [
                _build_assistant_mask(
                    input_ids[i].cpu(),
                    batch_messages[i],
                    chat,
                    tokenizer,
                    args.max_length,
                )
                for i in range(input_ids.size(0))
            ]
        ).to(device)  # bool, [B, S]

        # If no assistant tokens exist in the entire batch, skip the step to
        # avoid computing a meaningless zero loss.
        if not asst_masks.any():
            continue

        # -------------------------------------------- target vectors
        with torch.no_grad():
            target_vectors = model.encoder(input_ids, attention_mask=attention_mask)
        target_vectors = target_vectors.detach()

        # --------------------------------------------- forward pass
        output, router_loss = model(input_ids, target_vectors)

        # ------------------------------------ masked SFT task loss
        # Compute MSE only on assistant-response token positions.
        # ``asst_masks`` has shape [B, S]; broadcast to [B, S, H] for indexing.
        mask_expanded = asst_masks.unsqueeze(-1).expand_as(output)
        output_masked = output[mask_expanded].view(-1, output.size(-1))
        target_masked = target_vectors.float()[mask_expanded].view(-1, target_vectors.size(-1))

        if output_masked.numel() == 0:
            continue

        task_loss = F.mse_loss(output_masked.float(), target_masked)
        total_loss = task_loss + args.router_loss_weight * router_loss

        # ------------------------------------------- back-propagation
        optimizer.zero_grad()
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ---------------------------------------- expert count cap
        if len(model.moe.experts) > args.max_experts:
            model.moe.prune_least_used()
            logger.info(
                "Step %d: expert cap exceeded → pruned; experts now: %d",
                step, len(model.moe.experts),
            )

        step += 1
        running_task_loss += task_loss.item()
        running_router_loss += router_loss_scalar(router_loss)

        # --------------------------------------------------- logging
        if step % args.log_steps == 0:
            avg_task = running_task_loss / args.log_steps
            avg_router = running_router_loss / args.log_steps
            logger.info(
                "step=%6d  sft_loss=%.4f  router_loss=%.4f  experts=%d",
                step, avg_task, avg_router, len(model.moe.experts),
            )
            running_task_loss = 0.0
            running_router_loss = 0.0

        # ------------------------------------------------ checkpoint
        if step % args.save_steps == 0:
            save_checkpoint(model, optimizer, step, args.output_dir)

    save_checkpoint(model, optimizer, step, args.output_dir)
    logger.info("SFT complete.  Final checkpoint at step %d.", step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())

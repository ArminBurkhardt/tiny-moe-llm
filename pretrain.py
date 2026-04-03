"""Pre-training script for tiny-moe-llm on UltraFineWeb.

Training objective
------------------
Self-supervised representation learning: the model is trained to reproduce the
Gemma 3 encoder's hidden states at every sequence position.  Concretely,

    target_vectors = encoder(input_ids)   # computed with no_grad

and the loss is

    task_loss = MSE(output, target_vectors)   # output from FinalTransformer

The decoder's cross-attention uses the live encoder context (with gradients),
so the encoder is genuinely fine-tuned through the decoder path.

Expert lifecycle
----------------
The MixtureOfExperts module already handles this internally:

1. For the first ``steps_per_expert_add`` MoE steps in each cycle the
   existing experts are used for normal routing.
2. On step ``steps_per_expert_add`` a new expert is added and its
   parameters are solved in closed form via ``solve_from_batch``.
3. Immediately after solving, gradients are re-enabled on the new expert
   so it participates in back-propagation like any regular layer
   (see ``modules/model/moe.py``).
4. The least-used expert is pruned whenever:
   - ``global_step % prune_step_interval == 0``, handled by FinalTransformer,
     *or*
   - the number of live experts exceeds ``max_experts``, checked here after
     every forward pass.

Usage
-----
    # Minimal run (assumes model checkpoint at default path)
    python pretrain.py

    # Custom paths and hyperparameters
    python pretrain.py \\
        --model_dir      ckpts/pretrained/gemma-3-1b-it \\
        --data_dir       data/datasets/parquet/ultrafineweb_en_v1_4 \\
        --output_dir     ckpts/pretrained/tiny-moe-pretrained \\
        --batch_size     4 \\
        --max_steps      50000 \\
        --lr             1e-4 \\
        --max_experts    16 \\
        --steps_per_expert_add 500 \\
        --prune_step_interval  2000 \\
        --save_steps     1000 \\
        --log_steps      50
"""

import argparse
import copy
import logging
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modules.model.expert import ExpertModule
from modules.model.transformer import FinalTransformer
from modules.data.dataloader import DataLoader as FileDataLoader
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
        description="Pre-train tiny-moe-llm on UltraFineWeb (self-supervised).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        default=DIR.GEMMA_3_DIR,
        help="Path to the Gemma 3 checkpoint directory.",
    )
    parser.add_argument(
        "--data_dir",
        default=DIR.UFW_V1_4_DIR,
        help="Root directory of the UltraFineWeb parquet shards.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(DIR.BASE_DIR, "ckpts", "pretrained", "tiny-moe-pretrained"),
        help="Where to save checkpoints.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100_000,
        help="Total number of gradient update steps.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token sequence length (inputs are truncated/padded to this).",
    )
    parser.add_argument(
        "--minimum_score",
        type=float,
        default=0.5,
        help="Minimum UltraFineWeb quality score; records below this are dropped.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help=(
            "Latent dimension of the MoE / decoder.  Defaults to the encoder's "
            "hidden size (read from the Gemma config)."
        ),
    )
    parser.add_argument(
        "--num_initial_experts",
        type=int,
        default=4,
        help="Number of experts to initialise at the start of training.",
    )
    parser.add_argument(
        "--steps_per_expert_add",
        type=int,
        default=500,
        help="MoE routing steps between each new-expert addition.",
    )
    parser.add_argument(
        "--prune_step_interval",
        type=int,
        default=2000,
        help="Global training steps between automatic least-used-expert pruning.",
    )
    parser.add_argument(
        "--max_experts",
        type=int,
        default=16,
        help="Hard upper limit on the number of live experts; the least-used "
             "expert is immediately pruned when this is exceeded.",
    )
    parser.add_argument(
        "--max_recurrence",
        type=int,
        default=10,
        help="Maximum MoE recurrence iterations during a single forward pass.",
    )
    parser.add_argument(
        "--router_loss_weight",
        type=float,
        default=0.1,
        help="Coefficient for the router auxiliary loss.",
    )
    parser.add_argument(
        "--encoder_target_layer",
        type=int,
        default=12,
        help="Gemma 3 transformer layer to use as the encoder output.",
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
        help="Save a checkpoint every this many gradient steps.",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=50,
        help="Log training metrics every this many steps.",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Path to a checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: FinalTransformer, optimizer, step: int, output_dir: str) -> None:
    """Save model weights and optimizer state to ``output_dir/step-<step>/``."""
    ckpt_dir = os.path.join(output_dir, f"step-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    logger.info("Checkpoint saved to %s", ckpt_dir)


def load_checkpoint(model: FinalTransformer, optimizer, resume_from: str) -> int:
    """Load weights and optimizer state; return the step to resume from."""
    model.load_state_dict(
        torch.load(os.path.join(resume_from, "model.pt"), map_location="cpu")
    )
    optimizer.load_state_dict(
        torch.load(os.path.join(resume_from, "optimizer.pt"), map_location="cpu")
    )
    # Infer step from directory name: …/step-<N>/
    try:
        step = int(os.path.basename(resume_from.rstrip("/")).split("-")[-1])
    except ValueError:
        step = 0
    logger.info("Resumed from %s (step %d)", resume_from, step)
    return step


def enable_initial_expert_grads(model: FinalTransformer) -> None:
    """Enable gradient tracking on the experts created at initialisation time.

    Experts added *during training* have their gradients enabled automatically
    inside ``MixtureOfExperts.forward``.  The initial experts created by
    ``FinalTransformer.__init__`` use ``SolvableLinear`` with
    ``grad_enabled=False`` by default; this function activates them.
    """
    for expert in model.moe.experts:
        if hasattr(expert, 'consolidate'):
            # ExpertModule: convert SolvableLinear → nn.Linear and enable grad.
            expert.consolidate(disable_grad=False, dtype=torch.float32)
        elif hasattr(expert, 'enable_grad'):
            # SolvableLinear used directly.
            expert.enable_grad(True)
        else:
            for param in expert.parameters():
                param.requires_grad_(True)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ model
    logger.info("Loading tokenizer from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve latent_dim from the encoder config when not set explicitly.
    if args.latent_dim is None:
        from transformers import AutoConfig
        enc_cfg = AutoConfig.from_pretrained(args.model_dir)
        args.latent_dim = enc_cfg.hidden_size
        logger.info("latent_dim inferred from encoder config: %d", args.latent_dim)

    logger.info("Building FinalTransformer (latent_dim=%d, output_dim=%d)",
                args.latent_dim, args.latent_dim)
    model = FinalTransformer(
        model_dir=args.model_dir,
        latent_dim=args.latent_dim,
        output_dim=args.latent_dim,      # output_dim == latent_dim keeps decoder invertible
        num_initial_experts=args.num_initial_experts,
        steps_per_expert_add=args.steps_per_expert_add,
        prune_step_interval=args.prune_step_interval,
        max_recurrence=args.max_recurrence,
        expert_template=ExpertModule(args.latent_dim, args.latent_dim),
    )

    # Fine-tune the encoder (set it to training mode; the base model was put
    # into eval() inside Gemma3Encoder.__init__).
    model.encoder.train()
    # Enable gradients on the initial experts.
    enable_initial_expert_grads(model)

    model = model.to(device)
    model.train()

    # ------------------------------------------------------------ optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, args.resume_from)

    # --------------------------------------------------------------- data
    logger.info("Setting up DataLoader from %s", args.data_dir)
    data_loader = FileDataLoader(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        drop_last=True,
        minimum_score=args.minimum_score,
        target_column="content",
    )

    # ----------------------------------------------------------- training
    step = start_step
    running_task_loss = 0.0
    running_router_loss = 0.0

    logger.info(
        "Starting pre-training  (steps %d → %d, device=%s)",
        start_step, args.max_steps, device,
    )

    data_iter = iter(data_loader)

    while step < args.max_steps:
        # ------------------------------------------------- fetch next batch
        try:
            batch_df = next(data_iter)
        except StopIteration:
            # Restart the data iterator when the dataset is exhausted.
            data_loader.file_loader.reset()
            data_loader.load_next_file()
            data_iter = iter(data_loader)
            try:
                batch_df = next(data_iter)
            except StopIteration:
                logger.warning("Dataset exhausted after %d steps; stopping.", step)
                break

        texts = batch_df["content"].tolist()
        if not texts:
            continue

        # -------------------------------------------------- tokenise
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # -------------------------------------------- target vectors
        # Compute the encoder's hidden states as supervision signal.
        # Done under no_grad so the encoder's target representation is treated
        # as a fixed label; gradients for encoder fine-tuning come through the
        # decoder's cross-attention with the live encoder context.
        with torch.no_grad():
            target_vectors = model.encoder(input_ids, attention_mask=attention_mask)
        target_vectors = target_vectors.detach()

        # --------------------------------------------- forward pass
        output, router_loss = model(input_ids, target_vectors)

        # -------------------------------------------------- task loss
        # Mean-squared error between the decoder output and the encoder's
        # hidden states.  Both tensors are cast to float32 before the MSE
        # to avoid potential fp64 / bf16 type mismatches.
        task_loss = F.mse_loss(output.float(), target_vectors.float())
        total_loss = task_loss + args.router_loss_weight * router_loss

        # ------------------------------------------- back-propagation
        optimizer.zero_grad()
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ----------------------------------------- expert count cap
        # Prune immediately if the expert count exceeds the configured cap.
        if len(model.moe.experts) > args.max_experts:
            model.moe.prune_least_used()
            logger.info(
                "Step %d: expert cap (%d) exceeded → pruned; experts now: %d",
                step, args.max_experts, len(model.moe.experts),
            )

        step += 1

        running_task_loss += task_loss.item()
        running_router_loss += router_loss_scalar(router_loss)

        # ---------------------------------------------------- logging
        if step % args.log_steps == 0:
            avg_task = running_task_loss / args.log_steps
            avg_router = running_router_loss / args.log_steps
            logger.info(
                "step=%6d  task_loss=%.4f  router_loss=%.4f  experts=%d",
                step, avg_task, avg_router, len(model.moe.experts),
            )
            running_task_loss = 0.0
            running_router_loss = 0.0

        # -------------------------------------------------- checkpoint
        if step % args.save_steps == 0:
            save_checkpoint(model, optimizer, step, args.output_dir)

    # Final checkpoint
    save_checkpoint(model, optimizer, step, args.output_dir)
    logger.info("Pre-training complete.  Final checkpoint at step %d.", step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())

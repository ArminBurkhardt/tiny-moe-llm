#!/usr/bin/env python3
"""pretrain.py — Pretraining script for tiny-moe-llm on UltraFineWeb.

Training objective: next-token prediction (causal LM) via cross-entropy loss.

Expert lifecycle during pretraining
------------------------------------
The MoE module follows a fixed curriculum cycle of length
``steps_per_expert + 2``:

1. **Normal routing steps** (``cycle_pos < steps_per_expert``): the router
   distributes the latent across existing experts.
2. **Expert-addition step** (``cycle_pos == steps_per_expert``): a new expert
   is added whose parameters are solved analytically via closed-form
   least-squares (no back-prop for that step).  After the solve the expert is
   *consolidated* (fp64 → fp32) and gradient updates are *enabled*, so the
   new expert participates in all subsequent gradient steps alongside every
   other trainable parameter.
3. **OUTPUT-routing step** (``cycle_pos == steps_per_expert + 1``): the router
   is trained to select the special OUTPUT (identity) expert.

When the number of live experts reaches ``max_experts``, the least-used expert
is pruned (``moe.prune_least_used()``).  The optimiser is rebuilt at that point
to release all references to the removed expert's parameters.

Optimiser groups
----------------
* Encoder, decoder, and initial experts share a base learning rate.
* The router uses a separate (configurable) learning rate.
* Newly added experts are appended to the optimiser as a fresh param group
  right after their grad is enabled, so they receive gradient updates from the
  very next step.

Default hyperparameters
-----------------------
All model and training defaults live in :mod:`training_config` (:class:`~training_config.PretrainConfig`).
CLI flags override individual values; the merged config is saved in every
checkpoint for full reproducibility.

Usage
-----
    python pretrain.py \\
        --model_dir   ckpts/pretrained/gemma-3-1b-it \\
        --data_root   data/datasets/parquet/ultrafineweb_en_v1_4 \\
        --output_dir  ckpts/trained/pretrain \\
        --max_experts 16 \\
        --num_steps   10000
"""

import argparse
import importlib
import logging
import os

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from modules.data.dataset import Dataset
from modules.model.transformer import FinalTransformer
from training_config import EXPERT_TEMPLATES, PretrainConfig
from utils import DIR, logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _cfg = PretrainConfig()  # source of default values

    parser = argparse.ArgumentParser(
        description="Pretrain tiny-moe-llm on UltraFineWeb (next-token prediction).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        default=DIR.GEMMA_3_DIR,
        help="Path to the Gemma 3 checkpoint used as encoder.",
    )
    parser.add_argument(
        "--data_root",
        default=DIR.UFW_V1_4_DIR,
        help="Root directory of UltraFineWeb parquet shards.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(DIR.BASE_DIR, "ckpts", "trained", "pretrain"),
        help="Directory where checkpoints are written.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help=(
            "Latent space dimension.  Defaults to the encoder's hidden_size when "
            "not specified."
        ),
    )
    parser.add_argument(
        "--expert_template",
        default=_cfg.expert_template,
        choices=list(EXPERT_TEMPLATES),
        help="Expert class to use for the MoE.  Must be registered in training_config.EXPERT_TEMPLATES.",
    )
    parser.add_argument(
        "--max_experts",
        type=int,
        default=_cfg.max_experts,
        help=(
            "Maximum number of live experts.  When this limit is reached the "
            "least-used expert is pruned before a new one is added."
        ),
    )
    parser.add_argument(
        "--num_initial_experts",
        type=int,
        default=_cfg.num_initial_experts,
        help="Number of experts to pre-populate at model initialisation.",
    )
    parser.add_argument(
        "--steps_per_expert",
        type=int,
        default=_cfg.steps_per_expert,
        help="Normal-routing steps per MoE cycle before a new expert is added.",
    )
    parser.add_argument(
        "--prune_step_interval",
        type=int,
        default=_cfg.prune_step_interval,
        help=(
            "FinalTransformer global-step interval at which the least-used expert "
            "is additionally pruned (independent of the max_experts cap)."
        ),
    )
    parser.add_argument(
        "--max_recurrence",
        type=int,
        default=_cfg.max_recurrence,
        help="Maximum number of recurrent MoE iterations during inference.",
    )
    parser.add_argument("--lr", type=float, default=_cfg.lr, help="Base learning rate.")
    parser.add_argument(
        "--router_lr",
        type=float,
        default=_cfg.router_lr,
        help="Learning rate for router parameters (often benefits from a higher value).",
    )
    parser.add_argument(
        "--router_loss_weight",
        type=float,
        default=_cfg.router_loss_weight,
        help="Coefficient for the router auxiliary loss.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=_cfg.weight_decay, help="AdamW weight-decay."
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=_cfg.grad_clip,
        help="Gradient-norm clipping threshold (0 disables clipping).",
    )
    parser.add_argument("--batch_size", type=int, default=_cfg.batch_size)
    parser.add_argument(
        "--max_length", type=int, default=_cfg.max_length, help="Maximum token sequence length."
    )
    parser.add_argument(
        "--num_steps", type=int, default=_cfg.num_steps, help="Total training steps."
    )
    parser.add_argument(
        "--log_interval", type=int, default=_cfg.log_interval, help="Log metrics every N steps."
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=_cfg.save_interval,
        help="Save a checkpoint every N steps.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace, vocab_size: int, latent_dim: int) -> FinalTransformer:
    """Construct a :class:`FinalTransformer` for pretraining.

    The expert template class is resolved via the :data:`~training_config.EXPERT_TEMPLATES`
    registry using ``args.expert_template``.  The template must implement
    ``solve_from_batch`` so the MoE expert-addition cycle can solve its
    parameters analytically.

    Args:
        args: Parsed command-line arguments (expert_template and model-architecture
            flags are read here).
        vocab_size: Vocabulary size (from the tokeniser).
        latent_dim: Latent space dimension (typically the encoder hidden size).

    Returns:
        An untrained :class:`FinalTransformer` in training mode.
    """
    # Resolve the expert class from the registry
    fqn = EXPERT_TEMPLATES[args.expert_template]
    module_name, class_name = fqn.rsplit(".", 1)
    expert_cls = getattr(importlib.import_module(module_name), class_name)
    expert_template = expert_cls(latent_dim, latent_dim)

    model = FinalTransformer(
        model_dir=args.model_dir,
        latent_dim=latent_dim,
        vocab_size=vocab_size,
        num_initial_experts=args.num_initial_experts,
        steps_per_expert_add=args.steps_per_expert,
        prune_step_interval=args.prune_step_interval,
        max_recurrence=args.max_recurrence,
        expert_template=expert_template,
    )
    return model


# ---------------------------------------------------------------------------
# Optimiser helpers
# ---------------------------------------------------------------------------

def build_optimizer(model: FinalTransformer, args: argparse.Namespace) -> torch.optim.AdamW:
    """Build an :class:`~torch.optim.AdamW` with separate param groups.

    * Encoder, decoder, and expert parameters use ``args.lr``.
    * Router parameters use ``args.router_lr``.

    This function is also used to rebuild the optimiser after expert pruning, so
    that all references to the removed expert's parameters are released.

    Args:
        model: :class:`FinalTransformer` instance whose current parameters define
            the optimiser groups.
        args: Parsed command-line arguments (provides learning rates and
            weight-decay).

    Returns:
        Configured :class:`~torch.optim.AdamW` optimiser.
    """
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    router_params = list(model.moe.router.parameters())
    expert_params = [
        p
        for expert in model.moe.experts
        for p in expert.parameters()
        if p.requires_grad
    ]

    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr, "name": "encoder"},
            {"params": decoder_params, "lr": args.lr, "name": "decoder"},
            {"params": router_params, "lr": args.router_lr, "name": "router"},
            {"params": expert_params, "lr": args.lr, "name": "experts"},
        ],
        weight_decay=args.weight_decay,
    )


def _enable_expert_grad(expert: torch.nn.Module) -> None:
    """Consolidate a solved expert and enable gradient-based updates.

    Calls :meth:`~modules.model.expert.ExpertModule.consolidate` when available
    (converts the internal ``SolvableLinear`` to a standard ``nn.Linear`` at
    ``float32`` precision and enables ``requires_grad``), otherwise falls back
    to enabling ``requires_grad`` on all parameters directly.

    Args:
        expert: Newly solved expert module.
    """
    if hasattr(expert, "consolidate"):
        # ExpertModule.consolidate: fp64 → fp32, replaces SolvableLinear with
        # nn.Linear, enables grad
        expert.consolidate(force=True, disable_grad=False, dtype=torch.float32)
    else:
        for param in expert.parameters():
            param.requires_grad = True


def _add_expert_to_optimizer(
    optimizer: torch.optim.Optimizer, expert: torch.nn.Module, lr: float
) -> None:
    """Register a new expert's trainable parameters with the optimiser.

    Args:
        optimizer: The active :class:`~torch.optim.Optimizer`.
        expert: Expert module whose parameters should be trained.
        lr: Learning rate to assign to the new param group.
    """
    new_params = [p for p in expert.parameters() if p.requires_grad]
    if new_params:
        optimizer.add_param_group({"params": new_params, "lr": lr, "name": "expert"})


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model: FinalTransformer,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    vocab_size: int,
    device: str,
) -> tuple[dict, torch.optim.Optimizer | None]:
    """Execute a single pretraining step.

    The batch is expected to contain ``input_ids`` (with padding), optional
    ``attention_mask``, and ``labels`` (same as ``input_ids`` but with padding
    positions set to ``-100``).  The sequence is shifted by one position so the
    model predicts the *next* token at each position.

    After the MoE forward pass, any newly added expert is enabled for gradient
    updates and registered with the optimiser via
    :func:`_add_expert_to_optimizer`.

    If the live-expert count exceeds ``args.max_experts``, the least-used expert
    is pruned and the optimiser is **rebuilt** via :func:`build_optimizer`.
    This releases all references to the removed expert's parameters and prevents
    a memory leak from stale parameter tensors retained in optimiser state.

    Args:
        model: :class:`FinalTransformer` in training mode.
        batch: Dict with ``input_ids``, optionally ``attention_mask`` / ``labels``.
        optimizer: Active optimiser.
        args: Parsed command-line arguments.
        vocab_size: Vocabulary size used to build one-hot target vectors.
        device: Target device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns:
        A ``(metrics, new_optimizer)`` pair.  ``metrics`` is a dict with scalar
        keys ``lm_loss``, ``router_loss``, ``total_loss``, and ``num_experts``.
        ``new_optimizer`` is a freshly built :class:`~torch.optim.AdamW` when
        pruning occurred, or ``None`` when no pruning took place.
    """
    # Shift sequences: input = all but last token, target = all but first token
    input_ids = batch["input_ids"][:, :-1].to(device)
    labels = batch["labels"][:, 1:].to(device)

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask[:, :-1].to(device)

    # Build one-hot target vectors in vocabulary space [B, T, V].
    # Padding positions (label == -100) receive a zero vector so they do not
    # influence the expert-solving objective.
    valid_mask = labels != -100
    target_vectors = torch.zeros(
        labels.shape[0], labels.shape[1], vocab_size,
        device=device, dtype=torch.float32,
    )
    if valid_mask.any():
        clamped = labels.clone()
        clamped[~valid_mask] = 0  # avoid out-of-range index
        one_hot = F.one_hot(clamped, num_classes=vocab_size).float()
        target_vectors[valid_mask] = one_hot[valid_mask]

    # Track expert count before the forward pass to detect additions
    n_experts_before = len(model.moe.experts)

    # Forward pass (MoE cycle handles expert-addition internally)
    logits, router_loss = model(
        input_ids,
        target_vectors=target_vectors,
        attention_mask=attention_mask,
    )

    # Enable gradient updates on any experts that were just added and register
    # their parameters with the current optimiser
    n_experts_after = len(model.moe.experts)
    for idx in range(n_experts_before, n_experts_after):
        new_expert = model.moe.experts[idx]
        _enable_expert_grad(new_expert)
        _add_expert_to_optimizer(optimizer, new_expert, args.lr)
        logger.info("Expert %d added and enabled for gradient updates.", idx)

    # Prune if the live-expert count exceeds the configured cap, then rebuild
    # the optimiser so that stale references to the pruned expert's parameters
    # are released (avoids an effective memory leak in the optimiser state).
    new_optimizer: torch.optim.Optimizer | None = None
    if n_experts_after > args.max_experts:
        model.moe.prune_least_used()
        new_optimizer = build_optimizer(model, args)
        logger.info(
            "Pruned least-used expert (max_experts=%d).  Live experts: %d.  "
            "Optimiser rebuilt.",
            args.max_experts,
            len(model.moe.experts),
        )

    # Language-modelling loss (cross-entropy, ignoring padding)
    lm_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )

    # Combine LM loss with the router auxiliary loss
    router_loss_value = router_loss if isinstance(router_loss, torch.Tensor) else torch.tensor(router_loss, device=device)
    total_loss = lm_loss + args.router_loss_weight * router_loss_value

    # Backward pass and optimiser step (use the active optimiser before any
    # rebuild; the rebuilt optimiser takes effect from the next step)
    optimizer.zero_grad()
    total_loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
    optimizer.step()

    return (
        {
            "lm_loss": lm_loss.item(),
            "router_loss": router_loss_value.item(),
            "total_loss": total_loss.item(),
            "num_experts": len(model.moe.experts),
        },
        new_optimizer,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: FinalTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    config: dict,
) -> None:
    """Save model + optimiser state to *path*.

    The serialised payload includes a copy of the merged training configuration
    (CLI args merged on top of :class:`~training_config.PretrainConfig` defaults)
    so that every checkpoint is fully self-contained and the run can be
    reproduced exactly.

    Args:
        path: Destination file path (e.g. ``ckpts/trained/pretrain/ckpt_step1000.pt``).
        model: :class:`FinalTransformer` instance.
        optimizer: Active optimiser.
        step: Current global training step.
        args: Parsed command-line arguments (saved for reference).
        config: Serialised :class:`~training_config.PretrainConfig` dict (saved
            as the canonical hyperparameter record for reproducibility).
    """
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "config": config,
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Build a serialisable config dict from CLI args -----
    # PretrainConfig supplies typed defaults; CLI flags override individual
    # values.  The resulting dict is embedded in every checkpoint.
    training_cfg = PretrainConfig(
        expert_template=args.expert_template,
        num_initial_experts=args.num_initial_experts,
        steps_per_expert=args.steps_per_expert,
        prune_step_interval=args.prune_step_interval,
        max_recurrence=args.max_recurrence,
        max_experts=args.max_experts,
        lr=args.lr,
        router_lr=args.router_lr,
        router_loss_weight=args.router_loss_weight,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_steps=args.num_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )
    config_dict = training_cfg.as_dict()
    logger.info("Training config: %s", config_dict)

    # ----- Tokeniser -----
    logger.info("Loading tokeniser from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    vocab_size = len(tokenizer)
    logger.info("Vocabulary size: %d", vocab_size)

    # ----- Latent dimension -----
    if args.latent_dim is None:
        encoder_cfg = AutoConfig.from_pretrained(args.model_dir)
        latent_dim = encoder_cfg.hidden_size
        logger.info("Inferred latent_dim=%d from encoder config.", latent_dim)
    else:
        latent_dim = args.latent_dim

    # ----- Model -----
    logger.info("Building FinalTransformer (latent_dim=%d, vocab_size=%d)…", latent_dim, vocab_size)
    model = build_model(args, vocab_size, latent_dim)

    # Resume from checkpoint before moving to device to keep memory predictable
    start_step = 0
    optimizer_state = None
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        optimizer_state = ckpt.get("optimizer_state_dict")

    model.to(device)
    model.train()
    # Put the underlying Gemma model in train mode so dropout is active
    model.encoder.model.train()

    # ----- Optimiser -----
    optimizer = build_optimizer(model, args)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # ----- Dataset -----
    logger.info("Building UltraFineWeb dataset from %s", args.data_root)
    dataset = Dataset(
        data_root=args.data_root,
        batch_size=args.batch_size,
        similarity_delta=0.7,
        text_column="content",
        max_loaded_embeddings=100_000,
        device=device if torch.cuda.is_available() else None,
        tokenizer=tokenizer,
        max_length=args.max_length,
        padding="max_length",
    )

    # ----- Training loop -----
    logger.info(
        "Starting pretraining: %d steps (resuming from step %d).",
        args.num_steps,
        start_step,
    )
    data_iter = iter(dataset)
    global_step = start_step

    while global_step < args.num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart the dataset iterator when data is exhausted
            data_iter = iter(dataset)
            batch = next(data_iter)

        metrics, new_optimizer = train_step(model, batch, optimizer, args, vocab_size, device)
        # Replace the optimiser when pruning caused a rebuild
        if new_optimizer is not None:
            optimizer = new_optimizer
        global_step += 1

        if global_step % args.log_interval == 0:
            logger.info(
                "step %6d/%d | lm=%.4f | router=%.4f | total=%.4f | experts=%d",
                global_step,
                args.num_steps,
                metrics["lm_loss"],
                metrics["router_loss"],
                metrics["total_loss"],
                metrics["num_experts"],
            )

        if global_step % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"ckpt_step{global_step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, global_step, args, config_dict)

    # ----- Final checkpoint -----
    final_path = os.path.join(args.output_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, global_step, args, config_dict)
    logger.info("Pretraining complete.  Final model → %s", final_path)


if __name__ == "__main__":
    main()

"""training_config.py — Default hyperparameter configuration for pretraining.

All defaults that were previously spread across argparse arguments are
centralised here so that every run is fully reproducible.  The active
configuration is serialised into every checkpoint via :meth:`PretrainConfig.as_dict`.

Supported expert template names are listed in :data:`EXPERT_TEMPLATES`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass


# ---------------------------------------------------------------------------
# Expert template registry
# ---------------------------------------------------------------------------

#: Maps the ``expert_template`` config key to the fully-qualified class name
#: that will be resolved and instantiated in :mod:`pretrain`.
EXPERT_TEMPLATES: dict[str, str] = {
    "ExpertModule": "modules.model.expert.ExpertModule",
    "ExpertModuleWithSkip": "modules.model.expert.ExpertModuleWithSkip",
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    """Default hyperparameters for pretraining tiny-moe-llm.

    Attributes:
        expert_template: Name of the expert class to instantiate.  Must be one
            of the keys in :data:`EXPERT_TEMPLATES`.

        num_initial_experts: Number of experts pre-populated at model init.
        steps_per_expert: Normal-routing steps per MoE cycle before a new
            expert is solved and added.
        prune_step_interval: :class:`~modules.model.transformer.FinalTransformer`
            global-step interval at which the least-used expert is additionally
            pruned (independent of the ``max_experts`` cap).
        max_recurrence: Maximum number of recurrent MoE iterations during the
            inference / SFT forward pass.
        max_experts: Maximum number of live experts.  When this limit is reached
            the least-used expert is pruned before a new one is registered.

        lr: Base AdamW learning rate applied to encoder, decoder, and expert
            parameters.
        router_lr: AdamW learning rate for the router (often benefits from a
            higher value than the base rate).
        router_loss_weight: Coefficient multiplied with the router auxiliary
            cross-entropy loss before adding it to the LM loss.
        weight_decay: AdamW weight-decay regularisation.
        grad_clip: Gradient-norm clipping threshold.  Set to ``0`` to disable.

        batch_size: Number of samples per training batch.
        max_length: Maximum token sequence length; longer sequences are
            truncated.
        num_steps: Total number of pretraining gradient steps.
        log_interval: Print metrics every N steps.
        save_interval: Write a checkpoint every N steps.
    """

    # ---- Expert architecture ----
    expert_template: str = "ExpertModule"

    # ---- Model architecture ----
    num_initial_experts: int = 2
    steps_per_expert: int = 100
    prune_step_interval: int = 500
    max_recurrence: int = 10
    max_experts: int = 16

    # ---- Optimizer ----
    lr: float = 1e-4
    router_lr: float = 1e-3
    router_loss_weight: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # ---- Training loop ----
    batch_size: int = 4
    max_length: int = 512
    num_steps: int = 10_000
    log_interval: int = 10
    save_interval: int = 1_000

    def as_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON / pickle serialisation."""
        return asdict(self)

    def copy(self) -> "PretrainConfig":
        """Return a shallow copy of this config."""
        return copy.copy(self)

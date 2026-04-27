"""training_config.py — Default hyperparameter configuration for pretraining.

All defaults that were previously spread across argparse arguments are
centralised here so that every run is fully reproducible.  The active
configuration is serialised into every checkpoint via :meth:`PretrainConfig.as_dict`.

Supported expert template names are listed in :data:`EXPERT_TEMPLATES`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import importlib


# ---------------------------------------------------------------------------
# Expert template registry
# ---------------------------------------------------------------------------

#: Maps the ``expert_template`` config key to the fully-qualified class name
#: that will be resolved and instantiated in :mod:`pretrain`.
EXPERT_TEMPLATES: dict[str, str] = {
    "ExpertModule": "modules.model.expert.ExpertModule",
    "ExpertModuleWithSkip": "modules.model.expert.ExpertModuleWithSkip",
    "ExpertModuleWithSkipAndEmbedding": "modules.model.expert.ExpertModuleWithSkipAndEmbedding",
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
    expert_template: str = "ExpertModuleWithSkip"

    # ---- Model architecture ----
    hidden_size: int = 704 # 1408
    num_initial_experts: int = 16
    steps_per_expert: int = 100
    prune_step_interval: int = 500
    max_recurrence: int = 10
    max_experts: int = 256
    intermediate_size: int = 352 # 704
    num_gemma_layers: int = 8
    ir_num_entries: int = 65536 # 16384

    # ---- Optimizer ----
    lr: float = 1e-4
    router_lr: float = 1e-3
    router_loss_weight: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # ---- Training loop ----
    batch_size: int = 8
    max_length: int = 4096
    num_steps: int = 10_000
    log_interval: int = 10
    save_interval: int = 1_000

    def as_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON / pickle serialisation."""
        return asdict(self)

    def copy(self) -> "PretrainConfig":
        """Return a shallow copy of this config."""
        return copy.copy(self)


if __name__ == "__main__":
    # build model and print param count distribution
    from modules.model.transformer import FinalTransformer
    config = PretrainConfig()
    config_dict = config.as_dict()
    expert_name = EXPERT_TEMPLATES[config.expert_template]
    module_name, class_name = expert_name.rsplit(".", 1)
    expert_cls = getattr(importlib.import_module(module_name), class_name)
    config_dict["expert_template"] = expert_cls(config_dict["hidden_size"], config_dict["hidden_size"])
    
    config_dict["hidden_size"] = 704 # 960
    config_dict["intermediate_size"] = 352 # 480
    
    model = FinalTransformer(**config_dict)
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} params")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    
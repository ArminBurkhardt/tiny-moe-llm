import yaml


with open("config.yaml", "r") as f:
    Config = yaml.safe_load(f)


class ModelConfig:
    Params = {
        "vocab_size": int(Config["model"]["vocab_size"]),
        "max_seq_len": int(Config["model"]["max_seq_length"]),
        "hidden_size": int(Config["model"]["hidden_size"]),
        "intermediate_size": int(Config["model"]["intermediate_size"]),
        # routed + shared MoE experts only (Gemma4TextModel keeps plain intermediate_size);
        # defaults to intermediate_size so config.yaml can omit the key entirely.
        "moe_intermediate_size": int(Config["model"].get("moe_intermediate_size", Config["model"]["intermediate_size"])),
        "num_layers": int(Config["model"]["num_layers"]),
        "num_heads": int(Config["model"]["num_attention_heads"]),
        "head_dim": int(Config["model"]["head_dim"]),
        "num_mlp_experts": int(Config["model"]["num_mlp_experts"]),
        "num_attn_experts": int(Config["model"]["num_attn_experts"]),
        "num_ir_experts": int(Config["model"]["num_ir_experts"]),
        "num_ir_entries": int(Config["model"]["num_ir_entries"]),
        "ir_dim": int(Config["model"]["ir_dim"]),
        "dropout": float(Config["model"]["dropout"]),
        "top_k": int(Config["model"]["top_k"]),
        "n_loops": int(Config["model"]["n_loops"]),
        "ple_embeddings_size": int(Config["model"]["per_layer_embeddings_size"]),
        "mtp_num_extra_tokens": int(Config["model"]["mtp_num_extra_tokens"]),
        "lm_head_factor": int(Config["model"]["lm_head_factor"]),
    }
    
    Forward = {}

class TrainingConfig:
    Batch_size = int(Config["training"]["batch_size"])
    Seq_length = int(Config["training"]["seq_length"])
    lambda_mtp = float(Config["training"]["lambda_mtp"])
    aux_loss_weight = float(Config["training"]["aux_loss_weight"])
    num_epochs = int(Config["training"]["num_epochs"])
    learning_rate = float(Config["training"]["lr"])
    lr = float(Config["training"]["lr"])
    weight_decay = float(Config["training"]["weight_decay"])
    grad_clip = float(Config["training"]["grad_clip"])
    target_tokens = int(Config["training"]["target_tokens"])
    grad_accumulation_steps = int(Config["training"].get("grad_accumulation_steps", 1))
    total_steps = int((Config["training"]["target_tokens"] // (Config["training"]["batch_size"] * Config["training"]["seq_length"] * grad_accumulation_steps)))
    warmup_steps = int(Config["training"].get("warmup_steps", 0))
    noise_anneal_tokens = int(Config["training"].get("noise_anneal_tokens", 0))
    seed = int(Config["training"].get("seed", 42))
    data_dir = str(Config["training"].get("data_dir", "data/prepared"))
    phase = str(Config["training"].get("phase", "phase1"))

    # per-loop CE supervision: ascending weights, one per loop, so lm_head has
    # *some* incentive to make intermediate loops' hidden states legible without competing with
    # the final loop's dominant supervision. Consumed directly in compute_mtp_loss's call sites,
    # not passed into the model.
    loop_ce_weights = [float(w) for w in Config["training"]["loop_ce_weights"]]
    assert len(loop_ce_weights) == ModelConfig.Params["n_loops"], (
        f"loop_ce_weights ({loop_ce_weights}) must have exactly one weight per loop "
        f"(n_loops={ModelConfig.Params['n_loops']})"
    )

    # fraction of token positions supervised on the non-final loops. the final loop is always
    # supervised in full, so this only cheapens the low-weight intermediate readouts.
    loop_ce_subsample = float(Config["training"].get("loop_ce_subsample", 1.0))
    assert 0.0 < loop_ce_subsample <= 1.0, (
        f"loop_ce_subsample ({loop_ce_subsample}) must be in (0, 1]"
    )

    # stochastic loop depth: probability a step runs a reduced loop count (uniform 1..n_loops-1).
    # this is the whole depth policy during training now that the halt head is gone -- the
    # inference-time criterion (TinyMoETransformer's converge_tol) is parameter-free and needs
    # every depth to be a real operating point, which is exactly what this trains.
    # per-loop CE already supervises every prefix depth; this additionally makes the *model* see
    # shallow depths as real inputs to the rest of training, so an inference-time loop-count
    # override lands on a depth the model was actually trained at. 0.0 = always full depth.
    loop_count_sampling = float(Config["training"].get("loop_count_sampling", 0.0))
    assert 0.0 <= loop_count_sampling <= 1.0, (
        f"loop_count_sampling ({loop_count_sampling}) must be in [0, 1]"
    )

    # checkpoint lifecycle for the unattended run
    checkpoint_every_tokens = int(Config["training"].get("checkpoint_every_tokens", 400_000_000))
    keep_local_checkpoints = int(Config["training"].get("keep_local_checkpoints", 2))
    # None means "key absent, fall back to utils.HF_UPLOAD_REPO"; an explicit "" means "uploads
    # off". Collapsing the two (defaulting to "" and then `value or HF_UPLOAD_REPO` at the call
    # site) makes `hf_upload_repo: ""` silently upload anyway, which is the opposite of what the
    # yaml comment promises -- and it is only noticed once a 2GB checkpoint is already on the Hub.
    _raw_upload_repo = Config["training"].get("hf_upload_repo", None)
    hf_upload_repo = None if _raw_upload_repo is None else str(_raw_upload_repo)

    @classmethod
    def upload_repo(cls, default: str) -> str:
        """Resolve which repo to upload to.

        Args:
            default: utils.HF_UPLOAD_REPO, used only when config.yaml has no key at all.

        Returns:
            The repo id, or "" when uploads are explicitly disabled.
        """
        return default if cls.hf_upload_repo is None else cls.hf_upload_repo

    # phase 1 gets this fraction of target_tokens, phase 2 the rest. target_tokens itself stays
    # the COMBINED budget so total_steps and the cosine LR anchor are unchanged -- phase 2 must
    # continue the decay from where phase 1 left it, not restart it.
    phase1_fraction = float(Config["training"].get("phase1_fraction", 0.85))

    @classmethod
    def phase_target_tokens(cls, phase: str) -> int:
        """Token count at which the given phase stops training."""
        if phase == "phase1":
            return int(cls.target_tokens * cls.phase1_fraction)
        if phase == "phase2":
            return cls.target_tokens
        raise ValueError(f"unknown phase {phase!r}; expected 'phase1' or 'phase2'")


class SFTConfig:
    """Supervised fine-tuning knobs, read from config.yaml's ``sft:`` block.

    Only the things SFT genuinely does differently live here. Every loss weight -- lambda_mtp,
    aux_loss_weight, loop_ce_weights/loop_ce_subsample, loop_count_sampling and the whole ponder
    family -- is read from ``TrainingConfig`` by ``scripts/pretrain.train_step``, which
    ``scripts/sft.py`` reuses unchanged. That reuse is the point: the cheapest way to guarantee
    the objective stays *identical* across the two runs is to not have a second copy of it.
    """
    _Block = Config.get("sft", {}) or {}

    data_dir = str(_Block.get("data_dir", "data/prepared"))
    train_split = str(_Block.get("train_split", "sft_train"))
    val_split = str(_Block.get("val_split", "sft_val"))

    Batch_size = int(_Block.get("batch_size", 4))
    Seq_length = int(_Block.get("seq_length", 4096))
    grad_accumulation_steps = int(_Block.get("grad_accumulation_steps", 8))
    lr = float(_Block.get("lr", 3e-5))
    weight_decay = float(_Block.get("weight_decay", 0.01))
    num_epochs = int(_Block.get("num_epochs", 2))
    warmup_fraction = float(_Block.get("warmup_fraction", 0.03))
    lr_min_factor = float(_Block.get("lr_min_factor", 0.05))
    dropout = float(_Block.get("dropout", 0.05))
    # seeds the SFTDataset per-epoch document permutation. Changing it mid-run repoints every
    # checkpointed resume position into a different order, so it is checkpointed alongside them.
    seed = int(_Block.get("seed", 1234))

    checkpoint_every_tokens = int(_Block.get("checkpoint_every_tokens", 25_000_000))
    keep_local_checkpoints = int(_Block.get("keep_local_checkpoints", 3))
    eval_every_tokens = int(_Block.get("eval_every_tokens", 25_000_000))
    eval_max_batches = int(_Block.get("eval_max_batches", 40))

    # same None-vs-"" distinction as TrainingConfig.hf_upload_repo: absent means "fall back to
    # utils.HF_UPLOAD_REPO", explicit "" means uploads off (the default for a local run).
    _raw_upload_repo = _Block.get("hf_upload_repo", "")
    hf_upload_repo = None if _raw_upload_repo is None else str(_raw_upload_repo)

    @classmethod
    def upload_repo(cls, default: str) -> str:
        """Resolve which repo to upload SFT checkpoints to ("" disables uploads)."""
        return default if cls.hf_upload_repo is None else cls.hf_upload_repo

    @classmethod
    def model_params(cls) -> dict:
        """``ModelConfig.Params`` with the SFT dropout override applied.

        Dropout is not a parameter, so this changes nothing about checkpoint compatibility -- the
        pretrained state dict loads into the overridden model unchanged.
        """
        return {**ModelConfig.Params, "dropout": cls.dropout}


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
    # ponder loss (PLAN.md Step 3b): held at 0 while loop_scale grows, then ramped -- see the
    # deadlock note on LoopMixtureOfExperts.loop_scale for why the warmup is load-bearing.
    lambda_ponder = float(Config["training"].get("lambda_ponder", 3e-3))
    ponder_warmup_tokens = int(Config["training"].get("ponder_warmup_tokens", 1_000_000_000))
    ponder_ramp_tokens = int(Config["training"].get("ponder_ramp_tokens", 1_000_000_000))

    # per-loop CE supervision (PLAN.md Step 4a): ascending weights, one per loop, so lm_head has
    # *some* incentive to make intermediate loops' hidden states legible without competing with
    # the final loop's dominant supervision. Same yaml-block deviation as the ponder knobs above --
    # consumed directly in compute_mtp_loss's call sites, not passed into the model.
    loop_ce_weights = [float(w) for w in Config["training"]["loop_ce_weights"]]
    assert len(loop_ce_weights) == ModelConfig.Params["n_loops"], (
        f"loop_ce_weights ({loop_ce_weights}) must have exactly one weight per loop "
        f"(n_loops={ModelConfig.Params['n_loops']})"
    )

    # correctness head (PLAN.md Step 4b): weight on the correct_proj BCE loss, final loop only.
    # same yaml-block deviation as the ponder/loop_ce knobs above -- consumed directly in
    # compute_mtp_loss's call sites, not passed into the model.
    lambda_conf = float(Config["training"].get("lambda_conf", 0.05))


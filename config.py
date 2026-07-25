import yaml


with open("config.yaml", "r") as f:
    Config = yaml.safe_load(f)


class ModelConfig:
    Params = {
        "vocab_size": int(Config["model"]["vocab_size"]),
        "max_seq_len": int(Config["model"]["max_seq_length"]),
        "hidden_size": int(Config["model"]["hidden_size"]),
        "intermediate_size": int(Config["model"]["intermediate_size"]),
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
    
    Forward = {
        "identity_skew": float(Config["model"]["identity_skew"]),
    }
    
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
    max_tokens_per_shard = int(Config["training"].get("max_tokens_per_shard", 200_000_000))


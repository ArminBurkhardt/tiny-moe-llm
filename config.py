import yaml


with open("config.yaml", "r") as f:
    Config = yaml.safe_load(f)


class ModelConfig:
    Params = {
        "vocab_size": Config["model"]["vocab_size"],
        "max_seq_len": Config["model"]["max_seq_length"],
        "hidden_size": Config["model"]["hidden_size"],
        "intermediate_size": Config["model"]["intermediate_size"],
        "num_layers": Config["model"]["num_layers"],
        "num_heads": Config["model"]["num_attention_heads"],
        "head_dim": Config["model"]["head_dim"],
        "num_mlp_experts": Config["model"]["num_mlp_experts"],
        "num_attn_experts": Config["model"]["num_attn_experts"],
        "num_ir_experts": Config["model"]["num_ir_experts"],
        "num_ir_entries": Config["model"]["num_ir_entries"],
        "ir_dim": Config["model"]["ir_dim"],
        "dropout": Config["model"]["dropout"],
        "top_k": Config["model"]["top_k"],
        "n_loops": Config["model"]["n_loops"],
        "ple_embeddings_size": Config["model"]["per_layer_embeddings_size"],
    }
    
    Forward = {
        "identity_skew": Config["model"]["identity_skew"],
    }
    
class TrainingConfig:
    Batch_size = Config["training"]["batch_size"]
    Seq_length = Config["training"]["seq_length"]
    




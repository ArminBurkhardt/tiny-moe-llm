"""``TINY_LLM_CONFIG`` selects the yaml ``config.py`` reads, and unset means ``config.yaml`` as before.

Each reading runs in a subprocess because ``config.py`` parses its yaml once at import. Checks:

1. with ``TINY_LLM_CONFIG=config_micro.yaml`` the model shape is the ~30M one (hidden 256, 4 layers,
   exact IR read), the loop CE weights have one entry per loop, the loss weights and the evidence
   profile are the micro ones, and the shape constraints the model asserts at construction hold;
2. unset, every value equals what ``config.yaml`` says;
3. a path that does not exist fails at import, loudly;
4. neither import pulls in ``modules.model`` (so this test needs no GPU and no transformer_engine).

GPU free.
"""
import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml

READ = """
import json, sys
from config import ModelConfig, TrainingConfig, SFTConfig, EvidenceConfig
print(json.dumps({
    "params": ModelConfig.Params,
    "loop_ce_weights": TrainingConfig.loop_ce_weights,
    "lambda_mtp": TrainingConfig.lambda_mtp,
    "evidence_selection_weight": TrainingConfig.evidence_selection_weight,
    "groundedness_weight": TrainingConfig.groundedness_weight,
    "target_tokens": TrainingConfig.target_tokens,
    "sft_seq_length": SFTConfig.Seq_length,
    "sft_data_dir": SFTConfig.data_dir,
    "evidence": {
        "train_split": EvidenceConfig.train_split, "val_split": EvidenceConfig.val_split,
        "fixed_split": EvidenceConfig.fixed_split, "kill_tokens": EvidenceConfig.kill_tokens,
        "batch": EvidenceConfig.Batch_size, "accum": EvidenceConfig.grad_accumulation_steps,
        "lr": EvidenceConfig.lr, "fresh_lr": EvidenceConfig.fresh_lr,
        "max_evidence_tokens": EvidenceConfig.max_evidence_tokens,
        "conversation_loss_weighting": EvidenceConfig.conversation_loss_weighting,
        "cluster_refresh_tokens": EvidenceConfig.cluster_refresh_tokens,
        "dead_quantile": EvidenceConfig.dead_quantile,
    },
    "model_imported": any(m == "modules.model" or m.startswith("modules.model.") for m in sys.modules),
}))
"""


def read(config_path):
    env = {k: v for k, v in os.environ.items() if k != "TINY_LLM_CONFIG"}
    if config_path is not None:
        env["TINY_LLM_CONFIG"] = config_path
    done = subprocess.run([sys.executable, "-c", READ], cwd=ROOT, env=env, capture_output=True, text=True)
    return done


def main():
    # 1. the micro shape
    done = read("config_micro.yaml")
    assert done.returncode == 0, done.stderr
    micro = json.loads(done.stdout.strip().splitlines()[-1])
    p = micro["params"]
    assert p["hidden_size"] == 256 and p["num_layers"] == 4 and p["num_heads"] == 4
    assert p["head_dim"] == 64 and p["max_seq_len"] == 1024 and p["n_loops"] == 3
    assert p["num_mlp_experts"] == 8 and p["top_k"] == 2 and p["moe_intermediate_size"] == 768
    assert p["ir_dim"] == 384 and p["num_ir_entries"] == 256 and p["ir_num_clusters"] == 0
    assert p["ir_direct_read"] is True and p["evidence_encoder"] is True
    assert p["ple_embeddings_size"] == 8 and p["mtp_num_extra_tokens"] == 2
    assert len(micro["loop_ce_weights"]) == 3 == p["n_loops"]
    assert micro["lambda_mtp"] == 0.1 and micro["evidence_selection_weight"] == 0.1
    assert micro["groundedness_weight"] == 0.0 and micro["target_tokens"] == 300_000_000
    assert micro["sft_seq_length"] == 1024 and micro["sft_data_dir"] == "data/prepared_inject"
    ev = micro["evidence"]
    assert ev["train_split"] == "inject_full_train" and ev["val_split"] == "inject_val"
    assert ev["fixed_split"] == "" and ev["kill_tokens"] == 0
    assert ev["batch"] == 16 and ev["accum"] == 2 and ev["max_evidence_tokens"] == 3072
    assert ev["lr"] == ev["fresh_lr"] == 1e-3 and ev["conversation_loss_weighting"] is False
    assert ev["cluster_refresh_tokens"] == 0 and ev["dead_quantile"] == 0.0
    # the constructor's own assertions, restated: a bad shape would otherwise fail on the GPU box
    factor = p["lm_head_factor"]
    assert p["vocab_size"] % factor == 0 and p["hidden_size"] % factor == 0
    assert p["vocab_size"] % (2 * factor) == 0 and (p["hidden_size"] // 2) % (2 * factor) == 0
    assert p["num_ir_entries"] <= 65536 and p["vocab_size"] <= 65536
    assert p["hidden_size"] % p["num_heads"] == 0 and p["num_heads"] % 4 == 0
    assert not micro["model_imported"]
    print("1. TINY_LLM_CONFIG=config_micro.yaml gives the micro shape and profile     PASS")

    # 2. unset is today's behaviour
    done = read(None)
    assert done.returncode == 0, done.stderr
    default = json.loads(done.stdout.strip().splitlines()[-1])
    with open(os.path.join(ROOT, "config.yaml")) as f:
        raw = yaml.safe_load(f)
    assert default["params"]["hidden_size"] == raw["model"]["hidden_size"] == 768
    assert default["params"]["num_layers"] == raw["model"]["num_layers"]
    assert default["params"]["max_seq_len"] == raw["model"]["max_seq_length"]
    assert default["params"]["num_ir_entries"] == raw["model"]["num_ir_entries"]
    assert default["loop_ce_weights"] == raw["training"]["loop_ce_weights"]
    assert default["target_tokens"] == raw["training"]["target_tokens"]
    assert default["evidence"]["train_split"] == raw["evidence"]["train_split"]
    assert default["evidence"]["fixed_split"] == raw["evidence"]["fixed_split"]
    assert default["evidence"]["batch"] == raw["evidence"]["batch_size"]
    assert default != micro and not default["model_imported"]
    print("2. unset: every value is config.yaml's                                     PASS")

    # 3. a missing file fails loudly
    done = read("no_such_config.yaml")
    assert done.returncode != 0 and "FileNotFoundError" in done.stderr
    print("3. a missing config file fails at import                                   PASS")
    print("all config override checks passed")


if __name__ == "__main__":
    main()

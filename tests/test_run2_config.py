"""``config_run2.yaml`` and ``config_run2_control.yaml``: every decided override of the real run's spec,
read through ``config.py`` the way the trainer reads it (a subprocess per yaml, ``TINY_LLM_CONFIG`` set,
because ``config.py`` parses its yaml once at import). Checks:

1. the shape is shape L as ``config.yaml`` has it (width 768, 8 layers, 12 heads, 32 experts top 2, 3 loops,
   PLE 32, MTP 2, lm_head_factor 4), and ``num_ir_entries`` 256 with ``ir_num_clusters`` 0 (the exact read)
   so ``num_ir_entries % ir_num_clusters`` cannot fail and the capacity rule is not in play;
2. the decided training overrides: weight decay 0.1, loss-free balancing, segment weighting, no retriever
   term, AdamW, lr 4e-4, clip 1.0, per loop CE weights, depth sampling 0.3, noise anneal 1B;
3. the batch: 8 per GPU x accumulation 2 x 8 GPUs x 4096 = 524,288 tokens per update;
4. the evidence block: warm-up split, inject_val, evidence_fixed, kill at 5B / 0.1, frozen gate, plain CE,
   a save every 1B, the cosine floor 0.1;
5. the control differs from the run only in the split, the token budget, the epochs, the warmup fraction
   and the kill, and its warmup is the same 1000 steps of its own length.

GPU free; neither import pulls in ``modules.model``.
"""
import os, sys, json, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = r"""
import json, sys
import config as C
E = C.EvidenceConfig
T = C.TrainingConfig
P = C.ModelConfig.Params
out = {
    "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in P.items()},
    "weight_decay": T.weight_decay, "moe_balancing": T.moe_balancing, "segment": T.evidence_selection_weighting,
    "contrastive_weight": T.evidence_contrastive_weight, "contrastive_loops": list(T.evidence_contrastive_loops or []),
    "optimizer": getattr(T, "optimizer", None), "lr": T.lr, "grad_clip": T.grad_clip,
    "loop_ce_weights": list(T.loop_ce_weights), "loop_count_sampling": T.loop_count_sampling,
    "noise_anneal_tokens": T.noise_anneal_tokens, "target_tokens": T.target_tokens,
    "warmup_steps": T.warmup_steps,
    "ev": {k: getattr(E, k) for k in ("train_split", "val_split", "fixed_split", "kill_tokens", "kill_min_gain",
           "freeze_evidence_gate", "lr", "fresh_lr", "num_epochs", "weight_decay", "Batch_size",
           "grad_accumulation_steps", "warmup_fraction", "lr_min_factor", "checkpoint_every_tokens",
           "eval_every_tokens", "conversation_loss_weighting", "max_evidence_tokens", "seed")},
    "modules_model_imported": "modules.model" in sys.modules,
}
print("JSON" + json.dumps(out))
"""


def read(cfg):
    env = dict(os.environ, TINY_LLM_CONFIG=os.path.join(ROOT, cfg))
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{cfg} failed to load: {r.stderr[-2000:]}"
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")][-1]
    return json.loads(line[4:])


def main():
    run = read("config_run2.yaml")
    ctl = read("config_run2_control.yaml")
    for name, c in (("run", run), ("control", ctl)):
        p = c["params"]
        assert not c["modules_model_imported"], name
        assert p["hidden_size"] == 768 and p["intermediate_size"] == 2304 and p["moe_intermediate_size"] == 2304, name
        assert p["num_layers"] == 8 and p["num_heads"] == 12 and p["max_seq_len"] == 4096, name
        assert p["n_loops"] == 3 and p["top_k"] == 2, name
        assert p["num_ir_entries"] == 256 and p["ir_num_clusters"] == 0, name
        # the IR constraint: clusters 0 is the exact read; any other count must divide the entries
        assert p["ir_num_clusters"] == 0 or p["num_ir_entries"] % p["ir_num_clusters"] == 0, name
        assert p["ir_dim"] == 384 and p["vocab_size"] == 65536, name
        mlp_keys = [k for k in p if "mlp_experts" in k]
        assert mlp_keys and all(p[k] == 32 for k in mlp_keys), (name, {k: p[k] for k in p})
        assert c["weight_decay"] == 0.1 and c["ev"]["weight_decay"] == 0.1, name
        assert c["moe_balancing"] == "loss_free", name
        assert c["segment"] == "segment", name
        assert c["contrastive_weight"] == 0.0 and c["contrastive_loops"] == [0], name
        assert c["optimizer"] == "adamw", name
        assert c["lr"] == 4.0e-4 and c["ev"]["lr"] == 4.0e-4 and c["ev"]["fresh_lr"] == 4.0e-4, name
        assert c["grad_clip"] == 1.0, name
        assert c["loop_ce_weights"] == [0.2, 0.3, 1.0] and c["loop_count_sampling"] == 0.3, name
        assert c["noise_anneal_tokens"] == 1_000_000_000 and c["warmup_steps"] == 1000, name
        ev = c["ev"]
        assert ev["Batch_size"] == 8 and ev["grad_accumulation_steps"] == 2, name
        assert ev["Batch_size"] * ev["grad_accumulation_steps"] * 8 * 4096 == 524_288, name
        assert ev["val_split"] == "inject_val" and ev["fixed_split"] == "evidence_fixed", name
        assert ev["freeze_evidence_gate"] is True and ev["conversation_loss_weighting"] is False, name
        assert ev["lr_min_factor"] == 0.1 and ev["checkpoint_every_tokens"] == 1_000_000_000, name
        assert ev["max_evidence_tokens"] == 9216, name
    assert run["ev"]["train_split"] == "run2_warm_train"
    assert run["target_tokens"] == 280_000_000_000
    assert run["ev"]["kill_tokens"] == 5_000_000_000 and run["ev"]["kill_min_gain"] == 0.1
    assert run["ev"]["num_epochs"] == 3
    assert ctl["ev"]["train_split"] == "run2_control_train"
    assert ctl["target_tokens"] == 10_000_000_000 and ctl["ev"]["kill_tokens"] == 0 and ctl["ev"]["num_epochs"] == 1
    # the warmup is 1000 optimizer steps in both: fraction x steps, steps = tokens / 524,288
    for c, tokens in ((run, 280e9), (ctl, 10e9)):
        steps = tokens / 524_288
        assert abs(c["ev"]["warmup_fraction"] * steps - 1000) < 15, (c["ev"]["warmup_fraction"], steps)
    # everything else identical between the two files
    drop = {"train_split", "kill_tokens", "num_epochs", "warmup_fraction"}
    assert {k: v for k, v in run["ev"].items() if k not in drop} == {k: v for k, v in ctl["ev"].items() if k not in drop}
    assert run["params"] == ctl["params"]
    for k in ("weight_decay", "moe_balancing", "segment", "lr", "loop_ce_weights", "noise_anneal_tokens"):
        assert run[k] == ctl[k], k
    print("test_run2_config: ok")


if __name__ == "__main__":
    main()

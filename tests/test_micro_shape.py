"""The ~30M shape in ``config_micro.yaml`` builds, runs, and keeps the port's neutrality guarantees.

1. **Size.** The model builds under ``TINY_LLM_CONFIG=config_micro.yaml`` and has 25M to 40M
   parameters in total. The count, the active count and the FLOP estimate are printed so the run spec
   can quote them.
2. **No evidence is bit-identical to a model built without the port**, given the same weights.
3. **A fresh port with evidence attached is neutral too** (the reader's output projection is zero), and
   once that projection is nonzero the logits move and stay finite.
4. **The training forward** (``return_hidden``) returns one normed hidden state per loop at the micro
   width, with and without evidence.

GPU required (flash varlen, transformer_engine). The config is selected here, before ``config`` is
imported, so run it from anywhere; it changes into the repo root itself.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ["TINY_LLM_CONFIG"] = "config_micro.yaml"

import torch

from config import ModelConfig
from modules.model.transformer import TinyMoETransformer

BF16 = torch.bfloat16
DEVICE = "cuda"


def build(evidence_port, seed=0):
    torch.manual_seed(seed)
    model = TinyMoETransformer(**{**ModelConfig.Params, "evidence_port": evidence_port}).to(DEVICE).to(BF16)
    model.eval()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    return model


def main():
    params = ModelConfig.Params
    assert params["hidden_size"] == 256 and params["num_layers"] == 4, "the micro config was not selected"

    plain, ported = build(False), build(True)
    missing = ported.load_state_dict(plain.state_dict(), strict=False).missing_keys
    port_only = ("shared_evidence", "key_adapter", "value_adapter", "log_memory_scale",
                 "evidence_query_bias", "evidence_loop_scale", "evidence_gate_scale")
    assert all(any(p in k for p in port_only) for k in missing), missing

    # 1. size
    total = sum(p.numel() for p in ported.parameters())
    experts = sum(p.numel() for p in ported.moe.parallel_experts.parameters())
    active = total - experts + int(experts * params["top_k"] / params["num_mlp_experts"])
    embedding = sum(p.numel() for n, p in ported.named_parameters() if "embed_tokens" in n)
    print(f"micro shape: {total / 1e6:.1f}M total, {active / 1e6:.1f}M active, {embedding / 1e6:.1f}M in "
          f"the token table, {ported.flops_per_token_fwd / 1e6:.0f}M FLOP/token fwd at seq "
          f"{params['max_seq_len']}")
    assert 25e6 <= total <= 40e6, f"{total / 1e6:.1f}M parameters is outside 25M to 40M"
    print("1. the micro shape builds at 25M to 40M parameters                      PASS")

    B, S = 2, 48
    vocab = params["vocab_size"]
    input_ids = torch.randint(1, vocab, (B, S), device=DEVICE)
    with torch.inference_mode():
        base = plain(input_ids, skip_mtp=True)
        no_corpus = ported(input_ids, skip_mtp=True)
    assert torch.equal(base, no_corpus), (base - no_corpus).abs().max().item()
    print("2. no evidence attached is bit-identical to the port-free model         PASS")

    # one evidence set per query segment; an unpacked row is one segment
    S_ev, C = 16, 4
    ev_ids = torch.randint(1, vocab, (B, S_ev), device=DEVICE)
    ev_chunk = torch.tensor([[0] * 8 + [1] * 8, [2] * 8 + [3] * 8], device=DEVICE)
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device=DEVICE)
    chunk_segments = torch.tensor([0, 0, 1, 1], device=DEVICE)
    keys = torch.randn(C, params["ir_dim"], device=DEVICE, dtype=BF16)
    evidence = ported.build_evidence(ev_ids, ev_chunk, ev_segment, B,
                                     chunk_keys=keys, chunk_segments=chunk_segments)
    with torch.inference_mode():
        zero_reader = ported(input_ids, skip_mtp=True, evidence=evidence)
    assert torch.equal(base, zero_reader), (base - zero_reader).abs().max().item()
    with torch.no_grad():
        ported.moe.shared_evidence.attn.o_proj.weight.normal_(0, 0.02)
    with torch.inference_mode():
        awake = ported(input_ids, skip_mtp=True, evidence=evidence)
    assert torch.isfinite(awake.float()).all()
    assert (awake - base).abs().max().item() > 0.0, "a nonzero reader changed nothing"
    print("3. a fresh port is neutral with evidence attached, and moves once it is not zero   PASS")

    # 4. the training forward
    ported.train()
    ids = torch.randint(1, vocab, (B, S), device=DEVICE)
    for attached in (None, evidence):
        out = ported(ids, return_hidden=True, evidence=attached)
        hidden = out[0] if isinstance(out, tuple) else out
        assert hidden.shape == (params["n_loops"], B, S, params["hidden_size"]), hidden.shape
        assert torch.isfinite(hidden.float()).all()
    print("4. the training forward returns one hidden state per loop at the micro width   PASS")
    print("all micro shape checks passed")


if __name__ == "__main__":
    main()

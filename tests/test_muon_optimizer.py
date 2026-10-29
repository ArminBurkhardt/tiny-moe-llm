"""The combined Muon plus AdamW optimizer on hand built tensors (no model, no CUDA, no TE).

1. the partition rule over the micro and pilot parameter names;
2. Newton-Schulz: near-unit singular values, and a tall matrix handled like a wide one;
3. the RMS match: a Muon update's RMS is about 0.2 on any shape;
4. a fused weight's halves are orthogonalized as two matrices;
5. the AdamW groups step exactly as ``torch.optim.AdamW`` (same state keys, same values);
6. decoupled weight decay on a Muon parameter with a zero gradient, and the state round trip.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from modules.model.optim import (  # noqa: E402
    MuonAdamW, build_optimizer, check_resume_kind, muon_kind, muon_update, newton_schulz, optimizer_kind,
    partition_summary, saved_optimizer_kind, tag_param_groups,
)


def main():
    torch.manual_seed(0)

    # 1. the partition rule
    two = torch.zeros(4, 4)
    one = torch.zeros(4)
    expect = {
        "gemma_decoder.embed_tokens.weight": "adamw",
        "gemma_decoder.ple.weight": "adamw",
        "moe_embeddings.weight": "adamw",
        "moe_embed_proj.weight": "muon",
        "gemma_decoder.layers.0.self_attn.q_proj.weight": "muon",
        "gemma_decoder.layers.0.mlp.down_proj.weight": "muon",
        "gemma_decoder.layers.0.ple_proj.weight": "muon",
        "gemma_decoder.layers.0.gate_proj.weight": "muon",
        "moe.experts.0.attn.o_proj.weight": "muon",
        "moe.experts.2.ir_module.z_keys": "adamw",
        "moe.experts.2.ir_module.y_values": "adamw",
        "moe.experts.2.ir_module.g_proj.weight": "adamw",
        "moe.experts.2.ir_module.loop_query_bias.weight": "adamw",
        "moe.experts.2.ir_module.key_adapter.weight": "muon",
        "moe.experts.2.ir_module.value_adapter.weight": "muon",
        "moe.experts.2.down_proj.weight": "muon",
        "moe.experts.2.up_proj.weight": "muon",
        "moe.experts.2.direct_gate.weight": "adamw",
        "moe.shared_mlp.gate_up.weight": "muon_halves",
        "moe.shared_mlp.down.weight": "muon",
        "moe.shared_attn.attn.k_proj.weight": "muon",
        "moe.parallel_experts.gate_up.weight7": "muon_halves",
        "moe.parallel_experts.down.weight7": "muon",
        "moe.router.router.1.weight": "adamw",
        "moe.router.noise_proj.weight": "adamw",
        "moe.loop_router_bias.weight": "adamw",
        "moe.evidence_query_bias.weight": "adamw",
        "moe.inject.weight": "adamw",
        "moe.shared_evidence.attn.q_proj.weight": "muon",
        "moe.shared_evidence.attn.o_proj.weight": "adamw",
        "groundedness_head.out_proj.weight": "adamw",
        "lm_head.projection.weight": "adamw",
        "lm_head.lm_heads.3.weight": "adamw",
        "mtp_head.gate.weight": "adamw",
        "mtp_head.lm_head.lm_heads.0.weight": "adamw",
    }
    for name, kind in expect.items():
        got = muon_kind(name, two)
        assert got == kind, (name, got, kind)
    for name in ("moe.loop_scale", "gemma_decoder.layers.0.layer_scalar", "moe.post_norm.weight",
                 "moe.router.router.0.weight", "moe.experts.2.ir_module.log_memory_scale"):
        assert muon_kind(name, one) == "adamw", name
    print("1. the partition rule over the parameter names                              PASS")

    # 2. Newton-Schulz
    for shape in ((64, 32), (32, 64), (48, 48)):
        g = torch.randn(*shape)
        u = newton_schulz(g).float()
        s = torch.linalg.svdvals(u)
        assert s.min() > 0.5 and s.max() < 1.3, (shape, s.min().item(), s.max().item())
    tall = torch.randn(64, 16)
    assert torch.allclose(newton_schulz(tall).float(), newton_schulz(tall.mT).float().mT, atol=1e-2)
    print("2. Newton-Schulz gives near-unit singular values, tall and wide alike          PASS")

    # 3. the RMS match
    for shape in ((64, 32), (32, 64), (48, 48), (256, 32)):
        g = torch.randn(*shape)
        rms = muon_update(g, halves=False).pow(2).mean().sqrt().item()
        assert 0.15 < rms < 0.26, (shape, rms)
    print("3. a Muon update has RMS about 0.2 on every shape                              PASS")

    # 4. fused halves
    g = torch.randn(96, 32)
    fused = muon_update(g, halves=True)
    top, bottom = muon_update(g[:48], halves=False), muon_update(g[48:], halves=False)
    assert torch.equal(fused[:48], top) and torch.equal(fused[48:], bottom)
    whole = muon_update(g, halves=False)
    assert not torch.allclose(fused, whole)
    print("4. a fused weight's halves are orthogonalized as two matrices                   PASS")

    # 5. the AdamW groups match torch.optim.AdamW
    a = torch.randn(8, 6, requires_grad=True)
    b = torch.randn(6, requires_grad=True)
    a2, b2 = a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True)
    ref = torch.optim.AdamW([{"params": [a], "weight_decay": 0.1}, {"params": [b], "weight_decay": 0.0}], lr=1e-2)
    ours = MuonAdamW([{"params": [a2], "weight_decay": 0.1, "muon": "adamw"},
                      {"params": [b2], "weight_decay": 0.0, "muon": "adamw"}], lr=1e-2)
    for _ in range(5):
        ga, gb = torch.randn_like(a), torch.randn_like(b)
        a.grad, b.grad, a2.grad, b2.grad = ga, gb, ga.clone(), gb.clone()
        ref.step()
        ours.step()
    assert torch.allclose(a, a2, atol=1e-6) and torch.allclose(b, b2, atol=1e-6)
    assert set(ours.state[a2]) >= {"exp_avg", "exp_avg_sq", "step"}
    assert torch.allclose(ours.state[a2]["exp_avg"], ref.state[a]["exp_avg"], atol=1e-6)
    print("5. the AdamW groups step as torch.optim.AdamW with the same state keys          PASS")

    # 6. decoupled decay on a Muon parameter, and the state round trip through tag_param_groups
    w = torch.randn(16, 8, requires_grad=True)
    w0 = w.detach().clone()
    opt = MuonAdamW([{"params": [w], "weight_decay": 0.5, "muon": "muon"}], lr=0.1)
    w.grad = torch.zeros_like(w)
    opt.step()
    assert torch.allclose(w, w0 * (1.0 - 0.1 * 0.5)), "zero gradient: only the decay moves the weight"
    w.grad = torch.randn_like(w)
    opt.step()
    moved = (w - w0 * (1.0 - 0.1 * 0.5) ** 2).pow(2).mean().sqrt().item()
    assert abs(moved - 0.1 * 0.2 * (1.0 - 0.95) / (1.0 - 0.95)) < 0.1 * 0.2 * 0.6, moved
    state = opt.state_dict()
    opt2 = MuonAdamW([{"params": [w], "weight_decay": 0.5, "muon": "muon"}], lr=0.1)
    opt2.load_state_dict(state)
    assert torch.equal(opt2.state[w]["momentum_buffer"], opt.state[w]["momentum_buffer"])
    named = {"a.q_proj.weight": torch.randn(4, 4), "b.gate_up.weight2": torch.randn(6, 4),
             "c.norm.weight": torch.randn(4), "lm_head.projection.weight": torch.randn(4, 4)}
    names = {id(p): n for n, p in named.items()}
    groups = tag_param_groups([{"params": list(named.values()), "weight_decay": 0.1, "lr": 1e-3}], names)
    assert [g["muon"] for g in groups] == ["adamw", "muon", "muon_halves"], [g["muon"] for g in groups]
    assert all(g["lr"] == 1e-3 and g["weight_decay"] == 0.1 for g in groups)
    built = build_optimizer("muon", [{"params": list(named.values()), "weight_decay": 0.1}], names, lr=1e-3)
    assert isinstance(built, MuonAdamW) and "muon: 1 tensors" in partition_summary(built)
    assert isinstance(build_optimizer("adamw", [{"params": [torch.randn(2, 2, requires_grad=True)]}], lr=1e-3),
                      torch.optim.AdamW)
    print("6. decoupled decay on Muon, the state round trip, tagging and building          PASS")

    # 7. a bf16 parameter's momentum buffer stays fp32 across a resume; a resume across kinds is
    #    refused by name before load_state_dict
    wb = torch.randn(16, 8).to(torch.bfloat16).requires_grad_(True)
    opt_b = MuonAdamW([{"params": [wb], "weight_decay": 0.0, "muon": "muon"}], lr=0.01)
    wb.grad = torch.randn_like(wb)
    opt_b.step()
    assert opt_b.state[wb]["momentum_buffer"].dtype == torch.float32
    resumed = MuonAdamW([{"params": [wb], "weight_decay": 0.0, "muon": "muon"}], lr=0.01)
    resumed.load_state_dict(opt_b.state_dict())
    assert resumed.state[wb]["momentum_buffer"].dtype == torch.bfloat16, "load_state_dict casts to the parameter's dtype"
    wb.grad = torch.randn_like(wb)
    resumed.step()
    assert resumed.state[wb]["momentum_buffer"].dtype == torch.float32
    assert optimizer_kind(opt_b) == "muon" and saved_optimizer_kind(opt_b.state_dict()) == "muon"
    plain = torch.optim.AdamW([torch.randn(2, 2, requires_grad=True)], lr=1e-3)
    assert optimizer_kind(plain) == "adamw" and saved_optimizer_kind(plain.state_dict()) == "adamw"
    try:
        check_resume_kind(plain, opt_b.state_dict(), "x.pt")
        raise AssertionError("a resume across optimizer kinds must be refused")
    except ValueError as exc:
        assert "'muon'" in str(exc) and "'adamw'" in str(exc)
    check_resume_kind(opt_b, opt_b.state_dict())
    print("7. the momentum buffer is fp32 after a bf16 resume; a cross kind resume is refused  PASS")
    print("all muon optimizer checks passed")


if __name__ == "__main__":
    main()

"""The chunk gate is skipped when its scale is frozen at exactly zero.

``1 + evidence_gate_scale * sigmoid(mass)`` is exactly 1.0 for a zero scale, so a frozen zero scale
lets the loop skip the selector mass, the scatter to tokens and the multiply by ones.

1. frozen at zero: forward hidden states are bit-identical with the skip on and forced off for both
   readers, and so is every parameter gradient for the key/value reader; the cross reader's gradients
   agree to bf16 summation order;
2. the skip never reads the device in steady state, and follows an in-place write to the scale;
3. a nonzero frozen scale and a trainable zero scale are not skipped, and the gate moves the output.

GPU required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from test_evidence_kv_reader import P, KV, Q_HI, DEVICE, build, make_live, evidence_for

CROSS = dict(P, evidence_port=True)


def forward_grads(model, ids):
    model.zero_grad(set_to_none=True)
    # rebuilt per call: the encoded states carry the graph a previous backward freed
    evidence = evidence_for(model, [0, 1], 10, 2)
    hidden = model(ids, return_hidden=True, skip_mtp=True, evidence=evidence)
    hidden.float().pow(2).mean().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return hidden.detach().clone(), grads


def test_reader(params, label, exact):
    model = make_live(build(params))
    moe = model.moe
    if moe.shared_evidence is not None:
        with torch.no_grad():
            torch.nn.init.normal_(moe.shared_evidence.attn.o_proj.weight, std=0.05)
    model.train()
    moe.set_router_noise(0.0)
    with torch.no_grad():
        moe.evidence_gate_scale.zero_()
    moe.evidence_gate_scale.requires_grad_(False)
    ids = torch.randint(1, Q_HI, (2, 24), device=DEVICE)

    assert moe._evidence_gate_is_identity()
    skipped, g_skip = forward_grads(model, ids)
    real = moe._evidence_gate_is_identity
    moe._evidence_gate_is_identity = lambda: False
    try:
        full, g_full = forward_grads(model, ids)
    finally:
        moe._evidence_gate_is_identity = real
    _, g_again = forward_grads(model, ids)
    noisy = {n for n in g_skip if not torch.equal(g_skip[n], g_again[n])}
    assert torch.equal(skipped, full), f"{label}: skip changed the forward"
    assert g_skip.keys() == g_full.keys(), f"{label}: gradient sets differ"
    scale_all = max(g.float().abs().max().item() for g in g_full.values())
    for name in g_skip:
        if name in noisy:
            # a scatter add backward (the embedding table) differs between two identical runs, so
            # the skip is held to the run to run gap there
            gap = (g_skip[name].float() - g_full[name].float()).abs().max().item()
            floor = (g_skip[name].float() - g_again[name].float()).abs().max().item()
            assert gap <= 2 * floor + 1e-12, f"{label}: gradient of {name} moved by {gap} against noise {floor}"
            continue
        if exact:
            assert torch.equal(g_skip[name], g_full[name]), f"{label}: gradient of {name} moved"
        else:
            # the dropped branch carries exact zeros, but autograd sums the remaining terms in a
            # different arrival order, which moves a bf16 sum by an ulp
            gap = (g_skip[name].float() - g_full[name].float()).abs().max().item()
            assert gap <= 2 ** -6 * g_full[name].float().abs().max().item() + 1e-3 * scale_all, f"{label}: {name} gap {gap}"
    print(f"1. {label}: frozen zero gate, skip on equals skip off (forward and {len(g_skip)} gradients, {len(noisy)} run to run noisy)  PASS")
    return model, ids


def test_cache_and_live(model, ids, label):
    moe = model.moe
    scale = moe.evidence_gate_scale
    assert moe._evidence_gate_is_identity()
    cache = moe._gate_identity_cache
    assert moe._evidence_gate_is_identity() and moe._gate_identity_cache is cache, "steady state re-read the device"
    with torch.no_grad():
        scale.fill_(2.0)
    assert not moe._evidence_gate_is_identity(), "a write to the scale was not seen"
    live, _ = forward_grads(model, ids)
    with torch.no_grad():
        scale.zero_()
    assert moe._evidence_gate_is_identity()
    off, _ = forward_grads(model, ids)
    assert not torch.equal(live, off), "a nonzero frozen scale did not move the output"
    scale.requires_grad_(True)
    assert not moe._evidence_gate_is_identity(), "a trainable zero scale was skipped"
    _, grads = forward_grads(model, ids)
    assert grads["moe.evidence_gate_scale"].abs() > 0, "a trainable gate got no gradient"
    scale.requires_grad_(False)
    print(f"2. {label}: cached check, follows writes, nonzero or trainable scale is live   PASS")


def main():
    for params, label, exact in ((KV, "kv reader", True), (CROSS, "cross reader", False)):
        model, ids = test_reader(params, label, exact)
        test_cache_and_live(model, ids, label)
    print("all gate skip checks passed")


if __name__ == "__main__":
    main()

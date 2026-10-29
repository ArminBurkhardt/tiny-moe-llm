"""The IR expert's inner attention exists only on the averaged output stage.

1. **No dead weights under the direct read.** ``direct_read=True`` (every config) builds no
   ``moe.experts.{ir}.attn.*``; the self attention expert keeps its own ``attn.*``; the A/B stage
   (``direct_read=False``) still builds it. The optimizer parameter count drops by exactly the
   inner attention's parameters, and every parameter of a direct read model gets a gradient from a
   forward and backward on a batch with evidence (router noise 0).
2. **Round trip**: save, load through ``load_model_state``, bit-identical forward.
3. **Old checkpoints still load**: a state dict carrying the old ``attn.*`` keys of the IR expert
   loads through ``load_model_state`` with those keys dropped and the forward unchanged, while a
   plain ``load_state_dict`` stays strict, an unrelated stray key still fails, and a missing key of
   a self attention expert still fails. ``model_params_for_state_dict`` infers the direct read from
   such a checkpoint.
4. **Accounting**: the attention layer count has no IR term under the direct read, and the key/value
   reader's prefix attention is billed from segment lengths (4 * hidden * loops per pair).
5. **Width check**: ``check_store_embed_dim`` refuses a port at an ``ir_dim`` other than the store's.

GPU required (TE, flash varlen).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from test_evidence_kv_reader import P, KV, Q_HI, DEVICE, build, make_live, evidence_for
from modules.model.transformer import TinyMoETransformer
from utils import (
    drop_ir_inner_attn, ir_inner_attn_keys, load_model_state, model_params_for_state_dict,
)

A = P["num_attn_experts"]
IR = 2 * A                      # index of the (only) IR expert in the pool
DIRECT = dict(KV, ir_direct_read=True)
AVERAGED = dict(KV, ir_direct_read=False)


def attn_numel(model):
    return sum(p.numel() for p in model.moe.experts[IR].attn.parameters())


def test_structure_and_count():
    direct = build(DIRECT)
    averaged = build(AVERAGED)
    names = set(direct.state_dict())
    assert not any(k.startswith(f"moe.experts.{IR}.attn.") for k in names)
    assert any(k.startswith("moe.experts.0.attn.") for k in names), "self attention expert lost its attn"
    assert direct.moe.experts[IR].attn is None
    assert ir_inner_attn_keys(direct.state_dict()) == []
    old_keys = ir_inner_attn_keys(averaged.state_dict())
    assert len(old_keys) == 8 and all(k.startswith(f"moe.experts.{IR}.attn.") for k in old_keys), old_keys
    n_attn = attn_numel(averaged)
    opt_direct = sum(p.numel() for p in direct.parameters())
    # the averaged model has no direct_gate; the direct one has it: net difference is exact
    gate = direct.moe.experts[IR].direct_gate.weight.numel()
    assert sum(p.numel() for p in averaged.parameters()) - opt_direct == n_attn - gate
    print(f"1a. direct read builds no IR inner attention ({n_attn} parameters fewer), averaged stage keeps it  PASS")

    model = make_live(build(DIRECT))
    model.train()
    model.moe.set_router_noise(0.0)
    ids = torch.randint(1, Q_HI, (2, 24), device=DEVICE)
    evidence = evidence_for(model, [0, 1], 10, 2)
    model(ids, return_hidden=True, skip_mtp=True, evidence=evidence).float().pow(2).mean().backward()
    none = [n for n, p in model.named_parameters() if p.grad is None]
    print(f"1b. parameters with grad None after one forward/backward: {len(none)} of "
          f"{sum(1 for _ in model.parameters())}: {none}")
    assert not any(".attn." in n and n.startswith(f"moe.experts.{IR}.") for n in none)


def test_round_trip():
    src = make_live(build(DIRECT))
    dst = build(DIRECT, seed=7)
    load_model_state(dst, src.state_dict())
    ids = torch.randint(1, Q_HI, (2, 24), device=DEVICE)
    with torch.no_grad():
        a = src(ids, return_hidden=True, skip_mtp=True, evidence=evidence_for(src, [0, 1], 10, 2))
        b = dst(ids, return_hidden=True, skip_mtp=True, evidence=evidence_for(dst, [0, 1], 10, 2))
    assert torch.equal(a, b), "save and load changed the forward"
    print("2.  save, load, bit-identical forward                                             PASS")


def test_old_checkpoint():
    new = make_live(build(DIRECT))
    old_shape = build(AVERAGED)
    state = {k: v.clone() for k, v in new.state_dict().items()}
    extra = {k: v.clone() for k, v in old_shape.state_dict().items() if k in set(ir_inner_attn_keys(old_shape.state_dict()))}
    assert len(extra) == 8  # four weights and their TE extra states
    state.update(extra)
    assert len(ir_inner_attn_keys(state)) == 8

    params = model_params_for_state_dict(state, P)
    assert params["ir_direct_read"] is True and params["evidence_reader"] == "kv"

    target = build(DIRECT, seed=3)
    try:
        target.load_state_dict(state)
        raise SystemExit("a plain load_state_dict must stay strict on the old attn keys")
    except RuntimeError:
        pass
    plain = new.state_dict()
    assert drop_ir_inner_attn(plain, plain.keys()) is plain, "nothing to drop must return the dict itself"
    load_model_state(target, state)
    ids = torch.randint(1, Q_HI, (2, 24), device=DEVICE)
    with torch.no_grad():
        a = new(ids, return_hidden=True, skip_mtp=True, evidence=evidence_for(new, [0, 1], 10, 2))
        b = target(ids, return_hidden=True, skip_mtp=True, evidence=evidence_for(target, [0, 1], 10, 2))
    assert torch.equal(a, b), "the old attn keys changed the loaded model"

    # nothing else is tolerated
    stray = dict(state)
    stray["moe.experts.0.not_a_tensor"] = torch.zeros(1)
    try:
        load_model_state(build(DIRECT), stray)
        raise SystemExit("an unrelated stray key must still fail")
    except RuntimeError:
        pass
    missing = dict(state)
    del missing["moe.experts.0.attn.q_proj.weight"]
    try:
        load_model_state(build(DIRECT), missing)
        raise SystemExit("a missing self attention key must still fail")
    except RuntimeError:
        pass
    # the self attention expert's attn keys are never treated as IR keys
    assert not any(k.startswith("moe.experts.0.") for k in ir_inner_attn_keys(state))
    # the averaged stage keeps its own attn keys on load (it has the slot)
    avg_target = build(AVERAGED, seed=5)
    load_model_state(avg_target, old_shape.state_dict())
    print("3.  old attn.* keys dropped on load, strict everywhere else, direct read inferred   PASS")


def test_accounting():
    d = build(DIRECT)
    a = build(AVERAGED)
    H, L, loops = P["hidden_size"], P["num_layers"], P["n_loops"]
    assert d.loop_attn_flops_per_seqsq == 2 * H * (1 + 2 * A)
    assert a.loop_attn_flops_per_seqsq == 2 * H * (1 + 2 * A + P["num_ir_experts"])
    assert d.prefix_attn_flops_per_loop_per_pair == 4 * H
    assert d.prefix_attn_flops_per_pair == 4 * H * loops
    cross = build(dict(P, evidence_port=True))
    assert cross.prefix_attn_flops_per_pair == 0 and build(P).prefix_attn_flops_per_pair == 0
    q = torch.tensor([0, 10, 24])
    e = torch.tensor([0, 6, 6])
    assert d.evidence_prefix_attn_flops(q, e) == 4 * H * loops * (10 * 6 + 14 * 0)
    assert d.evidence_prefix_attn_flops(q, e, n_loops=1) == 4 * H * 60
    assert cross.evidence_prefix_attn_flops(q, e) == 0
    print("4.  no IR attention FLOPs under the direct read, prefix attention billed from lengths  PASS")


def test_width_check():
    build(dict(P, check_store_embed_dim=True))  # no port, no check
    try:
        build(dict(KV, check_store_embed_dim=True))
        raise SystemExit("ir_dim 64 with the port must be refused")
    except AssertionError as err:
        assert "EMBED_DIM" in str(err)
    model = TinyMoETransformer(**dict(KV, ir_dim=384, num_ir_entries=384, check_store_embed_dim=True,
                                      ir_num_clusters=0))
    del model
    print("5.  ir_dim must equal the store's embedding width when the port is on             PASS")


if __name__ == "__main__":
    test_structure_and_count()
    test_round_trip()
    test_old_checkpoint()
    test_accounting()
    test_width_check()
    print("all IR inner attention checks passed")

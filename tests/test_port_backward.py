"""Backward through a freshly built evidence port: who gets gradient at step 0, and who waits.

The port is grafted onto a converged trunk with zero-init tensors so that attaching evidence is a
no-op. A zero-init tensor is only safe if it can also LEAVE zero, and that depends on which
gradients reach it. Two structural claims, both checked here on a real backward pass:

1. **The reader's ``o_proj`` gets gradient at step 0 and everything behind it does not.**
   ``o_proj`` is the last matrix of the reader, so its gradient is ``upstream x attn_out`` and is
   nonzero, while the gradient reaching ``q_proj``/``k_proj``/``v_proj`` passes through
   ``o_proj^T = 0`` and is exactly zero (not merely small: an exact zero, no tolerance). After ONE
   optimizer step ``o_proj`` is nonzero and the second backward feeds q/k/v. This is why the first
   checkpoint of a run is the earliest a "reader learned nothing" verdict can be read, and why a
   flat ``|shared_evidence.o_proj|rms`` is a kill signal while flat q/k/v at step 0 is not.

2. **The IR expert's ``g_proj`` and ``direct_gate`` hold each other at zero.** Both are zero in the
   real seed. The expert's output is ``direct_gate(up_proj(g_proj(retrieved)))``, a product of two
   zero matrices in a chain: ``d/d direct_gate`` is ``information x upstream`` with
   ``information = up_proj(g_proj(.)) = 0``, and ``d/d g_proj`` passes through
   ``direct_gate^T = 0``. Neither can ever leave zero by gradient descent alone; a trainer that
   wants the read path alive has to break the symmetry (one of the two must start nonzero, or be
   moved by a separate lever). The claim is asserted only because it holds structurally; it is
   contrasted with a control where only ``direct_gate`` is zero, which does receive gradient.

GPU required (flash varlen).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from modules.model.experts import InformationRetrievalExpert

BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
    lm_head_factor=4, evidence_port=True,
)


def build(seed=0):
    torch.manual_seed(seed)
    model = TinyMoETransformer(**P).to("cuda").to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    return model


def ir_expert_of(model):
    return next(e for e in model.moe.experts if isinstance(e, InformationRetrievalExpert))


def attach(model, B, ev_len=12):
    ev_ids = torch.randint(1, P["vocab_size"], (B, ev_len), device="cuda")
    ev_chunk = torch.arange(B, device="cuda").unsqueeze(1).expand(B, ev_len).contiguous()
    ev_segment = torch.arange(B, device="cuda").unsqueeze(1).expand(B, ev_len).contiguous()
    keys = torch.randn(B, P["ir_dim"], device="cuda", dtype=BF16)
    chunk_segments = torch.arange(B, device="cuda")
    return model.build_evidence(ev_ids, ev_chunk, ev_segment, B, chunk_keys=keys,
                                chunk_segments=chunk_segments)


def loss_of(model, input_ids, evidence):
    logits = model(input_ids, skip_mtp=True, evidence=evidence)
    return logits.float().pow(2).mean() + logits.float().logsumexp(-1).mean()


def gnorm(p):
    return 0.0 if p.grad is None else p.grad.float().norm().item()


def main():
    B, S = 2, 24
    model = build()
    reader = model.moe.shared_evidence
    ir_expert = ir_expert_of(model)
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device="cuda")
    # detached evidence: the trunk's own gradient into the encoder is not what is under test
    with torch.no_grad():
        evidence = attach(model, B)

    # ---- claim 1: o_proj first, q/k/v after one step
    assert float(reader.attn.o_proj.weight.detach().abs().max()) == 0.0, "o_proj is not zero-init"
    model.zero_grad(set_to_none=True)
    loss_of(model, input_ids, evidence).backward()
    g_o = gnorm(reader.attn.o_proj.weight)
    g_qkv0 = {n: gnorm(getattr(reader.attn, n).weight) for n in ("q_proj", "k_proj", "v_proj")}
    print(f"step 0: |g o_proj| = {g_o:.3e}; " + ", ".join(f"|g {n}| = {v:.3e}" for n, v in g_qkv0.items()))
    assert g_o > 0.0, "o_proj received no gradient at step 0, so the reader can never wake up"
    for n, v in g_qkv0.items():
        assert v == 0.0, f"{n} got gradient {v} through a zero o_proj"
    print("1a. o_proj has gradient, q/k/v are exactly zero behind it at step 0      PASS")

    port_params = list(reader.parameters())
    opt = torch.optim.AdamW(port_params, lr=1e-2, weight_decay=0.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    assert float(reader.attn.o_proj.weight.abs().max()) > 0.0, "the optimizer step did not move o_proj"
    loss_of(model, input_ids, evidence).backward()
    g_qkv1 = {n: gnorm(getattr(reader.attn, n).weight) for n in ("q_proj", "k_proj", "v_proj")}
    print("step 1: " + ", ".join(f"|g {n}| = {v:.3e}" for n, v in g_qkv1.items()))
    for n, v in g_qkv1.items():
        assert v > 0.0, f"{n} still has no gradient after o_proj left zero"
    print("1b. q/k/v receive gradient once o_proj has moved                          PASS")

    # ---- claim 2: g_proj x direct_gate, both zero
    model = build()
    ir_expert = ir_expert_of(model)
    assert ir_expert.direct_gate is not None, "this test needs the direct read stage"
    with torch.no_grad():
        evidence = attach(model, B)
        # the real seed: the reshape migration zeroed g_proj, direct_gate is zero-init
        ir_expert.ir_module.g_proj.weight.zero_()
    assert float(ir_expert.direct_gate.weight.abs().max()) == 0.0
    # reader live so the run is not trivially all zeros, and the loss moves with the IR expert too
    with torch.no_grad():
        torch.nn.init.normal_(model.moe.shared_evidence.attn.o_proj.weight, std=0.02)
    model.zero_grad(set_to_none=True)
    loss_of(model, input_ids, evidence).backward()
    both = {
        "g_proj": gnorm(ir_expert.ir_module.g_proj.weight),
        "direct_gate": gnorm(ir_expert.direct_gate.weight),
        "up_proj": gnorm(ir_expert.up_proj.weight),
        "y_values": gnorm(ir_expert.ir_module.y_values),
    }
    print("both zero: " + ", ".join(f"|g {n}| = {v:.3e}" for n, v in both.items()))
    assert both["g_proj"] == 0.0 and both["direct_gate"] == 0.0, (
        f"g_proj and direct_gate do not hold each other at zero: {both}"
    )
    print("2a. g_proj = direct_gate = 0 -> neither gets gradient (mutual lock)      PASS")

    # control: only direct_gate zero, g_proj at its default init. direct_gate must now get gradient
    # (the lock is the PAIR of zeros, not direct_gate alone), while g_proj still does not because
    # its gradient passes through direct_gate^T
    model = build()
    ir_expert = ir_expert_of(model)
    with torch.no_grad():
        evidence = attach(model, B)
        torch.nn.init.normal_(model.moe.shared_evidence.attn.o_proj.weight, std=0.02)
    model.zero_grad(set_to_none=True)
    loss_of(model, input_ids, evidence).backward()
    ctrl = {
        "g_proj": gnorm(ir_expert.ir_module.g_proj.weight),
        "direct_gate": gnorm(ir_expert.direct_gate.weight),
    }
    print("control (g_proj nonzero): " + ", ".join(f"|g {n}| = {v:.3e}" for n, v in ctrl.items()))
    assert ctrl["direct_gate"] > 0.0, "direct_gate got no gradient even with a live g_proj"
    assert ctrl["g_proj"] == 0.0, "g_proj got gradient through a zero direct_gate"
    print("2b. control: a live g_proj alone unlocks direct_gate                      PASS")


if __name__ == "__main__":
    main()

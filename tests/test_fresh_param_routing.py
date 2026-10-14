"""Which tensors train at the fresh rate, and what the fp32 master machinery does with them.

The evidence profile grafts a zero-init port onto a converged checkpoint. Zero-init tensors need
the from-scratch learning rate or they sit at their init for the whole run, while the converged
table, trunk and heads must stay on the trunk rate or a from-scratch rate wrecks them. The split is
one name predicate (``moe.is_fresh_loop_param``) and a silent mistake in either direction costs a
whole run, so this file pins it from three sides.

1. **The predicate selects exactly the port.** On a model with the reader, the groundedness head,
   the loop injection and an IR expert, the selected names are exactly: the reader subtree
   (``shared_evidence.*``), ``key_adapter``, ``value_adapter``, ``log_memory_scale``, the IR
   module's and the reader's ``loop_query_bias`` / ``evidence_query_bias``, ``evidence_loop_scale``,
   ``direct_gate``, ``evidence_gate_scale``, ``groundedness_head.*`` and ``inject``. The expected set
   is written out independently of the predicate (as regexes), and the converged families
   (``z_keys``, ``y_values``, ``g_proj``, the IR ``down_proj``/``up_proj``, ``centroids``,
   ``loop_scale``, the router, ``lm_head``, the embeddings, every decoder layer) are asserted NOT
   selected by name, so a widened pattern fails loudly. The full selected list is printed.

2. **Optimizer grouping is a partition.** ``sft.build_sft_param_groups`` with the evidence
   predicate puts every trainable parameter in exactly one group with exactly one fp32 master,
   fresh ones in groups at ``fresh_lr`` and the rest at the run rate, and a parameter with
   ``requires_grad=False`` (here ``evidence_gate_scale``, the frozen gate) in no group and with no
   master.

3. **The by-hand gradient clear keeps accumulation and ends the window.** A parameter stepped
   through an fp32 master sits in no optimizer group, so ``optimizer.zero_grad()`` never reaches its
   bf16 ``.grad``; ``pretrain.train_step`` clears it by hand, gated on ``sync_gradients``. Driving
   the real ``train_step`` twice under ``gradient_accumulation_steps=2``: after the non-sync micro
   step the bf16 params still hold their gradient (clearing every micro step would throw the
   accumulation away), and after the sync step every bf16 param in ``master_pairs`` has
   ``.grad is None`` (without the clear the gradients sum over the whole run and the clip norm
   throttles every parameter).

GPU required (flash varlen, TE).
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import optim
from accelerate import Accelerator

from modules.model.transformer import TinyMoETransformer
from modules.model.moe import is_fresh_loop_param
from modules.model.attention import cu_seqlens_from_doc_ids
from scripts.sft import build_sft_param_groups
from scripts.pretrain import train_step

BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
    lm_head_factor=4, evidence_port=True, groundedness_head=True, loop_inject=True,
)

# the port, stated independently of the predicate under test
EXPECTED_FRESH = [
    r"^moe\.shared_evidence\.",
    r"^moe\.experts\.\d+\.ir_module\.key_adapter\.",
    r"^moe\.experts\.\d+\.ir_module\.value_adapter\.",
    r"^moe\.experts\.\d+\.ir_module\.log_memory_scale$",
    r"^moe\.experts\.\d+\.ir_module\.loop_query_bias\.",
    r"^moe\.evidence_query_bias\.",
    r"^moe\.evidence_loop_scale$",
    r"^moe\.experts\.\d+\.direct_gate\.",
    r"^moe\.evidence_gate_scale$",
    r"^groundedness_head\.",
    r"^moe\.inject\.",
]
# converged families that must stay on the trunk rate
NEVER_FRESH = [
    r"z_keys", r"y_values", r"g_proj", r"\.down_proj\.", r"\.up_proj\.", r"centroids",
    r"^moe\.loop_scale$", r"^moe\.router\.", r"^moe\.loop_router_bias", r"^lm_head\.",
    r"embed", r"^gemma_decoder\.layers\.", r"^mtp_head\.", r"^moe\.shared_mlp\.",
    r"^moe\.shared_attn\.", r"^moe\.parallel_experts\.", r"^moe\.post_norm",
]


def build():
    torch.manual_seed(0)
    model = TinyMoETransformer(**P).to("cuda").to(BF16).train()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    return model


def test_predicate(model):
    names = [n for n, _ in model.named_parameters()]
    selected = [n for n in names if is_fresh_loop_param(n)]
    print(f"is_fresh_loop_param selects {len(selected)} of {len(names)} tensors:")
    for n in selected:
        print(f"  {n}")
    expected = [n for n in names if any(re.search(p, n) for p in EXPECTED_FRESH)]
    assert set(selected) == set(expected), (
        f"selected but not expected: {sorted(set(selected) - set(expected))}; "
        f"expected but not selected: {sorted(set(expected) - set(selected))}"
    )
    # every pattern matched something, so a renamed tensor cannot make the expected list vacuous
    for p in EXPECTED_FRESH:
        assert any(re.search(p, n) for n in names), f"expected family {p} matches no parameter"
    leaked = [n for n in selected if any(re.search(p, n) for p in NEVER_FRESH)]
    assert not leaked, f"converged tensors selected as fresh: {leaked}"
    print("1. the predicate selects exactly the port and none of the converged families   PASS")
    return set(selected)


def test_groups(model, fresh_names):
    frozen = model.moe.evidence_gate_scale
    frozen.requires_grad_(False)
    frozen_name = "moe.evidence_gate_scale"
    base_lr, fresh_lr = 1e-5, 3e-4
    groups, master_pairs = build_sft_param_groups(
        model, 0.1, fresh_lr=fresh_lr, is_fresh_param=is_fresh_loop_param
    )
    opt = optim.AdamW(groups, lr=base_lr)

    trainable = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    pair_ids = [id(p) for p, _ in master_pairs]
    assert len(pair_ids) == len(set(pair_ids)), "a parameter has two masters"
    assert set(pair_ids) == set(trainable), "master pairs do not cover exactly the trainable params"
    assert id(frozen) not in set(pair_ids), "a frozen parameter got a master"

    master_group = {}
    for gi, g in enumerate(opt.param_groups):
        for m in g["params"]:
            assert id(m) not in master_group, "a master sits in two groups"
            master_group[id(m)] = gi
    assert len(master_group) == len(master_pairs), "some masters are in no group"
    for p, m in master_pairs:
        assert m.dtype == torch.float32 and m is not p
        g = opt.param_groups[master_group[id(m)]]
        name = trainable[id(p)]
        want = fresh_lr if name in fresh_names else base_lr
        assert g["lr"] == want, f"{name} trains at {g['lr']}, expected {want}"
        assert (g["weight_decay"] > 0) == (p.ndim >= 2), f"{name}: wrong decay split"
    assert frozen_name not in {trainable[i] for i in pair_ids}
    n_fresh = sum(1 for p, _ in master_pairs if trainable[id(p)] in fresh_names)
    print(f"2. {len(master_pairs)} masters, {n_fresh} at fresh lr {fresh_lr:.0e}, "
          f"{frozen_name} (requires_grad False) in no group   PASS")
    return opt, master_pairs


def test_grad_clear(model, opt, master_pairs):
    accelerator = Accelerator(gradient_accumulation_steps=2)
    model, opt = accelerator.prepare(model, opt)
    unwrapped = accelerator.unwrap_model(model)

    B, S, S_ev = 2, 24, 12
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device="cuda")
    cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(
        torch.arange(B, device="cuda").unsqueeze(1).expand(B, S).contiguous()
    )
    pad_mask = torch.zeros(B, S, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        # live reader and selector path so the gradient reaches the port as well as the trunk
        torch.nn.init.normal_(unwrapped.moe.shared_evidence.attn.o_proj.weight, std=0.02)
        ev_ids = torch.randint(1, P["vocab_size"], (B, S_ev), device="cuda")
        ev_chunk = torch.arange(B, device="cuda").unsqueeze(1).expand(B, S_ev).contiguous()
        evidence = unwrapped.build_evidence(
            ev_ids, ev_chunk, ev_chunk.clone(), B,
            chunk_keys=torch.randn(B, P["ir_dim"], device="cuda", dtype=BF16),
            chunk_segments=torch.arange(B, device="cuda"),
        )

    def step():
        train_step(
            model, input_ids, cu_seqlens, max_seqlen, input_ids.clone(), pad_mask,
            accelerator=accelerator, optimizer=opt, no_decay_master_pairs=master_pairs,
            evidence=evidence,
        )

    bf16_params = [p for p, _ in master_pairs]
    step()
    assert not accelerator.sync_gradients, "the first micro step should not be a sync step"
    held = [p for p in bf16_params if p.grad is not None and float(p.grad.abs().max()) > 0.0]
    assert len(held) > len(bf16_params) // 2, (
        f"only {len(held)} of {len(bf16_params)} bf16 params kept a nonzero grad after the "
        f"non sync micro step: accumulation is being thrown away"
    )

    step()
    assert accelerator.sync_gradients, "the second micro step should be the sync step"
    stale = [p for p in bf16_params if p.grad is not None]
    assert not stale, f"{len(stale)} bf16 params still hold a grad after the sync step"
    print(f"3. accumulation kept on the non sync step ({len(held)}/{len(bf16_params)} params held a "
          f"grad), all {len(bf16_params)} cleared after the sync step   PASS")


def main():
    model = build()
    fresh = test_predicate(model)
    opt, master_pairs = test_groups(model, fresh)
    test_grad_clear(model, opt, master_pairs)


if __name__ == "__main__":
    main()

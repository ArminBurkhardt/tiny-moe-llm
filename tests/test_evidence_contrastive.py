"""The retriever's contrastive loss on hand built tensors (1 to 6: no model, no CUDA, no tokenizer).

1. one positive per row: the query direction equals ``F.cross_entropy`` on the scaled cosines;
2. a row with no positive is excluded and does not change the loss; with no positive anywhere the
   loss is exactly zero and still carries a graph to both inputs;
3. several positives in a row: the target is their joint mass (logsumexp over the positives);
4. the key direction: a chunk that is gold for one query equals ``F.cross_entropy`` down its
   column; the symmetric loss is the mean of the two directions;
5. the temperature divides the cosine;
6. the gradient points the query toward its gold key and away from the other documents' keys, and
   a chunk from another document is a negative (the loss is higher when it is close to the query).

On a toy model (needs CUDA and transformer_engine, skipped otherwise), the trainer's entry
``LoopMixtureOfExperts.evidence_contrastive_term``:

7. with ``loops=None`` and ``trunk_grad=1.0`` it is bit-identical, value and every parameter's
   gradient, to the average over every loop's stashed query the term computed before those keys;
   the stashed query equals the old inline expression bit for bit, the rebuilt query lands on the
   same rows, and the input stash stays off (and a scaled term refuses) until it is turned on;
8. ``loops=[0]`` equals the loop-0 term alone, ``loops=[0, 2]`` the mean of those two;
9. ``trunk_grad=0.0`` leaves every parameter outside the query head (the IR expert's ``norm``,
   ``down_proj``, ``loop_query_bias`` and ``key_adapter``) without gradient from the term while
   ``down_proj`` and ``key_adapter`` get one, at the same value; at 1.0 the trunk does get one;
10. ``trunk_grad=0.5`` halves the trunk's gradient and leaves the head's; under ``no_grad`` the
    term reads the stashed query at any ``trunk_grad``.

The function is pure torch, so it is imported through ``modules.model.information_retrieval``
with transformer_engine stubbed when it is not installed (the module imports it at top level).
"""
import os
import sys
import types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

HAVE_TE = True
try:
    import transformer_engine.pytorch  # noqa: F401
except Exception:  # pragma: no cover - the CPU box has no TE
    HAVE_TE = False
    te = types.ModuleType("transformer_engine")
    te_pt = types.ModuleType("transformer_engine.pytorch")

    class _Linear(torch.nn.Linear):
        pass

    te_pt.Linear = _Linear
    te_pt.checkpoint = None
    te.pytorch = te_pt
    sys.modules["transformer_engine"] = te
    sys.modules["transformer_engine.pytorch"] = te_pt

from modules.model.information_retrieval import evidence_contrastive_loss  # noqa: E402


def close(a, b, tol=1e-5):
    assert abs(float(a) - float(b)) <= tol, f"{float(a)} != {float(b)}"


def unit(x):
    return F.normalize(x, p=2, dim=-1)


def main():
    torch.manual_seed(0)
    tau = 0.07

    # 1. one positive per row equals cross entropy over the scaled cosines
    q = unit(torch.randn(3, 8))
    k = unit(torch.randn(5, 8))
    positive = torch.zeros(3, 5, dtype=torch.bool)
    positive[0, 1] = positive[1, 3] = positive[2, 4] = True
    logits = q @ k.t() / tau
    expect = F.cross_entropy(logits, torch.tensor([1, 3, 4]))
    got = evidence_contrastive_loss(q, k, positive, temperature=tau, symmetric=False)
    close(got, expect)
    print("1. one positive per row equals cross entropy                                PASS")

    # 2. a row without a positive is excluded; none anywhere gives a connected zero
    positive2 = positive.clone()
    positive2[2] = False
    got2 = evidence_contrastive_loss(q, k, positive2, temperature=tau, symmetric=False)
    close(got2, F.cross_entropy(logits[:2], torch.tensor([1, 3])))
    q_live = q.clone().requires_grad_(True)
    k_live = k.clone().requires_grad_(True)
    zero = evidence_contrastive_loss(q_live, k_live, torch.zeros(3, 5, dtype=torch.bool), temperature=tau)
    close(zero, 0.0)
    zero.backward()
    assert q_live.grad is not None and k_live.grad is not None
    close(q_live.grad.abs().sum(), 0.0)
    print("2. rows without a positive are excluded; no positives gives a connected zero PASS")

    # 3. several positives: the target is their joint share
    positive3 = positive.clone()
    positive3[0, 2] = True
    row0 = torch.logsumexp(logits[0], 0) - torch.logsumexp(logits[0, [1, 2]], 0)
    rows = torch.stack([row0, -F.log_softmax(logits[1], 0)[3], -F.log_softmax(logits[2], 0)[4]])
    got3 = evidence_contrastive_loss(q, k, positive3, temperature=tau, symmetric=False)
    close(got3, rows.mean())
    print("3. several positives in a row target their joint mass                       PASS")

    # 4. the key direction, and the symmetric mean
    col = torch.stack([
        -F.log_softmax(logits[:, 1], 0)[0], -F.log_softmax(logits[:, 3], 0)[1], -F.log_softmax(logits[:, 4], 0)[2],
    ]).mean()
    got4 = evidence_contrastive_loss(q, k, positive, temperature=tau, symmetric=True)
    close(got4, 0.5 * (expect + col))
    print("4. the key direction equals cross entropy down the column; symmetric averages PASS")

    # 5. the temperature divides the cosine
    got5 = evidence_contrastive_loss(q, k, positive, temperature=1.0, symmetric=False)
    close(got5, F.cross_entropy(q @ k.t(), torch.tensor([1, 3, 4])))
    assert abs(float(got5) - float(got)) > 1e-3
    print("5. the temperature divides the cosine                                       PASS")

    # 6. the gradient pulls the query to its gold key; another document's chunk is a negative
    q6 = unit(torch.randn(1, 8)).requires_grad_(True)
    k6 = unit(torch.randn(4, 8))
    pos6 = torch.zeros(1, 4, dtype=torch.bool)
    pos6[0, 0] = True
    loss6 = evidence_contrastive_loss(q6, k6, pos6, temperature=tau, symmetric=False)
    loss6.backward()
    step = q6.detach() - 0.01 * q6.grad
    before = (q6.detach() @ k6[0]).item()
    after = (unit(step) @ k6[0]).item()
    assert after > before, (before, after)
    near = k6.clone()
    near[2] = unit(q6.detach()[0] + 0.05 * torch.randn(8))
    loss_near = evidence_contrastive_loss(q6.detach(), near, pos6, temperature=tau, symmetric=False)
    assert float(loss_near) > float(loss6), (float(loss_near), float(loss6))
    print("6. the gradient pulls the query to its gold key; a close foreign chunk costs   PASS")

    if HAVE_TE and torch.cuda.is_available():
        model_checks(tau)
    else:
        print("7-10. the model-level loop and trunk gradient checks need CUDA and TE         SKIP")
    print("all evidence contrastive checks passed")


MODEL_P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2, lm_head_factor=4,
    evidence_port=True,
)


def _grads(loss, named):
    # one backward per forward: transformer_engine's Linear frees its saved tensors on the first
    # backward, so a retained graph cannot be walked twice
    got = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    return {name: g for (name, _), g in zip(named, got)}


def _nonzero(g):
    return g is not None and g.abs().max().item() > 0


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def model_checks(tau):
    from modules.model.experts import InformationRetrievalExpert
    from modules.model.transformer import TinyMoETransformer

    device, bf16 = "cuda", torch.bfloat16
    torch.manual_seed(0)
    model = TinyMoETransformer(**MODEL_P).to(device).to(bf16).eval()
    model.set_checkpointing(False, False)
    moe = model.moe
    # a moved loop_query_bias makes the per-loop queries differ, so the loop filter is tested on
    # distinct terms rather than on three copies of one
    ir = moe.ir_modules[0]
    with torch.no_grad():
        torch.nn.init.normal_(ir.loop_query_bias.weight, std=0.5)

    B, S, S_ev = 2, 24, 12
    input_ids = torch.randint(1, MODEL_P["vocab_size"], (B, S), device=device)
    ev_ids = torch.randint(1, MODEL_P["vocab_size"], (B, S_ev), device=device)
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6, [2] * 6 + [3] * 6], device=device)
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device=device)
    chunk_segments = torch.tensor([0, 0, 1, 1], device=device)
    chunk_gold = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device)
    keys = torch.randn(4, MODEL_P["ir_dim"], device=device, dtype=bf16)
    evidence = model.build_evidence(
        ev_ids, ev_chunk, ev_segment, 2, chunk_keys=keys, chunk_segments=chunk_segments,
        chunk_gold=chunk_gold,
    )
    positions = torch.zeros(B, S, dtype=torch.bool, device=device)
    positions[:, S - 4:] = True
    positions = positions.reshape(-1)
    index = positions.nonzero(as_tuple=False).squeeze(-1)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def forward():
        model(input_ids, skip_mtp=True, evidence=evidence)

    def term(**kwargs):
        return moe.evidence_contrastive_term(chunk_gold, positions, temperature=tau, **kwargs)

    expert = next(e for e in moe.experts if isinstance(e, InformationRetrievalExpert))
    prefix = next(n for n, m in model.named_modules() if m is expert) + "."

    # the input stash is off until a trainer asks for it, and a scaled term without it says so
    forward()
    assert not ir.query_input_by_loop, "the expert input was stashed with the stash off"
    try:
        term(loops=[0], trunk_grad=0.0)
    except AssertionError:
        pass
    else:
        raise AssertionError("a scaled term without the stash did not refuse")
    moe.set_contrastive_query_stash(True)

    forward()
    assert sorted(ir.memory_query_by_loop) == [0, 1, 2], sorted(ir.memory_query_by_loop)
    assert sorted(ir.query_input_by_loop) == [0, 1, 2], sorted(ir.query_input_by_loop)
    # the refactored forward query is the old inline expression, bit for bit, and the rebuild
    # lands on the same rows
    with torch.no_grad():
        for loop in (0, 1, 2):
            source = ir.query_input_by_loop[loop]
            down = expert.down_proj(expert.norm(source)).reshape(-1, ir.latent_dim)
            inline = F.normalize(down + ir._loop_query_bias(loop, down.dtype), p=2, dim=-1)
            assert torch.equal(ir.memory_query_by_loop[loop], inline), f"loop {loop} query moved"
        flat = ir.query_input_by_loop[0].reshape(-1, ir.query_input_by_loop[0].shape[-1])
        rebuilt = expert.contrastive_query(flat.index_select(0, index), 0, 0.5)
        stashed = ir.memory_query_by_loop[0].index_select(0, index)
        assert torch.allclose(rebuilt.float(), stashed.float(), atol=1e-2), (rebuilt - stashed).abs().max()
    positive = moe.last_memory_visible.index_select(0, index) & chunk_gold.bool().unsqueeze(0)
    assert positive.any(dim=-1).all(), "every query row needs a visible gold chunk"

    def loop_term(loop):
        return evidence_contrastive_loss(
            ir.memory_query_by_loop[loop].index_select(0, index), ir.adapted_keys_by_loop[loop],
            positive, temperature=tau,
        )

    # 8 first, on values only, so the one backward of 7 can consume this forward's graph
    t0 = term(loops=[0])
    assert torch.equal(t0, loop_term(0)), (t0.item(), loop_term(0).item())
    t02 = term(loops=[0, 2])
    assert torch.equal(t02, torch.stack([loop_term(0), loop_term(2)]).mean())

    # 7. defaults are the term before the two keys, bit for bit. Value: equal. Gradient: both
    # graphs end on the same stashed tensors, so the gradient of their difference is exactly zero
    # everywhere iff every parameter's gradient agrees bit for bit
    old = torch.stack([loop_term(loop) for loop in ir.memory_query_by_loop]).mean()
    new = term()
    explicit = term(loops=None, trunk_grad=1.0)
    assert torch.equal(old, new) and torch.equal(new, explicit), (old.item(), new.item(), explicit.item())
    assert abs(t0.item() - new.item()) > 1e-4, "the per-loop terms coincide, the filter is untested"
    diff = _grads(old - explicit, named)
    moved = [n for n, g in diff.items() if _nonzero(g)]
    assert not moved, f"the default term's gradient moved on {moved}"
    print(f"7. loops=None, trunk_grad=1.0 is bit-identical, value and gradient ({new.item():.4f}) PASS")
    print(f"8. loops=[0] is the loop-0 term ({t0.item():.4f} against all loops {new.item():.4f})   PASS")

    # 9. trunk_grad=0.0: only the query head trains
    head = {prefix + s for s in (
        "norm.weight", "down_proj.weight", "ir_module.loop_query_bias.weight",
        "ir_module.loop_query_bias.bias", "ir_module.key_adapter.weight",
    )}
    forward()
    t_full = term(loops=[0])
    full = _grads(t_full, named)
    forward()
    t_head = term(loops=[0], trunk_grad=0.0)
    assert abs(t_head.item() - t_full.item()) < 1e-3, (t_head.item(), t_full.item())
    g_head = _grads(t_head, named)
    leaked = [n for n, g in g_head.items() if n not in head and _nonzero(g)]
    assert not leaked, f"trunk_grad=0.0 still reaches {leaked}"
    assert _nonzero(g_head[prefix + "down_proj.weight"]), "down_proj got no gradient from the term"
    assert _nonzero(g_head[prefix + "ir_module.key_adapter.weight"]), "key_adapter got no gradient"
    trunk = [n for n, g in full.items() if n not in head and _nonzero(g)]
    assert any("embed_tokens" in n for n in trunk), f"at trunk_grad=1.0 the embeddings got nothing: {trunk}"
    print(f"9. trunk_grad=0.0 trains only the query head ({len(trunk)} trunk tensors at 1.0, 0 at 0.0) PASS")

    # 10. trunk_grad=0.5 halves the trunk's gradient and keeps the head's; no_grad reads the stash
    forward()
    g_half = _grads(term(loops=[0], trunk_grad=0.5), named)
    worst_trunk = max(_rel(g_half[n], 0.5 * full[n]) for n in trunk)
    worst_head = max(_rel(g_half[n], full[n]) for n in head if _nonzero(full.get(n)))
    assert worst_trunk < 1e-2 and worst_head < 1e-2, (worst_trunk, worst_head)
    with torch.no_grad():
        forward()
        assert not ir.query_input_by_loop, "the expert input was stashed without autograd"
        assert torch.equal(term(loops=[0]), term(loops=[0], trunk_grad=0.0))
    print(f"10. trunk_grad=0.5 halves the trunk gradient (rel err {worst_trunk:.1e}, head {worst_head:.1e}) PASS")


if __name__ == "__main__":
    main()

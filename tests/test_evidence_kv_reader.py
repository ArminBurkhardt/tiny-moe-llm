"""The key/value evidence reader: evidence as leading keys of ``shared_attn`` in every loop.

1. **The mode lives in the state dict.** ``evidence_reader="kv"`` builds no ``shared_evidence`` and
   no ``evidence_query_bias``, registers ``moe.evidence_reader_kv``, and
   ``model_params_for_state_dict`` reads it back. A cross reader seed loads into it once its reader
   tensors are dropped. The no-rotary combination is refused.
2. **Prefix positions.** Each segment's evidence sits right before the segment
   (``prefix_position_ids``, negative values allowed), and ``rotary_emb.at`` matches the cache
   gather at non negative positions (to a bf16 rounding).
3. **The prefix attention is the text-before-the-document attention.** ``prefix_varlen_attention``
   equals plain causal attention over ``[evidence; own tokens]`` per segment, read at the own tokens,
   on the flash path and on the SDPA fallback.
4. **An empty prefix changes nothing**: a segment without evidence gets exactly the plain causal
   output (``torch.equal`` against ``varlen_attention``).
5. **No evidence attached is bit-identical** to the model without the port, same weights.
6. **The read is live and content dependent**: attaching evidence moves loop 1, other evidence moves
   it differently.
7. **Visibility**: in a packed row, changing document 1's evidence leaves document 0's hidden states
   unchanged and moves document 1's.
8. **Zero-init gate neutrality**: gate scale 0 equals the ungated forward bit for bit; a nonzero
   scale changes it.
9. **Read ablation**: ``reader_sites_kept`` None or at depth equals the full forward, 0 equals
   ``evidence=None`` with the reader output cleared, 1 keeps loop 1 only.
10. **Gradient reaches the evidence encoder** through ``shared_attn`` (token rows that occur only in
    the evidence get a gradient on the embedding table).
11. **Guards**: a KV cache with evidence raises.

GPU required (flash varlen, TE).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import modules.model.attention as attention
from modules.model.attention import (
    cu_seqlens_from_doc_ids, prefix_varlen_attention, varlen_attention,
)
from modules.model.evidence import prefix_position_ids
from modules.model.kv_cache import KVCache
from modules.model.transformer import TinyMoETransformer
from utils import load_model_state, model_params_for_state_dict

DEVICE = "cuda"
BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2, lm_head_factor=4,
)
KV = dict(P, evidence_port=True, evidence_reader="kv")
# evidence tokens come from the top of the vocabulary, queries from the bottom, so an embedding row
# with a gradient in test 10 can only have got it through the evidence
EV_LO, Q_HI = 400, 300


def build(params, seed=0):
    torch.manual_seed(seed)
    model = TinyMoETransformer(**params).to(DEVICE).to(BF16).eval()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    return model


def make_live(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("g_proj.weight", "direct_gate.weight")):
                torch.nn.init.normal_(param, std=0.05)
            elif name.endswith("log_memory_scale"):
                param.fill_(3.0)
    return model


def evidence_for(model, segments, length, num_segments, seed=1):
    """One evidence row per unpacked query row, ``length`` tokens in two chunks, no padding;
    ``segments[r]`` is the query segment row r's evidence serves."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, half = len(segments), length // 2
    ids = torch.randint(EV_LO, P["vocab_size"], (B, length), generator=g)
    chunk = torch.tensor([[2 * r] * half + [2 * r + 1] * (length - half) for r in range(B)])
    seg = torch.tensor([[s] * length for s in segments])
    keys = [torch.randn(2, P["ir_dim"], generator=g) for _ in segments]
    chunk_segments = [s for s in segments for _ in range(2)]
    return model.build_evidence(
        ids.to(DEVICE), chunk.to(DEVICE), seg.to(DEVICE), num_segments,
        chunk_keys=torch.cat(keys).to(DEVICE, BF16), chunk_segments=torch.tensor(chunk_segments, device=DEVICE),
    )


def test_mode_and_load():
    cross = build(dict(P, evidence_port=True))
    kv = build(KV)
    names = set(kv.state_dict())
    assert "moe.evidence_reader_kv" in names
    assert not any("shared_evidence" in k or "evidence_query_bias" in k for k in names)
    params = model_params_for_state_dict(kv.state_dict(), P)
    assert params["evidence_port"] and params["evidence_reader"] == "kv"
    params = model_params_for_state_dict(cross.state_dict(), P)
    assert params["evidence_port"] and params["evidence_reader"] == "cross"
    # a cross seed loads once its reader tensors are dropped; the marker is filled by the model
    state = {k: v for k, v in cross.state_dict().items()
             if not k.startswith(("moe.shared_evidence.", "moe.evidence_query_bias."))}
    load_model_state(kv, state)
    assert torch.equal(kv.moe.shared_attn.attn.k_proj.weight, cross.moe.shared_attn.attn.k_proj.weight)
    refused = False
    try:
        TinyMoETransformer(**dict(KV, evidence_reader_rotary=False))
    except ValueError:
        refused = True
    assert refused, "kv reader without rotary was built"
    print("1. kv mode in the state dict, inferred back, cross seed loads, no-rotary refused  PASS")


def test_prefix_positions(model):
    # one row, segments of 5 and 7 query tokens; evidence 3 tokens for segment 0, none for 1
    cu_q = torch.tensor([0, 5, 12], device=DEVICE, dtype=torch.int32)
    cu_p = torch.tensor([0, 3, 3], device=DEVICE, dtype=torch.int32)
    got = prefix_position_ids(cu_p, cu_q, 1, 12, 3)
    assert got.tolist() == [[-3, -2, -1]], got.tolist()
    # second segment owns the evidence: it starts at row position 5, so 2, 3, 4
    cu_p = torch.tensor([0, 0, 3], device=DEVICE, dtype=torch.int32)
    assert prefix_position_ids(cu_p, cu_q, 1, 12, 3).tolist() == [[2, 3, 4]]
    rot = model.moe.rotary_emb
    pos = torch.tensor([[0, 5, 17, 3]], device=DEVICE)
    # the model's cache was cast to bf16 by model.to(); .at computes in fp32 and casts the same way,
    # so the two agree to a bf16 rounding (host and device cos can straddle a rounding boundary)
    for a, b in zip(rot.at(pos, BF16), rot.gather(pos, BF16)):
        assert (a.float() - b.float()).abs().max().item() <= 4e-3, "rotary .at differs from the cache gather"
    print("2. prefix positions sit before the segment, .at matches the cache gather        PASS")


def reference_prefix_attention(q, k, v, kp, vp, cu_q, cu_p, scale):
    """Per segment: causal attention over [prefix; own], read at the own tokens. fp32."""
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    qf = q.transpose(1, 2).reshape(B * S, Hq, D).float()
    kf = k.transpose(1, 2).reshape(B * S, Hkv, D).float()
    vf = v.transpose(1, 2).reshape(B * S, Hkv, D).float()
    kpf = kp.transpose(1, 2).reshape(-1, Hkv, D).float()
    vpf = vp.transpose(1, 2).reshape(-1, Hkv, D).float()
    out = torch.zeros(B * S, Hq, D, device=q.device)
    for s in range(cu_q.numel() - 1):
        a, b = int(cu_q[s]), int(cu_q[s + 1])
        pa, pb = int(cu_p[s]), int(cu_p[s + 1])
        keys = torch.cat([kpf[pa:pb], kf[a:b]]).repeat_interleave(Hq // Hkv, dim=1)
        vals = torch.cat([vpf[pa:pb], vf[a:b]]).repeat_interleave(Hq // Hkv, dim=1)
        E, L = pb - pa, b - a
        scores = torch.einsum("qhd,khd->hqk", qf[a:b], keys) * scale
        allowed = torch.arange(E + L, device=q.device)[None, :] <= (E + torch.arange(L, device=q.device))[:, None]
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        out[a:b] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals)
    return out.view(B, S, Hq, D)


def test_prefix_attention():
    torch.manual_seed(3)
    B, S, S_p, Hq, Hkv, D = 2, 20, 9, 8, 2, 32
    q = torch.randn(B, Hq, S, D, device=DEVICE, dtype=BF16)
    k = torch.randn(B, Hkv, S, D, device=DEVICE, dtype=BF16)
    v = torch.randn(B, Hkv, S, D, device=DEVICE, dtype=BF16)
    kp = torch.randn(B, Hkv, S_p, D, device=DEVICE, dtype=BF16)
    vp = torch.randn(B, Hkv, S_p, D, device=DEVICE, dtype=BF16)
    # row 0: segments of 8 and 12; row 1: segments of 15 and 5. Prefix lengths 4, 0, 9, 5, with
    # every prefix slot owned by some segment (row 0's padding goes to its second segment)
    cu_q = torch.tensor([0, 8, 20, 35, 40], device=DEVICE, dtype=torch.int32)
    cu_p = torch.tensor([0, 4, 9, 18, 18], device=DEVICE, dtype=torch.int32)
    scale = D ** -0.5
    ref = reference_prefix_attention(q, k, v, kp, vp, cu_q, cu_p, scale)
    got = prefix_varlen_attention(q, k, v, kp, vp, cu_q, S, cu_p, S_p, softmax_scale=scale)
    err = (got.float() - ref).abs().max().item()
    assert err < 2e-2, f"flash prefix attention off the reference by {err}"
    has_flash = attention._HAS_FLASH
    attention._HAS_FLASH = False
    try:
        fallback = prefix_varlen_attention(q, k, v, kp, vp, cu_q, S, cu_p, S_p, softmax_scale=scale)
    finally:
        attention._HAS_FLASH = has_flash
    err_fb = (fallback.float() - ref).abs().max().item()
    assert err_fb < 2e-2, f"SDPA fallback off the reference by {err_fb}"
    print(f"3. prefix attention equals [evidence; own] causal attention (flash {err:.1e}, sdpa {err_fb:.1e}) PASS")

    # 4. the segments with an empty prefix (the last one of row 1) read exactly plain causal attention
    plain = varlen_attention(q, k, v, cu_q, S, softmax_scale=scale)
    empty = slice(15, 20)
    assert torch.equal(got[1, empty], plain[1, empty]), "an empty prefix changed the output"
    print("4. a segment with an empty prefix is bit-identical to plain causal attention  PASS")


def run(model, ids, **kw):
    with torch.inference_mode():
        return model(ids, return_hidden=True, skip_mtp=True, **kw).clone()


def main():
    test_mode_and_load()

    plain = build(P)
    kv = make_live(build(KV))
    test_prefix_positions(kv)
    test_prefix_attention()

    # 5. bit identity without evidence, same weights
    port_only = ("key_adapter", "value_adapter", "log_memory_scale", "evidence_loop_scale",
                 "evidence_gate_scale", "evidence_reader_kv")
    fresh = build(KV)
    missing = fresh.load_state_dict(plain.state_dict(), strict=False).missing_keys
    assert all(any(p in k for p in port_only) for k in missing), missing
    B, S = 2, 24
    ids = torch.randint(1, Q_HI, (B, S), device=DEVICE)
    assert torch.equal(run(plain, ids), run(fresh, ids)), "the kv port is not neutral without evidence"
    print("5. no evidence attached is bit-identical to the model without the port        PASS")

    # 6. live and content dependent, one row per document
    evidence = evidence_for(kv, [0, 1], 10, 2)
    other = evidence_for(kv, [0, 1], 10, 2, seed=2)
    base, full = run(kv, ids), run(kv, ids, evidence=evidence)
    assert not torch.equal(base[0], full[0]), "evidence did not reach loop 1"
    assert not torch.equal(full[0], run(kv, ids, evidence=other)[0]), "the read ignores the content"
    print("6. evidence moves loop 1, and different evidence moves it differently          PASS")

    # 7. visibility in a packed row: documents 0 (tokens 0..9) and 1 (10..23) of row 0, row 1 whole
    doc = torch.tensor([[0] * 10 + [1] * 14, [2] * 24], device=DEVICE)
    cu, max_len = cu_seqlens_from_doc_ids(doc)
    n_seg = int(cu.numel() - 1)

    def packed(seed_1):
        # row 0 holds both documents' evidence: 6 tokens for segment 0, then 8 for segment 1
        # row 1: one document, 14 evidence tokens in two chunks. Everything but document 1's
        # evidence comes from fixed seeds
        g = torch.Generator(device="cpu").manual_seed(seed_1)
        ev = torch.randint(EV_LO, P["vocab_size"], (2, 14), generator=torch.Generator().manual_seed(8))
        chunk = torch.tensor([[0] * 6 + [1] * 8, [2] * 7 + [3] * 7])
        seg = torch.tensor([[0] * 6 + [1] * 8, [2] * 14])
        ev[0, 6:] = torch.randint(EV_LO, P["vocab_size"], (8,), generator=g)
        keys = torch.randn(3, P["ir_dim"], generator=torch.Generator().manual_seed(9))
        keys[1] = torch.randn(P["ir_dim"], generator=g)
        return kv.build_evidence(ev.to(DEVICE), chunk.to(DEVICE), seg.to(DEVICE), n_seg,
                                 chunk_keys=keys.to(DEVICE, BF16),
                                 chunk_segments=torch.tensor([0, 1, 2], device=DEVICE))

    a = run(kv, ids, cu_seqlens=cu, max_seqlen=max_len, evidence=packed(5))
    b = run(kv, ids, cu_seqlens=cu, max_seqlen=max_len, evidence=packed(6))
    assert torch.equal(a[:, 0, :10], b[:, 0, :10]), (
        f"document 0 read document 1's evidence: max delta {(a[:, 0, :10] - b[:, 0, :10]).abs().max().item()}"
    )
    assert not torch.equal(a[:, 0, 10:], b[:, 0, 10:]), "document 1 ignored its own evidence"
    assert torch.equal(a[:, 1], b[:, 1]), "row 1 moved with row 0's evidence"
    print("7. a document reads only its own evidence in a packed row                      PASS")

    # 8. zero-init gate: exactly the ungated read; a nonzero scale is a live gate
    with torch.no_grad():
        kv.moe.evidence_gate_scale.zero_()
    gated0 = run(kv, ids, evidence=evidence)
    real_mass = kv.moe._selector_chunk_mass
    kv.moe._selector_chunk_mass = lambda memory: None
    try:
        ungated = run(kv, ids, evidence=evidence)
    finally:
        kv.moe._selector_chunk_mass = real_mass
    assert torch.equal(gated0, ungated), "a zero gate scale is not the ungated read"
    with torch.no_grad():
        kv.moe.evidence_gate_scale.fill_(2.0)
    assert not torch.equal(run(kv, ids, evidence=evidence), gated0), "the gate is dead"
    with torch.no_grad():
        kv.moe.evidence_gate_scale.zero_()
    print("8. zero gate scale equals the ungated read, a nonzero scale moves it           PASS")

    # 9. read ablation
    L = P["n_loops"]
    full = run(kv, ids, evidence=evidence)
    assert torch.equal(full, run(kv, ids, evidence=evidence, reader_sites_kept=L))
    assert torch.equal(run(kv, ids, evidence=evidence, reader_sites_kept=0), base)
    assert kv.moe.last_reader_output is None, "j = 0 left a reader output behind"
    one = run(kv, ids, evidence=evidence, reader_sites_kept=1)
    assert torch.equal(one[0], full[0]) and not torch.equal(one[0], base[0])
    assert not torch.equal(one[2], full[2])
    run(kv, ids, evidence=evidence)
    assert kv.moe.last_reader_output is not None
    print("9. read ablation: j = depth is full, j = 0 is no evidence, j = 1 keeps loop 1  PASS")

    # 10. gradient through shared_attn into the evidence encoder
    kv.train()
    kv.zero_grad(set_to_none=True)
    hidden = kv(ids, return_hidden=True, skip_mtp=True, evidence=evidence_for(kv, [0, 1], 10, 2))
    hidden = hidden[0] if isinstance(hidden, tuple) else hidden
    hidden.float().pow(2).mean().backward()
    grad = kv.gemma_decoder.embed_tokens.weight.grad
    assert grad is not None and grad[EV_LO:].abs().sum().item() > 0, "no gradient reached the evidence tokens"
    assert kv.moe.evidence_loop_scale.grad is not None and kv.moe.evidence_loop_scale.grad.abs().sum() > 0
    kv.eval()
    print("10. the evidence encoder and the per loop gain get a gradient                   PASS")

    # 11. guards
    cache = KVCache.for_model(kv)
    refused = False
    try:
        with torch.inference_mode():
            kv(ids[:1], kv_cache=cache, skip_mtp=True, evidence=evidence_for(kv, [0], 6, 1))
    except AssertionError:
        refused = True
    assert refused, "a KV cache with the kv reader and evidence did not raise"
    print("11. a KV cache with evidence on the kv reader raises                            PASS")
    print("all kv reader checks passed")


if __name__ == "__main__":
    main()

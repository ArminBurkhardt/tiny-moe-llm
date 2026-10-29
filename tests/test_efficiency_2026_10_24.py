"""Four exact efficiency and gradient changes in the model.

1. **Evidence padding is isolated in the encoder.** Every padding token (negative chunk id) is its
   own length-1 segment, so the encoder no longer runs quadratic attention over a row's padded
   tail. Real tokens keep the same segments, so their encoder states and the final logits are
   unchanged; the segment count grows by exactly the pad count.
2. **Null picks skip the grouped GEMM.** ``grouped_mlp_forward`` slices the dense-slot picks off
   before the GEMM and matches the old layer forward and gradients; the dense rows are exactly zero.
3. **Groundedness under the key/value reader is a head-only gradient**: the trunk's ``shared_attn``
   gets nothing from the term, the head does.
4. **The non-hidden eval path norms only the last loop**: logits equal the ``return_hidden`` path's
   last loop through the head.

GPU required (flash varlen, TE). Prints timings for 1 and 2.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import modules.model.transformer as transformer_mod
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.evidence import chunk_position_ids
from modules.model.experts import grouped_mlp_forward
from modules.model.moe import ParallelSparseMoELayer
from modules.model.transformer import TinyMoETransformer

DEVICE = "cuda"
BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=1024, hidden_size=256, intermediate_size=768,
    head_dim=32, num_layers=4, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2, lm_head_factor=4,
)


def build(**extra):
    torch.manual_seed(0)
    model = TinyMoETransformer(**dict(P, **extra)).to(DEVICE).to(BF16).eval()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    return model


def timed(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def test_encoder_padding():
    model = build(evidence_port=True)
    widths = [int(v) for v in os.environ.get("EV_WIDTHS", "3000,400").split(",")]
    S_ev = widths[0]
    ids = torch.randint(1, P["vocab_size"], (len(widths), S_ev))
    chunk = torch.full((len(widths), S_ev), -1, dtype=torch.long)
    for r, w in enumerate(widths):
        chunk[r, :w] = torch.arange(w) // 100 + 100 * r
    ids[chunk < 0] = 0
    ids, chunk = ids.to(DEVICE), chunk.to(DEVICE)
    pos = chunk_position_ids(chunk)
    real = chunk >= 0

    states = {}
    for flag in (False, True):
        transformer_mod.EVIDENCE_PAD_SEGMENT = 128 if flag else 0
        with torch.no_grad():
            states[flag] = model._encode_evidence(ids, chunk, pos)
    transformer_mod.EVIDENCE_PAD_SEGMENT = 128
    assert torch.equal(states[False][real], states[True][real]), "real encoder states moved"
    print("1a. real evidence token encoder states bit identical                          PASS")

    n_old = cu_seqlens_from_doc_ids(chunk)[0].numel() - 1
    # expected: every row's old pad run becomes one segment per 128-token block of absolute position
    padded_rows, blocks = 0, 0
    for r in range(len(widths)):
        p = (~real[r]).nonzero().flatten()
        if p.numel():
            padded_rows += 1
            blocks += int((p // 128).unique().numel())
    n_new = n_old - padded_rows + blocks
    seg = chunk.clone()
    seg[~real] = -1 - (torch.arange(S_ev, device=DEVICE).expand_as(chunk) // 128)[~real]
    assert cu_seqlens_from_doc_ids(seg)[0].numel() - 1 == n_new
    print(f"1b. segments {n_old} -> {n_new}: {int((~real).sum())} pad tokens in {blocks} blocks PASS")

    S = 64
    input_ids = torch.randint(1, P["vocab_size"], (2, S), device=DEVICE)
    seg = torch.tensor([[0], [1]], device=DEVICE).expand(len(widths), S_ev).contiguous()
    owners = []
    for r in range(len(widths)):
        owners += [r] * int(chunk[r][real[r]].unique().numel())
    keys = torch.randn(len(owners), P["ir_dim"], device=DEVICE, dtype=BF16)
    logits = {}
    for flag in (False, True):
        transformer_mod.EVIDENCE_PAD_SEGMENT = 128 if flag else 0
        with torch.no_grad():
            ev = model.build_evidence(ids, chunk, seg, 2, chunk_keys=keys,
                                      chunk_segments=torch.tensor(owners, device=DEVICE))
            logits[flag] = model(input_ids, skip_mtp=True, evidence=ev)
    transformer_mod.EVIDENCE_PAD_SEGMENT = 128
    assert torch.equal(logits[False], logits[True]), (logits[False] - logits[True]).abs().max()
    print("1c. final logits with evidence attached bit identical                         PASS")

    def enc(flag):
        transformer_mod.EVIDENCE_PAD_SEGMENT = 128 if flag else 0
        with torch.no_grad():
            model._encode_evidence(ids, chunk, pos)
    t_old, t_new = timed(lambda: enc(False)), timed(lambda: enc(True))
    transformer_mod.EVIDENCE_PAD_SEGMENT = 128
    print(f"    encoder forward, widths {widths} padded to {S_ev}: old {t_old:.2f} ms, new {t_new:.2f} ms")


def test_null_rows(dtype=BF16, tol=1e-2):
    torch.manual_seed(0)
    layer = ParallelSparseMoELayer(256, 768, 8).to(DEVICE).to(dtype)
    B, S, K = 4, 1024, 2
    x = torch.randn(B, S, 256, device=DEVICE, dtype=dtype)
    dense = torch.rand(B, S, K, device=DEVICE) < 0.4
    idx = torch.randint(0, 8, (B, S, K), device=DEVICE)
    idx[idx == 5] = 4  # expert 5 gets no rows: a zero-row expert stays legal
    w = torch.rand(B, S, K, device=DEVICE, dtype=dtype) + 0.1
    idx = torch.where(dense, torch.zeros_like(idx), idx)
    w = torch.where(dense, torch.zeros_like(w), w)

    def run(fn):
        layer.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        out = fn(xi, w, idx)
        (out.float() ** 2).sum().backward()
        grads = [p.grad.float().clone() for p in layer.parameters()]
        return out.float().detach(), xi.grad.float(), grads

    o0, gx0, g0 = run(layer.forward)
    o1, gx1, g1 = run(lambda a, b, c: grouped_mlp_forward(layer, a, b, c))

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()
    r_out, r_x, r_w = rel(o1, o0), rel(gx1, gx0), max(rel(a, b) for a, b in zip(g1, g0))
    print(f"2a. {dtype} rel diff: output {r_out:.2e}, x grad {r_x:.2e}, weight grad {r_w:.2e}")
    assert r_out < tol and r_x < tol and r_w < tol
    z = grouped_mlp_forward(layer, x, torch.zeros(B, S, K, device=DEVICE, dtype=dtype), torch.zeros_like(idx))
    assert z.abs().max().item() == 0.0
    both = dense.all(-1)
    assert both.any() and o1[both].abs().max().item() == 0.0
    print("2b. dense rows exactly zero, all-null batch legal                             PASS")
    t_old = timed(lambda: layer(x, w, idx))
    t_new = timed(lambda: grouped_mlp_forward(layer, x, w, idx))
    print(f"    MoE layer forward, 40% dense picks, 4x1024: old {t_old:.2f} ms, new {t_new:.2f} ms")

    def fb(fn):
        xi = x.clone().requires_grad_(True)
        fn(xi, w, idx).float().sum().backward()
    t_old = timed(lambda: fb(layer.forward))
    t_new = timed(lambda: fb(lambda a, b, c: grouped_mlp_forward(layer, a, b, c)))
    print(f"    forward + backward: old {t_old:.2f} ms, new {t_new:.2f} ms")


def test_groundedness_detach():
    model = build(evidence_port=True, evidence_reader="kv", groundedness_head=True)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("g_proj.weight", "direct_gate.weight")):
                torch.nn.init.normal_(param, std=0.05)
        torch.nn.init.normal_(model.groundedness_head.out_proj.weight, std=0.1)
    B, S, S_ev = 2, 24, 12
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device=DEVICE)
    ev_ids = torch.randint(1, P["vocab_size"], (B, S_ev), device=DEVICE)
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6, [2] * 6 + [3] * 6], device=DEVICE)
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device=DEVICE)
    gold = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE)
    evidence = model.build_evidence(
        ev_ids, ev_chunk, ev_segment, 2,
        chunk_keys=torch.randn(4, P["ir_dim"], device=DEVICE, dtype=BF16),
        chunk_segments=torch.tensor([0, 0, 1, 1], device=DEVICE), chunk_gold=gold)
    positions = torch.zeros(B, S, dtype=torch.bool, device=DEVICE)
    positions[:, S // 2 - 1] = True
    answerable = torch.ones(B, S, dtype=torch.long, device=DEVICE)
    model.zero_grad(set_to_none=True)
    model(input_ids, skip_mtp=True, evidence=evidence)
    assert model.moe.last_reader_output.abs().max().item() > 0
    term = model.groundedness_term(evidence.chunk_gold, answerable, positions)
    term.backward()
    head = model.groundedness_head.out_proj.weight.grad
    assert head is not None and head.abs().max().item() > 0, "head got no gradient"
    for name, param in model.named_parameters():
        if "groundedness_head" in name:
            continue
        g = param.grad
        assert g is None or g.abs().max().item() == 0.0, f"trunk param {name} got a gradient"
    print("3.  groundedness: head learns, no other parameter gets a gradient (kv reader)  PASS")


def test_eval_path_last_loop_norm():
    model = build()
    input_ids = torch.randint(1, P["vocab_size"], (2, 32), device=DEVICE)
    with torch.no_grad():
        logits = model(input_ids, skip_mtp=True)
        hidden = model(input_ids, skip_mtp=True, return_hidden=True)
        ref = model.lm_head(hidden[-1])
    assert torch.equal(logits, ref), (logits.float() - ref.float()).abs().max()
    print("4.  eval path norms the last loop only, logits equal the hidden path's last    PASS")


if __name__ == "__main__":
    test_encoder_padding()
    test_null_rows()  # bf16 outputs: a different M may pick another kernel, a few bf16 ulps
    test_null_rows(torch.float32, 1e-5)
    test_groundedness_detach()
    test_eval_path_last_loop_norm()

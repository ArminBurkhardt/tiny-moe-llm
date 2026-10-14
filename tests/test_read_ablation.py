"""Read ablation: ``reader_sites_kept`` cuts every evidence read above a site count, nothing else.

Both routes by which chunk content reaches the residual are made live first (the reader's output
projection, the IR expert's value path and direct gate, the selector's source scale and the chunk
gate), so a cut that only removed one of them would show up as a difference below.

1. **None is the forward of today.** ``reader_sites_kept=None`` equals not passing the argument,
   on every loop's hidden state (``torch.equal``), and so does any count at or above the depth.
2. **Zero sites is no evidence.** ``reader_sites_kept=0`` equals ``evidence=None`` on every loop,
   and the reader's last output is cleared like a no-evidence forward's.
3. **One site is one loop of reading.** With one kept site loop 1 equals the full forward's loop 1
   and the later loops differ from it; with two the first two loops agree and the third differs.
   Loop 1 also differs from the no-evidence forward, so the agreement is not a dead reader.
4. **The selector is cut with the reader.** After a one-site forward the IR module has weights and
   mass for loop 0 only.
5. **Depth override**: at ``n_loops=4`` the sites number past the trained depth and ``j = 4`` equals
   None at that depth.
6. **Guards**: training mode and a KV cache with the argument both raise, and a negative count too.

GPU required (flash varlen, TE).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from modules.model.kv_cache import KVCache

DEVICE = "cuda"
BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2, lm_head_factor=4,
    evidence_port=True,
)


def build():
    torch.manual_seed(0)
    model = TinyMoETransformer(**P).to(DEVICE).to(BF16).eval()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("shared_evidence.attn.o_proj.weight", "g_proj.weight", "direct_gate.weight")):
                torch.nn.init.normal_(param, std=0.05)
            elif name.endswith("log_memory_scale"):
                param.fill_(3.0)
            elif name.endswith("evidence_gate_scale"):
                param.fill_(1.0)
            elif name.endswith(("evidence_loop_scale",)):
                param.fill_(1.0)
    return model


def evidence_for(model, B, S_ev=12):
    ev_ids = torch.randint(1, P["vocab_size"], (B, S_ev), device=DEVICE)
    ev_chunk = torch.stack([torch.tensor([0] * 6 + [1] * 6) + 2 * r for r in range(B)]).to(DEVICE)
    ev_segment = torch.stack([torch.full((S_ev,), r) for r in range(B)]).to(DEVICE)
    keys = torch.randn(2 * B, P["ir_dim"], device=DEVICE, dtype=BF16)
    chunk_segments = torch.arange(B, device=DEVICE).repeat_interleave(2)
    return model.build_evidence(ev_ids, ev_chunk, ev_segment, B, chunk_keys=keys, chunk_segments=chunk_segments)


def main():
    model = build()
    B, S = 2, 24
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device=DEVICE)
    evidence = evidence_for(model, B)
    ir = model.moe.ir_modules[0]
    L = P["n_loops"]

    def run(**kw):
        with torch.inference_mode():
            return model(input_ids, return_hidden=True, skip_mtp=True, **kw).clone()

    full = run(evidence=evidence)
    plain = run()
    assert full.shape[0] == L and not torch.equal(full, plain), "the live reader changed nothing"

    # 1. None and over-large counts are the forward of today
    assert torch.equal(full, run(evidence=evidence, reader_sites_kept=None))
    assert torch.equal(full, run(evidence=evidence, reader_sites_kept=L))
    assert torch.equal(full, run(evidence=evidence, reader_sites_kept=L + 5))
    print("1. None, j = n_loops and j > n_loops equal the forward without the argument   PASS")

    # 2. zero sites is no evidence
    none = run(evidence=evidence, reader_sites_kept=0)
    assert torch.equal(none, plain), "j = 0 differs from evidence=None"
    assert model.moe.last_reader_output is None, "j = 0 left a reader output behind"
    print("2. j = 0 equals evidence=None on every loop, reader output cleared             PASS")

    # 3. one and two sites
    one = run(evidence=evidence, reader_sites_kept=1)
    assert torch.equal(one[0], full[0]), "loop 1 moved with one kept site"
    assert not torch.equal(one[0], plain[0]), "loop 1 reads nothing: the agreement above is vacuous"
    assert not torch.equal(one[1], full[1]) and not torch.equal(one[2], full[2])
    two = run(evidence=evidence, reader_sites_kept=2)
    assert torch.equal(two[0], full[0]) and torch.equal(two[1], full[1])
    assert not torch.equal(two[2], full[2])
    print("3. j = 1 keeps loop 1 and changes loops 2, 3; j = 2 changes loop 3 only      PASS")

    # 4. the selector goes with the reader
    run(evidence=evidence, reader_sites_kept=1)
    assert sorted(ir.memory_weights_by_loop) == [0], sorted(ir.memory_weights_by_loop)
    assert sorted(ir.memory_mass_by_loop) == [0], sorted(ir.memory_mass_by_loop)
    run(evidence=evidence)
    assert sorted(ir.memory_weights_by_loop) == list(range(L))
    print("4. an ablated loop leaves no selector weights or mass                         PASS")

    # 5. depth override
    full4 = run(evidence=evidence, n_loops=L + 1)
    assert full4.shape[0] == L + 1
    assert torch.equal(full4, run(evidence=evidence, n_loops=L + 1, reader_sites_kept=L + 1))
    assert torch.equal(full4[0], run(evidence=evidence, n_loops=L + 1, reader_sites_kept=1)[0])
    print(f"5. sites number past the trained depth (n_loops={L + 1})                              PASS")

    # 6. guards
    def raises(fn):
        try:
            fn()
        except AssertionError:
            return True
        return False

    model.train()
    assert raises(lambda: model(input_ids, return_hidden=True, skip_mtp=True, evidence=evidence, reader_sites_kept=1))
    model.eval()
    assert raises(lambda: run(evidence=evidence, reader_sites_kept=-1))
    cache = KVCache.for_model(model)
    assert raises(lambda: model(input_ids[:1], kv_cache=cache, skip_mtp=True, evidence=None, reader_sites_kept=1))
    print("6. training mode, a negative count and a KV cache raise                       PASS")


if __name__ == "__main__":
    main()

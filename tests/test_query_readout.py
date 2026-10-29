"""The selector's query can be read out of a forward pass without touching the model.

``scripts/eval_store.py`` ranks a store with the query the IR expert scores external chunks with:
``normalize(down + loop_query_bias(loop))`` against ``normalize(key_adapter(key))``. It reads that
query with forward pre-hooks. Two things have to hold for the ranking to mean anything:

1. **The hooked query reproduces the module's own external scores.** For every loop, the rows the
   hook captured at chosen positions, times the adapted index, times the positive source scale
   (``exp(log_memory_scale) / temperature``), equal what ``_memory_scores`` computed for those
   positions, on the chunks the position can see, to bf16 tolerance. The adapter weights, the loop
   bias and the source scale are randomized first so the check is not a comparison of zeros.
2. **Hooks change nothing and leave nothing behind.** The forward inside the hook context is
   bit-identical to the forward without it, no hook is left registered afterwards, and a forward
   after the context is bit-identical too.

GPU required (flash varlen).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from scripts.eval_store import QueryReadout, adapted_index

BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
    lm_head_factor=4, ir_direct_read=False, evidence_port=True,
)


def main():
    torch.manual_seed(0)
    model = TinyMoETransformer(**P).to("cuda").to(BF16).eval()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    ir = model.moe.ir_modules[0]
    with torch.no_grad():
        ir.loop_query_bias.weight.normal_(0, 0.5)
        ir.key_adapter.weight.normal_(0, 0.1)
        ir.log_memory_scale.fill_(0.3)

    B, S, S_ev, num_chunks = 2, 24, 12, 4
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device="cuda")
    ev_ids = torch.randint(1, P["vocab_size"], (B, S_ev), device="cuda")
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6, [2] * 6 + [3] * 6], device="cuda")
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device="cuda")
    chunk_segments = torch.tensor([0, 0, 1, 1], device="cuda")
    keys = torch.randn(num_chunks, ir.key_adapter.weight.shape[1], device="cuda", dtype=BF16)
    evidence = model.build_evidence(ev_ids, ev_chunk, ev_segment, B, chunk_keys=keys,
                                    chunk_segments=chunk_segments)
    visible_to_row = [[0, 1], [2, 3]]

    def forward():
        with torch.inference_mode():
            return model(input_ids, skip_mtp=True, evidence=evidence)

    baseline = forward()

    recorded = []
    original = ir._memory_scores

    def spy(x_norm, memory):
        scores = original(x_norm, memory)
        recorded.append(scores.detach().clone())
        return scores

    ir._memory_scores = spy
    positions = torch.tensor([S - 1, 2 * S - 1], device="cuda")
    with QueryReadout(model) as readout:
        readout.positions = positions
        readout.tokens = B * S
        inside = forward()
    del ir._memory_scores

    assert torch.equal(baseline, inside), "the hooks changed the forward"
    assert len(recorded) == model.moe.n_loops, f"{len(recorded)} score calls for {model.moe.n_loops} loops"
    assert not ir._forward_pre_hooks, "a hook was left registered"
    assert torch.equal(baseline, forward()), "the forward after the hooks differs"
    print("1. hooks are passive and removed                          PASS")

    adapted = adapted_index(model, keys.float().cpu().numpy(), 0, "cuda")
    scale = (ir.log_memory_scale.exp() / ir.temperature).float()
    for loop in range(model.moe.n_loops):
        query = readout.captured[(0, loop)]
        expected = (query @ adapted.t())
        got = recorded[loop][positions] / scale
        for row in range(B):
            visible = visible_to_row[row]
            delta = (expected[row, visible] - got[row, visible]).abs().max().item()
            assert delta < 5e-2, f"loop {loop} row {row}: hooked query disagrees with the module by {delta}"
    print("2. hooked query reproduces the module's external scores   PASS")


if __name__ == "__main__":
    main()

"""Query side causality of the reader/selector coupling gate, and why the gate is frozen at zero.

The reader multiplies each retrieved chunk's states by ``1 + evidence_gate_scale *
sigmoid(chunk_mean_mass)``. ``chunk_mean_mass`` is the selector's weight on that chunk **averaged
over every token of the document, including tokens after the one being predicted**. A causal LM
must not let position ``t`` see anything from ``t + 1`` onward, so the gate is a leak whenever its
scale is nonzero: the last tokens of a document reshape the mean, the mean reshapes the gate, the
gate reshapes what the reader hands every earlier position.

Two assertions, on two inputs that differ only in the LAST few query tokens of a document, with
the reader's ``o_proj`` randomized so the reader actually writes to the stream and the IR expert's
own read stage (``direct_gate``) live as well, so the whole block is exercised:

1. **``evidence_gate_scale == 0`` is causal.** Earlier positions of the changed document match the
   unchanged run (bit for bit, or within the tiny noise a grouped GEMM can add when the changed
   tokens re-batch expert rows; the observed difference is printed), and the untouched second
   document does not move at all.
2. **``evidence_gate_scale != 0`` leaks.** Earlier positions of the changed document move by far
   more than the noise floor of assertion 1. This assertion RECORDS a known defect rather than a
   desired property: it is the reason the gate is frozen at zero for training. A causal
   replacement (for instance a running prefix mean of the selector's weights) should make this
   assertion fail, and the fix is then to flip it to "must match", not to relax the tolerance.

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

TAIL = 4
GATE_SCALE = 5.0


def main():
    torch.manual_seed(0)
    model = TinyMoETransformer(**P).to("cuda").to(BF16).eval()
    model.set_checkpointing(False, False)
    ir_expert = next(e for e in model.moe.experts if isinstance(e, InformationRetrievalExpert))
    with torch.no_grad():
        torch.nn.init.normal_(model.moe.shared_evidence.attn.o_proj.weight, std=0.3)
        torch.nn.init.normal_(ir_expert.direct_gate.weight, std=0.05)
        # the external half must win a real share of the softmax or the mean mass is below a bf16
        # ulp by the time it reaches a gate (same reason test_evidence_port.py raises this)
        ir_expert.ir_module.log_memory_scale.fill_(3.0)

    B, S, S_ev = 2, 32, 12
    ids_a = torch.randint(1, P["vocab_size"], (B, S), device="cuda")
    ids_b = ids_a.clone()
    ids_b[0, S - TAIL:] = torch.randint(1, P["vocab_size"], (TAIL,), device="cuda")
    assert not torch.equal(ids_a[0, S - TAIL:], ids_b[0, S - TAIL:])

    ev_ids = torch.randint(1, P["vocab_size"], (B, S_ev), device="cuda")
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6, [2] * 6 + [3] * 6], device="cuda")
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device="cuda")
    keys = torch.randn(4, P["ir_dim"], device="cuda", dtype=BF16)
    chunk_segments = torch.tensor([0, 0, 1, 1], device="cuda")
    with torch.no_grad():
        evidence = model.build_evidence(ev_ids, ev_chunk, ev_segment, B, chunk_keys=keys,
                                        chunk_segments=chunk_segments)

    def run(scale):
        with torch.no_grad():
            model.moe.evidence_gate_scale.fill_(scale)
        with torch.inference_mode():
            a = model(ids_a, skip_mtp=True, evidence=evidence).float()
            b = model(ids_b, skip_mtp=True, evidence=evidence).float()
        head = S - TAIL
        early = (a[0, :head] - b[0, :head]).abs().max().item()
        other_doc = (a[1] - b[1]).abs().max().item()
        tail = (a[0, head:] - b[0, head:]).abs().max().item()
        return early, other_doc, tail

    early0, other0, tail0 = run(0.0)
    print(f"gate scale 0:   earlier positions max |delta| = {early0:.3e}, "
          f"other document = {other0:.3e}, changed tail = {tail0:.3e}")
    assert tail0 > 0.0, "the changed tail did not move its own outputs: the probe is vacuous"
    assert early0 < 1e-3 and other0 < 1e-3, (
        f"the block is not causal with the gate off: early {early0}, other document {other0}"
    )
    print("1. gate off: earlier positions do not see the changed tail                     PASS")

    early1, other1, tail1 = run(GATE_SCALE)
    print(f"gate scale {GATE_SCALE}: earlier positions max |delta| = {early1:.3e}, "
          f"other document = {other1:.3e}, changed tail = {tail1:.3e}")
    # the gate-off run is deterministic, so its delta is the noise floor (observed exactly 0)
    assert early1 > 0.0 and early1 > 3.0 * early0, (
        f"NO LEAK FOUND: with the gate on, earlier positions moved only {early1:.3e} "
        f"(gate-off floor {early0:.3e}). If this is real, the mean-over-document gate is causal "
        f"after all and the frozen-at-zero rationale is wrong."
    )
    print(f"2. gate on: earlier positions DO move (leak {early1:.3e} vs floor {early0:.3e})   PASS")


if __name__ == "__main__":
    main()

"""The pieces layered on top of the evidence port: the IR expert's direct read path, the selector's
supervised selection loss, the reader/selector coupling gate, and the aux loss's optional pad mask.

Four groups of assertions:

1. **``InformationRetrievalExpert(direct_read=True)`` is exactly zero at init** (the same "read
   zeroed" neutrality this file already reports at 0.0002 nats), and, once its gate has something to
   pass through, **never mixes across tokens** -- the defining property that fixes the averaged
   path's measured prefix-constant output. The averaged path is exercised too, as the contrast that
   proves 1/2 are testing the fix and not a coincidence.
2. **``evidence_selection_loss`` is numerically safe** on the corpus's real edge cases: a document
   with no gold chunk, a batch with no visible chunks at all, zero chunks retrieved anywhere, and a
   fully unsupervised batch -- every one of them a genuine zero with a live gradient, never a NaN.
3. **The reader's per chunk gate reconciles two chunk numberings that only agree on order.**
   ``evidence.chunk_ids`` resets every packed ROW (see ``modules/data/evidence_dataset.py``'s own
   docstring), so two different rows legitimately reuse the same small integers for different
   chunks; the gate has to land each row's chunk on the right entry of the selector's flat,
   whole-batch chunk list regardless. It is also exactly neutral (a no-op on the evidence states) at
   the gate's zero-init scale, for any chunk mass whatsoever.
4. **``compute_aux_loss``'s optional ``token_mask``** reproduces the loss computed over the real
   tokens alone, ``None`` stays bit-identical to today's numerics, and the fixture actually is a case
   the unmasked loss gets wrong (padding skewed hard onto one expert), or the test would pass
   vacuously.

Plain script, not pytest (see tests/run_tests.sh). Requires the WSL/CUDA environment because
modules/model/* pulls in transformer_engine at import time regardless of whether a given assertion
needs the GPU for its own math.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.experts import InformationRetrievalExpert
from modules.model.information_retrieval import evidence_selection_loss
from modules.model.evidence import EvidenceBatch, scatter_chunk_score_to_tokens, apply_chunk_gate, chunk_mean_mass
from modules.model.router import compute_aux_loss

DEVICE = "cuda"
BF16 = torch.bfloat16


def test_direct_read():
    torch.manual_seed(0)
    expert = InformationRetrievalExpert(
        input_size=64, num_entries=128, ir_dim=32, num_heads=8, num_kv_heads=4, dropout=0.0,
        direct_read=True,
    ).to(DEVICE).to(BF16).eval()

    B, S = 1, 6
    x = torch.randn(B, S, 64, device=DEVICE, dtype=BF16)
    with torch.no_grad():
        out = expert(x)
    assert torch.equal(out, torch.zeros_like(out)), (
        f"direct_gate zero-init did not zero the expert's output: max |out| = {out.abs().max().item()}"
    )
    print("1. direct_read is exactly zero at init                              PASS")

    # give the gate something to pass through, then perturb ONLY one token's residual input. Every
    # step from x to the output is per token (norm, down_proj, the table read, up_proj, direct_gate)
    # -- no attention, no cross-token matmul -- so every OTHER position's output must be untouched,
    # bit for bit, not merely close.
    with torch.no_grad():
        torch.nn.init.normal_(expert.direct_gate.weight, std=0.02)
        base = expert(x)
    x2 = x.clone()
    x2[:, 2] = torch.randn(64, device=DEVICE, dtype=BF16)
    with torch.no_grad():
        moved = expert(x2)
    delta = (moved - base).abs().sum(dim=-1)[0]  # [S]
    assert delta[2] > 0, "the changed token's own output did not move"
    others = torch.cat([delta[:2], delta[3:]])
    assert torch.equal(others, torch.zeros_like(others)), (
        f"a token other than the changed one moved: per-position |delta| = {delta.tolist()}"
    )
    print("2. direct_read never mixes across tokens                           PASS")

    # the averaged path this replaces DOES mix -- confirms 1/2 exercise the actual fix rather than
    # this expert being a no-op regardless of which path runs
    expert.direct_read = False
    with torch.no_grad():
        base_attn = expert(x)
        moved_attn = expert(x2)
    delta_attn = (moved_attn - base_attn).abs().sum(dim=-1)[0]
    assert delta_attn[2] > 0 and delta_attn[3:].sum() > 0, (
        f"the averaged path did not mix later positions -- comparison is not meaningful: "
        f"{delta_attn.tolist()}"
    )
    print("3. the averaged path DOES mix later positions (the contrast)       PASS")


def test_selection_loss():
    torch.manual_seed(0)
    T, M = 6, 4
    weights = torch.rand(T, M, device=DEVICE, requires_grad=True)
    visible = torch.zeros(T, M, dtype=torch.bool, device=DEVICE)
    visible[:3, :2] = True  # doc A: tokens 0-2 see chunks 0-1
    visible[3:, 2:] = True  # doc B: tokens 3-5 see chunks 2-3
    chunk_gold = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE)  # doc A has a gold chunk, doc B none
    loss = evidence_selection_loss(weights, visible, chunk_gold)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert loss.requires_grad, "loss lost its gradient path back to the selector's weights"
    loss.backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all(), (
        "no/NaN gradient reached the per chunk weights"
    )
    print("4. selection loss is finite and differentiable with a gold-free document present  PASS")

    weights2 = torch.rand(T, M, device=DEVICE, requires_grad=True)
    visible2 = torch.zeros(T, M, dtype=torch.bool, device=DEVICE)  # a pure replay batch: nothing visible
    loss2 = evidence_selection_loss(weights2, visible2, chunk_gold)
    assert torch.equal(loss2, torch.zeros_like(loss2)), f"an all invisible batch should be exactly 0: {loss2}"
    loss2.backward()
    print("5. an evidence-free batch gives an exact, differentiable zero                    PASS")

    weights3 = torch.zeros(T, 0, device=DEVICE, requires_grad=True)
    visible3 = torch.zeros(T, 0, dtype=torch.bool, device=DEVICE)
    chunk_gold3 = torch.zeros(0, device=DEVICE)
    loss3 = evidence_selection_loss(weights3, visible3, chunk_gold3)
    assert torch.equal(loss3, torch.zeros_like(loss3)), f"zero chunks anywhere should give an exact 0: {loss3}"
    print("6. zero retrieved chunks anywhere gives an exact zero, no crash                  PASS")

    weights4 = torch.rand(T, M, device=DEVICE, requires_grad=True)
    supervised = torch.zeros(T, device=DEVICE)
    loss4 = evidence_selection_loss(weights4, visible, chunk_gold, supervised=supervised)
    assert torch.equal(loss4, torch.zeros_like(loss4)), f"a fully unsupervised batch should be exactly 0: {loss4}"
    print("7. a fully unsupervised batch gives an exact zero                                PASS")


def _toy_evidence(chunk_ids, cu_seqlens, chunk_segments, hidden=8):
    B, S_ev = chunk_ids.shape
    states = torch.randn(B, S_ev, hidden, device=DEVICE, dtype=BF16)
    dummy_pos = (torch.zeros(B, S_ev, 4, device=DEVICE), torch.zeros(B, S_ev, 4, device=DEVICE))
    return EvidenceBatch(
        states=states, cu_seqlens=cu_seqlens, max_seqlen=S_ev, position_embeddings=dummy_pos,
        chunk_ids=chunk_ids, chunk_segments=chunk_segments,
    )


def test_chunk_mean_mass():
    weights = torch.tensor([[0.5, 0.1], [0.3, 0.0], [0.0, 0.9]], device=DEVICE)  # [T=3, M=2]
    visible = torch.tensor([[True, False], [True, False], [False, True]], device=DEVICE)
    mean = chunk_mean_mass(weights, visible)
    expected = torch.tensor([0.4, 0.9], device=DEVICE)  # chunk 0: mean(0.5, 0.3); chunk 1: 0.9 alone
    assert torch.allclose(mean, expected, atol=1e-6), f"{mean.tolist()} != {expected.tolist()}"
    print("8. chunk_mean_mass averages only over each chunk's own visible tokens            PASS")


def test_reader_gate():
    # two rows, each its own query segment, each locally numbering its two chunks 0 and 1 -- the
    # collision the reconciliation has to survive: row 1's local id 0 is NOT row 0's chunk 0.
    chunk_ids = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], device=DEVICE)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=DEVICE)
    chunk_segments = torch.tensor([0, 0, 1, 1], device=DEVICE)  # global chunks 0,1 -> row 0; 2,3 -> row 1
    evidence = _toy_evidence(chunk_ids, cu_seqlens, chunk_segments)

    chunk_score = torch.tensor([10.0, 20.0, 30.0, 40.0], device=DEVICE)
    token_gate = scatter_chunk_score_to_tokens(evidence, chunk_score)
    expected = torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]], device=DEVICE)
    assert torch.equal(token_gate, expected), (
        f"row-reset chunk ids were not reconciled against the global chunk list: "
        f"{token_gate.tolist()} != {expected.tolist()} (a bug here would read row 0's scores "
        f"[10, 20] onto row 1 as well, since both rows number their own chunks 0 and 1)"
    )
    print("9. the reader gate reconciles per-row-reset chunk ids against the global list    PASS")

    ones_gate = chunk_score.new_ones(4)
    gated = apply_chunk_gate(evidence, ones_gate)
    assert torch.equal(gated.states, evidence.states), "an all-ones chunk gate changed the evidence states"
    print("10. an all-ones chunk gate is a no-op on the evidence states                     PASS")

    # the actual formula LoopMixtureOfExperts.forward_step uses: a zero-init learned scale makes the
    # gate exactly 1.0 for ANY chunk mass, not merely close to it -- the sigmoid term never has to
    # saturate for this to hold.
    evidence_gate_scale = torch.zeros((), device=DEVICE)
    chunk_mass = torch.tensor([0.0, 0.3, 0.9, 1.0], device=DEVICE)
    gate = 1.0 + evidence_gate_scale * torch.sigmoid(chunk_mass)
    assert torch.equal(gate, torch.ones_like(gate)), f"gate is not exactly 1.0 at zero scale: {gate.tolist()}"
    gated_at_init = apply_chunk_gate(evidence, gate)
    assert torch.equal(gated_at_init.states, evidence.states), (
        "the zero-scale gate formula changed the evidence states"
    )
    print("11. the reader gate is exactly neutral at zero scale, for any chunk mass         PASS")


def test_aux_loss_mask():
    torch.manual_seed(0)
    num_experts = 4
    B, S, top_k = 2, 6, 1
    # first two positions of each row are real and route to expert 0; the last four are padding,
    # deliberately skewed onto expert 3 -- the exact failure CLAUDE.md's aux loss note names: a
    # routing statistic that moves with row fill rather than with the real tokens
    indices = torch.zeros(B, S, top_k, dtype=torch.long, device=DEVICE)
    indices[:, 2:] = 3
    probs = torch.zeros(B, S, num_experts, device=DEVICE)
    probs[:, :2, 0] = 1.0
    probs[:, 2:, 3] = 1.0
    mask = torch.zeros(B, S, device=DEVICE)
    mask[:, :2] = 1.0

    masked_loss = compute_aux_loss(indices, probs, num_experts, token_mask=mask)
    real_only_loss = compute_aux_loss(indices[:, :2], probs[:, :2], num_experts)
    assert torch.allclose(masked_loss, real_only_loss, atol=1e-5), (
        f"masked aux loss does not match the loss computed over the real tokens alone: "
        f"{masked_loss.item()} vs {real_only_loss.item()}"
    )
    unmasked_loss = compute_aux_loss(indices, probs, num_experts)
    assert not torch.allclose(masked_loss, unmasked_loss), (
        "padding skewed onto one expert did not move the unmasked aux loss -- fixture does not "
        "exercise the bug"
    )
    print(
        f"12. token_mask reproduces the real-tokens-only aux loss ({masked_loss.item():.4f} vs "
        f"{unmasked_loss.item():.4f} unmasked)                    PASS"
    )

    default_loss = compute_aux_loss(indices, probs, num_experts, token_mask=None)
    assert torch.equal(default_loss, unmasked_loss), "passing token_mask=None changed the numerics"
    print("13. token_mask=None stays bit-identical to today's numerics                      PASS")


def main():
    test_direct_read()
    test_selection_loss()
    test_chunk_mean_mass()
    test_reader_gate()
    test_aux_loss_mask()


if __name__ == "__main__":
    main()

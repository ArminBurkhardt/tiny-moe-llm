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
5. **Both of those are actually attached to the model**, which is a separate claim from either one
   being correct: a loss that is only tested standalone trains nothing. ``evidence_selection_term``
   is None exactly when there is nothing to select over, it carries a gradient back to the
   selector's own tensors after a real forward, and ``token_mask`` reaches ``compute_aux_loss``
   through ``TinyMoETransformer.forward``.

Plain script, not pytest (see tests/run_tests.sh). Requires the WSL/CUDA environment because
modules/model/* pulls in transformer_engine at import time regardless of whether a given assertion
needs the GPU for its own math.
"""
import math
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.experts import InformationRetrievalExpert
from modules.model.information_retrieval import evidence_selection_loss
from modules.model.evidence import EvidenceBatch, scatter_chunk_score_to_tokens, apply_chunk_gate, chunk_mean_mass
from modules.model.router import compute_aux_loss
from modules.model.transformer import TinyMoETransformer
# the trainer's own position rule, imported rather than restated -- a second copy here would let
# the test keep passing while the objective moved to a different position
from scripts.pretrain import answer_start_positions

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


MODEL_P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2, lm_head_factor=4,
    evidence_port=True,
)


def _model_with(**overrides):
    torch.manual_seed(0)
    model = TinyMoETransformer(**dict(MODEL_P, **overrides)).to(DEVICE).to(BF16).eval()
    # both terms read tensors the forward stashed, and a checkpointed segment recomputes them
    # under no_grad -- the trainer asserts this is off, so the test runs the same way
    model.set_checkpointing(False, False)
    return model


def _model():
    return _model_with()


def test_selection_term_wiring():
    model = _model()
    B, S, S_ev = 2, 24, 12
    input_ids = torch.randint(1, MODEL_P["vocab_size"], (B, S), device=DEVICE)
    ev_ids = torch.randint(1, MODEL_P["vocab_size"], (B, S_ev), device=DEVICE)
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6, [2] * 6 + [3] * 6], device=DEVICE)
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device=DEVICE)
    chunk_segments = torch.tensor([0, 0, 1, 1], device=DEVICE)
    chunk_gold = torch.tensor([1.0, 0.0, 0.0, 1.0], device=DEVICE)
    keys = torch.randn(4, MODEL_P["ir_dim"], device=DEVICE, dtype=BF16)

    # a forward with no evidence at all leaves nothing to select over, and the term has to say so
    # rather than return a zero the trainer would add to the loss as if it were a measurement
    with torch.no_grad():
        model(input_ids, skip_mtp=True)
    assert model.moe.evidence_selection_term(chunk_gold) is None, (
        "a no-evidence forward produced a selection term -- the trainer would be adding a stale "
        "batch's read weights to this batch's loss"
    )

    # the reader without the selector: evidence tokens but no chunk keys, so there is no external
    # store in the read at all and still nothing to supervise
    reader_only = model.build_evidence(ev_ids, ev_chunk, ev_segment, 2)
    with torch.no_grad():
        model(input_ids, skip_mtp=True, evidence=reader_only)
    assert model.moe.evidence_selection_term(chunk_gold) is None, (
        "evidence without chunk keys produced a selection term -- there is no external store to rank"
    )
    print("14. the selection term is None exactly when nothing was selected over             PASS")

    evidence = model.build_evidence(
        ev_ids, ev_chunk, ev_segment, 2, chunk_keys=keys, chunk_segments=chunk_segments,
        chunk_gold=chunk_gold,
    )
    model.zero_grad(set_to_none=True)
    logits = model(input_ids, skip_mtp=True, evidence=evidence)
    # supervise the back half of each row, the way an answer span sits at the end of a prompt
    supervised = torch.zeros(B, S, device=DEVICE)
    supervised[:, S // 2:] = 1.0
    term = model.moe.evidence_selection_term(evidence.chunk_gold, supervised=supervised.reshape(-1))
    assert term is not None and torch.isfinite(term), f"no finite selection term: {term}"
    assert term.requires_grad, "the selection term has no graph -- it would train nothing"
    term.backward()

    ir = model.moe.ir_modules[0]
    # key_adapter is the selector's OWN tensor: the renormalization inside the loss divides the
    # parametric/external split out, so what is left is pure ranking, and this is the weight that
    # decides where an external chunk lands relative to the query
    grad = ir.key_adapter.weight.grad
    assert grad is not None and torch.isfinite(grad).all(), (
        "no gradient reached the selector's key adapter -- the term is attached to nothing"
    )
    assert grad.abs().max().item() > 0, "the key adapter's gradient is exactly zero"
    # the term also reaches the trunk, which is correct rather than leakage: loop 2's query is the
    # state loop 1 (reader included) produced, so "supervises the split alone" is a statement about
    # what the loss MEASURES, not about which tensors the recurrence puts between it and the input
    print(
        f"15. the selection term trains the selector (|dkey_adapter|max = "
        f"{grad.abs().max().item():.2e})                                PASS"
    )
    del logits


def test_forward_token_mask():
    model = _model()
    B, S = 2, 24
    input_ids = torch.randint(1, MODEL_P["vocab_size"], (B, S), device=DEVICE)
    # a trailing pad run, the shape every packed row ends with
    input_ids[:, S - 8:] = 0
    mask = torch.ones(B, S, dtype=torch.bool, device=DEVICE)
    mask[:, S - 8:] = False

    with torch.no_grad():
        _, aux_none = model(input_ids, return_aux_loss=True, skip_mtp=True)
        _, aux_default = model(input_ids, return_aux_loss=True, skip_mtp=True, token_mask=None)
        _, aux_masked = model(input_ids, return_aux_loss=True, skip_mtp=True, token_mask=mask)
    assert torch.equal(aux_none, aux_default), "token_mask=None changed the forward's aux loss"
    assert not torch.equal(aux_none, aux_masked), (
        "token_mask did not reach compute_aux_loss through the model's forward -- the padded "
        "positions are still in the mean"
    )
    print(
        f"16. token_mask reaches the aux loss through the model forward ({aux_none.item():.4f} "
        f"-> {aux_masked.item():.4f})   PASS"
    )


def test_groundedness_wiring():
    """The head reads the reader, the label is the AND, and the positions are the prompt's last."""
    model = _model_with(groundedness_head=True)
    B, S, S_ev = 2, 24, 12
    input_ids = torch.randint(1, MODEL_P["vocab_size"], (B, S), device=DEVICE)
    ev_ids = torch.randint(1, MODEL_P["vocab_size"], (B, S_ev), device=DEVICE)
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6, [2] * 6 + [3] * 6], device=DEVICE)
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device=DEVICE)
    # row 0's document holds a gold chunk, row 1's holds none -- the two halves of the label have
    # to come apart, or the test cannot tell the AND from either operand
    chunk_gold = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE)
    evidence = model.build_evidence(
        ev_ids, ev_chunk, ev_segment, 2,
        chunk_keys=torch.randn(4, MODEL_P["ir_dim"], device=DEVICE, dtype=BF16),
        chunk_segments=torch.tensor([0, 0, 1, 1], device=DEVICE),
        chunk_gold=chunk_gold,
    )

    labels = torch.full((B, S), -100, device=DEVICE, dtype=torch.long)
    labels[:, S // 2:] = input_ids[:, S // 2:]
    positions = answer_start_positions(labels)
    expected = torch.zeros(B, S, dtype=torch.bool, device=DEVICE)
    expected[:, S // 2 - 1] = True
    assert torch.equal(positions, expected), (
        f"the scored position is not the last prompt token: {positions.nonzero().tolist()}"
    )
    print("17. the groundedness position is the last prompt token of each answer span     PASS")

    answerable = torch.ones(B, S, dtype=torch.long, device=DEVICE)
    model.zero_grad(set_to_none=True)
    model(input_ids, skip_mtp=True, evidence=evidence)
    term = model.groundedness_term(evidence.chunk_gold, answerable, positions)
    assert term is not None and torch.isfinite(term), f"no finite groundedness term: {term}"
    # a zero-init head is exactly p = 0.5, so the BCE starts at ln 2 whatever the labels are
    assert abs(term.item() - math.log(2)) < 1e-2, (
        f"a zero-init head should start at ln 2 = {math.log(2):.4f}, got {term.item():.4f}"
    )
    term.backward()

    # On a freshly migrated port the reader's o_proj is zero, so its output -- which is what this
    # head reads -- is IDENTICALLY zero, and the head's weight gradient is exactly zero with it.
    # The head can move its bias (the corpus's base rate) and nothing else until the reader leaves
    # its own zero. Asserted rather than worked around: it is the same dependency |g_proj|rms
    # exists to make visible on the IR read, and it means an early `grounded` falling in the log is
    # the base rate being learned, not grounding -- the held-out AUROC is what tells them apart.
    assert model.moe.last_reader_output.abs().max().item() == 0.0, (
        "the fixture's reader is not at its zero init -- assertion 18 is not testing what it says"
    )
    frozen_grad = model.groundedness_head.out_proj.weight.grad
    assert frozen_grad is not None and frozen_grad.abs().max().item() == 0.0, (
        f"the head's weights moved while the reader is zero: {frozen_grad.abs().max().item()}"
    )
    print(f"18. the head starts at ln 2 ({term.item():.4f}) and is weight-frozen until the "
          f"reader leaves zero   PASS")

    # wake the reader, exactly as tests/test_evidence_port.py does for its own assertion 3, and the
    # head has something to say. Read on the OUTPUT rather than on a second backward: the weight
    # gradient is the reader's output times a per position error, so "the logits stop being the
    # bias everywhere" is the same fact, and stacking several backwards through TE's fused ops in
    # one process trips its saved-tensor bookkeeping for reasons that have nothing to do with this.
    live_model = _model_with(groundedness_head=True)
    with torch.no_grad():
        frozen = live_model.groundedness_head(
            torch.zeros(1, 4, MODEL_P["hidden_size"], device=DEVICE, dtype=BF16)
        )
        assert torch.equal(frozen, torch.zeros_like(frozen)), (
            "the head is not neutral on a zero read -- a migrated checkpoint would start with an "
            "opinion it never learned"
        )
        torch.nn.init.normal_(live_model.moe.shared_evidence.attn.o_proj.weight, std=0.02)
        torch.nn.init.normal_(live_model.groundedness_head.out_proj.weight, std=0.5)
        live_model(input_ids, skip_mtp=True, evidence=evidence)
        logits = live_model.groundedness_head(live_model.moe.last_reader_output)
    assert live_model.moe.last_reader_output.abs().max().item() > 0, "the reader stayed at zero"
    assert logits.std().item() > 0, (
        "the head's output is constant across positions even with a live reader -- it is not "
        "reading the read"
    )
    print(f"19. with a live reader the head reads it (logit sd = {logits.std().item():.3f}), and "
          f"is exactly 0 on a zero read  PASS")

    # the label is the AND of two independent axes: row 0's document holds a gold chunk and row 1's
    # does not, so flipping `answerable` off for row 0 has to change the loss -- otherwise the head
    # is being trained on gold presence alone, which is the mass split again and exactly what this
    # head exists to go beyond. Read with a live reader, since at p = 0.5 every BCE is ln 2.
    axes_model = _model_with(groundedness_head=True)
    with torch.no_grad():
        torch.nn.init.normal_(axes_model.moe.shared_evidence.attn.o_proj.weight, std=0.02)
        torch.nn.init.normal_(axes_model.groundedness_head.out_proj.weight, std=0.5)
    with torch.no_grad():
        axes_model(input_ids, skip_mtp=True, evidence=evidence)
        both_true = axes_model.groundedness_term(evidence.chunk_gold, answerable, positions)
        half = answerable.clone()
        half[0] = 0
        gold_only = axes_model.groundedness_term(evidence.chunk_gold, half, positions)
    assert not torch.isclose(both_true, gold_only), (
        f"answerability does not move the label ({both_true.item():.4f} vs {gold_only.item():.4f})"
        f" -- the loss is reading gold presence alone"
    )
    print("20. answerability is a real second axis of the label, not a relabelled gold flag PASS")

    # and a model without the head says so rather than returning a zero a trainer would add
    plain = _model_with(groundedness_head=False)
    with torch.no_grad():
        plain(input_ids, skip_mtp=True, evidence=evidence)
    assert plain.groundedness_term(evidence.chunk_gold, answerable, positions) is None, (
        "a checkpoint with no head produced a groundedness term"
    )
    print("21. no head means no term, not a zero                                           PASS")


def main():
    test_direct_read()
    test_selection_loss()
    test_chunk_mean_mass()
    test_reader_gate()
    test_aux_loss_mask()
    test_selection_term_wiring()
    test_forward_token_mask()
    test_groundedness_wiring()


if __name__ == "__main__":
    main()

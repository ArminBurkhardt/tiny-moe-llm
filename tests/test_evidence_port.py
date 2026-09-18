"""The evidence port must be invisible without a corpus, and must actually read one with it.

The port's defining property is that **a batch with no evidence runs the forward the model ran
before the port existed, bit for bit**. Everything downstream leans on it: one checkpoint serves
both modes, the replay fraction of a finetune genuinely protects the trunk instead of training the
port, and any movement in a gate is attributable to evidence rather than to the port's mere
presence. A tolerance would hide exactly the wiring mistake that matters.

Six assertions:

1. **Building the port changes nothing** when no evidence is attached -- bit-identical to a model
   built without it, given the same weights.
2. **Zero-init ``o_proj`` makes attaching a corpus neutral too.** Structural absence covers the
   no-corpus case; this covers the case that a corpus IS attached to a freshly migrated checkpoint,
   where a default-initialized reader would inject noise into a converged trunk on step 0.
3. **A trained reader moves the logits**, so 1 and 2 are facts about the zero rather than about the
   module being disconnected.
4. **The evidence content matters.** Change the evidence tokens and the output must change -- a
   reader that attends to evidence but ignores what it says would pass 3 by reading its own query.
5. **Segments are isolated.** Document 0's output must not move when only document 1's evidence
   changes. This is the assertion that catches a mis-paired ``cu_seqlens_k``, which is otherwise
   silent: flash pairs the two sides by position and cheerfully points a document at the wrong
   evidence rather than raising.
6. **Per chunk positions restart**, so two adjacent retrieved chunks do not read as one passage.

The selector -- the IR table reading external chunk vectors under its own softmax -- is checked on
the same three axes, while the reader's output projection is still zero. That ordering is what makes
the selector assertions attributable: anything that moves with the reader switched off moved because
of the selector.

7. **Attaching chunk keys moves the output on its own**, with the reader still neutral.
8. **The selector's segments are isolated too.** It gets no help from flash's positional pairing --
   it scores a dense [tokens, chunks] matrix -- so its mask is a second, independent chance to point
   a document at the wrong evidence.
9. **The external mass is a real fraction and responds to the source scale.** It is the quantity the
   groundedness gate is read off, so "it exists, it is in [0, 1], and the learned scale moves it" is
   the minimum that makes it a signal rather than a constant.

GPU required (flash varlen); assertion 6 is pure tensor arithmetic and runs anywhere.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from modules.model.evidence import chunk_position_ids
from utils import model_params_for_state_dict

BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
    lm_head_factor=4,
)


def build(evidence_port, seed=0):
    torch.manual_seed(seed)
    model = TinyMoETransformer(**dict(P, evidence_port=evidence_port)).to("cuda").to(BF16).eval()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    return model


def test_chunk_positions():
    # two chunks of 3 and 2 tokens, then a 1-token chunk: positions must restart at each boundary
    chunk_ids = torch.tensor([[0, 0, 0, 1, 1, 2]])
    expected = torch.tensor([[0, 1, 2, 0, 1, 0]])
    got = chunk_position_ids(chunk_ids)
    assert torch.equal(got, expected), f"per chunk positions wrong: {got.tolist()}"

    # row padding carries a negative id and arrives as ONE long run. Numbering it like a chunk walks
    # its positions past the rotary cache, and that gather is unchecked -- the failure is a device
    # side assert that wedges the process with the GPU idle instead of raising. Padding is only read
    # by the trailing pad query segment, so 0 is correct as well as in range.
    padded = torch.tensor([[0, 0, 1, -1, -1, -1, -1]])
    got = chunk_position_ids(padded)
    assert torch.equal(got, torch.tensor([[0, 1, 0, 0, 0, 0, 0]])), (
        f"evidence padding is being numbered as a chunk: {got.tolist()}"
    )
    print("6. per chunk positions restart, and padding is not a chunk  PASS")


def main():
    test_chunk_positions()

    plain = build(False)
    ported = build(True)
    # same weights on both sides, so the port is the ONLY difference. The port is two halves and
    # both are new tensors: the reader over evidence tokens, and the selector's adapters plus its
    # source scale inside the IR module.
    port_only = ("shared_evidence", "key_adapter", "value_adapter", "log_memory_scale")
    missing = ported.load_state_dict(plain.state_dict(), strict=False).missing_keys
    assert all(any(p in k for p in port_only) for k in missing), (
        f"unexpected extra tensors: {[k for k in missing if not any(p in k for p in port_only)]}"
    )

    B, S = 2, 24
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device="cuda")

    with torch.inference_mode():
        base = plain(input_ids, skip_mtp=True)
        no_corpus = ported(input_ids, skip_mtp=True)
    assert torch.equal(base, no_corpus), (
        f"the port is not neutral without evidence: max |delta| = "
        f"{(base - no_corpus).abs().max().item()}"
    )
    print("1. no evidence attached is bit-identical                 PASS")

    # one evidence set per query segment. input_ids is unpacked, so each ROW is one segment and the
    # evidence needs exactly two segments in the same order
    S_ev = 12
    ev_ids = torch.randint(1, P["vocab_size"], (B, S_ev), device="cuda")
    # chunk ids only have to differ between adjacent chunks (they drive the position restart);
    # chunk_segments below is what says which document owns a chunk
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6, [2] * 6 + [3] * 6], device="cuda")
    # input_ids is unpacked, so each ROW is one query segment: two segments, numbered 0 and 1
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device="cuda")
    NUM_SEGMENTS, NUM_CHUNKS = 2, 4
    chunk_segments = torch.tensor([0, 0, 1, 1], device="cuda")

    def attach(keys=None):
        return ported.build_evidence(
            ev_ids, ev_chunk, ev_segment, NUM_SEGMENTS,
            chunk_keys=keys, chunk_segments=None if keys is None else chunk_segments,
        )

    evidence = attach()
    with torch.inference_mode():
        zero_reader = ported(input_ids, skip_mtp=True, evidence=evidence)
    assert torch.equal(base, zero_reader), (
        f"zero-init reader is not neutral with a corpus attached: max |delta| = "
        f"{(base - zero_reader).abs().max().item()}"
    )
    print("2. zero init reader is neutral WITH evidence attached     PASS")

    # ---- the selector, while the reader's output projection is still zero. Everything below moves
    # because the IR table started reading an external store, not because the reader woke up.
    ir = ported.moe.ir_modules[0]
    keys = torch.randn(NUM_CHUNKS, P["ir_dim"], device="cuda", dtype=BF16)
    with_keys = attach(keys)
    # the groundedness signal itself: a genuine fraction of ONE softmax's mass, and live under the
    # learned source scale rather than pinned by the two stores' score spreads
    with torch.inference_mode():
        selected = ported(input_ids, skip_mtp=True, evidence=with_keys)
    mass = ir.last_memory_mass
    assert mass is not None and mass.shape == (B * S,), f"no per token external mass: {mass}"
    assert 0.0 <= float(mass.min()) and float(mass.max()) <= 1.0, (
        f"external mass is not a fraction: [{float(mass.min())}, {float(mass.max())}]"
    )
    at_parity = float(mass.mean())
    assert at_parity > 0.0, "the external half won no mass at all -- it is not in the softmax"

    # This test's table is the 256-entry EXACT path, so at parity two chunks compete against 256
    # entries and win well under a percent -- which is below a bf16 ulp by the time it has been
    # through g_proj and up_proj, so the assertions below would be measuring rounding. The real
    # table reads a 32-entry top-k, where a handful of retrieved chunks compete on even terms. Raise
    # the source scale instead of weakening the assertions: the knob existing for exactly this
    # reason is the thing being tested.
    with torch.no_grad():
        ir.log_memory_scale.fill_(3.0)
    with torch.inference_mode():
        selected = ported(input_ids, skip_mtp=True, evidence=with_keys)
    mass_scaled = ir.last_memory_mass.clone()
    scaled = float(mass_scaled.mean())
    assert scaled != at_parity, "the source scale does not move the split -- the signal is pinned"
    print(f"7. external mass is a live fraction ({at_parity:.4f} -> {scaled:.4f} at scale e^3)  PASS")

    sel_delta = (selected - base).abs().max().item()
    assert sel_delta > 0, "chunk keys changed nothing -- the selector is not wired into the read"
    print(f"8. the selector reads external chunk keys (max |delta| = {sel_delta:.4f})  PASS")

    # Change only row 1's chunks. The selector does not get flash's positional pairing -- it builds
    # its own [tokens, chunks] mask -- so this is a second and independent chance to cross documents.
    #
    # Asserted on the MASS rather than on the logits, deliberately. Unlike the reader, the selector
    # is a ROUTED expert, so a token the router did not send to the IR slot has its read multiplied
    # by a zero gate and shows no logit movement however wrong the chunks it scored were -- the
    # assertion would pass vacuously for exactly the tokens it most needs to check (it did, on this
    # model, where row 1 routes away from the IR expert entirely). The mass is computed for every
    # token regardless of routing, because the non-MLP experts all run unconditionally, so it is
    # both the honest test and the quantity the groundedness gate actually reads.
    row1_keys = keys.clone()
    row1_keys[2:] = torch.randn(2, P["ir_dim"], device="cuda", dtype=BF16)
    row1 = attach(row1_keys)
    with torch.inference_mode():
        ported(input_ids, skip_mtp=True, evidence=row1)
    row1_mass = ir.last_memory_mass
    sel_leak = (row1_mass[:S] - mass_scaled[:S]).abs().max().item()
    sel_moved = (row1_mass[S:] - mass_scaled[S:]).abs().max().item()
    assert sel_leak == 0.0, f"document 0's read moved when only document 1's chunks changed ({sel_leak})"
    assert sel_moved > 0.0, "document 1's read did not move when its own chunks changed"
    print(f"9. selector segments are isolated (leak {sel_leak}, own move {sel_moved:.4f})  PASS")

    with torch.no_grad():
        ir.log_memory_scale.zero_()
    # the mass is cleared, not stale, on a batch that carried no evidence at all
    with torch.inference_mode():
        ported(input_ids, skip_mtp=True)
    assert ir.last_memory_mass is None, "a no-evidence batch kept the previous batch's mass"

    with torch.no_grad():
        torch.nn.init.normal_(ported.moe.shared_evidence.attn.o_proj.weight, std=0.02)
    with torch.inference_mode():
        live = ported(input_ids, skip_mtp=True, evidence=evidence)
    delta = (live - base).abs().max().item()
    assert delta > 0, "a trained reader changed nothing -- it is not wired into the forward"
    print(f"3. a trained reader moves the logits (max |delta| = {delta:.4f})   PASS")

    other_ids = torch.randint(1, P["vocab_size"], (B, S_ev), device="cuda")
    other_evidence = ported.build_evidence(other_ids, ev_chunk, ev_segment, NUM_SEGMENTS)
    with torch.inference_mode():
        other = ported(input_ids, skip_mtp=True, evidence=other_evidence)
    content_delta = (other - live).abs().max().item()
    assert content_delta > 0, "different evidence gave the same output -- the content is ignored"
    print(f"4. the evidence CONTENT matters (max |delta| = {content_delta:.4f})       PASS")

    # change only document 1's evidence; document 0's output must not move at all
    mixed_ids = ev_ids.clone()
    mixed_ids[1] = other_ids[1]
    mixed_evidence = ported.build_evidence(mixed_ids, ev_chunk, ev_segment, NUM_SEGMENTS)
    with torch.inference_mode():
        mixed = ported(input_ids, skip_mtp=True, evidence=mixed_evidence)
    leak = (mixed[0] - live[0]).abs().max().item()
    moved = (mixed[1] - live[1]).abs().max().item()
    assert leak == 0.0, f"document 0 moved when only document 1's evidence changed (leak {leak})"
    assert moved > 0.0, "document 1 did not move when its own evidence changed"
    print(f"5. evidence segments are isolated (leak {leak}, own move {moved:.4f})  PASS")

    # the checkpoint, not the yaml, decides whether a model has the reader
    assert model_params_for_state_dict(ported.state_dict(), P)["evidence_port"] is True
    assert model_params_for_state_dict(plain.state_dict(), P)["evidence_port"] is False
    print("10. evidence_port is inferred from the state dict         PASS")


if __name__ == "__main__":
    main()

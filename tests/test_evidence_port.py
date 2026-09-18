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
    print("6. per chunk positions restart at every chunk            PASS")


def main():
    test_chunk_positions()

    plain = build(False)
    ported = build(True)
    # same weights on both sides, so the reader is the ONLY difference
    missing = ported.load_state_dict(plain.state_dict(), strict=False).missing_keys
    assert all("shared_evidence" in k for k in missing), f"unexpected extra tensors: {missing}"

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
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6] * B, device="cuda")
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device="cuda")

    evidence = ported.build_evidence(ev_ids, ev_chunk, ev_segment)
    with torch.inference_mode():
        zero_reader = ported(input_ids, skip_mtp=True, evidence=evidence)
    assert torch.equal(base, zero_reader), (
        f"zero-init reader is not neutral with a corpus attached: max |delta| = "
        f"{(base - zero_reader).abs().max().item()}"
    )
    print("2. zero init reader is neutral WITH evidence attached     PASS")

    with torch.no_grad():
        torch.nn.init.normal_(ported.moe.shared_evidence.attn.o_proj.weight, std=0.02)
    with torch.inference_mode():
        live = ported(input_ids, skip_mtp=True, evidence=evidence)
    delta = (live - base).abs().max().item()
    assert delta > 0, "a trained reader changed nothing -- it is not wired into the forward"
    print(f"3. a trained reader moves the logits (max |delta| = {delta:.4f})   PASS")

    other_ids = torch.randint(1, P["vocab_size"], (B, S_ev), device="cuda")
    other_evidence = ported.build_evidence(other_ids, ev_chunk, ev_segment)
    with torch.inference_mode():
        other = ported(input_ids, skip_mtp=True, evidence=other_evidence)
    content_delta = (other - live).abs().max().item()
    assert content_delta > 0, "different evidence gave the same output -- the content is ignored"
    print(f"4. the evidence CONTENT matters (max |delta| = {content_delta:.4f})       PASS")

    # change only document 1's evidence; document 0's output must not move at all
    mixed_ids = ev_ids.clone()
    mixed_ids[1] = other_ids[1]
    mixed_evidence = ported.build_evidence(mixed_ids, ev_chunk, ev_segment)
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
    print("7. evidence_port is inferred from the state dict          PASS")


if __name__ == "__main__":
    main()

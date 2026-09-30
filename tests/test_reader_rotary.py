"""The evidence reader can run without rotary, and the checkpoint carries which mode it is in.

The reader rotates its query by the prompt position and its keys by their position inside the
chunk. Those two offsets have no relation to what a fact says, so the alternative mode passes no
position embeddings to the reader at all. The default must stay exactly what it was.

Five assertions:

1. **The mode lives in the state dict**: ``moe.evidence_reader_rotary_off`` exists only when rotary
   is off, and ``model_params_for_state_dict`` reads both modes back.
2. **No evidence attached is bit-identical between modes**, since the reader does not run.
3. **The flag reaches the reader**: with a live output projection and evidence attached, the two
   modes disagree.
4. **With rotary off the reader is handed no position embeddings**, with rotary on it is. This is
   the invariance that can be checked honestly: the trunk feeding the reader is itself rotary
   positioned, so the reader's output cannot be compared across shifted prompts end to end.
5. **Loading**: an on-mode state dict seeds an off-mode model through ``load_model_state``, and a
   strict ``load_state_dict`` of an off-mode state into an on-mode model (and the reverse) raises.

GPU required (flash varlen).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from utils import model_params_for_state_dict, load_model_state

BF16 = torch.bfloat16
KEY = "moe.evidence_reader_rotary_off"

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
    lm_head_factor=4, ir_direct_read=False,
)


def build(rotary, seed=0):
    torch.manual_seed(seed)
    model = TinyMoETransformer(**dict(P, evidence_port=True, evidence_reader_rotary=rotary))
    model = model.to("cuda").to(BF16).eval()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    return model


def main():
    on = build(True)
    off = build(False)

    assert KEY not in on.state_dict(), "rotary on must not add a key"
    assert KEY in off.state_dict(), "rotary off must carry its marker"
    assert model_params_for_state_dict(on.state_dict(), P)["evidence_reader_rotary"] is True
    assert model_params_for_state_dict(off.state_dict(), P)["evidence_reader_rotary"] is False
    rebuilt = TinyMoETransformer(**model_params_for_state_dict(off.state_dict(), dict(P, evidence_port=True)))
    assert KEY in rebuilt.state_dict()
    print("1. the mode is carried by the state dict and inferred back      PASS")

    # same weights on both sides
    load_model_state(off, on.state_dict())
    with torch.no_grad():
        torch.nn.init.normal_(on.moe.shared_evidence.attn.o_proj.weight, std=0.02)
        off.moe.shared_evidence.attn.o_proj.weight.copy_(on.moe.shared_evidence.attn.o_proj.weight)

    B, S, S_ev = 2, 24, 12
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device="cuda")
    ev_ids = torch.randint(1, P["vocab_size"], (B, S_ev), device="cuda")
    ev_chunk = torch.tensor([[0] * 6 + [1] * 6, [2] * 6 + [3] * 6], device="cuda")
    ev_segment = torch.tensor([[0] * S_ev, [1] * S_ev], device="cuda")

    def attach(model):
        return model.build_evidence(ev_ids, ev_chunk, ev_segment, 2)

    with torch.inference_mode():
        a = on(input_ids, skip_mtp=True)
        b = off(input_ids, skip_mtp=True)
    assert torch.equal(a, b), f"modes differ without evidence: {(a - b).abs().max().item()}"
    print("2. no evidence attached is bit-identical between modes         PASS")

    with torch.inference_mode():
        a = on(input_ids, skip_mtp=True, evidence=attach(on))
        b = off(input_ids, skip_mtp=True, evidence=attach(off))
    delta = (a - b).abs().max().item()
    assert delta > 0, "the flag changed nothing with a live reader"
    print(f"3. the flag reaches the reader (max |delta| = {delta:.4f})           PASS")

    seen = {}
    for name, model in (("on", on), ("off", off)):
        def hook(module, args, name=name):
            seen[name] = args[4]
        handle = model.moe.shared_evidence.register_forward_pre_hook(hook)
        with torch.inference_mode():
            model(input_ids, skip_mtp=True, evidence=attach(model))
        handle.remove()
    assert seen["on"] is not None, "rotary on must hand the reader position embeddings"
    assert seen["off"] is None, "rotary off must hand the reader no position embeddings"
    print("4. rotary off passes the reader no position embeddings          PASS")

    try:
        on.load_state_dict(off.state_dict())
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict load of an off state into an on model must raise")
    try:
        off.load_state_dict(on.state_dict())
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict load of an on state into an off model must raise")
    load_model_state(off, on.state_dict())
    print("5. seeding tolerates the marker, resume stays strict            PASS")


if __name__ == "__main__":
    main()

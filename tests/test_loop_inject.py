"""The loop input injection must be exactly neutral at its zero init, and must reach every loop.

Three assertions, in the order the migration depends on them:

1. **Zero-init neutrality is bit-identical**, not close. A migrated checkpoint is supposed to score
   exactly as the one it came from, which is what lets any later movement in a gate be read as the
   injection's doing rather than as migration damage. A tolerance would hide a wiring mistake that
   perturbs the model by "only a little" at step 0.
2. **A nonzero injection changes the output**, so the neutrality above is a fact about the zero and
   not about the tensor being disconnected from the forward -- the failure mode the first
   assertion, alone, would pass through happily.
3. **It reaches loops past the first.** The block's input is what loop 1 already starts from, so an
   injection wired only into loop 1 is a no-op in disguise: run with ``n_loops=1`` and ``n_loops=3``
   and require the deeper run to move by more.

GPU required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from utils import model_params_for_state_dict

BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
    lm_head_factor=4,
)


def build(loop_inject, seed=0):
    torch.manual_seed(seed)
    model = TinyMoETransformer(**dict(P, loop_inject=loop_inject)).to("cuda").to(BF16).eval()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    return model


def main():
    plain = build(False)
    injected = build(True)
    # same weights on both sides: the injection is the ONLY difference, which is the whole claim
    missing = injected.load_state_dict(plain.state_dict(), strict=False).missing_keys
    assert missing == ["moe.inject.weight"], f"unexpected extra tensors: {missing}"
    with torch.no_grad():
        injected.moe.inject.weight.zero_()

    B, S = 2, 24
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device="cuda")

    with torch.inference_mode():
        base = plain(input_ids, skip_mtp=True)
        neutral = injected(input_ids, skip_mtp=True)
    assert torch.equal(base, neutral), (
        f"zero-init injection is not neutral: max |delta| = {(base - neutral).abs().max().item()}"
    )
    print("1. zero init is bit-identical to no injection             PASS")

    with torch.no_grad():
        torch.nn.init.normal_(injected.moe.inject.weight, std=0.02)
    with torch.inference_mode():
        live = injected(input_ids, skip_mtp=True)
    delta_full = (live - base).abs().max().item()
    assert delta_full > 0, "a nonzero injection changed nothing -- it is not wired into the forward"
    print(f"2. nonzero injection moves the logits (max |delta| = {delta_full:.4f})  PASS")

    # loop 1 reads the block's input anyway, so the injection has to show up MORE at depth 3 than
    # at depth 1 or it is only being applied where it cannot matter
    with torch.inference_mode():
        base_1 = plain(input_ids, skip_mtp=True, n_loops=1)
        live_1 = injected(input_ids, skip_mtp=True, n_loops=1)
    delta_1 = (live_1 - base_1).abs().max().item()
    assert delta_full > delta_1, (
        f"injection at depth 3 ({delta_full:.4f}) moves no more than at depth 1 ({delta_1:.4f}) "
        f"-- it is not reaching the later loops"
    )
    print(f"3. reaches later loops (depth 3 {delta_full:.4f} > depth 1 {delta_1:.4f})   PASS")

    # the checkpoint, not the yaml, decides whether a model has the tensor
    assert model_params_for_state_dict(injected.state_dict(), P)["loop_inject"] is True
    assert model_params_for_state_dict(plain.state_dict(), P)["loop_inject"] is False
    print("4. loop_inject is inferred from the state dict            PASS")


if __name__ == "__main__":
    main()

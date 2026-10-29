"""``skip_mtp=True`` must be free: the logits on a fixed prompt are BIT-identical to the same
forward with the head running, and the return simply loses its third element.

Bit-identity, not a tolerance. The MTP head is a pure function of the final loop's normed hidden
state and feeds nothing back into the trunk, so skipping it can only be wrong if it changes control
flow somewhere it shouldn't -- and that failure would show up as a nonzero difference, however
small. A tolerance would hide exactly the bug this checks for.

Also runs the KV-cached decode path both ways, since that is where the saving actually lands (the
head is otherwise paid over the whole prefix on every generated token).

GPU required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TinyMoETransformer
from modules.model.kv_cache import KVCache

BF16 = torch.bfloat16

P = dict(
    vocab_size=512, max_seq_len=128, hidden_size=256, intermediate_size=512,
    head_dim=32, num_layers=2, num_heads=8, num_mlp_experts=8, num_attn_experts=1,
    top_k=2, n_loops=3, num_ir_experts=1, num_ir_entries=256, ir_dim=64,
    dropout=0.0, ple_embeddings_size=32, mtp_num_extra_tokens=2,
    lm_head_factor=4,
)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    model = TinyMoETransformer(**P).to(dev).to(BF16).eval()
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    assert model.has_mtp, "this test is meaningless without an MTP head"

    B, S = 2, 24
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device=dev)

    with torch.inference_mode():
        with_mtp = model(input_ids)
        without = model(input_ids, skip_mtp=True)

        assert isinstance(with_mtp, tuple) and len(with_mtp) == 2, type(with_mtp)
        assert torch.is_tensor(without), "skip_mtp must drop extra_token_outputs from the return"
        assert torch.equal(with_mtp[0], without), (
            (with_mtp[0].float() - without.float()).abs().max().item()
        )
        print(f"logits [{tuple(without.shape)}]: bit-identical with and without the MTP head")

        # return_hidden is the shape every eval script actually asks for
        hidden_with, _ = model(input_ids, return_hidden=True)
        hidden_without = model(input_ids, return_hidden=True, skip_mtp=True)
        assert torch.equal(hidden_with, hidden_without)
        print(f"per-loop hidden states [{tuple(hidden_without.shape)}]: bit-identical")

        # aux loss keeps its slot; only the MTP element leaves
        _, aux_with, _ = model(input_ids, return_aux_loss=True)
        logits_only, aux_without = model(input_ids, return_aux_loss=True, skip_mtp=True)
        assert torch.equal(aux_with, aux_without)
        assert torch.equal(logits_only, without)
        print("return_aux_loss arity: (logits, aux_loss) with skip_mtp, (logits, aux_loss, mtp) without")

        # the cached decode path -- where the head was being paid per generated token
        kv_a, kv_b = KVCache.for_model(model), KVCache.for_model(model)
        for t in range(S):
            step_with = model(input_ids[:, t:t + 1], kv_cache=kv_a)[0]
            step_without = model(input_ids[:, t:t + 1], kv_cache=kv_b, skip_mtp=True)
            assert torch.equal(step_with, step_without), f"step {t}"
        print(f"KV-cached decode over {S} steps: bit-identical at every step")

    print("\nMTP SKIP CHECKS PASSED")


if __name__ == "__main__":
    main()

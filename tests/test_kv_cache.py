"""KV cache correctness: cached incremental decoding must reproduce the same logits (within bf16
tolerance) as a single full-sequence forward with no cache -- across single-token decode steps,
multi-token steps (e.g. accepting several MTP-drafted tokens at once), and an n_loops override.
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


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


def main():
    torch.manual_seed(0)
    dev = "cuda"
    model = TinyMoETransformer(**P).to(dev).to(BF16).eval()
    model.set_checkpointing(False, False)

    B, S = 1, 20
    input_ids = torch.randint(1, P["vocab_size"], (B, S), device=dev)

    with torch.inference_mode():
        full_logits = _logits(model(input_ids))

        # one token at a time
        kv = KVCache.for_model(model)
        step_logits = []
        for t in range(S):
            step_logits.append(_logits(model(input_ids[:, t:t + 1], kv_cache=kv))[:, -1, :])
        step_logits = torch.stack(step_logits, dim=1)

        diff = (full_logits.float() - step_logits.float()).abs().max().item()
        rel = diff / full_logits.float().abs().max().item()
        print(f"full-sequence vs single-token KV-cached decode: max abs diff = {diff:.5f} (rel {rel:.5f})")
        assert kv.length == S
        assert diff < 0.5, diff  # generous: SDPA-vs-flash numerics can flip a marginal top-k routing pick

        # multi-token step: prefill half, then feed the rest in one batched call (simulates
        # accepting several MTP-drafted tokens in one step)
        half = S // 2
        kv2 = KVCache.for_model(model)
        model(input_ids[:, :half], kv_cache=kv2)
        logits2 = _logits(model(input_ids[:, half:], kv_cache=kv2))
        diff2 = (full_logits[:, half:, :].float() - logits2.float()).abs().max().item()
        print(f"full-sequence vs multi-token KV-cached decode (2 steps): max abs diff = {diff2:.5f}")
        assert kv2.length == S
        assert diff2 < 0.5, diff2

        # n_loops override must also work with the cache
        n_override = 2
        full_logits_ov = _logits(model(input_ids, n_loops=n_override))
        kv3 = KVCache.for_model(model, n_loops=n_override)
        logits3 = _logits(model(input_ids, kv_cache=kv3, n_loops=n_override))
        diff3 = (full_logits_ov.float() - logits3.float()).abs().max().item()
        print(f"n_loops override ({n_override}) full vs KV-cached prefill: max abs diff = {diff3:.5f}")
        assert kv3.length == S
        assert diff3 < 0.5, diff3

    print("\nKV CACHE CHECKS PASSED")


if __name__ == "__main__":
    main()

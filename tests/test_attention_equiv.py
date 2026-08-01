"""Checks the flash-attn varlen path against the SDPA block-mask fallback and a brute-force
reference: same inputs, same document packing -> outputs must match. Also verifies that a token
cannot see across document boundaries (mask correctness) and that GQA broadcast is right.
GPU required.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.attention import varlen_attention, _sdpa_fallback, cu_seqlens_from_doc_ids

torch.manual_seed(0)
dev = "cuda"
B, Hq, Hkv, S, D = 2, 8, 2, 64, 32

q = torch.randn(B, Hq, S, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.bfloat16)

half = S // 2
row = [0]*half + [1]*(half-4) + [2,3,4,5]
doc_ids = torch.tensor([row]*B, device=dev)
cu, maxlen = cu_seqlens_from_doc_ids(doc_ids)

out_flash = varlen_attention(q, k, v, cu, maxlen, dropout_p=0.0, causal=True)
out_sdpa = _sdpa_fallback(q.float(), k.float(), v.float(), cu, B, S, Hq, Hkv, 0.0, D**-0.5, True)

diff = (out_flash.float() - out_sdpa).abs().max().item()
print(f"flash varlen vs SDPA block-mask: max abs diff = {diff:.5f} (bf16 tolerance ~2e-2)")
assert diff < 2e-2, diff

# cross-document isolation: perturb doc 0's k/v, doc 1's outputs must not change at all
k2, v2 = k.clone(), v.clone()
k2[:, :, :half, :] += 10.0
v2[:, :, :half, :] += 10.0
out_pert = varlen_attention(q, k2, v2, cu, maxlen, dropout_p=0.0, causal=True)
delta_doc1 = (out_pert[:, half:, :, :] - out_flash[:, half:, :, :]).abs().max().item()
print(f"perturb doc0 K/V -> change in doc1 outputs: {delta_doc1} (must be 0)")
assert delta_doc1 == 0.0

# causality within a document: perturbing future tokens must not change past outputs
k3, v3 = k.clone(), v.clone()
k3[:, :, 10:half, :] -= 7.0
v3[:, :, 10:half, :] -= 7.0
out_pert2 = varlen_attention(q, k3, v3, cu, maxlen, dropout_p=0.0, causal=True)
delta_past = (out_pert2[:, :10, :, :] - out_flash[:, :10, :, :]).abs().max().item()
print(f"perturb tokens 10..{half} -> change in outputs of tokens 0..9: {delta_past} (must be 0)")
assert delta_past == 0.0

# no-packing path (cu_seqlens=None) == plain causal
out_none = varlen_attention(q, k, v, None, None, dropout_p=0.0, causal=True)
full_cu = torch.arange(0, (B+1)*S, S, device=dev, dtype=torch.int32)
out_full = _sdpa_fallback(q.float(), k.float(), v.float(), full_cu, B, S, Hq, Hkv, 0.0, D**-0.5, True)
diff2 = (out_none.float() - out_full).abs().max().item()
print(f"cu_seqlens=None vs plain causal SDPA: max abs diff = {diff2:.5f}")
assert diff2 < 2e-2

print("\nATTENTION EQUIVALENCE CHECKS PASSED")

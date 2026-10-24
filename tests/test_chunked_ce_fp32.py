"""The chunked LM head cross entropy accumulates in fp32 even when the logits are bf16.

A bf16 sum over thousands of tokens sits on a 1/64 nat grid and a bf16 correct count can read
above 1.0, so the unweighted path must sum per token losses and counts in fp32. CPU-only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from torch import nn

from modules.model.mtp import _chunked_linear_ce

torch.manual_seed(0)
T, H, V = 8192, 32, 64
head = nn.Linear(H, V, bias=False).to(torch.bfloat16)
hidden = torch.randn(T, H, dtype=torch.bfloat16)
labels = torch.randint(0, V, (T,))

with torch.no_grad():
    logits = head(hidden)
    per_token = F.cross_entropy(logits, labels, reduction="none").float()
    want = per_token.sum() / T
    old = (F.cross_entropy(logits, labels, reduction="sum") / torch.tensor(T).to(torch.bfloat16)).float()
    assert (old - want).abs().item() > 1e-3, f"bf16 sum should miss the fp32 sum, gap {(old - want).abs().item()}"

    got = _chunked_linear_ce(head, hidden, labels)
    assert got.dtype == torch.float32, got.dtype
    assert abs(got.item() - want.item()) < 1e-4, f"got {got.item()} want {want.item()}"
    print(f"1. fp32 accumulation: bf16 {old.item():.5f}, fp32 {want.item():.5f}, got {got.item():.5f}   PASS")

    # every token correct: top1_acc exactly 1.0, with some ignored labels in the mix
    argmax = logits.argmax(-1)
    perfect = argmax.clone()
    perfect[::7] = -100
    _, m = _chunked_linear_ce(head, hidden, perfect, collect_metrics=True)
    assert m["top1_acc"].dtype == torch.float32 and m["top1_acc"].item() == 1.0, m["top1_acc"].item()
    assert 0.0 < m["p_max"].item() <= 1.0
    print("2. all tokens correct reads top1_acc exactly 1.0                    PASS")

    # weighted path stays fp32
    w = torch.rand(T)
    lw = _chunked_linear_ce(head, hidden, labels, weights=w)
    wantw = (per_token * w).sum() / w.sum()
    assert lw.dtype == torch.float32 and abs(lw.item() - wantw.item()) < 1e-4
    print("3. weighted path is fp32 and matches                                PASS")

# gradient flows and is finite in the lm_head
head.zero_grad()
_chunked_linear_ce(head, hidden, labels).backward()
assert torch.isfinite(head.weight.grad.float()).all() and head.weight.grad.abs().sum() > 0
print("4. backward reaches the head                                        PASS")

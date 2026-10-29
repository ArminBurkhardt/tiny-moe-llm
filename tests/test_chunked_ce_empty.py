"""``_chunked_linear_ce`` gives the same results without branching on its valid-token count.

The count is a device tensor, so ``if n_valid == 0`` was a host sync on every call (five per
step). The count is now clamped at one instead: with no valid label every chunk sums to exactly
zero, so ``0 / 1`` is the zero the old early return gave, and with any valid label the clamp
changes nothing.

1. **No valid label**: the loss is zero, with and without weights, and the metrics are zero, as
   the old early return gave; the result is still in the autograd graph and every gradient is zero.
2. **Some valid labels**: the loss equals the full ``F.cross_entropy`` and the metrics equal the
   token-level ``p_max`` and top-1 accuracy, exactly the old arithmetic.
3. **No rows at all** keeps the old early return, graph-connected through the head.

CPU unless a GPU is visible; ``modules/model/mtp.py`` imports Transformer Engine, so run it in WSL
after env_init.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
import torch.nn.functional as F

from modules.model.mtp import _chunked_linear_ce

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB, HIDDEN = 64, 16


def _head():
    torch.manual_seed(0)
    return nn.Linear(HIDDEN, VOCAB, bias=False).to(DEVICE)


def test_no_valid_label():
    head = _head()
    # a hidden state that requires grad routes through Transformer Engine's checkpoint, which
    # needs CUDA; on CPU the gradient is read on the head alone
    hidden = torch.randn(23, HIDDEN, device=DEVICE, requires_grad=DEVICE == "cuda")
    labels = torch.full((23,), -100, dtype=torch.long, device=DEVICE)
    weights = torch.rand(23, device=DEVICE)

    loss = _chunked_linear_ce(head, hidden, labels, chunk_size=7)
    assert loss.item() == 0.0 and not torch.isnan(loss), loss
    assert loss.requires_grad, "the loss left the autograd graph"
    loss.backward()
    if hidden.requires_grad:
        assert hidden.grad is not None and torch.count_nonzero(hidden.grad) == 0
    assert head.weight.grad is not None and torch.count_nonzero(head.weight.grad) == 0

    weighted = _chunked_linear_ce(head, hidden.detach(), labels, chunk_size=7, weights=weights)
    assert weighted.item() == 0.0, weighted

    loss, metrics = _chunked_linear_ce(head, hidden.detach(), labels, chunk_size=7,
                                       collect_metrics=True)
    assert loss.item() == 0.0, loss
    assert metrics["p_max"].item() == 0.0 and metrics["top1_acc"].item() == 0.0, metrics
    print("1. no valid label: zero loss and metrics, graph kept, zero gradients   PASS")


def test_some_valid_labels():
    head = _head()
    hidden = torch.randn(23, HIDDEN, device=DEVICE)
    labels = torch.randint(0, VOCAB, (23,), device=DEVICE)
    labels[::3] = -100
    valid = labels != -100
    with torch.no_grad():
        loss, metrics = _chunked_linear_ce(head, hidden, labels, chunk_size=7, collect_metrics=True)
        logits = head(hidden)
        ref = F.cross_entropy(logits, labels, ignore_index=-100)
        p = logits.softmax(-1).max(-1).values[valid].mean()
        top1 = (logits.argmax(-1) == labels)[valid].float().mean()
    assert torch.allclose(loss, ref, atol=1e-5), (loss.item(), ref.item())
    assert torch.allclose(metrics["p_max"], p, atol=1e-5), (metrics["p_max"].item(), p.item())
    assert torch.allclose(metrics["top1_acc"], top1, atol=1e-6), (metrics["top1_acc"].item(), top1.item())
    print(f"2. some valid labels: CE {loss.item():.5f} equals F.cross_entropy, metrics token-level   PASS")


def test_no_rows():
    head = _head()
    hidden = torch.randn(0, HIDDEN, device=DEVICE, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)
    loss, metrics = _chunked_linear_ce(head, hidden, labels, collect_metrics=True)
    assert loss.item() == 0.0 and loss.requires_grad, loss
    assert metrics["p_max"].item() == 0.0 and metrics["top1_acc"].item() == 0.0
    loss.backward()
    assert head.weight.grad is not None, "the head left the graph on an empty input"
    print("3. no rows: the old early return, head still in the graph   PASS")


if __name__ == "__main__":
    test_no_valid_label()
    test_some_valid_labels()
    test_no_rows()
    print("all chunked CE checks passed")

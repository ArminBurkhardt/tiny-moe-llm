"""The retriever's contrastive loss on hand built tensors (no model, no CUDA, no tokenizer).

1. one positive per row: the query direction equals ``F.cross_entropy`` on the scaled cosines;
2. a row with no positive is excluded and does not change the loss; with no positive anywhere the
   loss is exactly zero and still carries a graph to both inputs;
3. several positives in a row: the target is their joint mass (logsumexp over the positives);
4. the key direction: a chunk that is gold for one query equals ``F.cross_entropy`` down its
   column; the symmetric loss is the mean of the two directions;
5. the temperature divides the cosine;
6. the gradient points the query toward its gold key and away from the other documents' keys, and
   a chunk from another document is a negative (the loss is higher when it is close to the query).

The function is pure torch, so it is imported through ``modules.model.information_retrieval``
with transformer_engine stubbed when it is not installed (the module imports it at top level).
"""
import os
import sys
import types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

try:
    import transformer_engine.pytorch  # noqa: F401
except Exception:  # pragma: no cover - the CPU box has no TE
    te = types.ModuleType("transformer_engine")
    te_pt = types.ModuleType("transformer_engine.pytorch")

    class _Linear(torch.nn.Linear):
        pass

    te_pt.Linear = _Linear
    te_pt.checkpoint = None
    te.pytorch = te_pt
    sys.modules["transformer_engine"] = te
    sys.modules["transformer_engine.pytorch"] = te_pt

from modules.model.information_retrieval import evidence_contrastive_loss  # noqa: E402


def close(a, b, tol=1e-5):
    assert abs(float(a) - float(b)) <= tol, f"{float(a)} != {float(b)}"


def unit(x):
    return F.normalize(x, p=2, dim=-1)


def main():
    torch.manual_seed(0)
    tau = 0.07

    # 1. one positive per row equals cross entropy over the scaled cosines
    q = unit(torch.randn(3, 8))
    k = unit(torch.randn(5, 8))
    positive = torch.zeros(3, 5, dtype=torch.bool)
    positive[0, 1] = positive[1, 3] = positive[2, 4] = True
    logits = q @ k.t() / tau
    expect = F.cross_entropy(logits, torch.tensor([1, 3, 4]))
    got = evidence_contrastive_loss(q, k, positive, temperature=tau, symmetric=False)
    close(got, expect)
    print("1. one positive per row equals cross entropy                                PASS")

    # 2. a row without a positive is excluded; none anywhere gives a connected zero
    positive2 = positive.clone()
    positive2[2] = False
    got2 = evidence_contrastive_loss(q, k, positive2, temperature=tau, symmetric=False)
    close(got2, F.cross_entropy(logits[:2], torch.tensor([1, 3])))
    q_live = q.clone().requires_grad_(True)
    k_live = k.clone().requires_grad_(True)
    zero = evidence_contrastive_loss(q_live, k_live, torch.zeros(3, 5, dtype=torch.bool), temperature=tau)
    close(zero, 0.0)
    zero.backward()
    assert q_live.grad is not None and k_live.grad is not None
    close(q_live.grad.abs().sum(), 0.0)
    print("2. rows without a positive are excluded; no positives gives a connected zero PASS")

    # 3. several positives: the target is their joint share
    positive3 = positive.clone()
    positive3[0, 2] = True
    row0 = torch.logsumexp(logits[0], 0) - torch.logsumexp(logits[0, [1, 2]], 0)
    rows = torch.stack([row0, -F.log_softmax(logits[1], 0)[3], -F.log_softmax(logits[2], 0)[4]])
    got3 = evidence_contrastive_loss(q, k, positive3, temperature=tau, symmetric=False)
    close(got3, rows.mean())
    print("3. several positives in a row target their joint mass                       PASS")

    # 4. the key direction, and the symmetric mean
    col = torch.stack([
        -F.log_softmax(logits[:, 1], 0)[0], -F.log_softmax(logits[:, 3], 0)[1], -F.log_softmax(logits[:, 4], 0)[2],
    ]).mean()
    got4 = evidence_contrastive_loss(q, k, positive, temperature=tau, symmetric=True)
    close(got4, 0.5 * (expect + col))
    print("4. the key direction equals cross entropy down the column; symmetric averages PASS")

    # 5. the temperature divides the cosine
    got5 = evidence_contrastive_loss(q, k, positive, temperature=1.0, symmetric=False)
    close(got5, F.cross_entropy(q @ k.t(), torch.tensor([1, 3, 4])))
    assert abs(float(got5) - float(got)) > 1e-3
    print("5. the temperature divides the cosine                                       PASS")

    # 6. the gradient pulls the query to its gold key; another document's chunk is a negative
    q6 = unit(torch.randn(1, 8)).requires_grad_(True)
    k6 = unit(torch.randn(4, 8))
    pos6 = torch.zeros(1, 4, dtype=torch.bool)
    pos6[0, 0] = True
    loss6 = evidence_contrastive_loss(q6, k6, pos6, temperature=tau, symmetric=False)
    loss6.backward()
    step = q6.detach() - 0.01 * q6.grad
    before = (q6.detach() @ k6[0]).item()
    after = (unit(step) @ k6[0]).item()
    assert after > before, (before, after)
    near = k6.clone()
    near[2] = unit(q6.detach()[0] + 0.05 * torch.randn(8))
    loss_near = evidence_contrastive_loss(q6.detach(), near, pos6, temperature=tau, symmetric=False)
    assert float(loss_near) > float(loss6), (float(loss_near), float(loss6))
    print("6. the gradient pulls the query to its gold key; a close foreign chunk costs   PASS")
    print("all evidence contrastive checks passed")


if __name__ == "__main__":
    main()

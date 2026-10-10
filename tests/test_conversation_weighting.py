"""The per-conversation loss weighting is numerically what it claims to be (NEXT.md Phase 2, fix #3).

The dataset side is covered GPU-free in ``tests/test_sft_dataset.py``; this is the loss side, which
needs ``modules/model/mtp.py`` and therefore transformer_engine. Four properties, in the order they
can silently break:

  1. **``weights=None`` is bit-identical to before.** Pretraining passes None on every step, so a
     regression here is a regression in the 16B-token objective, not just in SFT.
  2. **Uniform weights == the unweighted mean.** ``sum(w*ce)/sum(w)`` has to reduce to
     ``sum(ce)/n`` when every valid token carries the same weight, or the whole scheme is just a
     scale change on the learning rate.
  3. **Per-conversation weights == mean-over-conversations of mean-CE-within-conversation.** This
     is the property the phase exists for: a 2-token refusal and a 40-token answer must pull
     equally. Checked against a reference computed the slow, obvious way.
  4. **The metrics stay token-level.** ``p_max``/``top1_acc`` are compared across pretraining, SFT
     and eval_calibration.py; weighting them would redefine a reported number without renaming it.

Plus an end-to-end pass through ``compute_mtp_loss`` (per-loop CE + MTP heads), because the label
and weight tensors are shifted by different amounts in each term and an off-by-one there would look
like nothing worse than a slightly odd loss value.

Run: `bash tests/run_tests.sh tests/test_conversation_weighting.py` (needs a GPU).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
import torch.nn.functional as F

from modules.model.mtp import _chunked_linear_ce, compute_mtp_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB, HIDDEN = 64, 16
TOL = 2e-5


def reference_ce(lm_head, hidden, labels):
    """The obvious implementation: full logits, one F.cross_entropy call."""
    return F.cross_entropy(lm_head(hidden), labels, ignore_index=-100)


def make_case(n_conversations=5, seed=0):
    """Hidden states, labels and per-conversation weights for a few uneven conversations.

    Conversation ``i`` has ``i + 1`` supervised tokens preceded by two prompt tokens, so the
    unweighted mean is dominated by the long ones and the weighted mean is not -- which is the
    entire difference the test is here to measure.
    """
    torch.manual_seed(seed)
    labels, weights = [], []
    for i in range(n_conversations):
        n_supervised = i + 1
        labels += [-100, -100] + [(i * 7 + j) % VOCAB for j in range(n_supervised)]
        weights += [0.0, 0.0] + [1.0 / n_supervised] * n_supervised
    hidden = torch.randn(len(labels), HIDDEN, device=DEVICE)
    return (
        hidden,
        torch.tensor(labels, dtype=torch.long, device=DEVICE),
        torch.tensor(weights, dtype=torch.float32, device=DEVICE),
    )


def test_unweighted_path_is_unchanged():
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False).to(DEVICE)
    hidden, labels, _ = make_case()
    with torch.no_grad():
        # a chunk size that does not divide the token count, so the chunk loop actually runs twice
        got = _chunked_linear_ce(lm_head, hidden, labels, chunk_size=7)
        want = reference_ce(lm_head, hidden, labels)
    assert torch.allclose(got, want, atol=TOL), f"{got.item()} != {want.item()}"
    print("  weights=None matches plain cross entropy: OK")


def test_uniform_weights_equal_the_token_mean():
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False).to(DEVICE)
    hidden, labels, _ = make_case()
    valid = (labels != -100).float()
    with torch.no_grad():
        # 0.37 rather than 1.0: the normalization has to come from sum(w), not from an assumption
        # that the weights are probabilities
        got = _chunked_linear_ce(lm_head, hidden, labels, chunk_size=7, weights=valid * 0.37)
        want = reference_ce(lm_head, hidden, labels)
    assert torch.allclose(got, want, atol=TOL), f"{got.item()} != {want.item()}"
    print("  uniform weights reduce to the token mean: OK")


def test_conversation_weights_average_over_conversations():
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False).to(DEVICE)
    n_conversations = 5
    hidden, labels, weights = make_case(n_conversations)

    with torch.no_grad():
        got = _chunked_linear_ce(lm_head, hidden, labels, chunk_size=7, weights=weights)

        # reference: per-conversation mean CE, then the plain mean of those. Conversations are laid
        # out contiguously by make_case, so slicing them back out is unambiguous.
        per_conversation, start = [], 0
        for i in range(n_conversations):
            end = start + 2 + (i + 1)
            per_conversation.append(reference_ce(lm_head, hidden[start:end], labels[start:end]))
            start = end
        want = torch.stack(per_conversation).mean()
        token_mean = reference_ce(lm_head, hidden, labels)

    assert torch.allclose(got, want, atol=TOL), f"{got.item()} != {want.item()}"
    # the two must actually differ here, or the test above would pass vacuously on any corpus
    assert not torch.allclose(got, token_mean, atol=1e-3), (
        "uneven conversation lengths must make the weighted and unweighted means differ"
    )
    print("  per-conversation weights average over conversations: OK")


def test_metrics_stay_token_level():
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False).to(DEVICE)
    hidden, labels, weights = make_case()
    with torch.no_grad():
        _, plain = _chunked_linear_ce(lm_head, hidden, labels, chunk_size=7, collect_metrics=True)
        _, weighted = _chunked_linear_ce(
            lm_head, hidden, labels, chunk_size=7, collect_metrics=True, weights=weights,
        )
    for key in ("p_max", "top1_acc"):
        assert torch.allclose(plain[key], weighted[key], atol=TOL), (
            f"{key} must not depend on the loss weighting ({plain[key]} vs {weighted[key]})"
        )
    print("  p_max / top1_acc stay unweighted: OK")


def test_gradients_flow_through_the_weighted_path():
    """The weighted branch runs under transformer_engine's checkpoint, the unweighted one too.

    Worth its own case: the weights are an extra tensor argument to the checkpointed function, and a
    recompute that dropped or reordered them would still produce a finite loss -- just the wrong
    gradient. Comparing against the same slow reference catches that.
    """
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False).to(DEVICE)
    hidden, labels, weights = make_case()
    hidden = hidden.requires_grad_(True)

    loss = _chunked_linear_ce(lm_head, hidden, labels, chunk_size=7, weights=weights)
    loss.backward()
    got = hidden.grad.clone()

    hidden2 = hidden.detach().clone().requires_grad_(True)
    logits = lm_head(hidden2)
    per_token = F.cross_entropy(logits, labels, ignore_index=-100, reduction="none")
    ((per_token * weights).sum() / weights.sum()).backward()

    assert torch.allclose(got, hidden2.grad, atol=1e-5), "checkpointed weighted CE has a wrong grad"
    print("  gradients survive the checkpointed weighted path: OK")


def test_compute_mtp_loss_shifts_weights_with_labels():
    """End to end: per-loop CE and the MTP heads must each shift the weights like their labels.

    ``loss_weights`` comes in aligned with ``targets``; the main term reads ``[:, 1:]`` and MTP head
    ``i`` reads ``[:, i+2:]``. Passing a weight vector that is uniform on supervised positions makes
    every term reduce to its unweighted self, so any misalignment shows up as a mismatch rather than
    as a plausible-looking number.
    """
    torch.manual_seed(3)
    B, S, n_loops, n_extra = 2, 24, 2, 2
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False).to(DEVICE)
    mtp_lm_head = nn.Linear(HIDDEN // 2, VOCAB, bias=False).to(DEVICE)

    hidden = torch.randn(n_loops, B, S, HIDDEN, device=DEVICE)
    mtp_outputs = torch.randn(B, S, n_extra, HIDDEN // 2, device=DEVICE)
    targets = torch.randint(0, VOCAB, (B, S), device=DEVICE)
    targets[:, :4] = -100  # a prompt region, so the ignore_index paths are exercised
    weights = (targets != -100).float() * 0.5

    kwargs = dict(
        mtp_outputs=mtp_outputs, lm_head=mtp_lm_head, lambda_mtp=0.1, main_lm_head=lm_head,
        loop_ce_weights=[0.3, 1.0], loop_ce_subsample=1.0,
    )
    with torch.no_grad():
        plain, plain_ce = compute_mtp_loss(hidden, targets, **kwargs)
        weighted, weighted_ce = compute_mtp_loss(hidden, targets, loss_weights=weights, **kwargs)

    assert torch.allclose(plain, weighted, atol=TOL), (
        f"uniform weights must reproduce the unweighted total loss: {plain.item()} vs {weighted.item()}"
    )
    assert torch.allclose(plain_ce, weighted_ce, atol=TOL)
    print("  compute_mtp_loss shifts weights with labels: OK")


if __name__ == "__main__":
    print("=== per-conversation loss weighting (NEXT.md Phase 2) ===")
    test_unweighted_path_is_unchanged()
    test_uniform_weights_equal_the_token_mean()
    test_conversation_weights_average_over_conversations()
    test_metrics_stay_token_level()
    test_gradients_flow_through_the_weighted_path()
    test_compute_mtp_loss_shifts_weights_with_labels()
    print("all conversation-weighting tests passed")

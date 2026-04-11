import torch
import torch.nn.functional as F


def compute_mtp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mtp_steps: int = 1,
    mtp_lambda: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute multi-token prediction (MTP) loss from token logits.

    For ``offset == 0`` this matches standard token-level cross-entropy. Higher
    offsets add auxiliary losses where each position predicts further-future
    tokens.
    """
    if mtp_steps < 1:
        raise ValueError("mtp_steps must be >= 1")
    if mtp_lambda <= 0:
        raise ValueError("mtp_lambda must be > 0")

    seq_len = logits.size(1)
    vocab_size = logits.size(-1)
    effective_steps = min(mtp_steps, seq_len)

    weighted_sum = logits.new_tensor(0.0)
    weight_total = 0.0

    for offset in range(effective_steps):
        step_logits = logits[:, : seq_len - offset, :]
        step_labels = labels[:, offset:]

        if not torch.any(step_labels != ignore_index):
            continue

        step_loss = F.cross_entropy(
            step_logits.reshape(-1, vocab_size),
            step_labels.reshape(-1),
            ignore_index=ignore_index,
        )
        weight = mtp_lambda ** offset
        weighted_sum = weighted_sum + step_loss * weight
        weight_total += weight

    if weight_total == 0:
        return logits.new_tensor(0.0)
    return weighted_sum / weight_total

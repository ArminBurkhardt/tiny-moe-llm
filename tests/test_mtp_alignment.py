"""Verifies next-token / multi-token-prediction loss alignment in compute_mtp_loss,
and that the dataset's MTP padding prevents cross-document MTP supervision.
CPU-only (uses the pure-torch paths of compute_mtp_loss).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn

from modules.model.mtp import compute_mtp_loss as _compute_mtp_loss

# compute_mtp_loss returns (loss, loss_ce.detach()); these checks only need the total loss
def compute_mtp_loss(*args, **kwargs):
    return _compute_mtp_loss(*args, **kwargs)[0]

torch.manual_seed(0)
B, S, V = 1, 12, 16
targets = torch.arange(S).remainder(V).view(B, S)  # token at pos p is p % V

def onehot_logits(ids):
    return torch.nn.functional.one_hot(ids, V).float() * 100.0

# --- main loss alignment: logits[p] must predict targets[p+1] ---
logits = onehot_logits(torch.roll(targets, shifts=-1, dims=1))  # logits[p] -> targets[p+1]
loss = compute_mtp_loss(logits, targets)
assert loss.item() < 1e-3, f"aligned main loss should be ~0, got {loss.item()}"
loss_mis = compute_mtp_loss(onehot_logits(targets), targets)  # predict self instead
assert loss_mis.item() > 10, f"misaligned main loss should be huge, got {loss_mis.item()}"
print(f"[ok] main NTP loss: logits[p] is trained against targets[p+1] "
      f"(aligned={loss.item():.4f}, misaligned={loss_mis.item():.1f})")

# --- MTP alignment: head i at pos p must predict targets[p + i + 2] ---
num_extra = 2
H = V  # let hidden == vocab and lm_head == identity so hidden one-hots are logits
class IdHead(nn.Module):
    def forward(self, x): return x

mtp = torch.zeros(B, S, num_extra, H)
for i in range(num_extra):
    shift = i + 2
    mtp[:, :S - shift, i, :] = onehot_logits(targets[:, shift:])
    mtp[:, S - shift:, i, :] = 1.0  # tail rows are sliced off inside the loss
loss_full = compute_mtp_loss(logits, targets, mtp_outputs=mtp, lm_head=IdHead(), lambda_mtp=1.0)
assert loss_full.item() < 1e-3, f"aligned MTP loss should be ~0, got {loss_full.item()}"
mtp_wrong = torch.roll(mtp, shifts=1, dims=1)
loss_wrong = compute_mtp_loss(logits, targets, mtp_outputs=mtp_wrong, lm_head=IdHead(), lambda_mtp=1.0)
assert loss_wrong.item() > 10, f"misaligned MTP loss should be huge, got {loss_wrong.item()}"
print(f"[ok] MTP head i at pos p is trained against targets[p+i+2] "
      f"(aligned={loss_full.item():.4f}, misaligned={loss_wrong.item():.1f})")

# --- pad_mask: positions whose SOURCE token is pad get no MTP supervision ---
pad_mask = torch.zeros(B, S, dtype=torch.bool); pad_mask[:, 3] = True
bad = mtp.clone(); bad[:, 3, :, :] = 1.0  # garbage at masked source position
loss_masked = compute_mtp_loss(logits, targets, mtp_outputs=bad, lm_head=IdHead(),
                               lambda_mtp=1.0, pad_mask=pad_mask)
assert loss_masked.item() < 1e-3, f"pad-masked MTP loss should be ~0, got {loss_masked.item()}"
print("[ok] pad_mask removes MTP supervision at pad source positions")

# --- cross-document MTP leakage with dataset-style labels ---
# Packed row: docA tokens [0..5], then num_mtp_tokens pads, then docB.
# Dataset labels: -100 at pads and at the first token of each doc.
def packed_labels(lenA, num_pad, lenB, S):
    lab = torch.full((S,), -100, dtype=torch.long)
    lab[1:lenA] = 1                      # docA continuation tokens
    b0 = lenA + num_pad
    lab[b0 + 1: b0 + lenB] = 2           # docB continuation tokens
    return lab.view(1, S)

def leaks(num_pad, num_extra_tokens, lenA=6, lenB=5):
    """True if any MTP head of a docA position is supervised on a docB token."""
    S = lenA + num_pad + lenB
    lab = packed_labels(lenA, num_pad, lenB, S)
    for i in range(num_extra_tokens):
        shift = i + 2
        for p in range(lenA):            # p inside docA
            t = p + shift
            if t < S and t >= lenA + num_pad and lab[0, t].item() != -100:
                return True
    return False

assert not leaks(num_pad=2, num_extra_tokens=2), "config (2 pads, 2 MTP heads) should NOT leak"
print("[ok] current config (num_mtp_tokens=2, mtp_num_extra_tokens=2): no cross-doc MTP supervision")
assert leaks(num_pad=2, num_extra_tokens=3), "3 MTP heads with only 2 pads should leak"
print("[!!] coupling: mtp heads=3 with num_mtp_tokens=2 WOULD leak docB tokens into docA's MTP loss"
      " -> num_mtp_tokens must stay >= mtp_num_extra_tokens")

# --- NaN edge case: a fully masked MTP target row must yield 0, not NaN ---
all_pad = torch.ones(B, S, dtype=torch.bool)
nan_loss = compute_mtp_loss(logits, targets, mtp_outputs=mtp, lm_head=IdHead(),
                            lambda_mtp=1.0, pad_mask=all_pad)
assert torch.isfinite(nan_loss), f"all-masked MTP batch must not produce NaN, got {nan_loss}"
print(f"[ok] all-masked MTP batch -> loss={nan_loss.item()} (safe cross entropy, no NaN)")

print("\nMTP ALIGNMENT CHECKS PASSED")

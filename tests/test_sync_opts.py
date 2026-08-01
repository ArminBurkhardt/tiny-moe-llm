"""Verifies the TokenTracker sync-removal optimization preserves behavior:
TokenTracker accumulates non-pad counts on-device and only materializes them on sync(); the synced
total must equal the exact non-pad count, and get_count() must not change between syncs.
CPU is sufficient (the tracker bookkeeping is device-agnostic).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from modules.model.transformer import TokenTracker

# --- TokenTracker device accumulator: exact count, sync-free reads ---
tr = TokenTracker()
tr.pad_token_id = 1
x1 = torch.tensor([[5, 5, 1, 5], [1, 1, 5, 5]])   # 5 non-pad
x2 = torch.tensor([[1, 1, 1, 9]])                  # 1 non-pad
exact = int((x1 != 1).sum()) + int((x2 != 1).sum())
tr.count_tokens(x1); tr.count_tokens(x2)
# before sync, the host-visible count has not advanced (work is pending on the accumulator)
assert tr.get_count() == 0, f"get_count must be sync-free / unchanged before sync, got {tr.get_count()}"
assert tr.sync() == exact, f"synced total must equal exact non-pad count {exact}, got {tr.sync()}"
assert tr.get_count() == exact, "after sync the cached count reflects the total"
# draining must not double count on a second sync with no new tokens
assert tr.sync() == exact, "re-sync without new tokens must be idempotent"
tr.count_tokens(x2)
assert tr.sync() == exact + 1, "subsequent counts accumulate on top of the drained total"
# resume/dry-run path: setting num_tokens replaces the total and clears pending
tr.count_tokens(x1)              # pending, not yet synced
tr.num_tokens = 1234
assert tr.get_count() == 1234 and tr.sync() == 1234, "num_tokens setter resets total and clears pending"

# no-pad mode counts every element with no device work
tr2 = TokenTracker()
tr2.count_tokens(torch.zeros(2, 8))
assert tr2.get_count() == 16 and tr2._device_count is None, "pad-less counting stays on the host"
print("[ok] TokenTracker: exact non-pad count, sync-free reads, idempotent drain, setter resets")

print("\nSYNC-OPTIMIZATION CHECKS PASSED")

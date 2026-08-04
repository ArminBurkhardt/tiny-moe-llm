"""Per phase token targets and the new training config keys. No GPU, no TE."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainingConfig as T

assert T.checkpoint_every_tokens > 0
assert T.keep_local_checkpoints >= 1
assert 0.0 < T.phase1_fraction < 1.0
print(f"[ok] new keys load: every {T.checkpoint_every_tokens:,} tokens, keep {T.keep_local_checkpoints}, phase1 {T.phase1_fraction}")

p1 = T.phase_target_tokens("phase1")
p2 = T.phase_target_tokens("phase2")
assert p1 == int(T.target_tokens * T.phase1_fraction), (p1, T.target_tokens)
assert p2 == T.target_tokens, (p2, T.target_tokens)
assert p1 < p2, "phase 1 must stop before the combined budget so phase 2 has the anneal"
print(f"[ok] phase1 stops at {p1/1e9:.2f}B, phase2 runs to the combined {p2/1e9:.2f}B")

# the LR schedule stays anchored to the COMBINED budget -- phase 2 continues the cosine, it does
# not restart it
tokens_per_step = T.Batch_size * T.Seq_length * T.grad_accumulation_steps
assert T.total_steps == T.target_tokens // tokens_per_step
print(f"[ok] total_steps ({T.total_steps:,}) still derives from the combined target, not the phase target")

# an explicitly empty hf_upload_repo must stay empty. this was a real bug: the call site read
# `TrainingConfig.hf_upload_repo or HF_UPLOAD_REPO`, so `hf_upload_repo: ""` uploaded anyway --
# discovered only when a local smoke run started pushing a 2GB checkpoint to the real repo.
T.hf_upload_repo = ""
assert T.upload_repo("owner/fallback") == "", "an explicit empty repo must disable uploads"
T.hf_upload_repo = None
assert T.upload_repo("owner/fallback") == "owner/fallback", "an absent key falls back to the default"
T.hf_upload_repo = "owner/configured"
assert T.upload_repo("owner/fallback") == "owner/configured", "config.yaml wins over the default"
print("[ok] upload_repo distinguishes 'disabled' from 'not configured'")

try:
    T.phase_target_tokens("phase3")
    raise AssertionError("an unknown phase must raise rather than silently returning a target")
except ValueError:
    pass
print("[ok] an unknown phase name raises")

print("\nPHASE TARGET CHECKS PASSED")

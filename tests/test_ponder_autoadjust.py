"""PonderController: the runtime lambda_ponder auto-adjust. Pure Python, no GPU, no TE."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.runtime.ponder import PonderController


def make(**overrides):
    kwargs = dict(lambda_ponder=0.15, target=0.30, band=0.12, factor=1.20,
                  cooldown_tokens=100, lambda_min=0.01, lambda_max=1.0)
    kwargs.update(overrides)
    return PonderController(**kwargs)


# --- ramp gating ------------------------------------------------------------------------------
c = make()
c.observe(0.01, tokens=0, ramp_complete=False)
c.observe(0.01, tokens=1000, ramp_complete=False)
assert c.lambda_ponder == 0.15, "must not adjust before the ponder warmup+ramp has finished"
print("[ok] no adjustment while the ramp is still in progress")

# --- first observation only seeds the cooldown, never adjusts on its own ----------------------
c = make()
c.observe(0.01, tokens=0, ramp_complete=True)
assert c.lambda_ponder == 0.15
print("[ok] the first post-ramp observation only starts the cooldown window")

# --- cooldown: repeated out-of-band readings inside the window do not adjust ------------------
c = make(cooldown_tokens=1000)
c.observe(0.01, tokens=0, ramp_complete=True)
c.observe(0.01, tokens=500, ramp_complete=True)
assert c.lambda_ponder == 0.15, "must wait out the full cooldown before the first real adjustment"
print("[ok] cooldown suppresses adjustment until it elapses")

# --- p_halt too low -> lambda_ponder increases (more pressure to halt) ------------------------
c = make(cooldown_tokens=100)
c.observe(0.01, tokens=0, ramp_complete=True)     # seed
c.observe(0.01, tokens=200, ramp_complete=True)    # EMA still low, cooldown elapsed -> adjust up
assert c.lambda_ponder > 0.15, c.lambda_ponder
assert abs(c.lambda_ponder - 0.15 * 1.20) < 1e-9, c.lambda_ponder
print(f"[ok] p_halt below the healthy band raises lambda_ponder ({0.15} -> {c.lambda_ponder:.4f})")

# --- p_halt too high -> lambda_ponder decreases (less pressure) -------------------------------
c = make(cooldown_tokens=100)
c.observe(0.9, tokens=0, ramp_complete=True)
c.observe(0.9, tokens=200, ramp_complete=True)
assert c.lambda_ponder < 0.15, c.lambda_ponder
assert abs(c.lambda_ponder - 0.15 / 1.20) < 1e-9, c.lambda_ponder
print(f"[ok] p_halt above the healthy band lowers lambda_ponder ({0.15} -> {c.lambda_ponder:.4f})")

# --- inside the healthy band: no adjustment ----------------------------------------------------
c = make(cooldown_tokens=100)
c.observe(0.30, tokens=0, ramp_complete=True)
c.observe(0.30, tokens=200, ramp_complete=True)
assert c.lambda_ponder == 0.15, "p_halt at the target must never trigger an adjustment"
print("[ok] p_halt inside the healthy band leaves lambda_ponder untouched")

# --- clamping: repeated low readings stop growing past lambda_max ------------------------------
c = make(cooldown_tokens=100, lambda_max=0.20)
tokens = 0
for _ in range(20):
    c.observe(0.0, tokens=tokens, ramp_complete=True)
    tokens += 100
assert c.lambda_ponder == 0.20, c.lambda_ponder
print("[ok] repeated adjustments clamp at lambda_max instead of growing unbounded")

c = make(cooldown_tokens=100, lambda_min=0.10)
tokens = 0
for _ in range(20):
    c.observe(1.0, tokens=tokens, ramp_complete=True)
    tokens += 100
assert c.lambda_ponder == 0.10, c.lambda_ponder
print("[ok] repeated adjustments clamp at lambda_min instead of shrinking unbounded")

# --- disabled controller never adjusts, regardless of p_halt -----------------------------------
c = make(cooldown_tokens=1, enabled=False)
c.observe(0.0, tokens=0, ramp_complete=True)
c.observe(0.0, tokens=100, ramp_complete=True)
assert c.lambda_ponder == 0.15
print("[ok] a disabled controller never adjusts")

# --- EMA smooths a single noisy reading ---------------------------------------------------------
c = make(cooldown_tokens=100, ema_alpha=0.05)
c.observe(0.30, tokens=0, ramp_complete=True)      # settle EMA at the target
for t in range(1, 50):
    c.observe(0.30, tokens=t, ramp_complete=True)
c.observe(0.0, tokens=5000, ramp_complete=True)     # one wild outlier
assert c.lambda_ponder == 0.15, "a single outlier reading must not swing a slow EMA out of band"
print("[ok] a single noisy reading does not trigger an adjustment through the EMA")

# --- state_dict / load_state_dict round-trip ----------------------------------------------------
c = make(cooldown_tokens=100)
c.observe(0.01, tokens=0, ramp_complete=True)
c.observe(0.01, tokens=200, ramp_complete=True)
state = c.state_dict()
assert state["lambda_ponder"] == c.lambda_ponder

c2 = make()  # fresh controller, config-seeded at 0.15
assert c2.lambda_ponder == 0.15
c2.load_state_dict(state)
assert c2.lambda_ponder == c.lambda_ponder
assert c2._ema == c._ema
assert c2._last_adjust_tokens == c._last_adjust_tokens
print("[ok] state_dict/load_state_dict round-trips the adjusted value across a resume")

# load_state_dict(None) is a no-op -- covers a legacy checkpoint with no ponder_state key
c3 = make()
c3.load_state_dict(None)
assert c3.lambda_ponder == 0.15
print("[ok] load_state_dict(None) leaves a fresh controller at its config-seeded value")

print("\nPONDER AUTO-ADJUST CHECKS PASSED")

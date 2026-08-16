"""Runtime auto-adjustment of the ponder loss weight.

TrainingConfig.lambda_ponder (config.yaml) is only a STARTING point, re-read unmoved at every
relaunch. Auto-adjustment has to live outside that: its whole purpose is to move away from the
configured value in response to how p_halt actually behaves, and that move must survive every
preemption restart -- otherwise a relaunch (routine on an interruptible box) would silently reset
it back to the yaml value. So the live value lives in PonderController, and the checkpoint payload
carries it (see save_checkpoint's ponder_state), seeded from the checkpoint on resume or from
TrainingConfig.lambda_ponder on a cold start.

Deliberately GPU-free (no torch import) so it tests like the rest of modules/runtime/.
"""
from utils import logger


class PonderController:
    """Bang-bang controller nudging lambda_ponder to keep p_halt's steady state in a healthy band.

    Direction, from moe.py's ponder term (``loss += lambda_ponder_now * mean(1 - p_halt)``):
    minimizing loss means minimizing ``1 - p_halt``, i.e. pushing p_halt UP. So a higher
    lambda_ponder means a higher p_halt equilibrium, and the controller pushes lambda_ponder up
    when the observed p_halt is below the healthy band, down when it's above. Confirmed
    empirically in the Gate 4 test (see config.yaml's lambda_ponder comment): 0.05 -> p_halt
    collapsed to ~0.01-0.02 (too weak, CE pressure wins), 0.5 -> ~0.73-0.75 (too strong, the loop
    collapses to a near no-op and visibly flattens the loss curve), 0.15 -> a genuine plateau at
    ~0.28-0.30 (the ``target``/``band`` defaults below).
    """

    def __init__(self, lambda_ponder: float, target: float = 0.30, band: float = 0.12,
                 factor: float = 1.20, cooldown_tokens: int = 500_000_000,
                 lambda_min: float = 0.01, lambda_max: float = 1.0, ema_alpha: float = 0.05,
                 enabled: bool = True):
        self.lambda_ponder = float(lambda_ponder)
        self.target = target
        self.band = band
        self.factor = factor
        self.cooldown_tokens = cooldown_tokens
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.ema_alpha = ema_alpha
        self.enabled = enabled
        self._ema = None
        # tokens at which the cooldown window last reset -- None until the first observation, so
        # the cooldown starts counting from when the ramp actually completes, not from token 0
        self._last_adjust_tokens = None

    def observe(self, p_halt_mean: float, tokens: int, ramp_complete: bool) -> None:
        """Feed one log-interval's p_halt_mean; may adjust self.lambda_ponder in place.

        Args:
            p_halt_mean: this interval's mean p_halt, already host-synced by the caller.
            tokens: live (real, non-pad) token count, used for the cooldown.
            ramp_complete: whether the ponder warmup+ramp has finished. Before that, lambda_ponder
                itself is still moving from 0 toward its target, so p_halt is chasing a target
                that hasn't settled -- reacting to it would be adjusting against noise.
        """
        if not self.enabled or not ramp_complete:
            return
        self._ema = (
            p_halt_mean if self._ema is None
            else self.ema_alpha * p_halt_mean + (1.0 - self.ema_alpha) * self._ema
        )
        if self._last_adjust_tokens is None:
            self._last_adjust_tokens = tokens
            return
        if tokens - self._last_adjust_tokens < self.cooldown_tokens:
            return

        low, high = self.target - self.band, self.target + self.band
        if self._ema < low:
            new_lambda, direction = min(self.lambda_ponder * self.factor, self.lambda_max), "too low"
        elif self._ema > high:
            new_lambda, direction = max(self.lambda_ponder / self.factor, self.lambda_min), "too high"
        else:
            return

        # cooldown always resets on a check, even a no-op one at a clamp bound -- otherwise a
        # pinned EMA outside the band would retry (and log) every single log interval forever
        self._last_adjust_tokens = tokens
        if new_lambda == self.lambda_ponder:
            return
        logger.warning(
            f"ponder auto-adjust: p_halt {direction} (EMA {self._ema:.3f}, healthy band "
            f"[{low:.2f}, {high:.2f}]) -- lambda_ponder {self.lambda_ponder:.4g} -> {new_lambda:.4g}"
        )
        self.lambda_ponder = new_lambda

    def state_dict(self) -> dict:
        return {
            "lambda_ponder": self.lambda_ponder,
            "ema": self._ema,
            "last_adjust_tokens": self._last_adjust_tokens,
        }

    def load_state_dict(self, state) -> None:
        """Restore from a checkpoint's ponder_state. A no-op on None (legacy checkpoint / cold
        start), leaving __init__'s config-seeded values in place."""
        if not state:
            return
        self.lambda_ponder = float(state.get("lambda_ponder", self.lambda_ponder))
        self._ema = state.get("ema", None)
        self._last_adjust_tokens = state.get("last_adjust_tokens", None)

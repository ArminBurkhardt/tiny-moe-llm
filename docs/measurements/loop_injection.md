# Loop input injection (arm D)

Every loop currently reads only the residual stream, which by loop 3 is the dense decoder's output
plus two of the block's own updates. The diagnostics said the later loops behave like that is the
problem — `cos(q2, q3) = 0.977`, 60% of a token's expert selections repeated from the previous
loop, `cos(Δ3, Δ2) = 0.685` — and every published looped transformer hands the block's input back
to each iteration in some form. Arm D adds that: a zero-init `768 x 768` `inject`, so every loop's
router and experts read `hidden_states + inject(e)` where `e` is what the decoder handed over. The
residual update is untouched, so the injection reaches a readout only through what the experts
compute from it.

**Gate G2c fails, and the run was stopped at 123M of 208M tokens** once two matched-token readings
agreed and the mechanism itself had stopped growing. Numbers below.

## 1. The comparison

Arm C is the control: same corpus, same settings, same migrated seed, differing in this one tensor.
Both arms checkpoint at the same token counts, so the comparison is free at each one. Read with
`.loop_gain_probe.py` on the standard held-out slice, 80 batches of 2 x 4096.

| | 50M tokens | 100M tokens | G2c wants |
|---|---|---|---|
| loop 3 gain, control | 0.0123 | 0.0122 | |
| loop 3 gain, arm | 0.0130 | 0.0132 | |
| **gain delta** | **+0.0007** | **+0.0009** | **+0.01** |
| `cos(Δ3, Δ2)` delta | +0.0038 | +0.0067 | negative |
| loop 1→2 `cos` delta | +0.0230 | +0.0250 | negative |
| loop 2→3 overlap delta | +0.0068 | +0.0064 | negative |
| loop 1→2 overlap delta | +0.0078 | +0.0100 | negative |
| final-loop CE delta | +0.0222 | +0.0112 | ~0 |

Two things in that table, and the second is the finding.

**The CE penalty was a transient and it closed.** At 50M the arm was uniformly ~0.022 nats behind
its control at every loop, which is the signature of a run lagging rather than of a loop-specific
defect; by 100M it had halved. That was the one reading that could have meant "too early to tell",
which is why the run continued past the first checkpoint.

**Closing it bought nothing.** The gain delta moved from 11% of its bar to 9% of it, while both
repetition measures moved monotonically the *wrong* way. The injection makes consecutive loops
slightly **more** alike, not less — the opposite of the mechanism's purpose, at both readings and
on both transitions.

## 2. The mechanism stopped growing

`|inject|rms`, logged every 10 steps:

| ir tokens | 1.65M | 19.7M | 39.3M | 59.0M | 78.6M | 98.3M | 117.9M |
|---|---|---|---|---|---|---|---|
| `\|inject\|rms` | 2.56e-04 | 2.37e-03 | 3.27e-03 | 3.83e-03 | 4.18e-03 | 4.17e-03 | 4.19e-03 |

Flat from 78M onward. It plateaued at 4.2e-3, which is essentially where `g_proj` plateaued in the
same runs (4.6e-3 at 208M) — two tensors, different jobs, same ceiling, both at `fresh_lr` against
the same trunk. The remaining 85M tokens would have trained a flat tensor toward a number 11x short
of its gate, so they were not spent.

## 3. What it rules out, and what it does not

The redundancy in loops 2 and 3 is **not** caused by those loops lacking access to the block's
input. They were given it, through a projection with 0.59M parameters trained at the from-scratch
rate, and they used it to become marginally more redundant. Geiping-style input injection is the
cheapest published fix for exactly this symptom and it does not transfer to this model.

Two things keep this from being a verdict on the recurrence itself:

- **The loop-conditioned mechanism the model already has does work.** Zeroing `loop_router_bias` at
  eval costs 0.0018 nats and raises consecutive-loop expert overlap by 0.017–0.029 — a 1,120
  parameter zero-init tensor moving the quantity arm D's 0.59M could not. Whatever is wrong is not
  that per-loop conditioning is inert here.
- **The depth is doing something.** Out to 8 loops each transition still moves the hidden state
  (‖Δh‖/‖h‖ ≈ 0.09–0.12) and flips ~5% of top-1 predictions, and the oracle keeps finding tokens
  that only a deeper loop gets right (first-correct still adds 0.3–0.5% per loop at depth 8). The
  loops compute; they also break as much as they fix (`regressed` climbs 0.026 → 0.057), which is
  what keeps net top-1 flat after loop 3.

So the failure is retention and control, not access. That is not something another zero-init tensor
on this trunk is going to fix, and this is the second arm to establish the same shape of result —
the remaining candidates (a coda block trained from scratch, the loop-conditioned evidence query)
are architecture changes that belong in a from-scratch comparison rather than a graft onto a
converged checkpoint.

## 4. Provenance

`scripts/migrate_loop_inject.py` produced the seed from
`checkpoint_phase2_final_phase0_irc.pt` — the same seed arm C started from, differing in exactly
one tensor, verified by state-dict comparison (added `moe.inject.weight` at RMS 0, nothing else
added, dropped or changed). Zero-init neutrality is asserted bit-identical in
`tests/test_loop_inject.py`, along with the injection reaching loops past the first.

Checkpoints kept at 50M, 100M and 122M in `ckpts/ir_d/`. The run stopped through the STOP sentinel,
so the 122M checkpoint is a clean save rather than a killed process.

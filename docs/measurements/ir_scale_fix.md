# The IR scale fix (arm C)

Phase 3 measured the read at 0.0002 nats and left *why* open. Phase 3b answered it with a pathway
probe: the value side was never normalized, so the read arrived ~250x below the query it was issued
from. Arm C is that diagnosis acted on — value rows normalized in the forward and seeded as a
rotated orthonormal set per cluster, the migration's neutrality zero moved off `y_values` onto
`g_proj.weight`, and `ir_dim` widened 256 → 384.

**Gate G2b fails.** The read's per-token content is worth **0.0002 nats**, against a bar of 0.01 and
against the 0.0002 the untrained seed and both earlier arms measured. The scale fix worked on the
stage it targeted and the attenuation relocated one stage downstream.

## 1. The run

208M tokens in 71 minutes at ~50k tokens/sec, 23.56 GB peak, micro batch `4 x 4096` in BF16, seeded
from `checkpoint_phase2_final_phase0_irc.pt`. Identical corpus and settings to arms A and B.

The finetune's own objective improved as before: `ir_val` CE 2.9355 → **2.9105**, monotone after
100M tokens. Ten cluster refreshes, candidate recall@32 between 0.892 and 0.995 — the 0.9 warning
tripped once, at the 181M-token refresh, matching arm B's behaviour rather than arm A's.

## 2. Three arms, one result

`scripts/eval_stage0.py` and `scripts/eval_calibration.py`, same held-out slice and identical flags
as the seed's (`--start-doc-idx 0 --max-batches 40 --batch-size 4`):

| | seed | arm A | arm B | **arm C** |
|---|---|---|---|---|
| per-loop CE | 3.5152 / 3.3922 / **3.3843** | 3.5515 / 3.4234 / 3.4109 | 3.5516 / 3.4235 / 3.4109 | 3.5517 / 3.4236 / **3.4112** |
| `ir_val` CE, final | — | 2.9109 | 2.9101 | **2.9105** |
| entropy `E / ln 32`, loops 1–3 | — | 0.9109 / 0.9862 / 0.9866 | 0.9232 / 0.9839 / 0.9848 | **0.9020 / 0.9821 / 0.9835** |
| max weight, loop 1 | — | 0.1451 | 0.1385 | **0.1582** |
| top-1 accuracy | 0.3963 | 0.3950 | 0.3949 | **0.3948** |
| ECE(`p_max`) | 0.0089 | 0.0341 | 0.0346 | **0.0345** |
| AUROC(`p_max` → correct) | 0.8376 | 0.8358 | 0.8360 | **0.8360** |
| dCE, read zeroed (final loop) | 0.0002 | 0.0001 | 0.0002 | **0.0002** |
| dCE, read set to its mean | — | −0.0001 | 0.0000 | **0.0000** |
| IR routed weight (per loop) | — | 0.0199 / 0.0537 / 0.0413 | 0.0199 / 0.0540 / 0.0405 | **0.0197 / 0.0532 / 0.0405** |
| `cos(q1, q2)` | — | 0.3710 | 0.3176 | **0.3393** |

Three key initializations, two table widths, one of them with the scale bug fixed, and the three
runs agree to three or four decimals on every number the gate reads. Arm C sharpens *slightly*
harder on loop 1 (0.9020 against 0.9109 and 0.9232) and its loops 2 and 3 stay as close to uniform
as ever. Held-out CE is 0.027 nats worse than the seed, the same cost arms A and B paid, consistent
with a finetune on a different mixture rather than with damage.

The ECE regression reproduces to the third decimal (0.0089 on the seed, 0.0341 / 0.0346 / 0.0345
across the arms), which is the clearest evidence in the table that the read is not involved in it:
three tables with different content and different read magnitudes cannot land on the same
calibration shift through the retrieval. The trunk moved, as Phase 3 already concluded.

## 3. The attenuation moved one stage downstream

`.ir_pathway_probe.py` on arm C's final checkpoint, RMS over tokens per loop, against the same
probe's numbers from Phase 3b:

| stage | arms A / B | **arm C** |
|---|---|---|
| `x_norm` (the residual) | 0.992 | 0.992–0.998 |
| `down` (the query) | 0.76–1.35 | 0.76–1.19 |
| `retrieved_y` | 0.0032–0.0049 | **0.011–0.015** |
| `ir_out` (after `g_proj`) | — | **0.0030–0.0046** |
| `information` (the expert's K/V) | 0.003–0.008 | 0.004–0.008 |
| `expert_out` | 0.025–0.104 | 0.025–0.102 |

Normalization did what it was supposed to: the read off the table is ~4x larger, `y_row_norm_mean`
is 0.9983 with no row near zero, and `retrieved_y` now lands where a near-uniform read over 32 unit
rows should. **`g_proj` then puts it back.** At RMS 0.0047 it re-attenuates the read to exactly the
magnitude arms A and B delivered without it, so `information` is unchanged and the model is once
again seeing a retrieval worth ~1e-3 of a unit-RMS residual.

The `g_proj` trajectory is the finding. It left zero immediately (5.8e-6 at step 10, 3.7e-3 at 50M
tokens) and then essentially stopped: 3.8e-3 at 91M, 4.3e-3 at 150M, 4.6e-3 at 208M. Against a pure
random walk at `fresh_lr` (~0.034 per element over 12,722 steps) it moved **7x less**, which is the
same signature `y_values` showed in Phase 3 — steps that cancel rather than accumulate. It is not a
vanishing gradient: `‖grad‖/‖w‖` is 3.2e-4 on `g_proj` against 7.0e-4 on a trunk attention weight,
the same order. Weight decay is not the explanation either, at ~7% of the norm over the run.

## 4. What this rules out

Phase 3b's scale diagnosis was correct as a description and insufficient as a cause. The read was
too small, it is now four times larger at the table's output, and the model closed a different valve
by the same amount. Moving the neutrality zero from the content tensor to the gate tensor changes
which parameter is held at zero, not whether the objective wants the read open.

That leaves the hypothesis the three arms were designed to separate as the one they all support:
**LM cross-entropy on this corpus supplies no consistent gradient toward using a parametric read
whose query is derived from the same hidden state the readout already has.** The table cannot offer
the model information it does not already hold, so nothing pays for opening the gate, and every
tensor on the pathway drifts back toward closed whichever one is nominally free.

Two subsidiary readings, both now better supported than before:

- **The key-init A/B was uninformative, and stays uninformative.** Arm C ran at 4x the read
  magnitude and reproduced arms A and B to four decimals, so the regime, not the init, was setting
  the result.
- **The router is not the mechanism.** IR's selection rate is 5.6% / 11.7% / 8.5% against a 5.7%
  uniform, and its routed weight is unchanged across all three arms. Nothing is routing around the
  read; there is nothing in the read to route around.

## 5. Consequence for the plan

Phase 3's branch stands: the parametric table's size is frozen out of the real run spec, and the
evidence pathway carries the retrieval mechanism alone. The three changes are kept regardless —
normalized value rows, the gate-side zero and the 384-d width are all correct on their own terms and
all of them are what the external path wants — but none of them is load-bearing for a gate any more.

The open question this leaves is whether the failure is specific to a *parametric* store or general
to retrieval in this model. A read over externally supplied evidence has content the trunk provably
does not hold, which is the one condition arm C could not create; that is exactly what the evidence
pathway is for, and it is the next thing to test.

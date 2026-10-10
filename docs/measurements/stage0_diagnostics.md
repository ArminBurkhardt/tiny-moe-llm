# Stage 0 diagnostics (2026-08-20)

Phase 1 of [NEXT.md](../plans/NEXT.md): the no-training diagnostic pass every later design is
conditioned on. `scripts/eval_stage0.py`, both migrated checkpoints, the same held-out slice Gate
P0 used (local `phase1`, doc 0 onward, 40 × 4 × 4096 = 654,128 supervised tokens), run at
`--max-loops 6`. Loop-3 CE and top-1 reproduce the
[Gate P0 table](phase0_migration.md) to the last digit on both checkpoints, so this harness is
reading the same quantity that one did. The per-run report files are untracked (`.gitignore`
swallows `*.json`); regenerate with the command in CLAUDE.md.

Row 1's quantity is **also logged during training**, in the same `E / ln N` units: `pretrain.py` and
`sft.py` print `IR E/lnN: [...]` (one entry per loop) at every log interval, off `moe.ir_tracker`'s
per-loop EMA of the retrieval softmax. So the 99.5% below is a number you can watch move rather than
one that needs an eval pass — which is what Gate G2 ("post-anneal entropy well below `ln 65536`")
actually needs to be observable during the Phase 3 run.

| | `sft_final` | `phase2_final` |
|---|---|---|
| 1. retrieval entropy, loop 3 (max `ln 8192` = 9.011) | 8.9671 (99.51%) | 8.9650 (99.49%) |
| max weight / top-32 mass (uniform: 0.000122 / 0.0039) | 0.000229 / 0.0069 | 0.000245 / 0.0073 |
| read dispersion, loop 1 / loops 2–6 | 0.53 / 0.056–0.065 | 0.20 / 0.023–0.034 |
| 2. **ΔCE with the read zeroed (Gate G1 wants > 0.02)** | **0.0004** | **0.0002** |
| ΔCE with the read replaced by its own batch mean | 0.0000 | −0.0000 |
| IR routed weight / selection rate, best loop (uniform: 0.0286 / 0.0571) | 0.0249 / 0.070 | 0.0368 / 0.081 |
| 3. `cos(q1, q2)` / `cos(q2, q3)` | 0.078 / 0.988 | −0.659 / 0.989 |
| 4. top-1 flip 1→2 / 2→3 | 0.189 / 0.073 | 0.191 / 0.054 |
| `‖Δh‖/‖h‖`, loops 1 / 2 / 3 | 0.80 / 0.32 / 0.14 | 0.87 / 0.36 / 0.10 |
| `cos(Δ2, Δ1)` / `cos(Δ3, Δ2)` | 0.371 / 0.733 | 0.299 / 0.634 |
| 5. earliest-correct-loop histogram, loops 1/2/3 | 0.349 / 0.032 / 0.009 | 0.383 / 0.035 / 0.006 |
| never correct at any depth | 0.594 | 0.562 |
| oracle vs. actual top-1 at loop 3 (headroom) | 0.389 vs 0.363 (0.026) | 0.424 vs 0.396 (0.028) |
| 6. CE at loops 3 / 4 / 5 / 6 | 3.7604 / 3.7594 / 3.7724 / 3.7965 | 3.3843 / 3.3889 / 3.4026 / 3.4231 |

## Readings

**1 + 2 — the table stores nothing, and the read is a constant.** Entropy is 99.5% of `ln 8192` on
every loop of both checkpoints, and the top 32 of 8192 entries carry 0.7% of the mass against a
uniform 0.39%. Zeroing the read costs 0.0004 nats; replacing it with its own batch mean costs
0.0000, so *all* of that already-negligible contribution is content-free. The read-dispersion
column says the same thing in vector form — past loop 1 a token's read deviates from the mean read
by 2–6% of its norm. **Gate G1 fails by ~50x**, which is the branch the plan predicted:
re-initializing `z_keys`/`y_values` in Phase 3 costs nothing, and the router's above-uniform
interest in the IR slot (`phase2_final`: 0.037 mean routed weight vs 0.029 uniform) is now
confirmed to be a preference for a bias term. The post-reshape ablation is the real test.

**3 — query drift is zero after loop 1, so the loop-conditioned IR query (NEXT.md 5c item 1) is
mandatory, not conditional.** `cos(q2, q3)` is 0.988/0.989 and every later pair is ≥ 0.96. Loop
1's query is the only distinct one, and on the pretrained checkpoint it points the other way
entirely (−0.659) before the query snaps to a fixed direction and stays there. Re-executed
retrieval cannot make loop 3 differ from loop 2 without the loop-conditioned query bias.

**4 — loop 3 is not idle, it is redundant.** It still moves the residual stream by 14%/10% of
`‖h‖` and flips 7.3%/5.4% of top-1 predictions, but 73%/63% of that movement is aligned with loop
2's direction, and it buys 0.021/0.008 nats. "Ship `n_loops=2`" is not the reading; "later loops
repeat the previous update because nothing feeds them anything new" is.

**5 — oracle headroom is 2.6/2.8 points of top-1, and each loop's slice of it shrinks.** Read this
as a floor on today's checkpoint, not a depth recommendation. It is measured on plain LM
continuation, where most next tokens are settled by loop 1 in *any* looped LM, and with none of
the mechanisms that are supposed to give a later loop something new to work on (evidence buffer,
loop-conditioned query, novelty pressure) in existence yet. The number that decides depth is Gate
G5's EM-vs-`n_loops` curve with a corpus attached, and this histogram is expected to change under
Phases 4/5 — if it doesn't, that is the finding.

**6 — depth past the trained count degrades gracefully but buys nothing.** Loops 4–6 run without
incident (the sinusoidal loop encoding and `loop_scale`'s clamp hold), and the CE curve is
flat-then-slightly-worse rather than divergent. That flat curve is the baseline the depth
curriculum has to beat, and the reason "minimum depth well above 3" is a training goal rather than
a config change.

## Consequences for the plan

The loop-conditioned IR query is promoted from conditional to required and is a precondition for
any depth story, not a Stage-5 refinement; the IR re-init is confirmed free; `n_loops=3` stays the
shipping depth until something actually feeds later loops new information.

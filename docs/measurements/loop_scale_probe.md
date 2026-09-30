# Loop scale probe (2026-09-30)

R0 of the ladder in [NEXT.md](../plans/NEXT.md): is the last loop weak because its gain is small
(`loop_scale[2]` is 0.113 on `ir_c` and 0.098 on `phase2_final`, the halt gate's fold) or because
its update has nothing to add? `scripts/eval_stage0.py --loop-scale-mult F [--loop-scale-loops 3]`,
eval only, the Stage 0 slice (local `phase1`, doc 0 onward, 40 x 4 x 4096 = 654,128 supervised
tokens), `--max-loops 4`. Both x1 rows reproduce [stage0_diagnostics.md](stage0_diagnostics.md)
(3.4112 on `ir_c`, 3.3842 against 3.3843 on `phase2_final`). One log per row in
`ckpts/r0_probe/` (gitignored).

Pass bar: CE gain from loop 2 to 3 at least 0.02 nats, loops 1 and 2 not worse.

## `ir_c` (`ckpts/ir_c/checkpoint_ir_final.pt`, loop_scale [0.633, 0.318, 0.113])

| multiplier | on | loop 1 | loop 2 | loop 3 | loop 4 | gain 2 to 3 |
|---|---|---|---|---|---|---|
| 1 | - | 3.5517 | 3.4236 | **3.4112** | 3.4115 | +0.0124 |
| 2 | all | 3.7462 | 3.5107 | 3.5051 | 3.5210 | +0.0056 |
| 3.5 | all | 4.0302 | 3.6205 | 3.6218 | 3.6490 | -0.0013 |
| 5.9 | all | 4.2573 | 3.6963 | 3.7030 | 3.7379 | -0.0067 |
| 2 | loop 3 | 3.5517 | 3.4236 | 3.4132 | 3.4374 | +0.0104 |
| 3.5 | loop 3 | 3.5517 | 3.4236 | 3.4384 | 3.5154 | -0.0148 |
| 5.9 | loop 3 | 3.5517 | 3.4236 | 3.5220 | 3.6916 | -0.0984 |

## `phase2_final` (`ckpts/trained/checkpoint_phase2_final_phase0.pt`, loop_scale [0.637, 0.326, 0.098])

| multiplier | on | loop 1 | loop 2 | loop 3 | loop 4 | gain 2 to 3 |
|---|---|---|---|---|---|---|
| 1 | - | 3.5150 | 3.3921 | **3.3842** | 3.3889 | +0.0079 |
| 2 | all | 3.8691 | 3.5087 | 3.5031 | 3.5213 | +0.0056 |
| 3.5 | all | 4.3162 | 3.6512 | 3.6510 | 3.6808 | +0.0002 |
| 5.9 | all | 4.6544 | 3.7528 | 3.7566 | 3.7943 | -0.0038 |
| 2 | loop 3 | 3.5150 | 3.3921 | 3.3915 | 3.4236 | +0.0006 |
| 3.5 | loop 3 | 3.5150 | 3.3921 | 3.4256 | 3.5133 | -0.0335 |
| 5.9 | loop 3 | 3.5150 | 3.3921 | 3.5251 | 3.7108 | -0.1330 |

## Readings

**R0 fails on both checkpoints, monotonically.** No multiplier beats x1 at loop 3, and none brings
the loop 2 to 3 gain near 0.02. Raising the last entry to `1/sqrt(3)` (x5.9) costs 0.11 nats on
`ir_c` and 0.14 on `phase2_final`. Scaling every entry wrecks loop 1 first (+0.19 to +1.14 nats),
so the converged trunk is tuned to the migrated gains everywhere, not only at loop 3.

**What it says.** The last loop's update, as trained, points somewhere a larger step makes worse:
its small gain is the model's own correct estimate of how little that direction is worth, not a
throttle holding back a useful update. A bigger post-hoc gain amplifies error. This does not say a
loop 3 trained with a fresh gain would stay useless; it says the checkpoint has no latent loop-3
signal waiting to be released by the scalar.

**Decision.** `loop_scale` stays out of the fresh LR group in the graft arms, and R3 does not
reset it: it keeps the migrated values at the trunk's rate. A fresh `loop_scale` is a from-scratch
question, which R4 already tests. Loop 3's weakness on the grafted lineage is
a property of what that loop learned, which is one more reason the loop test belongs to the
from-scratch pilot and not to a graft.

# Phase 0 — heads out: the migration record (2026-08-19)

The head-removal phase of [NEXT.md](../plans/NEXT.md). Both learned heads (correctness, halt)
failed for structural reasons, not tuning ones — the analysis is in
[CONCLUSION.md](../CONCLUSION.md) — so Phase 0 was a subtraction: delete `correct_proj`, delete
`halt_proj` and the whole ponder subsystem, and fold the halt gate's measured effect into
`loop_scale` so the checkpoint keeps working.

## The fold

`scripts/migrate_phase0.py` does the fold and the strip. It **measures** the gate per checkpoint
rather than taking a constant, because the planning guess was wrong:

| | plan's guess | measured, `checkpoint_sft_final` | measured, `checkpoint_phase2_final` |
|---|---|---|---|
| mean `(1 - p_halt)` per loop | `[0.22, 0.22, 0.22]` (implied) | `[0.290, 0.134, 0.084]` | `[0.367, 0.179, 0.074]` |
| folded `loop_scale` | `[0.38, 0.40, 0.29]` | `[0.501, 0.242, 0.115]` | `[0.637, 0.326, 0.098]` |

The single logged `p_halt` scalar was a mean over all three loops and hid a strong decreasing
trend, so a flat fold would have left loop 1 ~25% too weak and loop 3 ~2.6x too strong. The gate is
stable across corpora within a checkpoint (~5%) but differs a lot between checkpoints — hence
"measure it, per checkpoint". Its per-token std is small (≈0.03–0.06), which is what makes folding
to a mean legitimate at all.

## Gate P0

Re-run `eval_abstention.py` / `eval_calibration.py` on the folded checkpoint; loss and top-1 must
be within noise of the pre-removal numbers — the phase is meant to change nothing behaviorally.

**Passed on both checkpoints** (local `phase1` slice, 654k tokens, identical settings
before/after):

| | `sft_final` before → after | `phase2_final` before → after |
|---|---|---|
| final-loop CE | 3.7564 → 3.7604 | 3.3720 → 3.3843 |
| top-1 | 0.3644 → 0.3628 | 0.3979 → 0.3963 |
| ECE(`p_max`) | 0.0925 → 0.0887 | 0.0088 → 0.0089 |
| AUROC(`p_max`) | 0.8236 → 0.8230 | 0.8390 → 0.8376 |

`p_correct` lost to `p_max` again on both checkpoints, so the correctness-head revert (Gate 5) is
confirmed a third time.

## Deviations from the plan text, both deliberate

- **Only the convergence criterion of the depth policy is implemented** (`converge_tol` /
  `min_loops` on `TinyMoETransformer.forward`, surfaced in `inference.py` and the Gradio app). The
  "evidence still arriving" criterion reads an append-only evidence buffer that does not exist
  until NEXT.md's Phase 4.
- **The convergence exit and the KV cache are mutually exclusive.** An exited loop appends no K/V
  for that token, so a later full-depth step would attend over a cache with a hole in it. Filling
  those caches cheaply (K/V projections only, skipping the skipped loops' experts) is real plumbing
  through every attention expert and was left out of a subtraction-shaped change.

## Convergence statistics

Measured for picking `converge_tol` (`eval_calibration.py` prints these):

| transition | top-1 agreement | mean `\|Δ log p_top\|` |
|---|---|---|
| loop 1 → 2 | 0.808 | 0.232 |
| loop 2 → 3 | 0.925–0.944 | 0.067–0.091 |

A migrated checkpoint drops its optimizer/scheduler state — it is a finetune **seed**, not a
resume point, and `utils.load_checkpoint` says so by name if you try.

# TEMP.md — handoff to the 5090 machine

Scratch note, not part of the permanent docs — delete once Step 1 is committed and validated.
See [PLAN.md](PLAN.md) for the full step list and [CLAUDE.md](CLAUDE.md) for invariants.

## Status: Step 1 code done, NOT validated yet

Implemented on this machine (RTX 5060, TE unavailable) purely as a code edit — could not run the
acceptance tests here.

**Changed:**
- [modules/model/moe.py](modules/model/moe.py) — `LoopMixtureOfExperts.__init__` gained
  `self.loop_scale = nn.Parameter(torch.full((1,), 0.1))`. `forward_step`'s tail now does a
  residual update (`hidden_states = hidden_states + loop_scale * dropout(post_norm(output))`)
  instead of replacing `hidden_states`. No caller signature changes needed elsewhere.
- [CLAUDE.md](CLAUDE.md) — added a bullet under "Model invariants" documenting the above.

**Not yet done:** committing. Per PLAN.md's rules, commit only after the acceptance test passes.

## Do this on the 5090

```bash
source env_init
git status   # confirm you're looking at the same working tree state as below
```

1. **Local housekeeping first** (both are pre-existing local changes from TE-install attempts on
   the 5060, not from me):
   - `requirements.txt`'s `transformer-engine[pytorch]>=2.15.0` line is currently commented out.
     Uncomment it (the 5090 already has TE installed some other way, but requirements.txt should
     stay the source of truth) — or just leave it commented if you installed TE outside pip on
     that box and want to keep it that way. Your call.
   - There's a stray untracked file literally named `=2.15.0` in the repo root. That's the classic
     bash gotcha: `pip install transformer-engine[pytorch]>=2.15.0` unquoted lets the shell treat
     `>` as a redirect and `=2.15.0` as the output filename, silently swallowing the real install
     command. Safe to `rm '=2.15.0'`.

2. **Run Step 1's acceptance tests:**
   ```bash
   bash tests/run_tests.sh tests/test_attention_equiv.py tests/test_overfit.py
   ```
   - `test_overfit.py` must reach a **lower** loss in the same number of steps than it did before
     this change (i.e. compare against `main`/pre-Step-1 behavior — if you don't have a recent
     baseline number, run the test once on the commit before this edit too). **If it's not lower,
     stop** — per PLAN.md, nothing downstream (Steps 2+) is worth doing until this passes.
   - `test_attention_equiv.py` must still pass unmodified (sanity that the residual change didn't
     break the attention path).

3. **If both pass:** commit on `train-build`:
   ```bash
   git add modules/model/moe.py CLAUDE.md
   git commit -m "feat: loop residual + loop_scale gate (PLAN.md Step 1)"
   ```
   (Leave `requirements.txt` and any `=2.15.0` cleanup as separate commits/decisions — not part of
   Step 1.)

4. **If it fails:** don't touch Step 2. Report back what `test_overfit.py` printed (loss curve,
   whether `loop_scale` moved at all) so we can debug — likely suspects: `loop_scale` not receiving
   gradient, or the residual accidentally applied on top of an already-normalized `output` twice.

## What's next after Step 1 (Step 2, PLAN.md)

**Shared (always-on) experts + `moe_intermediate_size`.** In `LoopMixtureOfExperts`:
- Add `self.shared_mlp` (routed-MLP-sized) and `self.shared_attn` (reuse `SelfAttention`), neither
  in the router pool, seeding the accumulator every `forward_step`.
- Add a new `moe_intermediate_size` config key (default to `intermediate_size` if absent) threaded
  through `TinyMoETransformer.__init__` -> `LoopMixtureOfExperts` -> `ParallelSparseMoELayer` +
  `shared_mlp`. `Gemma4TextModel` keeps plain `intermediate_size`.
- Note left in PLAN.md itself under Step 2: "There should be an actual residual stream here, with
  actual skip connections to keep gradients stable" — worth re-reading Step 1's residual change
  alongside this before starting, they're related.
- Acceptance: param count grows by the expected amount, `test_overfit.py` still passes, router
  output dim unchanged, watch expert-selection plots for routed-weight collapse toward the shared
  path.

Don't start Step 2 until Step 1's commit is in.

## Sidenote: why TE is such a pain

Transformer Engine ships prebuilt against a specific CUDA + cuBLAS + PyTorch ABI combination, and
its pip package doesn't vendor CUDA the way `flash-attn`'s wheel-per-cuda-version scheme sort of
does — it links against whatever's on `LD_LIBRARY_PATH`/`ldconfig` at import time. Swapping CUDA
12.4 -> 13 mid-stream is exactly the failure mode you hit: cuBLAS moved to a different `.so` name/
path under CUDA 13, TE's compiled extension still expects the 12.x layout, import succeeds
partially and then fails resolving symbols. The two things it needs (flash-attn's CUDA toolkit
version and TE's) have to match on the same box, which is why CLAUDE.md's `env_init` hardcodes
CUDA 12.9 paths rather than "whatever's newest" — it's the one combination someone already got
working here.

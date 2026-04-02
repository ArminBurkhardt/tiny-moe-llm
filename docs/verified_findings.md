# Verified Findings: TODOs and Potential Bugs

This file records findings checked against current source files.

## Explicit TODO/FIXME markers

- Searched the repository for `TODO`, `FIXME`, `BUG`, `XXX`, and `HACK` markers.
- No explicit markers were found.

## Source-verified inconsistencies

1. **Method name mismatch in expert solving path**
   - `modules/model/expert.py` defines `ExpertModule.solve_for_batch(...)`
   - `modules/model/moe.py` calls `new_expert.solve_from_batch(...)`
   - If `ExpertModule` is used as the expert template, this mismatch can raise `AttributeError` during the "add expert" training phase.

2. **Unreachable lines in MoE inference return path**
   - In `modules/model/moe.py`, after `return output, probs` in inference branch, there are additional lines (`self.current_step += 1` and `return output`) that are unreachable.

3. **Test expectation mismatch with current code path**
   - `modules/model/test_final_transformer.py` expects `model.moe.current_step` not to change in inference.
   - The current unreachable `self.current_step += 1` line in inference supports that observed behavior, but the line itself is dead code and should be cleaned for clarity.

## Recommended follow-up

- Align expert API naming (`solve_for_batch` vs `solve_from_batch`) across expert implementations.
- Remove unreachable code in `modules/model/moe.py` for maintainability.
- Keep tests aligned with intended inference-step semantics.

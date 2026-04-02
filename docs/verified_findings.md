# Verified Findings: TODOs and Potential Bugs

This file records findings checked against current source files.

## Explicit TODO/FIXME markers

- None as of now

## Source-verified inconsistencies (task list)

- [ ] **Unify expert solving method name**
  - `modules/model/expert.py` defines `ExpertModule.solve_for_batch(...)`.
  - `modules/model/moe.py` calls `new_expert.solve_from_batch(...)` during expert addition.
  - Action: choose one method name (`solve_from_batch` or `solve_for_batch`) and align implementation + call sites.

- [ ] **Remove unreachable inference code in MoE**
  - In `modules/model/moe.py`, lines after `return output, probs` in the inference branch are unreachable (`self.current_step += 1` and `return output`).
  - Action: delete unreachable lines to avoid ambiguous behavior.

- [ ] **Lock intended inference-step behavior with tests**
  - `modules/model/test_final_transformer.py` currently expects `model.moe.current_step == 0` after inference.
  - Action: keep or change this expectation intentionally, then align code and test to the same semantics.

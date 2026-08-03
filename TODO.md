# TODO — outstanding testing

Everything below is *not yet verified* — either because it needs network/GPU/paid resources this
environment doesn't have, or because it's the next item in PLAN.md's sequence. Check items off as
they're actually run; this file tracks testing status, not implementation status.

## `scripts/prepare_data.py` (PLAN.md Step 11) — untested against the real Hub

The current test coverage (`tests/test_prepare_data.py`) is entirely network-free: synthetic
in-memory sources exercise the resumable interleave/write/checkpoint core (`run_phase`,
`truncate_to_state`), but nothing has touched huggingface_hub for real yet.

- [ ] **Live file discovery per source**: confirm `list_repo_files` + each `SourceSpec`'s
      `file_prefix`/`file_suffix` actually resolves a non-empty, sensible file list for all seven
      sources. The prefixes (`data/CC-MAIN-2025-26/`, `global-shard_01_of_10/`, `eng_Latn/`, ``
      (root, `code_edu`), `4plus/`, `20231101.en/`, `SFT/`) were inferred from best-effort lookups
      during Step 11, not confirmed by actually downloading anything (except `code_edu`'s, which
      was confirmed live: 95 `stack-edu-NNNN.json.gz` files at repo root) — the Hub layout may
      have shifted or be wrong for one of the others.
- [ ] **Nemotron-CC-Math gated access**: needs `HF_TOKEN` set *and* the dataset's access request
      accepted at huggingface.co first. Untested: whether the accepted token actually authorizes
      `list_repo_files`/`hf_hub_download`, and whether the failure message on a missing/unaccepted
      token is actually clear when it happens for real (only the code path, not the real 401/403,
      has been exercised). `code_edu` (`common-pile/stackv2_edu_filtered`) replaced the originally
      gated `nvidia/Nemotron-CC-Code-v1` and needs no token at all — confirmed via a live
      `list_repo_files`/`hf_hub_download` call during this change, including the `text` column.
- [ ] **`code_edu`'s `"jsonl.gz"` decompression path** (`read_jsonl_gz`, stdlib `gzip`): only
      exercised against one live shard file so far (schema confirmed: `text`, `score`,
      `int_score`, `metadata.detected_licenses`, etc.) — not yet run through the full
      `prepare_data.py` pipeline end-to-end.
- [ ] **Real column-schema auto-detection** (`pick_text_column`): `nemotron_math`'s candidate list
      (`("text", "content")`) is still a guess — its schema couldn't be verified offline (gated).
      First real run against it should be watched closely for a `pick_text_column` failure.
- [ ] **`dclm`'s `.jsonl.zst` decompression path** (`read_jsonl_zst`) has never run against a real
      shard — only the parquet path has any indirect real-world grounding (via prune_vocab.py's
      local parquet reading). Confirm `zstandard` streaming + line-by-line `json.loads` works
      against an actual DCLM shard and that `text` is really the right field.
- [ ] **`smoltalk2` chat rendering**: confirm the `messages` column actually contains
      role/content dicts as expected, that the `_no_think` file filter matches real filenames
      under `SFT/`, and that rendered text looks reasonable (not e.g. empty after filtering).
- [ ] **End-to-end small-scale smoke run**: `python scripts/prepare_data.py --phase1-tokens 2000000
      --phase2-tokens 500000 --checkpoint-docs 50` (or similar) before ever pointing it at the real
      25.5B/4.5B targets. Confirms the whole pipeline (download -> tokenize -> write -> delete)
      against live data, not just the synthetic core logic.
- [ ] **Real interruption safety**: kill `prepare_data.py` (SIGKILL, not KeyboardInterrupt) mid-run
      against real Hub downloads and confirm it resumes cleanly — the synthetic test proves the
      *algorithm* is gap/repeat-free, not that a real, half-downloaded `hf_hub_download` file or an
      OS-level kill behaves the same way.
- [ ] **Acceptance criteria from PLAN.md Step 11**, once a full real run happens:
  - realized per-source token counts within 2% of target
  - `phase1.idx`/`phase2.idx` monotonic, last entry == `len(bin)` (already asserted in `main()`,
    but only ever exercised against synthetic data so far)
  - peak disk under ~70GB during the run (only a post-hoc size check exists today, not live
    monitoring)
  - `manifest.json`'s `data_prep` key has correct repo ids/revisions/filenames once real data
    populates it

## PLAN.md "Local validation gates (5090) — do not rent until all five pass"

- [x] Gate 1 — `bash tests/run_env_check.sh`
- [x] Gate 2 — `tests/test_attention_equiv.py`
- [x] Gate 3 — `tests/test_overfit.py`
- [x] **Gate 4** — run at smoke scale (config.yaml's "GATE4 TEST" values: 45M tokens, not the full
      500M) against a real (not synthetic) local `prepare_data.py` slice. mean loops 2.70, 45.17M
      tokens, 59702 tok/s, MFU 31.8%, peak mem 25.45GB, 12.60min. Not re-run at the full 500M scale.
- [x] **Gate 5** — `scripts/eval_calibration.py` written and run against the Gate 4 checkpoint.
      ECE(p_correct)=0.013 (passes <0.15) but `p_max` beat `p_correct` on both ECE and abstention
      AUROC — by PLAN.md's literal rule that's "revert Step 4b", but deferred instead: only one real
      cloud run remains, the 45M-token checkpoint is ~0.15-0.2% of the real token target, and
      `correct_proj` is proven gradient-isolated/free, so the head stays wired in through the real
      run and the revert-or-keep call gets re-made against the real final checkpoint. See memory
      `project_step4b_correctness_head_deferred`.

## Repo prep before renting (PLAN.md)

- [x] `tests/` un-ignored (already tracked)
- [x] Write `vast_init` (committed env script for the rented box — no WSL paths, no venv activation;
      installs `requirements.txt`'s non-CUDA deps only, since the NGC image ships
      torch/transformer_engine/flash-attn prebuilt for sm90)
- [x] Make `tests/run_tests.sh` / `tests/run_env_check.sh`'s hardcoded repo root / `env_init` path
      overridable by env var (`TINY_LLM_ROOT`, `TINY_LLM_ENV_INIT`)
- [ ] ~~Commit a real `manifest.json`~~ — not actually a before-renting task: `prepare_data.py` runs
      *on* the rented box (PLAN.md), so its `manifest.json` `data_prep` key can only be produced and
      committed *from* the box, after a real run. Nothing to do locally now.

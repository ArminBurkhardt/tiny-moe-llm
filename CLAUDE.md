
## Conventions

- Comments are lowercase, explanatory, and justify *why* (especially around sync avoidance,
  checkpoint recompute, and accelerate's batch handling). Match that density; don't strip them.
- Google-style docstrings with an `Args:` block on the public modules.
- Config values flow yaml -> `config.py` -> kwargs. Don't read `config.yaml` from a module under
  `modules/`.
- `utils.logger` (yellow-formatted) is the logging channel; scripts use `print` only in the
  inference CLI.

## Git

Current branch `train-build`; PRs target `main`. Commit style is
`feat:` / `docs:` / `chore:` / `merge:`. Note the `.gitignore` swallows `*.json` (so
`data_config.json` is untracked), `*.cmd`, `ckpts/`, `venv/`, `env_init`, and `data/prepared`
(the Step 9/11 `{phase}.bin`/`.idx` corpus). `tests/` is tracked.

## Known rough edges

- `flash-attn` and `transformer-engine` in `requirements.txt` need CUDA builds matched to the GPU;
  a plain `pip install -r requirements.txt` will usually fail on them.
- `huggingface.key` sits in the repo root (gitignored via `*.key`).
- `inference.py` runs the model with no `cu_seqlens` (plain causal) and no KV cache — it re-runs
  the full prefix per token, which is fine for smoke-testing checkpoints and slow for anything else.
- README notes token counts can be inflated by tens of tokens per batch.

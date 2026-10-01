# CLAUDE.md

Operational map: where things live, invariants that bite, how to run things, what is next. Prose
lives in [docs/](docs/): [architecture.md](docs/architecture.md), [moe.md](docs/moe.md),
[training.md](docs/training.md), [configuration.md](docs/configuration.md),
[runbook.md](docs/runbook.md). Every gate is in [docs/measurements/](docs/measurements/). The plan
is [docs/plans/NEXT.md](docs/plans/NEXT.md) (older notes call it `PLAN.md`).

## Now (keep current)

Mirror of the "Now" section of NEXT.md; update both when the next step changes. As of 2026-09-30
the evidence port (Phase 4) is built, its instruments fixed, R0 read, and it has not trained. The
plan was rewritten on 2026-09-30 around the goal "facts in the store through the retrieval
pathway, reasoning in the looped trunk"; the design is `docs/evidence_path_design.html`, the
findings are `docs/review_2026-09-29.md`.

- Corpus `evidence_train`: 963,011 conversations, 140.1M prompt / 525.6M evidence tokens, ratio
  3.75. Held out (`prepare_evidence_data.py --heldout`, SQuAD v2 dev + HotpotQA dev):
  `evidence_dev` 18,966 rows (the objective's held-out copy, ratio 11.2) and `evidence_fixed`
  13,333 questions x gold/mixed/distractors/none, real answer every time (the kill number).
  `evidence_val` is a train-loss slice for QA.
- Seed: `ckpts/repair/checkpoint_repair_final_irrandom_evidence_grounded.pt`. Step-0 readings:
  fixed-target gold gain -0.0020 (noise), pooled chunk AUROC 0.43 / 0.41 / 0.42 by loop (the old
  metric, uniform-selector chance 0.421: the selector is uniform, gold share 0.334), grounded
  AUROC 0.5000 (zero head). Per-token chunk AUROC (chance 0.5) 0.515 / 0.484 / 0.490 by loop,
  mass/chunk AUROC 0.481 / 0.470 / 0.469, content gain -0.0009, per-loop reader gain about
  -0.002 at every loop (smoke2, 2026-09-30). `g_proj` and `direct_gate` both exactly 0: the IR
  value path is dead for this run, the selector trains from the selection loss alone.
  `evidence_fixed` ratio 6.25.
- **R0 failed** ([loop_scale_probe.md](docs/measurements/loop_scale_probe.md)): any `loop_scale`
  multiplier makes loop 3 worse; `loop_scale` stays at the trunk's rate in graft arms.
- **Next, in order:**
  1. Arm A `python scripts/sft.py --evidence -c <seed>`, then arm B with
     `--reader-no-rotary --run-name norope -c <seed>`, each in the background under a Monitor
     watch teed to a gitignored log, 10M tokens. Readings: step-0 baseline plus four at
     2.5 / 5 / 7.5 / 10M, the fourth decides. Automatic kill at the first `[eval fixed]` past 10M
     (save, exit 10) if gold gain or content gain (gold minus distractors) is under 0.1 nats;
     otherwise `touch ckpts/evidence/STOP` (arm A) or `touch ckpts/evidence_norope/STOP` (arm B)
     after the 10M reading. Pass: gold gain at least 1.6 nats (50% of 3.23), distractors gain at
     least 0. A mechanism check of the reader; it decides nothing about where facts live.
  2. R0b, the externalization micro-pilot: about 30M params, 0.3B tokens per arm, arms (a) full
     CE no retrieval, (b) span weights no retrieval, (c) span weights + swaps + anonymization
     with the store; fictional biographies at 1 / 10 / 100 / 1000 exposures; closed-book rank
     among 100 candidates by exposure. Needs the biography generator and the closed-book rank
     scorer from Phase 4b first. If (b) or (c) climb with exposure like (a), the recipe fails
     before any 1B-token spend.
  3. Phase 4b instruments (closed-book rank scorer, A3 / A4 / A7, read-ablation chain eval, R1b,
     the ceiling on `evidence_fixed`), then the pilot (Phase 5).
- Relaunch a killed or stopped arm with the same `-c` and flags (without `-c` the strict load
  fails; `kill_checked` is persisted). `ckpts/instrsmoke.log` is the memory and throughput
  reference (batch 2 x 4096, cap 14336, 26.36 GB peak allocated over 110 micro steps, 6.0k to
  14.8k tok/s, about 10k typical, about 42k real prompt tokens per optimizer step at 64% fill);
  `ckpts/evidence_smoke2.log` (2026-09-30) reproduces it on HEAD with the new eval lines.
  `ckpts/evidence/` does not exist, so the first `--evidence` launch honours `-c`; later ones resume.

## Ending a turn

- A turn that did work closes with three short lines in the reply: what was done; what changed
  architecturally (tensor, loss term, invariant, on-disk format; "nothing structural" is valid);
  what is next.
- Every turn, never skipped, ends with **"Something you should know"**: one to three sentences on
  the one non-obvious thing (a risk, a stale assumption, a cheaper path, a number that does not
  add up).

## Running things

- **Everything runs through WSL, from the repo root, after `source env_init`** (gitignored: CUDA
  12.9, flash-attn, TE and `venv/` exist only in WSL). Without it `modules/model/` does not import
  (TE at module scope); from another cwd `config.py` fails (relative `config.yaml`). Git is the
  exception, see Commits.
  ```powershell
  wsl bash -lc "cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init && python scripts/inference.py --help"
  ```
- `tests/run_tests.sh` wraps this; `TINY_LLM_ROOT` / `TINY_LLM_ENV_INIT` override it on the
  rented box (`TINY_LLM_ENV_INIT=/dev/null`).
- **Every training launch (`sft.py`, `pretrain.py`, `run_training.py`) runs in the background
  under a Monitor watch**, however short. The filter matches failures and progress:
  `Traceback|Error|Killed|OOM|assert|device not ready` plus `Step` / `Tokens/sec`, and for
  `--evidence` also `eval fixed|KILL|kill check`. Tee the watched output to a gitignored `*.log`
  so the eval blocks survive a crash.
- STOP files are per run directory: `ckpts/evidence/STOP`, `ckpts/evidence_<run-name>/STOP`
  (polled every 10 micro steps, save, exit 10). A glob such as `ckpts/evidence*/STOP` creates
  nothing.
- A device-side assert prints nothing: process in `R`, GPU ~1%, progress line goes quiet.
  Reproduce one micro step outside the trainer.
- Kill a run as soon as its own instrumentation says it cannot pass. Zero-init tensors are logged
  per step (`|g_proj|rms`, `|inject|rms`, `|shared_evidence.o_proj|rms`); one flat by the first
  checkpoint is the answer. Say out loud whether the rest is kept as a matched-token control.
- **Fix slow runs immediately.** Without evidence, 4 x 4096 fits the 5090 (21-27 GB); 8 x 4096
  spills into shared memory at 3-4x cost and does not OOM. With evidence the peak is
  **~3.5 GiB + 0.61 MiB per evidence token** (prompt axis negligible), hence 2 x 4096. Evidence
  overflow shows up as `CUDA driver error: device not ready` at an arbitrary op, with
  `dxgkio_make_resident: Ioctl failed: -12` in `dmesg`. That is an OOM, not a driver fault.
- All local finetunes run BF16. Never set `USE_FP8`.

## Commits

- Conventional Commits, one of `feat: fix: docs: style: refactor: perf: test: build: ci: chore:
  revert:` (history also has `merge:`). **One line only**, plain unhyphenated phrasing, no body,
  no `Co-Authored-By`, no generated-with footer.
- **Never commit or push.** Finish by handing over one copy-paste line. The user runs it in
  PowerShell on Windows, because WSL git rewrites line endings on every tracked file it touches:
  ```
  git add path/one path/two; git commit -m "fix: thing"
  ```
- Never name a plan document, phase, gate or step in a commit message or code comment (plans get
  renumbered); write the reason. Docs under `docs/` are exempt.
- Branch `ir-train-build`; PRs target `prototype`.

## Code conventions

- Comments are the exception: write one only where the code cannot explain itself (sync
  avoidance, checkpoint recompute, accelerate's batch handling are the usual cases). Lowercase.
  Do not strip existing ones.
- **No em dashes anywhere**, code or prose, this file and docs included.
- Google-style docstrings with an `Args:` block on public modules.
- Config flows yaml -> `config.py` -> kwargs; nothing under `modules/` reads `config.yaml`.
- `utils.logger` for logging; `print` only in the inference CLI and eval report blocks.
- `modules/*/__init__.py` are empty; imports are fully qualified.
- Research code: no packaging, no pytest, no CI. Tests are plain assert scripts in `tests/`.

## Subagents

- Sonnet only. Every subagent runs caveman full or above (`/caveman:caveman full`, `ultra` for a
  pure lookup); say so in the spawn prompt, the mode is not inherited. Caveman is for the report
  only: code, comments and docs are plain English in the repo's style.
- Subagents share one checkout and cannot coordinate, so they never run git that changes the
  working tree or index (`stash`, `reset`, `checkout`, `restore`, `switch`, `clean`, `add`,
  `commit`, ...). Read-only git is fine; baselines come from `git show <rev>:<path>`. The main
  agent may still use them.
- Each task is self-contained and finishes fast. Bundle work that shares a file into one agent and
  name its owned files; split genuinely independent work across parallel agents in one message.

## What this is

`tiny-moe-llm`, ~383M params (224M active). A dense Gemma4-style decoder feeds **one MoE block
applied `n_loops` times** (LoopLM recurrence): heterogeneous expert pool (self-attn / cross-attn /
IR / MLP) behind one router, always-on shared experts, MTP heads, and an **evidence port**
(retrieved chunks encoded by the same decoder, read every loop by an always-on cross-attention
reader; the IR expert scores the chunks' external keys as a selector). Document-packed flash-attn
varlen; optional FP8/NVFP4 via TE.

History: 16B tokens of pretraining on a rented H100 ([CONCLUSION.md](docs/CONCLUSION.md)), then
local finetunes (SFT, abstention repair, three IR sharpening arms, loop injection) and the Phase 4
build. **The benchmark suite is the quality instrument**, diffed against
[benchmark_snapshot.md](docs/measurements/benchmark_snapshot.md); local-slice CE is a health check.
~2.5k H100-hours planned for shape L, 5k ceiling (estimates until the constructor prints them;
the run spec fixes a token target and derives hours). Pre-Phase-4 review:
[review_2026-09-18.md](docs/review_2026-09-18.md).

**Binding verdicts** (each measured, each recorded):
- The learned IR table stores nothing the trunk lacks (zeroing its read: 0.0002 nats, any init or
  width). Frozen out of the real run spec. A retry must move the ablation, not the entropy.
- In-context gold evidence is +3.23 nats on the answer span; a distractor costs 0.63 nats more
  than nothing. External evidence is the mechanism.
- Every graft onto the converged checkpoint (G1, G2, G2b, G2c) measured 0 for 4. The real run
  pretrains with retrieval from token 0.
- `p_max` carries no answerability signal; a linear trunk probe reads 0.584 everywhere.
  Abstention precision is pinned at ~0.578 by the data lever.
- Looping is a requirement; a weak later loop is a defect to fix, never a reason to cut depth.
  "Loops buy computation, not storage" is an axiom, not a measurement (NEXT.md Decisions, with its
  L1 falsifier).

## Layout

```
config.py / .yaml     ModelConfig / TrainingConfig / SFTConfig (+ RepairConfig, IRConfig, EvidenceConfig)
utils.py              logger, BASE_DIR, dtypes, TOKENIZER_*/HF_UPLOAD_REPO, save/load_checkpoint,
                      model_params_for_state_dict + load_model_state
scripts/
  run_training.py     unattended pretrain supervisor: phase1 -> phase2, relaunches through preemption
  pretrain.py         THE training loop; train_step is shared with sft.py
  sft.py              post-training: default / --repair / --ir / --evidence -> ckpts/{sft,repair,ir,evidence}
  prepare_data.py     pretrain mix (rented box); `--phases ir` builds the IR corpus locally, NEVER omit --manifest-key
  prepare_sft_data.py, prepare_evidence_data.py, archive_corpus.py  (the builders delete shards as they go)
  migrate_*.py        phase0, ir_reshape (--arm random|warm), loop_inject, evidence_port, groundedness_head
  eval_abstention.py  THE acceptance metric (SQuAD v2); --evidence-port scores gold/none/distractors/mixed
  eval_benchmarks.py  fixed 13-task suite, one scoring path for this model and the peers
  eval_calibration.py eval_probe.py eval_stage0.py evidence_ceiling_probe.py
  inference.py gradio_app.py prune_vocab.py fetch_tokenizer.py setup.sh onstart.sh run_sft_after_pretrain.sh
modules/model/        transformer gemma4 moe router experts information_retrieval evidence mtp attention kv_cache
modules/data/         dataset (pretrain) sft_dataset evidence_dataset chat abstention
modules/runtime/      run lifecycle; MUST NOT import torch.nn / TE / modules.model (its tests are GPU-free)
tests/                tracked plain assert scripts
```
Gitignored: `ckpts/`, `data/datasets/`, `data/prepared*`, `data/benchmarks`, `*.json`, `*.log`,
`*.cmd`, `*.key` (`huggingface.key` sits in the root), `venv/`, `env_init`.

## Commands

```bash
python scripts/run_training.py                     # the real run; or pretrain.py --phase phase1
python scripts/prepare_evidence_data.py --target-tokens 150000000 --max-evidence-tokens 4608 --max-source-epochs 4
python scripts/prepare_evidence_data.py --heldout   # evidence_dev + evidence_fixed from SQuAD v2 dev / HotpotQA dev
python scripts/prepare_data.py --phases ir --ir-tokens 210000000 --val-tokens 2000000 --manifest-key ir_prep
python scripts/archive_corpus.py pack --all | list
python scripts/sft.py [--from-hub | --repair | --ir | --evidence] -c CKPT
python scripts/migrate_evidence_port.py -c CKPT; python scripts/migrate_groundedness_head.py -c CKPT_evidence.pt
python scripts/eval_abstention.py -c CKPT --max-examples 2000 --batch-size 16 [--skip-forced] [--example-offset 2000]
python scripts/eval_abstention.py -c CKPT --evidence-port --evidence-condition gold,none,distractors,mixed
python scripts/eval_benchmarks.py -c CKPT --compare docs/measurements/benchmarks/*.json
python scripts/eval_benchmarks.py --peer pythia-410m --validate --json-out docs/measurements/benchmarks/pythia-410m.json
python scripts/eval_calibration.py -c CKPT --start-doc-idx 0 --max-batches 40 --batch-size 4
python scripts/eval_stage0.py -c CKPT --start-doc-idx 0 --max-batches 40 --batch-size 4 --max-loops 6
python scripts/eval_stage0.py -c CKPT --start-doc-idx 0 --max-batches 40 --batch-size 4 --max-loops 4 --loop-scale-mult F [--loop-scale-loops 3]
python scripts/eval_probe.py -c CKPT --json-out docs/measurements/probe_CKPT.json
python scripts/evidence_ceiling_probe.py -c CKPT
python scripts/inference.py -c CKPT -p PROMPT -n 200 [--evidence chunks.json] [--converge-tol T]
bash tests/run_env_check.sh; bash tests/run_tests.sh tests/test_attention_equiv.py tests/test_overfit.py
touch ckpts/training/STOP                           # clean stop (exit 10); kill -USR1 <pid> saves now
```
GPU-free tests: the `modules/runtime/` ones (`test_checkpoint_lifecycle`, `test_hf_sync`,
`test_control`, `test_supervisor`, `test_phase_targets`, `test_hf_token`, `test_checkpoint_atomic`)
plus `test_prepare_data`, `test_sft_dataset`, `test_dataset_packing`, `test_token_tracker`.

## Config

Tables in [configuration.md](docs/configuration.md).
- `ModelConfig.Params` is splatted into `TinyMoETransformer(**...)`.
- `TrainingConfig` holds **every loss weight** (`lambda_mtp`, `aux_loss_weight`,
  `evidence_selection_weight`, `groundedness_weight`, `loop_ce_weights` + `loop_ce_subsample`,
  `loop_count_sampling`). `total_steps = target_tokens // (batch * seq * grad_accum)`.
  `loop_ce_weights` length is asserted against `n_loops` at import.
- `RepairConfig` / `IRConfig` / `EvidenceConfig` subclass `SFTConfig` and read only the keys their
  block names; the rest inherits. Never duplicate a shared number. No loss weights here.
  `hf_upload_repo: ""` (default, all four) = uploads off; `None` = use `utils.HF_UPLOAD_REPO`.
  Never collapse the two.
- `IRConfig`: `fresh_lr`, geometric temperature anneal clamped at 1.0, `cluster_refresh_tokens`,
  `dead_quantile`.
- `EvidenceConfig`: `fresh_lr` (port tensors only), refresh cadence, `loss_weight_floor_tokens`
  (64: a conversation weighs `min(n_supervised/64, 1)`, so a 5-token SQuAD answer counts ~0.08;
  read token, conversation and weighted shares separately), and `max_evidence_tokens`. Also
  `fixed_split` + `fixed_eval_max_batches` (the `[eval fixed]` pass: token-level answer CE per
  condition, gain = CE(none) - CE(cond), content gain = gold minus distractors, per-loop reader
  gain from the same forward, per-loop selector lines with chunk AUROC per token, chance 0.5),
  `kill_tokens` / `kill_min_gain` (one automatic decision at the first fixed eval past
  `kill_tokens`: save and exit 10 if gold gain or content gain is under the bar; re-armed when the
  fixed pass read nothing; `kill_checked` persisted in the checkpoint), `checkpoint_every_tokens`
  10M (a save at the decision), `freeze_evidence_gate` (requires_grad off: the gate is non-causal
  and breaks cached decode once its scale leaves 0).
- **`max_evidence_tokens` pulls two ways.** Under `ratio * seq_length` rows close early on the
  evidence budget and the empty prompt slots are padding the body still pays ~502M FLOP/token for;
  over it, 0.61 MiB per evidence token times the batch overflows the card. The ratio is a corpus
  composition statistic (SQuAD 7.6, HotpotQA 6.3, web 0.45, replay 0), so re-read it for every
  corpus. Fill lags the ratio (greedy packing; QA rows close early), so raising the cap pays
  sublinearly. Read `fill` and `closed by the evidence budget` off the packing line in minute one.

Constraints:
- `moe_intermediate_size` sizes routed and shared MoE experts only (defaults to `intermediate_size`).
- `mtp_num_extra_tokens <= num_mtp_tokens`; both come from one value in `pretrain.py`.
- `vocab_size`, `hidden_size` divisible by `lm_head_factor` (and `hidden_size // 2` by
  `lm_head_factor * 2` with MTP); `vocab_size <= 65536`.
- `num_ir_entries % ir_num_clusters == 0`; capacity `<= ir_dim`. `ir_num_clusters: 0` is the exact
  full-table read.
- **The checkpoint, not the yaml, decides shape and mode.** `model_params_for_state_dict` reads IR
  sizes from `z_keys`, forces `ir_num_clusters=0` without centroids, infers `loop_inject`,
  `evidence_port`, `ir_direct_read`, groundedness head, and reader rotary (off iff the buffer
  `moe.evidence_reader_rotary_off` exists; `sft.py --reader-no-rotary` sets it on a seed) from tensor
  presence. `load_model_state` is strict except `IR_TEMPERATURE_KEYS` and `NEUTRAL_LOOP_KEYS`
  (whose inits are the old behaviour) and `READER_MODE_KEYS` (a marker with no learned value).
  Resume paths (`utils.load_checkpoint`, `sft.load_sft_checkpoint`) are fully strict.
  `evidence_encoder` always comes from the yaml.
- Hardcoded: `NUM_DATA_WORKERS=4`, `LOG_INTERVAL` 20 (pretrain) / 10 (sft), expert heads 16/4,
  `rope_theta`, `CE_CHUNK_SIZE=8192`, `ROUTER_NOISE_SCALE=0.3`, `MAX_CHUNKS_PER_SEGMENT=256`.
- The constructor prints 383.5M total / 224.2M active / ~502M FLOP/token at seq 4096, +118M per
  evidence token. Budget math keyed to it goes stale silently. FLOPs split into body, LM head (per
  loop, 4x under chunk checkpointing) + MTP, attention (`sum(seg^2)` from `cu_seqlens`), and the
  evidence encoder (kept out of `flops_per_token_fwd`).

## Model invariants

Detail in [moe.md](docs/moe.md) and [architecture.md](docs/architecture.md).

**Expert pool.** Order `[SelfAttention x A | CrossAttention x A | IR x I | MLP x M]`,
`first_mlp_index = 2A + I`, no identity expert. MLP indices are remapped into
`ParallelSparseMoELayer`'s local space; non-MLP slots become `(index 0, weight 0)`.
- Non-MLP experts run unconditionally once per `forward_step` (attention sees the whole
  sequence); only MLP experts are sparse (grouped GEMM). Routing weights fold into one
  `[B, S, first_mlp_index]` gate via `scatter_add_` and a mask multiply, **never `mask.sum()` or
  boolean indexing** (a device sync per expert).
- `shared_mlp` + `shared_attn` seed the accumulator every loop, outside router and aux loss.
  `shared_evidence` joins only when the port exists and evidence is attached.
- `h = h + loop_scale[loop] * dropout(post_norm(output))`. `loop_scale` is `[n_loops]`, init
  `1/sqrt(n_loops)`, excluded from weight decay; indices past the end reuse the last entry.
- Routing is loop-conditioned by a zero-init `loop_router_bias` over a **sinusoidal buffer** of
  the absolute loop index (any depth runs without reshaping). The same `loop_enc` feeds
  `ir_module.loop_query_bias` and `moe.evidence_query_bias`.
- `loop_inject` (zero-init 768 x 768, inferred) measured useless as a graft; kept loadable. The
  from-scratch design reuses it for gradient flow through the detached depth prefix.
- **Convergence exit** (`forward(..., converge_tol=, min_loops=)`) reads the last position's
  readout, not `||dh||`. Asserted inference-only and exclusive with `kv_cache`.
- `ParallelSparseMoELayer` runs under `te.autocast(enabled=False)` (NVFP4 needs rows % 16);
  `argsort(stable=True)` for recompute determinism; `m_splits.tolist()` is the one accepted sync.
- Aux loss is on the router softmax, normalized by `loops_run`, one `topk` feeds loss and
  selection. Both trainers pass `token_mask = input_ids != pad` (`None` is bit-identical), so aux
  values before that are not comparable. Selection fraction is the signal, not mean weight.
- Router noise is scaled by 0.3 and annealed to 0 over `noise_anneal_tokens`.
- Trackers guard recompute double counting (`begin_forward(expected_updates)`), sample every 8th
  forward, never sync in `update()`. IR entropy is logged as `IR E/ln32` (1.0 = uniform).

**IR expert** ([information_retrieval.py](modules/model/information_retrieval.py)).
- Query: norm -> `down_proj` 768->384 -> `+ loop_query_bias` -> normalize. Learned table and
  external store (when `chunk_keys` exist) share **one softmax over the union**. Then `g_proj` ->
  `up_proj` -> output stage.
- Two stage read: 256 centroids, open top 8, score members exactly, softmax over top 32 divided by
  `log_temperature` and `temperature_scale`. Clusters are exactly equal size (asserted) so scoring
  is one `bmm`. Dispatch overflow goes to a trash row masked to `-1e4` (finite: uniform, not NaN).
  No `torch.bincount` in the step path.
- Keys and values are unit-normalized in the forward. Normalize the table, then gather.
- `g_proj.weight` zero-init is the neutrality zero; `y_values` gets no gradient until `g_proj`
  moves, hence `|g_proj|rms` per step.
- `reset_values()` seeds rotated orthonormal sets per cluster; call after the partition is known.
  Recycled entries get a random unit direction, never zero.
- `temperature_scale` **is** a persistent buffer. `entry_usage` / `query_reservoir` /
  `reservoir_ptr` are plain fp32 attributes, lazy, training-only.
- `refresh_clusters`: warm-started balanced spherical k-means; recall measured on real queries;
  recycles entries below `dead_quantile` AND below `0.01 x mean(entry_usage)`. Recall under 0.9
  warns: raise `ir_probe_clusters` before blaming the anneal.
- External store adapters are square **orthogonal**, not zero (zero teaches "ignore evidence";
  the reader's `o_proj` carries neutrality). A `[tokens, chunks]` `visible` mask keeps documents on
  their own chunks, asserted on the mass. `last_memory_mass` (detached, per loop, every token) is
  the G3b signal; `last_memory_weights` keeps gradient for the selection loss.
- Output stage `ir_direct_read` (inferred): each token's own read through a zero-init
  `direct_gate`. The old inner attention stays loadable for the A/B.

**Evidence port** ([evidence.py](modules/model/evidence.py), `transformer.build_evidence`).
- **No evidence attached is bit-identical to the model without the port**
  (`test_evidence_port.py`, equality). Nothing may touch a path a `None` evidence batch reaches.
  `o_proj` is zero, so a fresh port with evidence is neutral too.
- Evidence is encoded once per forward by the model's own decoder, chunk-causal with its own
  `cu_seqlens`, positions restarting per chunk, no LM head, cached across loops.
  `evidence_encoder: true | N | false` (false = raw embedding, A/B only). **Padding is not a
  chunk**: negative id, position 0 (otherwise positions overrun the rotary cache: a silent
  device-side assert).
- One evidence segment per query segment, paired by position; a document that retrieved nothing
  gets a zero-length segment, never dropped (else later documents read the wrong evidence).
- The reader reads `step_input + evidence_query_bias` every loop, bidirectional, scaled by
  one-init `evidence_loop_scale[loop]`; evidence states are gated per chunk by
  `1 + evidence_gate_scale * sigmoid(selector mass)` before `k_proj`/`v_proj` (zero-init scale is
  exactly neutral). IR experts run before the shared seed so this loop's selector gates. No KV
  slot; cached and uncached greedy decodes can diverge by reduction order (`--no-kv-cache`).
- Nothing ragged leaves the dataloader: `chunk_keys [B, C, 384]` + `[B, C]` slot map, flattened
  in-thread by `evidence_from_batch`. `chunk_gold` / `condition_ids` ride along when the corpus
  has `.evgold`/`.cond` (warned once otherwise). `chunk_segments` comes from packing, never
  re-derived.
- `evidence_selection_loss`: BCE per visible chunk vs gold, renormalized over the token's visible
  chunks, averaged over every (loop, IR expert), weight 0.1. Supervised at the positions that
  produce a supervised token (`labels[:, 1:] != -100` shifted to `:-1`), the positions
  `answer_start_positions` and the readouts use. **Asserts gradient checkpointing off**
  (it reads stashed tensors a recompute would produce under `no_grad`). Logged `selection:`.
- `GroundednessHead` + `groundedness_loss`: label "gold present AND answerable" from the corpus
  (`.ans`, not the condition; never the model's argmax). Reads `moe.last_reader_output` at
  `pretrain.answer_start_positions` (last prompt token of each supervised span), weight 0.1. **Its
  weight gradient is zero until `o_proj` leaves zero**: an early falling `grounded:` is the bias
  learning the base rate; the held-out AUROC in `[eval]` tells them apart.
- `SQUAD_INSTRUCTION` stays verbatim; evidence mode has its own. `is_fresh_loop_param` names every
  port/loop tensor on `fresh_lr`.

**Removed heads.** `correct_proj` and `halt_proj` + ponder are gone; `migrate_phase0.py` folded
the measured per-loop `mean(1 - p_halt)` into `loop_scale`, so a migrated `loop_scale` far below
`1/sqrt(n_loops)` is correct (`[0.637, 0.326, 0.098]` on `phase2_final`). `load_checkpoint`
returns a 6-tuple; a four-value forward unpack is pre-migration code. **`p_max` is the confidence
signal everywhere**, as `1 / sum(exp(l_j - l_max))`; a replacement must add information.

**Forward plumbing.**
- Gradient checkpointing uses TE's `checkpoint`, never `torch.utils.checkpoint`; both levels off.
- Training forward returns per-loop normed hidden states `[loops_run, B, S, H]`, not logits
  (`return_hidden=True` + `delayed_mtp_loss(True)`); `compute_mtp_loss` applies the head in a
  chunked checkpointed CE. **New call sites must pass `main_lm_head=`** or double the activation
  peak; `loop_ce_weights` is then required. `loss_ce` is the final loop's raw CE.
- `return_aux_loss=True` -> `(x, aux_loss)` plus `extra_token_outputs` with MTP. `skip_mtp=True`
  drops it (bit-identical logits); evals and non-drafting inference pass it. `evidence=` takes an
  `EvidenceBatch`.
- The dataset emits `document_ids [B, S]`; the trainer builds `cu_seqlens` in-thread. **Never put
  `cu_seqlens` in the batch dict** (ragged; accelerate truncates dim 0). `max_seqlen` is `S`.
- `TokenTracker` counts on-device, drains on `sync()`; `get_count()` is sync-free; assign
  `.num_tokens` to restore. No `.item()` in the step path.
- KV cache: a slot per decoder layer, per (loop, non-MLP expert), per (loop, `shared_attn`).
  Default `kv_cache=None`; single sequence only.
- Tokenizer: DeepSeek, pruned 129280 -> 65536 (uint16 corpus), `pad == eos`, BOS = 0,
  `embed_tokens` has no `padding_idx`. Path `utils.TOKENIZER_DIR` (`$TINY_LLM_TOKENIZER`
  overrides), fetched from `utils.TOKENIZER_REPO` by `fetch_tokenizer.py`.

## Data prep

**`prepare_data.py`** builds the pretrain mix on the rented box. **Local `data/prepared/` is an
outdated stand-in**: compare before/after deltas on it, not absolutes against CONCLUSION.md.
- One shard in flight per source; sources interleaved by smooth weighted round robin baked into
  on-disk order; checkpointed every 2000 docs, state advances only on a committed document.
- Only `nvidia/Nemotron-CC-Math-v1` is gated. Code comes from `common-pile/stackv2_edu_filtered`.
- smoltalk2 hashes go into `manifest.json` `smoltalk2_holdout_hashes`; `prepare_sft_data.py` and
  `prepare_evidence_data.py` honour them by **importing** the renderer (`--ignore-holdout`).
- Tokenize with the fast tokenizer's Rust threads, never a `ProcessPoolExecutor`.

**`prepare_evidence_data.py`** (local). Conditions `gold`, `mixed`, `many` (16-32 distractors, QA
only), `distractors` and `none` (abstain), plus >= 20% replay. Sources SQuAD v2, HotpotQA, web text
(fixed 384-token held span, long documents windowed), smoltalk2 replay. The passage leaves the
prompt. `--target-tokens` counts prompt tokens.
- `.ans` is not derivable from `.cond`: a natively unanswerable SQuAD row under `gold` still
  abstains. Replay/lm rows are answerable but label 0 for groundedness without a gold chunk.
- QA sources are finite (~10.7M prompt tokens per pass); `EvidenceSource.repeat` redraws
  conditions and distractors up to `--max-source-epochs`.
- **`--max-evidence-tokens` default 1536 silently deletes `many`** (over-budget rows are dropped,
  never truncated). Pass 4608.
- Read each source's conversation share, token share and `too_long` count from the print.

## Training loop ([pretrain.py](scripts/pretrain.py))

- Order: tokenizer -> Dataset -> `DataLoader(batch_size=None, num_workers=4)` -> model ->
  optimizer/scheduler -> resume on the unwrapped model -> `dry_run` -> `Accelerator.prepare`.
- LR: warmup -> cosine to `0.1 * lr`, re-anchored by tokens on resume.
- Main CE = weighted per-loop CE, non-final loops subsampled at 0.25. Stochastic depth
  (`loop_count_sampling` 0.3) picks depth `1..n_loops-1` and rescales weights; log steps run full
  depth.
- Weight decay only on `ndim >= 2`. **A parameter stepped through an fp32 master is in no
  optimizer group, so `zero_grad()` never clears its `.grad`; `train_step` clears it by hand,
  gated on `accelerator.sync_gradients`.** Without it grads sum all run and the clip norm
  silently throttles every parameter. Every parameter in `sft.py` is shadowed.
- A checkpoint that exists but fails to load is never a fresh start.
- Host syncs are throttled to `LOG_INTERVAL`. `has_mtp`, `lm_head`, `mtp_head`,
  `_token_tracker`, `moe` go through `accelerator.unwrap_model`.
- Stops at the phase's token target; `target_tokens` is the combined budget so phase 2 continues
  the cosine.

## Checkpoints, datasets, lifecycle

- `ckpts/training/checkpoint_{phase}_tok{N}M_loss{L}.pt` rolling + `checkpoint_{phase}_final.pt`.
  "Latest" = newest mtime that loads. `global_offset = min(seen) + NUM_DATA_WORKERS`, resume is
  document-granular. New payload fields need `.get(..., default)`. Writes are atomic. Retention
  deletes only outside `keep_local_checkpoints` AND confirmed uploaded; `*_final.pt` exempt.
  Cadence is in tokens.
- Pretrain `Dataset`: `{phase}.bin` (uint16) + `.idx`, memmapped inside `_batch_iterator`, read
  once in on-disk order (the mix is baked in). Each document is followed by
  `EOS + (num_mtp_tokens - 1)` pads; BOS prepended if absent.
- `SFTDataset`: `.mask`, conversations never split across rows, per-epoch `(seed, epoch)`
  permutation (changing `sft.seed` mid-run is a hard error). `EvidenceDataset` adds the evidence
  stream, packs to `max_length - 1`, caps at `max_evidence_tokens`, logs fill.
- `modules/runtime/` stays GPU-free. Exit codes: `0` complete, `10` user stop, `20` preempted,
  `30` resume verification failed; `run_training.py` restarts on all but 10 and 30 (change both
  sides and the runbook table together). Signal handlers only set flags. Upload failures never
  propagate. `run_training.py` starts at `resume_phase_index`. **Crossing a phase resets
  offset/epoch/step but keeps `token_count`.** `verify_resume` aborts (30) when more than
  `2 * checkpoint_every_tokens` behind `run_state.json`.

## SFT ([sft.py](scripts/sft.py))

- Local, single GPU, BF16, same stop contract (SIGTERM -> 20, STOP -> 10, SIGUSR1 saves).
  Reuses `pretrain.train_step` verbatim. Profiles swap config class, phase label and checkpoint
  dir only; `load_sft_checkpoint` refuses another phase's checkpoint. `-c` seeds drop optimizer
  state and must be migrated.
- Every parameter gets an fp32 master (bf16 ulp is ~3x the AdamW step at 3e-5); masters are
  reseeded on resume.
- The SFT cosine floors at `lr_min_factor: 0.05` of lr (pretrain floors at 0.1).
- `fresh_lr` group: `--ir` = `is_rebuilt_ir_param or is_fresh_loop_param`; `--evidence` =
  `is_fresh_loop_param` alone (the IR table trains at the trunk's rate by choice).
- Per-conversation loss weighting is off for `SFTConfig`, on for repair and evidence.
  `p_max`/`top1_acc` stay unweighted. The global token counter continues.
- Chat control tokens are resolved from the tokenizer and asserted. Only assistant text + EOS is
  supervised; conversations with roles outside system/user/assistant are dropped.
- **`eval_abstention.py` generates**: left-padded batched decode, no KV cache, numbers comparable
  only at fixed `--batch-size`. **Read the generated-answers block first** (token-level
  calibration passed while behaviour failed). `--evidence-port`: G3 is answer-span CE with the real
  answer under every condition; G3b is external-mass AUROC at the last prompt position, per loop.
- Abstention phrasings are a closed set (`abstention.py`): 5 forced by the eval inside 15 used by
  corpora.

## Inference

`inference.py` is the reference; `gradio_app.py` imports `stream_generate`. KV-cached by default
(`--no-kv-cache` for the reference path). `--num-mtp-tokens` drafts with no rejection sampling.
`--converge-tol` turns the cache off. `--evidence FILE` builds one `EvidenceBatch` per session.

## Known rough edges

- `flash-attn` / `transformer-engine` need CUDA builds matched to the GPU.
- `eval_abstention.py` has no KV cache; the reader has no cache slot.
- The append-only evidence buffer and the "evidence still arriving" depth criterion are unbuilt.
- `eval_benchmarks.py` / `eval_calibration.py` / `eval_stage0.py` cannot attach evidence.
- Token counts can be inflated by tens of tokens per batch.
- `evidence_from_batch` flattens with boolean mask indexing in the per-step path: one host sync
  per micro step, accepted for the 10M evidence run.

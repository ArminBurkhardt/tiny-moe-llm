# CLAUDE.md

Operational map: where things live, invariants that bite, how to run things, what is next. Prose
lives in [docs/](docs/): [architecture.md](docs/architecture.md), [moe.md](docs/moe.md),
[training.md](docs/training.md), [configuration.md](docs/configuration.md),
[runbook.md](docs/runbook.md). Every gate is in [docs/measurements/](docs/measurements/). The plan
is [docs/plans/NEXT.md](docs/plans/NEXT.md) (older notes call it `PLAN.md`).

## Now (keep current)

Mirror of the "Now" section of NEXT.md; update both when the next step changes. As of 2026-10-05
R0b arms (a), (b) and (c) have run in full: the key/value reader copies completely in distribution
and arm (c) fails, on a leak present before copying existed. With a copy-first warm-up the R0b
rule holds through tier 100 and tier 1000 still leaks. Decided 2026-10-05: R0b is accepted on the
warm-up arm with the tier 1000 leak known, the warm-up enters the pilot as a requirement, and a
swap rate 0.30 branch (read 2026-10-06) left the tier 1000 leak where it was. Decided 2026-10-06
(the user): the pilot's swap rate stays at 0.15; the placeholder name arm and a chain depth arm
are approved; the learned exit gate arm is dropped on the per-exit read. On 2026-10-08 the
placeholder name arm falsified the dose lever, so the copy-criterion span weight is dropped and the
pilot's lever list is the warm-up alone at swap rate 0.15. The same day the chain depth arm read
inconclusive ([chain_depth_micro.md](docs/measurements/chain_depth_micro.md)): the micro model
never learned 1-hop lookup, so L1 stays unmeasured at micro scale. On 2026-10-09 the pilot spec
was written ([PILOT.md](docs/plans/PILOT.md)): the pilot runs the R0b recipe on today's code at
99.1M parameters (width 512, 6 prelude layers, 8 experts, seq 4096, key/value reader) on a merged
corpus from the existing builders, 1B tokens per arm against a matched full-CE control, with a
1-hop gate before L1; the unbuilt blueprint items and the rungs R4, R4b and R6 are deferred. Its
build list is next; the user decides the launch.
The plan was rewritten on 2026-09-30 around the goal "facts in the store through the retrieval
pathway, reasoning in the looped trunk"; the
design is `docs/evidence_path_design.html`, the findings are `docs/review_2026-09-29.md`, the R0b
record is [r0b_micro_pilot.md](docs/measurements/r0b_micro_pilot.md).

**The design page moves with the architecture and the plan.** Any change to the architecture (a
tensor, a module, a loss term, an invariant) or to the plan (NEXT.md, this section, a decision, a
gate, a measured verdict) is also made in `docs/evidence_path_design.html` in the same turn, and
the page is republished to its artifact, https://claude.ai/artifact/GS2S79cV2ACydtXaaHzemW (the
Artifact tool with that `url`; a new session reads the artifact first). The repo file is the
source and the artifact stays identical to it; the file goes into the same commit line.

- **Arm (a), full CE, 300M tokens, reads the instrument**: closed-book entity `delta` in
  distribution 0.009 / 0.063 / 0.378 / 0.467 by tier 1 / 10 / 100 / 1000 (tier 1000 rank 0),
  held-out template 0.001 / 0.009 / 0.023 / 0.104 (surface-form memorization). Checkpoints
  `ckpts/evidence_inject_full/` (50M steps to final).
- **The 20M arm (c) smoke was read before copying existed**: `closed_book_rank.py --evidence
  prompt` (the card as prompt text, no port) equals the closed-book read for entities on that save.
- **Matched 100M, arm (c): the key/value reader copies, the cross reader does not.**
  `sft.py --reader-kv` (evidence as leading keys of `shared_attn` every loop, prefix rotation):
  gold card 0.003 / 0.003 / 0.002 / 0.000 in distribution (top-1 0.92), 0.02 held out, swapped card
  followed 92%, `mr_ll` under 0.03. Cross reader resumed to 100M: 0.47 / 0.46 / 0.35 / 0.02,
  followed 2%, while its own prompt copy reads 0.10 to 0.13. Saves
  `ckpts/evidence_inject_retrieval_kv/checkpoint_evidence_tok100M_loss7.7670.pt` and
  `ckpts/evidence_inject_retrieval/checkpoint_evidence_tok100M_loss8.4518.pt`; reads in
  `ckpts/inject/rank_*100M_*`.
- **Arm (c) leaks at 100M under both readers**: closed-book `delta` in distribution at tier 100 /
  1000 key/value 0.026 (z 5.2) / 0.149 (z 12.3), cross 0.037 / 0.203, arm (a) 0.099 / 0.332 at the
  same tokens. Mid-cosine then; the full arm (c) confirmed it.
- **Arm (b), masked spans, 300M: nothing stored at 3 sigma.** Closed-book entity `delta` in
  distribution 0.001 / 0.003 / 0.002 / 0.011 (z at most 1.7), held out at most 0.013 (z 2.4); flat
  at 100M too. Watch the tier 1000 entity cell (0.012 at z 2.8 pooled over forms, rising from
  100M). The in-context input path does not leak, so the arm (c) leak is most likely on its
  supervised spans, sharpened 2026-10-05 by the warm-up arm: written by supervised real-value
  spans, not the input or card path, and also after the reader copies. Prompt
  copy 0.15 in distribution, 0.05 held out, worse at tier 1000. Checkpoints
  `ckpts/evidence_inject_masked/`, reads `ckpts/inject/rank_masked*`. `compare` at matched 100M:
  held out holds, in distribution fails on (c) at tier 100 (the held-out hold did not last).
- **Arm (c) in full, key/value reader, 300M: it copies, and fails on both forms** (2026-10-05).
  299.22M tokens, final filler CE 3.8842 (arm (a) 3.9037, arm (b) 3.8769). In distribution gold
  card top-1 1.000 at every tier and class (0.999 for entities at tier 10), swapped card followed
  on every item, `mr_ll` 0.000; held out gold 0.010 to 0.024 (top-1 0.58 to 0.74), followed 0.61
  to 0.64. Closed-book entity `delta` in distribution 0.002 / 0.003 / 0.027 (z 4.1) / 0.120
  (z 8.8), held out 0.000 / -0.001 / 0.013 (z 3.6) / 0.071 (z 8.2). `compare full masked
  retrieval_kv` FAILS on both forms at tier 100; tier 1000 leaks on both. Saves
  `ckpts/evidence_inject_retrieval_kv/` (50M steps to final), reads
  `ckpts/inject/rank_retrieval_kv*`, `compare_full_masked_kv.log`.
- **The arm (c) leak is present before copying and flat after 100M.** Entity `delta` in
  distribution at 100 / 150 / 200 / 250 / 300M: tier 100 0.026 / 0.029 / 0.023 / 0.024 / 0.027,
  tier 1000 0.149 / 0.140 / 0.115 / 0.112 / 0.120; paired 100M to final +0.001 (z 0.2) and -0.029
  (z -2.0), arm (a) +0.279 / +0.134 over the same tokens. Gold top-1 0.91 at 100M, 1.00 from 150M.
  At 50M no copying (gold top-1 0.007 to 0.013 through tier 100) and already 0.010 (z 4.0) / 0.105
  (z 14.4), equal to arm (a) at 50M at tier 100 (paired +0.000, z 0.1). Established: present before
  copying, no growth after 100M; read by the warm-up arm (next bullet) as both an offset written
  before copying and a level training maintains. A dose prediction (about 0.68 of spans carry the real value
  with the real name) failed on `delta`. Copying emerged between 50M and 100M.
- **Loop reading on arm (a) final (`--n-loops`): at tier 100 recall grows with depth, mostly in
  pass 2.** Entity `delta` in distribution at depth 1 / 2 / 3: 0.269 / 0.369 / 0.378; paired 1 to 2
  +0.100 (z 19.7), 2 to 3 +0.008 (z 6.5), prior unmoved. On the real name only (1.21 nats against
  0.14 for the prior); held out nothing grows. Tier 1000 is saturated after one pass (rank 0.0008);
  its `delta` growth (0.393 / 0.453 / 0.467) is the prior drifting, not recall. The storage
  falsifier's condition is met at tier 100; the weights are shared across passes, so it reads as
  two-step recall, not capacity. The axiom is reworded to say so (NEXT.md Decisions). Leak reads
  stay at full depth.
- **Per-exit read (`eval_exit.py`, 2026-10-05): depth pays evenly on this corpus.** Arm (a) final
  on `inject_val`: CE 4.0072 / 3.9132 / 3.9071 at exits 1 / 2 / 3 (+0.0940, then +0.0061). Gain 1
  to 3 is +0.107 to +0.122 over the seven least confident deciles at exit 1, +0.072 in the ninth,
  +0.024 in the most confident; a confidence exit rule is no better than fixed depths at the same
  passes (3.9237 against 3.9106 at tau 0.5); the beta 0.1 optimum is near uniform (0.343 / 0.286 /
  0.371, a bound). Arm (c) reads the same. Reads `ckpts/inject/exit_full.*`, `exit_retrieval_kv.*`.
- **The copy-first warm-up arm, done 2026-10-05: the rule holds through tier 100, tier 1000
  leaks** (`inject_retrieval_kv_cf`, `ckpts/evidence_inject_retrieval_kv_cf/`): arm (c) with
  `--reader-kv` from `seed_micro.pt`, first 100M on `inject_retrieval_s100a50_train` (swap rate
  1.0: every span in a gold-present document carries a substitute the card also carries), then
  `--train-split inject_retrieval_train` to 299.19M (logs `inject_retrieval_kv_cf_warm.log`,
  `inject_retrieval_kv_cf_main.log`); final filler CE 3.9106 against 3.8842 for the original arm,
  a 0.026 gap that opens after the switch and that the logs do not explain. Switch criteria met:
  gold top-1 in distribution 0.973 / 0.973 / 0.959 / 0.953 at 100M, closed-book `delta` within 3
  sigma in every cell at 50M and 100M (original arm (c) at 50M 0.010, z 4.0 / 0.105, z 14.4).
  Entity `delta` in distribution at 100 / 150 / 200 / 250 / 300M: tier 100 -0.006 / 0.011 / 0.013 /
  0.013 / 0.015 (z 2.6), tier 1000 -0.001 / 0.081 / 0.095 / 0.087 / 0.085 (z 6.4); held out final 0.007 (z 2.0) /
  0.047 (z 5.1); paired 200M to final -0.011 (z -1.2) at tier 1000. `compare full masked
  retrieval_kv_cf` HOLDS on both forms through tier 100; tier 1000 leaks on both; dates 0.106
  (z 3.6) at tier 1000. The 0.03 expectation failed; the falsifier triggered at tier 1000 (and,
  100M to final, at tier 100: +0.020, z 3.3). Against the original arm at the final save, paired
  per form: tier 100 -0.012 (z -1.5) / -0.006 (z -1.4), tier 1000 -0.035 (z -2.2) / -0.024
  (z -2.2): lower in every cell, under 3 sigma per form. Copying final: gold top-1 0.999 to 1.000
  and swapped followed on every item in distribution, held out followed 0.66 to 0.69, prompt
  copy held out 0.008 (top-1 0.84, original 0.017 / 0.76). Established: nothing stored without a
  real-value target; real-value supervision after copying still writes frequent facts (tier 1000
  to 0.08 to 0.10, then flat); tier 100 inside 3 sigma on both forms. The original arm's leak is
  in part written before copying, in part a level training maintains. Open: what sets that level
  (the swap rate, the gold-drop documents, residual span loss after copying). Decided 2026-10-05:
  R0b accepted on this arm, tier 1000 leak known, "no real-value fact supervision before the
  reader copies" a pilot requirement; the warm-up is the better of two arms on one seed, not an
  optimum, and its gain over the original arm is a direction, not established per form.
- **Swap rate 0.30 branch, read 2026-10-06** (`inject_retrieval_kv_cf30`, from the warm-up arm's
  100M save on `inject_retrieval_s30a50_train`, 299.19M tokens, filler CE 3.9131): final entity
  `delta` in distribution 0.005 / 0.004 / 0.010 (z 1.7) / 0.083 (z 6.2), held out -0.002 / -0.001 /
  0.003 (z 0.9) / 0.039 (z 4.6). Paired against the warm-up arm, tier 1000 -0.002 (z -0.3) /
  -0.008 (z -1.4), -0.005 (z -1.0) pooled: criterion (z at or beyond -3) not met, no large effect (a
  dose-sized one is under the power); tier 100 -0.004 (z -1.0 / -1.4); `compare` HOLDS on both
  forms. The raw-rank z -4.0 in `compare_cf_vs_cf30.log` is the real name ranking worse (+0.014,
  z 3.8, all classes pooled), the prior also worse, so `delta` moves -0.009 (z -2.1). Copying
  unchanged; prompt copy held out 0.008 (top-1 0.83). Of the middle saves only 150M `none` was read.
- **Placeholder name arm, read 2026-10-08: the dose lever is falsified** (`inject_retrieval_kv_cf_p50`,
  `ckpts/evidence_inject_retrieval_kv_cf_p50/`). Split `inject_retrieval_s15a50p50_train`
  (`--placeholder-rate 0.5`: 64,664 of 129,438 gold-present documents share a placeholder with their
  gold card; keys, gold flags, swaps and the rest byte identical to `inject_retrieval_train`;
  298.26M tokens), from the warm-up arm's 100M save to 298.27M in 68.5 minutes, filler CE 3.9161.
  Final entity `delta` in distribution 0.005 / 0.001 / 0.0185 (z 3.0) / 0.0986 (z 6.7), held out
  0.000 / -0.003 / 0.0133 (z 3.7) / 0.0559 (z 6.4). Paired against the warm-up arm: tier 1000
  +0.014 / +0.009 (z +1.3 each), +0.011 (z +1.8) pooled; tier 100 +0.005 (z +1.8) pooled; the real
  name and the prior both rank better (-0.025, z -4.6; -0.014, z -2.5). Real-name-with-real-value
  supervision halved (0.686 to 0.346 of exposures), level unchanged: falsified (pooled z above -2).
  `compare` HOLDS in distribution, FAILS held out at tier 100. Held-out copying pays (gold top-1
  0.53 to 0.68 against 0.60 to 0.74, swapped followed 0.51 to 0.54 against 0.60 to 0.63, z down to
  -8); prompt copy held out better (top-1 0.84 to 0.87). Decided 2026-10-08: the copy-criterion span
  weight is dropped. Open: what writes the level (gold-absent documents with the real name, the
  input path, the prior drifting). Reads `ckpts/inject/rank_retrieval_kv_cf_p50final_*`,
  `compare_cf_vs_cf_p50.log`, `compare_full_masked_kv_cf_p50.log`.
- **Chain depth arm, read 2026-10-08: inconclusive, not readable** ([chain_depth_micro.md](docs/measurements/chain_depth_micro.md)):
  `chains_kv` (`ckpts/evidence_chains_kv/`), 2M questions (104.66M prompt tokens, 439M evidence,
  ratio 4.20), `--evidence --reader-kv` from `seed_micro.pt`, 102.66M tokens in 51.5 minutes on
  batch 16 x accum 2 (`ckpts/inject/config_micro_chains.yaml`; batch 32 spilled in minute one).
  `[eval]` answer CE on `chains_val` 0.628 / 0.501 / 0.455 at 50 / 75 / 100M (top-1 per token
  0.87); selector near uniform (`selection:` 0.495 to 0.440). Every accuracy cell at chance at 50M
  and final: `chains_eval` depth 3 all sites 1-hop 0.104 (chance 0.115), 2-hop 0.187 (0.185),
  3-hop 0.194 (0.206); held-out 2-hop kept 1 / 3 0.188 / 0.194 (0.191), Delta +0.006 at sigma
  0.013. The read carries content (4-hop answer CE per token 3.82 with no site, 0.47 with them)
  but not the choice: about 3.3 nats per answer against about 1.9 for a uniform pick among the
  candidates, so it copies an answer-type entity and does not select which; no bug in
  `eval_chains.py`. Readable precondition fails, so inconclusive; not a kill. `loop_scale` ended
  [1.34, 0.48, 0.07], the third under 0.01 at 50M. No change to depth or the chain slice; the
  pilot's L1 decides. Instrument gaps (no per-question JSON, answer CE for hop4 only, `*` glued to
  the previous column) are in the record. Reads `ckpts/inject/chains_kv_{50M,final}_{d123,hop4}.*`.
- **Seed-side instruments, done 2026-10-06** ([seed_instruments.md](docs/measurements/seed_instruments.md)):
  PopQA with its prior control, seed entity `delta` 0.0287 (z 11.7) / 0.0263 (z 12.8) / 0.0356
  (z 19.4) by tail / mid / head, concentrated in four cue relations (father, mother, capital,
  capital of: 0.151; tail 0.261 against head 0.113); the other 9,430 entity items 0.0088 / 0.0075 /
  0.0070. At the entity tail the 10M arms move the raw rank +0.0090 (z 7.4, arm A) but the prior
  +0.0102, so `delta` +0.0012 (z 1.0): the finetune moved the prior, stored nothing. Counterfactual
  likelihood (1,409 items): seed `mr_ll` 0.295 to 0.766 by stratum is a frequency-ratio prior
  (neutral port; correlation 0.636 with the log count ratio); the arms follow in no stratum (paired
  gap +0.01, z 0.1, arm A). Gates A1 and A3 carry the method notes (prior control and cue relations;
  ratio-binned or baseline-paired `mr_ll`). Raw `ckpts/instruments/popqa_*`, `cfll_*`.
- Graft lineage, binding: Phase 4 arms A and B killed at 10.08M on content gain (+0.058 / +0.081);
  in-context ceiling on `evidence_fixed` 3.18 nats pooled; R0 failed; the graft branch is closed.
- **Next, in order** (the full ladder with its done steps is in NEXT.md):
  1. The pilot spec, with the copy-first warm-up requirement, swap rate 0.15 and no further lever
     against the leak (decided 2026-10-06 by the user: a leak this size matters little for the
     pilot, and RL planned later can act on small leaks; caveat on record: RL is Parked and shapes
     behaviour at conflicts rather than removing what is stored, so for small leaks the goal is read
     as behavioural at conflicts). The chain slice stays 10% at hops 25 / 50 / 25 and the pilot's
     L1 is the deciding read on the loop clause.
  2. The pilot (Phase 5) on that spec with the key/value reader, ladder cut to budget.
  Spec written 2026-10-09 ([PILOT.md](docs/plans/PILOT.md)). Its build list, in order: the split
  merge tool `scripts/merge_evidence_splits.py` with its test, the `eval_chains.py` instrument
  fixes (per-question JSON, per-answer gold NLL against ln K, the `*` marker), `config_pilot.yaml`
  and its seed, the three biography builds at seq 4096 (retrieval s15, retrieval s100, full) and
  the three merges (warm-up, main, control), the launchers and per-save read scripts. No change to
  `modules/`.
- Pilot shape probe (2026-10-09, `ckpts/inject/config_pilot_probe*.yaml`, logs
  `ckpts/inject/pilot_probe*.log`): 99.1M total, 84.9M active, 249M FLOP/token at seq 4096, 39.4M
  per evidence token. Batch 8 x 4096 x 1 on `inject_retrieval_train`: 44k to 50k tok/s, 20.4 GB;
  batch 4 x 2: 41k to 45k, 12.7 GB. On `evidence_train` (ratio 3.75, cap 4608) rows close on the
  evidence budget at fill 17 to 30% and the rate is 2.7k to 3.9k, GPU at 99%: the evidence ratio
  sets the rate, not the batch; the pilot's merged corpus (ratio about 0.73) fits to about 16k.
- Micro runs: 55k to 60k tok/s, 13.3 GB peak with the card buffer, `--batch-size 1024` for
  `closed_book_rank.py`. The batch move to 32 x 1 changed no tokens per update; if copying ever
  emerges late, the lever is fewer tokens per update (16 x 1). Relaunch a stopped arm with the same
  `-c` and every flag; a stop leaves `STOP` in the run directory. `ckpts/instrsmoke.log` stays the
  full-shape memory reference (batch 2 x 4096, 26.36 GB peak).

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
  "Passes add computation, not capacity; recalling a weakly stored fact can take more than one
  pass" is an axiom, not a measurement (NEXT.md Decisions, with its L1 falsifier; reworded
  2026-10-04 from "loops buy computation, not storage"). On the micro arm (a) closed-book recall of
  weakly stored facts (tier 100) grows with depth, mostly in pass 2, so every leak read is taken at
  full depth.

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
  prepare_injection_data.py  fictional biographies at 1/10/100/1000 exposures, arms full/masked/retrieval -> data/prepared_inject
  prepare_chain_data.py      synthetic k-hop chains with .evhop and .chains.jsonl sidecars -> data/prepared/chains_*
  build_store.py      retrieval stores (NQ, TriviaQA, HotpotQA) and edit stores -> data/index/<name>/
  init_scratch_seed.py       random-init seed for a from-scratch sft.py run (TINY_LLM_CONFIG picks the shape)
  migrate_*.py        phase0, ir_reshape (--arm random|warm), loop_inject, evidence_port, groundedness_head
  eval_abstention.py  THE acceptance metric (SQuAD v2); --evidence-port scores gold/none/distractors/mixed/counterfactual
  eval_benchmarks.py  fixed 13-task suite, one scoring path for this model and the peers; popqa is a rank task outside `all`
  closed_book_rank.py likelihood rank of the gold among same-type candidates (bios: by exposure tier; compare: paired)
  eval_chains.py      accuracy by hops and by kept read sites (reader_sites_kept) at fixed depth
  eval_exit.py        per-exit CE without evidence: paired gains, oracle and beta optimum bounds, confidence exit vs fixed depths
  eval_store.py       store recall of the model's query vs bge, open-book pathway EM vs oracle, store-edit flip rate
  entity_frequency.py exact token-sequence counts in a .bin (the counterfactual strata)
  eval_calibration.py eval_probe.py eval_stage0.py evidence_ceiling_probe.py (--fixed-split: ceiling on evidence_fixed)
  inference.py gradio_app.py prune_vocab.py fetch_tokenizer.py setup.sh onstart.sh run_sft_after_pretrain.sh
modules/model/        transformer gemma4 moe router experts information_retrieval evidence mtp attention kv_cache
modules/data/         dataset (pretrain) sft_dataset evidence_dataset chat abstention biographies chains entity_swap store
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
python scripts/evidence_ceiling_probe.py -c SEED --fixed-split evidence_fixed --max-questions 2000 --json-out docs/measurements/ceiling_fixed.json
python scripts/prepare_evidence_data.py --heldout --sources-only --max-evidence-tokens 4608   # byte-checked .src backfill, CPU
python scripts/sft.py --evidence -c SEED --ceiling-json docs/measurements/ceiling_fixed.json [--data-dir D --train-split S --val-split V]
python scripts/eval_abstention.py -c CKPT --evidence-port --evidence-condition gold,counterfactual,distractors,none --json-out OUT.json
python scripts/eval_abstention.py -c CKPT --evidence-port --evidence-condition counterfactual --counterfactual-likelihood-only   # mr_ll only, abstaining checkpoints
python scripts/prepare_injection_data.py --out-dir data/prepared_inject --target-tokens 300000000 --filler-phases ir,phase1,phase2 --seq-length 1024 --swap-rate 0.15 --anon-rate 0.5 --gold-drop-rate 0.2 --distractors 3 --seed 42 --device cuda
TINY_LLM_CONFIG=config_micro.yaml python scripts/init_scratch_seed.py --out ckpts/inject/seed_micro.pt --seed 0
TINY_LLM_CONFIG=config_micro.yaml python scripts/sft.py --evidence -c ckpts/inject/seed_micro.pt --run-name inject_full --data-dir data/prepared_inject --train-split inject_full_train --val-split inject_val   # also inject_masked, inject_retrieval
TINY_LLM_CONFIG=config_micro.yaml python scripts/sft.py --evidence --reader-kv -c ckpts/inject/seed_micro.pt --run-name inject_retrieval_kv --data-dir data/prepared_inject --train-split inject_retrieval_train --val-split inject_val   # the key/value reader arm (c)
python scripts/prepare_injection_data.py --out-dir data/prepared_inject --target-tokens 300000000 --filler-phases ir,phase1,phase2 --seq-length 1024 --swap-rate 1.0 --anon-rate 0.5 --gold-drop-rate 0.2 --distractors 3 --seed 42 --arms retrieval --suffix s100a50 --device cuda   # the copy-first warm-up split
python scripts/prepare_injection_data.py --out-dir data/prepared_inject --target-tokens 300000000 --filler-phases ir,phase1,phase2 --seq-length 1024 --swap-rate 0.15 --anon-rate 0.5 --gold-drop-rate 0.2 --placeholder-rate 0.5 --distractors 3 --seed 42 --arms retrieval --suffix s15a50p50 --device cuda   # the placeholder name split
TINY_LLM_CONFIG=config_micro.yaml python scripts/sft.py --evidence --reader-kv -c ckpts/inject/seed_micro.pt --run-name inject_retrieval_kv_cf --data-dir data/prepared_inject --train-split inject_retrieval_s100a50_train --val-split inject_val   # warm-up to 100M, then the same line with --train-split inject_retrieval_train
TINY_LLM_CONFIG=config_micro.yaml python scripts/closed_book_rank.py bios -c CKPT --form both --evidence none|gold|swapped|prompt --batch-size 1024 [--n-loops N] --json-out ckpts/inject/rank_full.json
python scripts/closed_book_rank.py compare rank_full.json rank_masked.json rank_retrieval.json
TINY_LLM_CONFIG=config_micro.yaml python scripts/eval_exit.py -c CKPT --data-dir data/prepared_inject --split inject_val --max-batches 40 --json-out OUT.json
python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains --splits eval,heldout_tmpl,hop4 --eval-questions-per-hop 1000 --seed 42 --device cpu   # train,val on cuda
python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains --splits train,val --train-questions 2000000 --val-questions 2000 --seed 42 --device cuda
python scripts/eval_chains.py -c CKPT --data-dir data/prepared --splits chains_eval,chains_heldout_tmpl,chains_hop4 --depths 3,4 --sites all --json-out OUT.json
python scripts/build_store.py --name openqa --sources nq,triviaqa,hotpotqa --max-questions 3000 --chunk-tokens 128 --device cuda --seed 42
python scripts/build_store.py --edit-from openqa --name openqa_edit --edit-fraction 0.5 --neighbour-cos 0.85 --device cpu --seed 42
python scripts/eval_store.py recall|pathway|edit -c CKPT --store data/index/openqa [--edit-store data/index/openqa_edit] --buffer 4 --loop 1 --json-out OUT.json
python scripts/eval_benchmarks.py -c CKPT --tasks popqa --rank-candidates 20 --json-out OUT.json
python scripts/inference.py -c CKPT -p PROMPT -n 200 [--evidence chunks.json] [--converge-tol T]
bash tests/run_env_check.sh; bash tests/run_tests.sh tests/test_attention_equiv.py tests/test_overfit.py
touch ckpts/training/STOP                           # clean stop (exit 10); kill -USR1 <pid> saves now
```
GPU-free tests: the `modules/runtime/` ones (`test_checkpoint_lifecycle`, `test_hf_sync`,
`test_control`, `test_supervisor`, `test_phase_targets`, `test_hf_token`, `test_checkpoint_atomic`)
plus `test_prepare_data`, `test_sft_dataset`, `test_dataset_packing`, `test_token_tracker`,
`test_prepare_evidence_heldout`, `test_heldout_sources`, `test_entity_swap`, `test_entity_frequency`,
`test_counterfactual_condition`, `test_biographies`, `test_prepare_injection`, `test_closed_book_rank`,
`test_config_override`, `test_chain_generator`, `test_store`, `test_popqa`, `test_eval_exit`. `TINY_LLM_CONFIG=path.yaml`
swaps the config yaml for every script (the checkpoint still decides shape and mode on load, so a
micro checkpoint under the default yaml fails the strict load, as it should).

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
  `moe.evidence_reader_rotary_off` exists; `sft.py --reader-no-rotary` sets it on a seed) and the
  reader kind (`evidence_reader` "kv" iff `moe.evidence_reader_kv` exists, which also implies the
  port; `sft.py --reader-kv` sets it and drops a cross seed's `shared_evidence.*` and
  `evidence_query_bias` by name) from tensor presence. `load_model_state` is strict except
  `IR_TEMPERATURE_KEYS` and `NEUTRAL_LOOP_KEYS` (whose inits are the old behaviour) and
  `READER_MODE_KEYS` (markers with no learned value).
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
- **Key/value reader** (`evidence_reader="kv"`, marker buffer `moe.evidence_reader_kv`, `sft.py
  --reader-kv`): no `shared_evidence`, no `evidence_query_bias`. The evidence states go through
  `shared_attn`'s own norm, the same chunk gate (after the norm, which would divide it out), and
  its own `k_proj`/`v_proj`, and become leading keys of each query segment under one softmax
  (`attention.prefix_varlen_attention`: keys interleaved `[prefix_s, own_s]`, one flash call,
  causal bottom-right alignment). Prefix keys are rotated at `start_s - E_s + i`
  (`evidence.prefix_position_ids`, negative allowed, `rotary_emb.at` rebuilds fp32 frequencies
  because `model.to(bf16)` casts `inv_freq`). `evidence_loop_scale` scales the evidence values.
  Not neutral with evidence attached; bit-identical without. `last_reader_output` is
  `shared_attn`'s output. No KV cache with evidence (asserted; `inference.py` turns it off).
  `tests/test_evidence_kv_reader.py`.
- Nothing ragged leaves the dataloader: `chunk_keys [B, C, 384]` + `[B, C]` slot map, flattened
  in-thread by `evidence_from_batch`. `chunk_gold` / `condition_ids` ride along when the corpus
  has `.evgold`/`.cond` (warned once otherwise). `chunk_segments` comes from packing, never
  re-derived. `source_ids [B, S]` rides along when `{split}.src` exists (uint8 per document,
  index into `prepare_evidence_data.SOURCE_KEYS`, 255 unknown; -1 on padding); the `[eval fixed]`
  pass then prints one line per source, the kill stays on the pooled numbers.
- **Read ablation** (`forward(..., reader_sites_kept=j)`, inference only, asserted off in training,
  with a KV cache and with the convergence exit): read sites are numbered
  `loop * moe.read_sites_per_loop + 1` in execution order; a site above `j` gets `evidence=None`
  and `memory=None` for that loop, cutting the reader and the IR expert's external read together
  (the second content route). `None` is bit-identical to today, `j >= sites` equals `None`, `j = 0`
  equals `evidence=None` on the residual. After an ablated loop `last_reader_output` keeps the last
  live loop's read and the IR `memory_weights_by_loop` has no entry for it. `test_read_ablation.py`.
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
- `--heldout` writes `{split}.src` (per document, `SOURCE_KEYS` index). `--heldout --sources-only`
  backfills it onto an existing build: a zero-embedder replay that must match every existing
  file but `.evkey` byte for byte, then copies only the `.src` files; it aborts naming the first
  differing file. Never rebuild `evidence_dev` / `evidence_fixed` while an arm reads them.
- Other evidence-format builders: `prepare_injection_data.py` (biographies plus filler from
  `data/prepared/{ir,phase1,phase2}.bin`; arms `full` and `masked` share `.bin` bytes and differ
  in `.mask` only; `retrieval` carries the store card buffer with swaps and name placeholders;
  `--placeholder-rate` (default 0, byte identical at 0, its own rng stream) renames gold-present
  documents: the document and its gold card share a placeholder, each distractor its own;
  `inject_val` is filler only and identical across arms) and `prepare_chain_data.py` (fictional
  k-hop chains, `.evhop` per chunk and `.chains.jsonl` per document, eval-only sidecars; held-out
  2-hop compositions are absent from every seen split including 3-hop). Stores
  (`data/index/<name>/`: `chunks.jsonl`, `keys.npy` fp16 bge, `meta.json`, optional
  `questions.jsonl`, `query_keys_bge.npy`, `edits.jsonl`) are read by `modules/data/store.py`.

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
- `--data-dir`, `--train-split`, `--val-split` override the profile's data for any profile (logged
  as `data overrides`, not stored in the checkpoint, so a relaunch passes them again).
  `--ceiling-json` reads `evidence_ceiling_probe.py --fixed-split` output and prints the gold gain
  against the same-rows ceiling, pooled and per source. The from-scratch micro runs are
  `TINY_LLM_CONFIG=config_micro.yaml sft.py --evidence -c ckpts/inject/seed_micro.pt` with
  `conversation_loss_weighting: false`: the span weight travels through `.mask` (0 on unsupported
  fact spans), so the MTP targets carry it too; `.factspan` (uint8 per token: 0 none, 1 to 5 the
  attribute, 6 name, 7 placeholder) is eval-only.
- Chat control tokens are resolved from the tokenizer and asserted. Only assistant text + EOS is
  supervised; conversations with roles outside system/user/assistant are dropped.
- **`eval_abstention.py` generates**: left-padded batched decode, no KV cache, numbers comparable
  only at fixed `--batch-size`. **Read the generated-answers block first** (token-level
  calibration passed while behaviour failed). `--evidence-port`: G3 is answer-span CE with the real
  answer under every condition; G3b is external-mass AUROC at the last prompt position, per loop.
  The `counterfactual` condition (needs `gold` in the same run) swaps the answer entity in the gold
  chunk for a same-type entity from a gazetteer built over the slice's own answers
  (`modules/data/entity_swap.py`, no NER model; PROPER answers are mostly ineligible by design)
  and reports follow rate, `mr_gen` (Longpre) and `mr_ll` by frequency stratum counted in
  `data/prepared/ir.bin` (`entity_frequency.py`, cached under `data/benchmarks/entity_freq/`).
  `--json-out` writes per-record entries in evidence mode; `memory_mass_by_expert` reads every IR
  expert (the old `last`/`by_loop` keep reading expert 0).
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

# CLAUDE.md

Working notes for this repo. Prose docs live in [docs/](docs/); this file is the operational map:
what lives where, the non-obvious invariants, how to run things, and **what is next**.

## What is next (keep this section current)

Authoritative copy: the "Now" section at the top of [docs/plans/NEXT.md](docs/plans/NEXT.md)
(the plan; older notes call it `PLAN.md`). Update both when the next step changes. As of 2026-09-20:

1. **Corpus: built** (`--target-tokens 150000000 --max-evidence-tokens 4608 --max-source-epochs 4`,
   28 min). `evidence_train`: 963,011 conversations, 140.1M prompt tokens, 525.6M evidence tokens,
   3.74M chunks; `evidence_val` 9,707. Both QA sources ran the full 4 passes, so QA is 32.6% of
   tokens and ~90% of conversations; `many` is 58,767 rows; 59.9% of rows answerable; replay 25.0%
   of tokens. The one-pass build is archived as `data/prepared/evidence_nomany_*` (and is no longer
   loadable — it predates `.ans`).
2. **Seed: migrated.** `ckpts/repair/checkpoint_repair_final_irrandom_evidence_grounded.pt`, 4
   tensors / 1.5K parameters added, every other tensor bit-identical to its source.
3. **Batch and cap: retuned** to `batch_size: 2`, `grad_accumulation_steps: 8`,
   `max_evidence_tokens: 14336` — same tokens per optimizer step, 64% fill instead of 55%, ~25 GiB
   peak. Measured: peak ≈ **3.5 GiB + 0.61 MiB per evidence token**, the prompt axis negligible
   beside it, so batch 4 on this corpus wants ~30 GiB before the optimizer's +4.3 GiB against
   ~30.2 GiB free and dies inside WSL as `CUDA driver error: device not ready`. ~0.7 s/step
   expected, so 10M tokens is roughly 20 minutes.
4. Train: `python scripts/sft.py --evidence -c <the _grounded.pt seed>` under a watch; kill at 10M
   tokens if the per-condition `[eval]` gold-vs-none CE gap is < ~0.1 nats. Watch `selection:`,
   `grounded:` (and its held-out AUROC), `|shared_evidence.o_proj|rms`, `external mass`.
5. Read G3/G3b with `eval_abstention.py --evidence-port`, then the benchmark suite. Baseline on the
   migrated seed: gold-vs-none gap 0.0000 nats.

All three losses are wired now (`evidence_selection_loss`, `groundedness_loss`, the aux
`token_mask`). **`ckpts/evidence_smoke/` is empty** — the rehearsal checkpoints were deleted, so
the only way to exercise the `--evidence` profile now is the real run. Its log survives as
`ckpts/evsmoke.log` and is the memory and throughput reference: batch 4 × 4096, cap 12288, a
ratio-2.93 corpus, **peak 24.29 GB with full optimizer state**, 18–29k tok/s. `ckpts/evidence/`
does not exist yet, so the first `--evidence` launch honours its `-c` seed; once it does, a later
launch resumes from it instead.

**End every turn with a short "something you should know in my opinion"** — one thing the user
did not ask about but should hear: a risk noticed in passing, a stale assumption, a cheaper path,
a number that does not add up. Plain prose, one to three sentences, never skipped.

**Close every turn that did work with three short lines**: what was done, what changed
architecturally (a new tensor, a new loss term, a moved invariant, a changed on-disk format —
"nothing structural" is a valid and common answer), and what is next. One or two sentences each,
in the reply and not in a file, before the "something you should know". A turn that only answered
a question skips it.

## Running anything: WSL + `env_init`

**Every command runs under WSL, from the repo root, after `source env_init`.** The dev box is
Windows; CUDA 12.9, flash-attn and Transformer Engine live only in the WSL Ubuntu install.
`env_init` is gitignored: it exports `CUDA_HOME`, include/library paths, `LD_LIBRARY_PATH`
(incl. `/usr/lib/wsl/lib`), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and activates `venv/`.

```powershell
wsl bash -lc "cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init && python scripts/inference.py --help"
```

Without `source env_init` nothing under `modules/model/` imports (`transformer_engine` is a
module-scope dependency); from any other cwd `config.py` fails (it opens `config.yaml` by relative
path). `tests/run_tests.sh` encodes this and takes `TINY_LLM_ROOT` / `TINY_LLM_ENV_INIT` overrides
for the rented box (`TINY_LLM_ENV_INIT=/dev/null`, where `scripts/setup.sh` installed into the
system python).

**Anything that trains runs in the background under a `Monitor` watch, always** — every `sft.py` /
`pretrain.py` / `run_training.py` launch, however short. A foreground run blocks for its whole
duration and an unwatched background run is only read when someone remembers, so a stall, OOM or
crash three minutes in burns the slot. The watch filter must match the failure signatures as well as
the progress line (`Traceback|Error|Killed|OOM|assert` alongside `Step`/`Tokens/sec`). **A failure
that prints nothing looks like a slow step**: a device-side assert leaves the process in `R` with the
GPU at ~1%; if the progress line goes quiet, reproduce one micro step outside the trainer.

**Kill a run the moment its own instrumentation says it cannot pass.** The zero-init tensors are
logged per step for exactly this (`|g_proj|rms`, `|inject|rms`, `|shared_evidence.o_proj|rms`); one
that has stopped climbing by the first checkpoint is the answer. The remaining hour only buys a
control at matched tokens — worth having sometimes, and worth saying out loud either way.

**Fix slow runs immediately.** 4 × 4096 stays resident on the 5090 (21–27GB peak); 8 × 4096 spills
into shared system memory at a ~3–4x throughput cost and does not OOM. **That is the no-evidence
figure.** With evidence attached the peak is set by the evidence axis, not the prompt axis —
measured ≈ **3.5 GiB + 0.61 MiB per evidence token**, with the prompt axis lost in the noise beside
it — so `--evidence` runs at 2 × 4096 with the accumulation doubled. **An evidence run that
overflows does not degrade gracefully**: WSL's paravirt layer reports it as
`CUDA driver error: device not ready` at an arbitrary op (backward, rotary, an MoE activation),
with `dxgkio_make_resident: Ioctl failed: -12` in `dmesg`, which reads like a driver fault and is
an ordinary out-of-memory. **All local finetunes run in BF16 — do not set `USE_FP8`.**

## Subagents

**Every subagent runs on Sonnet, in caveman mode** (`/caveman:caveman full`, or `ultra` for a pure
lookup). Say so in the spawn prompt — a subagent does not inherit this session's mode and has to
invoke the skill itself. Reason: its report is injected verbatim into this context, and an
uncompressed one is 2–3x the size of what it says. Caveman applies to the **report**, not to what
the subagent writes into the repo (code, comments, docstrings, docs, commit messages stay normal
prose). One subagent per task; combine tasks that share a file, state the owned files in the prompt,
run disjoint groups in parallel. Any documentation or code comments must be written in the same style as the rest of the repo, not in caveman.

## What this is

`tiny-moe-llm`: an experimental ~383M-param LM (224M active). A dense Gemma4-style decoder feeds a
**single MoE block applied `n_loops` times** (LoopLM-style recurrence) with a heterogeneous expert
pool (self-attn / cross-attn / IR / MLP) behind one router plus always-on shared experts, multi-token
prediction heads, and an **evidence port**: retrieved chunks encoded by the same decoder, read at
every loop by an always-on cross-attention reader, with the IR expert scoring the chunks' external
embeddings as a selector. Document-packed flash-attn varlen; optional FP8/NVFP4 via TE.

Research code, not a library: no packaging, no test framework, no CI. Entry points are under
[scripts/](scripts/).

One real run exists: 16B tokens of pretraining on a rented H100, then a chain of local finetunes —
SFT, abstention repair, three IR sharpening arms, a loop injection arm — and the Phase 4 build.
[docs/CONCLUSION.md](docs/CONCLUSION.md) is the pretraining write-up; every gate since is in
[docs/measurements/](docs/measurements/); the plan is [docs/plans/NEXT.md](docs/plans/NEXT.md)
(Phases 0, 1, 1b, 2, 3, 3b, 3c done; 4 built, smoke-tested and its corpus built, not yet run); the
pre-Phase-4 review is [docs/review_2026-09-18.md](docs/review_2026-09-18.md). **The benchmark suite
is the quality instrument** — CE on the local slice is a health check —
[benchmark_snapshot.md](docs/measurements/benchmark_snapshot.md) is the baseline every change is
diffed against. A cluster with ~5k H100-hours is expected for the real post-POC run.

**The verdicts that bind everything below**, each measured, each with a record:

- The learned IR table stores nothing the trunk lacks: zeroing its read costs 0.0002 nats across
  three key inits and two widths, trained or not. Its size is frozen out of the real run spec.
- In-context gold evidence is worth +3.23 nats on the answer span; a distractor passage costs 0.63
  nats *more than nothing*. The external evidence pathway is the mechanism.
- Every graft onto the converged checkpoint (G1, G2, G2b, G2c) measured 0 for 4. The real run
  pretrains with retrieval from token 0.
- `p_max` carries no answerability signal; a linear probe of the trunk reads 0.584 on every
  checkpoint alike. Abstention precision is pinned at ~0.578 by the data lever.
- Loops buy computation, not storage. Looping is a spec requirement, not a hypothesis; a weak later
  loop is a defect to fix, never a reason to cut depth.

## Layout

```
config.py / config.yaml     hyperparameters -> ModelConfig / TrainingConfig / SFTConfig /
                             RepairConfig / IRConfig / EvidenceConfig
utils.py                    logger, BASE_DIR, dtype aliases, TOKENIZER_REPO/DIR, HF_UPLOAD_REPO,
                             get_hf_token, save/load_checkpoint, model_params_for_state_dict +
                             load_model_state (the checkpoint, not the yaml, is the authority on shape)
env_init                    WSL/CUDA env + venv activation (gitignored)
scripts/
  run_training.py           THE unattended entry point: phase1 -> phase2, relaunches through preemptions
  pretrain.py               THE training loop; `pretrain(phase=None) -> exit code`; train_step is shared
  setup.sh / onstart.sh     rented box setup; vast.ai onstart hook
  run_sft_after_pretrain.sh unattended pretrain -> SFT -> abstention eval chain
  fetch_tokenizer.py        pulls TOKENIZER_REPO into TOKENIZER_DIR
  prepare_data.py           phase1/phase2.bin/.idx from the Hub mix (on the box). `--phases ir` builds
                             the IR sharpening corpus LOCALLY; NEVER omit --manifest-key there
  prepare_sft_data.py       sft_train/sft_val .bin/.idx/.mask (local); `--profile repair`
  prepare_evidence_data.py  the five-condition oracle-evidence corpus (+ .ev/.evidx/.evchunk/.evkey/
                             .evkeyidx/.evgold/.cond); honours the smoltalk2 holdout by import
  archive_corpus.py         pack/list/restore a prepared split (both builders delete shards as they go)
  sft.py                    THE post-training entry point: `--repair` / `--ir` / `--evidence` profiles,
                             one train_step, ckpts/{sft,repair,ir,evidence}
  migrate_phase0.py         folds the deleted halt gate into loop_scale, strips both old heads
  migrate_ir_reshape.py     rebuilds the IR table at 65536 x 384; `--arm random` / `--arm warm`
  migrate_loop_inject.py    adds the zero-init loop input injection (arm D, failed)
  migrate_evidence_port.py  adds the reader (o_proj zero) and the selector's adapters (orthogonal)
  migrate_groundedness_head.py  adds the groundedness readout (zero-init, read by no forward)
  eval_calibration.py       p_max ECE/AUROC, early-exit curve, loop-convergence stats; Gate P0 harness
  eval_abstention.py        THE acceptance metric: SQuAD v2 abstention precision/recall + ECE;
                             `--evidence-port` scores gold/none/distractors/mixed through the port
  eval_benchmarks.py        the fixed 13-task suite, one scoring path for this model and the peers
                             (downloads to data/benchmarks, peers to ckpts/peers)
  eval_probe.py             linear answerability probe on the final loop's last-position hidden state
  eval_stage0.py            IR entropy + ablation, per-loop query drift, loop dynamics; read-only hooks
  evidence_ceiling_probe.py in-context gold / none / distractor CE on the answer span
  inference.py              KV cache, MTP drafting, convergence exit, `--evidence FILE`
  gradio_app.py             browser UI over inference.stream_generate
  prune_vocab.py            one-shot 129280 -> 65536 vocab prune
modules/model/
  transformer.py            TinyMoETransformer + TokenTracker + evidence encoder + convergence exit
  gemma4.py                 dense decoder: GQA, RoPE, RMSNorm(te), per-layer embeddings
  moe.py                    LoopMixtureOfExperts, ParallelSparseMoELayer, is_fresh_loop_param
  router.py                 Router (+ annealed noise), compute_aux_loss (optional token_mask)
  experts.py                SelfAttention / CrossAttention / InformationRetrievalExpert
  information_retrieval.py  the table, two stage read, external store, refresh, evidence_selection_loss
  evidence.py               EvidenceBatch, chunk positions/segments, reader gate, GroundednessHead
  mtp.py                    MTPHead, chunked LM-head CE, compute_mtp_loss
  attention.py              varlen_attention (separate K segments), cu_seqlens_from_doc_ids, SDPA fallback
  kv_cache.py               one slot per decoder layer and per (loop, non-MLP expert); additive only
  modules.py / embeddings.py / utils.py   SmallLMHead; RoPE; EncoderOutput
modules/data/               dataset.py (pretraining), sft_dataset.py, evidence_dataset.py, chat.py,
                             abstention.py (the closed phrasing set)
modules/runtime/            unattended-run machinery; MUST NOT import torch.nn / TE / modules.model
                             (checkpoints.py, hf_sync.py, control.py, status.py) — every test GPU-free
tests/                      tracked plain assert scripts, no pytest
ckpts/, data/datasets/, data/prepared*, data/benchmarks   gitignored
```

`modules/*/__init__.py` are empty; imports are always fully qualified.

## Commands

```bash
source env_init                      # required, see above
bash scripts/setup.sh --hf-token X   # rented box only
python scripts/run_training.py       # the real run
python scripts/pretrain.py --phase phase1
python scripts/prepare_sft_data.py [--profile repair]
python scripts/prepare_data.py --phases ir --ir-tokens 210000000 --val-tokens 2000000 --manifest-key ir_prep
python scripts/prepare_evidence_data.py --target-tokens 150000000
python scripts/archive_corpus.py pack --all | list
python scripts/sft.py --from-hub
python scripts/sft.py --repair -c ckpts/trained/checkpoint_sft_final_phase0.pt
python scripts/migrate_ir_reshape.py -c CKPT --arm random|warm
python scripts/sft.py --ir -c CKPT_irrandom.pt
python scripts/migrate_evidence_port.py -c CKPT
python scripts/migrate_groundedness_head.py -c CKPT_evidence.pt
python scripts/sft.py --evidence -c ckpts/repair/checkpoint_repair_final_irrandom_evidence_grounded.pt
python scripts/migrate_phase0.py -c CKPT
python scripts/eval_calibration.py -c CKPT --start-doc-idx 0 --max-batches 40 --batch-size 4
python scripts/eval_abstention.py -c CKPT --max-examples 2000 --batch-size 16 [--skip-forced] [--example-offset 2000]
python scripts/eval_abstention.py -c CKPT --evidence-port --evidence-condition gold,none,distractors,mixed
python scripts/eval_probe.py -c CKPT --json-out docs/measurements/probe_CKPT.json
python scripts/eval_stage0.py -c CKPT --start-doc-idx 0 --max-batches 40 --batch-size 4 --max-loops 6
python scripts/eval_benchmarks.py --peer pythia-410m --validate --json-out docs/measurements/benchmarks/pythia-410m.json
python scripts/eval_benchmarks.py -c CKPT --compare docs/measurements/benchmarks/*.json
python scripts/evidence_ceiling_probe.py -c CKPT
python scripts/inference.py -c CKPT -p PROMPT -n 200 [--evidence chunks.json] [--converge-tol T]
bash tests/run_env_check.sh
bash tests/run_tests.sh tests/test_attention_equiv.py tests/test_overfit.py
touch ckpts/training/STOP            # clean stop (exit 10); kill -USR1 <pid> checkpoints now
```

Tests are plain scripts (`sys.path.insert` + asserts). GPU-free ones: the `modules/runtime/` tests
(`test_checkpoint_lifecycle`, `test_hf_sync`, `test_control`, `test_supervisor`, `test_phase_targets`,
`test_hf_token`, `test_checkpoint_atomic`) plus `test_prepare_data`, `test_sft_dataset`,
`test_dataset_packing`, `test_token_tracker`. Operational detail lives in
[docs/runbook.md](docs/runbook.md).

## Config

`config.yaml` -> `config.py`, five surfaces ([docs/configuration.md](docs/configuration.md) has the tables):

- `ModelConfig.Params` — splatted into `TinyMoETransformer(**...)`. `ModelConfig.Forward` is empty.
- `TrainingConfig` — `total_steps` is derived as `target_tokens // (batch * seq * grad_accum)`.
  Holds **every loss weight** (`lambda_mtp`, `aux_loss_weight`, `evidence_selection_weight`,
  `groundedness_weight`, `loop_ce_weights` + `loop_ce_subsample`, `loop_count_sampling`);
  `loop_ce_weights`' length is asserted against `n_loops` at import time.
- `SFTConfig`, and its **subclasses** `RepairConfig` / `IRConfig` / `EvidenceConfig` — only the keys
  a block names are read from it; everything else inherits by attribute lookup. That inheritance is
  the invariant: each profile is the SFT run with different data, and a second copy of the shared
  numbers is a second thing that drifts. No loss weights here on purpose (`sft.py` reuses
  `pretrain.train_step`). `hf_upload_repo` defaults to `""` (uploads off) for all four; `None`
  (key absent) means "use `utils.HF_UPLOAD_REPO`" — the two must never be collapsed.
- `IRConfig` adds `fresh_lr`, the temperature anneal (`temperature_scale(progress)` is geometric,
  clamped at 1.0), `cluster_refresh_tokens`, `dead_quantile`. `EvidenceConfig` adds `fresh_lr` (port
  tensors only), `max_evidence_tokens` (**set against the corpus's evidence-to-prompt ratio, and
  it is also the dominant term in peak memory** — the two pull opposite ways, which is the whole
  difficulty. A row needs `ratio * seq_length` of evidence to fill its token budget, so under that
  the evidence budget closes every row early and the unused prompt slots are padding the body still
  pays 502M FLOP/token to run; over it, `0.61 MiB` per evidence token times the batch overflows the
  card. At 12288 against the rebuilt corpus's ratio of 3.75 the dataset closes 206 of its first 256
  rows on the evidence budget, for 55% fill; 14336 at `batch_size: 2` is 64% fill at ~25 GiB —
  read `fill` and `closed by the evidence budget` off the packing line in the first minute of any
  run. **Fill lags what the ratio predicts** because the ratio is a corpus mean and packing is
  greedy: a row that draws QA conversations (short prompts, 6–7.6× evidence) closes on evidence
  with its token budget half empty, so the return on raising the cap is sublinear — 12288 → 14336
  bought 9 points. A corpus whose ratio is below 1 makes 12288 oversized, the harmless direction),
  `loss_weight_floor_tokens`
  (64: it caps a conversation's total weight at `min(n_supervised/64, 1)`, so gradient share is
  **not** the conversation share for a source with short answers — a 5-token SQuAD answer counts
  ~0.08 against a long continuation's 1.0, and the three shares to read are tokens, conversations
  and that product),
  and the refresh cadence (the table still trains, at `lr`).

Constraints:
- `moe_intermediate_size` sizes the routed and shared MoE experts only; defaults to `intermediate_size`.
- `mtp_num_extra_tokens <= num_mtp_tokens` (the dataset's separator budget); both come from the same
  value in `pretrain.py`.
- `vocab_size` and `hidden_size` divisible by `lm_head_factor` (and by `lm_head_factor * 2` on
  `hidden_size // 2` when MTP is on); `vocab_size <= 65536`. Asserted at construction.
- `num_ir_entries % ir_num_clusters == 0`; capacity `<= ir_dim` for the orthonormal value init.
  `ir_num_clusters: 0` is the exact full-table read a pre-reshape checkpoint loads as.
- **The checkpoint, not the yaml, decides shape and mode.** `utils.model_params_for_state_dict` reads
  `num_ir_entries`/`ir_dim` from `z_keys`, forces `ir_num_clusters=0` without centroids, and infers
  `loop_inject`, `evidence_port`, `ir_direct_read` from tensor presence. `utils.load_model_state`
  loads strictly except for `IR_TEMPERATURE_KEYS` and `NEUTRAL_LOOP_KEYS` (loop query biases,
  `evidence_loop_scale`, `evidence_gate_scale`), whose inits *are* the old behaviour. The two resume
  paths (`utils.load_checkpoint`, `sft.load_sft_checkpoint`) stay fully strict: there, absence means
  lost state. `evidence_encoder` is read from the yaml for every checkpoint (no parameters).
- Hardcoded: `NUM_DATA_WORKERS=4`, `LOG_INTERVAL=20` (pretrain) / `10` (sft), expert heads 16/4,
  `rope_theta`, `CE_CHUNK_SIZE=8192`, `ROUTER_NOISE_SCALE=0.3`, `MAX_CHUNKS_PER_SEGMENT=256`.
- `TinyMoETransformer.__init__` prints total/active params and FLOP/token (383.5M / 224.2M / ~502M
  at seq 4096, +118M per evidence token). Budget math keyed to it goes stale silently. FLOPs are
  three separately scaled components — `body_flops_per_token` (decoder once + MoE × `n_loops`, the
  IR table billed by its own `flops_per_token` rather than `2 × params`), `lm_head_flops_per_token`
  (per application, once per loop, 4x under chunk checkpointing) + `mtp_flops_per_token`, and
  `attn_flops_per_seqsq` (scales with `sum(seg²)`, accumulated on-device from `cu_seqlens`) — plus
  `evidence_encoder_flops_per_token`, deliberately not folded into `flops_per_token_fwd`.

## Model invariants

**Expert index layout** (one router over the pool; order matters everywhere):
`[ SelfAttention × A | CrossAttention × A | IR × I | MLP × M ]`, `first_mlp_index = 2A + I`.
`_num_attn_experts = num_attn_experts * 2`. No identity expert. Indices `>= first_mlp_index` are
remapped into `ParallelSparseMoELayer`'s local space; non-MLP slots become `(index 0, weight 0)`.

- **Non-MLP experts run unconditionally**, once per `forward_step`, cached across top-k slots
  (attention must see the whole sequence). Only MLP experts are sparse (grouped GEMM over sorted
  assignments). Their routing weights are folded into one `[B, S, first_mlp_index]` gate via
  `scatter_add_`, a mask multiply — never `mask.sum()`/boolean indexing (a device sync per expert).
- **`shared_mlp` + `shared_attn` seed the accumulator unconditionally every loop**, outside the router
  pool and the aux loss, sized by `moe_intermediate_size`, static row count so they run inside the
  outer `te.autocast`. `shared_evidence` joins them only when the port exists *and* evidence is attached.
- **`forward_step` returns an updated `hidden_states`**:
  `h = h + loop_scale[loop] * dropout(post_norm(output))`. `loop_scale` is `[n_loops]`, init
  `1/sqrt(n_loops)` (at the old 0.1 the whole block was a ~1.5% perturbation), **excluded from weight
  decay**, indices past `n_loops - 1` reuse the last entry. Distinct from the decoder's `layer_scalar`.
- **Routing is loop-conditioned**: `loop_router_bias(loop_enc[loop])`, zero-init, over a **sinusoidal
  encoding of the absolute loop index** (`[max_enc_loops, loop_enc_dim]` non-persistent buffer), not a
  learned table — so `forward(..., n_loops=N)` runs any depth without reshaping a weight. The same
  `loop_enc` feeds `ir_module.loop_query_bias` and `moe.evidence_query_bias`.
- **`loop_inject`** (zero-init `768 × 768`, inferred from the state dict) adds `inject(e)` of the
  block's input to what the router and experts *read*; the residual is untouched. Measured not to
  help (G2c); kept loadable.
- **Depth policy is the parameter-free convergence exit** (`_convergence_exit`, via
  `forward(..., converge_tol=, min_loops=)`, plumbed as `exit_check` into the MoE). Reads the **last
  position's readout** (not `‖Δh‖` — `loop_scale` still moves the state while the prediction is
  stationary); **asserted inference-only** (a short `hidden_states_all` breaks per-loop CE);
  **asserted exclusive with `kv_cache`** (an exited loop appends no K/V; `inference.py` turns the cache
  off under `--converge-tol`). `eval_calibration.py` prints per-transition agreement / `|Δ log p|`
  (~0.81 / 0.23 for 1→2, ~0.93 / 0.08 for 2→3 on the 16B checkpoints).
- `ParallelSparseMoELayer` runs under `te.autocast(enabled=False)` (NVFP4 needs row counts % 16);
  `torch.argsort(stable=True)` for determinism across the checkpoint recompute; `m_splits` via
  `.tolist()` is the one accepted host sync.
- **Aux loss** is on the router's softmax, normalized by `loops_run`; one `torch.topk` feeds it and the
  selection. It takes an optional `token_mask` (`None` is bit-identical), plumbed through
  `TinyMoETransformer.forward`; **both trainers now pass `input_ids != pad`**, so its value is not
  comparable to a run from before that (it used to move ~3x with row fill). Mean routed weight is flat by
  construction (top-2 renormalized); selection *fraction* is the signal.
- **Router noise is scaled by `ROUTER_NOISE_SCALE = 0.3`** (a ceiling on the initial level; still
  annealed to 0 over `noise_anneal_tokens`).
- `_ExpertTracking` and `RetrievalEntropyTracking` guard against checkpoint-recompute double counting
  (`begin_forward(expected_updates)`) and sample every 8th forward; no host sync in `update()`,
  `get_stats()` at log cadence only. IR entropy is **`E / ln(width)`** where width is the softmax the
  module actually takes (`read_top_k` on the two stage path) — logged as `IR E/ln32: [...]` per loop,
  same units as `eval_stage0.py`; 1.0 = uniform.

**IR expert** ([information_retrieval.py](modules/model/information_retrieval.py),
[experts.py](modules/model/experts.py)). Query: norm → `down_proj` (768→384) → `+ loop_query_bias`
→ `F.normalize`. Reads the learned table and, when a batch carries `chunk_keys`, the external store
under **one softmax over the union** (two normalized reads summed would have no notion of which store
won). Then `g_proj` → `up_proj` → output stage.

- **Two stage read** (`ir_num_clusters > 0`): score 256 centroids, open the top `probe_clusters` (8),
  score their members exactly, global top `read_top_k` (32), softmax over those ÷ learned
  `log_temperature` ÷ `temperature_scale`. 2.58M FLOP/token/loop vs ~101M for an exact read.
  **Clusters are exactly equal in size** (asserted) so `cluster_members` is `[C, capacity]` and
  candidate scoring is one `bmm` — no ragged gather, no host sync. Query dispatch is capacity-limited
  MoE style; overflow goes to a **trash row** masked to `-1e4` (finite, so a fully-overflowed token
  gets a uniform read, not NaN). **No `torch.bincount` in the step path** (`_group_starts`).
- **Both sides unit-normalized in the forward**: keys (cosine) and values (`_value_table`; otherwise
  the read is whatever the rows drifted to — RMS ~0.003 for 16B tokens). Normalize the table, then
  gather, never the reverse. Read magnitude lands in `[1/√k, 1]` = the read's confidence.
- **`g_proj.weight` zero-init is the migration's neutrality zero** (bias-free, so the value stream is
  exactly zero); deliberately not on `y_values` (50M params that would have to travel 250x under decay).
  `∂L/∂y_values` is zero until `g_proj` moves, hence `|g_proj|rms` is logged every step.
- **`reset_values()` seeds each cluster with a rotated orthonormal set** (needs capacity `<= ir_dim`);
  call it after the partition is known (after `refresh_clusters` in the migration). Recycled entries get
  a fresh random unit direction, never a zero row.
- **`temperature_scale` IS a persistent buffer** written through by `set_temperature_scale` — the
  sharpness the anneal ends at is part of what the table means; as a float it reloaded at 1.0 and read
  20x flatter than it trained. **`entry_usage` / `query_reservoir` / `reservoir_ptr` are plain fp32
  attributes, not buffers** (an EMA would not survive bf16), lazily created, **training-only** so an
  eval pass cannot move the next refresh.
- **`refresh_clusters`** re-runs balanced spherical k-means warm-started, measures `_candidate_recall`
  on the reservoir (**real queries** — isotropic random ones score at chance; `test_ir_two_stage.py`
  checks the curve's shape), recycles entries that are both below `dead_quantile` *and* below
  `0.01 × mean(entry_usage)` (a cap, not a target). Warn below recall 0.9: raise `ir_probe_clusters`
  before blaming the anneal.
- **External store**: `key_adapter` / `value_adapter` are square (384→384) **orthogonal** rotations,
  not zero — a zero there teaches "ignore evidence" first, and the reader's `o_proj` already carries
  the neutrality; `exp(log_memory_scale)` puts the external logits on the table's scale. A
  `[tokens, chunks]` `visible` mask keeps documents on their own chunks (asserted on the **mass**,
  not the logits — an assertion downstream of the router can be silenced by the router).
  `last_memory_mass` (summed external share, detached, per loop in `memory_mass_by_loop`) is the
  G3b signal, computed for **every** token whether or not the router opened its gate;
  `last_memory_weights` keeps the per-chunk breakdown with gradient for `evidence_selection_loss`.
- **Output stage** (`ir_direct_read`, inferred from the state dict): each token's own read through a
  zero-init `direct_gate` (default). The original inner attention over every position's read
  prefix-averaged the document's reads (batch-mean replacement cost 0.0000 nats); kept loadable for
  the A/B. `kv_cache` is accepted and unused on the direct path.
- **The verdict** ([ir_reshape.md](docs/measurements/ir_reshape.md),
  [ir_sharpening.md](docs/measurements/ir_sharpening.md), [ir_scale_fix.md](docs/measurements/ir_scale_fix.md)):
  the table sharpens on loop 1 and the model does not use what it retrieves; three arms agree to four
  decimals. Anything that tries again has to move the ablation, not the entropy, and explain where
  content the trunk lacks would come from.

**Evidence port** ([evidence.py](modules/model/evidence.py), `transformer.build_evidence`,
`moe.shared_evidence`). Added by `migrate_evidence_port.py`; `evidence_port` inferred from the state dict.

- **No evidence attached → bit-identical to the model without the port** (`test_evidence_port.py`
  asserts equality, not a tolerance). Nothing may touch a code path a `None` evidence batch reaches.
  The reader's `o_proj` is zero, so attaching a corpus to a fresh port is neutral too.
- **Evidence is encoded by the model's own dense decoder, once per forward** (`_encode_evidence`),
  chunk-causal with its own `cu_seqlens` from `evidence_chunk_ids` (finer than the reader's
  per-document segments), positions restarting at 0 per chunk (`chunk_position_ids`), no LM head,
  cached across loops. `evidence_encoder: true | N | false`; `false` = raw `_moe_ple` embedding, kept
  only for the A/B. **Padding is not a chunk**: it carries a negative id and gets position 0 — numbering
  it like a chunk ran positions past the rotary cache into an unchecked gather (a silent device-side
  assert). `RotaryEmbedding.gather` now clamps.
- **One evidence segment per query segment, paired by position.** `evidence_cu_seqlens` counts per
  segment over `num_segments` from the query side, so a document that retrieved nothing contributes
  a **zero-length segment** rather than dropping out (flash returns exact zeros for it; the SDPA
  fallback agrees). A run-based construction would silently point every later document at the wrong
  evidence. `_segment_ids` accumulates into an oversized buffer for the same reason (a repeated or
  end-of-axis boundary broke the old scatter).
- **The reader** reads `step_input + evidence_query_bias(loop_enc[loop])` at every loop, bidirectional,
  scaled by `evidence_loop_scale[loop]` (one-init, so `loop_scale`'s `[0.63, 0.32, 0.11]` does not
  stunt a re-read), and its evidence states are gated per chunk by
  `1 + evidence_gate_scale * sigmoid(selector chunk mass)` before `k_proj`/`v_proj` — the `1 +`
  makes a zero-init scale *exactly* neutral (flash varlen takes no additive bias, so a multiplicative
  gate on the states is the coupling). The IR experts run before the shared seed so this loop's own
  selector output is what gates. No KV-cache slot: the reader recomputes over the fixed evidence
  states every step (correct; cached and uncached greedy decodes can diverge after a few tokens by
  reduction order — `--no-kv-cache` for exact reproducibility).
- **Nothing ragged leaves the dataloader**: `chunk_keys` is emitted `[B, C, 384]` with a `[B, C]` slot
  map and flattened in-thread by `evidence_from_batch` (accelerate truncates dim 0 to the batch size).
  `chunk_gold` and `condition_ids` ride along; a corpus without `.evgold`/`.cond` omits both from
  every batch (warned once). `chunk_segments` is supplied by the packing loop, never re-derived.
- **`evidence_selection_loss`** (BCE per visible chunk against the gold flag, renormalized to the
  token's visible chunks so it trains ranking, not the split; safe with zero or several gold chunks)
  **is wired**, through `LoopMixtureOfExperts.evidence_selection_term`: it pairs each IR module's
  `memory_weights_by_loop` with the recurrence's own `last_memory_visible`, averages over every
  (loop, IR expert) read with uniform weights, and `train_step` adds it at
  `TrainingConfig.evidence_selection_weight` (0.1). Absent — hence the identical objective — for any
  batch without evidence or without the corpus's gold flag. **It reads tensors the forward stashed,
  so it asserts gradient checkpointing is off**: a checkpointed segment recomputes them under
  `no_grad` and the term would silently train nothing. Logged as `selection:` in the train line and
  in `[eval]`.
- **`GroundednessHead` + `groundedness_loss`** (label "gold present AND answerable", from the corpus,
  never the model's own argmax — the deleted correctness head's failure) **are wired**, added by
  `migrate_groundedness_head.py` and inferred from the state dict. `TinyMoETransformer.groundedness_term`
  reads `moe.last_reader_output` (the ungated, unscaled read) at the positions
  `pretrain.answer_start_positions` picks — the last prompt token of each supervised span, which is
  where `eval_abstention.py --evidence-port` reads the mass. `gold_present` comes from
  `last_memory_visible @ chunk_gold`; `answerable` from the corpus's `.ans` sidecar and **not** from
  the condition (a natively unanswerable SQuAD row is built under `gold` and still abstains).
  Weight: `TrainingConfig.groundedness_weight` (0.1). **Its weight gradient is exactly zero until
  the reader's `o_proj` leaves zero** — the head reads a zero vector on a fresh port, so only its
  bias (the base rate) can move; an early falling `grounded:` is that, not grounding. The held-out
  AUROC in `[eval]` is what separates them. Asserted in `test_evidence_selector.py`.
- `SQUAD_INSTRUCTION` stays verbatim for teacher-forced comparability; evidence mode has its own
  instruction constant. `is_fresh_loop_param` names every port/loop tensor that trains at `fresh_lr`.

**Removed heads (Phase 0).** `correct_proj` (BCE against `lm_head`'s own argmax → learned to
reproduce `p_max`; lost on ECE and AUROC) and `halt_proj` + the ponder subsystem (`p_halt` saturated at
~0.78, 0.92 on the final loop; 11 λ cuts did nothing; `loop_scale` grew to `[1.73, 1.81, 1.32]` to
compensate). `migrate_phase0.py` folds the gate's **measured** per-loop mean into `loop_scale`
(`new[k] = old[k] * mean_k(1 - p_halt)`; measured, because the log only recorded the all-loop
average and hid a strong decreasing trend):

| | `sft_final` | `phase2_final` |
|---|---|---|
| mean `(1 - p_halt)` | `[0.290, 0.134, 0.084]` | `[0.367, 0.179, 0.074]` |
| folded `loop_scale` | `[0.501, 0.242, 0.115]` | `[0.637, 0.326, 0.098]` |

So a migrated `loop_scale` far below `1/sqrt(n_loops)` is correct. Gate P0 passed on both (CE
3.7564→3.7604 / 3.3720→3.3843). `load_checkpoint` returns a **6-tuple** (was 7) and names a
migrated checkpoint if you try to resume from one. Any call site unpacking four forward values is
pre-Phase-0. **`p_max` is the confidence signal everywhere**, computed as `1 / Σ exp(l_j − l_max)`
(avoids two ~2GB fp32 transients per chunk); anything replacing it has to *add* information.

**Gradient checkpointing**: `from transformer_engine.pytorch import checkpoint`, never
`torch.utils.checkpoint` (breaks FP8/NVFP4). `set_checkpointing(stage_level, sub_level)`; both **off**.

**Training-mode forward returns hidden states, not logits** (`return_hidden=True` +
`delayed_mtp_loss(True)`): `compute_mtp_loss` applies the LM head inside a chunked, checkpointed CE.
New call sites must pass `main_lm_head=` or silently double the activation peak. **Per-loop CE**: the
returned "hidden states" are `self.norm` applied at *every* loop, `[loops_run, B, S, H]`; `loop_ce_weights`
required (length-checked) whenever `main_lm_head` is set; `loss_ce` is the final loop's raw CE.
Ascending weights do not guarantee descending per-loop CE deep into overfit (`test_per_loop_ce.py`
samples early for that reason).

**Forward return arity.** `return_aux_loss=True` → `(x, aux_loss)`, plus `extra_token_outputs` third
when MTP is on. **`skip_mtp=True` drops it and skips the head** (bit-identical logits,
`test_mtp_skip.py`); every eval and the non-drafting inference path pass it. `evidence=` is the
optional `EvidenceBatch`.

**Document packing**: the dataset emits `document_ids [B, S]`; the trainer builds `cu_seqlens`
**in-thread** (`cu_seqlens_from_doc_ids`). Never put `cu_seqlens` in the batch dict (ragged; accelerate
truncates dim 0). `max_seqlen` is passed as `S`, not the true max (avoids an `.item()`).

**Token counting**: `TokenTracker` accumulates on-device, drains on `sync()` at log/checkpoint cadence;
`.get_count()` is sync-free; assign `.num_tokens` to restore. No `.item()` in the step path.

**KV cache**: one `LayerKVCache` per decoder layer, plus per `(loop, non-MLP expert)` and per
`(loop, shared_attn)` — the block is the same weights re-applied, so each loop needs its own slot.
Every `forward()` defaults `kv_cache=None` (exact training path). Single-sequence only (`cu_seqlens`
must be None), so `eval_abstention.py`'s batched left-padded decode does not use it.

**Tokenizer**: DeepSeek tokenizer, `pad_token_id == eos_token_id`, id 0 is BOS; `embed_tokens` has
**no `padding_idx`** (it froze BOS). Pruned 129280 → 65536 by `prune_vocab.py` so the corpus is
uint16; every entry point reads `utils.TOKENIZER_DIR` (`$TINY_LLM_TOKENIZER` overrides), fetched from
`utils.TOKENIZER_REPO` (`ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536`, public) by `fetch_tokenizer.py`.
The prune keeps all 1283 special/added tokens and the 256-byte alphabet unconditionally (no
`byte_fallback`), closes kept tokens under BPE merge dependency, renumbers kept old ids ascending
(so BOS stays 0 and pad==eos needs no special case; `id_remap.json`), and is gated on **fertility**
(last regression 0.15%, bound 3%, worst source wiki 0.37%), not round-trip identity.

## Data prep

**`scripts/prepare_data.py`** builds `phase1/phase2.bin/.idx` from the seven-source Hub mix, meant for
the rented, interruptible box. **`data/prepared/` on the dev box is an outdated local stand-in**:
absolute numbers against it are not comparable to CONCLUSION.md; before/after deltas on one slice are.

- One shard in flight per source: download → tokenize → append → delete. Peak disk ≈ final bin size.
- Sources interleaved document-by-document by smooth weighted round-robin (the dataset reads
  sequentially, so the mix ratio must be baked into on-disk order); per-phase weights renormalized.
- Checkpointed every `--checkpoint-docs` (2000): `bin`/`idx` fsynced, `_prepare_state_{phase}.json`
  sidecar; on restart `truncate_to_state` trims back before reopening. **State advances only on a
  committed document, never at pick time** (a real bug `test_prepare_data.py` caught). Two separate
  runs do not reproduce the interleave order, but each source's sequence is gap- and repeat-free.
- Only `nvidia/Nemotron-CC-Math-v1` is gated (`HF_TOKEN` + accepted request; fails with a hint on
  401/403). The code source is `common-pile/stackv2_edu_filtered` (NVIDIA's code gate rejected
  independent accounts). Text column auto-detected (`pick_text_column`), failing loudly otherwise.
- smoltalk2 `messages` rendered as `"role: content"` (non-reasoning `_no_think` splits only); every
  consumed document's `sha1(text)[:16]` goes into `manifest.json`'s `smoltalk2_holdout_hashes`.
- Tokenization uses the fast tokenizer's Rust threads, not a `ProcessPoolExecutor` (fork deadlock risk).
- `--phases ir` builds the IR sharpening corpus **locally** with its own weight column; **never omit
  `--manifest-key`**.

**`scripts/prepare_evidence_data.py`** (local): five conditions — `gold`, `mixed` (gold among 3
distractors), `many` (16–32 distractors, QA only, 8% of the QA mix), `distractors` (abstain), `none`
(abstain) — plus ≥20% replay; SQuAD v2 (a natively unanswerable row abstains under **every**
condition, incl. `gold`), HotpotQA distractor setting, web text with a fixed 384-token held-out span
(long documents **windowed**, not dropped), smoltalk2 replay. **The passage leaves the prompt.**
Writes `.ev/.evidx/.evchunk/.evkey/.evkeyidx` plus `.evgold` (per chunk), `.cond` and `.ans` (per
document); `--target-tokens` counts **prompt** tokens (evidence reported separately). Shard order is
shuffled from `(seed, source)` — never from build progress — and the smoltalk2 holdout is imported
from `prepare_sft_data.py`; `--ignore-holdout` overrides. Distractors are written per occurrence.
Prints each source's conversation share next to its token share and its `too_long` count — read both.

- **`.ans` is not derivable from `.cond`.** A natively unanswerable SQuAD row is built under `gold`
  like any other and still takes an abstention target; separating those is the whole job of the
  groundedness readout. Written where `apply_condition` decides it. An `lm`/replay row is always
  answerable (its target is its continuation), and still labels 0 for groundedness when no gold
  chunk was retrieved.
- **The QA sources are exhausted by one pass** (~10.7M prompt tokens for both), so
  `EvidenceSource.repeat` gives them more, bounded by `--max-source-epochs` (4) and by a guard that
  stops any source whose whole pass produced no tokens. Each pass redraws conditions and
  distractors; the answer text does repeat.
- **`--max-evidence-tokens` (1536 by default) silently deletes the `many` condition**: 16–32
  distractors is 2048–4096 evidence tokens, and an over-budget row is dropped (never truncated —
  truncation would drop whichever chunk landed last, which is the gold one a third of the time).
  The first real build produced 2 `many` rows out of 304,662 and threw away ~18k QA rows this way.
  Pass 4608.
- **The evidence-to-prompt ratio is a composition statistic**, not a property of the rows: 7.6 for
  SQuAD, 6.3 for HotpotQA, 0.45 for web text, 0 for replay. The 2.93 that sized
  `EvidenceConfig.max_evidence_tokens` came from a QA-heavy smoke corpus; a one-pass real corpus
  reads 0.96 because web text and replay dominate its tokens, and the four-pass rebuild reads
  **3.75** because repeating the QA sources moves the composition back. So the cap has to be
  re-read against every corpus rather than carried over.

## Training loop notes ([scripts/pretrain.py](scripts/pretrain.py))

- Order: tokenizer → `Dataset` → `DataLoader(batch_size=None, num_workers=4)` → model →
  optimizer/scheduler → **checkpoint resume on the unwrapped model** → `dry_run` (synthetic packed
  batch, asserts finite loss, restores the token counter) → `Accelerator.prepare`.
- LR: linear warmup → cosine to `0.1 * lr`, **re-anchored by tokens on resume**
  (`resume_token_count // tokens_per_step`). Router noise anneals 1→0 over `noise_anneal_tokens`.
- Main CE = weighted sum of per-loop CE; non-final loops token-subsampled by `loop_ce_subsample`
  (0.25, unbiased). **Stochastic depth** (`loop_count_sampling` 0.3): random depth in `1..n_loops-1`,
  `loop_ce_weights_for(n)` truncates and rescales so the deepest loop run carries 1.0. **Log steps
  are pinned to full depth.** This is the entire training-time depth mechanism.
- **Two param groups** (`build_param_groups`): decay only `ndim >= 2`; norms/biases/gates excluded
  because their zero is degenerate (`loop_scale` → off; `layer_scalar` decay compounds ~0.5x over 8
  layers). **A parameter stepped through an fp32 master is in no optimizer group, so `zero_grad()`
  never clears its `.grad` — `train_step` clears it by hand, gated on `accelerator.sync_gradients`.**
  Without that the bf16 grads accumulate for the run, the clip norm grows unbounded and throttles
  every other parameter (a silent run-wide LR collapse behind a descending loss curve), and
  `sync_master_grads_` copies a run-length sum. Bites harder in `sft.py`, where every parameter is shadowed.
- **A checkpoint that exists but fails to load is never a fresh start**: "no file" warns, "files exist
  but none loads" raises. `find_resume_checkpoint` falls back one corrupt newest file; `verify_resume`
  bounds how far back that may silently go.
- `collect_metrics` gated on the log cadence; every host sync (loss `.item()`, token sync, tokens/sec,
  peak mem) throttled to `LOG_INTERVAL`. `compute_mtp_loss(..., return_metrics=True)` returns
  on-device `per_loop_ce`, `p_max`, `top1_acc` from already-materialized chunks.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, `moe` goes through
  `accelerator.unwrap_model(model)`.
- `KeyboardInterrupt`'s `input()` is gated on `sys.stdin.isatty()`; vast preemption sends SIGTERM.
- **Training stops at the PHASE's token target** (`n_tokens >= phase_target`, checked in the
  `LOG_INTERVAL` block); `num_epochs: 1` is a safety net; `target_tokens` stays the **combined** budget
  so phase 2 continues the cosine.

## Checkpoints & resume

`ckpts/training/checkpoint_{phase}_tok{N}M_loss{L}.pt` rolling + one `checkpoint_{phase}_final.pt`
per phase (`modules/runtime/checkpoints.rolling_name`/`final_name`); token count is the only figure
that says where a checkpoint sits.

- **"Latest" = newest mtime that actually LOADS** (`find_resume_checkpoint` walks newest-first, skips
  one that raises, raises only when all fail). Payload: model/optimizer/scheduler states, `token_count`,
  `losses`, `phase`, **`global_offset`** — a doc index into the flat stream; sharding is
  `doc_idx % num_workers`, so `min(seen) + NUM_DATA_WORKERS` (`snapshot_global_offset`) resumes without
  skipping any worker's documents; resume is document-granular (`test_dataset_resume.py`).
- `load_checkpoint` extras use `.get(..., default)`; add new fields to `save_checkpoint` with defaults.
- **Writes are atomic** (`.tmp`, fsync, `os.replace`); stale `.pt.tmp` swept by `cleanup_stale_files`.
- **Retention deletes only when BOTH outside `keep_local_checkpoints` AND confirmed uploaded**
  (`prune_checkpoints`, `HFSync.is_uploaded`). Sustained upload failure fills the disk on purpose.
  `*_final.pt` is exempt.
- **Cadence is in TOKENS** (`checkpoint_every_tokens`, in the `LOG_INTERVAL` block). The old
  `step % 1500` counted micro steps: ~608 checkpoints, ~1.2TB, on a 120GB disk.

## Dataset ([modules/data/dataset.py](modules/data/dataset.py))

`IterableDataset` yielding **fully assembled batches** from `{phase}.bin` (uint16) + `{phase}.idx`
(uint64 offsets + trailing `len(bin)`), memmapped **inside** `_batch_iterator` (a long-lived memmap
across worker restarts leaks). Read **once, in on-disk order, no shuffling** (the mix is baked in).
Packing: concatenate to `max_length`, split across boundaries, each document followed by
`EOS + (num_mtp_tokens - 1)` pads; trailing padding = length-1 segments; labels `-100` except document
interiors + terminating EOS; BOS prepended if absent (bin stores BOS-less content). Batches carry
`doc_idx / worker_id` as `[B]` tensors for accelerate.

## SFT / post-training ([scripts/sft.py](scripts/sft.py))

Local, single GPU, BF16; same stop contract as pretraining (SIGTERM → exit 20, STOP → 10, SIGUSR1 →
save and continue), no phase supervisor. Rented-box variant in [docs/runbook.md](docs/runbook.md) §10.

- **Reuses `pretrain.train_step` verbatim** (one copy of the objective); prompt masking needs nothing
  there (`-100` labels, every term honours `ignore_index`). Every loss weight lives in `TrainingConfig`.
- **Four profiles** (`--repair`, `--ir`, `--evidence`, default SFT) swap the config class, the phase
  label and the checkpoint dir (`ckpts/{sft,repair,ir,evidence}`; `--run-name` suffixes) and nothing
  else. `load_sft_checkpoint` refuses another phase's checkpoint, so no run adopts another's AdamW
  moments; `-c` seeds are initializers (`load_pretrained_weights`, optimizer state dropped) and must be
  migrated (`*_phase0.pt` or later; a pre-Phase-0 file still has the old heads).
- **Every parameter gets an fp32 master** (`build_sft_param_groups`): at `lr=3e-5` a bf16 weight near
  init std has an ulp ~3x the AdamW step, so it would never move. Masters are not checkpointed
  (reseeded from bf16 on resume, sub-ulp progress lost, bounded by one interval).
- **Fresh-LR group** (`fresh_lr`, `--ir` and `--evidence`), **per-profile predicate**: `--ir` →
  `is_rebuilt_ir_param or is_fresh_loop_param`; `--evidence` → `is_fresh_loop_param` **alone** (its
  table carries a full sharpening run and would be wrecked at 3e-4). Membership logged by name.
  Cluster refresh runs in any profile whose config sets `cluster_refresh_tokens`; the temperature
  anneal is `--ir` only, on micro-step progress.
- **Per-conversation loss weighting** (`SFTConfig` off — it describes the run that produced the
  existing SFT checkpoint; `RepairConfig`/`EvidenceConfig` on; evidence floored at 1/64 per token).
  `SFTDataset` always emits `loss_weights [B, S] = 1/(supervised tokens in the conversation)`; not
  passing them is bit-for-bit the per-token mean. Aligned with `targets` (each term shifts them as it
  shifts its labels; the denominator is masked by `labels != -100`). **`p_max`/`top1_acc` stay
  unweighted.** Under conversation weighting a source's influence is its share of *conversations*
  (a SQuAD row ~206 tokens vs HotpotQA ~1340): retune against the realized counts prep prints.
- **The global token counter is continued** (router noise anneal reads it); progress is
  `token_count - start_token_count`; the SFT payload is a strict superset of pretraining's.
- **`SFTDataset`** differs in exactly three forced ways: `{split}.mask`; **conversations never split
  across rows** (over-long dropped, ~5–10% padding cost); per-epoch `(seed, epoch)` permutation
  (`global_offset` is a position in it; `sft.seed` is checkpointed and a mid-run change is a hard
  error). Separator slots are all pads (the supervised EOS is already there). **`EvidenceDataset`** adds
  the evidence stream, packs to `max_length - 1` so evidence padding always has a trailing pad
  segment, caps a row at `max_evidence_tokens`, and logs realized fill / rows closed by the evidence
  budget (the first smoke run collapsed to 221 tok/s on ~96%-padding rows, invisible in the loss).
- **Chat template control tokens are resolved from the tokenizer and asserted** (`ChatTemplate._control_id`):
  DeepSeek's `<｜User｜>` / `<｜Assistant｜>` / `<｜begin▁sys｜>` / `<｜end▁sys｜>` with explicit
  FULLWIDTH VERTICAL LINE / LOWER ONE EIGHTH BLOCK escapes. Only the assistant's text + EOS is supervised.
  Conversations with roles outside {system, user, assistant} are dropped whole (tool turns).
- `prepare_sft_data.py` honours the smoltalk2 holdout by **importing** `render_pretrain_chat`
  (a drifted reimplementation would exclude nothing); refuses an empty holdout unless
  `--ignore-holdout`. Only train splits are consumed (SQuAD v2 val / GSM8K test are eval sets).
- **`eval_abstention.py` is the acceptance metric and it generates.** Batched decode is left-padded
  with the pad run as its own segment (exact isolation through attention — `test_pad_isolation.py`
  asserts bit-identity on the decoder; the MoE's `m_splits` tiling makes numbers comparable only at
  fixed `--batch-size`). No KV cache (quadratic; `--max-new-tokens 32`, `--max-examples`). Two
  calibration numbers: answer-level (`p_max` vs correctness, the claim) and token-level teacher-forced
  (readable on the pretrained baseline, conservative). **The token-level number passed while the
  behavioural one failed on the real run — read the generated-answers block first.** ECE/AUROC are
  imported from `eval_calibration.py`; `eval_probe.py` and `evidence_ceiling_probe.py` reuse this
  script's loader/renderer/slice by import. **`--evidence-port`**: passage out of the prompt, into the
  port; scores any subset of conditions over one slice; G3 is the **answer-span CE with the real answer
  as target under every condition** (the condition-target gap compares "answer" vs "refusal" and reads
  −3.25 on a zero reader); G3b is the external-mass AUROC at the last prompt position, per loop.
- **Abstention phrasings are a closed set** (`abstention.py`): `ABSTENTIONS_PASSAGE` (5, what the eval
  forces) ⊂ `ABSTENTIONS_PASSAGE_TRAIN` (15, what corpora draw from); `is_abstention` matches the
  union. The SFT run collapsed onto `"The passage doesn't say."` (78.4% of answerables); repair took
  false abstention to 16.1%, recall 0.81 → 0.22, precision pinned ~0.578 at any ratio
  ([abstention_repair.md](docs/measurements/abstention_repair.md)).
- `estimate_packed_rows` replays the packing rule (and the evidence budget) to anchor the cosine.

## Run lifecycle (`modules/runtime/`, `scripts/run_training.py`)

- **`modules/runtime/` stays GPU-free** (imports `utils` for `logger` only); phase-reset logic lives in
  `resolve_resume_scope` for that reason.
- **Exit-code contract** (`control.py`): `0` complete, `10` user stop, `20` preempted, `30` resume
  verification failed; `run_training.py` restarts on anything but `TERMINAL_CODES = (10, 30)`.
  Changing a number means both sides plus the runbook table.
- **Signal handlers set a flag and nothing else**; `RunControl.poll()` reads flags and stats `STOP` at
  the `LOG_INTERVAL` cadence (~3s). A stale `STOP` is cleared at startup.
- **Upload failures never propagate** (`HFSync` retries 3x, tripling backoff, then logs; the file stays
  unmarked so retention keeps it). Every successful Hub delete also `super_squash_history` (throttled
  to `squash_min_interval` 1800s) — acceptable only because `temp-train` is a scratch mirror.
  `HFSync.drain()` waits for the in-flight job (`_busy`), called in `pretrain()`'s `finally`.
- **`run_training.py` starts at `checkpoints.resume_phase_index(...)`**, not `PHASE_ORDER[0]` (a
  reclaim relaunches a fresh supervisor against a disk that may hold a phase-2 checkpoint; the old loop
  overwrote `checkpoint_phase1_final.pt` with phase-2 weights).
- **`setup.sh` fails hard with no HF token and a non-empty resolved upload repo** (`onstart.sh` runs
  `set -euo pipefail`). `hf_upload_repo: ""` is the legitimate opt-out.
- **Crossing a phase resets `global_offset`/epoch/step but PRESERVES `token_count`**
  (`resolve_resume_scope`): phase 1's ~23M-doc offset in phase 2's ~4M-doc corpus yields zero batches
  and a "successful" run; the token count anchors the cosine.
- **`verify_resume`**: `run_state.json` (`{phase, token_count, checkpoint}`, atomic) records where the
  last process got to; resuming more than `2 * checkpoint_every_tokens` behind aborts with 30.
- Graphs are fixed filenames, overwritten at checkpoint cadence.

## Inference

`scripts/inference.py` is the reference path; `gradio_app.py` imports `stream_generate`. KV-cached by
default (`--no-kv-cache` for the full-prefix reference; exactness is structural — every attention call
is causal). `--num-mtp-tokens` drafts self-speculatively with **no rejection sampling** (default 0).
`--converge-tol` turns the cache off. `--evidence FILE` (JSON list or one chunk per line) builds one
`EvidenceBatch` for the session; composes with the cache structurally but can diverge from the
uncached decode after a few tokens (see the port invariants). Streaming re-decodes the full sequence
each step and yields the new suffix.

## Conventions

- Comments are lowercase, explanatory, and justify *why* (especially sync avoidance, checkpoint
  recompute, accelerate's batch handling). Match that density; don't strip them.
- Google-style docstrings with an `Args:` block on the public modules.
- Config flows yaml → `config.py` → kwargs. Don't read `config.yaml` from a module under `modules/`.
- `utils.logger` is the logging channel; `print` only in the inference CLI and the eval report blocks.

## Git

Current branch `ir-train-build`; PRs target `prototype`. **Commit from Windows, not WSL** (WSL git
rewrites line endings on every tracked file it touches).

**Never run `git commit`. Stop before committing and hand back the commit message instead.**

**Commit messages are a single line. No body, no bullets, no `Co-Authored-By` trailer.** Style is
`feat:` / `fix:` / `docs:` / `chore:` / `merge:` plus a short description. No config/version labels;
plain unhyphenated phrasing ("construction time assertions").

**Never name a plan document, phase, gate or step in a commit message or a code comment.** Plans get
renumbered; write the reason instead ("the corpus builders delete shards from here"). Docs under
[docs/](docs/) are the exception. Older comments still carrying `PLAN.md Step N` are left alone
unless that line is being edited anyway.

`.gitignore` swallows `*.json`, `*.log`, `*.cmd`, `*.key`, `ckpts/`, `venv/`, `env_init`,
`data/prepared*`, `data/benchmarks`. `tests/` is tracked.

## Known rough edges

- `flash-attn` / `transformer-engine` in `requirements.txt` need CUDA builds matched to the GPU.
- `huggingface.key` sits in the repo root (gitignored).
- Convergence exit ↔ KV cache exclusivity is real plumbing, not a flag (NEXT.md 7a).
- `eval_abstention.py` has no KV cache (quadratic in answer length); the reader has no cache slot.
- The QA sources are finite and small (SQuAD v2 ~130k rows / 5.9M prompt tokens, HotpotQA ~90k /
  4.9M). A one-pass corpus puts QA at ~10% of tokens and ~18% of the gradient; `EvidenceSource.repeat`
  (QA only, capped by `--max-source-epochs`) is what closes that, at the cost of repeating answer text.
- The append-only evidence buffer and the "evidence still arriving" depth criterion are unbuilt.
- `eval_benchmarks.py` / `eval_calibration.py` / `eval_stage0.py` cannot attach evidence.
- Token counts can be inflated by tens of tokens per batch.

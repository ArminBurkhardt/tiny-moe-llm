# Run 2 corpus build sheet (rewritten 2026-10-27 from the code)

The corpus of the real run, built with the existing builders by `scripts/build_run2_corpus.py` on
MareNostrum 5. Decision of the user, 2026-10-26: the pilot recipe at scale. The old build item 4 of
[RUN2.md](RUN2.md) (per-window bge retrieval on plain text, the fact tagger and span weights, the
near-duplicate filter, the staircase visibility question) is not on the launch path. What is built is
the pilot's three merges (`pilot_warm_train`, `pilot_main_train`, `pilot_control_train`; Record bullet
"Pilot corpus" in CLAUDE.md, `ckpts/inject/build_pilot_bios.sh`, `merge_pilot.sh`, `merge_pilot2.sh`)
at the Run 2 token target, with `--support-gold-only` on the biography builds (RUN2 decision 3), the
written bridge chains beside the plain ones (R9), QA from `prepare_evidence_data.py` at its finite
passes and the filler from `prepare_data.py`.

Every number below is `python scripts/build_run2_corpus.py plan` (`derive`, `disk_schedule`,
`hours_plan`), unless it says "measured" (a pilot log) or "estimate". `plan` takes the same flags as
`build` (`--scale`, `--chain-tokens`, `--chain-share`, `--qa-target`, ...) and exits 2 when the disk
plan is over budget (peak 3.4 TB, standing 2.0 TB). Tests: `tests/test_build_run2_corpus.py`.

## 1. The numbers

Unit: prompt tokens of the merged split, as `merge_evidence_splits.py --target-tokens` counts them.
The evidence trainer runs whole epochs and ignores `target_tokens`, so the split is sized as the
token target over the epoch count.

| quantity | value | derivation |
|---|---|---|
| token target / epochs | 800B / 3 | `TOKEN_TARGET`, `EPOCHS` |
| tokens per split T | 266,666,666,667 | `TOKEN_TARGET / EPOCHS`; the main split, each epoch a fresh permutation |
| warm-up read | at most 28B | the warm split is the first 28B of the main order (`merge_warm`, prefix mode) |
| control | 10B | the first 10B of the main order, no evidence files (`merge_control`) |
| bios slice | 262,339,919,341 (share 0.9838) | T minus qa minus chains; the injection build's `--target-tokens`; taken whole |
| qa slice | 1,526,747,326 (0.0057) | sum of the arms below |
| chains, plain / bridge | 1.4B / 1.4B (0.00525 each) | absolute, `--chain-tokens 2.8e9` (a chain prompt token carries about 177 bytes of chunk keys, so a share of the split would grow the disk with the corpus); `--chain-share` replaces it |
| chain questions | 28,090,116 plain, 19,491,610 bridge | 52.33 and 75.41 tokens per question (measured on the pilot build and the smoke), times 1.05 |
| filler built | phase 1 259,963,318,534; phase 2 5,000,000,000; ir 2,500,000,000 | phase 1 is the bios need times 1.01 minus phase 2; phase 2 and ir are assumptions bounded by the smoltalk2 supply; the injection build reads phase 1 and 2 as filler, ir belongs to the QA web arm |
| persons, renderings | 10,800, 486,000 | the pilot's tiers at 3x the persons (`1000:300,100:1500,10:3000,1:6000`), so the leak read has about 1.7x lower sigma than the pilot's |
| validation windows | cutoff 5 per million, about 1,837 | the builder's hash gives whole millionths only |
| prefix builds | bios full 10.5B, bios warm 29.4B | token count of the split plus 5%, `--stop-after-tokens`: byte prefixes of a build |
| merge documents | 477,832,853 | 477.8M at 37k documents per second with evidence, 118k without |
| evidence | 11.98B tokens, ratio 0.045 per prompt token | the real split's ratio is about 0.045 (bios 90M, chains and QA nearly all), against RUN2's 1.49 |

QA arms (the builder stops an arm at `weight x target` or at its supply; `--qa-target 2.5e9` is more
than the sources hold, so the slice is the supply; SQuAD and HotpotQA at `--max-source-epochs 2`):

| arm | weight | realised, estimate | supply source |
|---|---|---|---|
| squad_v2 | 0.20 | 11.7M | 5.86M per pass (pilot state file) |
| hotpot_qa | 0.15 | 9.7M | 4.86M per pass |
| webtext | 0.40 | 1,000M | `ir` tokens times 0.402 prompt tokens per `ir` token |
| smoltalk2 | 0.12 | 300M | 1.8B in 24 files minus phase 2 and ir |
| ultrachat | 0.08 | 200M | 207k rows at 1,163 tokens |
| no_robots | 0.05 | 5.3M | pilot state |

Chat replay at RUN2's 20% is not reachable (the non-holdout chat left is about 0.5B); the share goes
to the bios slice. The tail curriculum is not built.

### Filler: sources, volume, repeat

Phase 1 is capped per source at its share of the Hub's supply (code_edu 95%, wikipedia 90%, the
shortfall to nemotron_math up to 65%, then fineweb). Need against supply, all three phases:

| source | need | files | supply (fineweb widened) |
|---|---|---|---|
| fineweb | 165.2B | 559 | 713B (2,410 files, 110 crawls); one crawl is 14.8B |
| dclm | 28.9B | 265 | 304B |
| finepdfs | 20.9B | 22 | 99B |
| code_edu | 25.5B | 91 | 26.9B (95% used) |
| nemotron_math | 20.9B | 30 | 32.2B (gated) |
| wikipedia | 4.5B | 37 | 5.0B (90% used) |
| smoltalk2 | 1.5B | 20 | 1.8B (binding for the QA replay arm) |

Phase 1 weights after the cap: fineweb 0.6315, dclm 0.1111, finepdfs 0.0778, code_edu 0.0923,
nemotron_math 0.0724, wikipedia 0.0150 (`--phase1-weights`, passed by the stage). The filler is
reachable with no repeat only with fineweb's prefix widened to all crawls (`wrap-prepare-data
--fineweb-prefix data/`, a patch in memory); phases 2 and ir skip the files an earlier phase used
(`files_used` of the manifest); `data_prep` holds phase 1 and 2 (the QA builder reads its
`smoltalk2_holdout_hashes`), `ir_prep` holds ir. **Filler repeat is checked**: after each biography
build the driver reads `common.filler_repeat_factor` of `inject_build.json` and fails the stage above
1.001 (a smoke with `--link-filler-from` or `--allow-filler-repeat` skips it). `load_filler_windows`
holds every training window as a Python tuple: size the node for tens of GB of RAM in the biography
builds (estimate).

## 2. Stages

`build --stage NAME[,NAME]` (default all), in this order. Each stage runs its commands through bash
(`cd ROOT && source ENV_INIT && ...`, `TINY_LLM_ROOT` and `TINY_LLM_ENV_INIT=/dev/null` on the
cluster), tees `DATA_ROOT/run2_logs/STAGE.log`, checks its outputs and writes `run2_done/STAGE.json`.

| stage | tool | writes | rerun after a kill |
|---|---|---|---|
| filler | `prepare_data.py` x3 through `wrap-prepare-data` (phase1, phase2, ir) | `prepared/{phase1,phase2,ir}.{bin,idx}`, `manifest.json` entries | resumes from its state files; `--force` on a pruned filler now fails (the state records more bytes than the bins hold) |
| qa | `prepare_evidence_data.py` | `prepared/evidence_run2_train.*` | resumes |
| bios_main | `prepare_injection_data.py`, arm retrieval, swap 0.15, gold-only support | `prepared_run2/inject_retrieval_s15a50g1_train.*`, `inject_val.*`, `index/run2_bios` | resumes (state file; a kill before the first state write is rebuilt) |
| val_clean | `prepare_clean_val.py` | `inject_val_clean.*` | partial outputs deleted, rerun |
| bios_full | injection builder, arm full | `inject_full_s15a50g1_train.{bin,idx,mask}`, a 10.5B prefix | resumes |
| bios_warm | injection builder, arm retrieval, swap 1.0 | `inject_retrieval_s100a50g1_train.*`, a 29.4B prefix | resumes |
| prune_filler | `rm` | deletes the filler bins | idempotent |
| chains_plain | `prepare_chain_data.py` | `prepared/chains_run2_train.*` | resumes |
| chains_bridge | `wrap-chains --hops 4`, `--bridge` | `prepared/chains_bridge_run2_train.*` | resumes |
| qa_shuffle | `merge_evidence_splits.py` over the qa build alone | `prepared_run2/run2_qa_shuf_train.*` | resumes (merge state) |
| order | `build_run2_corpus.py order`, numpy, about 9 GB | `run2_order.slice`, `run2_order.json` | partial deleted, rerun |
| merge_main | `merge_evidence_splits.py --order-prefix-from run2_order.slice`, whole order | `run2_main_train.*`, `.slice`, `.merge.json` | resumes (merge state, `--overwrite`) |
| merge_warm | the same, prefix of main to 28B | `run2_warm_train.*`, then `run2_main_train.order.json` (`epoch0_head_docs`) | resumes |
| merge_control | the same, `--no-evidence`, prefix to 10B | `run2_control_train.{bin,idx,mask,slice,merge.json}` | resumes |
| rows | `sft.py --evidence --estimate-rows-only`, CPU | `run2_{main,control}_train.rows_estimate.json` (entries for 1, 2, 4, 8, 16 ranks) | partial deleted, rerun |
| prune_sources | `rm` | deletes the bios, qa (build and shuffled copy) and chain train builds | idempotent |
| heldout_qa | `prepare_evidence_data.py --heldout` | `evidence_dev.*`, `evidence_fixed.*` | partial deleted, rerun |
| heldout_chains | `prepare_chain_data.py --prefix chains` | `chains_{eval,heldout_tmpl,hop4}.*` (same seed as the pilot) | partial deleted, rerun |
| store_openqa, store_edit | `build_store.py` | `index/openqa`, `index/openqa_edit` | partial deleted, rerun |

Rules the driver enforces:

- A stage with a marker and its outputs (or pruned by a later prune stage) is skipped; `--force`
  reruns. Its post checks still run on a skip (the md5 and prefix checks of the merges, the filler
  repeat check); an inbox directory deleted after the filler is fine on that path.
- `run2_done/STAGE.lock`: a non-blocking `flock` per stage; a second driver process reaching a held
  stage fails it at once instead of running the same merge beside the first.
- SIGTERM is forwarded to the child's process group, the child is waited for, no marker is written,
  exit 20; rerun the same command. A stop that lands after the work is done is checked and marked first.
- `DATA_ROOT/run2_logs/last_child_output` is touched when any child prints a line (at most once per
  10 s) and at every stage start and end, in `--parallel` mode for every child; the driver's 300 s
  heartbeat does not touch it. The corpus job's stall check reads its mtime (`STALL_MIN`, default 120).
- Exit codes of `build`: 0 done, 1 a stage failed, 2 refused before running (disk plan over budget,
  filler preflight: a `manifest.json` that holds `data_prep.phase1` this root did not write; this
  root's own `run2_data_root` stamp, a filler marker, or its `_prepare_state_phase1.json` is proof),
  20 interrupted.

The three merges are one document stream. `order` interleaves the four slices in proportion to their
size, storage order inside a slice (document k of a slice of n sorts at (k + 0.5) / n). The merge
tool's prefix mode takes slice i's k-th document in storage order, so a slice must already be in
random order: the biography layout and the chain generator's seeded questions are, **the QA builder's
rows are not** (smooth weighted round robin over its arms: SQuAD, HotpotQA and no_robots run out in
about the first 30% of the build, so the warm prefix and the control would carry about 3x the run's
share of extractive QA). `qa_shuffle` therefore rewrites the QA build as a one-slice merge (share 1,
every document, the tool's seeded permutation of seed, label and count; seed 42) and every later
stage reads that copy; `run2_order.json` records it under `permuted_slices` (seed, md5, documents).
The warm and control splits are byte prefixes in slice order of the main merge, checked by the driver
(md5 of the `.slice` prefix and the non-biography documents); the biography documents differ in length
between the forms (swaps), so `md5_by_slice` differs between the splits (each is the md5 of its own prefix).

## 3. Disk and time (`plan`, estimates from the pilot rates)

| stage | writes | deletes | running |
|---|---|---|---|
| filler | 537.6 GB (+30 GB scratch) | | 537.6 GB |
| qa | 18.8 GB | | 556.4 GB |
| bios_main | 800.4 GB | | 1.36 TB |
| bios_full / bios_warm | 31.6 / 89.6 GB | | 1.48 TB |
| prune_filler | | 537.6 GB | 940.7 GB |
| chains_plain / chains_bridge | 315.2 / 239.1 GB | | 1.50 TB |
| qa_shuffle | 18.8 GB | | 1.51 TB |
| merge_main | 1.37 TB | | 2.89 TB |
| merge_warm / merge_control | 144.2 / 30.2 GB | | 3.06 TB |
| prune_sources | | 1.51 TB | 1.55 TB |
| held out, stores | 1.8 GB | | 1.55 TB |

Peak 3.06 TB (budget 3.4 TB, the filler's inbox window counted when the mirror is on), standing
1.55 TB (budget 2.0 TB). The cause of the size is the chains (12 chunk keys of 768 bytes per 52 token
question, 177 bytes per prompt token against 0.8 for the bios slice) and the injection builders'
`.mask` and `.factspan` beside every token.

Hours, one process each: bios_main 35.2, bios_full 1.4, bios_warm 3.9, chains_plain 12.3,
chains_bridge 12.3, qa_shuffle under 0.1, order 0.3, merge_main 3.6, merge_warm 0.4, merge_control
under 0.1; filler, qa and val_clean unmeasured. Every stage over 60 hours must be resumable (a 72 h
job loses a non-resumable one); `plan` flags it. Independent stages can share a node
(`--parallel chains_plain,chains_bridge,bios_main`; the driver refuses two injection builds in one
group).

## 4. The cluster: no internet on any node

No MareNostrum 5 node, login nodes included, reaches the Hub, so the filler reads its shards through
the mirror (`scripts/mirror_sources.py`, demand driven) and the other Hub reads come from a staged
cache. Recipe in `scripts/slurm/README.md` ("The filler through the mirror"); in short:

1. On the PC, once: `python scripts/mirror_sources.py list --out listing.json` (the plan flags as for
   the build; 444 GB in 1,127 files at the full plan, limit 2 TB) and `python scripts/mirror_sources.py
   stage-small --hf-home DIR` (tokenizer, bge, SQuAD, HotpotQA, smoltalk2, ultrachat, no_robots, NQ,
   TriviaQA, and the 13 benchmark task datasets of `eval_benchmarks.py` for the per save suite); copy `DIR` to `$HF_HOME` and `listing.json` to `$DATA_ROOT/listing.json` on the cluster.
   `python scripts/build_run2_corpus.py check-offline --offline-stage $HF_HOME` lists what a stage lacks.
2. Cluster, the filler job (a whole node, 72 h jobs chained, resumable):
   `STAGES=filler CORPUS_ARGS="--mirror-inbox $DATA_ROOT/inbox --mirror-listing $DATA_ROOT/listing.json --offline-stage $HF_HOME" bash scripts/slurm/submit_chain.sh CORPUS 3`
3. PC, at the same time: `python scripts/mirror_sources.py push --listing listing.json --remote
   USER@transfer1.bsc.es:$DATA_ROOT/inbox --window-gb 120 --transfer-limit-gb 2000`. The inbox holds at
   most the window; the file the consumer waits on is sent even when later files of other sources fill
   the window. The push ends when the driver drops `.filler_done` into the inbox after the filler;
   delete the inbox afterwards (later driver runs, `--stage all` included, do not mind).
4. Every other stage group, in the order of section 2: `STAGES=qa,bios_main,... bash
   scripts/slurm/submit_chain.sh CORPUS 3` with `CORPUS_ARGS="--offline-stage $HF_HOME"` where the stage
   reads the Hub (qa, the chains' embedder, held out, stores); the Slurm side (stall watchdog, locks,
   chain ids) is `scripts/slurm/corpus.sbatch` and its README. The CPU-only stages (the merges, `rows`)
   may use a CPU partition.
5. The `rows` stage (CPU) must have run before any training job: the trainer exits 2 without a row
   estimate for its rank count (entries for 1, 2, 4, 8 and 16 ranks are written).

On a machine with internet (local smoke, the earlier rented box) none of this applies: `python
scripts/build_run2_corpus.py build --data-root DIR --device cuda` with `HF_TOKEN` for nemotron_math
and no `manifest.json` in the repo root. `--dry-run` prints every command and runs nothing.

## 5. What the builders lack, and how the driver works around it

| missing | workaround |
|---|---|
| `prepare_data.py`: a source's file prefix, skipping an earlier phase's files, a mirror | `wrap-prepare-data` patches the imported module in memory (`--fineweb-prefix`, `--exclude-files-of`, `--phase1-weights`, `--mirror-inbox`) |
| `prepare_chain_data.py`: the training hop mix 1..4 at equal weight | `wrap-chains --hops 4` sets `chains.sample_hops.__defaults__`; seed unchanged, so the held out compositions are the pilot's |
| `prepare_evidence_data.py` and `build_store.py` without internet | `wrap-hub --stage-dir` answers the Hub dataset calls from the staged directory |
| QA rows in storage order are not random | the `qa_shuffle` stage (section 2) |
| a separate size for the `full` arm | `--stop-after-tokens` builds a prefix; the control reads 10.5B of it |
| `merge_evidence_splits.py` has no sharding | none; 3.6 h at the estimated rate |

## 6. History: the local smoke of 2026-10-26 (the 100B plan, not the current one)

Run 22:06 to 22:30 at scale 0.0005 on the 100B plan with 10% chains, before the split became 93.3B and
then 266.7B and before `qa_shuffle` existed; the mechanics and the md5 check are unchanged by the
resize. Wall time 1,424 s, exit 0, 3.4 GB, stages filler (linked), bios_main, bios_warm, qa,
chains_plain, chains_bridge, merge_warm, merge_main, merge_control, heldout_qa, heldout_chains,
val_clean. `md5_by_slice` of warm and main agreed on qa, chains and bridge and differed on bios, as
they must (`ckpts/inject/run2_corpus_smoke/`). Read off it: the bridge at hops 1 to 4 builds through
`wrap-chains --hops 4` (75.41 tokens and 12.6 chunks per question, which the plan uses), and
`--support-gold-only` supports a span exactly when the gold card is in the buffer.

## 7. Local smoke of the final code, 2026-10-27

`python scripts/build_run2_corpus.py build --data-root /tmp/r2smoke_final --scale 0.0005 --link-filler-from data/prepared --heldout-max-questions 200 --chain-eval-per-hop 200 --store-max-questions 100`
on the final code (the plan at scale 0.0005: T 133M per split; log `/tmp/r2smoke_final.log`, outside the repo). Every stage ok, driver exit 0,
1,710 s (28.5 min), 2.8 GB (deleted afterwards). `qa_shuffle` 66 s, order 1 s, merges 65 to 69 s each, rows 207 s.
`run2_order.json` records `permuted_slices.qa`: split `run2_qa_shuf_train`, seed 42, 4,461 documents,
md5 `7db095139554f65da39574b25185451e` (equal to the main merge's qa `md5_by_slice`: the main merge takes the whole
permuted slice). `epoch0_head_docs` 68,748. `md5_by_slice`, warm / main / control (the warm and control splits are
prefixes of the main stream, so their md5 differ from the main's by construction; the driver checks the `.slice`
prefix and the non-biography documents instead):

| slice | warm | main | control |
|---|---|---|---|
| bios | `214d50dde974c2150fd78f866b13f378` | `6923079cbbcf91654989fae71839fd09` | `bfae38143e1555a0341f789c4d48353c` |
| qa | `2950ee50bc644b47c2a66757ddcbd495` | `7db095139554f65da39574b25185451e` | `30da3e247ba57ec359671ca703058cd9` |
| chains | `03e59667804498724abc8335b6e842f4` | `1b568a08ce8a8a9bf86a3d6d45adc18f` | `feb4616bb3971a0541c77b65e8984361` |
| bridge | `2ae17fc41cb5a60591b71723d6d3eafe` | `9da0b67de96b6f1967735f0200c2b4f5` | `23cb6a05ea0b9807c65a9f4c8a271627` |

Filler repeat 1.49x: the local stand-in filler is small, and `--link-filler-from` skips the repeat check.

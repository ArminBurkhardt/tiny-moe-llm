# Run 2 corpus build sheet (2026-10-26)

The corpus of the real run, built with the existing builders by `scripts/build_run2_corpus.py`.
Decision of the user, 2026-10-26: the pilot recipe at scale. The old build item 4 of
[RUN2.md](RUN2.md) (per-window bge retrieval on plain text, the fact tagger and span weights, the
near-duplicate filter, the staircase visibility question) is not on the launch path. What is built
is the pilot's three merges (`pilot_warm_train`, `pilot_main_train`, `pilot_control_train`, see the
Record bullet "Pilot corpus" in CLAUDE.md and `ckpts/inject/build_pilot_bios.sh`,
`merge_pilot.sh`, `merge_pilot2.sh`) scaled to the RUN2 token target, with `--support-gold-only` on
the biography builds (RUN2 decision 3), the written bridge chains beside the plain ones (R9), QA
from `prepare_evidence_data.py` at its finite passes and the filler from `prepare_data.py`.

Every number below is `scripts/build_run2_corpus.py plan` (the function `derive`), unless it says
"measured" (a pilot log) or "estimate". Tests: `tests/test_build_run2_corpus.py`.

## What the recipe gives, against RUN2's Corpus table

RUN2's table (web 70%, QA at its limit, chains 10%, chat replay 20%, biographies under 0.01%) was
written for a builder that retrieves on every window. The pilot recipe has retrieval only on the
QA, chain and biography rows; filler windows carry none. So the merged split is four slices and the
evidence ratio is far below RUN2's 1.49 to 1.55.

| slice | builder | share of the split | RUN2 line it stands for |
|---|---|---|---|
| bios (filler windows plus the biographies, evidence only on the 3,600 persons' cards) | `prepare_injection_data.py` | 95.4% (derived: the rest) | web, edu, code, math 70%, plus the chat replay share nobody can supply (below) |
| qa (SQuAD v2, HotpotQA, web text with a held span, chat replay) | `prepare_evidence_data.py` | 1.6% (supply limited) | evidence QA at its limit, part of chat replay |
| chains (plain, hops 1 / 2 / 3 at 25 / 50 / 25) | `prepare_chain_data.py` | 1.5% | synthetic chains, half of the chain share (default 3%, RUN2 10%) |
| bridge (written bridge, hops 1 to 4 at 25 each) | `prepare_chain_data.py --bridge` | 1.5% | synthetic chains, half of the chain share |

Shares RUN2 leaves undefined, and what is assumed (each is one flag of the driver):

| undefined in RUN2 | assumption | flag |
|---|---|---|
| the chain share (RUN2: 10%) and the plain against bridge split | 3% by default (RUN2's 10% costs 6.8 TB and 3 to 4 days, section 4); half and half | `--chain-share`, `--bridge-fraction 0.5` |
| hop mix of the chain rows in the main body (RUN2 gives 25 x 4 for the tail only) | plain at the pilot's 25 / 50 / 25 over hops 1 to 3, bridge at 25 each over hops 1 to 4 | none (the wrapper's `--hops 4`) |
| the QA builder's target | 2.5B prompt tokens, which is more than its sources hold, so the realised slice is the supply (table below) | `--qa-target` |
| chat replay at 20% | not reachable: the non holdout smoltalk2 left after phase 2 and ir is about 0.3B, ultrachat 0.24B, no_robots 5M; the 18.5 points go to the bios slice | none |
| web text with retrieval | only the QA builder's web arm (a held 384 token span of an `ir` document as evidence), 1.0B tokens | `--ir-tokens` |
| tiers over the run's epochs | the pilot's tiers per split, 1 / 10 / 100 / 1000 exposures per pass, so the run sees 3x that (3 / 30 / 300 / 3000); 3,600 persons, 162,000 renderings, 0.016% of the split | `--persons-per-tier` |
| the tail curriculum (RUN2 Schedule: chains 25%, reasoning weighted) | not built: the main split covers the whole 280B with the main shares. A tail split is one more merge (phase 2 filler, chains 25%) and one more injection build | none |

## 1. The arithmetic

Unit: the merged split's prompt tokens, as `merge_evidence_splits.py --target-tokens` counts them.

| quantity | value | derivation |
|---|---|---|
| tokens per split T | 93,333,333,333 | `TOKEN_TARGET / EPOCHS` = 280B / 3; RUN2's 100B unique at up to 4 epochs is replaced because the evidence trainer runs whole epochs (`num_epochs` times the split's rows, it ignores `target_tokens`), so 100B at 3 epochs would train 300B |
| run tokens | 280,000,000,000 | RUN2 decision 6 |
| epochs of the split in the run | 3 (`EPOCHS`) | T times 3 is exactly 280B (`TOKEN_TARGET`); a fresh permutation each epoch |
| warm-up read | at most 28B | RUN2 Schedule: stop and decide if unmet by 28B; the first 28% of the warm split |
| control | 10,000,000,000 | RUN2 decision 8 |
| chains slice | 2,800,000,000 | 3% of the split (`--chain-share`) |
| plain chains | 1,400,000,000 tokens, 28,090,116 questions | 1.4B over 52.33 tokens per question (measured: `chains_train.mask` 104.66M for 2M questions), times 1.05 margin |
| bridge chains | 1,400,000,000 tokens, 19,491,610 questions | 1.4B over 75.41 tokens per question (measured on the smoke build at hops 1 to 4 at 25 each: 2,351,504 tokens in 31,180 questions, 12.6 chunks per question; the pilot's hops 1 to 3 form was 67.35), times 1.05 |
| qa slice | 1,526,747,326 | the sum of the arms below |
| bios slice (the injection build's `--target-tokens`) | 89,006,586,007 | T minus qa minus chains |
| filler built, phase1 | 84,896,651,867 | bios slice times 1.01 (the 1% margin for validation windows and the window cut) minus phase 2 |
| filler built, phase2 | 5,000,000,000 | assumption: smoltalk2 (15% of phase 2) is the scarce source, see section 2 |
| filler built, ir | 2,500,000,000 | the QA builder's web arm reads it; assumption, same reason |
| persons, renderings | 3,600 persons, 162,000 renderings | the pilot's `--persons-per-tier 1000:100,100:500,10:1000,1:2000` unchanged, so the pilot's A1(b) probe reads at the same noise |
| inject_val fraction | 1.63e-5 | 2,000 windows, as the pilot's 1,961 (the default 0.5% would hold 440M tokens at this size) |

QA arms (the builder stops an arm at `weight x target` or at its supply; SQuAD and HotpotQA at
`--max-source-epochs 2`, and the merge takes the slice for one pass, so the run sees them
3 times; the pilot's QA slice was one build pass at two merge passes):

| arm | builder weight | per pass or supply (source) | realised, estimate |
|---|---|---|---|
| squad_v2 | 0.20 | 5.86M per pass (pilot state file) | 11.7M |
| hotpot_qa | 0.15 | 4.86M per pass (pilot state file) | 9.7M |
| webtext | 0.40 | 2.5B of `ir` times 0.402 prompt tokens per `ir` token (pilot: 60.1M prompt from 149.3M `ir` tokens) | 1,000M |
| smoltalk2 | 0.12 | 1.8B in the 24 no_think files, minus phase 2 (0.75B) and ir (0.75B) | 300M |
| ultrachat | 0.08 | 207k rows at 1,163 tokens (pilot state) | 200M |
| no_robots | 0.05 | 5.3M (pilot state) | 5.3M |

The three merge commands, as printed by `build --dry-run` (shares: bios 0.9727, qa 0.016358, chains
0.015, bridge 0.015; the bios share is 2% over its size so the slice is taken whole and the document
counts of the warm and main merges match, as the pilot's 0.6001 did for the control):

- warm: `--split run2_warm_train --target-tokens 100000000000`, bios from `inject_retrieval_s100a50g1_train`
- main: `--split run2_main_train`, bios from `inject_retrieval_s15a50g1_train`, `--order-from run2_warm_train.slice`
- control: `--split run2_control_train --target-tokens 10000000000 --no-evidence`, bios from
  `inject_full_s15a50g1_train`, no `--order-from`: each slice's subsequence is a prefix of the same
  seeded permutation, so the control is the first 10B of the stream at slice level (its
  `md5_by_slice` is a prefix, not equal; the driver checks warm against main only)

## 2. The filler: sources, volume, repeat

`prepare_data.py` as it stands has one crawl of fineweb-edu (`data/CC-MAIN-2025-26/`, 50 files, 14.78B
tokens: the 16B run's "ran dry") and one tenth of dclm. Needed tokens per source across the three
phases at the plan above, against supply (files times tokens per file, the per-file yield from the
16B manifest: realised tokens over `files_used`):

| source | needed | files needed | supply as coded | supply with `--fineweb-prefix data/` | verdict |
|---|---|---|---|---|---|
| fineweb | 52.9B | 179 | 14.8B (50 files) | about 713B (2,410 files, 110 crawls) | short as coded, fine patched |
| dclm | 9.4B | 87 | 304B (2,790 files) | same | fine |
| finepdfs | 7.3B | 8 | 99B | same | fine |
| code_edu | 12.9B | 46 | 26.9B (95 files) | same | fine, uses 48% |
| nemotron_math | 4.9B | 8 | 32B (46 files, gated, HF_TOKEN) | same | fine |
| wikipedia | 3.4B | 29 | 5.0B (41 files) | same | fine, uses 68% |
| smoltalk2 | 1.5B | 20 | 1.8B (24 files) | same | binding: phase 2 and ir take 1.5B, the QA replay arm gets 0.3B |

Verdict: the 89.0B filler is reachable with no repeat only if fineweb's prefix is widened to all
crawls. The driver does that without touching the builder (`wrap-prepare-data --fineweb-prefix data/`
patches `SourceSpec.file_prefix` in memory). As coded, the build ends at about 48B of the 84B
phase 1 target (every other source reaches its weight, fineweb stops at 14.8B), the biography
build then cycles the filler 1.7x (the pilot repeated 2.18x); the driver does not default to that.
Two more patches in the same wrapper: phase 2 and ir skip the files an earlier phase read (the
manifest's `files_used`), because the builders each start at file 0 of a source and would copy
phase 1's first documents into phase 2 and ir; and the `data_prep` manifest key is written by phase 1
and 2 (the QA builder reads `data_prep.smoltalk2_holdout_hashes` from it), `ir_prep` by ir (the
builder refuses `ir` without a key). A fresh cluster has no `manifest.json`; the driver refuses to run
the filler stage over a manifest that already holds `data_prep.phase1` (the 16B run's counts).

Filler repeat implied by the plan: 1.00x (phase 1 plus phase 2 supply 89.4B against 89.0B plus
margin). The injection build reads `phase1,phase2` as filler (not `ir`, which belongs to the QA
builder's web arm, so no document is both plain filler and a held span web row). The 0.5% default
validation fraction becomes 1.63e-5. Memory: `load_filler_windows` holds every training window as a
Python tuple, about 90M windows of up to 1,000 tokens at this size: plan for 20 to 40 GB of RAM in
`bios_main` and `bios_warm` (estimate; the pilot's 375k windows did not show it).

## 3. Stages, outputs and the cluster command

| # | stage | builder | main outputs | check |
|---|---|---|---|---|
| 1 | filler | `prepare_data.py` x3 through `wrap-prepare-data` | `prepared/{phase1,phase2,ir}.{bin,idx}` | marker, outputs |
| 2 | bios_main | `prepare_injection_data.py --arms full,retrieval --swap-rate 0.15 --support-gold-only --suffix s15a50g1` | `prepared_run2/inject_{full,retrieval}_s15a50g1_train.*`, `inject_val.*`, `index/run2_bios` | files |
| 3 | bios_warm | the same, `--arms retrieval --swap-rate 1.0 --suffix s100a50g1` | `inject_retrieval_s100a50g1_train.*` | files |
| 4 | qa | `prepare_evidence_data.py --max-evidence-tokens 4608 --max-source-epochs 2 --split-prefix evidence_run2 --webtext-phase ir` | `prepared/evidence_run2_train.*` | files |
| 5 | chains_plain | `prepare_chain_data.py --prefix chains_run2` | `prepared/chains_run2_train.*` | files |
| 6 | chains_bridge | `wrap-chains --hops 4` then `prepare_chain_data.py --bridge --prefix chains_bridge_run2` | `prepared/chains_bridge_run2_train.*` | files |
| 7 | merge_warm | `merge_evidence_splits.py`, four slices | `prepared_run2/run2_warm_train.*`, `.slice`, `.merge.json` | `md5_by_slice` has bios, qa, chains, bridge |
| 8 | merge_main | the same with `--order-from run2_warm_train.slice` | `run2_main_train.*` | `md5_by_slice` qa, chains, bridge equal the warm merge's, bios differs |
| 9 | merge_control | `--no-evidence`, 10B | `run2_control_train.{bin,idx,mask,slice,merge.json}` | the four md5 lines |
| 10 | heldout_qa | `prepare_evidence_data.py --heldout --max-evidence-tokens 4608` | `evidence_dev.*`, `evidence_fixed.*` | files |
| 11 | heldout_chains | `prepare_chain_data.py --prefix chains --splits eval,heldout_tmpl,hop4 --seed 42` | `chains_{eval,heldout_tmpl,hop4}.*` | files; same seed, so the held out compositions are the pilot's |
| 12 | val_clean | `prepare_clean_val.py` | `inject_val_clean.*` | files |
| 13 | store_openqa, store_edit | `build_store.py` | `index/openqa`, `index/openqa_edit` | `meta.json` |

Each stage is skipped when its marker (`DATA_ROOT/run2_done/STAGE.json`) and outputs exist; a
failure leaves no marker, so rerunning the same command resumes at that stage (`prepare_data.py`
and the QA builder resume themselves from their state files; the injection, chain and merge
builders restart the stage). Logs: `DATA_ROOT/run2_logs/STAGE.log`. `--stage A,B` runs some,
`--force` reruns, `--dry-run` prints every command.

Cluster command (the rented box has no `env_init`; `HF_TOKEN` for nemotron_math and the gated
reads; move any `manifest.json` out of the repo root first):

```
TINY_LLM_ROOT=/workspace/tiny-llm TINY_LLM_ENV_INIT=/dev/null HF_TOKEN=... \
  nohup python scripts/build_run2_corpus.py build --data-root /data/run2 --device cuda \
  > /data/run2/build.log 2>&1 &
```

The chain share defaults to 0.03; pass `--chain-share 0.10` for RUN2's 10%, see section 4. Stages 1 to 6 are
independent of each other except that 2, 3 and 12 read stage 1 and 4 reads stages 1's `ir` and
phase 2 manifest entry, so on a node with several CPUs run `--stage chains_plain`,
`--stage chains_bridge` and `--stage bios_main` in parallel processes after `--stage filler`
(one process per stage; the driver itself is sequential).

## 4. Storage and time, scaled from the pilot logs

Pilot rates (measured): the two biography builds of 600M tokens took 5 minutes each (12:35 to 12:45,
`build_pilot_bios.log`); 2M chain questions took about 55 minutes on cuda (`chain_depth_micro.md`);
the warm and main merges took about 30 minutes each for 3.49M documents (13:11 and 13:41, 1,940
documents per second, one process) and the evidence-free control 2 minutes. Bytes per token from
the pilot files: bin 2, mask 1, factspan 1, evidence 2 per evidence token (ev and evchunk each) and
768 per chunk occurrence for the keys. The QA builder's and `prepare_data.py`'s wall clocks are not
on disk (`prepare_data.py` ran on the rented box; the QA log was not kept): unmeasured, the
builders print documents per second in minute one.

Default chain share 0.03 against RUN2's 10%, nothing deleted:

| item | chains 3% (default) | chains 10% (RUN2) |
|---|---|---|
| filler bins | 185 GB | 172 GB |
| bios main build (full + retrieval) | 804 GB | 745 GB |
| bios warm build | 448 GB | 415 GB |
| qa build | 19 GB | 19 GB |
| chains plain build | 275 GB | 916 GB |
| chains bridge build | 201 GB | 671 GB |
| merge warm | 853 GB | 1.94 TB |
| merge main | 853 GB | 1.94 TB |
| merge control | 30 GB | 30 GB |
| **total, nothing deleted** | **3.7 TB** | **6.8 TB** |
| left after deleting the build sources (warm, main, control, filler kept) | 1.9 TB | 4.1 TB |
| chain questions, plain / bridge | 28.1M / 19.5M | 93.6M / 65.0M |
| hours, chains plain / bridge builds (one process each) | 12 / 12 | 41 / 41 |
| hours, bios main / warm builds | 25 / 12 | 23 / 12 |
| hours, merge warm / main (one process each; 194M / 289M documents) | 28 / 28 | 42 / 42 |
| hours, merge control | 0.3 | 0.4 |

Against RUN2 Storage (corpus about 1.5 TB in the per-occurrence format, about 0.5 TB with chunks
referenced by store id, plus 140 GB of kept saves): the default needs 1.9 TB standing and 3.7 TB at
the peak, RUN2's shares 4.1 TB and 6.8 TB. The cause is the chains: a 52 token question carries 12
chunk keys (9.2 KB), 177 bytes of key per prompt token, against 0.8 for the bios slice. Two other
costs are the recipe's: the injection builds write `.mask` and `.factspan` beside every token (4
bytes per token in all, before the merge copies it again), and the warm split must be built at the
main split's full size because `--order-from` needs equal document counts although the run reads
only its first 28B (the warm merge is 30% used).

Wall clock with stages in parallel where independent: filler (unmeasured, the download of about 90B
tokens of parquet and jsonl), then the longest chain of bios_main (25 h) and the merges (28 h, warm
then main, or in parallel on two cores after bios_warm and the chains), so about 2 days after the
filler at 3% and 3 days at 10%; "day 1 on CPUs" holds for the arithmetic and the builders that
start, not for the finished corpus. The merge is the longest single process and has no sharding.

Decision needed from the user before the cluster command: the chain share. The driver defaults to
3% (2.8B tokens of chains per split, 28M plain and 19M bridge questions, each seen 3 times). RUN2's
row-level 10% costs 6.8 TB at peak (7.3 TB at the earlier 100B split size) and 3 to 4 days
(`--chain-share 0.10`). RUN2 itself names the 10% as a share with no rung behind it.

## 5. What the builders lack, and how it is worked around

| missing | exact flag and meaning | workaround in the driver |
|---|---|---|
| `prepare_data.py`: a source's file prefix or a crawl list | `--source-prefix KEY=PREFIX` (repeatable): replaces `SourceSpec.file_prefix` for the key, so fineweb can read `data/` (all crawls) | `wrap-prepare-data --fineweb-prefix` patches the imported `SourceSpec` in memory |
| `prepare_data.py`: skip the files an earlier phase read | `--skip-files-of PHASE[,PHASE]`: drops the files listed under `manifest[key][PHASE].per_source[*].files_used` from the listing | `wrap-prepare-data --exclude-files-of` does it through `file_filter` |
| `prepare_chain_data.py`: the training hop mix | `--train-hop-weights 1:25,2:25,3:25,4:25` for the train and val splits (the default stays 25 / 50 / 25 over 1..3; the hop 4 compositions are the seen ones, `seen_compositions(4)`) | `wrap-chains --hops 4` sets `chains.sample_hops.__defaults__`; the seed is unchanged, so the held out compositions and the eval splits are the pilot's |
| `merge_evidence_splits.py`: a shorter warm split | `--order-from-prefix`: take the first N documents of the `.slice` file as the first N of the output and continue the interleave, so the warm split can stop at 28B | none: the warm split is built at full size (the disk cost above) |
| `merge_evidence_splits.py`: sharding | `--shard I/N` on the output (the merge is one process, 28 h at 3% chains, 42 h at 10%) | none |
| `prepare_chain_data.py`: sharded train build | `--shard I/N` over the question stream (the build is one process, 12 h at 3% chains, 41 h at 10%) | run plain and bridge as two processes |
| `prepare_injection_data.py`: a separate size for the `full` arm | `--full-target-tokens N` (the control needs only 10B, the `full` arm writes 89B) | none: delete `inject_full_*` after `merge_control` |
| `load_filler_windows` memory | a numpy accumulation instead of a Python list of tuples | none; size the node |

## 6. Local smoke

Run 2026-10-26 22:06 to 22:30, before T became 93.33B and the chain share 0.03 (so at the 100B plan and 10% chains; the mechanics and the md5 check are unchanged by the resize, which only moves targets), in WSL on the 5090 box while other test jobs shared the disk:

```
python scripts/build_run2_corpus.py build --data-root data/run2_smoke --scale 0.0005   --link-filler-from data/prepared --persons-per-tier 1000:4,100:20,10:40,1:80   --heldout-max-questions 200 --chain-eval-per-hop 200 --store-max-questions 100   --stage filler,bios_main,bios_warm,qa,chains_plain,chains_bridge,merge_warm,merge_main,merge_control,heldout_qa,heldout_chains,val_clean
```

Wall time 1,424 seconds (23.7 minutes), driver exit 0, 3.4 GB on disk (removed afterwards; logs, the
three `merge.json` files and the markers are in `ckpts/inject/run2_corpus_smoke/`). The split is
50M prompt tokens (scale 0.0005), the filler the local stand-in bins `phase1` and `phase2` (60M
tokens, repeat factor 1.00x), the filler stage replaced by symlinks. Stage seconds: bios_main 136,
bios_warm 127, qa 196, chains_plain 184, chains_bridge 172, merge_warm 105, merge_main 93,
merge_control 100, heldout_qa 104, heldout_chains 127, val_clean 80 (each is mostly the model and
embedder load on a contended disk, not the work).

`md5_by_slice` lines (the driver's check passed: warm and main agree on qa, chains and bridge, the
bios slices differ, as they must):

| slice | warm | main | control (10% of the split, a prefix) |
|---|---|---|---|
| bios | `82fd520e514957e659d28d612de74746` | `03538acb47df94becdc1539243d2a3f9` | `bc97ec46867f615098d1b37460c654be` |
| qa | `703e87853d673716912e9b0e798643c3` | `703e87853d673716912e9b0e798643c3` | `b80b08c1306929acecd2896e074d37f1` |
| chains | `91ab61bf2fa7d41ebbc0417acc21907c` | `91ab61bf2fa7d41ebbc0417acc21907c` | `5f9938c1a0560e7a8d3c89633057a835` |
| bridge | `da02ddff7c05b3e693c8e7bca6469b82` | `da02ddff7c05b3e693c8e7bca6469b82` | `db394ecaee4251a9c725df53ccfaf169` |

Merge totals (warm): 49.83M prompt tokens, 21.18M evidence tokens (ratio 0.43), 153,334 documents;
slice shares realised bios 0.879, qa 0.024, chains 0.050, bridge 0.047 (target 0.884, 0.015, 0.05,
0.05; the QA slice is above its share at this scale because no arm is exhausted, the bridge slice
is short because the smoke built it with the pre-smoke estimate of 84.2 tokens per question
against the measured 75.4: the constant is corrected). The main merge reused the warm order
(`--order-from`) with equal document counts per slice. The evidence ratio of the real split is
about 0.4 (chains and QA carry nearly all of it), against RUN2's 1.49: the evidence per row cap
(4608) and the packing line are the pilot's, the Cost section's rate estimate (RUN2 Cost) is
conservative for this corpus.

Read off the smoke: the bridge at hops 1 to 4 builds through `wrap-chains --hops 4` (hops
7,742 / 7,871 / 7,788 / 7,779 of 31,180), 75.41 tokens and 12.6 chunks per question; the
`--support-gold-only` biography build reports supported spans 80.2% against 80.2% gold in the
buffer (a gold-less document supports nothing); `wrap-prepare-data` loads, patches fineweb and, for
the later phases, reports the files to skip (checked against the local 16B manifest with
`--help`).

Not run locally, and what the cluster needs:

| stage | why not | what the cluster needs |
|---|---|---|
| filler (`prepare_data.py` x3) | a full source download (about 90B tokens, 179 fineweb files, 87 dclm, 46 code_edu and the rest, listed above) | `HF_TOKEN` with access accepted for `nvidia/Nemotron-CC-Math-v1`, about 0.2 TB for the bins plus a few shards of scratch; no `manifest.json` in the repo root; the wrapper's fineweb prefix and file skipping are exercised only up to `--help` |
| store_openqa, store_edit | NQ and TriviaQA downloads (`build_store.py` fetches the NQ file whole) | disk and network for NQ; the driver passes `--store-max-questions` through |
| the real scale of everything | the point of the cluster | sections 1 to 4 |

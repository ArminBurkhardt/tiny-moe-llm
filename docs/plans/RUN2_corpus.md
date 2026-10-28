# Run 2 corpus build sheet (rewritten 2026-10-27 from the code)

The corpus of the real run, built with the existing builders by `scripts/build_run2_corpus.py` on
MareNostrum 5. Decision of the user, 2026-10-26: the pilot recipe at scale. The old build item 4 of
[RUN2.md](RUN2.md) (per-window bge retrieval on plain text, the fact tagger and span weights, the
near-duplicate filter, the staircase visibility question) is not on the launch path. What is built is
the pilot's three merges (`pilot_warm_train`, `pilot_main_train`, `pilot_control_train`; Record bullet
"Pilot corpus" in CLAUDE.md, `ckpts/inject/build_pilot_bios.sh`, `merge_pilot.sh`, `merge_pilot2.sh`)
at the Run 2 token target, with `--support-gold-only` on the biography builds (RUN2 decision 3), the
written bridge chains beside the plain ones (R9), QA from `prepare_evidence_data.py` at its finite
passes and the filler from `prepare_data.py`. Added 2026-10-28 (RUN2.md row 16): a fifth merge, the tail split
`run2_tail_train` (42B prompt tokens, read for the run's last 40B on a higher quality mix), section 2b.

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
| tail split | 42B prompt tokens, the run reads 40B (760B to 800B) | `TAIL_TOKENS`, `TAIL_READ`; mix and sources in section 2b; its plain text is 28.95B more filler tokens (28.67B of windows) in a fourth phase; evidence ratio about 0.44 |

QA arms (the builder stops an arm at `weight x target` or at its supply; `--qa-target 2.5e9` is more
than the sources hold, so the slice is the supply; SQuAD and HotpotQA at `--max-source-epochs 2`):

| arm | weight | realised, estimate | supply source |
|---|---|---|---|
| squad_v2 | 0.20 | 11.7M | 5.86M per pass (pilot state file) |
| hotpot_qa | 0.15 | 9.7M | 4.86M per pass |
| webtext | 0.40 | 1,000M | `ir` tokens times 0.402 prompt tokens per `ir` token |
| smoltalk2 | 0.12 | 300M | 1.8B in 24 files minus phase 2 and ir (0.75B each; ir runs with `--exclude-files-of phase1,phase2`, so the two read disjoint files and the remainder is 1.8B - 1.5B = 0.3B, exactly the 0.3B asked: a point estimate from the per file token yield with no extra slack; the qa stage checks the realized tokens of this arm and fails under 90% of its target, warns under 100%) |
| ultrachat | 0.08 | 200M | 207k rows at 1,163 tokens |
| no_robots | 0.05 | 5.3M | pilot state |

Chat replay at RUN2's 20% is not reachable (the non-holdout chat left is about 0.5B); the share goes
to the bios slice. The tail split (section 2b) repeats that chat 4.1 times (it is a third of the QA build) and is 5% of the
tokens, not 20%.

### Filler: sources, volume, repeat

Phase 1 is capped per source at its share of the Hub's supply (code_edu 95%, wikipedia 90%, the
shortfall to nemotron_math up to 65%, then fineweb). Need against supply, all four phases (phase 1, phase 2, ir and the tail's
plain text of section 2b, which takes the files left):

| source | need | files | supply (fineweb widened) |
|---|---|---|---|
| fineweb | 168.6B | 570 | 713B (2,410 files, 110 crawls); one crawl is 14.8B |
| dclm | 41.3B | 379 | 304B |
| finepdfs | 28.0B | 29 | 99B |
| code_edu | 26.6B | 95 | 26.9B (99% used: 95% in the three phases, the tail takes the rest) |
| nemotron_math | 24.9B | 36 | 32.2B (gated) |
| wikipedia | 4.9B | 41 | 5.0B (99% used, the same) |
| smoltalk2 | 1.5B | 20 | 1.8B (binding for the QA replay arm) |

Phase 1 weights after the cap: fineweb 0.6315, dclm 0.1111, finepdfs 0.0778, code_edu 0.0923,
nemotron_math 0.0724, wikipedia 0.0150 (`--phase1-weights`, passed by the stage). The filler is
reachable with no repeat only with fineweb's prefix widened to all crawls (`wrap-prepare-data
--fineweb-prefix data/`, a patch in memory); phases 2 and ir skip the files an earlier phase used
(`files_used` of the manifest); `data_prep` holds phase 1 and 2 (the QA builder reads its
`smoltalk2_holdout_hashes`), `ir_prep` holds ir and, since 2026-10-28, `smoltalk2_ir_hashes`, the hashes of the smoltalk2 conversations the ir phase drew; the QA builder excludes those too, so the chat replay arm never repeats a conversation any filler phase used. The QA web text arm reads `ir.bin` documents as plain text and does not filter them: the ir bin has no per document source tag, so the chat conversations in it (30% of ir tokens, rendered as `role: content` lines) are also seen as web text rows, once as a plain language row, which is accepted. An `ir.bin` built before this change has no `smoltalk2_ir_hashes` and is excluded by nothing but phase 2's list. **Filler repeat is checked**: after each biography
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
| filler | `prepare_data.py` x4 through `wrap-prepare-data` (phase1, phase2, ir, then the tail's plain text: fresh files, `--manifest-key tail_prep`, its own directory) | `prepared/{phase1,phase2,ir}.{bin,idx}`, `prepared_tail/phase1.{bin,idx}`, `manifest.json` entries | resumes from its state files; `--force` on a pruned filler now fails (the state records more bytes than the bins hold) |
| qa | `prepare_evidence_data.py` | `prepared/evidence_run2_train.*` | resumes (since 2026-10-28 the state file carries the rng states, distractor reservoirs and round robin credits, so a resumed build is byte identical to an uninterrupted one; a state file from before has none and restarts them) |
| bios_main | `prepare_injection_data.py`, arm retrieval, swap 0.15, gold-only support | `prepared_run2/inject_retrieval_s15a50g1_train.*`, `inject_val.*`, `index/run2_bios` | resumes (state file; a kill before the first state write is rebuilt) |
| val_clean | `prepare_clean_val.py` | `inject_val_clean.*` | partial outputs deleted, rerun |
| bios_full | injection builder, arm full | `inject_full_s15a50g1_train.{bin,idx,mask}`, a 10.5B prefix | resumes |
| bios_warm | injection builder, arm retrieval, swap 1.0 | `inject_retrieval_s100a50g1_train.*`, a 29.4B prefix | resumes |
| tail_text | `build_run2_corpus.py tail-text`, numpy | `prepared_tail/tail_text.{bin,idx,mask,json}`: the tail's plain text in windows of at most 1000 tokens | partial deleted, rerun |
| prune_filler | `rm` | deletes the filler bins and the tail phase's bins | idempotent |
| chains_plain | `prepare_chain_data.py` | `prepared/chains_run2_train.*` | resumes |
| chains_bridge | `wrap-chains --hops 4`, `--bridge` | `prepared/chains_bridge_run2_train.*` | resumes |
| qa_shuffle | `merge_evidence_splits.py` over the qa build alone | `prepared_run2/run2_qa_shuf_train.*` | resumes (merge state) |
| order | `build_run2_corpus.py order`, numpy, about 9 GB | `run2_order.slice`, `run2_order.json` | partial deleted, rerun |
| merge_main | `merge_evidence_splits.py --order-prefix-from run2_order.slice`, whole order | `run2_main_train.*`, `.slice`, `.merge.json` | resumes (merge state, `--overwrite`) |
| merge_warm | the same, prefix of main to 28B | `run2_warm_train.*`, then `run2_main_train.order.json` (`epoch0_head_docs`) | resumes |
| merge_control | the same, `--no-evidence`, prefix to 10B | `run2_control_train.{bin,idx,mask,slice,merge.json}` | resumes |
| rows | `sft.py --evidence --estimate-rows-only`, CPU | `run2_{main,control,tail}_train.rows_estimate.json` (entries for 1, 2, 4, 8, 16 ranks; the tail's is checked against the rows the last 5% of the plan needs) | partial deleted, rerun |
| prune_bios | `rm` | deletes the three biography builds and the raw QA build (the merges and the permuted copy hold them) | idempotent |
| merge_tail | `merge_evidence_splits.py`, five slices, seed 43 (section 2b) | `run2_tail_train.*`, `.slice`, `.merge.json` | resumes (merge state, `--overwrite`) |
| prune_sources | `rm` | deletes the chain builds, the permuted QA copy and `tail_text` | idempotent |
| heldout_qa | `prepare_evidence_data.py --heldout` | `evidence_dev.*`, `evidence_fixed.*` | partial deleted, rerun |
| heldout_chains | `prepare_chain_data.py --prefix chains` | `chains_{eval,heldout_tmpl,hop4}.*` (same seed as the pilot) | partial deleted, rerun |
| store_openqa, store_edit | `build_store.py` | `index/openqa`, `index/openqa_edit` | partial deleted, rerun |

Rules the driver enforces:

- A stage with a marker and its outputs (or pruned by a later prune stage) is skipped; `--force`
  reruns. On a skip the stage's check runs again (a merge's md5 lines, the `.slice` md5 against the order file or the main split, token totals; not the deep document comparison, which ran when the merge finished; not run if the outputs were pruned) and so do its post checks (the filler
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

## 2b. The tail split (decided 2026-10-28, RUN2.md row 16)

The last 5% of the run, 40B tokens from 760B to 800B, is read from its own split on a higher quality mix, built with the corpus
(days 1 to 5 on the cluster, not mid-run). The cosine is unchanged: one schedule fitted to the 800B plan from step 0 on
`run2_main_train` (`--plan-split`), and the tail is where it ends. The main chain stops by itself after the first save at or past
760B (`STOP_AFTER` defaults to `760000M` on the main split) and waits for the user's switch (`scripts/slurm/README.md`, Day 1
step 10).

**Size.** `run2_tail_train` is 42B prompt tokens, 5% over the 40B the run reads, so the run reaches its last optimizer step before the
split ends. The trainer ends a run at the end of its split (`num_epochs` times the rows), not at the plan's step count, so the tail
launch passes `--end-at-plan`: the first log step on or past the plan's `total_steps` ends the run (final validation, `_final.pt`,
exit 0), with the cosine at its floor. A split that ended first would end the run early, before the floor; so the tail's own row
estimate (the `rows` stage, every rank count) is checked against the rows the last 5% of the plan needs (`rows_check`: the main
rows times the 3 epochs times 5%), and the trainer refuses the switch (exit 30) when the tail holds fewer optimizer steps than the
plan has left. The tail's tokens per row differ from the main split's (more evidence per row, so rows close earlier), so the end
lands near 800B, not exactly on it; the step count is what is exact.

**Mix** (prompt token shares, `TAIL_SHARES`; the slice labels are the merge's):

| slice | share | tokens | input | passes of the input (cap) |
|---|---|---|---|---|
| text | 0.65 | 27.3B | `tail_text` (the filler's fourth phase, windowed) | 1 (fresh files only) |
| main | 0.12 | 5.04B | `run2_main_train` itself: a seeded sample of its documents (1.9%) | 1 |
| qa | 0.15 | 6.3B | `run2_qa_shuf_train`: the QA build with its chat arms (a third of its tokens: about 10% QA, 5% chat) | 4.13 (5) |
| chains | 0.04 | 1.68B | `chains_run2_train` | 1.20 (2) |
| bridge | 0.04 | 1.68B | `chains_bridge_run2_train` | 1.20 (2) |

The merge runs with its own seed (`TAIL_SEED` 43), so its permutations differ from the main run's, and the biography documents in the
`main` slice keep the main form (swap 0.15, gold only support, anonymization 0.5, the same masks) because they are copied from the
main merge, evidence files included; the leak reads stay comparable. That slice is mostly the filler windows the biography build
mixed with the renderings (the renderings are 0.016% of the main split), so a 12% sample of it is, to a good approximation, more of
the main run's filler and 0.0019% renderings: the persons' exposure counts move by 1.9% of one pass, not by a tier.

**The plain text** (65%) is a fourth filler phase: the filler stage's fourth command, `wrap-prepare-data --exclude-files-of
phase1,phase2,ir --phase1-weights ... --phases phase1 --data-dir DATA_ROOT/prepared_tail --manifest-key tail_prep`, so the tail reads
only files none of the three earlier phases used (no file repeats; the validation windows of `inject_val` cannot appear in it). The
weights prefer dclm, finepdfs, nemotron_math, code_edu and wikipedia over fineweb (`TAIL_WEIGHTS`) and are held by `cap_tail` at the
fresh supply, 99% of the Hub's files minus what the three phases took, the shortfall going to dclm, finepdfs and fineweb in order:

| source | asked | gets | fresh limit |
|---|---|---|---|
| dclm | 11.00B | 13.13B | 272.18B |
| finepdfs | 7.24B | 7.24B | 75.24B |
| nemotron_math | 4.05B | 4.05B | 10.18B |
| code_edu | 2.32B | 0.86B | 0.86B (capped: the main filler took 95% of its files) |
| wikipedia | 0.87B | 0.19B | 0.19B (capped, the same) |
| fineweb | 3.47B | 3.47B | 540.47B |

(Counted in whole files since 2026-10-28: every earlier phase marks its partly used last file as used and the tail phase skips it, so `cap_tail` charges each phase its files rounded up. Before, the plan charged tokens: wikipedia was planned at 0.45B and had about 0.24B of fresh files, code_edu 1.08B against less, and the tail plain text could have ended about 0.2B short after two days of filler. The text margin is also 5%, was 3%, because `prepare_data` does not move an exhausted source's share to the others: the per source realized counts are read after the build.)

That is 28.95B filler tokens for 28.67B of windows (`tail_text`: windows of at most 1000 tokens with BOS and EOS, a last window under
32 tokens dropped, mask 1 but on BOS: the biography build's own rule, so the tail's text has the main run's shape, numpy, about
10M tokens per second, an assumption; the merge permutes the documents). Repeats of already used files were allowed by the
request; none are used (the mirror deletes behind its consumer, so a second read of an early file would wait for a file the pusher
no longer sends), and the price is that code_edu and wikipedia, the highest quality sources, are 1.5B of the 27B.

**Repeat factors of the evidence slices.** QA at 4.13 passes of its build (the request said about 3 for QA and about 2 for chat; the
chat arms are inside the build, so both repeat 4.13 times, and the web text arm with its held span evidence repeats 4.1 times on
top of the 3 epochs it already had), the chains at 1.2. The cap on the QA passes is 5, on the chains 2 (`plan` refuses beyond them).
A second QA build over the tail's fresh text would remove the repeat; it was not built (a GPU embedder job of about the QA build's
size).

**Evidence ratio** about 0.44 evidence tokens per prompt token (the main run's 0.045), from the QA and chain slices: about 10% slower
per token (459M FLOP per token body plus 118.4M per evidence token: 511M against 464M, so about 84k tokens per second per GPU
against 93k at the corpus's own ratio), about 12 more H100 hours for the 40B tokens (an estimate; the rate has not been measured at
this ratio).

**Stage order and the disk.** `tail_text` runs before `prune_filler` (the tail phase's bins are pruned with the filler), and the
tail merge reads the main split, so it runs after the three first merges. To keep the peak under the budget `prune_bios` deletes
the three biography builds and the raw QA build (the merges and the permuted QA copy hold them) before `merge_tail` writes, and
`prune_sources` (after `merge_tail` and `rows`) deletes the chain builds, the permuted copy and `tail_text`. The tail split is
standing data like the three merges: standing 2.32 TB (the standing budget moved from 2.0 TB to 2.6 TB), peak 3.15 TB, at the end of
`merge_control`, before `prune_bios` (section 3). `merge_tail` is resumable like the other merges; `tail_text` restarts clean (it
takes under an hour). `rows` estimates the tail split too (1, 2, 4, 8 and 16 ranks); the trainer reads the main split's estimate for
the plan and the tail's only for the switch check.

**What the trainer does** (`scripts/sft.py`, docstring "Tail switch"): `--switch-from-start` (with `--switch-split`) lets the switch land in
the checkpoint's last epoch, planned on the same `--plan-split`, and reads the new split from its first document: the position and the
step in the epoch restart at zero; the optimizer, the scheduler (the step count: one cosine), the token count, the fp32 masters
and the RNG states carry. `--end-at-plan` is above. Tests: `tests/test_build_run2_corpus.py` 31 to 34, `tests/test_sft_contracts_2026_10_27.py`
4, `tests/test_run2_launchers.py`, `tests/test_slurm_chain.py` (the 760B stop), `tests/test_ops_tools.py` (the "tail switch due" alert).

## 3. Disk and time (`plan`, estimates from the pilot rates)

| stage | writes | deletes | running |
|---|---|---|---|
| filler | 594.7 GB (+30 GB scratch; 537.6 GB without the tail phase) | | 594.7 GB |
| qa | 18.8 GB | | 613.5 GB |
| bios_main | 800.4 GB | | 1.41 TB |
| bios_full / bios_warm | 31.6 / 89.6 GB | | 1.54 TB |
| tail_text | 84.7 GB | | 1.62 TB |
| prune_filler | | 594.7 GB | 1.03 TB |
| chains_plain / chains_bridge | 315.2 / 239.1 GB | | 1.58 TB |
| qa_shuffle | 18.8 GB | | 1.60 TB |
| merge_main | 1.37 TB | | 2.97 TB |
| merge_warm / merge_control | 144.2 / 30.2 GB | | 3.15 TB |
| prune_bios | | 940.1 GB | 2.21 TB |
| merge_tail | 771.1 GB | | 2.98 TB |
| prune_sources | | 657.8 GB | 2.32 TB |
| held out, stores | 1.8 GB | | 2.32 TB |

Peak 3.15 TB (budget 3.4 TB, the filler's inbox window counted when the mirror is on; it was 3.06 TB before the tail: the tail's
text bins and `tail_text` add 140 GB at the peak and `prune_bios` runs before `merge_tail` writes its 771 GB), standing 2.32 TB
(budget 2.6 TB, moved from 2.0 TB for the tail split; the project quota is 4 TB and the checkpoints, about 0.4 TB kept saves plus
the rolling two, come out of the rest). The tail merge is mostly its chain slices (3.36B tokens at about 197 bytes of keys and
evidence per token, 660 GB of the 771 GB). The cause of the size is the chains (12 chunk keys of 768 bytes per 52 token
question, 177 bytes per prompt token against 0.8 for the bios slice) and the injection builders'
`.mask` and `.factspan` beside every token.

Hours, one process each: bios_main 35.2, bios_full 1.4, bios_warm 3.9, tail_text 0.8, chains_plain 12.3,
chains_bridge 12.3, qa_shuffle under 0.1, order 0.3, merge_main 3.6, merge_warm 0.4, merge_control
under 0.1, merge_tail 0.9; filler, qa and val_clean unmeasured (the filler's fourth phase adds about 28B tokens of
download and tokenizing, about 6% of the filler). Every stage over 60 hours must be resumable (a 72 h
job loses a non-resumable one); `plan` flags it. Independent stages can share a node
(`--parallel chains_plain,chains_bridge,bios_main`; the driver refuses two injection builds in one
group).

## 4. The cluster: no internet on any node

No MareNostrum 5 node, login nodes included, reaches the Hub, so the filler reads its shards through
the mirror (`scripts/mirror_sources.py`, demand driven) and the other Hub reads come from a staged
cache. Recipe in `scripts/slurm/README.md` ("The filler through the mirror"); in short:

1. On the PC, once: `python scripts/mirror_sources.py list --out listing.json` (the plan flags as for
   the build; 490 GB in 1,280 files at the full plan with the tail phase, 444 GB in 1,127 without it, limit 2 TB) and `python scripts/mirror_sources.py
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
| `prepare_chain_data.py`: the training hop mix 1..4 at equal weight | `wrap-chains --hops 4` sets `chains.sample_hops.__defaults__`; seed unchanged, so the held out compositions are the pilot's. Caveat: the 4-hop questions are 25% of the training chains (plain and bridge), over the same 4-hop compositions `chains_hop4` draws from, so the 4-hop read on this run is not an unseen depth read; only the held-out 2-hop compositions (`chains_heldout_tmpl`) are unseen |
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

## 8. Local smoke of the tail, 2026-10-28

Corpus: `python scripts/build_run2_corpus.py build --data-root /tmp/r2smoke_tail --scale 0.0005 --link-filler-from data/prepared
--heldout-max-questions 200 --chain-eval-per-hop 200 --store-max-questions 100`, run as four stage groups in the foreground on the
5090 (each under 20 minutes; the stages resume, so a group boundary is a rerun with `--stage`), scratch root in WSL, nothing in the
repo. Every stage ok, exit 0 each time, 45 markers. The group walls were 758 s (filler linked, qa, bios_main, val_clean,
bios_full, bios_warm, tail_text, prune_filler), 705 s (chains, qa_shuffle, order, the three merges), the third group
(prune_bios, merge_tail, rows, prune_sources, the held out builds, store_openqa) and 246 s for `store_edit` rerun alone (it hit
the group's timeout while a second build shared the machine). Single stages: tail_text 103 s (the local `ir` bins stand in for the
tail phase, `--allow-repeat` not needed: no repeat), merge_tail 102 s, rows 386 s (three estimates), qa 144 s.

The tail split at this scale (`run2_tail_train`, the plan's 21,000,000 tokens, gap to the target 4,504): text 13,649,545 tokens
(0.97 of the windows built), main 2,519,156 (12,244 documents, 1.9% of the main split's, the biography documents among them in
the main form), qa 3,146,869 (2.55 passes of the smoke's QA build, evidence ratio 1.87 there), chains 839,967 and bridge 839,959
(1.14 passes each), 65,611 documents, 13.86M evidence tokens, ratio 0.660 (the full plan's estimate is 0.44: the smoke's QA
slice carries more evidence per token than the real build's, 0.96). `rows`: main 35,306 packed rows per epoch at 1 rank (4
workers), tail 5,910: the last 5% of the plan needs 0.05 x 3 x 35,306 = 5,296, so the tail holds 11.6% more rows than the plan
has left (its rows hold fewer tokens, 3,553 against 3,774, because evidence closes them early), and `rows_check` passed. The
real run's tail should therefore end on the plan's last step at about 799B tokens, not exactly 800B.

Training smoke on a second, smaller corpus (`--scale 0.00005 --persons-per-tier 1000:3,100:15,10:30,1:60`: main 13.34M tokens,
26,007 documents, 3,631 rows per epoch; warm 1.40M tokens, 2,742 documents, head of the main order; tail 2.10M tokens, 5,040
documents, 602 rows; the persons are 1/100 of the plan's because the biography slice cannot shrink below its renderings),
`TINY_LLM_CONFIG=ckpts/smoke_tail/config_run2_smoke.yaml` (`config_run2.yaml` with batch 2, saves every 1M tokens, evals every 2M, no
fixed split, warmup 1%), `ckpts/smoke_tail/smoke_train.sh SPLIT STOP_AFTER FLAGS`, one GPU, the stop watcher
`stop_after_save_run2.sh` as in the launchers. Plan: 3,631 rows per epoch x 3 epochs, 5,446 optimizer steps (cache found).
Checked with `ckpts/smoke_tail/ckinfo.py` on the checkpoints at each stop: (1) warm launch (`--max-epochs-this-split 1`,
`STOP_AFTER=1M`): save `tok1M`, epoch 0, position 1,929 of the warm split's 2,742 documents, 1,029,145 tokens, optimizer step
141 of 5,446, lr 3.99769e-4 (the optimizer's lr equals the cosine of `build_sft_scheduler` at that step), the watcher dropped STOP, exit 10.
(2) switch to the main split (`--switch-split`): "Resumed evidence at epoch 0, position 1,929", epoch 0 of the main split ended
at 13.35M tokens, epoch 1 began; `STOP_AFTER=19M`: save `tok19M`, epoch 1, step in the epoch 770, 19,010,698 tokens, position 10,821
documents, optimizer step 2,593 of 5,446 (total_steps unchanged), lr 2.36442e-4 (the cosine's value), exit 10. Not run: the third
launch (main to the 38M save, which is epoch 2, then `--train-split run2_tail_train --switch-split --switch-from-start --end-at-plan`):
the WSL root filesystem failed with I/O errors on the machine (D: had no free space left at the same time, cause not found) and did
not start again, so the tail switch, the LR and token continuity across it, and the run's end at step 5,446 (final save, exit 0) were
verified by the unit tests and the code only (`tests/test_sft_contracts_2026_10_27.py` 4, the launcher and chain tests).
The commands, to run once WSL is back, from the repo root in WSL after `source env_init`, with the smoke corpus at `/tmp/r2tiny`
(rebuild it with the command above if /tmp is gone): `bash ckpts/smoke_tail/smoke_train.sh run2_main_train 38M --switch-split
--plan-split run2_main_train`, then `bash ckpts/smoke_tail/smoke_train.sh run2_tail_train '' --switch-split --plan-split
run2_main_train --switch-from-start --end-at-plan`; expect "--end-at-plan: the plan's 5,446 optimizer steps are done", a
`checkpoint_evidence_final.pt` and exit 0.

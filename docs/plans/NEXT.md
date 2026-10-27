# NEXT.md

The plan (older notes and commit messages call it `PLAN.md`). Rewritten 2026-09-30 around the goal
stated below; the previous version is in git history (`git show aea3372:docs/plans/NEXT.md`). The
record of the 16B-token run is [docs/CONCLUSION.md](../CONCLUSION.md). Read
[CLAUDE.md](../../CLAUDE.md) first, it is authoritative for everything already built. Every
measurement lives in [docs/measurements/](../measurements/). The two reviews this plan rests on
are [docs/review_2026-09-18.md](../review_2026-09-18.md) (the pre-run code review) and
[docs/review_2026-09-29.md](../review_2026-09-29.md) (the direction review, with the loop section
and the new references). The design itself, with blueprints of both runs, is
[docs/evidence_path_design.html](../evidence_path_design.html).

## Now (2026-10-27)

Mirrored in CLAUDE.md's "Now": the State block and the Open items table are the same text in both
files. Restructured 2026-10-24 after the repository review; the history moved under Record.

### State (2026-10-27)

- **Ran and read 2026-10-24 and 2026-10-25:** the three micro retriever arms the review proposed
  (`bash ckpts/inject/launch_review_arms.sh`, about 40 minutes each plus the store recall read, on
  the code in the tree, three passes over `evidence_nomany_train` from `seed_micro.pt`), the first
  two launched 2026-10-24 about 20:25 in sequence: `w0e3b`, the `w0e3` recipe rerun on the
  reviewed code as the twin the other two are read against
  (`ckpts/inject/config_micro_infonce0_e3.yaml`); `l0e3`, the `w25e3` recipe with the contrastive
  term on the first pass only and the trunk gradient on (`config_micro_infonce_l0_e3.yaml`). The
  third, `seg_e3`, the `w0e3` recipe with the selection loss weighted per query segment
  (`evidence_selection_weighting: segment`, `config_micro_segsel_e3.yaml`), ran 2026-10-25. Run
  directories `ckpts/evidence_infonce_{ARM}`, logs `ckpts/inject/infonce_{ARM}.log`, reads
  `ckpts/inject/infonce_{ARM}_store_recall.{json,log}`. **Order changed the same evening:** the
  watch on the launcher expired at 20:58 and took the launcher and the twin (at 74M, last save
  50M) down with it; the relaunch ran detached from any watch. The resumed twin read epoch 0's
  permutation from the epoch 1 position (the SFT resume epoch bug, found and fixed that evening:
  accelerate's prepared loader sets the dataset's epoch from its own pass counter, which starts
  at zero in a new process, so a resume past epoch 0 trained on epoch 0's order; `sft.select_epoch`
  sets it through the loader, `tests/test_sft_resume_state.py` test 8; no recorded run ever resumed
  past epoch 0, so nothing on record is affected); it was killed before any save. `l0e3` ran
  first (fresh, 21:02; 104.46M tokens in 39.3 minutes at about 45k tokens per second, exit 0),
  then `w0e3b` resumed at 21:45 from its 50M save under the fix
  (`ckpts/inject/queue_twin_after_l0e3.sh`; its 60M eval reproduced the original run's within
  0.0001, 4.6262 against 4.6261; 104.56M tokens, exit 0). The card paused around 22:30 on
  2026-10-24 (the user). `seg_e3` ran 2026-10-25 (its yaml parked as
  `config_micro_segsel_e3.yaml.hold` so the first launcher skipped it, renamed back before `bash
  ckpts/inject/launch_review_arms.sh seg_e3`): launched 10:24 detached under the queue script
  `ckpts/inject/queue_2026_10_25.sh`, 104.46M tokens in 40.0 minutes, peak 16.6 GB, exit 0, the
  step 0 eval identical to the twin's (same seed); the user stopped the queue after this arm to
  pause the card at about 11:05, so its later steps (the leak arms `m1` and `g1`, then
  `read_clean_val.sh`) did not run then. They ran the same afternoon, detached, under
  `ckpts/inject/queue_2026_10_25b.sh` (the paired `seg_e3` exit read first): `m1` warm-up 12:33 to
  13:04 (about 54k tokens per second, 12.2 GB, exit 10 on the 100M save; the stop fell within one
  log interval of that save, so no second save, the rule of 2026-10-24), main 13:11 to 14:13 (60.5
  minutes, 299.19M tokens, exit 0), reads done 14:24; `g1` warm-up 14:24 to 14:56 (exit 10), main
  15:03 to 16:05 (60.6 minutes, exit 0), reads done 16:16 (run directories
  `ckpts/evidence_leak_{m1,g1}`, logs `ckpts/inject/leak_{m1,g1}.log`, reads
  `ckpts/inject/rank_leak_{ARM}_{100M,final}_*` and `compare_cf_vs_leak_{ARM}.log`); then `bash
  ckpts/inject/read_clean_val.sh` 16:16 to 16:25 (`eval_exit.py`, 40 batches of 4, no evidence, on
  the arm P, C and E4 finals on `inject_val_clean` and `inject_val`; outputs
  `ckpts/inject/pilot_{p,c,e4}_exit_{inject_val_clean,inject_val}.{json,log}`), every read exit 0.
  The day's queue is complete and the card is idle.
- **Decided 2026-10-24 (the user):** R4 (loss placement by coda) is dropped, obsolete on R9: the
  model composes at 0.997 on held-out compositions once the intermediate entity is written out,
  so the loop no longer has to compose in silence; the coda stays unbuilt. R6 (the
  pretraining-text leak probe) is skipped: its mechanism is already read at micro (arm (a) stores
  plain-text facts under the normal loss, tier 100 `delta` 0.378; arm (b), fact spans masked,
  stores nothing; arm C repeats arm (a) at pilot shape), so the run spec uses that lever (the
  spans to keep out of the weights are masked, or plain-text facts are accepted into them) in
  place of a new arm. A4's flip read gets a bge routed variant (retrieval through the store's bge
  query keys, the serving configuration, in place of the model's own query, which finds nothing).
- **Decided 2026-10-24, evening (the user, on the review's recommendations):** the review's held
  code changes land (landed the same evening, committed as c79002e, tests passing on the GPU, a detached
  Sonnet review with no high or medium finding; the Open items row and the Record), except the
  eval-pass rewrite (deferred to 6a) and the `eval_chains.py` bridge-mode items. The evidence
  reader keeps dropping the 4094-token rows: future evidence arms on the existing builds stay
  matched to the arms already run (the pilot merges keep dropping the 12,240 rows; new evidence
  builds cap a training row at 4093 tokens). Run order: the three micro retriever arms, then the
  two leak arms, then the clean-slice CE re-read, with R5's eval set between them when the card
  is idle.
- **Ran and read 2026-10-25, evening:** `cfb`, the recorded warm-up arm's recipe on the flagless splits
  rerun on the code in the tree (`bash ckpts/inject/launch_leak_arm.sh cfb`, launched 2026-10-25
  16:49 detached; warm-up exit 10 on the 100M save at 17:19, main 17:26 to 18:25, 299.19M tokens in
  57.0 minutes, exit 0, reads and compares done 18:36; run directory `ckpts/evidence_leak_cfb`, log
  `ckpts/inject/leak_cfb.log`, reads `ckpts/inject/rank_leak_cfb_*`, `compare_cf_vs_leak_cfb.log`
  and the two same-code compares `compare_cfb_vs_leak_{m1,g1}.log`): the same-code twin of `m1` and
  `g1` for the paired leak compare, and the pin on the filler CE gap between today's arms and the
  recorded warm-up arm (the fp32 CE accounting explains 0.002 of it). Paired on the same code both
  leak candidates stay falsified (every cell within z 1.1), `cfb` copies at 0.95 to 0.96 at the
  switch, so the 0.88 to 0.90 of `m1` and `g1` was the flags, and it ends at filler CE 3.8885, so
  the 0.022 gap to the recorded arm's 3.9106 is the code since 2026-10-05, not the flags. The
  recipe itself reads tier 100 at z 2.9 in distribution and 3.7 held out on today's code (the
  recorded arm 2.6 and 2.0), over 3 sigma held out on all three arms of the matched set, a caveat
  for the user, not a verdict change.
- **Decided 2026-10-25 (the user, on `docs/plans/RUN2.md`'s first table, every row on the
  recommended option):** (0) shape L is the configured `config.yaml` shape, width 768, not the
  blueprint's 1024; (1) 32 routed experts kept; (2) `evidence_selection_weighting: segment` is the
  real run's default; (3) `--support-gold-only` is carried into the real corpus builder as a data
  rule, a build item behind a flag; (4) the goal's reasoning clause is reworded to "facts in the
  store through the retrieval pathway, reasoning in written tokens, the looped trunk as computation
  per token", the original "reasoning in the looped trunk" kept on record as the original goal and
  the aspiration to revisit after the real run (NEXT.md "The goal"); (5) today's one block per
  pass, the three-sublayer core not built (R4b's sublayer half closed); (6) the token target 280B,
  the application asking 2.5k H100 hours plus the corpus build line plus a 20% buffer (about
  3.0k), ceiling 5k; (7) R5 tonight on the local card; (8) a short matched control to 10B tokens,
  about 30 H100 hours (about 4 hours on an 8-GPU node if DDP scales linearly); (9) the thin tier
  100 held-out margin at micro accepted as the recipe's known state on one seed, stated in the
  caveats, no second seed.
- **Just read:** on the evening of 2026-10-25, R5, the winner's eval set
  (`docs/measurements/r5_bridge_ladder.md`; the main thread's verdict). Arm P's final already
  carried the complete ladder (2026-10-14, the decode reads redone on the fixed packer 2026-10-24,
  the A4 flip on the bge route) and its 4-hop reads at depths 3 and 4 by read sites (at chance at
  every save); the bridge final carried the 4-hop generate read (0.206 against chance 0.210, two
  steps written); "HotpotQA bridge by read sites" has no instrument (the bridge decode exists only
  for the synthetic chains), a build item. New: the A-gate ladder on the written-bridge final
  (`ckpts/evidence_pilot_bridge/checkpoint_evidence_final.pt`, arm P's final plus 66.35M tokens of
  chain-only bridge SFT, the form 6f builds on; `bash ckpts/inject/r5_bridge_ladder.sh`, launched
  18:57 detached, one process per read, complete 19:51, log `ckpts/inject/r5_bridge_ladder.log`,
  outputs `ckpts/inject/pilot_bridge_ladder_*`). Closed book every cell at chance: entity `delta`
  at tier 100 / 1000 in distribution 0.004 (z 0.9) / -0.000 (z 0.0), held out 0.000 (z 0.1) / 0.006
  (z 0.7), against arm P's 0.010 / 0.064 and 0.013 / 0.046 (the tier 1000 residual is gone).
  Copying halved: gold top-1 0.755 / 0.757 / 0.755 / 0.797 by tier in distribution and 0.40 to 0.47
  held out (arm P 1.000 and 0.95), swapped followed 0.71 to 0.73 and 0.50 to 0.53 (arm P 1.000 and
  0.94 to 0.97). The in-context ceiling probe reads answer-span CE 12.33 / 12.41 / 14.14 / 15.41
  under gold / mixed / distractors / none (arm P 3.49 / 4.10 / 6.20 / 6.05; ceiling 3.08, content
  1.81, the bridge final's own; arm P's are 2.56 / 2.71), the levels set by the answer format (the model expects "Step 1:" first). A3 in generation:
  EM 0.000 and F1 0.008 under gold, non-abstain 1.000 under none and distractors, the completions
  the bridge template filled with degenerate or invented strings ("Step 1: 1: 1: 1:", "Answer: The
  Basisines."); A7's pathway EM 0.000 on every route (arm P's oracle 0.088); A4 on the bge route
  has nothing to flip (0 of 2,000 answered before the edit). Clean-slice CE 8.687 / 8.974 / 9.062
  at exits 1 / 2 / 3, token top-1 0.058, against arm P's 3.348 / 3.274 / 3.267 and 0.42: plain
  language destroyed and depth now hurts. The 13-task suite: mean multiple-choice headroom over
  chance +0.019 over 9 tasks against arm P's +0.062 and the control's +0.073 (lambada 0.002 against
  0.115, sciq 0.451 against 0.599, arc_easy 0.297 against 0.370; gsm8k 0.001 against 0.014; the
  generations degenerate repeats; restated 2026-10-25: BoolQ chance was set at 0.50 while the majority class is about 0.62 (arm P 0.4098, control 0.4881, bridge 0.3810 against all-no 0.378), so on the 8-task mean arm P reads +0.0917, the control +0.0845 and the bridge +0.0513 (9-task +0.0615 / +0.0725 / +0.0191); A6's "within noise of the control" holds on both (z about 1.2 and 0.76), the sign of the control's lead flips, and R5's suite gap is about half, not a third; the verdict stands on the other gates). Verdict: the chain-only bridge SFT at the chain continuation's
  recipe is a demonstration that composition happens in written tokens (R9 stands, 0.997 held out),
  not a usable checkpoint. For the spec: 6f's mix carries the pretraining slices (evidence QA, the
  abstention rows, the biographies at their tier structure, the plain-answer form) beside the
  bridge rows at a lower LR than the pilot's continuation, with the A gates and the clean-slice CE
  read at every 6f save and the kill rule on them; every A gate is taken on the shipped checkpoint,
  after 6e and 6f; the pilot's tier 1000 residual is a property of the pretrained trunk that a
  short post-training removes, so the leak reads after post-training are the binding ones; the
  HotpotQA bridge instrument joins the build list (`docs/plans/RUN2.md`). Earlier the same evening,
  `cfb`, the same-code twin of the two leak arms (`docs/measurements/leak_candidates_micro.md`, the
  same-code twin section; the main thread's verdict). Paired prior-controlled `delta`, `cfb` minus the arm, against `m1` held out tier 100
  +0.005 (z 1.1) and tier 1000 +0.005 (z 0.4), in distribution -0.006 (z -0.7) and -0.003
  (z -0.2); against `g1` held out +0.001 (z 0.3) and +0.002 (z 0.2), in distribution -0.001
  (z -0.1) and -0.013 (z -0.7): both candidates falsified on the same code. At 100M every cell
  inside 3 sigma (largest |z| 1.8) with gold top-1 0.964 / 0.961 / 0.949 / 0.950 in distribution
  (the recorded arm 0.973 / 0.973 / 0.959 / 0.953), so the slower copying of `m1` and `g1` at the
  switch was the flags. Final entity `delta` in distribution 0.018 (z 2.9) at tier 100 and 0.070
  (z 5.0) at tier 1000, held out 0.016 (z 3.7) and 0.038 (z 3.5); copying complete in distribution,
  held-out swapped following 0.64 to 0.65 (the recorded arm 0.66 to 0.69, `m1` 0.65 to 0.67, `g1`
  0.78 to 0.80). Filler CE 3.8885 against
  the recorded arm's 3.9106 and `m1` / `g1` 3.886 / 3.889: the 0.022 gap is the code since
  2026-10-05, the change not identified. The held-out tier 100 cell is over 3 sigma on all three
  arms (`m1` 3.1, `g1` 3.3, `cfb` 3.7), by the letter of the R0b rule a fail at tier 100 held out
  on one seed, stated as a caveat; tier 1000 is the known residual. On the afternoon of
  2026-10-25, the two leak candidate arms and the clean-slice CE re-read. The leak arms (`docs/measurements/leak_candidates_micro.md`; the copy-first warm-up
  recipe on the flagged splits, read against the recorded warm-up arm): both candidates are
  falsified, no cell near the bar. The paired prior-controlled `delta` at tier 1000, warm-up arm
  minus leak arm (the lever needed z +3, the leak arm binding clearly less), reads for `m1` (the
  first name mention masked) +0.0115 (z 0.7) in distribution and +0.0147 (z 1.3) held out, for
  `g1` (support from the gold card only) +0.0017 (z 0.1) and +0.0118 (z 0.9); tier 100 sits inside
  noise the other way (z -0.6 to -1.5). Final entity `delta` at tier 100 / 1000 in distribution
  `m1` 0.024 (z 3.6) / 0.073 (z 5.2), `g1` 0.020 (z 3.0) / 0.083 (z 5.6), against the warm-up arm's
  0.015 (z 2.6) / 0.085 (z 6.4); at 100M every cell inside 3 sigma on both arms (copying there at
  0.88 to 0.90 in distribution, the recorded arm 0.95 to 0.97); at the final gold top-1 1.000 and
  swapped followed 1.000 in distribution on both. So the tier 1000 residual is neither the name's
  first mention nor the gold-absent documents' value spans; the prior drifting and the input path
  stay open. Side finding: `--support-gold-only` raises held-out swapped-card following from about
  0.67 (`m1` 0.65 to 0.67, the warm-up arm 0.66 to 0.69) to about 0.79 (0.78 to 0.80) at no cost,
  a candidate for the real-run data spec. And the warm-up recipe on the reviewed code reaches
  filler CE 3.886 (`m1`) and 3.889 (`g1`) against the recorded warm-up arm's 3.9106 and the
  original arm (c)'s 3.8842, so the 0.026 gap the record calls unexplained was the code of that
  week, not the warm-up. The clean-slice re-read (`docs/measurements/clean_validation_slice.md`;
  CE per exit 1 / 2 / 3 on `inject_val_clean` arm P 3.3478 / 3.2739 / 3.2669, arm C 3.3503 /
  3.2696 / 3.2650, arm E4 3.3474 / 3.2806 / 3.2740; at exit 3 on `inject_val` 3.4092, 3.3844 and
  3.4201; these 40 batches are not the trainer's): the control's lead over arm P at exit 3 is
  0.025 on `inject_val` and 0.002 on clean text, so the control's lower held-out CE in the record
  (3.2195 against 3.2321) was the validation documents' presence in rows only the control trains
  on (the review's 6.7%), not better language; on clean text arm P and the control are equal and
  E4 is 0.007 behind P (the pilot's noise 0.013); the depth gains repeat (exit 1 to 3 about 0.08 on
  every arm and slice). Earlier on 2026-10-25, the third micro retriever arm, `seg_e3`
  (`docs/measurements/infonce_micro.md`, the same section as the other two): the `w0e3` recipe
  with the selection loss weighted per query segment ends at held-out CE 4.3671 against the fresh
  twin `w0e3b`'s 4.3996 (0.033 better; the gap -0.008 at 20M and 40M, then -0.018 / -0.028 /
  -0.031 / -0.033 at 60M / 80M / 100M / final), better in every condition (gold 4.6171 against
  4.6540, mixed 4.4814 against 4.5230, distractors 3.9025 against 3.9179, none 4.3248 against
  4.3563), selection equal (the eval's `selection:` 0.7503 against 0.7538, lower is better; token
  top-1 0.3138 against 0.3070), store recall@5 / @20 at loop 1 0.001 / 0.004 against 0.000 /
  0.002 (gate z -205.0, failed, as expected with no contrastive term). The weighting costs nothing
  on selection and the gain reaches the rows with no evidence, so it is a language gain (the token
  weighting's pull toward the long documents through the selection term is the candidate
  mechanism; the review measured that pull on the pilot corpus's biographies, which this QA slice
  does not have). One seed, and 0.033 sits at the edge of the micro seed spread of about 0.03, so
  on the QA slice alone it is a direction. The paired per-token read ran 2026-10-25 on
  `inject_val` (filler only; `eval_exit.py --pair` refuses a split with evidence; 148,450 supervised tokens in 162 clusters, the 1,104,743 being the trainer's eval count;
  `ckpts/inject/exit_seg_e3_vs_w0e3b.{json,log}`): the twin 0.085 / 0.062 / 0.060 nats behind at
  exits 1 / 2 / 3, per-token z +10.0 / +8.5 / +8.1, about 1.9x the same instrument's seed spread of 0.032, so the
  gain is real between these two runs and general to text the arms never trained on; its
  attribution to the weighting rests on one seed. The segment default was the user's call before
  the real-run recipe (decided 2026-10-25: `segment`). On the evening of 2026-10-24, the other two micro retriever
  arms (the first pass only section): `l0e3`, the contrastive
  term on the first pass alone, reaches store recall@5 / @20 at loop 1 of 0.049 / 0.122 against
  the every-loop arm `w25e3`'s 0.059 / 0.133 (bge 0.829 / 0.897), loops 2 and 3 at 0.010 and
  0.005 at k5, held-out CE 4.4404 against the fresh twin `w0e3b`'s 4.3996 (a cost of 0.041 nats;
  the recorded every-loop pair cost 0.074), gate failed (z -174.3); the twin reads 0.000 / 0.002
  (gate z -205.6) at final CE 4.3996, 0.019 from the old `w0e3`'s 4.3809, inside the micro seed
  spread of about 0.03. The term on the first pass alone keeps about five sixths of the small
  recall gain at a bit over half the language cost, so the cost sits in the trunk's change and not
  in the later passes, which find almost nothing; the decision of 2026-10-21 (no retriever term in
  the real-run spec, the model's own retrieval revisited at the real run's size) is unchanged.
  Earlier on 2026-10-24: R9, the written bridge
  (`docs/measurements/written_bridge_pilot.md`): held-out 2-hop 0.997 against chance 0.191 at the
  final (0.982 already at 10M), 3-hop 0.997, read sites past the first adding +0.002 +- 0.001,
  4-hop at chance because the model writes the two steps it was trained on; at this shape
  reasoning composes in written tokens, not in the loop. The eval packer bug (fixed 2026-10-24,
  same record): every left padded decode with evidence before the fix read its buffer shifted.
  The A3 generation and A7 pathway re-reads on arm P's final: A3 weak in generation (EM 0.05,
  abstention 0.77 under gold; likelihood unchanged), A7 oracle-buffer EM 0.09 against the model's
  query at 0.001; the A4 edit flip read still unreadable on the model's query route (1 of 2,000
  answered before the edit). The A4 flip read on the bge route (`eval_store.py edit --route bge`,
  the serving configuration; 95 of 2,000 answered before the edit, same record): delete flip 0.90
  +- 0.035 (n 72, stuck 7) against the bar 0.70, so the facts the model answers from live in the
  store; edit flip 0.13 +- 0.07 (n 23, follow 3, stuck 1), and 0.09 follow with the edited chunk
  attached directly, a generation weakness on real questions, not a store failure. Chunk-order caveat: store chunks are 128 tokens, best first, while training used whole passages with the gold shuffled; under the prefix rotation best-first puts the top chunk farthest from the query. The size of this effect is unmeasured; a shuffled re-read is running 2026-10-25. Arm E4
  (`docs/measurements/expert_count_pilot.md`: the expert count is a cost choice); the repository
  review (`docs/review_2026-10-24.md`).
- **Built 2026-10-26:** launch path L1 to L5 (`docs/plans/RUN2.md`, the launch-path table; uncommitted
  in the tree). (1) The model set: the IR expert's inner attention is no longer built under
  `direct_read` (every config), an old checkpoint's `moe.experts.{i}.attn.*` keys of IR experts are
  dropped on load (`utils.drop_ir_inner_attn`, one warning naming the count; the strict resume
  paths stay strict, so a run checkpoint written before today cannot resume, none needs to),
  `flops_per_token_fwd` no longer counts that attention and `evidence_prefix_attn_flops` bills the
  key/value reader's prefix attention (not wired into a trainer), `detach_loops` clamped to
  `n_loops - 1` and the sampled depth, the fp32 balance shadow stamped on the buffer's version, data
  pointer and device, `check_store_embed_dim` (`ir_dim == 384` with the port); 24 test files pass,
  the three recorded checkpoints (`seed_micro.pt`, the warm-up arm's 100M save, arm P's final) load
  strictly with logits bit-identical to HEAD, a detached Sonnet review found no high or medium issue
  (two lows folded in). Constructor prints: `config_micro` 34.9M / 31.3M / 61M FLOP per token (seq
  1024), `config_pilot` 97.4M / 83.3M / 227M, `config.yaml` 380.2M / 221.0M / 464M (HEAD 381.7M /
  222.4M / 492M; the 383.5M / 224.2M / 502M in older notes was already stale); the per-evidence-token
  encoder figure is unchanged (micro 6.6M, pilot 39.4M, shape L 118.4M). (2) The trainer: multi-GPU
  on `sft.py --evidence`, one process per GPU, `batch_size` per GPU (8 x 4096 x 2 x 8 = 524,288
  tokens per update), per-rank interleaved document shares, lockstep loaders, gradients averaged in
  fp32 buckets by `pretrain.all_reduce_grads` (no DDP wrap: the LM heads run outside the model
  forward), counts, stop requests and loss means in one float64 all-reduce per log step, rank 0
  alone writes saves behind barriers, sharded evals merged; resume verification with exit 30 on the
  sft path, `total_steps` stored and warned on change, the token grid snap in both trainers,
  `keep_every_tokens`, the groundedness gradient-checkpointing tripwire; a detached Sonnet review
  found two high and two medium issues, all folded in; the 2-GPU against 1-GPU check cannot be
  bit-exact and runs in the cluster's first hour. (3) The run's config and launch: `config_run2.yaml`
  and `config_run2_control.yaml`, the seed `ckpts/inject/seed_run2_grounded.pt` (gitignored),
  `scripts/launch_run2.sh` and `scripts/launch_run2_control.sh`, `scripts/reads/` (the per-save read
  scripts, tracked); the trainer prints 330.4M total / 171.1M active / 459M FLOP per token with the
  key/value reader (body 252M + heads 100M + attention 107M at seq 4096), the encoder 118.4M per
  evidence token, so RUN2's Model and Cost replace their estimates (about 333M / 174M / 502M): the
  with-evidence line is 459 + 118.4 x 1.55 = 643M (635M at RUN2's own ratio 1.49), about 67k tokens
  per second per H100 and 1,040 to 1,220 hours for 280B (the control 27 hours); the 30-step smoke
  at shape L on the 5090 read 25.4k to 27.4k tokens per second at 16.5 GB peak, step 0 CE 11.17,
  step 30 10.96, save and exit 10 clean. (4) The corpus pipeline: `scripts/build_run2_corpus.py`
  (`docs/plans/RUN2_corpus.md`), the pilot recipe at scale on the existing builders, stages
  idempotent with markers, T = 93.33B prompt tokens per split and 3 epochs (the trainer counts
  epochs, not tokens), chain share 0.03 (RUN2's 0.10 would cost 6.8 TB peak and 3 to 4 days; 3.7 TB
  and about 2 days after the filler download at 0.03), chat replay held at the supply limit (about
  0.5B unique non-holdout chat; the share goes to the filler), per-slice targets bios 89.0B, QA
  1.53B, plain chains 1.4B, bridge chains 1.4B, control 10B; the smoke at scale 0.0005 ran every
  stage in 23.7 minutes with `md5_by_slice` agreeing across warm and main; not run locally: the
  filler download (reachable at 1.00x repeat only with fineweb widened to all crawls, a
  patch the driver applies in memory) and the two store builds; missing builder flags are listed in
  the sheet; the evidence ratio of this corpus is about 0.4, not RUN2's 1.49, so the with-evidence
  rate is better than the Cost table's estimate (about 85k per second and 910 hours at 0.4). The
  chain share and the chat replay are decisions of the main thread as routine (RUN2.md rows 12 and
  13), pending the user's confirmation. (Superseded 2026-10-27 in the token target, the split size,
  the chain share and the evidence ratio: the next bullet.)
- **Built 2026-10-27 (and decided, the user: 800B):** for the allocation on MareNostrum 5 (BSC AI
  Factories, the ACC partition, 4 x H100 64 GB per node, 72 hour jobs, 5,000 GPU hours applied and
  4,500 targeted, 4 TB project disk, 2 TB transfer, no internet on any node, Singularity only;
  RUN2.md "The allocation and the cluster"), the token target is 800B (RUN2.md row 14: 266.67B
  unique prompt tokens per split x 3 epochs, was 280B as 93.33B x 3): about 3,300 H100 hours at 67k
  tokens per second per GPU (about 2,400 at 93k, the corpus's evidence ratio 0.045), about 206 hours
  on 16 GPUs in at least 3 chained 72 hour jobs, the floor about 54k per GPU. `config_run2.yaml`:
  `target_tokens` 800e9, `warmup_fraction` 0.000655, the kill at 8B, evals every 4B, accumulation 1
  at 16 GPUs, uploads off; the control at constant LR after its warmup (`lr_min_factor` 1.0). Built,
  uncommitted: the corpus driver at 800B (`docs/plans/RUN2_corpus.md`: per split bios about 262.3B,
  QA 1.53B, plain and bridge chains 1.4B each, persons 3x the pilot's, a seeded `qa_shuffle` before
  the order, the warm split (about 28B) and the control (10B) as prefixes of the main order, epoch 0
  of the main split led by the warm split's documents, row estimates for 1 to 16 ranks, resumable
  stages with locks and a progress file; disk peak about 3.06 TB, standing 1.55 TB; the smoke at
  scale 0.0005 every stage ok in 28.5 minutes), `scripts/mirror_sources.py` (the filler pushed from
  the PC through a transfer node), resumable builders, the trainer additions (`--plan-split`,
  `--max-epochs-this-split`, `--grad-accumulation-steps`, `--estimate-rows-only`, `--keep-stop`, the
  switch and head checks, rows cache version 2), `scripts/slurm/` (the job chain with its watchdog
  and HALT rules), `scripts/ops/` (push, pull with a 2 TB ledger, `supervise.py` polled by a local
  Claude Code `/loop`), `scripts/export_weights.py`. Three Opus final reviews (Slurm and ops,
  corpus, trainer) left no high issue in the corpus and the trainer, the Slurm review's two highs and
  every medium and low fixed, and a detached Sonnet review of the trainer fixes found no high or
  medium issue. Nothing has run under real Slurm. The user's steps: `docs/plans/RUN2_handover.md`.
- **Next, in order:** the user's cluster access and day 1 (`scripts/slurm/README.md` "Day 1": the
  `module` command in a non-interactive shell, one supervisor action seen in `squeue`, the 2-GPU
  against 1-GPU check stopped after about 100 steps, the first hour's rate against 54k per GPU), the
  corpus build, the 10B control, then the warm-up with `STOP_AFTER='4000M'` and the switch
  decision; during the run table B of `docs/plans/RUN2.md`.
- **Open for the user:** the control's exposure (at 10B it sees about 3.75% of each person's
  renderings: probe tiers at 2,667 / 26,667 exposures per pass, or a denser build of its own);
  whether to read the warm-up at 12B in addition to 4B; where the corpus is built (the cluster,
  about 400 GPU hours, or the PC, about 3 TB of local disk where D: has about 125 GB free); 6f's
  abstention reward against the A3 re-read.

### Open items

Mirrored in both "Now" sections; a row leaves when it is done.

| item | status | the read that gates it | waiting on |
|---|---|---|---|
| the review's code changes | landed 2026-10-24, committed as c79002e (tests pass on the GPU; a detached Sonnet review found no high or medium issue): fp32 CE, top-1 and `p_max` sums in `mtp.py`; the log-space selection loss with `evidence_selection_weighting` (`token` default, `segment`); `evidence_contrastive_loops` default `[0]`; the groundedness head input detached under the key/value reader; the row estimator replaying each reader's rule and a 4093-token training cap in the evidence builder (the reader itself unchanged by decision); the evidence padding cut into 128-token blocks; the dense-slot GEMM skip; the frozen-gate skip; fp32 selection scores under loss-free. Held: the eval-pass rewrite, deferred to 6a; the `eval_chains.py` bridge-mode items (in `--generate-bridge` the latent verdict lines suppressed and the kept 3 minus kept 1 line labelled bridge-internal; the evidence built once per batch) | none | 6a (the eval pass); a free card for the `eval_chains.py` identity check |
| A4 edit, a flip read routed through bge | closed: read 2026-10-24 (`scripts/eval_store.py edit --route bge` on arm P's final, retrieval through the store's bge query keys, the serving configuration; `ckpts/inject/pilot_p_ladder2_store_edit_bge.*`; 95 of 2,000 answered before the edit): delete holds (flip 0.90 +- 0.035, n 72, bar 0.70), edit fails on generation (flip 0.13 +- 0.07, n 23; the edited chunk attached directly followed 0.09); the edit half goes to 6f with A3 | none | nothing |
| R5, the winner, eval only | closed: read 2026-10-25 evening (RUN2.md row 7; `docs/measurements/r5_bridge_ladder.md`): arm P's final already carried A1 to A7 and its 4-hop reads by read sites (at chance), the bridge final the 4-hop generate read (0.206 against 0.210); the new read, the A-gate ladder on the bridge final (`ckpts/inject/r5_bridge_ladder.sh`), makes the chain-only bridge SFT a demonstration of written composition, not a usable checkpoint (biographies and the tier 1000 residual erased, copying halved, answering and abstention replaced by the bridge form, clean-slice CE 2.8x, the suite's headroom down to a third), so 6f's mix carries the pretraining slices and every A gate is read on the shipped checkpoint; the HotpotQA bridge by read sites has no instrument, a build item | held-out 4-hop by read sites and at D = 4; HotpotQA bridge by read sites; A1 to A7, done | nothing |
| a clean validation slice (documents neither builder used) | closed: built 2026-10-24 (`scripts/prepare_clean_val.py`, `data/prepared_pilot/inject_val_clean`, 974 windows, 650,125 tokens; verbatim 64-token overlap with the QA training rows' supervised tokens 0.99% against 13.3% for `inject_val`), read 2026-10-25 (`bash ckpts/inject/read_clean_val.sh`; `docs/measurements/clean_validation_slice.md`): the control's exit 3 lead over arm P is 0.025 on `inject_val` and 0.002 on clean text, so its lower held-out CE was contamination, not language; E4 0.007 behind P; depth gains about 0.08 on every arm | `eval_exit.py` on the arm P, C and E4 finals on both slices, done | nothing |
| a value-to-name closed-book probe | proposed, mode not built | arm P's final | a build |
| R4, the coda corner | closed: dropped 2026-10-24 (the user), obsolete on R9 (composition at 0.997 held out once the intermediate entity is written out); the coda stays unbuilt | none | nothing |
| R4b, the sublayer half | closed: decided 2026-10-25 (the user, RUN2.md row 5): today's one block per pass, the three-sublayer core not built, a cost choice like the expert count, no arms | none | nothing |
| three micro retriever arms (a fresh `w0e3` twin on the reviewed code; the contrastive term on the first pass with the trunk gradient on; per-segment selection weighting) | all three read, two 2026-10-24 and `seg_e3` 2026-10-25 (`bash ckpts/inject/launch_review_arms.sh`; `docs/measurements/infonce_micro.md`): the twin `w0e3b` final CE 4.3996, recall@5 0.000; `l0e3` recall@5 / @20 at loop 1 0.049 / 0.122 against `w25e3`'s 0.059 / 0.133 at 0.041 nats against the twin (the every-loop pair 0.074), gate failed, so the cost sits in the trunk's change and the decision of 2026-10-21 stands; `seg_e3` (per-segment selection weighting, no contrastive term) held-out CE 4.3671 against the twin's 4.3996 (0.033 better, every condition including no evidence, the gap growing from 60M on), selection equal (0.7503 against 0.7538), recall@5 0.001; the paired per-token read on `inject_val` (2026-10-25) puts the twin 0.060 nats behind at exit 3, z +8.1, about 1.9x the same instrument's seed spread of 0.032, so the gain is real between these two runs, attributed to the weighting on one seed; `segment` decided as the real run's default 2026-10-25 (the user, RUN2.md row 2) | store recall and held-out CE against `w0e3b`; the paired exit read, done | nothing |
| two leak candidates (value-first templates; distractor-collision spans) | closed: builder flags `--mask-first-name-mention` and `--support-gold-only` (`.mask` only), splits `inject_retrieval_s100a50m1_train`, `s100a50g1`, `s15a50m1`, `s15a50g1` built 2026-10-24; arms `bash ckpts/inject/launch_leak_arm.sh m1` and `g1` run and read 2026-10-25 (`docs/measurements/leak_candidates_micro.md`): both falsified, the paired tier 1000 `delta` against the warm-up arm at z 0.7 / 1.3 (`m1`) and 0.1 / 0.9 (`g1`) by form against the bar z 3, so the residual is neither the first name mention nor the collision spans; side finding: `--support-gold-only` raises held-out following from 0.67 to 0.79 at no cost (decided 2026-10-25: carried into the real corpus builder behind a flag, RUN2.md row 3); the same-code twin `cfb` (`bash ckpts/inject/launch_leak_arm.sh cfb`, the flagless splits on today's code, read 2026-10-25 evening) confirms both falsified (every paired cell within z 1.1), puts the slower copying at the switch (0.88 to 0.90 against `cfb`'s 0.95 to 0.96) on the flags and the 0.022 filler CE gap to the recorded arm on the code, and reads tier 100 at z 2.9 / 3.7 by form (held out over 3 sigma on all three arms; accepted 2026-10-25 as the recipe's known state on one seed, no second seed, RUN2.md row 9) | closed-book rank at 100M and at the final against the copy-first warm-up arm (`compare_cf_vs_leak_{ARM}.log`) and against `cfb` (`compare_cfb_vs_leak_{m1,g1}.log`), done | nothing |
| R6, pretraining-text facts | closed: skipped 2026-10-24 (the user); its mechanism is read at micro (arm (a) stores plain-text facts, tier 100 `delta` 0.378; arm (b), spans masked, stores nothing; arm C repeats arm (a) at pilot shape) and the run spec uses that lever | none | nothing |
| the goal's reasoning clause reworded | closed: decided 2026-10-25 (the user, RUN2.md row 4), open since 2026-10-17, on R9 (composes in written tokens, not in the loop): "facts in the store through the retrieval pathway, reasoning in written tokens, the looped trunk as computation per token"; the original "reasoning in the looped trunk" kept as the original goal and the aspiration to revisit after the real run (NEXT.md "The goal") | none | nothing |
| 6f, reasoning post-training | planned; its reasoning form written tokens (R9, 2026-10-24), with multi-hop rows at the serving depth; its mix carries the pretraining slices beside the bridge rows at a lower LR than the pilot's continuation, the A gates and the clean-slice CE read at every save under a kill rule (R5, 2026-10-25: chain-only bridge SFT lost every other gate) | the A3 generation re-read (weak, 2026-10-24) and A4's edit half (fails on generation, 2026-10-24); decode with evidence through a KV cache | the real run |
| 6a throughput, 6c corpus on the rented cluster, 6d run spec | 6d decided 2026-10-25 (the user, every row of `docs/plans/RUN2.md`'s first table on the recommended option; drafted the same evening: recipe by rung, cost table for the application, build list; the draft found the plan's 2.5k-hour figure was for the blueprint width 1024 while `config.yaml` is width 768, the evidence trainer single GPU, the real corpus builder unbuilt with the staircase visibility unresolved under the key/value reader); 6c and the rest of the spec's build list next, in RUN2.md's order; 6a after the spec (2026-10-21), carrying the deferred eval-pass rewrite | none | the build list (R5 read 2026-10-25) |
| the real run's launch | built 2026-10-26 (launch path L1 to L5, RUN2.md) and 2026-10-27 for MareNostrum 5 at 800B (RUN2.md row 14 and "Built 2026-10-27"): the corpus driver at 800B with resumable builders and the filler mirror, the trainer additions for chained jobs, the Slurm job chain with its watchdog, the PC-side ops and supervisor, the weight export; the final reviews left no high issue; the user's steps in `docs/plans/RUN2_handover.md` | day 1 on the cluster (`scripts/slurm/README.md`): the `module` command in a non-interactive shell, one supervisor action seen in `squeue`, the 2-GPU against 1-GPU check (STOP after about 100 steps; cannot be bit-exact, the loss and the per-rank counts agree within noise), the first hour's rate against the 54k floor (about 67k expected at RUN2's ratio, about 93k at the corpus's 0.045) | the user's cluster access; the user's three calls (the control's exposure, a 12B warm-up read, where the corpus is built) |

### Record

In date order, oldest first. Superseded sentences are in the past tense; nothing is cut. The
ladder's done steps and the standing notes close the record.

Earlier, still binding: Phase 4's graft arms were killed at 10.08M tokens on the content gain
(+0.058 / +0.081, [evidence_arm_a.md](../measurements/evidence_arm_a.md),
[evidence_arm_b.md](../measurements/evidence_arm_b.md)); the graft branch is closed (seven failed
gates, one graft condition). The in-context ceiling on `evidence_fixed` is 3.18 nats pooled
([evidence_ceiling.md](../measurements/evidence_ceiling.md) section 5), so the 10M arms reached a
sixth of it on gold gain and 2% on content gain. R0 failed: `loop_scale` stays at the trunk's rate
in graft arms.


**The key/value reader copies completely in distribution, and arm (c) fails R0b on a leak present
before copying existed. With a copy-first warm-up the R0b rule holds through tier 100 and tier
1000 still leaks. Decided 2026-10-05: R0b is accepted on the warm-up arm with the tier 1000 leak
known, and the warm-up enters the pilot as a requirement. The swap rate 0.30 branch (read
2026-10-06) leaves the tier 1000 leak where it was. Decided 2026-10-06: the pilot's swap rate stays
at 0.15, the placeholder name arm and a chain depth arm are approved, and the learned exit gate arm
is dropped. On 2026-10-08 the placeholder name arm ran and falsified the dose lever (halving the
real-name-with-real-value supervision left tier 1000 where it was), so the copy-criterion span
weight is dropped and the pilot's lever list is the copy-first warm-up alone at swap rate 0.15. The
same day the chain depth arm ran and read inconclusive
([chain_depth_micro.md](../measurements/chain_depth_micro.md)): the micro model copies an
answer-type entity from the buffer but never learned which one (1-hop accuracy at chance), so the
readable precondition fails and L1 stays unmeasured at micro scale.** R0b ran from
2026-10-02 to 2026-10-05 ([r0b_micro_pilot.md](../measurements/r0b_micro_pilot.md)), all at the
micro shape (35M, `config_micro.yaml`), from one seed, on `data/prepared_inject`:

- **Arm (a), full CE, 300M tokens: the instrument reads.** Closed-book entity `delta` in
  distribution 0.009 / 0.063 / 0.378 / 0.467 at tiers 1 / 10 / 100 / 1000 (tier 1000 rank 0,
  top-1 1.000); on the held-out template only 0.001 / 0.009 / 0.023 / 0.104. Memorization is mostly
  surface form. In context, the same checkpoint copies a card from its prompt at 0.29 / 0.26 / 0.12
  / 0.003 (in distribution) and 0.20 flat over tiers (held out).
- **The 20M arm (c) smoke was read before any copy circuit existed.** The new prompt read
  (`closed_book_rank.py bios --evidence prompt`, the card as prompt text, no port) equals the
  closed-book read for entities at every tier on that save (0.491 / 0.496 / 0.480 / 0.394), so the
  smoke said nothing about the reader.
- **At matched 100M tokens the cross attention reader does not copy and the key/value reader
  does.** The key/value reader (`sft.py --reader-kv`: the evidence states become leading keys and
  values of `shared_attn` in every loop, through its own projections, rotated as if the evidence
  were the text right before the document) ranks the gold card's value at 0.003 / 0.003 / 0.002 /
  0.000 in distribution (top-1 0.92, flat in exposure) and 0.02 held out, and follows a swapped card
  92% of the time with a memorization ratio under 0.03. The cross reader, resumed to 100M as the
  matched control, ranks it at 0.47 / 0.46 / 0.35 / 0.02 and follows a swapped card 2% of the time,
  although the same model copies from its prompt at 0.10 to 0.13. Training the key/value read also
  trained in-context copying (prompt 0.003). The selector separated in both runs by 100M on the
  selection loss alone (gate frozen): external mass 0.39 to 0.46 with gold, 0.02 to 0.12 without.
- **Arm (c) already leaks at 100M under either reader.** Closed-book `delta` in distribution at
  tier 100 / 1000: cross 0.037 (z 10.5) / 0.203 (z 20.3), key/value 0.026 (z 5.2) / 0.149 (z 12.3),
  arm (a) at the same tokens 0.099 / 0.332. The reader that copies leaks least but still fails the
  pass rule at tier 100. These were mid-cosine readings; the full arm (c) confirmed them (below).
- **Arm (b), masked fact spans, 300M tokens: nothing stored at 3 sigma** (2026-10-04). Closed-book
  entity `delta` in distribution 0.001 / 0.003 / 0.002 / 0.011 (z at most 1.7), held out -0.001 /
  0.001 / -0.000 / 0.013 (z at most 2.4); flat at matched 100M as well. The tier 1000 entity cell
  is the one to watch (0.012 at z 2.8 pooled over forms, up from 0.001 and 0.003 at 100M). The
  in-context input path does not leak, so the arm (c) leak most likely comes from the spans arm
  (c) supervises (81% of the fact tokens, real name in the prompt, unswapped target 85% of the
  time); sharpened 2026-10-05 by the warm-up arm: the store is written by supervised real-value
  spans, not by the input or the card path, and also after the reader copies. The
  masked arm copies from its prompt at 0.15 in distribution and 0.05 held out, flat over tiers 1 to
  100 and worse at 1000; better than arm (a) held out and at tiers 1 and 10. `compare` at matched
  100M: held-out form holds, in-distribution form fails on arm (c) at tier 100 (corrected
  2026-10-05: the held-out hold did not last, see arm (c) in full).
- **Arm (c) in full, key/value reader, 300M tokens: it copies, and it fails the rule on both
  forms** (2026-10-05). Resumed from the 100M save, 299.22M tokens, final filler validation CE
  3.8842 (arm (a) 3.9037, arm (b) 3.8769). In distribution the gold card ranks the value at 0 with
  top-1 1.000 at every tier in every class (0.999 for entities at tier 10), a swapped card is
  followed on every item with `mr_ll` 0.000, and the prompt copy reads 0.0002 or better; held out
  the gold card ranks entities at 0.010 to 0.024 (top-1 0.58 to 0.74) and a swapped card is
  followed 0.61 to 0.64 of the time. Closed-book entity `delta` in distribution 0.002 (z 0.3) /
  0.003 (0.5) / 0.027 (4.1) / 0.120 (8.8), held out 0.000 / -0.001 / 0.013 (z 3.6) / 0.071 (z 8.2).
  `compare full masked retrieval_kv`: FAILS on both forms at tier 100, and tier 1000 is a leak on
  both. The held-out tier 100 cell held at 100M (0.006, z 1.6) and sits 3.3 to 3.9 sigma off its
  prior from 150M on.
- **The arm (c) leak is present before copying and flat after 100M** (2026-10-05). Entity
  `delta` in distribution at 100 / 150 / 200 / 250 / 300M: tier 100 0.026 / 0.029 / 0.023 / 0.024 /
  0.027, tier 1000 0.149 / 0.140 / 0.115 / 0.112 / 0.120; paired from 100M to final +0.001 (z 0.2)
  and -0.029 (z -2.0), where arm (a) gains +0.279 and +0.134 over the same tokens. Gold top-1 in
  distribution is 0.91 at 100M and 1.00 from 150M. The 50M save does not copy yet (gold top-1 0.007
  to 0.013 through tier 100) and already leaks: 0.010 (z 4.0) at tier 100 and 0.105 (z 14.4) at
  tier 1000, equal to arm (a) at 50M at tier 100 (paired +0.000, z 0.1). Established: the leak is
  there before copying and does not grow after 100M. The warm-up arm (next bullet) reads it as both
  an offset written before copying and a level that training maintains. A dose prediction
  (about 0.68 of the arm (c) spans carry the real value next to the real name, so less leak than
  arm (a) at 50M) failed on `delta`. At the final save arm (c) holds 7% / 26% of arm (a)'s stored
  signal at tier 100 / 1000 in distribution. Copying emerged between 50M and 100M for the key/value
  reader.
- **The loop reading on arm (a): a weakly stored fact is recalled better after two passes**
  (2026-10-04, `closed_book_rank.py bios --n-loops`). In distribution the entity `delta` at depth
  1 / 2 / 3 is 0.269 / 0.369 / 0.378 at tier 100: paired, depth 1 to 2 is +0.100 (z 19.7), depth 2
  to 3 +0.008 (z 6.5), with the prior unmoved, so the storage falsifier's condition is met at tier
  100 (in every class). The gain is on the real name (1.21 nats at tier 100 against 0.14 for the
  fresh-name prior), depth 1 reaches 71% of the depth 3 `delta`, and held out nothing grows with
  depth. At tier 1000 the real name is saturated after one pass (rank 0.0008, top-1 0.98); the
  `delta` there grows (0.393 / 0.453 / 0.467) only because the prior drifts toward chance. The
  block's weights are shared across passes, so this reads as recall of a weakly stored fact being a
  two-step computation, not as extra capacity in later passes. The axiom is reworded to say so
  (Decisions, loops), and every closed-book leak read is taken at full depth.
- **The per-exit read: depth pays evenly on this corpus** (2026-10-05, `scripts/eval_exit.py`,
  eval only, no evidence attached). On `inject_val` (1,104,743 tokens, 1,492 documents) arm (a)
  final reads CE 4.0072 / 3.9132 / 3.9071 at exits 1 / 2 / 3: the second pass +0.0940, the third
  +0.0061. The gain from exit 1 to 3 is +0.107 to +0.122 over the seven least confident deciles of
  `p_max` at exit 1, +0.072 in the ninth and +0.024 in the most confident; a confidence exit rule
  is no better than a mix of fixed depths at the same mean passes (3.9237 against 3.9106 at 2.43
  passes for tau 0.5, equal within 0.0002 at tau 0.99); the entropy-regularized optimum at beta
  0.1 is near uniform over exits (0.343 / 0.286 / 0.371), and that is a bound, since it chooses
  with the label. Arm (c) final reads the same (3.9902 / 3.8919 / 3.8859). A learned exit gate has
  little to learn here; the gate arm was dropped on 2026-10-06 (ladder step 8).
- **The copy-first warm-up arm, done 2026-10-05: the rule holds through tier 100, tier 1000
  leaks** (`inject_retrieval_kv_cf`). Arm (c) with the key/value reader, first 100M on a split where
  every fact span in a gold-present document carries a substitute the card also carries (swap rate
  1.0), then on the arm (c) split to 299.19M tokens; final filler CE 3.9106 against 3.8842 for the
  original arm (c), a 0.026 gap that opens after the switch and that the logs do not explain (explained 2026-10-25: the code of that week, not the warm-up). At the
  switch both criteria were met: gold top-1 in distribution 0.973 / 0.973 / 0.959 / 0.953, and
  closed-book `delta` within 3 sigma in every cell at 50M and 100M (the original arm (c) at 50M read
  0.010, z 4.0 and 0.105, z 14.4). After the switch, entity `delta` in distribution at 100 / 150 /
  200 / 250 / 300M: tier 100 -0.006 / 0.011 / 0.013 / 0.013 / 0.015 (z 2.6 at the final save), tier
  1000 -0.001 / 0.081 / 0.095 / 0.087 / 0.085 (z 6.4); held out at the final save 0.007 (z 2.0) / 0.047 (z 5.1); paired
  200M to final +0.002 (z 0.5) and -0.011 (z -1.2). `compare full masked retrieval_kv_cf`: HOLDS on
  both forms through tier 100; tier 1000 is a leak on both forms by the rule's own reading, and
  dates leak at tier 1000 in distribution (0.106, z 3.6). The pre-registered expectation "tier 1000
  at most about 0.03" failed, and the falsifier "a `delta` climbing after the switch, paired z above
  3" triggered at tier 1000 (and, from 100M to final, at tier 100: +0.020, z 3.3). Against the
  original arm at the final save, paired per form: tier 100 -0.012 (z -1.5) / -0.006 (z -1.4),
  tier 1000 -0.035 (z -2.2) / -0.024 (z -2.2) in distribution / held out; lower everywhere, under 3
  sigma per form. Copying at the final save: gold top-1 0.999 to 1.000 and a swapped card followed
  on every item in distribution; held out followed 0.66 to 0.69, prompt copy 0.008 (top-1 0.84,
  against 0.017 and 0.76 in the original arm). Established: nothing is stored without a real-value
  target; real-value supervision after the reader copies still writes frequent facts (tier 1000 to
  0.08 to 0.10 within 50M to 100M tokens, then flat); tier 100 is inside 3 sigma on both forms with
  the warm-up. So the original arm's leak is in part written before copying and in part a level
  that training maintains while real values are supervised next to the real name. Open: what sets
  that level (the swap rate, the gold-drop documents, residual span loss after copying). Decided
  2026-10-05: R0b is accepted on this arm, the tier 1000 leak recorded as known, and "no real-value
  fact supervision before the reader copies" enters the pilot as a requirement. The warm-up is the
  better of two arms on one seed, not an optimum; its improvement over the original arm is a
  direction, not established per form.
- **The swap rate 0.30 branch, started 2026-10-05, read 2026-10-06**: the swap rate does not move the tier 1000 level in any large way. Final closed-book entity `delta` in distribution 0.005 / 0.004 / 0.010 (z 1.7) / 0.083 (z 6.2), held out -0.002 / -0.001 / 0.003 (z 0.9) / 0.039 (z 4.6), against the warm-up arm's 0.015 (z 2.6) / 0.085 (z 6.4) and 0.007 (z 2.0) / 0.047 (z 5.1) at tiers 100 / 1000. Paired branch minus warm-up arm (same items), entity: tier 1000 -0.002 (z -0.3) in distribution and -0.008 (z -1.4) held out, -0.005 (z -1.0) pooled over forms, so the pre-registered criterion (z at or beyond -3 per form or pooled) is not met; tier 100 -0.004 (z -1.0) and -0.004 (z -1.4), lower on both forms as a direction, and the rule holds with more margin (z 1.7 and 0.9). The raw-rank z -4.0 in `compare_cf_vs_cf30.log` (all classes, pooled over forms) is mostly the real name ranking worse in the branch (+0.014, z 3.8) with the prior also worse (+0.005, z 1.2), so the `delta` moves only -0.009 (z -2.1). Within the stated power this is no large effect, not no effect: a dose-sized effect (about 0.07) could not be separated from none. Copying holds: gold top-1 in distribution 0.999 to 1.000 in both arms, swapped card followed on every item in distribution, held out 0.654 to 0.673 against 0.663 to 0.690 (all classes), `mr_ll` the same within 0.005; prompt copy held out 0.0084 (top-1 0.834), the same as the warm-up arm. Filler CE 3.9131 against 3.9106 for the warm-up arm. The 150M `none` read was taken (0.003, z 0.6 / 0.072, z 5.3 in distribution); the 200M and 250M reads and the other 150M modes were not taken. Decided 2026-10-06 (the user): the pilot's swap rate stays at 0.15, the default. The user's reasoning: a leak of this size is not much of an issue for the pilot, and RL is planned later (for reasoning chains and generally), which can also act on small leaks. Caveat on record, not a counter-decision: RL is a Parked item (it unparks when pass@8 on the target task exceeds about 15%), and RL shapes behaviour at conflicts (prefer the card over memory) rather than removing what is stored, so the closed-book instrument would still read the leak; for small leaks the goal "facts in the store, not in the weights" is then read as behavioural at conflicts. The placeholder name arm is approved (ladder step 6; read 2026-10-08, below).
- **The seed-side instruments, done 2026-10-06** ([seed_instruments.md](../measurements/seed_instruments.md)).
  PopQA (20 candidates, prior control, 14,233 items): seed entity `delta` 0.0287 (z 11.7) / 0.0263
  (z 12.8) / 0.0356 (z 19.4) by tail / mid / head, concentrated in four cue relations (father,
  mother, capital, capital of: 0.151, tail 0.261 against head 0.113); the other 9,430 entity items
  read 0.0088 / 0.0075 / 0.0070. Paired arm minus seed at the entity tail the raw rank moved +0.0090
  (z 7.4, arm A) but the prior moved +0.0102, so `delta` moved +0.0012 (z 1.0): 10M of evidence
  finetuning shifted the answer prior and stored nothing. Counterfactual likelihood (1,409 SQuAD v2
  items): seed `mr_ll` 0.295 to 0.766 across frequency strata, a frequency-ratio prior (the port is
  neutral; correlation 0.636 with the log count ratio); the arms gain about 1.6 to 1.8 nats on both
  answers under a buffer and follow the swapped card in no stratum (paired gap +0.01, z 0.1 pooled,
  arm A). Method notes now in gates A1 and A3.
- **The placeholder name arm, done 2026-10-08: the dose lever is falsified**
  (`inject_retrieval_kv_cf_p50`, [r0b_micro_pilot.md](../measurements/r0b_micro_pilot.md)). Split
  `inject_retrieval_s15a50p50_train` (`--placeholder-rate 0.5`, swap 0.15, 298.26M tokens; 64,664 of
  129,438 gold-present documents renamed, everything else byte identical to the arm (c) split),
  branched from the warm-up arm's 100M save, 298.27M tokens in 68.5 minutes, final filler CE 3.9161
  (warm-up arm 3.9106). Final entity `delta` in distribution 0.005 / 0.001 / 0.0185 (z 3.0) / 0.0986
  (z 6.7), held out 0.000 / -0.003 / 0.0133 (z 3.7) / 0.0559 (z 6.4). Paired against the warm-up
  arm: tier 1000 +0.014 (z +1.3) / +0.009 (z +1.3), +0.011 (z +1.8) pooled; tier 100 +0.005 (z +1.8)
  pooled (the real name and the prior both rank better, -0.025 at z -4.6 and -0.014 at z -2.5 at
  tier 1000, so `delta` does not move). Criterion (follows dose at z at or beyond -3, falsified at
  pooled z above -2): **falsified**. Halving the real-name-with-real-value supervision (0.686 to
  0.346 of exposures) left tier 1000 where it was; neither dose prediction (0.043, 0.064) came true.
  `compare full masked retrieval_kv_cf_p50` HOLDS in distribution and FAILS held out at tier 100
  (0.0133, z 3.7), worse than the warm-up arm as a direction. Copying in distribution unchanged;
  held out degraded (gold top-1 0.53 to 0.68 against 0.60 to 0.74, swapped followed 0.51 to 0.54
  against 0.60 to 0.63, z down to -8), prompt copy held out better (top-1 0.84 to 0.87). Decided
  2026-10-08: the copy-criterion span weight is dropped; what writes the maintained level stays
  open (the gold-absent documents with the real name, the input path, the prior drifting); the
  held-out copy loss reads as the reader binding on the name (a reading, not established).
- **The chain depth arm, run and read 2026-10-08: inconclusive, the 1-hop curve is not readable**
  ([chain_depth_micro.md](../measurements/chain_depth_micro.md); ladder step 8). One epoch of 2M
  chain questions (104.66M prompt tokens, 439M evidence tokens, ratio 4.20) from `seed_micro.pt`
  with `--evidence --reader-kv`, run `chains_kv`, 102.66M tokens in 51.5 minutes on batch 16 x
  accumulation 2 after the batch 32 launch spilled in minute one. `[eval]` answer CE on
  `chains_val` 11.13 / 2.259 / 0.628 / 0.501 / 0.455 at 0 / 25 / 50 / 75 / 100M (top-1 per token
  0.871 at 100M); the selector stayed near uniform (`selection:` 0.495 to 0.440). Every
  `eval_chains.py` accuracy cell sits at chance at both saves: `chains_eval` 1-hop at depth 3 with
  all sites 0.104 (chance 0.115), 2-hop 0.187 (0.185), 3-hop 0.194 (0.206); held-out 2-hop kept 1
  0.188, kept 3 0.194 (chance 0.191), Delta +0.006 at sigma 0.013. The read sites carry content
  (answer CE per token 3.82 without a site, 0.49 with one) but not the choice: 0.455 per token is
  about 3.3 nats per answer against ln K of about 1.9 nats for a uniform pick among the
  candidates, so the model copies an answer-type entity and does not select which. No bug found in
  `eval_chains.py`. Readable precondition fails, so inconclusive under the pre-registered criteria;
  not a kill (every sub-hop cell at chance). `loop_scale` went to [1.34, 0.48, 0.07], the third
  pass's scale under 0.01 at 50M: the weak later loop case, on a task where no pass composed.


**The pilot spec is written ([PILOT.md](PILOT.md), 2026-10-09), and it cuts the pilot to what is
built.** A code inventory found most of the Phase 5 blueprint unbuilt (no coda, no tied table, the
selector is still the routed IR expert, one read site per loop, today's depth sampling and per-loop
CE, no union selection loss, no InfoNCE; no per-window retrieval or fact tagger on natural text, no
near-duplicate filter, no 0.4B-token store). The pilot therefore runs the R0b recipe on today's code
at a larger shape (99.1M total, 84.9M active, width 512, 6 prelude layers, 8 experts, seq 4096,
key/value reader) on a merged corpus from the existing builders (biographies plus filler 60%, QA
plus web plus replay 25%, synthetic chains 10%; 1B tokens per arm), with the copy-first warm-up
and swap rate 0.15, against a matched full-CE control. It reads A1(b), a new 1-hop gate before L1,
L1, A2, A3, A4, A5, A7 and A6 as benchmarks within noise; it cannot read A1(a), A6 through the
port or L2 at depth 4, and the ladder rungs R4, R4b and R6 are cut from the bottom (superseded 2026-10-23: the run order
brings them back staged). Measured
2026-10-09 on the probe shape: 44k to 50k prompt tokens per second at batch 8 x 4096 (20.4 GB) on
rows with little evidence, 2.7k to 3.9k on `evidence_train` (ratio 3.75) at the 4608 cap (fill 17
to 30%); a fit through the two gives about 16k on the merged corpus (ratio about 0.73), 16 to 28
hours per pathway arm, 35 to 45 GPU hours for the pilot, all estimates until the first hour prints
its rate; the chain slice (ratio 4.2) is the heavy one. Before the launch: a split merge tool, the
`eval_chains.py` instrument fixes, the pilot config and seed, three biography builds at 4096 and
three merges. The user decided the launch (arm P, 2026-10-10).**


**The pilot's build list was done earlier on 2026-10-10.** Built: the
split merge tool `scripts/merge_evidence_splits.py` with its test (seeded per-slice permutations,
an interleave drawn by remaining tokens, re-indexed chunk tables, a per-document chunk cap, an
evidence-free form, a per-slice md5 and a `.slice` sidecar; `--order-from` reuses another merge's
document order so a run resumed by position on the main split continues where the warm-up split
stopped); the `eval_chains.py` instrument fixes (per-question records, gold NLL against ln K for
every split and hop, the NLL at the first divergent token, a paired bootstrap delta between kept
site counts, the `*` marker in its own column); `config_pilot.yaml` (one file for both arms: the
groundedness and selection terms are never computed on the control's evidence-free split) and the
seed `ckpts/inject/seed_pilot_grounded.pt` (99.1M parameters, the head added); the biography
builds at seq 4096 and 600M tokens in `data/prepared_pilot` (974,525 documents per arm, filler
repeat 2.18x over 270M filler tokens, bio share 2.3%; the store `data/index/pilot_bios`, the facts,
pools and `inject_val` are byte identical to the micro build, so the closed-book reads are
comparable); the three merges `pilot_warm_train`, `pilot_main_train`, `pilot_control_train`
(3,488,744 documents, 910.2M prompt tokens each, so the per-arm budget is 910M, not 1B: the QA
slice at 2 passes holds 210.6M; evidence 652M tokens at ratio 0.717, 419.6M of it in the chain
slice; QA and chain subsequences md5 identical across the three) and `evidence_fixed` copied with
its `.src`; the launchers `ckpts/inject/launch_pilot_{p,c}.sh` and the per-save read script
`ckpts/inject/read_pilot_save.sh`. Not done before the launch: the dry run of the read scripts on a pilot save (no
probe save existed; arm P's own saves served and timed the read set). The spec stood as
written on 2026-10-09; the paragraph above is its summary.


**The warm-up and the main split, as they ran (2026-10-10, evening).** Launched at 19:03 on the warm-up
split, it ran at 42k to 48k tokens per second (50M per 21 minutes, 27.1 GB peak), three times the
fitted rate, so the whole pilot is one night. Stopped right after the 300M save and read: gold card
top-1 1.000 in every tier in distribution (0.95 to 1.00 held out), swapped card followed on every
item, and closed book at chance in every cell on both forms (largest |z| 2.3), so both halves of
the switch criterion hold with margin and no counterpart of the micro arm's 50M leak. The 1-hop
gate passed at the same save: `chains_eval` 1-hop 0.32 against chance 0.115 (z +21) with one site
kept, which the micro model never reached; 2-hop, 3-hop and the held-out composition sit at chance
and the paired kept 3 minus kept 1 delta is +0.000, so L1 is readable and reads flat at 300M of
warm-up. The trainer's own reads: gold gain on `evidence_fixed` +3.73 nats at 275M (above the
3.23 ceiling measured on other rows), grounded AUROC 0.95, selector still weak (chunk AUROC 0.61).
Relaunched on `pilot_main_train` from the 301M save at 21:50 (document 1,221,989, the same
document on both splits by the shared order). A full read set takes 26 minutes against 21 minutes
per 50M, and the trainer keeps the GPU, so each read is a stop, read, relaunch; rank reads at
350M and 400M (the micro leak appeared within 50M of the switch), then every 100M with the chains.
The 351M read: copying intact (gold top-1 1.000 / 0.985, swapped followed 1.000 / 0.97) and the
tier 1000 leak written within 50M of the switch as at micro (entity `delta` 0.043, z 3.9 in
distribution; 0.012, z 1.3 held out; tier 100 flat). At 426M the leak grows on both forms (0.050,
z 5.0 / 0.029, z 3.3 at tier 1000; tier 100 still flat) and held-out copying slips (gold top-1
0.92, swapped followed 0.93, from 0.985 / 0.97). The 530M full read (2026-10-11, after a user
pause at 429M): the 1-hop gate holds at its 500M bar and is rising (1-hop 0.52 to 0.53 against
chance 0.115, z +40, from 0.32 at 300M); L1 flat (2-hop, 3-hop, the held-out composition and 4-hop
at chance at every depth, paired kept 3 minus kept 1 -0.004 +- 0.005); the tier 1000 leak level in
distribution (0.040, z 4.5) and caught up held out (0.043, z 5.9), tier 100 flat on both forms;
held-out copying back to 0.97 (gold top-1 0.973, swapped followed 0.97). The 631M full read: 1-hop
0.59 (z +47, the gold answer now at ln K); L1 still flat (every composition cell at chance, paired
delta -0.006 +- 0.005); tier 1000 leak 0.041 (z 4.3) / 0.028 (z 3.5), the lower tiers creeping in
distribution (tier 1 z 2.8, tier 100 z 2.0, the pool over tiers z 4.3) but no lower-tier cell at
3 sigma on either form; held-out copying 0.92 again, so it oscillates between 0.92 and 0.98 across
reads rather than sliding. The 731M full read: 1-hop 0.80 (z +68) with the gold answer at half
the uniform-pick cost, so lookup selects now; L1 still flat (every composition cell at chance,
paired delta +0.002 +- 0.005); the leak grows again through the cosine's tail, tier 1000 0.064
(z 7.0) / 0.045 (z 5.7), and **the A1(b) tier 100 cell crosses 3 sigma in distribution (0.013,
z 3.2; held out 0.008, z 2.4)**, the first breach of the pilot's externalization criterion on one
form at one read; the verdict is the final save's, read against arm C; held-out copying at its
low (gold top-1 0.935 entities, 0.833 nouns). The 832M full read: the tier 100 breach moves to
the other form (in distribution 0.008, z 1.8; held out 0.013, z 3.4), so over the last two reads
one form is at 3 sigma each time and neither on both; tier 1000 0.068 (z 7.2) / 0.046 (z 5.6);
1-hop 0.85; L1 flat (paired delta +0.004 +- 0.005); held-out copying back to 0.96. Record: the
"Run" section of [PILOT.md](PILOT.md). The final save was read next (the next paragraph), then arm C, then the
reads of the ladder, with A1(b) decided on the paired `compare`.


**Arm P was complete and read on 2026-10-11.** Arm P ended at 860.61M tokens,
one epoch (the trainer stops at the epoch's end, so the cosine fitted to 910M stopped at LR about
6e-5), final filler CE 3.2321. Its final read: the 1-hop gate holds (0.85 against chance 0.115,
gold answer 0.82 nats against ln K 2.20); **L1 reads in the spec's "falsified" cell** (held-out
2-hop Delta kept 3 minus kept 1 +0.002 +- 0.005, every composition cell at chance at depths 3 and
4, Delta + 2 sigma 0.012 against the 0.05 bar); **the arm P half of A1(b) is not met on one
form** (tier 100 entity `delta` 0.0125 at z 3.2 held out, 0.010 at z 2.2 in distribution; tier
1000 0.064, z 7.2 / 0.046, z 5.5; the forms have swapped sides across the last three reads and
pooled over forms the tier 100 cell has sat at z 4 to 5 since 731M); copying intact (gold top-1
1.000 in distribution, 0.95 held out, swapped followed 1.000 / 0.94 to 0.97); A2 and A5 clear on
the trainer's numbers (gold gain +4.00 nats, content gain +3.97, chunk AUROC 0.78, grounded AUROC
0.994). Verdicts then waited on arm C's final `none` read (the paired `compare`, arm C above its own
prior first; read 2026-10-14) and the user's decision on the decision table (taken 2026-10-17): the first matching row is "A1(b)
fails on either form" (the recipe fails at shape; no blueprint work until the leak is
understood), with the failing cell at z 3.2 on one form only; the L1 row (falsified at this shape,
looping stays a requirement, the reasoning clause is reworded before the real-run spec) applies on
its own terms. Arm C (`launch_pilot_c.sh`, `ckpts/evidence_pilot_c/`, log
`ckpts/inject/pilot_c.log`) ran the control split from the same seed to its epoch's end; reads
`read_pilot_save.sh CKPT c TAG` (`none` only) at its saves and final, then `closed_book_rank.py
compare` arm C against arm P, then the ladder's final reads on arm P (A3 `eval_abstention.py`,
A4, A7 `eval_store.py`, A6 `eval_benchmarks.py` on both arms, A5 `--ceiling-json`). Arm C at
325M (2026-10-12, after a pause at 74M): the control stores the biographies as the micro arm (a)
did, entity `delta` in distribution 0.299 (z 43) at tier 100 and 0.390 (z 25, top-1 1.000) at
tier 1000, held out 0.029 (z 7.7) / 0.105 (z 10.3), so A1(b) is readable; against arm P at the
same tokens (0.001 / 0.043) the control stores about 300x more at tier 100 and 9x more at tier
1000. Record: the "Run" section of [PILOT.md](PILOT.md).


**Muon for the real run, proposed 2026-10-12 (the user), decided by rung R7 (closed 2026-10-21).** The hidden matrices
step with Muon and everything else stays on AdamW (Decisions, optimizer). It changes nothing in the
pilot: arms P and C share AdamW, as a matched pair must. R7 was to run once arm C freed the GPU and the
pilot verdicts were read: first a build (one optimizer over the fp32 masters in `pretrain.py` and
`sft.py`, which share `train_step`), then two micro arms against their AdamW twins on the R0b
recipe, then one pilot-shape arm against arm P if the micro arms pass. The read that decides is the
leak, not the loss: Muon learns rare associations better than Adam (arXiv 2509.26030), and the
tier 100 cell is a rare association. Decided 2026-10-21 (the user), on R7's micro arms: Muon is
dropped from the real-run spec and the real run stays on AdamW.


**Both pilot arms complete and read; the ladder's remaining reads ran the same day (2026-10-14).**
Arm C ended at 909.72M (one epoch; final `inject_val` CE 3.2195 against arm P's 3.2321; on
2026-10-25 a clean validation slice put that gap down to contamination, the control's exit 3 lead
0.025 on `inject_val` and 0.002 on `inject_val_clean`). Its
final closed-book read: entity `delta` in distribution 0.016 / 0.201 / 0.467 / 0.468 by tier
(tier 100 z 62, top-1 0.96), held out -0.006 / 0.005 / 0.049 (z 12.3) / 0.118 (z 10.8). The
paired `compare` against arm P (`ckpts/inject/compare_pilot_c_vs_p.log`): the control ranks the
real name better by 0.480 (z -81) at tier 100 and 0.372 (z -29) at tier 1000 in distribution,
0.033 (z -6.4, all classes pooled; on entities only raw z -3.8 and prior-controlled `delta` z +7.0,
both with the control storing more) and 0.050 (z -3.7) held out. **A1(b) as written: the control half holds with
margin; the arm P half fails on the held-out form at tier 100 (0.0125, z 3.2), a cell 4x under the
control on that form and 47x under it in distribution.** The decision table's first matching row
is "A1(b) fails on either form"; the user was to decide whether a per-form 3-sigma miss at a 4x to 47x
reduction is that row (decided 2026-10-17, below: A1(b) passes with the leak known). The ladder's remaining reads are done (`ckpts/inject/ladder_pilot.sh`,
outputs `pilot_{p,c}_ladder_*`): **A5 holds** (the port's +4.00 nats is 1.56x the in-context
ceiling of 2.56 measured on arm P's own final save); **A6 holds** (13 tasks, arm P within two
sigma of arm C on all but boolq's raw accuracy and mmlu, both arms at chance on most tasks);
**A3 holds in likelihood** (answer CE 2.13 under gold, 5.87 with nothing attached, 6.11 under
distractors; the swapped answer wins in likelihood, `mr_ll` 0.11) **and is unreadable in
generation** (the model abstained on 96% of answerable SQuAD questions with the gold passage
attached, EM 0.01; that read was void, the packer bug; on the fixed packer 2026-10-24 it abstains on 77%, EM 0.05); **A7's recall half fails** (the model's query at recall@5 0.004
against bge 0.829 over 274k chunks: the selector was trained over a document's few visible chunks
and never against a store); **A7's pathway half unreadable** (oracle-buffer EM 2.6%, model query EM 0.000 with
97% abstention), **the A4 store edit unreadable** (zero correct answers to flip; the A3
generation, A7 pathway and A4 edit reads were taken before the 2026-10-24 packer fix and are void;
re-read 2026-10-24: A3 weak in generation, EM 0.05 and abstention 0.77 under gold, A7 oracle EM
0.09 against the model's query at 0.001, the A4 flip read still unreadable on the model's query
route). The exit reads repeat the micro shape on both arms
(pass 2 +0.07 to +0.08, pass 3 under +0.01, even across deciles, no confidence rule beats fixed
depth) and the depth reads show the control recalling tier 100 in pass 2 (0.348 / 0.464 / 0.467)
with arm P flat. In one line each: copy holds; A1(b) control half holds, arm P half fails on
one form by z 3.2 at a 4x to 47x reduction; the 1-hop gate holds; L1 "falsified"; A2, A5, A6
hold; A3 split; A7's recall half fails; A7's pathway half and A4 unreadable. A research round on 2026-10-16
([review_2026-10-16.md](../review_2026-10-16.md)) found the nearest published relatives (LMLM,
Co-LMLM) measuring the same kind of residual against a standard model, the pilot's chain read
matching the slow phase before an abrupt in-context two-hop transition, from-scratch two-hop
composing only under looping and only after grokking-scale training, and the A7 recall fix in Co-LMLM's
in-batch InfoNCE on the model's own query; its seven proposed directions (a chain-overtraining
arm with Poisson loop counts read every 10 to 20M, zero-hop identity questions, InfoNCE store
negatives, a mask-rate leak lever, the deletion audit as the A4 read, a loop stability term, the
anonymization lever read on grounded tasks) are proposals. **Decided 2026-10-17 (the user):**
A1(b) passes with the leak recorded as known and its criterion amended to the paired reading
(the read table in PILOT.md); L1 is recorded as failed at this shape unless the chain
continuation shows composition, in which case it was under-dosed; the retriever fix is built; A3's
behaviour is left to 6f's RL and that reward is marked the one RL step that must not fail; the
queue is the chain arm, the InfoNCE micro arm, then R7, R8, R9. **The chain continuation of arm P
ran and was read on 2026-10-17 ([chain_continuation_pilot.md](../measurements/chain_continuation_pilot.md)):
L1 is failed at this shape.** Arm P's final save trained one epoch on `chains_train` alone
(`ckpts/evidence_pilot_chains/`, 102.66M tokens in 5.1 hours at batch 4 x 2 after batch 8
spilled, saves every 10M). The held-out answer CE decays smoothly 0.315 to 0.201 with top-1 per
token flat at 0.91 and the loop scales healthy ([1.23, 0.49, 0.27] at the end). The final
save's chain read: 1-hop 0.99 (from 0.85), every composition cell at chance at depths 1 to 4
(2-hop 0.19, 3-hop 0.18, 4-hop 0.21, held-out 2-hop 0.20; paired kept 3 minus kept 1 +0.004
+- 0.008), the composed answer at a uniform pick among the type-matched candidates (1.93 nats
against ln K 1.72). The published two-hop transition did not come in 2.1 epochs of the chain
split at 99M parameters; the decision table's "falsified" row applies (looping stays a
requirement; the goal's reasoning clause is reworded before the real-run spec; R9's written
bridge is the next read on the question). **Built 2026-10-17: the retriever's contrastive term**
(`evidence_contrastive_loss`, in-batch InfoNCE on the selector's query at the answer-start
positions against every chunk in the batch, both directions, temperature 0.07;
`evidence_contrastive_weight`, 0 by default; the CLAUDE.md port bullet and
`tests/test_evidence_contrastive.py`). **The InfoNCE micro arm, one epoch, read 2026-10-17
([infonce_micro.md](../measurements/infonce_micro.md)): the term moves store recall from zero
to small.** `ckpts/inject/launch_infonce.sh`, `config_micro_infonce.yaml` (weight 0.25)
against `config_micro_infonce0.yaml` (weight 0), one pass each over `evidence_nomany_train`
from `seed_micro.pt` with the key/value reader (34.82M tokens, 15.5 minutes each; the long web
rows are skipped at seq 1024). Held out: the arm's `contrastive` 5.49 to 2.18, `selection`
equal on both arms (0.712 / 0.714), answer CE within 0.025 nats, the port unread on both at this
dose. `eval_store.py recall` on `data/index/openqa` (274k chunks), loop 1: the arm 0.002 / 0.012 /
0.035 at k 1 / 5 / 20 against the twin's 0.000 / 0.000 / 0.001 (arm P's final read 0.004 at k
5; bge 0.829). The in-batch separation over about 250 chunks transfers weakly to the store at
35M tokens on a 35M model. **The same pair at three epochs, read 2026-10-18** (`w25e3`, `w0e3`,
`config_micro_infonce{,0}_e3.yaml`, 104.46M tokens in 40 minutes each; the twin relaunched
clean from the seed after a stop at 85M): recall 0.016 / 0.059 / 0.133 at k 1 / 5 / 20 on the
arm against 0.000 / 0.000 / 0.002 on the twin (bge 0.662 / 0.829 / 0.897), HotpotQA 0.237 at
k 20 against NQ 0.019; the arm's held-out answer CE 4.45 against the twin's 4.38, a 0.07 nat
cost that widens through the run (0.05 at 20M, 0.08 at 100M); `selection` 0.695 against 0.706;
gold costs more than none on both (the port unread at this dose on either arm); the training
rows' external mass under gold 0.123 against 0.076. The term is the whole of the model's store
recall at this dose, one sixth of bge and growing; whether the retrieved mass becomes answer
gain is unread at micro scale. **Decided 2026-10-21 (the user): the term enters the pilot-shape
recipe at 0.25, and its language cost is read on one pilot-size arm before the real-run spec
commits to it.** **That arm, PI (`config_pilot_infonce.yaml`: arm P's recipe with the term at
0.25 and loss-free balancing, `ckpts/evidence_pilot_pi`, `pilot_pi.log`), was stopped at 87M the
same day on the language bar** ([infonce_micro.md](../measurements/infonce_micro.md)): held-out
CE on `inject_val` 0.03 / 0.11 / 0.30 nats behind arm P at steps 890 / 1770 / 2660 against a
bar of the pilot's own noise (0.013), the load fields even, while store recall@5 on the 75M save
read 0.006 against arm P's final 0.004 and the port read lagged arm P's (content gain +0.43
against +1.76 at step 2660). The term as built, with its gradient into the trunk, stays out of
the real-run spec; next are an offline probe of the hidden state and one head-only micro arm
(trunk gradient stopped, loop 1 only; both read the same day, below), with a separate search tower
the other direction (held 2026-10-21).
A research round on
training the model's own retrieval without the language cost started the same day; its
write-up is [review_2026-10-21_retriever.md](../review_2026-10-21_retriever.md): the term trains loops 2 and 3, which already hold the gold chunk, while the eval reads loop 1; the cheap arms are the term at loop 1 only, a stopped trunk gradient, bge query distillation and a fixed key bank of negatives, after an offline probe of whether the loop 1 state carries the question at all. The model-keyed variant (keys from the model's own encoder, Co-LMLM's) stays
held: A7's recall half is the model's query against a bge-keyed store, which is exactly the pairing the term
trains, and model keys would change the store format and the gate's meaning.
**The query head probe, read 2026-10-21 ([query_head_probe.md](../measurements/query_head_probe.md),
`scripts/probe_query_head.py`, eval only), caps a head on the frozen state:** ridge and MLP maps
from the IR expert's loop 1 input onto bge's question vector reach at best recall@5 / @20 0.025 /
0.055 on `w0e3` and 0.055 / 0.122 on `w25e3` (mean and last token concatenated, MLP), under 7% of
bge at k5 at any position or pooling; the term's gain split between the head and the trunk state
(a head on the untouched trunk reaches about 0.025 at k5, on the term-trained trunk about 0.055,
the model's own query 0.059), so the head-only arm reads what the head alone gets from an
untouched trunk, and the trunk's share cost 0.30 nats at pilot shape; a tower on detached states is capped the same way, so a working tower would be a small
bge-style encoder over the prompt text beside the model (held, decided 2026-10-21, below).
**The head-only micro arm, read the same day ([infonce_micro.md](../measurements/infonce_micro.md), `bash
ckpts/inject/launch_infonce_head.sh`, `config_micro_infonce_head_e3.yaml`: the `w25e3` recipe
with `evidence_contrastive_loops: [0]` and `evidence_contrastive_trunk_grad: 0.0`, 104.46M
tokens, log `infonce_head_e3.log`, read `infonce_head_e3_store_recall.{json,log}`), costs no
language and finds nothing:** held-out CE 4.3904 against `w0e3`'s 4.3809 (within 0.02 at every
eval, `w25e3` 0.05 to 0.08 behind) and store recall@5 / @20 0.003 / 0.007 at loop 1 against
`w25e3`'s 0.059 / 0.133 (bge 0.829 / 0.897), the probe's weakest row. The whole of `w25e3`'s
search therefore needed the trunk to change, so training only the search head is closed at this
size. **Decided 2026-10-21 (the user): the separate search tower (a small bge-style encoder over
the prompt text beside the model) is held, not built; bge keys the store and embeds the question
at serving, A7's recall half stays untested, and the model's own retrieval is revisited at the real run's size
(`evidence_contrastive_loops` and `evidence_contrastive_trunk_grad` stay as the knob); the
real-run spec carries no retriever term.** **R1b, the depth memory probe, run 2026-10-21
([depth_memory_probe.md](../measurements/depth_memory_probe.md)): depth 5 is trainable at pilot
shape with full gradient at batch 4 x 2 (17.3 GB, about 31 GB at batch 8 over the card, 1.27x
the time per token of depth 3), and the detached prefix (`detach_loops`, built the same day)
fits at 21.5 GB but cuts the prelude off from the language loss until a straight-through route
exists; nothing changes in the real-run spec.** **Built
2026-10-18 while that pair ran: the optimizer rung's code** (R7 in the ladder: `MuonAdamW` in
`modules/model/optim.py`, the `optimizer` key, `eval_exit.py --pair`, the test); its three micro
arms (`ckpts/inject/launch_muon.sh`) ran after the InfoNCE pair (the partition at micro: 60
tensors, 25.7M on AdamW, 78 tensors, 9.8M on Muon, the embedding table being most of the
model). **Read 2026-10-18 ([muon_micro.md](../measurements/muon_micro.md)).** Loss: Muon's
held-out CE below its AdamW twin's at every read on both recipes, 0.07 to 0.15 nats, z 16 to
25 paired per token, against a one-pair seed spread of 0.03 at the end (0.04 to 0.06 mid-run,
itself z 6 paired); passes. Copying: both warm-up arms switch between 50M and 100M, gold top-1
at 100M 0.98 to 0.99 against 0.95 to 0.97, held-out copying at the final 0.90 against 0.60 to
0.74, swapped held-out card followed 0.90 against 0.67; passes. The leak (the warm-up arm's
closed-book entity `delta` at the final, Muon against AdamW): in distribution 0.023 / 0.035 /
0.136 against 0.000 / 0.015 / 0.085 at tiers 10 / 100 / 1000, held out 0.004 / 0.012 / 0.037
against -0.001 / 0.007 / 0.047; paired on the rank the largest cell is z 2.6 (tier 1000 in
distribution), so the bar as written holds, while in distribution Muon leaks more at every
tier above 1 (each cell z 2.4 to 2.6 on `delta`) and the storage read on arm (a) says why:
Muon stores the biographies about twice as early (its 50M read sits where AdamW's 100M does;
tier 100 in distribution 0.072 against 0.010 at 50M, 0.451 against 0.378 at the final, z 22
paired) and more at every tier above 1, far outside the seed spread at tiers 10 and 100 in
distribution (z 0.6 and 0.7 between two AdamW seeds), while the seed spread at tier 1000 held
out is z 3.6, as large as the leak difference. Nothing is stored before copying on either arm
(50M and 100M within 2.1 sigma), so the extra leak is written after the switch by real-value
supervision, as the AdamW leak is. The attention-logit bound is unread. **Decided 2026-10-21
(the user): Muon is out of the real-run spec, the pilot-shape arm is not run, the real run
stays on AdamW.** Every save of the chain
continuation is read (2026-10-19, the per-save table in
[chain_continuation_pilot.md](../measurements/chain_continuation_pilot.md)): 1-hop climbs
0.82 to 0.99 smoothly, the four composition columns sit at chance at all eleven reads, the
paired held-out Delta within one sigma throughout; no transition anywhere inside the epoch.
**R8 built, run and read 2026-10-19 and 2026-10-20 ([lossfree_micro.md](../measurements/lossfree_micro.md)):
loss-free balancing passes on all three bars.** Held-out CE 3.9055 against arm (a)'s 3.9037
(paired at exit 3 -0.0055, z -1.8, one fifth of the 0.029 seed spread), storage inside the two
AdamW seeds cell for cell, MLP load max over mean 1.03 to 1.05 per pass. The load comparison is
unread: the logged aux value saturates in bf16 (hard counts stop at 256) and reads the batch,
so arm (a)'s aux term was a floor against collapse, not a balancer. Fixed 2026-10-20 (fp32
counts in `compute_aux_loss`, `test_evidence_selector.py`); the comparison half of the load bar
was read on an arm (a) rerun under the fixed term on 2026-10-20 (`bash ckpts/inject/launch_auxfix.sh`, `ckpts/evidence_auxfix_full`, reads `ckpts/inject/auxfix_full_{TAG}_{rank_none,exit}` paired against arm (a)). **Loss-free is better on all three bars:** the working aux term ends at CE 3.9142 (paired -0.0076 at exit 3, z -2.3), puts 0.35 to 0.54 of the picks on the dense slots with the MLP load at 1.08 to 1.26, and stores the biographies more in distribution (tier 100 z 14.5 against arm (a), z 14.4 against the reseed, far outside the seed spread). **Decided 2026-10-21 (the user): loss-free balancing enters the real-run spec.** Decided the same day (the user): the 100B-token corpus build (6c) runs on the rented cluster, not locally, and the throughput work (6a) starts only after every other spec decision is made, so nothing optimized has to be redone.


**Run order, decided 2026-10-23 (the user, after a cost read):** R4b's expert arm first (it needs
no build) with builds beside it, R9 next (hours; its read gates R4's corner arm), R6 last (its data
build is the longest CPU job). One training run at a time on the card (a second GPU process spills
the first into shared memory); builds, data preparation, docs and save reads run beside it, but
corpus builds and evals are not resumable, and a run paused through its STOP file loses the tokens
since its last save and resumes on the code in the tree, so builds land behind flags that default
to the old behaviour. **R4 is staged:** arm P is the per-pass-loss, no-coda cell (at chance on
every composition read, run under the aux term, accepted as the baseline); the far corner
(last-pass loss through the coda; the coda is a half-day build behind a flag plus review) runs
first as one arm, a chance corner makes the table a null, and only a positive corner earns the two
single-switch arms and second seeds of the corner and the baseline (five arms in all); the corner
runs only if R9 says the loop, not written tokens, must compose (optional if the bridge composes;
if neither does, R4 waits for the real run's size). **R4b's sublayer half is not a config change:**
today's core is one block per pass, and 2 against 3 sublayers is the unbuilt blueprint core (reader
in sublayers 1 and 3, per-sublayer norm and scale), a day of architecture work plus review before
two arms; the question for the user was to build it, or read one arm of the three-sublayer core against today's
one block (decided 2026-10-25: one block per pass, not built, RUN2.md row 5). **Arm E4 was launched 2026-10-23 17:09** (read 2026-10-24, below; `bash ckpts/inject/launch_pilot_e4.sh`,
`ckpts/inject/config_pilot_e4.yaml`: `config_pilot.yaml` with `num_mlp_experts: 4` and
`moe_balancing: loss_free`; seed `ckpts/inject/seed_pilot_e4_grounded.pt` from
`init_scratch_seed.py` seed 0 then `migrate_groundedness_head.py`, 89.6M total, 84.9M active (the probe's count; the trainer printed 88.9M / 84.2M, and arm P's 98.4M / 84.2M against the 99.1M / 84.9M used everywhere), arm
P's active compute; run directory `ckpts/evidence_pilot_e4`, log `ckpts/inject/pilot_e4.log`; arm
P's recipe: the warm-up on `pilot_warm_train` stopped after the 300M save by `stop_after_save.sh e4
'30[0-4]M'`, then `SPLIT=pilot_main_train bash ckpts/inject/launch_pilot_e4.sh`; reads `bash
ckpts/inject/read_pilot_save.sh CKPT e4 TAG`, which loads the e4 yaml): 44k to 48k tokens per
second, peak 26.3 GB at the first look (26.4 GB for most of the run, 26.9 GB the maximum), step 0 evals neutral. Read against arm P on held-out CE on `inject_val`,
closed-book rank by tier on both forms, the 1-hop gate and L1, the port read and the load fields;
caveat on record: E4 runs loss-free where arm P ran under the aux term the bf16 saturation had made
a constant, two changes, and at micro loss-free alone moved CE and storage inside the seed spread,
so a difference is read as the expert count with that caveat. **R9 built 2026-10-23** (code by a
subagent, review pending at the time; run 2026-10-24, below): `prepare_chain_data.py --bridge` renders the assistant turn
as "Step 1: e1. Step 2: e2. Answer: ek." (1 hop: "Answer: e1.") with `.bin`, `.idx` and `.mask` differing from the plain build (the answer text changes), `.chains.jsonl` gaining `bridge`, and `.ev`, `.evchunk`, `.evidx`, `.evkeyidx`, `.evgold`, `.cond`, `.ans` and `.evhop` byte identical (`.evkey` too given a deterministic embedder), and `bridge` in the meta; `eval_chains.py --generate-bridge
[--bridge-max-tokens 48]` greedily decodes the bridge with evidence attached (left padded, no KV
cache, stop at "Answer:") and scores the candidates after the generated prefix through the
unchanged scoring path, with per-record `generated`, `bridge_marker`, `bridge_hits`, `bridge_all`
and one extra table (marker rate, hit rate per intermediate hop, accuracy with and without a fully
written bridge); helpers in `modules/data/chains.py`, tests in `test_chain_generator.py` (5c, 8b)
and `test_eval_chains_bridge.py`. Then built and run as planned: `bash ckpts/inject/build_chains_bridge.sh` (500k train
and 2000 val questions, seed 42, GPU embedder, in E4's 300M stop window), then `bash
ckpts/inject/launch_pilot_bridge.sh` (`config_pilot_bridge.yaml`: the chain continuation config on
the bridge splits, two epochs, saves every 10M, from arm P's final; `ckpts/evidence_pilot_bridge`,
`pilot_bridge.log`), planned at about 50M tokens (it ran 66.35M), read save by save in the generate mode on `chains_eval`,
`chains_heldout_tmpl` and `chains_hop4` against arm P's latent read; bar: held-out 2-hop 3 sigma
above chance with the 1-hop gate kept.


**Arm E4 complete and read 2026-10-24 ([expert_count_pilot.md](../measurements/expert_count_pilot.md)):
halving the routed experts from 8 to 4 at top 2 (10M fewer parameters, the same active compute)
costs nothing measurable at this shape and dose.** 858.40M tokens, one epoch (killed once at about
680M on the user's request on 2026-10-23 and resumed from the 652M save, which replays the same
documents; the 675M eval reproduced within 0.001). Held-out CE 3.2396 against arm P's 3.2321, E4
above at every read by 0.001 to 0.019, under the pilot's noise of 0.013 at every read but 25M and
276M (E4's evals after 450M sit 2.7 to 4.1M tokens before arm P's nearest, about 0.002 to 0.003 nats of the gap at arm P's slope there, so the raw gaps of 0.005 to 0.012 read about 0.003 to 0.010; still above at every read, under the noise of 0.013.); the switch criterion met at 300M as on arm P; copying equal (gold top-1 1.000 in
distribution, 0.96 to 0.97 held out; swapped followed 1.000 / 0.91 to 0.93); the 1-hop gate 0.875
against 0.848, every composition cell at chance, so L1 fails as on arm P; port gold gain +4.19,
content +3.82, grounded AUROC 0.972 against +4.00, +3.97, 0.994; the leak equal at tier 100 (entity
`delta` 0.017 / 0.012 by form against 0.010 / 0.013, paired z -1.2) and smaller at tier 1000
(0.053 / 0.041 against 0.064 / 0.046; paired raw rank z -3.8, about one sigma on `delta`, since
E4's prior ranks worse too); MLP load max over mean 1.01 to 1.06 per pass. Caveat on record: two
changes (loss-free against the saturated aux term) and one seed each. For shape L the expert count
is a cost choice, not a quality one, until a read at the real run's size says otherwise; R4b's
sublayer half stays open. **R9's bridge decode smoke ran 2026-10-23** on arm P's final (8
questions per hop, `ckpts/inject/bridge_smoke.json`, moved on 2026-10-24 to
`ckpts/inject/bridge_prefix_bug/` since it was decoded before the packer fix `af24c70`): the decode runs end to end and writes
entity-like text, marker rate 0 as expected on a model that never saw the form. **R9's SFT
ran 2026-10-24** (`bash ckpts/inject/launch_pilot_bridge.sh`, `ckpts/evidence_pilot_bridge`,
log `ckpts/inject/pilot_bridge.log`; the bridge split 500k questions, 33.7M prompt tokens per pass
at evidence ratio 3.26, two passes; its record is below).


**R9, the written bridge, ran 2026-10-24; its 30M read meets the bar.** From arm P's final, SFT on
the 500k-question bridge split (33.7M prompt tokens per pass at evidence ratio 3.26, two passes,
saves every 10M; `ckpts/evidence_pilot_bridge`, `ckpts/inject/pilot_bridge.log`, target about 72M
tokens), trained to its end at 17:18 at 66.35M tokens in 184.5 minutes (exit 0). Reads of the 10M
to 40M saves by `ckpts/inject/read_bridge_save.sh`; the generate-mode reads taken before the packer
fix `af24c70` (the decode smoke and early save reads) sit in `ckpts/inject/bridge_prefix_bug/`.
The 30M read (`ckpts/inject/pilot_bridge_30M_gen_d3.log`, `_hop4.log`), `chains_eval` at depth 3:
one read site kept 0.998 / 0.992 / 0.980 at 1 / 2 / 3 hops (chance 0.115 / 0.185 / 0.206), three
sites 0.999 / 0.994 / 0.988; `chains_heldout_tmpl` 2-hop 0.989 at one site and 0.993 at three
(chance 0.191), paired kept 3 minus kept 1 +0.004 +- 0.002; no read site at chance (0.132 / 0.199 /
0.220, held out 0.218, z at most +2.1). The bridge marker is written on every question with a site
kept, the intermediate hop hit rate 0.98 to 0.99. The 40M read is the same shape (three sites
1.000 / 0.997 / 0.996, no site 0.137 / 0.192 / 0.192). The 4-hop split at depth 4 reads 0.176
against chance 0.210 (z -2.6): the model writes exactly two steps and then "Answer:" on 999 of 1000
questions (it trained on hops 1 to 3, so at most two intermediate steps; hit rate 0.99 / 0.97 /
0.00 by hop), not a decode truncation (the bridges run 52 to 88 characters against the 48-token
cap). The log's `instrument valid: no` and its four FAIL lines are the latent read's sub-hop rule
applied in a mode it does not fit: with the bridge written every decode step is a fresh forward,
so one read site composes by writing (the review note; their suppression in that
mode is held). R9's bar (held-out 2-hop 3 sigma above chance with the 1-hop gate kept) is met at 30M
on this read; the verdict was left to the user after the final read (read 2026-10-24, below).

**The repository review, 2026-10-24 ([review_2026-10-24.md](../review_2026-10-24.md)).** Six
detached reviewers read the tree against the goal after `fee6a5c` and `af24c70` were committed.
Verdicts: externalization is demonstrated at pilot shape relative to the matched control, with the
tier 1000 residual known; at this shape reasoning composes in written tokens, not in the latent
loop; the retriever and the abstention behaviour are the open halves, and both rest on reads the
review found flawed. Findings that change a recorded reading: every generation read with evidence
before `af24c70` read a shifted buffer (the A3 generation read, the A7 pathway EM and the A4 edit
flip rate, the G3b `last` mass past loop 0, the pre-fix bridge smoke); arm C trained on 12,240
documents and 50.11M tokens of 4094-token QA rows the evidence arms drop, and its cosine reached the
6.0e-5 floor where arms P, E4 and PI stopped at 6.55e-5; about 7.4% of `inject_val` tokens appear
verbatim as supervised text both arms train on and another 6.7% in the rows only arm C trains on;
the logged and `[eval]` CE on `inject_val` accumulate in bf16 (about 0.002 of the 0.013 noise);
the closed-book sigma was estimated to be understated by person clustering (up to 1.7x, 2.2x in
`compare`; run 2026-10-24 on the pilot rank JSONs: sigma within 10%, the decided reading stands)
and `compare` pairs the raw rank, not `delta`; a 2-hop shortcut scores 0.256 where chance is 0.185;
the selection loss is carried by long biography documents (63% of gold-document positions); the
contrastive term at loops 2 and 3 can match on evidence already read; under the key/value reader
the groundedness head reads the whole attention output. Landed 2026-10-24 (default behaviour
byte identical): the merge tool reports documents over the evidence reader's limit; the eval
packer assert in `evidence_cu_seqlens`; the splits, fp32 masters and RNG states stored in the
checkpoint with `--switch-split` required to change split on a relaunch; the save and eval grid
fixed across relaunches; a finished run exits 0 instead of resuming from its final (and, the same
evening, the epoch permutation selected through the prepared loader, `sft.select_epoch`: a resume
past epoch 0 had read epoch 0's order again; found on the `w0e3b` twin's resume, no recorded run
affected); the `n_valid`
syncs removed; the clustered sigma and a paired `delta` compare as extra output lines (run on the arm P and E4
rank JSONs the same day: sigma moves by 0.9x to 1.1x; arm P held-out tier 100 z 3.2 item and 3.2
clustered, in distribution 2.2 and 2.3, tier 1000 7.2 and 8.0 in distribution, 5.5 and 5.9 held
out); the shortcut baseline
printed by the chain builder; the training-distractor gold exclusion in the builder (next build
only); `migrate_groundedness_head.py` accepting pre-2026-10-19 sources. Held (the Open items row):
the changes that move a number or need the GPU for their bit-identity test, landed the same
evening but for the eval-pass rewrite (below). The bridge eval's two items (verdict lines
suppressed, evidence built once per batch) were first listed here as landed; they were not, and
stay held. Its plan proposals
are in Decisions ("repository review"), decided 2026-10-24.

**The eval packer bug, found and fixed 2026-10-24 ([written_bridge_pilot.md](../measurements/written_bridge_pilot.md),
its packer section).** `eval_abstention.py: _pack_evidence_batch` laid each row's evidence out as
[the row's own tokens, then the dump width], while `evidence_cu_seqlens` counts tokens per query
segment and requires the flattened evidence axis sorted by segment id. Right padded batches (real
segment first) are sorted; in a left padded batch the pad run is the row's first segment, so the
count-based `cu_seqlens` gave the first (S_ev minus n) tokens of each row, its own evidence, to the
dump segment and the last n to the real one: every left padded decode with evidence attached read
its buffer shifted by the dump width (correct only for the widest row in the batch). Affected:
every generation with the port attached (`eval_abstention.py --evidence-port` generation, the A3
generation read of 2026-10-14; `eval_store.py pathway`, A7's pathway half, and `edit`, A4;
`eval_chains.py --generate-bridge`). Not affected: every likelihood read (closed-book rank, chain
candidate scoring, the trainers' evals and the port read, the teacher forced calibration, all
right padded), the dataset's packing and `inference.py`. Found on R9's 30M save: teacher forced
argmax over the gold bridge of 32 two-hop val rows 100% right padded against 69.6% left padded
(the intermediate entity's first token 100% against 47%) with the incremental decode matching the
left padded forward at 99.85%; 100% / 100% / 100% after the fix. The fix places a row's tokens
after the dump width when the dump id is below the real one, with a non-decreasing assert on CPU
tensors (`tests/test_evidence_pack_layout.py`, GPU free, fails on the old layout; a detached
Sonnet review found no bugs, two nits and one test gap folded). The reads on the broken path (the
bridge decode smoke of 2026-10-23, the 10M and 20M bridge reads and the 30M val read of the
morning: held-out 2-hop 0.61 at 10M, the first hop hit 41 to 57% of the time) are void and kept in
`ckpts/inject/bridge_prefix_bug/`. Re-read on arm P's final the same day (`bash
ckpts/inject/ladder_pilot_redo.sh`, `ckpts/inject/pilot_p_ladder2_{abstention,store_pathway}.*`):
**A3 is readable in generation now and weak** (under gold false abstention 0.770 against 0.956 in
the void read, EM 0.046, F1 0.063; the likelihood numbers identical, answer CE 2.1257 / 6.1133 /
5.8720 under gold / distractors / none, so only the generation path changed; counterfactual: 28
items answered right with the unswapped chunk, follow 0.429, `mr_ll` 0.026, every stratum still
too small); **A7's pathway gate still fails** (EM 0.001 with the model's query against 0.088 with
the oracle buffer, z -23.7, bge 0.044; the void read had the oracle at 0.026) because the model's
query finds nothing; **the oracle route is readable now (EM 0.09), the A4 flip read still
unreadable on the model's query route** (`ckpts/inject/pilot_p_ladder2_store_edit.*`: 1 of 2,000
answered before the edit through the model's own query, edit n 1 flip 0.000, delete n 0; the
edited gold chunk attached directly was followed, n 1); a flip read routed through bge would need
`eval_store.py edit` extended.

**R9, the written bridge, read 2026-10-24: passes by a wide margin
([written_bridge_pilot.md](../measurements/written_bridge_pilot.md)).** From arm P's final, two
passes of the 500k-question bridge split (33.67M prompt and 109.8M evidence tokens per pass, ratio
3.26), 66.35M tokens in 184.5 minutes, 5.7k to 9.1k tokens per second, 16 GB; held-out bridge CE
on `chains_bridge_val` 4.895 at step 0, 0.0101 at 10M, 0.0018 at 30M, 0.0004 at the final (top-1
per token 0.9998). Read on the fixed packer (`read_bridge_save.sh`, 1,000 questions per split and
hop count, generate then score): held-out 2-hop at three read sites 0.982 / 0.993 / 0.993 / 0.992
/ 0.997 at 10M / 20M / 30M / 40M / final (chance 0.191; z 64.7 at the final). Final `chains_eval`
at depth 3: 1-hop 0.998, 2-hop 0.998, 3-hop 0.997 (chance 0.115 / 0.185 / 0.206); no read site at
chance (0.136 / 0.198 / 0.220); marker rate 1.000 and intermediate hit rates 1.00 at every hop
count; paired held-out Delta (kept 3 minus kept 1) +0.002 +- 0.001, so read sites past the first
add nothing once the hop is written. 4-hop at depth 4 (`chains_hop4`, never trained at that
length): 0.206 against 0.210, hits 1.00 / 0.99 / 0.00, the model writes the two steps the mix
trained and answers, a length limit of the training mix, not of composition. The eval's
`instrument valid: no` is the plain mode's sub-hop check, stale in this mode. Splits: 464 of 2,000
val and 610 of 3,000 `chains_eval` questions repeat a train question verbatim (name pool
collisions, the plain splits alike), `chains_heldout_tmpl` none, so the held-out cell is the clean
one. **Verdict: the model composes two and three facts as soon as the intermediate entity is a
token it writes and reads back, on compositions never seen in training, after 20M tokens of
fine-tuning on the form (the bar was already passed at 10M, 0.982); the loop never did this in silence at this size.** What it decides: 6f's
reasoning form is written tokens (the loop keeps lookup and copying); R4's corner arm is no longer
needed for the real run and is recommended dropped (dropped 2026-10-24, the user); the real run's mix needs
multi-hop rows at the depth wanted at serving, since the written depth does not generalize past the
trained one. Unmeasured: transfer to real questions (A3, HotpotQA bridge questions) and to
retrieved rather than handed chunks; one seed, synthetic chains with the gold chunks in the buffer.

**The review's code changes, landed 2026-10-24 (evening; decided by the user on the review's
recommendations).** Uncommitted at the time of writing, tests passing on the GPU, a detached Sonnet
review with no high or medium finding. Landed, each with its test: the unweighted CE, top-1 and
`p_max` sums of `mtp.py` accumulate in fp32, so the logged CE no longer sits on a 1/64-nat grid and
`top1_acc` cannot exceed 1 (`tests/test_chunked_ce_fp32.py`); under loss-free balancing the
selection scores are formed in fp32 for the topk (a zero bias is bit-identical; at bias 0.30 the
bf16 sum picked differently on 2% of tokens); the frozen chunk gate (every config sets
`freeze_evidence_gate: true`, scale exactly 0) is skipped, with no selector mass, scatter or
multiply by ones per loop, the check cached against the tensor's `_version` so there is no
per-step sync, forward and key/value reader gradients bit-identical
(`tests/test_evidence_gate_skip.py`); `ParallelSparseMoELayer.forward` delegates to
`experts.grouped_mlp_forward` with `null_mask=~mlp_mask`, so the dense-slot picks sort after the
real experts, are counted in the same `bincount().tolist()` sync and are sliced off before the
grouped GEMM (exact zeros as before; outputs differ from the old path only by bf16 summation order;
about 5% of the MoE layer's time at micro shape, more at larger expert widths;
`tests/test_efficiency_2026_10_24.py`); evidence padding is cut into 128-token blocks
(`EVIDENCE_PAD_SEGMENT`) before the encoder's `cu_seqlens`, so it is no longer one long causal
segment (real-token states and logits bit-identical; at pilot shape 9.61 to 7.57 ms on a
4608/600/600/600 batch, no change when every row is full); `groundedness_term` detaches the head
input under the key/value reader (the head still learns, no trunk parameter gets a gradient from
it; the evidence-only signal the review proposed for finding 9 is not built); the cross reader's
rotary gather is skipped under the key/value reader; the eval path norms only the last loop when
`return_hidden` is False. `evidence_selection_loss` is a log-space BCE with no clamp on the
renormalized share (a gold share under 1e-6 now gets a gradient; the old clamp zeroed it), rows
with fewer than two visible chunks are excluded, and a new `TrainingConfig.evidence_selection_weighting`
(`token`, default, the mean over supervised positions as before; `segment`, each query segment
averaged first from `cu_seqlens`) is plumbed through `moe.evidence_selection_term(segment_ids=,
num_segments=)` and the `pretrain.py` call site. `evidence_contrastive_loops` falls back to `[0]`
(first pass only) in `config.py` and the three yamls, null still accepted; the recorded arms'
gitignored configs under `ckpts/inject/` are pinned to null. `sft.py estimate_packed_rows` replays
each reader's rule (evidence reader `len + mtp <= max_length - 1`, plain reader `<= max_length`;
`tests/test_packed_rows_estimate.py`): `pilot_main_train` estimates 241,398 rows (about 30.17k
steps) against the old 258,852 (32,356), matching arm P's observed epoch end at about 30.3k; the
control's 252,912 is unchanged. `prepare_evidence_data.py` caps a training row at 4093 tokens
(`MAX_DOC_TOKENS_TRAIN`), the held-out cap 4094 unchanged; `evidence_dataset.py` logs the documents
and tokens over the row limit at startup and per worker. The reader itself is unchanged: decided
2026-10-24, future evidence arms stay matched to the arms already run on the existing builds, and
the existing pilot merges keep dropping the 12,240 rows. `prepare_injection_data.py` gains two
builder flags, default off and byte identical at the default, each changing only `.mask`:
`--mask-first-name-mention` (the first run of name tokens per document gets mask 0; 1,474,885
tokens of 300M on the micro builds) and `--support-gold-only` (gold-absent documents count no
value as supported). Built the same day: `inject_retrieval_s100a50m1_train`, `s100a50g1`,
`s15a50m1`, `s15a50g1` (44 of 44 companion files byte identical to the flagless builds;
`ckpts/inject/build_leak_arms.sh`, log `build_leak_arms.log`). `scripts/prepare_clean_val.py`
(new, `tests/test_prepare_clean_val.py`) built `data/prepared_pilot/inject_val_clean`: 974
windows, 650,125 tokens, from `inject_val`'s windows whose documents the evidence QA builder never
used (the first 80,672 `ir.bin` documents excluded; phase1 and phase2 kept); its verbatim 64-token
overlap with the QA training rows' supervised tokens is 0.99% against 13.3% for `inject_val` under
the same check (the review's 7.4% was a narrower definition). The read, `ckpts/inject/read_clean_val.sh`
(`eval_exit.py` on the P, C and E4 finals on both slices), was not yet run (run 2026-10-25,
below). Launchers:
`ckpts/inject/launch_review_arms.sh [w0e3b|l0e3|seg_e3]` (`config_micro_infonce0_e3.yaml`, the
fresh `w0e3` twin on the reviewed code; `config_micro_infonce_l0_e3.yaml`, the `w25e3` recipe with
`evidence_contrastive_loops: [0]` and the trunk gradient on; `config_micro_segsel_e3.yaml`, the
`w0e3` recipe with `evidence_selection_weighting: segment`; run directories
`ckpts/evidence_infonce_{ARM}`, logs `infonce_{ARM}.log`, store recall reads
`infonce_{ARM}_store_recall.{json,log}`), launched 2026-10-24 about 20:25 and running at the time (all three read by 2026-10-25), the three
arms in sequence, about 40 minutes each plus the recall read; `ckpts/inject/launch_leak_arm.sh
m1|g1` (the copy-first warm-up recipe on the flagged splits, run directories
`ckpts/evidence_leak_{m1,g1}`, reads `rank_leak_{ARM}_{100M,final}_*` and
`compare_cf_vs_leak_{ARM}.log`), queued after the retriever arms (run and read 2026-10-25,
below). Held: the eval-pass rewrite
(deferred to 6a) and the two `eval_chains.py` bridge-mode items (the latent verdict lines
suppressed and the paired line labelled bridge-internal in `--generate-bridge`; the evidence built
once per batch). The review paragraph above first recorded those two as landed;
`scripts/eval_chains.py` is unchanged since `fee6a5c`. Run order (the user): the three micro
retriever arms, then the two leak arms, then the clean-slice CE re-read, with R5's eval set between
them when the card is idle.

**2026-10-25: the third micro retriever arm, the two leak candidates and the clean slice; the
day's queue complete.** The three micro retriever arms of the review
([infonce_micro.md](../measurements/infonce_micro.md)) were all read by the late morning: the
fresh twin `w0e3b` (the `w0e3` recipe on the reviewed code) at final CE 4.3996 and store recall@5
0.000; `l0e3` (the contrastive term on the first pass only, trunk gradient on) at recall@5 / @20
0.049 / 0.122 against `w25e3`'s 0.059 / 0.133 for 0.041 nats against the twin (the every-loop
pair 0.074), so the cost sits in the trunk's change and the decision of 2026-10-21 stands;
`seg_e3` (the selection loss weighted per query segment; launched 10:24 under
`ckpts/inject/queue_2026_10_25.sh`, 104.46M tokens in 40.0 minutes, exit 0) at CE 4.3671, 0.033
better than the twin in every condition, selection equal (0.7503 against 0.7538), recall@5 0.001,
and the paired per-token read on `inject_val` puts the twin 0.060 nats behind at exit 3 (z +8.1),
real between the two runs and attributed to the weighting on one seed; the default is the user's
call. The twin's resume on the evening of 2026-10-24 found the SFT resume epoch bug (accelerate's
prepared loader sets the dataset's epoch from its own pass counter, zero in a new process, so a
resume past epoch 0 trained on epoch 0's order again; fixed by `sft.select_epoch`,
`tests/test_sft_resume_state.py` test 8; no recorded run resumed past epoch 0), and the expired
watch that took the first launcher down at 20:58 set the detached launch rule. The afternoon queue
(`ckpts/inject/queue_2026_10_25b.sh`, detached) ran the paired `seg_e3` read, the two leak arms
and the clean-slice read.

The leak candidates ([leak_candidates_micro.md](../measurements/leak_candidates_micro.md)): the
copy-first warm-up recipe on the flagged splits of 2026-10-24, `m1` (`--mask-first-name-mention`,
1,474,885 tokens of 300M masked) and `g1` (`--support-gold-only`), each 100M on the swap rate 1.0
split, STOP on the 100M save, two rank reads, then `--switch-split` to the swap rate 0.15 split to
the epoch's end at 299.19M (`m1` warm-up 12:33 to 13:04, main 13:11 to 14:13, reads done 14:24;
`g1` warm-up 14:24 to 14:56, main 15:03 to 16:05, reads done 16:16; mains 60.5 and 60.6 minutes,
exit 10 at each stop and exit 0 at each end; `ckpts/evidence_leak_{m1,g1}`,
`ckpts/inject/leak_{m1,g1}.log`). At 100M every closed-book cell is inside 3 sigma on both arms
(largest |z| 1.6 and 2.0), with copying at 0.88 to 0.90 in distribution (the recorded warm-up arm
0.95 to 0.97; the pilot's switch bar is 0.9, and the micro launcher switches at a fixed 100M). At
the final, entity `delta` at tier 100 / 1000 in distribution `m1` 0.024 (z 3.6) / 0.073 (z 5.2),
held out 0.011 (z 3.1) / 0.032 (z 3.6); `g1` 0.020 (z 3.0) / 0.083 (z 5.6), held out 0.015
(z 3.3) / 0.035 (z 3.4); the warm-up arm 0.015 (z 2.6) / 0.085 (z 6.4) and 0.007 (z 2.0) / 0.047
(z 5.1). Paired prior-controlled `delta`, warm-up arm minus leak arm (`compare_cf_vs_leak_{ARM}.log`;
the lever needed z +3, the leak arm binding clearly less), in distribution / held out: tier 1000
`m1` +0.0115 (z 0.7) / +0.0147 (z 1.3), `g1` +0.0017 (z 0.1) / +0.0118 (z 0.9); tier 100 `m1`
-0.0094 (z -1.2) / -0.0039 (z -0.9), `g1` -0.0050 (z -0.6) / -0.0078 (z -1.5); the clustered z
within 0.1 of each. Copying is complete in distribution on both (gold top-1 1.000, swapped followed
1.000); held out the swapped card is followed 0.646 to 0.673 on `m1` and 0.782 to 0.804 on `g1`
(the warm-up arm 0.66 to 0.69). Verdict: both candidates are falsified, no cell near the bar; the
tier 1000 residual is neither the name's first mention nor the gold-absent documents' value spans,
and what the record lists as open stays open (the prior drifting, the input path). Side finding:
`--support-gold-only` raises held-out swapped-card following from about 0.67 to about 0.79 at no
cost, a candidate for the real-run data spec (decided 2026-10-25: carried into the real corpus builder behind a flag, RUN2.md row 3). Filler CE on `inject_val` ends at
3.886 (`m1`) and 3.889 (`g1`) against the recorded warm-up arm's 3.9106 and the original arm (c)'s
3.8842, the new arms tracking arm (c) within 0.002 to 0.018 after the switch: the 0.026 gap the
warm-up arm's record calls unexplained was the code of that week, not the warm-up. On the letter of
the R0b rule both arms sit at or past 3 sigma off their prior at tier 100 on at least one form
while every paired tier 100 cell is inside noise: the recipe's tier 100 margin is thin on one seed.

The clean-slice CE re-read ([clean_validation_slice.md](../measurements/clean_validation_slice.md);
`bash ckpts/inject/read_clean_val.sh`, 16:16 to 16:25, `eval_exit.py`, 40 batches of 4, no
evidence): CE per exit 1 / 2 / 3 on `inject_val_clean` (588,155 supervised tokens) arm P 3.3478 /
3.2739 / 3.2669, arm C 3.3503 / 3.2696 / 3.2650, arm E4 3.3474 / 3.2806 / 3.2740; on `inject_val`
(598,164 tokens) 3.4898 / 3.4166 / 3.4092, 3.4697 / 3.3894 / 3.3844 and 3.4932 / 3.4274 / 3.4201
(not the trainer's `[eval]` batches, so compared within this read only). The control's lead over
arm P at exit 3 is 0.025 on `inject_val` and 0.002 on clean text, so the control's lower held-out
CE (3.2195 against 3.2321 in the record) was the validation documents' presence in rows only the
control trains on (the review's 6.7%), not better language; on clean text arm P and the control are
equal and E4 is 0.007 behind P (the pilot's noise 0.013); the depth gains repeat (exit 1 to 3 about
0.08 on every arm and slice). The day's queue is complete; next is the run spec, with R5's eval set
when the card is idle.

**2026-10-25, evening: R5, the winner's eval set, resolved; the A-gate ladder on the written-bridge
final read** ([r5_bridge_ladder.md](../measurements/r5_bridge_ladder.md)). Two of R5's three parts
were already on record: arm P's final carries the complete ladder (2026-10-14, the decode reads
redone on the fixed packer 2026-10-24, the A4 flip on the bge route) and its 4-hop reads at depths
3 and 4 by read sites, at chance at every save (final 0.18 to 0.19 against 0.210); the bridge final
carries the 4-hop generate read (`pilot_bridge_final_gen_hop4`, 2026-10-24: 0.206 against chance
0.210, two steps written). "HotpotQA bridge by read sites" has no instrument: the bridge decode
exists only for the synthetic chain splits, so it is a build item, not a read. New tonight: the
ladder on `ckpts/evidence_pilot_bridge/checkpoint_evidence_final.pt`, arm P's final plus 66.35M
tokens of chain-only bridge SFT (R9; `config_pilot_bridge.yaml`, lr 3e-4 re-warmed on a trunk whose
cosine had ended at about 6.5e-5), the checkpoint form 6f builds on, to read whether the SFT keeps
the gates. `bash ckpts/inject/r5_bridge_ladder.sh` (launched 18:57 detached under
`TINY_LLM_CONFIG=config_pilot.yaml`; one process per read, skipped on relaunch when its json
exists; log `ckpts/inject/r5_bridge_ladder.log`, every read exit 0, complete 19:51), outputs
`ckpts/inject/pilot_bridge_ladder_{rank_none,rank_gold,rank_swapped,ceiling_fixed,abstention,store_pathway,store_edit_bge,exit_clean,bench}.{json,log}`.

- **A1(b), closed book: every cell at chance.** Entity `delta` in distribution 0.0023 / -0.0009 /
  0.0039 (z 0.9) / -0.0001 (z 0.0) by tier 1 / 10 / 100 / 1000, held out 0.0014 / -0.0036 / 0.0003
  (z 0.1) / 0.0063 (z 0.7); the largest other cell dates in distribution at tier 1000, 0.029 (z
  2.3). Arm P's final: 0.0096 (z 2.2) / 0.0639 (z 7.2) in distribution and 0.0125 (z 3.2) / 0.0458
  (z 5.5) held out. The pilot's tier 1000 residual is gone, with the weakly stored biographies.
- **Copying halved.** Gold card entity top-1 in distribution 0.755 / 0.757 / 0.755 / 0.797 by tier
  (normalized rank 0.006 to 0.009), held out 0.403 / 0.408 / 0.416 / 0.467 (rank 0.042 to 0.048);
  dates 0.35 to 0.41 on both forms, nouns 0.96 to 0.99. Arm P: 1.000 in every cell in distribution,
  0.94 to 0.97 held out. Swapped card followed 0.710 to 0.734 in distribution and 0.496 to 0.525
  held out, `mr_ll` 0.03 to 0.06, against 1.000 and 0.943 to 0.968.
- **The in-context ceiling probe (A2 / A5 in likelihood form).** Answer-span CE on `evidence_fixed`
  under gold 12.33, mixed 12.41, distractors 14.14, none 15.41 (arm P 3.49 / 4.10 / 6.20 / 6.05);
  ceiling 3.08, content 1.81 (arm P 2.56 / 2.71). The absolute levels are the answer format: the
  model expects to write "Step 1:" before an answer.
- **A3 in generation, gone.** 2,000 SQuAD v2 questions with the port: EM 0.000 and F1 0.008 under
  gold, false abstention 0.000; non-abstain 1.000 under none and distractors (arm P under gold EM
  0.046, false abstention 0.770; under none and distractors abstaining on 99 to 100%). Every
  completion starts with "Step 1:" (59%) or "Answer:" (41%): "Step 1: 1: 1: 1: 1:", "Answer: The
  Basisines.", "Step 1: Sustiigraphers Dive Biomcigginghers. Answer: Geesthraphers ...". The
  likelihood half still prefers the answer with the passage (real answer's span CE 11.69 under gold
  against 15.20 under none, +3.51 nats; arm P +3.75); memory mass at the last prompt position 0.52
  (arm P 0.16), AUROC 0.51; counterfactual follow 0.000 (n 218), `mr_ll` 0.328.
- **A7, pathway: EM 0.000** on the model, bge, oracle and none routes (n 6,000; F1 at most 0.013),
  against arm P's 0.001 / 0.044 / 0.088 / 0.000.
- **A4 on the bge route: unreadable**, 0 of 2,000 answered before the edit (arm P 95: delete flip
  0.90, edit 0.13).
- **Language.** `eval_exit.py` on `inject_val_clean` (40 batches, the same 588,155 tokens as arm
  P's read): CE 8.687 / 8.974 / 9.062 at exits 1 / 2 / 3, token top-1 0.059 / 0.058 / 0.058, the
  gain exit 1 to 3 -0.375 (z -64.4 clustered); arm P 3.348 / 3.274 / 3.267, top-1 0.42, +0.081. The
  most confident decile at exit 1 (p_max 0.996) sits at CE 14.4. The 13-task suite: mean
  multiple-choice headroom over chance +0.0191 over 9 tasks against arm P's +0.0615 and the
  control's +0.0725; lambada 0.002 against 0.115, sciq 0.451 against 0.599, arc_easy 0.297 against
  0.370, piqa 0.527 against 0.575, gsm8k 0.001 against 0.014, triviaqa and nq_open 0.000 on all
  three; the cells that rose (arc_challenge, openbookqa, mmlu) stay within about 0.03 of chance.

Verdict (the main thread's): the chain-only bridge SFT at the chain continuation's recipe (66M
tokens, the LR and schedule of `config_pilot_bridge.yaml`) is a demonstration that composition
happens in written tokens (R9's verdict stands: 0.997 held-out 2-hop) and not a usable checkpoint:
it erased the weakly stored biographies (the tier 1000 residual with them), halved the port's
copying, replaced answering and abstention on real questions with the bridge form or degenerate
text, and multiplied the clean-slice CE by about 2.8. Consequences for the spec, written into
[RUN2.md](RUN2.md) and 6f: 6f's mix carries the pretraining slices (evidence QA, the abstention
rows, the biographies at their tier structure, the plain-answer form) beside the bridge rows, at a
lower LR than the pilot's continuation, with the A gates and the clean-slice CE read at every save
of 6f and the kill rule on them; every A gate in the spec is taken on the shipped checkpoint, after
6e and 6f, not only at the end of pretraining; the pilot's tier 1000 residual is a property of the
pretrained trunk that a short post-training removes, so the leak reads after post-training are the
binding ones; "HotpotQA bridge by read sites" needs an instrument (the bridge decode over retrieved
real chunks), now in the build list with the 6f mix builder.

**2026-10-26: launch path L1 to L5 built** (`docs/plans/RUN2.md` launch-path table,
[RUN2_corpus.md](RUN2_corpus.md); uncommitted in the tree). The State block's "Built 2026-10-26"
bullet is the summary with the numbers; the details by item. Model set: no IR inner attention under
the direct read, the load-time drop of old checkpoints' keys, the FLOP line and the prefix attention
bill, the `detach_loops` clamp, the balance shadow stamp, the `ir_dim` assert; prints 34.9M / 31.3M
/ 61M (micro, seq 1024), 97.4M / 83.3M / 227M (pilot), 380.2M / 221.0M / 464M (`config.yaml`);
24 tests pass, the three recorded checkpoints load strictly bit-identical to HEAD, a detached Sonnet
review found no high or medium issue. Trainer: multi-GPU on the evidence path (per-rank document
shares, lockstep loaders, fp32 bucketed gradient averaging in place of DDP, one float64 all-reduce
per log step, rank 0 writes saves), exit 30 resume verification on the sft path, the token grid
snap, `keep_every_tokens`; the review's two high and two medium findings folded in. Run 2 config and
launch: 330.4M / 171.1M / 459M printed, 118.4M per evidence token; cost recomputed (643M with
evidence, about 67k tokens per second, 1,040 to 1,220 H100 hours for 280B, the control 27 hours,
ledger 1.26k to 1.54k, wall clock 6.0 days on 8 GPUs); the 30-step smoke at shape L on the 5090.
Corpus: T = 93.33B per split x 3 epochs, chain share 0.03, chat replay at supply, the 23.7 minute
smoke at scale 0.0005, filler and store builds not run locally. Next: the cluster.

**2026-10-27: MareNostrum 5, 800B, the cluster tooling built** ([RUN2.md](RUN2.md) row 14, "The
allocation and the cluster", the Schedule and Cost notes and "Built 2026-10-27";
[RUN2_corpus.md](RUN2_corpus.md); [RUN2_handover.md](RUN2_handover.md); uncommitted in the tree).
The allocation: MareNostrum 5 (BSC AI Factories), the ACC partition, 4 x H100 64 GB and 80 CPUs per
node, 72 hour jobs, 5,000 GPU hours applied and 4,500 targeted, 4 TB project storage, 2 TB transfer
in and out, no outbound internet on any node (login nodes included), Singularity only, QOS
`acc_ehpc`, exclusive nodes, probably billed per whole node. Decided (the user): the token target is
800B, 266.67B unique prompt tokens per split x 3 epochs (was 280B, 93.33B x 3). Config:
`target_tokens` 800e9, `warmup_fraction` 0.000655 (1000 steps; about 1.1k in practice, the plan
derived from the packed row estimate), the kill at 8B (5B could fire falsely at 16 GPUs), evals every
4B (2B cost about 10%), accumulation 1 at 16 GPUs (the job scripts pass 16 / GPUs), keep every 10B,
uploads off; the control constant at 4e-4 after its warmup (`lr_min_factor` 1.0, so the paired leak
read is not biased by an annealed control) with the kill off. Corpus per split: bios about 262.3B,
QA 1.53B at its supply, plain and bridge chains 1.4B each (absolute, not a share), persons
`1000:300,100:1500,10:3000,1:6000`, filler phase 1 about 260B plus phase 2 5B and ir 2.5B under
source caps; the QA build permuted once by a seeded `qa_shuffle` stage before the order (the QA
builder's round robin front-loads SQuAD and HotpotQA; found by the final corpus review); the warm
split the first about 28B of the main order and the control the first 10B (no evidence, the full
biography arm), both prefixes; epoch 0 of the main split led by the warm split's documents in their
order (`run2_main_train.order.json`, `epoch0_head_docs`), so the switch continues the stream; evidence
ratio about 0.045; disk peak about 3.06 TB, standing about 1.55 TB; the smoke at scale 0.0005 on the
final code every stage ok in 28.5 minutes and 2.8 GB. Cost: about 3,300 H100 hours at 67k tokens per
second per GPU, about 2,400 at 93k, the floor about 54k to fit 4,500; about 206 hours of wall time at
16 GPUs, at least 3 chained 72 hour jobs; the corpus build on the cluster about 400 GPU hours
(whole-node billing, estimate) plus the filler shards pushed in, or on the user's PC (about 3 TB of
local disk). Built: the corpus driver at 800B with its new stages (`qa_shuffle`, `order`, prefix
merges, `rows` for 1, 2, 4, 8 and 16 ranks, the prunes), resumable stages, per-stage locks, a
heartbeat and the `last_child_output` progress file; `scripts/mirror_sources.py`; the builders
resumable (`prepare_injection_data.py`, `prepare_chain_data.py` streaming, `merge_evidence_splits.py`
resume and prefix mode); in `sft.py` `--plan-split`, `--max-epochs-this-split`,
`--grad-accumulation-steps`, `--estimate-rows-only`, `--keep-stop`, the switch and head-docs checks,
the rows cache version 2 keyed on size and a content fingerprint, a moved `data_dir` as a plain
resume, the has-gradient flags in `all_reduce_grads`, log steps on sync steps; `scripts/slurm/` (the
job chain with its watchdog: a stall at 15 minutes, a save limit scaled by the GPU count, 49 minutes
at 16 GPUs, the loss-rise halt, the crash-loop rules, the save 20 minutes before the limit, HALT on
deterministic failures, a cancel respected); `scripts/ops/` (`push_code.sh`, `push_corpus.sh`,
`pull_exports.sh` with a 2 TB transfer ledger, `supervise.py` run by a local Claude Code `/loop` on
the user's PC every 30 to 60 minutes); `scripts/export_weights.py`. Reviews: three Opus final
reviewers (Slurm and ops, corpus, trainer and contracts) found no high-severity issue left in the
corpus and the trainer; the Slurm review's two highs (supervisor actions silently skipped, a finite
divergence unseen) and every medium and low were fixed; a detached Sonnet review of the trainer
fixes found no high or medium issue. Replaced in the State block: the "Next, in order" of
2026-10-26 (day 1 the filler download and the corpus stages on cluster CPUs, the 2-GPU check, the
10B control, the warm-up) and the launch row's wait on the two corpus confirmations (RUN2.md rows
12 and 13; row 12's share replaced in the 800B plan by absolute chain tokens; neither is on the
user's open list of 2026-10-27). Next: the user's cluster access and day 1.

**The ladder, in order** (done steps kept so the order reads whole):

1. **Done 2026-10-02: arm (a) in full.** The instrument reads (above).
2. **Done 2026-10-03: the copy control.** `--evidence prompt` on the 20M arm (c) save and on arm
   (a) final. It showed the 20M smoke was read before copying existed.
3. **Done 2026-10-03: the reader redesign and its matched control.** The key/value reader built and
   tested (`tests/test_evidence_kv_reader.py`), a 100M arm (c) smoke with it, and the cross reader
   resumed to the same 100M. The key/value reader copies; the cross reader does not.
4. **Done 2026-10-04: arm (b), `inject_masked`, in full.** Within 3 sigma of its prior at every
   tier on both forms; the in-context input path does not leak (above).
5. **Done 2026-10-05: arm (c) in full with the key/value reader. Fails.** `inject_retrieval_kv`
   resumed from its 100M save with the same flags to 299.22M tokens; every save from 100M on read
   with `gold`, `swapped`, `none` and `prompt` at `--batch-size 1024` (the 50M save with `none`
   and `gold`), then `compare full masked retrieval_kv`. The reader copies (in distribution top-1
   1.000, a swapped card followed on every item, from 150M on), and the verdict FAILS on both forms:
   tier 100 off its prior (0.027, z 4.1 in distribution; 0.013, z 3.6 held out), tier 1000 a leak
   on both (0.120, z 8.8; 0.071, z 8.2). The leak is there at 50M before copying and does not grow
   after 100M (above). The cross reader is closed at micro scale; its 100M save stays as the
   control.
6. **Done 2026-10-05: the copy-first warm-up arm. The R0b rule holds through tier 100, tier 1000
   leaks; R0b accepted on this arm 2026-10-05.**
   The sweep arm at swap rate 0.30 (`s30a50`) was set aside on 2026-10-05, on the reading that the
   swap rate acts only before copying exists (after it a real and a swapped span both cost about
   zero) and that 0.30 in place of 0.15 cuts the real-value share of the spans from 0.68 to 0.56,
   an expected 10 to 20% less leak. The warm-up arm's 150M read removes the support for that
   reading, so `s30a50` is reopened as a candidate, not run. The anonymization arm (`s15a90`)
   stays dropped (2026-10-04: it reaches at most about 0.7% of the supervised spans that see the
   real name). The warm-up arm,
   `inject_retrieval_kv_cf` (`ckpts/evidence_inject_retrieval_kv_cf/`), trains from
   `seed_micro.pt` with `--reader-kv` on `inject_retrieval_s100a50_train`, built with
   `prepare_injection_data.py ... --swap-rate 1.0 --anon-rate 0.5 --gold-drop-rate 0.2 --arms
   retrieval --suffix s100a50` (same seed, documents, order and card selection). Every span in a
   gold-present document carries a substitute that the card also carries, so there is no
   consistent name-to-value target to store; the value collisions in gold-less documents, about
   0.6% of spans, are the only real values left next to the real name. It trains until copying
   holds, planned stop at the 100M save; then the same run is relaunched with every flag and
   `--train-split inject_retrieval_train` to 300M (the data overrides are not stored in the
   checkpoint, and the resume position indexes the same permutation). No code change. Staged:
   `ckpts/inject/launch_kv_cf_warm.sh` (touches the `STOP` at the 100M save, then reads `gold`,
   `none` and `swapped` on the 50M and 100M saves). Criteria, fixed before the read:
   - At the switch: gold top-1 in distribution at least 0.9, and closed-book `delta` within 3 sigma
     of its prior at every tier. Above 3 sigma there falsifies "nothing to store": the input or
     encoder path stores before copying.
   - At 300M: the R0b rule on both forms (tier 100 within 3 sigma), tier 1000 read alongside with
     an expectation of at most about 0.03. A `delta` that climbs after the switch (paired z above
     3) falsifies the copy-first reading.
   - If it passes, the pilot inherits a requirement: fact spans are not supervised with real values
     before the reader copies (a warm-up or a copy criterion). Decided 2026-10-05: it does.
   - **First phase, read 2026-10-05.** Stopped at the 100M save (trainer exit 10,
     `checkpoint_evidence_tok100M_loss7.7318.pt`; filler validation CE 4.4870 against 4.4807 for
     the original arm at the same step). Both switch criteria are met: gold top-1 in distribution
     0.953 to 0.973, closed-book `delta` within 3 sigma of its prior in every cell at 50M and 100M
     (largest |z| 2.3). "Nothing to store" is not falsified: with no consistent real-value target
     the input side and the card path store nothing detectable, so the arm (c) store is written
     by supervised real-value spans. Copying is learned at least as well on fully swapped spans
     (paired at 100M against the original arm, entity in distribution: gold top-1 +0.037 to +0.061
     at tiers 1 to 100, z 5.0 to 8.0; swapped follow +0.057 to +0.113).
   - **Second phase, done 2026-10-05**: relaunched from that save with every flag and
     `--train-split inject_retrieval_train` (`ckpts/inject/inject_retrieval_kv_cf_main.log`,
     `launch_kv_cf_main.sh`), 299.19M tokens, 62.6 minutes, final filler CE 3.9106 (original arm
     (c) 3.8842: a 0.026 gap that opens after the switch and that the logs do not explain (explained 2026-10-25: the code of that week, not the warm-up)). The
     150M read already triggered the falsifier at tier 1000 (0.081, z 6.0; paired from 100M
     +0.082, z 5.9). Entity `delta` at the final save: in distribution 0.004 / 0.000 / 0.015
     (z 2.6) / 0.085 (z 6.4), held out 0.001 / -0.001 / 0.007 (z 2.0) / 0.047 (z 5.1); no tier 100
     cell in any class or form beyond 3 sigma; dates at tier 1000 in distribution 0.106 (z 3.6).
     Tier 1000 in distribution went -0.001 / 0.081 / 0.095 / 0.087 / 0.085 at 100 / 150 / 200 /
     250 / 300M
     (paired 200M to final -0.011, z -1.2): up within 50M tokens, then flat. `compare full masked
     retrieval_kv_cf`: HOLDS on both forms through tier 100 (margin in distribution z 2.6); tier
     1000 is a leak on both forms by the rule's own reading. The 0.03 expectation failed; the
     falsifier triggered at tier 1000 (and from 100M to final at tier 100, +0.020, z 3.3). Against
     the original arm at the final save, paired per form: tier 100 -0.012 (z -1.5) / -0.006
     (z -1.4), tier 1000 -0.035 (z -2.2) / -0.024 (z -2.2) in distribution / held out: lower in
     every cell, under 3 sigma per form. The `gold`, `swapped`, `prompt` reads at 150M, 200M and
     250M are on disk and not tabulated yet (250M `none`: 0.013, z 2.3 and 0.087, z 6.0).
   - **Reading.** Established: nothing is stored without a real-value target; real-value
     supervision after the reader copies still writes frequent facts into the weights; with the
     warm-up, tier 100 is inside 3 sigma on both forms. Both earlier readings hold in part: some of
     the original arm's leak was written before copying (the warm-up arm ends lower in every
     cell), the rest is a level that training maintains while real values are supervised next to
     the real name. Open: what sets that level. Candidates: the swap rate (swapped exposures
     penalize answering from memory at every point of training; untested), the gold-drop
     documents, residual span loss after copying.
   - **Decided by the user 2026-10-05 (decision A closed), both**: (1) the copy-first warm-up is
     accepted as the basis for R0b (the rule holds through tier 100 on both forms with it,
     marginally in distribution at z 2.6), the tier 1000 leak is recorded as known, and "no
     real-value fact supervision before the reader copies" goes into the pilot as a requirement, a
     new element of the pilot's schedule. The warm-up is the better of two arms on one seed, not an
     optimum, and its improvement over the original arm is a direction, not established per form.
     (2) The swap rate 0.30 branch runs before the pilot spec is frozen, because it sets a pilot
     parameter either way.
   - **The swap rate 0.30 branch, started 2026-10-05, read 2026-10-06.** Run name
     `inject_retrieval_kv_cf30` (`ckpts/evidence_inject_retrieval_kv_cf30/`, log
     `ckpts/inject/inject_retrieval_kv_cf30.log`, launcher `ckpts/inject/launch_kv_cf30.sh`). The
     split `inject_retrieval_s30a50_train` is built with `--swap-rate 0.30 --anon-rate 0.5
     --gold-drop-rate 0.2 --arms retrieval --suffix s30a50`; the warm-up arm's 100M save
     (`checkpoint_evidence_tok100M_loss7.7318.pt`) and a `run_state.json` at 100,419,895 tokens are
     copied into the new run directory, and the run goes to 300M with `--reader-kv` and
     `--train-split inject_retrieval_s30a50_train`. No code change. Every save past 100M gets
     `none`, `gold`, `swapped` and `prompt` reads into
     `ckpts/inject/rank_retrieval_kv_cf30<tag>_<mode>`. Criteria, fixed before the read:
     - The question: does the level that training maintains follow the swap rate. Read
       closed-book entity `delta` at tiers 1000 and 100, paired against the warm-up arm at the
       same saves (150M, 200M, 250M, final), per form. It follows the swap rate if tier 1000 is
       lower than the warm-up arm's at paired z at or beyond -3 per form at the final save, or on
       both forms pooled.
     - Power: the paired sigma at tier 1000 is about 0.016 in distribution and 0.011 held out (from
       the warm-up against original comparison), so only a drop of about 0.05 in distribution (the
       0.085 falling to about 0.035 or below) reads at 3 sigma per form. A pure dose effect
       (real-value share 0.68 to 0.56) predicts about 0.07, which this arm cannot separate from no
       effect. A null therefore means "no large effect of the swap rate", not "no effect".
     - Alongside: copying must not degrade (gold top-1 in distribution, swapped follow, held-out
       follow against the warm-up arm), tier 100 must stay within 3 sigma, and the filler CE gap is
       read again (the warm-up arm ended 0.026 above the original arm after its switch).
     - After it: if the level follows the swap rate, the pilot's swap rate is set from it; if not,
       the placeholder name in gold-present documents with the card carrying the same placeholder
       (about 25 lines in the builder and `biographies.render_store_chunk`) is the next candidate
       against tier 1000, and the pilot can start with the warm-up alone (the user's call then).
     - **Read 2026-10-06**: the swap rate does not move the tier 1000 level in any large way. Final closed-book entity `delta` in distribution 0.005 / 0.004 / 0.010 (z 1.7) / 0.083 (z 6.2), held out -0.002 / -0.001 / 0.003 (z 0.9) / 0.039 (z 4.6), against the warm-up arm's 0.015 (z 2.6) / 0.085 (z 6.4) and 0.007 (z 2.0) / 0.047 (z 5.1) at tiers 100 / 1000. Paired branch minus warm-up arm (same items), entity: tier 1000 -0.002 (z -0.3) in distribution and -0.008 (z -1.4) held out, -0.005 (z -1.0) pooled over forms, so the pre-registered criterion (z at or beyond -3 per form or pooled) is not met; tier 100 -0.004 (z -1.0) and -0.004 (z -1.4), lower on both forms as a direction, and the rule holds with more margin (z 1.7 and 0.9). The raw-rank z -4.0 in `compare_cf_vs_cf30.log` (all classes, pooled over forms) is mostly the real name ranking worse in the branch (+0.014, z 3.8) with the prior also worse (+0.005, z 1.2), so the `delta` moves only -0.009 (z -2.1). Within the stated power this is no large effect, not no effect: a dose-sized effect (about 0.07) could not be separated from none. Copying holds: gold top-1 in distribution 0.999 to 1.000 in both arms, swapped card followed on every item in distribution, held out 0.654 to 0.673 against 0.663 to 0.690 (all classes), `mr_ll` the same within 0.005; prompt copy held out 0.0084 (top-1 0.834), the same as the warm-up arm. Filler CE 3.9131 against 3.9106 for the warm-up arm. The 150M `none` read was taken (0.003, z 0.6 / 0.072, z 5.3 in distribution); the 200M and 250M reads and the other 150M modes were not taken.
     - Decided 2026-10-06 (the user): the pilot's swap rate stays at 0.15, the default. The user's reasoning: a leak of this size is not much of an issue for the pilot, and RL is planned later (for reasoning chains and generally), which can also act on small leaks. Caveat on record, not a counter-decision: RL is a Parked item (it unparks when pass@8 on the target task exceeds about 15%), and RL shapes behaviour at conflicts (prefer the card over memory) rather than removing what is stored, so the closed-book instrument would still read the leak; for small leaks the goal "facts in the store, not in the weights" is then read as behavioural at conflicts. The placeholder name arm is approved (ladder step 6; read 2026-10-08, below).
     - **The placeholder name arm, designed 2026-10-05/06; approved 2026-10-06, read 2026-10-08.** `--placeholder-rate 0.5` for gold-present documents: the subject's name is replaced by a placeholder in the document, the gold card is rendered with the same placeholder, each distractor card with its own placeholder; swaps on top; gold-absent anonymization unchanged; keys canonical; the eval renders its own cards with the real name, so `closed_book_rank.py` needs no change. About 40 lines in `prepare_injection_data.py` (`RetrievalBuilder.build`, argparse, metrics) and `biographies.render_store_chunk(name_override=)` plus about 30 lines of tests. Branched from the warm-up arm's 100M save (paired sigma about 0.006 against the warm-up arm, against 0.016 from the seed); build about 5 minutes and 1.8 GB, training about 64 minutes, reads 21 to 48 minutes. Supervision shares per exposure at p = 0.5 and swap 0.15: real value with real name 0.346 (0.686 at p = 0), real value with placeholder 0.346, substitute with real name 0.060, substitute with placeholder 0.060, unsupervised 0.189. Expected tier 1000 `delta` at the final save against the warm-up arm's 0.085: unchanged under the "written before copying" reading; 0.043 (z about -6) if the maintained level is linear in dose; 0.064 (z about -3) if it saturates. Criterion: the leak follows dose if paired z is at or beyond -3 per form or pooled; the lever is falsified if pooled z is above -2; p = 1.0 only on a null (it passes by construction and tests nothing). Rate 1.0 is not used because the real name would never be supervised with a value and the closed-book instrument would read zero by construction. The placeholder cannot carry to the pilot (nothing in the repo finds a subject name in real text; `entity_swap.py` is a heuristic gazetteer with no NER); what carries over is the result: if the leak follows dose, the trainer-side span weight by a copy criterion (an n-gram support match between document and evidence, the same test the builder uses) is worth building. Recommendation on record: run it before freezing the pilot's lever list and write the pilot spec in parallel.
     - **Read 2026-10-08: the dose lever is falsified.** Built with `--placeholder-rate 0.5 --arms
       retrieval --suffix s15a50p50` (swap 0.15, anon 0.5, gold drop 0.2, seed 42; 4 minutes):
       `inject_retrieval_s15a50p50_train`, 558,657 documents, 298.26M tokens, 64,664 of 129,438
       gold-present documents renamed (0.4996); `.evkey`, `.evgold`, `.cond`, `.evkeyidx`, `.ans`,
       the 97,395 swaps and the 16,294 anonymized gold-absent documents identical to the arm (c)
       split. The builder change (`--placeholder-rate`, default 0 and byte identical to the old code
       at 0; `render_store_chunk(name_override=)`; placeholder draws on their own rng stream) is
       tested in section 8 of `tests/test_prepare_injection.py`. Run `inject_retrieval_kv_cf_p50`
       (`ckpts/evidence_inject_retrieval_kv_cf_p50/`, log `ckpts/inject/inject_retrieval_kv_cf_p50.log`),
       from the warm-up arm's 100M save to 298.27M tokens in 68.5 minutes; filler CE within 0.003 of
       the warm-up arm at every matched step, final 3.9161 (warm-up 3.9106, swap 0.30 branch 3.9131).
       Final entity `delta` in distribution 0.005 / 0.001 / 0.0185 (z 3.0) / 0.0986 (z 6.7), held out
       0.000 / -0.003 / 0.0133 (z 3.7) / 0.0559 (z 6.4). Paired against the warm-up arm, entity: tier
       1000 +0.014 (z +1.3) in distribution, +0.009 (z +1.3) held out, +0.011 (z +1.8) pooled; tier
       100 +0.004 (z +0.8), +0.006 (z +2.2), +0.005 (z +1.8) pooled; the real name and the prior both
       rank better (tier 1000 -0.025, z -4.6 and -0.014, z -2.5), so `delta` does not move. Pooled z
       +1.8 is above -2: falsified; the "unchanged" reading came true, neither dose prediction did.
       `compare full masked retrieval_kv_cf_p50`: HOLDS in distribution (tier 100 at z 3.0), FAILS
       held out at tier 100 (0.0133, z 3.7), where the warm-up arm held on both forms. Copying in
       distribution unchanged (gold top-1 1.000, swapped followed on every item, `mr_ll` 0.000); held
       out degraded (entity gold top-1 0.53 / 0.53 / 0.53 / 0.68 against 0.60 / 0.63 / 0.61 / 0.74,
       z -2.5 to -8.0; swapped followed 0.51 to 0.54 against 0.60 to 0.63, z -3.4 to -8.1; `mr_ll`
       0.034 to 0.066 against 0.027 to 0.049); prompt copy held out better (top-1 0.84 to 0.87,
       z +3 to +3.7). Reading: the maintained tier 1000 level is not proportional to the supervised
       spans that pair the real name with the real value, so a span weight by a copy criterion,
       which cuts exactly that count, is not worth building. What writes the level stays open (the
       gold-absent documents with the real name and unsupervised values, the input path, the prior
       drifting under real-name exposure; the masked arm's 0.012 at tier 1000 is that family's
       floor). The held-out copy loss reads as the reader binding on the name: with half the cards on
       a placeholder, name matching is trained less and the held-out templates suffer (a reading,
       not established). Record: [r0b_micro_pilot.md](../measurements/r0b_micro_pilot.md).
     - **Dropped 2026-10-08: the span weight by a copy criterion in the trainer** (the item that was
       to follow these arms), on the placeholder arm's null (Decisions, real run ordering). The goldfish loss is expected to do no
       better than a dose cut, since the renders are paraphrased; it is not planned.
   - **Then: the pilot spec** (written 2026-10-09), with the copy-first warm-up requirement and swap rate 0.15 and no
     further lever against the leak; the chain depth arm (step 8) read inconclusive on 2026-10-08
     and changes nothing in it.
7. **Done 2026-10-04: the loop reading on arm (a).** `closed_book_rank.py bios --n-loops 1|2|3` on
   arm (a) final. At tier 100 the `delta` grows with depth beyond its paired sigma, almost all of it
   in the second pass; at tier 1000 the real name is saturated after one pass (above). The axiom
   was reworded on the same day (Decisions, loops).
8. **Dropped 2026-10-06: the learned exit gate arm. In its place: the chain depth arm, approved
   2026-10-06, spec written, run and read 2026-10-08: inconclusive.** The gate (an Ouro-style per-token exit distribution that only
   reweights the per-pass losses, never built) is dropped on the per-exit read: the second pass's
   gain is spread evenly over tokens, the third adds 0.006 nats, and the gate's own optimum is
   near uniform. Its three open questions (the skip-compute exemption, the equal-weight control
   arm, the pilot spec without exits) are closed as moot. The depth reading moves to the chain
   corpus as one micro arm: build the chain training and validation splits
   (`prepare_chain_data.py`, the `train,val` splits on cuda; only `chains_eval`,
   `chains_heldout_tmpl` and `chains_hop4` exist), train one micro arm from `seed_micro.pt` with
   `--evidence --reader-kv` on them, and read it with `eval_chains.py` by depth and by kept read
   sites, which is the loop axiom's computation-half falsifier at micro scale (L1: read sites past
   the first pass must add at least 5 points on held-out 2-hop at depth 3), plus the per-exit
   read. Its spec (token target, splits, pass and kill criteria, cost) is written before launch.
   - **Spec written 2026-10-08:** [chain_depth_micro.md](../measurements/chain_depth_micro.md)
     (the commands, packing, memory fallback and secondary reads are there). Token target about
     104.7M tokens, one epoch of 2M questions (`--train-questions 2000000 --val-questions 2000`; no
     flag sets tokens, the split is the budget). `chains_heldout_tmpl` is rebuilt at 2,000 questions
     for power (5 points is 3.9 sigma at n 2,000 against 2.7 at 1,000); its first 1,000 draws are
     the old questions. L1 at the final save, `Delta = acc(held-out 2-hop, D 3, kept 3) - acc(same,
     D 3, kept 1)`: pass at Delta at least 0.05 and at least 3 sigma; falsified when the
     preconditions (valid, readable, composes in distribution) are met and Delta plus 2 sigma is
     under 0.05; otherwise inconclusive. `eval_exit.py` cannot read this arm (it refuses splits
     with evidence, and the evidence-free splits are prose the arm never saw); the full-site cells
     at depth 1, 2 and 3 replace the per-exit read. `[eval fixed]` and the kill stay off. Cost:
     build 30 to 60 minutes, training 32 to 58 minutes, reads 10 to 30 minutes.
   - **Run 2026-10-08:** splits built (held-out at 2,000, train 2M questions, 21 GB, about 55
     minutes); trained on the fallback config (batch 16 x accum 2, the same tokens per update)
     after the batch 32 launch spilled into shared memory in minute one (32 GB, 10k tokens/s), as
     the spec allowed for: 102.66M tokens in 51.5 minutes, every packed row closed by the evidence
     budget.
   - **Read 2026-10-08: inconclusive.** Every accuracy cell at chance at 50M and final; the
     readable precondition (1-hop at depth 3 at least 3 sigma over chance, 0.146) fails at 0.104
     (chance 0.115); held-out Delta +0.006 at sigma 0.013. The model copies an answer-type entity
     from the buffer (answer CE per token 3.82 without a read site, 0.47 with them) and never
     learned to select which one, so 0.455 nats per answer token is about 3.3 nats per answer,
     more than a uniform pick among the candidates (about 1.9). The selector stayed near uniform
     and the third pass's `loop_scale` fell under 0.01 by 50M (0.07 at the end). Under the
     "Inconclusive" clause nothing changes: depth and the pilot's chain slice (10%, hops 25 / 50 /
     25) stay as they are, and the pilot's own L1 (sublayer sites against passes, at its shape and
     dose) is the deciding read. A micro read would need a separate decision (a larger model or
     dose, a 1-hop curriculum first, or a selector loss that is not left uniform), recorded as an
     option, not a step. Instrument gaps found (no per-question record in the JSON, answer CE for
     the 4-hop split only, the `*` marker glued to the previous column) are listed in the record.
9. **Done 2026-10-06: the seed-side instruments**
   ([seed_instruments.md](../measurements/seed_instruments.md)). PopQA with its prior control
   resolves the seed's weak stored knowledge (entity `delta` 0.029 / 0.026 / 0.036 by tail / mid /
   head, most of it in four cue relations), and 10M tokens of evidence finetuning moved neither the
   PopQA `delta` (arm A -0.0002, z -0.4 over all entity items) nor counterfactual following (paired
   gap +0.01, z 0.1 pooled; the seed's `mr_ll` gradient is a frequency-ratio prior). The store
   build, the edit store and the chain splits are CPU work that can run beside any GPU arm.
10. **Spec written 2026-10-09 ([PILOT.md](PILOT.md)), build list done 2026-10-10, arm P launched
    the same evening and switched to the main split at 300M with the criterion met and the 1-hop
    gate passed; arm P complete 2026-10-11, arm C 2026-10-14, verdicts decided 2026-10-17**
    (Phase 5): the key/value reader and its rotation, the copy-first warm-up as a requirement, and
    swap rate 0.15, with no further lever against the leak (the copy-criterion span weight dropped
    2026-10-08); the chain slice stays at 10% with hops 25 / 50 / 25, and the pilot's L1 is the
    deciding read on the loop clause (the micro chain read was inconclusive), behind a 1-hop gate.
    The spec cuts the ladder to today's code: the blueprint items (coda, tied table, always-on
    selector, sublayer read sites, depth draw with a detached prefix, last-pass loss, union
    selection loss, InfoNCE, the natural-text corpus builder) are deferred to arms after the pilot
    reads or to the real-run spec, and R4, R4b and R6 are cut (superseded 2026-10-23, the run order). Cost measured on the probe shape
    (99.1M params): 44k to 50k tokens per second on low-evidence rows, 2.7k to 3.9k on
    `evidence_train` at the 4608 cap; the pilot is about 35 to 45 GPU hours (estimate), two arms of
    about 1B tokens.


Batch caveat: `config_micro.yaml` moved from batch 16 x accumulate 2 to 32 x 1 for throughput.
That changes nothing the induction-head result keys on: formation is set by tokens per update x
updates (batch x context, arXiv 2511.16893), and both settings put 32 rows of 1024 tokens into
every optimizer step. Copying emerged between 50M and 100M for the key/value reader (gold top-1
0.01 at 50M, 0.91 at 100M, 1.00 from 150M), so nothing is blocking. If a later
arm copies late or weakly, the lever is fewer tokens per optimizer step (16 x 1, twice the updates
per token, a larger batch giving weaker heads), not the accumulation split.

Relaunching: a stopped arm is relaunched with the same `-c` and every flag (`--data-dir`,
`--train-split`, `--val-split`, `--reader-kv`, `--run-name`). Since 2026-10-24 the checkpoint
stores its splits, and a relaunch on a different split than the checkpoint recorded is refused
unless `--switch-split` is passed (the warm-up to main switch is the one place that passes it). The STOP file stays in the run
directory after a stop (`ckpts/evidence_inject_retrieval/STOP`, `.../evidence_inject_retrieval_kv/STOP`);
delete it before a resume or the run exits at its first poll. The warm-up arm's launcher leaves one
in `ckpts/evidence_inject_retrieval_kv_cf/` at its 100M save; the relaunch on
`inject_retrieval_train` deletes it first. Throughput at the micro shape: 55k to 60k tok/s typical,
13.3 GB peak with the card buffer, under either reader.

Seed baselines for the graft lineage (2026-09-30, unchanged): fixed-target gold gain -0.0020 nats,
per-token chunk AUROC 0.515 / 0.484 / 0.490 by loop, mass/chunk AUROC 0.481 / 0.470 / 0.469,
grounded AUROC 0.5000; `g_proj` and `direct_gate` exactly zero. Corpus `evidence_train` 963,011
conversations, ratio 3.75; held-out `evidence_dev` and `evidence_fixed` from `--heldout`.

## The goal

> All world facts come from an external store through the retrieval pathway: the selector chooses
> chunks by embedding, the evidence reader carries their exact tokens into the model. The trunk
> holds language, common sense and reasoning skill; reasoning over the retrieved facts happens in
> written tokens, and the looped block is computation per token.

In short: facts in the store through the retrieval pathway, reasoning in written tokens, the looped
trunk as computation per token. Reworded 2026-10-25 (the user, on the run spec's first table, row
4, [RUN2.md](RUN2.md)) on the measurement: latent composition at chance at every pilot read (L1
failed at this shape, 2026-10-17), written composition 0.997 on held-out 2-hop (R9, 2026-10-24).

The original goal (2026-09-30), kept as the aspiration to revisit after the real run:

> All world facts come from an external store through the retrieval pathway: the selector chooses
> chunks by embedding, the evidence reader carries their exact tokens into the model. The trunk
> holds language, common sense and reasoning, and the looped block is where reasoning over the
> retrieved facts happens.

In short: facts in the store through the retrieval pathway, reasoning in the looped trunk.

Two things the wording settles:

- **The IR expert is the selector, not the information channel.** Its external value is a
  projection of one pooled 384-d vector, so no span can be copied through it. Content flows through
  the key/value reader (the evidence tokens as leading keys and values of the shared
  self-attention; Decisions, roles). Making the selector carry content would compress chunks into one
  vector, which published work shows recovers gist, not facts. The goal is stated over the pathway.
- **"No facts in the weights" is the strong form and nobody has demonstrated it.** The reachable
  target is no long-tail or episodic facts in the weights, with a language and common-sense core
  kept. What the trunk keeps and what the store holds:

| information type | lives in |
|---|---|
| entity, episodic and long-tail facts, numbers, dates | store |
| rare-word and domain definitions | store |
| syntax, common word meaning, coreference | trunk |
| common sense and basic priors | trunk |
| procedural and reasoning skill, code, arithmetic | trunk |

Two mechanisms carry the goal and both are measured separately:

- **Externalization**: the trunk is never rewarded for producing a fact the buffer does not supply,
  and it is penalized where recall from weights and evidence disagree. This is a pretraining
  property. Every graft onto the converged checkpoint measured zero, and the published systems
  that externalize facts impose it from token 0.
- **Reasoning over evidence**: the selector and the reader query the state at their read site,
  and chunks are encoded independently, so a chunk that depends on a fact in another chunk is
  unreachable until that fact is in the state. k hops over evidence need at least k sequential
  read sites before readout. In the built POC that is one read per loop from `step_input`, so
  there k hops need k loops. In the pilot design the reader sits in sublayers 1 and 3 of every
  pass, so 2 hops fit in one pass (4 with the prelude-read arm). The architecture permits depth;
  the data and the objective have to ask for it, and whether the loop, rather than the sublayers
  of one pass, does the composing is an axiom with a falsifier (Decisions, L1). L1 failed at
  pilot shape (2026-10-17), and since 2026-10-25 the goal puts the composing in written tokens, so
  L1 is read descriptively in the run spec.

## What must be proven

The gates below replace the POC's older G-gates for everything after Phase 4. Each is a metric, a
dataset, a bar against a noise floor, and a script that holds it. Thresholds are relative to
[noise_floor.md](../measurements/noise_floor.md); it has no CE sigma yet, so a paired-bootstrap CE
sigma is recorded before any CE gate is quoted.

| gate | metric | bar | home |
|---|---|---|---|
| A1 closed-book externalization | (a) closed-book likelihood rank (or MC accuracy) of the gold answer among at least 20 same-type candidates, pathway off, neutral prompt, head and tail obscurity tiers, entity-valued relations; (b) fact injection: fictional biographies (bioS style, Allen-Zhu and Li, Physics of Language Models 3.1) inserted at 1 / 10 / 100 / 1000 exposures with their chunks in the store | (a) tail-tier rank worse than the matched no-retrieval control by at least 3 sigma (bootstrap); (b) closed-book rank at chance up to 100 exposures where the control memorizes (the micro form, R0b); at pilot power, amended 2026-10-17 by the user: the control at least 3 sigma above its prior at tier 100 or 1000, the paired `compare` puts the pathway arm under the control at tiers 100 and 1000 on both forms by at least 3 sigma, and the pathway arm's own tier 100 residual is recorded as a known leak when under a quarter of the control's (PILOT.md read table) | new closed-book rank scorer, PopQA and the tiers (Phase 4b build item); R0b. Method (2026-10-06, [seed_instruments.md](../measurements/seed_instruments.md)): read `delta` against the subject-blind prior, never the raw rank; report the cue relations (father, mother, capital, capital of) separately; the no-retrieval control must sit above its prior at the tail before the pathway arm is read against it (the seed's non-cue tail `delta` is 0.0088, paired sigma about 0.0012) |
| A2 evidence recovery | fixed-target answer-span CE on held-out rows, gain = CE(none) - CE(cond) | gold gain at least 1.6 nats (50% of the 3.23-nat in-context ceiling); distractors gain at least 0 | `sft.py` `[eval fixed]`; `eval_abstention.py --evidence-port` cross-check |
| A3 counterfactual faithfulness | memorization ratio p_orig / (p_orig + p_swap) (Longpre et al. 2021) under an entity-swapped gold chunk, on items answered correctly with the unswapped chunk, stratified by entity frequency, injected high-exposure facts included so conflicts exist at pilot scale; n reported per stratum | memorization ratio at most 0.05 and follow rate at least 90% in every stratum | new condition in `eval_abstention.py`. Method (2026-10-06, [seed_instruments.md](../measurements/seed_instruments.md)): read the high-frequency strata paired against a neutral-port baseline or binned by the frequency ratio of original to substitute, never by the original's count alone (the seed's `mr_ll` runs 0.050 to 0.957 across ratio bins with a neutral port) |
| A4 store edit | edit or delete the gold chunk and its near neighbours in the store, re-retrieve with the model's own query, answer; not the oracle buffer | flip rate at least 70% on items answered correctly before the edit | two-store swap through the model-query retrieval path |
| A5 selector | chunk AUROC (per token, chance 0.5) per loop; mass/chunk AUROC gold-present vs distractors; grounded AUROC (evidence rows) | chunk AUROC at least 0.59 per loop (3 sigma of 0.030 over 0.5); mass/chunk and grounded AUROC at least 0.674 (0.584 + 3 x 0.030) | `sft.py` `[eval fixed]` (Phase 4 home); `eval_abstention.py` cross-check |
| A6 reasoning retained | HotpotQA, BoolQ and SciQ through the port against in-prompt, counted only on tasks whose in-prompt score is at least 3 sigma above chance; language benchmarks within noise | port at least in-prompt minus 2 points, judged at the binomial sigma; control is the pilot's own in-prompt score | `eval_benchmarks.py` with an evidence path |
| A7 store-level retrieval | held-out NQ, TriviaQA and HotpotQA with gold passages in the store: recall@k of the model's query; open-book EM through the whole pathway | model-query recall@k at least bge-query recall@k; pathway EM read against oracle-buffer EM | new, Phase 4b |
| L1 composition | synthetic k-hop chains with fictional entities, accuracy by hops and by sequential read sites, at fixed D = 3 with the reader output zeroed at every site after site j (read ablation) | every cell with fewer read sites than hops at chance, 1 / (type-matched candidate answers in the buffer); held-out-template 2-hop above the control (the read-ablated model or the per-loop-CE arm) | synthetic eval, Phase 4b |
| L2 depth pays | held-out 4-hop and HotpotQA bridge answer CE by read sites at D = 3 (read ablation), plus D = 4 against D = 3 | rises with read sites past the first pass; at least 0.05 nats on bridge CE | synthetic eval plus `eval_abstention.py` |

The old A1 bar (tail tiers near chance, at most 2x the seed) could not separate "facts
externalized" from "too small to know": pythia-410m at 300B tokens already reads TriviaQA 0.021
and NQ 0.000, under that bar, and every 120M pilot control would pass it. The rank bar is relative
to the matched control, and the injection probe makes exposure count the variable. The A1 claim
covers entity-valued relations only, because the fact tagger covers only those; extending it to
attribute nouns (occupation, nationality, genre) extends the claim. A3 is read at every
checkpoint, because context reliance is known to rise early and then decay under finetuning on
context that agrees with the weights; the injected high-exposure facts are what makes a
weights-against-chunk conflict exist at pilot scale.

L1's validity rule: hops are bounded by sequential read sites before readout, so anything above
chance in a cell with fewer sites than hops is a leak in the data. It is scored by read ablation at
fixed D = 3, not by truncating depth: the minimum depth is 3, so 1- and 2-loop exits are untrained
and a truncated cell at chance would be readout failure. Chance is 1 over the number of
type-matched candidate answers in the buffer. The control is never the no-retrieval arm, which
cannot see the chains.

## Rules (carried over)

- Commit per logical change (`feat:` / `docs:` / `chore:`, branch `ir-train-build`, PRs to
  `prototype`), single-line subjects.
- Lowercase explanatory comments that justify why; Google-style docstrings with `Args:`.
- Anything touching `has_mtp`, `lm_head`, `mtp_head`, `_token_tracker`, `moe` in the training loop
  goes through `accelerator.unwrap_model(model)`.
- **Never add `.item()` / `.tolist()` / `.cpu()` / boolean mask indexing to the per-step path.**
- **Every reported number is measured at fixed eval flags**, quoted against the noise floor.
- **Looping is a requirement.** A later loop that does not earn its compute is a defect to fix,
  never a reason to ship shallower.

## Decisions

| decision | choice |
|---|---|
| goal | The reworded goal above. Strong form is not a gate; A1 with A3, A4 and A7 is. Its reasoning clause reworded 2026-10-25 (the user; Decisions, run spec 2026-10-25), the original kept in the section as the aspiration to revisit after the real run. |
| A1(b) reading | Amended 2026-10-17 (the user) after the pilot read: the control at least 3 sigma above its prior at tier 100 or 1000, the paired `compare` puts arm P under the control at tiers 100 and 1000 on both forms by at least 3 sigma, and arm P's own tier 100 residual is recorded as a known leak when it is under a quarter of the control's. The original "arm P within 3 sigma of its prior per form" was written for micro power and flips on noise at pilot power; the published relatives (LMLM, Co-LMLM) judge the same residual against the standard model. Passed on the pilot with the leak known (arm P held-out tier 100 0.0125 at z 3.2, 4x under the control; 47x under it in distribution). The person-clustered sigma, run 2026-10-24 on the pilot rank JSONs, moves the sigma by 0.9x to 1.1x: within 10%, the decided reading stands. |
| roles | Selector = the IR path, chooses by embedding. Reader = the evidence tokens as leading keys and values of the shared self-attention in every loop, carries content. Measured 2026-10-03 at micro scale: at matched 100M tokens this read copies (gold card top-1 0.92, swapped card followed 92%) where the separate cross-attention reader does not (top-1 0.01, followed 2%). In the full arm (c) (2026-10-05) it copies completely from 150M on in distribution (top-1 1.000, a swapped card followed on every item); held out it follows a swapped card 0.61 to 0.64 of the time. The cross reader stays loadable as the legacy mode. |
| learned IR table | Dropped from the real run. Zeroing its read costs 0.0002 nats on every arm. |
| selector in the real run | Always on, not routed; scores raw bge keys plus one learned null key; no value read, no key adapter. Superseded 2026-10-25: not in the RUN2 spec (RUN2.md "Not in the spec"). |
| selector to reader coupling | The selector's own logit enters the reader's attention logit through a one-hot chunk channel, per token, causal. Replaces the document-mean sigmoid gate. Carries over to the key/value read as a head-dim extension: the channel on the evidence keys, zeros on the segment's own keys. Superseded 2026-10-25: not in the RUN2 spec (RUN2.md "Not in the spec"). |
| reader positions | Prefix rotation: each segment's evidence keys are rotated at the positions the evidence would hold as the text right before the segment, so a query sees the offsets of in-context copying. Supersedes "no rotary", which was a cross-reader decision (Phase 4 arm B: +40% content gain without rotary, still an order under the bar). |
| reader output | No separate output: the read lands through the shared self-attention's own `o_proj`, inside the always-on seed; `evidence_loop_scale[k]` (init 1) scales the evidence values per loop. Not neutral when evidence is attached (the read shares the softmax), which from-scratch training does not need; with no evidence attached the forward is bit-identical to the model without the port. |
| evidence encoder | The trunk's own dense decoder, chunk-causal. All 8 layers in the POC; the first 4 prelude layers under TE checkpoint from scratch. No cached encoder states in the store. |
| readout | A 2-layer coda after the loop, own KV slots. The loop refines, the coda reads out. 2026-10-24 (the user): the coda stays unbuilt; R4, the arm that would have read it, is dropped as obsolete on R9. |
| loss placement | One CE on the last pass run, through the coda. Per-loop CE is the control arm, not the default. 2026-10-24 (the user): R4, the arm that would have decided this row, is dropped as obsolete on R9 (composition in written tokens); the row stands as an unread proposal and arm P's per-pass loss without a coda is the measured recipe. |
| depth | Drawn per step from {3: 0.7, 4: 0.2, 5: 0.1}; the first D minus 3 passes run under `no_grad` and are detached; minimum 3, so no pass is an exit. Inference 3, at most 4. R1b (2026-10-21, [depth_memory_probe.md](../measurements/depth_memory_probe.md)): depth 5 trains at pilot shape at batch 4 x 2; the detached prefix (`detach_loops`, built) cuts the prelude off the language loss until a straight-through route to the first live pass exists, so it is not a training recipe yet. Superseded 2026-10-25: not in the RUN2 spec (RUN2.md "Not in the spec"). |
| loop tensors | `loop_scale` and `evidence_loop_scale` length 5, fresh `1/sqrt(5)`; `loop_inject` on; loop code clamped to the trained depth. Superseded 2026-10-25: not in the RUN2 spec (RUN2.md "Not in the spec"). |
| selection supervision | Union over the passes in the gradient window: `u_c = 1 - prod(1 - p_hat[k, c])`, BCE against the gold flag. Never a target per hop and per loop. Superseded 2026-10-25: not in the RUN2 spec (RUN2.md "Not in the spec"). |
| query training | Decided 2026-10-21 (the user): the real-run spec carries no retriever term and no separate search tower (held, not built); bge keys the store and embeds the question at serving, A7's recall half stays untested, and the model's own retrieval is revisited at the real run's size, with `evidence_contrastive_loops` and `evidence_contrastive_trunk_grad` kept in the code as the knob for that revisit ([infonce_micro.md](../measurements/infonce_micro.md), [query_head_probe.md](../measurements/query_head_probe.md)). The review of 2026-10-24 proposed two micro arms before the spec closes this row (the term at loop 1 only with the trunk gradient on; the selection loss weighted per segment; Decisions, repository review); approved by the user the same day and running since about 20:25 with a fresh `w0e3` twin on the reviewed code (`ckpts/inject/launch_review_arms.sh`: `w0e3b`, `l0e3`, `seg_e3`). Read 2026-10-24 and 25 ([infonce_micro.md](../measurements/infonce_micro.md)): the term on the first pass alone keeps 0.049 of `w25e3`'s 0.059 recall@5 at 0.041 nats against 0.074, the per-segment selection weighting costs nothing and reads 0.033 to 0.060 nats better (decided 2026-10-25 as the default); the row is closed as decided 2026-10-21, no retriever term; `evidence_contrastive_loops` defaults to `[0]` since 2026-10-24. Was (the proposal before the pilot): infoNCE on the in-model query against the bge key of the gold chunk, in-batch negatives plus store-mined hard negatives, same-document and near-duplicate negatives masked. Keys stay bge, so the index is never rebuilt; Atlas (Izacard et al. 2022) justifies query-side-only tuning against a fixed index. |
| token tables | One table, tied to the LM head and the MTP head through small adapters; no weight decay on it. The per-layer and router tables become projections of it, gated on the per-layer table zero-ablation on the seed (Phase 4b). G9 gates the head tie only. Superseded 2026-10-25: not in the RUN2 spec (RUN2.md "Not in the spec"). |
| shape | Shape L: width 1024, 16 heads x 64, 4 KV heads; 8 prelude layers; looped core of 3 sublayers per pass with 8 MLP experts each, top 2; 2 coda layers. About 470M total, 300M active per token. Sublayer count and expert count are pilot arms. After arm E4 (2026-10-24, [expert_count_pilot.md](../measurements/expert_count_pilot.md): 4 experts against 8 at the same active compute cost nothing measurable, two changes and one seed each) the expert count is a cost choice, not a quality one, until a read at the real run's size; the sublayer count is open (the review proposes a cost choice too). Decided 2026-10-25 (the user; Decisions, run spec 2026-10-25): shape L is the configured `config.yaml` shape, width 768, with 32 experts and today's one block per pass; this row stays the blueprint on record. |
| context | 4096 for the pilot and the main run; a short 8k to 16k extension phase at the end on long documents with retrieval attached. Never 32k from the start. |
| real run ordering | Retrieval-augmented pretraining from token 0 with fact spans kept out of the loss. No plain pretraining plus a graft. Decided 2026-10-05: fact spans are not supervised with real values before the reader copies (a copy-first warm-up, a new element of the pilot's schedule). R0b arm (c) leaked before its reader copied (50M) and did not grow after (100M on); the copy-first warm-up arm stored nothing without real-value targets, held the R0b rule through tier 100 at 300M, and still stored tier 1000 facts after copying (0.085, z 6.4 at the final save), recorded as a known leak. The warm-up is the better of two arms on one seed, not an optimum. The swap rate 0.30 branch (read 2026-10-06) left tier 1000 where it was (paired -0.002, z -0.3 in distribution), and the pilot's swap rate stays at 0.15 (decided 2026-10-06 by the user: a leak this size matters little for the pilot, and RL planned later can act on small leaks; caveat on record: RL is Parked and shapes behaviour at conflicts rather than removing what is stored, so for small leaks the goal is read as behavioural at conflicts). The placeholder name arm (read 2026-10-08) halved the supervision that pairs the real name with the real value (0.686 to 0.346 of exposures) and left tier 1000 where it was (paired +0.011, z +1.8 pooled; falsified at pooled z above -2). 2026-10-08: the copy-criterion span weight is dropped on the placeholder arm's null; the pilot's lever list against the leak is the copy-first warm-up alone, at swap rate 0.15. 2026-10-24 (the user): R6, the pretraining-text leak probe, is skipped; the spec takes the lever already read at micro (arm (a) stores plain-text facts under the normal loss, arm (b) with the fact spans masked stores nothing, arm C repeats arm (a) at pilot shape): the spans to keep out of the weights are masked, or plain-text facts are accepted into them. |
| real run budget | A token target is fixed and the hours derived from it (6b). About 2.5k H100 hours planned for shape L, which buys about 270 to 290B tokens with evidence at today's MFU; the 5k-hour ceiling stands. Estimates until the constructor prints them; recomputed after Phase 6a's throughput work. Decided 2026-10-25 (the user; Decisions, run spec 2026-10-25): the target is 280B tokens and the application asks 2.5k H100 hours plus the corpus build line plus a 20% buffer (about 3.0k), ceiling 5k; the 2.5k figure was computed for the blueprint width, at the configured width 280B costs about 1.1k to 1.3k ([RUN2.md](RUN2.md), Cost). |
| MoE balancing | Proposed 2026-10-12; R8 passed for loss-free on all three bars 2026-10-19 and 2026-10-20 ([lossfree_micro.md](../measurements/lossfree_micro.md)): loss-free balancing (a per-expert bias on the routing score for selection only, moved by the centred sign of the load error) is the record's recommendation for the real run; the aux term, once its bf16 counts were fixed, balanced the dense slots too and stored the biographies more (z 14 at tier 100). Decided 2026-10-21 (the user): loss-free balancing enters the real-run spec in place of the aux term. Not RL: balance has an exact dense signal. |
| run order | Decided 2026-10-23 (the user, after a cost read): R4b's expert arm first (arm E4, no build), builds beside it; R9 next, its read gating R4's corner arm; R6 last (the longest CPU data build). R4 is staged (arm P the baseline cell, the far corner first, five arms at most) and R4b's sublayer half waits on the user's choice about the three-sublayer core. One training run at a time on the card; a paused run resumes on the code in the tree, so builds land behind flags that default to the old behaviour. Decided 2026-10-24 (the user): R4 is dropped, obsolete on R9; R6 is skipped, its mechanism read at micro (arms (a) and (b), arm C at pilot shape); next the A4 flip read routed through bge, then R5 on the chosen shape and the run spec. |
| reasoning post-training | Proposed 2026-10-12 (the user): 6f, in written tokens with the loop as computation per token unless R9 says otherwise. SFT on short and mixed chains, on-policy distillation, then RL past the pass@8 gate with RLTT credit, a depth record and routing replay; rewards on outcomes with visible evidence, abstention (three-way) and evidence conflicts. RL only for decisions without a gradient. R9 read 2026-10-24: composes in written tokens ([written_bridge_pilot.md](../measurements/written_bridge_pilot.md)), so 6f's reasoning form is written tokens and its mix needs multi-hop rows at the serving depth. |
| optimizer | Proposed 2026-10-12 (the user). Decided 2026-10-21 (the user): Muon is dropped from the real-run spec, the pilot-shape Muon arm P is not run, the real run stays on AdamW; R7 is closed (loss and copying passed at micro, the leak was worse by the sign and Muon stored the biographies about twice as early, [muon_micro.md](../measurements/muon_micro.md)). The rest of this row is the proposal as it stood, kept for a revisit. The real run would step the hidden matrices with Muon: Nesterov momentum 0.95, 5 Newton-Schulz steps in bf16, the update scaled by 0.2 sqrt(max(m, n)) to AdamW's RMS so the lr and weight decay carry over, decoupled weight decay kept (Moonlight, arXiv 2502.16982: both are what made Muon scale). The partition is a rule, not a list. On Muon: the hidden 2D projection matrices of attention, MLP, expert and adapter modules (q, k, v, o of every attention; the dense and shared MLPs; the routed experts one matrix at a time, since TE `GroupedLinear` keeps one 2D `weight{i}` per expert; the selector's `down_proj`; the PLE projections). Where gate and up are fused into one `gate_up` weight (the routed experts and the shared MLP; the prelude MLP is already split), rows `[:I]` and `[I:]` are orthogonalized as two matrices. On AdamW: every table (the embedding or tied `E`, the per-layer and router tables, the IR `z_keys` and `y_values`, whose Adam moments `apply_ir_refresh` zeroes by row), every head (the LM head and its sub-heads, the MTP head, the groundedness head, the head adapters), the router, `noise_proj` and `loop_router_bias`, every tensor of ndim 1 or less, and the zero-init neutrality matrices (`loop_inject`, `direct_gate`, IR `g_proj`), which a normalized update would move to full RMS on the first step. Where it does not apply: the pilot arms (a matched pair shares its optimizer) and any graft or finetune of an AdamW-pretrained checkpoint (Moonlight reports its finetuning gain for matched pretrain and finetune optimizers). A Muon trunk is finetuned on Muon, so 6e and 6f then run on Muon. R7 tests today's parameter set; the coda and the tied `E` (absent from the micro and pilot shapes) inherit the rule unverified. The budget in 6b keeps AdamW's numbers: Moonlight's about 2x compute efficiency is measured at compute optimal, and shape L at about 280B tokens was about 600 tokens per parameter at the blueprint's 470M (about 840 at the configured 333M), a regime it does not cover. |
| loops (axiom) | "Passes add computation, not capacity; recalling a weakly stored fact can take more than one pass" (reworded 2026-10-04 from "loops buy computation, not storage") is an axiom, not a measurement: the record is 0.008 to 0.012 nats from loop 2 to 3, confounded by the halt gate. Ouro (arXiv 2510.25741) measures looped and non-looped models at the same ~2 bits per parameter, with the loop gain in knowledge manipulation. Two falsifiers. Storage half: closed-book rank by loop count on arm (a) (depth 1, 2, 3; every exit was trained through `loop_count_sampling`); if the tier 100 and 1000 `delta` grows with depth beyond its paired sigma, later loops carry stored facts and the axiom is wrong for this design. **Read 2026-10-04: the condition is met at tier 100.** In distribution the entity `delta` at depth 1 / 2 / 3 is 0.269 / 0.369 / 0.378 at tier 100 (depth 1 to 2 +0.100, z 19.7; 2 to 3 +0.008, z 6.5; prior unmoved); the gain is on the real name (1.21 nats against 0.14 for the fresh-name prior), and held out nothing grows. At tier 1000 the real name is saturated after one pass (rank 0.0008), and the `delta` growth there (0.393 / 0.453 / 0.467) is the prior drifting toward chance, not recall. Since the block's weights are shared, a second pass adds a step of computation over the same weights, not capacity: recall of a weakly stored fact is a two-step computation here. Decided 2026-10-04: the axiom is reworded as above and the storage-half falsifier is retired as read, since it measured recall depth, not capacity. The rule it leaves: every closed-book leak read is taken at full depth. Not planned, available if the axiom ever has to be quoted as measured: an arm with equal CE weight on every exit, which removes the weaker training of the early exits (a last-loop-only arm cannot be read by depth at all). It was also the control the learned exit gate arm would have needed; that arm was dropped on 2026-10-06 (Decisions, depth allocation), so the question is moot. Computation half (L1): if read sites past the first pass add under 5 points on held-out 2-hop at D = 3, the loop clause of the goal is wrong. Looping stays a requirement either way. **Micro read 2026-10-08 ([chain_depth_micro.md](../measurements/chain_depth_micro.md)): inconclusive, the precondition "readable" fails.** The 35M arm at 105M tokens copies an answer-type entity from the buffer but does not select the right one, so 1-hop accuracy is at chance and so is every L1 cell (held-out Delta +0.006 at sigma 0.013). That the Delta also sits under the falsified bound (Delta + 2 sigma 0.032 against 0.05) does not count: a model that cannot look up one hop cannot show whether later sites compose. The read neither supports nor falsifies the computation half; the axiom stays an axiom, and the pilot's L1 (sublayer sites against passes, at the pilot's shape and dose) is the deciding read. The third pass's `loop_scale` falling under 0.01 by 50M (0.07 at the end) is the weak-later-loop case the rule says to fix, read on a task where no pass composed. **Pilot read 2026-10-11 to 2026-10-17: the computation half is falsified at this shape.** Arm P (99.1M, 860M tokens) passes the 1-hop gate at 0.85 with every composition cell at chance (held-out Delta +0.002 +- 0.005), and the chain continuation ([chain_continuation_pilot.md](../measurements/chain_continuation_pilot.md), one more epoch of the chain split) takes lookup to 0.99 with the loop scales healthy and still composes nothing (Delta +0.004 +- 0.008, composed answers at a uniform pick among candidates). Decided 2026-10-17 (the user): L1 failed at this shape; the loop clause is recorded as falsified here, looping stays a requirement, the goal's reasoning clause is reworded before the real-run spec, and the written bridge (R9) is the next read on whether reasoning composes in written tokens where the loop did not. **Bridge read 2026-10-24 (R9, the 30M save, `pilot_bridge_30M_gen_d3.log`):** written as a bridge in the answer, the same chains compose at one read site (held-out 2-hop 0.989 against chance 0.191, the loop adding +0.004 +- 0.002), so at this shape reasoning composes in written tokens and the loop's measured value is language and the recall of weakly stored facts; R9's verdict is the user's. Decided 2026-10-25 (the user; Decisions, run spec 2026-10-25): the goal's reasoning clause is reworded to reasoning in written tokens with the looped trunk as computation per token, L1 is descriptive in the run spec, and the loop clause is kept as the original goal to revisit after the real run. |
| depth allocation | Dropped 2026-10-06 (the user): the learned exit gate arm (an Ouro-style exit distribution that only reweights the per-pass losses, never built) is dropped on the per-exit read on micro arms (a) and (c) (`eval_exit.py`, 2026-10-05): the second pass's gain (+0.094) is spread evenly over confidence deciles, the third pass adds 0.006 nats, and the entropy-regularized optimum is near uniform even as a bound (0.343 / 0.286 / 0.371 at beta 0.1). Its three open questions (the skip-compute exemption, the equal-weight control arm, the pilot spec without exits) are closed as moot. The depth reading moves to the chain corpus: one micro arm with `--evidence --reader-kv` on chain training splits, read with `eval_chains.py` by depth and by kept read sites (L1 at micro scale); approved 2026-10-06, spec written 2026-10-08 ([chain_depth_micro.md](../measurements/chain_depth_micro.md), ladder step 8). `eval_exit.py` cannot read that arm (it refuses evidence splits); the full-site cells at depth 1, 2 and 3 replace the per-exit read. **Read 2026-10-08: inconclusive, not readable.** The arm never learned 1-hop lookup (every accuracy cell at chance at 50M and final), so the full-site cells by depth (answer CE 0.485 / 0.468 / 0.468 at depth 1 / 2 / 3 on the 4-hop split) say nothing about how depth should be allocated on a task that composes. The fixed draw {3: 0.7, 4: 0.2, 5: 0.1} stands; depth allocation is not reopened. |
| abstention | The groundedness head reads the reader output and the null mass; the preference pass stays deferred until an A-gate has a pilot reading. |
| POC role | Phase 4 is a mechanism check of the reader, read per loop. It decides nothing about externalization. |
| matched compute | SMELT's definition: equal compute per token, equal non-embedding parameters and equal KV-cache size, or the comparison is not quoted as compute-matched. |
| IR key init, `num_ir_experts > 1`, B1 | Moot: the table is dropped. |
| gate names | 2026-10-24: the records from 2026-10-14 to 2026-10-23 called the store recall read (the model's query at recall@5 0.004 against bge 0.829) "A4 fails" and the oracle-buffer pathway read (EM 2.6%) "A7 unreadable". By the gate table the recall read is A7's recall half, the pathway read A7's pathway half, and A4 is the store edit (flip rate), read in the same run with zero correct answers to flip. The records in this file, CLAUDE.md, PILOT.md and the design page are renamed to match; logs, JSON and script names keep the old labels (`ladder_pilot.sh`, `pilot_{p,c}_ladder_*`), and so does [review_2026-10-16.md](../review_2026-10-16.md) (its "A4 fix" is the recall fix, its deletion audit "as the A7 read" replaces the flip rate, so A4). |
| repository review 2026-10-24 | [review_2026-10-24.md](../review_2026-10-24.md). Proposals (decided by the user 2026-10-24): R4's coda corner not run (optional under the 2026-10-23 rule now that the bridge composes at one read site; R4 waits for the real run's size); R4b's sublayer count chosen on cost and the two-sites-per-pass argument and recorded as a cost choice like the expert count, no arms; R6 kept (the only read of pretraining-text facts reaching the weights); the two corpus flaws fixed and the matching rule for arms after P decided before the next pilot-shape arm; two micro retriever arms (the contrastive term at loop 1 with the trunk gradient on; the selection loss weighted per segment), about 1.5 hours each, before the spec closes "no retriever term"; A3 in generation and the A7 pathway and A4 edit reads re-read with the fixed packer before 6f's abstention reward is called the step that must not fail; the goal's reasoning clause read as "reasoning in written tokens, the loop as computation per token". Instrument caveats on record (facts): arm C trained on 12,240 documents, 50.11M tokens, of 4094-token QA rows the evidence arms drop (the evidence reader packs to 4095 and adds two MTP pads), so arm P against arm C is matched on the biographies and the chains, not on the QA slice or the schedule (arm C's cosine reached the 6.0e-5 floor, the evidence arms stopped at 6.55e-5), while P against E4 stays matched; about 7.4% of `inject_val` tokens appear verbatim as supervised text in rows both arms train on and another 6.7% in the rows only arm C trains on, so every held-out CE on `inject_val` (the pilot's language bar, arm PI's stop, the E4 read) leans toward the arm that saw more of it. The rank reads that decide A1 are unaffected. Decided 2026-10-24 (the user): R4's corner dropped as proposed (obsolete on R9); R6 skipped against the proposal to keep it (its mechanism read at micro); the A4 flip re-read gets a bge route (`eval_store.py edit --route bge`). Decided 2026-10-24, evening (the user): the held code changes land and landed the same evening (fp32 CE sums in `mtp.py`; the log-space selection loss with `evidence_selection_weighting` `token` or `segment`; `evidence_contrastive_loops` default `[0]`; the groundedness head input detached under the key/value reader, the evidence-only signal not built; the row estimator replaying each reader's rule and a 4093-token training cap in the evidence builder; the evidence padding in 128-token blocks; the dense-slot GEMM skip; the frozen-gate skip; fp32 selection scores under loss-free); held: the eval-pass rewrite (deferred to 6a) and the two `eval_chains.py` bridge-mode items. The matching rule: the evidence reader keeps dropping the 4094-token rows, so arms after P on the existing builds stay matched to the arms already run (the corpus flaw is kept, not fixed); the `inject_val` flaw gets `inject_val_clean` (built, 0.99% verbatim overlap against 13.3%) and a CE re-read of the three finals. Both leak candidates are built as builder flags with their splits; the two micro retriever arms and the two leak arms run in that order, then the clean-slice re-read, with R5's eval set between them when the card is idle. Decided 2026-10-25 (the user; Decisions, run spec 2026-10-25): the reasoning clause read as proposed, and R4b's sublayer count a cost choice (today's one block per pass, no arms). |
| run spec 2026-10-25 | Decided 2026-10-25 (the user) on the first table of [RUN2.md](RUN2.md), every row on the recommended option: (0) shape L is the configured `config.yaml` shape, width 768, not the blueprint's 1024 (the shape row stays the blueprint on record); (1) 32 routed experts kept, a cost choice; (2) `evidence_selection_weighting: segment` is the real run's default; (3) `--support-gold-only` is carried into the real corpus builder as a data rule, a build item behind a flag; (4) the goal's reasoning clause reworded ("The goal": reasoning in written tokens, the looped trunk as computation per token; the original kept there as the aspiration to revisit after the real run), so L1 is descriptive and the written bridge read is the reasoning gate; (5) today's one block per pass, the three-sublayer core not built; (6) the token target 280B, the application asking 2.5k H100 hours plus the corpus build line plus a 20% buffer (about 3.0k), ceiling 5k; (7) R5 on the local card tonight; (8) a short matched control to 10B tokens, about 30 H100 hours (about 4 hours on an 8-GPU node if DDP scales linearly); (9) the thin tier 100 held-out margin at micro accepted as the recipe's known state on one seed, stated in the caveats, no second seed. |

## Where this stands

Done, with the full records linked:

- **Phase 0**: both learned heads removed, the halt gate's measured pass-through folded into
  `loop_scale` ([phase0_migration.md](../measurements/phase0_migration.md)). `p_max` beat
  `p_correct` a third time.
- **Phase 1**: Stage 0 diagnostics ([stage0_diagnostics.md](../measurements/stage0_diagnostics.md)).
  The IR table stores nothing (entropy 99.5% of max, zeroing the read 0.0002 to 0.0004 nats). Query
  drift after loop 1 is about zero. Loop 3 is redundant, not idle.
- **Phase 1b**: the benchmark suite validated against Pythia-410m on 11 of 11 anchors, four peers
  frozen, slice noise measured, the three-checkpoint snapshot recorded
  ([benchmark_suite.md](../measurements/benchmark_suite.md),
  [noise_floor.md](../measurements/noise_floor.md),
  [benchmark_snapshot.md](../measurements/benchmark_snapshot.md)). The answerability probe reads
  0.584 on all three checkpoints ([answerability_probe.md](../measurements/answerability_probe.md)).
  Mean MC headroom +0.088 / +0.081 / +0.084 against gpt2-medium's +0.193. BoolQ with the passage in
  the prompt reads 0.46 / 0.44 / 0.42, below chance.
- **Phase 2**: abstention repair ([abstention_repair.md](../measurements/abstention_repair.md)).
  The 0.55 arm (the shipped repair): false abstention 0.783 to 0.161, recall 0.81 to 0.22
  (the 0.40 arm reached 0.136 false abstention at recall 0.18); precision pinned at 0.578 six
  readings running. The data lever is exhausted.
- **Phase 3, 3b, 3c**: table reshape and sharpening, the value-scale fix, loop input injection
  ([ir_sharpening.md](../measurements/ir_sharpening.md),
  [ir_scale_fix.md](../measurements/ir_scale_fix.md),
  [loop_injection.md](../measurements/loop_injection.md)). All three gates failed on the same
  number: the read is worth 0.0002 nats whatever the init, width or scale, and the injection made
  later loops marginally more alike. The graft record is 0 for 4.
- **The evidence ceiling** ([evidence_ceiling.md](../measurements/evidence_ceiling.md)): in-context
  gold evidence is worth +3.23 nats on the answer span; a distractor costs 0.63 nats more than
  nothing.
- **Phase 4 build**: port, corpus builder, `--evidence` profile, selection loss, groundedness head,
  `eval_abstention.py --evidence-port`, all built; the 2026-09-18 review's thirteen findings acted
  on; corpus built and seed migrated 2026-09-20; batch and cap retuned. Not trained.
- **The direction review** (2026-09-29): seven audits, five literature sweeps, three judges. It
  found the kill metric, the validation leak, the non-causal gate and the reader's rotary phase, and
  established that the corpus and the old real-run recipe both keep rewarding recall from weights.
  Its loop section found the objective and the migrated `loop_scale` working against depth, and
  merged three loop designs into the one in Decisions.
- **R0b, first from-scratch runs** (2026-10-02 to 05,
  [r0b_micro_pilot.md](../measurements/r0b_micro_pilot.md)): arm (a) reads the instrument; the
  key/value reader copies from the card at 100M where the cross reader does not; arm (c) leaks at
  tier 100 at 100M under both. Arm (b) in full (2026-10-04) stores nothing at 3 sigma, and the loop
  reading on arm (a) shows closed-book recall at tier 100 growing with depth, mostly in the second
  pass. Arm (c) in full (2026-10-05) copies completely in distribution and fails on both forms, on
  a leak present at 50M before copying and flat after 100M. The copy-first warm-up arm stored
  nothing without real-value targets, then stored tier 1000 facts again after the switch; at 300M
  it holds the R0b rule through tier 100 and leaks at tier 1000; R0b was accepted on it on
  2026-10-05; a swap rate 0.30 branch (read 2026-10-06) and a placeholder name arm (read
  2026-10-08, the dose lever falsified) left tier 1000 where it was, and the copy-criterion span
  weight was dropped. The per-exit read finds the second pass's gain spread evenly over tokens; the
  learned exit gate arm was dropped on it (2026-10-06) and the depth reading moved to a chain arm,
  which ran on 2026-10-08 and read inconclusive: the micro model never learned 1-hop lookup, so L1
  stays unmeasured at micro scale.

## Binding measurements

Each measured, each recorded; a retry must move the ablation, not the entropy.

- The learned IR table stores nothing the trunk lacks: zeroing its read costs 0.0002 nats at any
  init, width or scale. Dropped from the real run.
- In-context gold evidence is worth +3.23 nats on the answer span; a distractor costs 0.63 nats more
  than nothing. External evidence is the mechanism, and 3.23 is the ceiling every reader reading
  is a fraction of.
- Every graft onto the converged checkpoint measured zero for four. The real run pretrains with
  retrieval from token 0.
- `p_max` carries no answerability signal; a linear trunk probe reads 0.584 on every checkpoint.
  Abstention precision is pinned at 0.578 by the data lever.
- No per-loop number on record is free of the halt gate: it passed 8% of the loop-3 update for all
  16B pretraining tokens, and every later arm grafted onto that lineage. Only the from-scratch
  pilot can separate "loops cannot reason" from "loops were never allowed to".
- The trunk is already near-empty of facts (TriviaQA 0.014, NQ-open 0.002). The pending run tests
  reader capacity, not fact removal.
- Depth past 3 degrades on plain text: CE 3.4112 / 3.4115 / 3.4213 / 3.4384 / 3.4618 / 3.4900 at
  depth 3 to 8 on `ir_c`, with no evidence attached
  ([loop_scale_probe.md](../measurements/loop_scale_probe.md), "Depth past the trained 3").

"Loops buy computation, not storage" is no longer listed here: nothing measured supports it, so it
is an axiom under Decisions, next to its falsifier. It was reworded on 2026-10-04 to "passes add
computation, not capacity; recalling a weakly stored fact can take more than one pass" after the
loop reading on micro arm (a).

---

## Phase 4: the evidence finetune as a mechanism check

One question: can the port be read at all. It runs the architecture as built, on the grafted
checkpoint, with the gate frozen. It cannot show where facts live: the seed scores near zero
closed-book, and between 54 and 92% of the corpus's supervised tokens (depending on the weighting)
carry content the buffer does not supply.

### Before the launch

**Done 2026-09-30**, all eight: `[eval fixed]` in `sft.py` over `evidence_fixed` with the kill at
10M tokens, `evidence_dev` / `evidence_fixed` from `prepare_evidence_data.py --heldout`, the 2.5M
cadence plus a step-0 eval, `freeze_evidence_gate`, `--reader-no-rotary` (inferred from
`moe.evidence_reader_rotary_off`), one chunk per passage and same-passage exclusion in
`eval_abstention.py`, the seed's value path printed, and the tests (`test_port_backward`,
`test_gate_causality`, `test_evidence_decode_cache`, `test_fresh_param_routing`,
`test_reader_rotary`, `test_prepare_evidence_heldout`). The gate test measured the leak it exists
for: at scale 5 earlier positions move 4.2e-3 against an exact 0, and cached against uncached
decode differs by 8.1e-3.

**Follow-up, 2026-09-30.** The kill checks two numbers (gold gain and content gain, gold minus
distractors) and is armed again when the fixed pass read nothing; `kill_checked` is persisted in
the checkpoint payload. The fixed pass prints a per-loop reader gain from the per-loop CE of the
same forward and a per-token chunk AUROC (chance 0.5) beside the old pooled one. The selection
loss supervises the positions that produce a supervised token (`labels[:, 1:] != -100` shifted to
`:-1`), the positions `answer_start_positions` and the readouts use; the old `labels != -100` was
off by one. `checkpoint_every_tokens: 10000000` puts a save at the decision.

The eight items as specified, in the order of how badly each would have misled the run:

1. **The kill number.** `sft.py`'s per-condition `[eval]` CE uses each row's own target, the answer
   under `gold` and a refusal under `none`, and read minus 3.25 nats on a seed with a dead reader
   (on the old leaked split; `evidence_dev` reads about -2.3). The `none` bucket also holds replay
   rows. The fixed-target pass teacher-forces the real answer under every condition over the
   natively answerable QA rows and prints gain = CE(none) - CE(cond); the kill reads it. The seed
   reads -0.0020 there, and `eval_abstention.py --evidence-port` reads +0.0000 as a cross-check.
2. **The validation split.** `prepare_evidence_data.py` split per rendered row after the QA
   sources repeat, so every QA validation question was also in train (7,440 of 7,440). The
   held-out splits `evidence_dev` and `evidence_fixed` come from SQuAD dev and HotpotQA dev, split
   by source id before the condition draw. The current `evidence_val` is a train-loss slice.
3. **Cadence.** `eval_every_tokens: 2500000`, and an eval before step 1, so the decision at 10M
   tokens has a step-0 baseline plus four readings at 2.5 / 5 / 7.5 / 10M, the fourth being the
   decision. Warmup is 99 optimizer steps, about 4.2M tokens.
4. **The chunk gate.** `chunk_mean_mass` averages the selector's weight over every token of the
   document and that number gates the states every position reads, so training sees future tokens
   and cached, uncached and training gates differ. Freeze `evidence_gate_scale` at zero:
   `requires_grad = False` and out of the fresh group.
5. **Reader rotary.** The query is rotated by its prompt position and the key by its position in
   the chunk; the offset is unrelated to content. Add a flag that passes `position_embeddings=None`
   to the reader. Arm B.
6. **Eval chunking.** The primary G3 reading uses one chunk per passage, as trained; the 128-token
   re-chunking stays as a secondary robustness row. Exclude same-context passages from the eval
   distractor pool.
7. **Seed check.** Print `|g_proj|rms` once. If it is zero, `g_proj` and the zero-born
   `direct_gate` hold each other at zero and the IR value path is dead for the run.
8. **Tests.** Three: backward on a zero-init port (`o_proj` gradient nonzero at step 0, q/k/v zero,
   then nonzero after one step), causality of the gate when its scale is nonzero, cached against
   uncached decode with evidence attached. Plus an assert on the `is_fresh_loop_param` predicate
   and the by-hand gradient clear in `train_step`, which route exactly the tensors this run trains
   (`test_fresh_param_routing`, a GPU test: it builds the model).

### The run

- Arm A: as built, gate frozen. Arm B: same, reader without rotary (`--reader-no-rotary
  --run-name norope`). 10M tokens each, 11 to 19 minutes each at 6.0k to 14.8k tok/s (about 10k
  typical), in the background under a Monitor watch teed to a gitignored log.
- Sign convention, stated once: gain = CE(none) - CE(cond) on the fixed-target answer span;
  positive means the evidence helped.
- Kill: at the first fixed-target eval past 10M tokens the run saves and exits 10 if the gold gain
  or the content gain (gold minus distractors) is under 0.1 nats. The check is armed again when
  the fixed pass read nothing; `kill_checked` is persisted, so a resume does not decide twice.
- Stop by hand after the 10M reading: `touch ckpts/evidence/STOP` (arm A) or
  `touch ckpts/evidence_norope/STOP` (arm B). Polled every 10 micro steps; the run saves and exits
  10. The glob form does not work. Relaunch a killed or stopped arm with the same `-c` and the
  same flags: without `-c` the yaml builds a model without the port and the strict load fails.
- Pass: gold gain at least 1.6 nats and distractors gain at least 0. The printed ceiling fraction
  is against the SQuAD-only 3.23 nats, measured on other rows and another checkpoint, so it is
  indicative until the Phase 4b ceiling on the `evidence_fixed` rows exists.
- Read per loop, all from the same `[eval fixed]` forward: the reader gain lines (final loop equals
  the headline), external mass and mass per chunk per condition, mass/chunk AUROC gold-present vs
  distractors, chunk AUROC (per token, chance 0.5) and gold share. A rising
  `|shared_evidence.o_proj|rms` with a flat gain means the reader learned a bias, not retrieval;
  RMS alone is not the signal.
- Log gradient norms of `key_adapter`, `value_adapter`, `down_proj` and `loop_query_bias` at steps
  10 and 100.
- Under 50% of the ceiling: the next arm is a reader per loop or reads in dense layers, before more
  data. Arm C, only after a pass: the one-hot chunk channel with `gamma = 0` (arm C only; from
  scratch `gamma` starts at 1).

### What it decides

A pass says the grafted reader can carry a span; it does not change the real run's design. A fail
on the graft is weak evidence, given the record, but it changes which reader arm the pilot runs
first (a reader per loop or dense-layer reads). Phase 5 does not wait for it either way.

**Gate G3** (restated): gold gain on the held-out fixed-target rows at least 1.6 nats, 50% of the
ceiling; distractors gain at least 0; benchmarks within noise. **Gate G3b**: chunk AUROC (per
token, chance 0.5) at least 0.59 per loop (3 sigma of 0.030 over 0.5); mass/chunk AUROC and
grounded AUROC (evidence rows) at least 0.674, three sigma over the 0.584 probe (0.65 was 2.2
sigma). `sft.py`'s `[eval fixed]` is the home; `eval_abstention.py --evidence-port` is the
cross-check.

---

## Phase 4b: instruments

Everything the goal is measured with. Most of it is eval-only code and runs on existing
checkpoints; it is on the critical path because the pilot cannot be read without it.

- **R0, the loop scale probe.** `eval_stage0.py` gets a flag that multiplies `loop_scale` at eval.
  Run on `ir_c` and `phase2_final` with factors 1, 2, 3.5 and 5.9. Read the CE gain from loop 2 to
  3. Pass at 0.02 nats with loops 1 and 2 not worse. Decides whether `loop_scale` joins the fresh
  LR group in graft arms. **FAIL 2026-09-30** on both checkpoints, monotone in the multiplier
  ([loop_scale_probe.md](../measurements/loop_scale_probe.md)): it does not join.
- **R1b, the depth memory probe.** One micro step at depth 3 and at depth 5 with the detached
  prefix, pilot shape; read peak memory. Decides whether the depth schedule is trainable. **Run
  2026-10-21: fits** ([depth_memory_probe.md](../measurements/depth_memory_probe.md)): depth 5
  with full gradient OOMs at batch 8 x 4096 and fits at batch 4 (17.3 GB, 1.27x the time per token
  of depth 3), and the detached prefix fits at 21.5 GB but leaves the prelude without a language
  gradient until a straight-through route at the first live pass is built.
- **The in-context ceiling on the `evidence_fixed` rows**, read with the seed, and a source
  sidecar on `evidence_fixed` so every gain splits per source (SQuAD, HotpotQA). The printed
  ceiling fraction is against the SQuAD-only 3.23 nats on other rows and another checkpoint until
  then.
- **A1, the closed-book rank scorer.** Likelihood rank (or MC accuracy) of the gold answer among
  at least 20 same-type candidates (100 in R0b), pathway off, neutral prompt that does not licence
  abstention, head and tail obscurity tiers, bootstrap sigma. PopQA and the tiers are not in
  `eval_benchmarks.py` today: a build item. Plus the fictional biography generator (bioS style) with
  per-fact exposure counts 1 / 10 / 100 / 1000 and the matching store chunks, shared with R0b.
- **A3 in `eval_abstention.py`.** A counterfactual condition following the Faithfulness-QA recipe:
  swap the answer entity in the gold chunk for a same-type entity, and report the memorization
  ratio p_orig / (p_orig + p_swap) (Longpre et al. 2021) and the follow rate on items answered
  correctly with the unswapped chunk, stratified by entity frequency, with n per stratum. Injected
  high-exposure facts are included so conflicts exist at pilot scale. Per-record JSON in evidence
  mode, EM and non-abstain rate under distractors and none.
- **A4, the store edit.** Two stores on the same questions: the gold chunk edited or deleted with
  its near neighbours; re-retrieve with the model's own query, not the oracle buffer; report the
  flip rate on items answered correctly before the edit.
- **A5 readouts.** Built in `sft.py`'s `[eval fixed]` (chunk AUROC per token, mass/chunk AUROC,
  grounded AUROC, per loop). Left for the cross-check: gold-chunk mass per IR expert in
  `eval_abstention.py` (it reads `ir_modules[0]` only today).
- **A7, store-level retrieval.** Held-out NQ, TriviaQA and HotpotQA with the gold passages in the
  store; recall@k of the model's query against bge's; open-book EM through the whole pathway
  against the oracle-buffer EM.
- **The synthetic chain generator and eval** (L1, L2). Fictional entities, hops 1 / 2 / 3, 6 to 14
  distractors of three kinds (same relation with other entities, bridge relation with other
  entities, one full decoy chain), shuffled chunk order, a hop index per gold chunk written to an
  `.evhop` sidecar for evaluation only, 4-hop chains and unseen hop-2 templates held out. Score
  accuracy by hops and by sequential read sites at fixed D = 3, zeroing the reader output at every
  site after site j. The validity rule: every cell with fewer read sites than hops sits at chance,
  1 / (type-matched candidate answers in the buffer).
- **An evidence path in `eval_benchmarks.py`.** SciQ and BoolQ carry their passage; put it through
  the port and diff against the in-prompt score (A6, only where in-prompt is 3 sigma over chance).
  TriviaQA-rc and NQ with DPR gold passages for the attach delta. HotpotQA validation with both
  gold paragraphs in the port, by read site.
- **Per-loop readouts in the training log.** `cos(q_k, q_k+1)`, per-loop selector mass, per-loop
  reader gain, and `||h||` per loop (the readout blind spot from the first review).
- **The oracle head (G9).** Train a dense `768 -> 65536` head on frozen final-loop states for
  100 to 200M tokens; `CE_factored - CE_oracle` is the damage the block-diagonal head does. Tie if
  at least 0.02 nats. Plus the free reading: least-squares fit of `E A^T` to the trained head. G9 is
  the head tie only.
- **The per-layer table zero-ablation on the seed** (the PLE ablation). Zero the per-layer table's
  read and measure CE; near the IR table's 0.0002 nats, the per-layer and router tables become
  projections of `E`; otherwise the per-layer table stays.
- **A paired-bootstrap CE sigma** on the standard slice, so "within noise" is defined for CE.

---

## Phase 5: the from-scratch pilot

The first experiment that can falsify the goal. About 120M parameters at width 512, 1 to 2B
tokens per arm on the 5090, with a matched-token control without retrieval. Run times are
unmeasured; the R1b memory probe ran 2026-10-21 (fits, depth 5 at batch 4 x 2), and the R0b micro-pilot (ladder below) reads the
externalization recipe before any 1B-token arm.

### The model, from scratch

Every item is in the design page's from-scratch blueprint. Grouped by what it touches.

**Token tables and readout.**
- One table `E`, init std `1/sqrt(hidden)`, no weight decay (it is tied, so its input and output
  roles cannot be decayed separately). LM head `norm(h) @ A @ E^T` with `A` hidden x hidden; MTP
  head through a `hidden/2 -> hidden` adapter onto `E^T`, CE on a 25% token subsample. Fused
  linear plus cross-entropy in `_chunked_linear_ce`. The per-layer and router tables become
  `E W_ple` and `E W_moe` if the PLE ablation (Phase 4b) lands near the IR table's 0.0002 nats;
  otherwise the per-layer table stays.
- A 2-layer coda between the loop and the norm, a separate `ModuleList`, zero-init residual
  branch, own KV slots allocated in `KVCache.__init__`, keys in `NEUTRAL_LOOP_KEYS` and
  `is_fresh_loop_param`. MTP reads the coda output. The convergence exit stays off.

**Selector.**
- Always on, outside the router pool. `q = normalize(down_proj(norm(h)) + loop_query_bias[k])`,
  384-d. `z[t, c] = cos(q_t, key_c) / tau` over the candidates plus one learned null key. Keys are
  raw bge; `key_adapter`, `value_adapter`, `g_proj`, `up_proj`, `direct_gate` and the learned table
  are gone. The null mass feeds the groundedness head.
- The selection loss is the union over the passes in the gradient window on the renormalized
  per-pass share. Returned from `forward_step` like the aux loss, so gradient checkpointing can
  stay on.

**Reader.** Superseded in form on 2026-10-03: the measured reader is the key/value read in the shared
self-attention (Decisions, roles and reader positions). The items below carry over to it where they
are about the keys (the one-hot channel, the sink slot, 4 KV heads) and lapse where they are about a
separate module (its own `o_proj`, its own output gate).
- `k'_j = [W_k e_j ; onehot(rank(chunk(j)))]`, `q'_t = [W_q x_t ; sqrt(d) * b_t]`, head dim
  `d + M` rounded to a multiple of 8, `softmax_scale = d^-0.5` passed explicitly, so
  `logit(t, j) = content + b[t, chunk(j)]`. `b[t, c] = gamma_h z[t, c] - lambda log n_c + vis[t, c]`,
  `gamma` per head init 1, `lambda` init 0, `vis` the visibility mask at minus 1e4. One sink slot
  per segment with a learned key and a zero value. No rotary. 4 KV heads. K and V are
  pass-invariant, one cache per buffer. `M` up to 40. Check first that the installed flash-attn
  build accepts the head size.
- Output: after `post_norm`, `h = h + g_loop[k] * RMSNorm(o_proj(read))`, `o_proj` at default
  init, `g_loop` init 1.
- Read sites: sublayers 1 and 3 of every pass, so 2 sequential read sites per pass and 2 hops can
  fit in one pass. Reads in dense prelude layers 3 and 6 are a pilot arm (4 sites before the
  first pass ends).

**Loop.**
- 3 sublayers per pass: self-attention (shared), always-on MLP plus 8 routed experts at top 2, the
  reader in sublayers 1 and 3, `post_norm` and `loop_scale[k]` per sublayer.
- Depth `D` from {3: 0.7, 4: 0.2, 5: 0.1}; the first `D - 3` passes under `no_grad`, detached;
  the last 3 carry gradient; log steps at `D = 3`. `loop_inject` on, because a detached prefix
  otherwise leaves the decoder without main-CE gradient on those steps. `loop_scale` and
  `evidence_loop_scale` length 5, fresh `1/sqrt(5)`. The loop code clamped to the trained depth.
- `loop_ce_weights` are gone; the loss is `CE(head(coda(h_D)))`. The control arm keeps per-loop
  CE and `loop_count_sampling: 0.3`. Note that `loop_ce_weights: [0, 0, 1]` alone is a trap:
  `loop_ce_weights_for` returns the zeros unrescaled, so 30% of steps carry no CE.

**Encoder.** The first 4 prelude layers, chunk-causal, TE checkpoint, encoded once per forward.

**Objective.** `L = CE x span_weight + 0.1 MTP + 0.01 aux + 0.1 L_sel + 0.1 contrastive + 0.1
groundedness` (the aux term leaves if R8 passes), with the MTP targets span-weighted like the main CE. The control arm swaps the first
term for per-loop CE.

**Optimizer (pilot).** Its own spec, in tokens: warmup 2 to 5% of the pilot's tokens (the design's
earlier "1000 steps" is 524M tokens at the pretrain batch), the router noise anneal and the
checkpoint cadence scaled to the pilot's budget, LR from a width-512 sweep, cosine to 0.1 lr, no
weight decay on `E` or the loop scales.

### The pilot corpus (a new builder mode)

Retrieval is attached to every slice. The only evidence-free case is the abstention condition.

| slice | tokens | sources | evidence | supervision |
|---|---|---|---|---|
| web and edu text | 45% | the pretrain mix, fineweb-edu weighted up | top 2 chunks per 256-token window, staircase visibility | span weights, counterfactual swaps, tail anonymization |
| code and math | 10% | stackv2 edu, Nemotron math | same | plain CE |
| evidence QA | 15% | SQuAD v2, HotpotQA, NQ, TriviaQA with passages, MuSiQue, 2Wiki; at most 2 passes each | gold, mixed, distractors, none, counterfactual, partial hop; buffer capped at 8 chunks, so no `many` | answer or abstain |
| synthetic chains | 10% | generated | hops 1 / 2 / 3 at 25 / 50 / 25 | answer |
| chat replay | 20% | smoltalk2, instruction and format rows | same per-window retrieval | assistant text, span weights |

- **Windows.** Each 256-token window is queried with the bge embedding of the preceding window's
  text, offline. Top 2 from the store, the document itself excluded. Window 0 is queried from its
  first 64 tokens and readable from token 64. Tokens in window w see the chunks retrieved for
  windows up to w; the mask lives in `vis[t, c]`.
- **Near-duplicate filter.** Drop a candidate whose longest common token run with the document is
  32 or more, or whose 8-gram Jaccard is 0.3 or more. Hash sets per document; the cost is
  negligible. Report the language-model gain by overlap tier, because a paraphrased mirror passes
  both thresholds.
- **Span weights.** Entities, numbers and dates are tagged (numbers need two tokens or a matching
  context n-gram). A span is supported when it occurs in a visible chunk: weight 1, chunk flagged
  gold. Unsupported: weight 0. Language tokens: weight 1. The MTP targets get the same span
  weight, or the MTP head relearns what the main loss masks. The tagger covers entity-valued
  relations only; attribute nouns (occupation, nationality, genre) stay in the loss, so the A1
  claim is restricted to entity-valued relations unless the tagger is extended to them.
- **Counterfactual swaps.** On 15% of supported spans the entity is swapped for a same-type entity
  in every visible chunk that holds it, in the target, and in every later mention in the document.
- **Tail anonymization.** In 50% of documents, unsupported entities below the head-frequency
  threshold get a typed placeholder in input and target, re-drawn per epoch. The 15% and 50% rates
  are first guesses. The micro-pilot (R0b) dropped the anonymization sweep (2026-10-04: it barely
  reaches the supervised spans) and set the swap rate sweep aside on 2026-10-05, then reopened it
  as a candidate the same day, after the copy-first warm-up arm stored tier 1000 facts again once
  real values were supervised with copying already in place. At 300M the warm-up arm holds the
  rule through tier 100 and leaks at tier 1000; a swap rate 0.30 second phase branched from its
  100M save (read 2026-10-06) found no large effect of the swap rate on the remaining level
  (tier 1000 0.083 against 0.085), so the pilot's swap rate is chosen on copy quality and tier 100
  margin.
- **Copy-first warm-up** (decided 2026-10-05, a requirement). Fact spans are not supervised with
  real values before the reader copies: the pilot opens with a warm-up in which every supported
  span carries a substitute that its visible chunk also carries, and switches to real values once
  the reader copies (on R0b the switch was at 100M with gold top-1 0.95 to 0.97). This is a new
  element of the pilot's schedule; how long the warm-up runs at pilot scale, and whether a copy
  criterion replaces a fixed length, is part of the pilot spec.
- **QA.** Distractors half bge hard negatives, half random. Partial hop: a HotpotQA row with one
  supporting paragraph withheld, target abstains. Hop labels from MuSiQue and 2Wiki are for
  evaluation only. A wider abstention phrasing set.
- **Store.** Training corpus plus Wikipedia, at least 0.4B tokens, 128-token sentence-aligned
  chunks, MinHash deduplicated, raw bge keys. Closed-book probe answers held out of the pretraining
  text. No cached encoder states (600 GB at 0.4B tokens).
- **Build metrics.** Share of tagged spans supported by a visible chunk (under about 20% fails
  the build); gold recall of bge top 2 on the QA slices; near-duplicate drop rate per source;
  evidence ratio and weighted gradient share per slice; the contamination check.

### The ladder

| rung | tokens | runs | read | pass | decides |
|---|---|---|---|---|---|
| R0 | 0 | loop scale probe on existing checkpoints | CE gain loop 2 to 3 | 0.02 nats, loops 1 and 2 not worse | `loop_scale` in the fresh group for grafts. **FAIL**: it stays at the trunk's rate |
| R1 | 10M | the Phase 4 run | chunk AUROC (per token) and reader gain per loop | descriptive: read, not decided (a 0.02 bar is under one sigma of the 0.030 floor, and the POC objective gives every loop the same target); a decided form needs a 0.09 bar | nothing; describes whether reads differ by loop. **Read 2026-10-01**: both arms killed on the content gain (0.058 / 0.081); chunk AUROC 0.650 / 0.643 / 0.644 by loop and the per-loop content gain 0.070 / 0.078 / 0.081 (arm B), so loop 1 reads best and later loops add little |
| R0b | 3 x 0.3B | externalization micro-pilot, 35M params, about 1.5 5090 hours per arm: (a) full CE, no retrieval; (b) span weights with facts masked, no retrieval (the input-side leak test); (c) span weights plus swaps plus anonymization, with store retrieval through the key/value reader. Fictional biographies injected at 1 / 10 / 100 / 1000 exposures; the anonymization sweep dropped 2026-10-04; the swap rate sweep set aside 2026-10-05 for a copy-first warm-up arm of (c), reopened as a candidate by the warm-up arm's 150M read | closed-book likelihood rank among 100 same-type candidates, by exposure, against a fresh-name prior; on every (c) save also gold, swapped and prompt reads | (b) and (c) within 3 sigma of their prior through tier 100 while (a) is 3 sigma above at tier 100 or 1000; tier 1000 in distribution read alongside: a (b) or (c) arm 3 sigma off its prior there is a leak even when tier 100 holds, because the 95% filler weakens tier-100 storage (Physics 3.3, arXiv 2404.05405), so tier 1000 is where (a) stores reliably and a leak shows first. **(a) read 2026-10-02**: 0.378 (z 51) at tier 100, 0.467 at tier 1000. **(c) at 100M**: 0.026 (z 5.2) at tier 100, 0.149 (z 12.3) at tier 1000 with the key/value reader. **(b) read 2026-10-04**: 0.002 (z 0.6) at tier 100, 0.011 (z 1.7) at tier 1000, within its prior on both forms. **(c) read 2026-10-05: fails** on both forms. In distribution 0.027 (z 4.1) at tier 100 and 0.120 (z 8.8) at tier 1000, held out 0.013 (z 3.6) and 0.071 (z 8.2), while the reader copies (top-1 1.000, swapped card followed on every item); the leak is present at 50M before copying (0.010, z 4.0; 0.105, z 14.4) and flat after 100M (paired +0.001, z 0.2 at tier 100). **Copy-first warm-up arm, read 2026-10-05**: nothing stored at 50M and 100M without real-value targets (every cell within 3 sigma), copying at 100M (top-1 0.95 to 0.97); at 300M 0.015 (z 2.6) at tier 100 and 0.085 (z 6.4) at tier 1000 in distribution, held out 0.007 (z 2.0) and 0.047 (z 5.1): the rule holds through tier 100 on both forms, tier 1000 is a leak on both. **Accepted 2026-10-05 on the warm-up arm**, the tier 1000 leak recorded as known; the swap rate 0.30 branch (read 2026-10-06) leaves tier 1000 at 0.083 (z 6.2), paired -0.002 (z -0.3) against the warm-up arm, and holds the rule with more margin (tier 100 0.010, z 1.7; 0.003, z 0.9); the pilot's swap rate stays at 0.15 (decided 2026-10-06). **Follow-ups built 2026-10-24**: the review's two leak candidates (value-first templates supervising the name from the value; gold-absent documents counting a value supported on a distractor collision) as the builder flags `--mask-first-name-mention` and `--support-gold-only`, read by two copy-first warm-up arms, `m1` and `g1` (`ckpts/inject/launch_leak_arm.sh`), queued after the micro retriever arms | if (b) or (c) climb with exposure like (a), the recipe fails before any 1B-token spend |
| R1b | 0 | one micro step at depth 3 and at depth 5 with the prefix, pilot shape | peak memory | fits | whether the depth schedule is trainable. **Run 2026-10-21** ([depth_memory_probe.md](../measurements/depth_memory_probe.md)): fits; depth 5 full gradient needs batch 4 x 2 at pilot shape (about 31 GB at batch 8, 17.3 at batch 4, 1.27x the time per token); the detached prefix fits (21.5 GB) but cuts the prelude off from the language loss until a straight-through route exists. |
| R2 | 0 | synthetic chains scored on the R1 checkpoint | accuracy by hops and read sites (loops, on the POC) | after the 1-hop curve saturates: chance wherever read sites are fewer than hops | whether the instrument is valid |
| R3 | 2 x 30M | graft: proposed objective against the current recipe, 15% synthetic, `loop_scale` kept at the migrated values and the trunk's rate | 2-hop accuracy at 3 loops minus 1 loop | +0.15 and proposed beats current | a pass helps; a null decides nothing |
| R4 | 4 x 2 x 1B | from scratch: loss placement by coda, 2 x 2, two seeds per compared arm (training-seed sigma is unmeasured), shared fresh `loop_scale`, inject and data | held-out-template 2-hop at D = 3 | best arm 10 points over per-loop loss without coda, outside the seed spread; 3-hop above chance | objective and coda. **Staged 2026-10-23 (the user): arm P is the per-pass-loss, no-coda cell, accepted as the baseline though run under the aux term; the far corner (last-pass loss through the coda, a half-day coda build behind a flag plus review) runs first as one arm and only if R9 says the loop must compose; a chance corner makes the table a null, and only a positive corner earns the two single-switch arms and second seeds of the corner and the baseline (five arms in all).** **2026-10-24: the bridge composes at one read site (R9's 30M read), so the corner is optional under that rule; the review recommends not running it (Decisions, repository review); the user decides.** **R9's final read (2026-10-24, composition lives in written tokens) leaves the corner unneeded for the real run: recommended dropped, open for the user.** **Dropped 2026-10-24 (the user), obsolete on R9: the model composes at 0.997 on held-out compositions once the intermediate entity is written out, so the loop no longer has to compose in silence; the coda stays unbuilt.** |
| R4b | 2 x 1B | sublayers per pass 2 against 3 at matched compute; experts 4 against 8 | L1 plus closed-book rank plus language benchmarks | see gates | shape L's core. **2026-10-23: the expert half runs first as arm E4 (4 experts, loss-free, against arm P, launched 17:09); the sublayer half needs the unbuilt three-sublayer core (a day of architecture work plus review), and whether to build it or read one three-sublayer arm against today's one block is open for the user (decided 2026-10-25: one block per pass, not built).** **Arm E4 read 2026-10-24 ([expert_count_pilot.md](../measurements/expert_count_pilot.md)): 4 experts cost nothing measurable against arm P's 8 at the same active compute (held-out CE 3.2396 against 3.2321, under the noise; copying and the port equal; 1-hop 0.875 against 0.848, composition at chance on both; the leak equal at tier 100 and smaller at tier 1000, paired raw rank z -3.8), with two changes (loss-free) and one seed each on record; the expert count is a cost choice for shape L, not a quality one, until a read at the real run's size; the sublayer half stays open.** |
| R5 | 0 | the winner, eval only | held-out 4-hop by read sites and at D = 4; HotpotQA bridge by read sites; A1 to A7 | L2 and the A gates | whether the design enters the real run. **Read 2026-10-25 ([r5_bridge_ladder.md](../measurements/r5_bridge_ladder.md)): arm P's final already carried A1 to A7 (the decode reads on the fixed packer, the A4 flip on the bge route) and its 4-hop reads by read sites at depths 3 and 4, at chance at every save; the bridge final its 4-hop generate read, 0.206 against 0.210 (two steps written); the HotpotQA bridge by read sites has no instrument, a build item (RUN2.md). New: the A-gate ladder on the bridge final, the form 6f builds on: closed book at chance in every cell (the tier 1000 residual gone), copying halved (held-out gold top-1 0.40 to 0.47 against 0.95), SQuAD EM 0.000 with no abstention, pathway EM 0.000 on every route, nothing to flip on A4, clean-slice CE 9.06 against 3.27 at exit 3, the suite's headroom +0.019 against +0.062. The main thread's verdict: a demonstration of written composition, not a usable checkpoint; the design enters the real run as decided 2026-10-25 (RUN2.md), with 6f's mix rule and every A gate read on the shipped checkpoint.** |
| R6 | 1 to 2B | hop 1 seen only in pretraining text (about 100 exposures), hop-1 documents excluded from the store, hop 2 retrieved | closed-book hop-1 accuracy and 2-hop accuracy | at chance on both (the A1 leak probe: above chance means pretraining-text facts reached the weights) | whether the recipe keeps pretraining-text facts out of the weights. **Skipped 2026-10-24 (the user): the mechanism is already read at micro (arm (a) stores plain-text facts under the normal loss, tier 100 `delta` 0.378; arm (b), fact spans masked, stores nothing; arm C repeats arm (a) at pilot shape), so the run spec uses that lever (the spans to keep out of the weights are masked, or plain-text facts are accepted into them) in place of a new arm.** |
| R7 | 3 x 0.3B, then 0.9B | Muon on the hidden matrices (Decisions, optimizer) against AdamW, at the AdamW arm's lr and weight decay through the RMS-matched update, no sweep. Build first: one combined Muon plus AdamW optimizer class over the fp32 masters (`train_step` takes one optimizer and one `LambdaLR`); `eval_exit.py --pair OTHER.json` (paired per-token CE against a twin; today it keeps aggregates only); a sampled max of q·k per layer (flash varlen exposes no logits). **Built 2026-10-18**: `modules/model/optim.py` (`MuonAdamW`, the name rule `muon_kind`, `build_optimizer`; `tests/test_muon_optimizer.py`), `training: optimizer: adamw | muon` read by both trainers, `eval_exit.py --pair TWIN_CKPT` (the twin read on the same tokens in one process, `pair_stats`); the sampled q·k max is not built. A detached Sonnet review the same day found two bugs, fixed: the momentum buffer came back bf16 on a pretraining resume (now kept fp32), and three zero-init tensors were misrouted to Muon (`moe.inject`, the groundedness head, the cross reader's `o_proj`); a resume across optimizer kinds is now refused by name (`check_resume_kind`). Open risk from the review: no foreach batching, one Newton-Schulz per expert matrix, so the step time at the pilot shape is measured before the pilot-shape arm. Arms: `ckpts/inject/launch_muon.sh` (`full`, `cf`, `reseed`; `config_micro_muon.yaml`, `config_micro_seed4321.yaml`), each read by `closed_book_rank` at 50M, 100M and the final and by the paired exit read at every save against the twin's save at the same tokens. Micro, from `seed_micro.pt` on the R0b splits: Muon arm (a) full CE against arm (a), the Muon copy-first warm-up arm of (c) against the warm-up arm (same seed, data and switch at 100M), and one AdamW reseed of arm (a) for the run-to-run spread (the paired sigma is per item, not per run); about 1.5 5090 hours each. **Run and read 2026-10-18** ([muon_micro.md](../measurements/muon_micro.md)): loss bar passes (0.07 to 0.15 nats below the twin, z 16 to 25 paired, 2 to 4x the seed spread), copying no later (both warm-up arms switch between 50M and 100M; held-out copying 0.90 against 0.67), the leak passes by the letter (largest cell z 2.6, tier 1000 in distribution) and is worse by the sign in distribution at every tier above 1 (1.6x at tier 1000, 2.4x at tier 100; held out equal); arm (a) stores about twice as early under Muon, far outside the seed spread; the logit bound unread. **Decided 2026-10-21 (the user): R7 is closed, the pilot-shape arm is not run and Muon is dropped from the real-run spec, which stays on AdamW.** Then, only if both micro reads pass, one Muon arm P from `seed_pilot_grounded.pt` on the pilot splits against arm P, about one night | filler CE on `inject_val` at 50M steps (`eval_exit.py`, paired against the twin); gold top-1 at 50M and 100M (when copying emerges); closed-book entity `delta` by tier on both forms, `compare` against the twin; arm (a) by tier as the storage read (descriptive: whether Muon stores the tail more, arXiv 2509.26030); max attention logit per layer (no QK norm in the trunk); on the pilot shape also the 1-hop gate, L1 and A2 | Muon filler CE below its twin's at matched tokens beyond the paired sigma and beyond the AdamW reseed spread; copying no later (the warm-up's switch criterion met by 100M at micro, by 300M at pilot shape); the (c) and P `delta` not above the twin's by 3 sigma paired at tier 100 or tier 1000 on either form; the logit not growing without bound | the real run's optimizer. A loss gain with a worse leak fails: the goal outranks the speed |
| R8 | 0.3B | loss-free balancing (arXiv 2408.15664: a bias added to the routing score for top-k selection only, moved after each optimizer step by u times the sign of the load error, u 0.001) against the aux loss; micro arm (a) recipe from `seed_micro.pt`, paired against arm (a). Build: the bias per pass, `[n_loops, E]`, since routing is loop-conditioned; over the MLP slots only (the attention and IR experts run densely anyway, and shape L routes MLP experts only); updated on device from the router's hard counts once per sync step; the buffer on `load_model_state`'s allow-list; expert load logged per pass (today the tracker pools passes into one EMA). **Built 2026-10-19**: `training: moe_balancing: aux | loss_free` with `moe_balance_bias_rate`, the buffer `moe.balance_bias` added to the routing probabilities for the selection only (weights unbiased, zero exact, absent buffer loads as zero), `update_balance_bias` on an fp32 shadow after each real optimizer step with the signs centred over the MLP slots (a detached review the same day found the raw signs drift every MLP bias up against the dense slots under a skewed load; the first launch was stopped at 3M and relaunched on the centred update), counts across micro steps, skipped passes untouched, `load max/mean` and `dense share` per pass in the step line, the strict resume paths tolerating an absent buffer, `tests/test_loss_free_balancing.py`; arm `ckpts/inject/launch_lossfree.sh` (`config_micro_lossfree.yaml`, `ckpts/evidence_lossfree_full`, reads `lossfree_full_{TAG}_*`) launched the same day against `ckpts/evidence_inject_full`. **Run and read 2026-10-19** ([lossfree_micro.md](../measurements/lossfree_micro.md)): loss passes (CE 3.9055 against 3.9037, paired -0.0055 at z -1.8, inside the seed spread), storage unchanged within the two seeds, MLP load max over mean 1.03 to 1.05; the load comparison unread (the logged aux value saturates in bf16 and reads the batch, so the aux side was a floor, not a balancer); aux count fix 2026-10-20 and the arm (a) rerun under the fixed term **run and read 2026-10-20**: R8 passes for loss-free on all three bars (the working aux term ends 0.0105 above arm (a), puts 0.35 to 0.54 of the picks on the dense slots with the MLP load at 1.08 to 1.26, and stores more in distribution, tier 100 z 14.5). **Decided 2026-10-21 (the user): loss-free balancing enters the real-run spec in place of the aux term.** | filler CE on `inject_val`, paired; expert load per pass (max over mean); closed-book entity `delta` by tier | CE not worse beyond the paired sigma and the AdamW seed spread (0.03 at the end, from R7's reseed); load at least as even | the real run's balancing |
| R9 | about 500k questions, two passes | from arm P's final save (`ckpts/evidence_pilot_p/checkpoint_evidence_final.pt`): SFT on a written-bridge form of the chain splits (the first hop's answer written before the final answer; `.chains.jsonl` already holds start, bridge and answer, so the build is a template flag in `prepare_chain_data.py`). Budgeted in questions, not tokens: at the chain slice's evidence ratio 4.20 the rate is about 3k tok/s, so 0.1B tokens would take 7 to 10 hours. Read with a generate-then-score mode added to `eval_chains.py`, which today scores candidates right after the prompt and cannot read a written bridge. A short RL run on the chain answer only after 6f's rollout prerequisites exist | held-out 2-hop and 3-hop accuracy with the written bridge against arm P's latent read (at chance); the 1-hop gate kept | held-out 2-hop 3 sigma above chance | whether 6f reasons in written tokens or in the loop. **Built 2026-10-23: `prepare_chain_data.py --bridge` and `eval_chains.py --generate-bridge`; run 2026-10-24, two epochs of a 500k-question bridge split from arm P's final (`launch_pilot_bridge.sh`, 66.35M tokens against a plan of about 50M), read save by save; its read gates R4's corner arm. The 30M read met the bar: held-out 2-hop 0.989 at one read site (chance 0.191), 1-hop 0.998, the loop adding +0.004 +- 0.002.** **Read 2026-10-24 ([written_bridge_pilot.md](../measurements/written_bridge_pilot.md)), on the fixed eval packer: passes by a wide margin; held-out 2-hop 0.982 at the first save (10M) and 0.997 at the final (z 64.7), 3-hop 0.997, read sites past the first adding +0.002 +- 0.001, 4-hop at chance since the model writes only the two steps the mix trained; the model composes as soon as the intermediate entity is a token it writes and reads back, which the loop never did in silence at this size, so 6f reasons in written tokens.** |

A 1-hop curve must saturate before any null is read. R4 replaces the previous plan's per-loop CE
test, which changed loss placement, depth sampling and the coda in one arm. Pilot arms that do not
fit the budget are dropped from the bottom of this table, not the top; R0b runs before all of them.
R7 to R9 (added 2026-10-12) sit outside that rule: they decide the run spec and 6f, not the
pilot's claims, and cost hours, not 1B-token arms. They queue behind arm C and its reads.
Outside the table, arm PI (launched 2026-10-21) is arm P's recipe with the retriever's InfoNCE
term at 0.25 under loss-free balancing, read against arm P's final on held-out CE, store recall
at k5, the leak and copying, before the term enters the real-run spec; stopped at 87M the same
day on the language gap.

### Later, after the pilot passes

- Reader-attention distilled into the selector where no gold label exists.
- The append-only buffer and per-pass re-query at serving, with a staged two-hop eval. Untrained
  today, so deferred.
- A learned bridge-entity span as a small auxiliary, as an arm.

---

## Phase 6: the real run

### 6a. Throughput first

Build plan with estimates, 2026-10-25, re-cut 2026-10-26: 6a throughput is optional and not on the launch path (the launch path is about 3 days of build across parallel lanes, 2 days wall clock, `docs/plans/RUN2.md` table A; the 6a items are in its table B, each non-bit-identical item with a 40 minute micro A/B against its twin; every efficiency percentage is an estimate until a profile at shape L; the token target, hours and ask do not change on them).

Decided 2026-10-21 (the user): this work starts only after every other spec decision is made, so
nothing optimized has to be redone; it is the last step before the run, not the first.

The 16B run reached about 11% MFU on an H100. The step is overhead-bound, not GEMM-bound: routing
with host syncs through `m_splits` every loop, several attention passes per loop, a chunked and
checkpointed head at 4x, no `torch.compile`, no CUDA graphs. In order: profile the step;
`torch.compile` the dense decoder and the shared MLP; remove the per-loop host syncs; a bigger
micro-batch on the 80 GB card; the fused linear plus cross-entropy head and the subsampled MTP
head; then FP8 through Transformer Engine, validated by a 2B-token A/B within about 1% of BF16.

Caveats and items found 2026-10-25 (estimates until the profile): `torch.compile` needs the evidence width `S_ev` bucketed, since it changes every batch; evidence padding to the batch max runs through all the encoder layers (5 to 15% estimated); prompt fill is 88% (8 to 10%); the head's chunk-checkpoint recompute costs 2 to 3% and is bit-identical to skip when memory fits; the step has 7 to 8 host syncs per micro step, not one; the eval pass must be rank-sharded for multi-GPU; and the FLOP line counts the dead IR inner attention, so the 685M per token is high by about 28M.

**Gate G8**: at least 2x the sustained tokens per second that shape L reaches at today's MFU
(about 30 to 32k/s with evidence was the blueprint-width rate behind the 6b estimate; at the configured width 768 the rate is about 63k/s, so the bar is stated as 2x the first-hour rate; an estimate until measured),
defined in tokens per second at shape L, measured over an hour with checkpointing and upload live;
FP8 A/B loss-matched. "Equivalent shape" is not a definition and is not used.

### 6b. Budget

Blueprint, superseded 2026-10-25 by RUN2.md Cost (shape L is the configured width 768, token target 280B, the application about 3.0k H100 hours). Shape L costs about 1.3 GFLOP per token forward at 3 passes: body 2 x (109 + 3 x 98 + 27)M =
860M, tied head 134M, MTP at 25% x 2 tokens 67M, attention 160M, reader sites 80M; the expected
depth of 3.4 passes adds to the body. Evidence multiplies it by about 1.29 at the slice-mix ratio
of 1.55 (0.75 x 1.0 + 0.15 x 2 + 0.10 x 5, QA buffer capped). At today's MFU, 2.5k H100 hours buy
about 270 to 290B tokens; shape M buys about 235B per 1k H100 hours.

The data cap is 400B tokens (100B unique at 4 epochs). Passing G8 at 2x doubles the tokens an
hour buys, which makes a 2.5k-hour run data-bound at that cap, so the hours and the token target
cannot both hold. The run spec fixes a token target and derives the hours from the measured rate.
Every figure here is an estimate until the constructor prints it; recompute after 6a, because
the FLOP line is the anchor and it goes stale silently.

### 6c. The corpus at 100B tokens

The pilot builder mode at scale. Offline cost estimates at 100B tokens: the fact tagger about
15 H100 hours; bge window queries about 5; about 400M ANN queries against the store, CPU hours with
a flat index on GPU; the near-duplicate filter negligible; store deduplication minutes. Corpus
prep is its own interruption-safe job with its own budget line in the run spec. Decided 2026-10-21
(the user): the build runs on the rented cluster, not locally.

### 6d. The run spec (`docs/plans/RUN2.md`)

Written before anything is rented. It names, per component, the pilot rung that admitted it and
its margin. Shape L from the pilot's R4b reading. A token target, with the hours derived from it.
The optimizer is AdamW (Muon dropped 2026-10-21 on R7) and the balancing is loss-free in place
of the 0.01 aux term (decided 2026-10-21 on R8); the LR transfer sweep at reduced width runs on
AdamW, or muP if it can be adopted cheaply. If Muon is ever revisited: DDP repeats the
Newton-Schulz steps on every rank, a sharded optimizer needs the distributed form (Moonlight), and
a growing attention logit gets QK-clip (MuonClip, Kimi K2, arXiv 2507.20534). The gate table A1 to
A7 and L1, L2, with the seed noise measured. The
two-phase curriculum with a reasoning-weighted tail. The per-loop and per-pass readouts in the log
from step 0. The early signals for a run whose kill rules no longer have zero-init tensors to
watch: the selector's gold mass per pass, the reader gain per pass, the supported-span share of the
loss, and A3 at every checkpoint. Drafted 2026-10-25 and decided the same night (the user, every row of
its first table on the recommended option; Decisions, run spec 2026-10-25): shape L is the
configured width 768 with 32 experts and one block per pass, not an R4b reading.

### 6e. After the main run

- **Context extension**: 8k or 16k on long documents with retrieval attached, rotary base raised or
  YaRN-scaled, a few billion tokens. A context change is also a corpus rebuild, because packing,
  fill and the per-row evidence cap all change with the row length. Sliding-window attention in
  some prelude layers only if this phase shows the cost matters.
- **Consolidation SFT** on the real trunk with the evidence segment in the template, then the
  preference pass on abstention pairs mined from the model's own samples, if the groundedness
  signal did not bend the abstention curve on its own. Gate G7 as before: benchmarks within noise,
  false abstention under 10% with recall at least 0.5 and precision at least 0.65, A-gates
  re-confirmed on the shipped checkpoint. If 6f's RL stage runs, its three-way abstention reward
  replaces the preference pass; the pass stays the fallback.

### 6f. Reasoning post-training

Proposed 2026-10-12 (the user): a reasoning model on the real trunk after 6e. The rule for what
gets RL: only decisions without a gradient (discrete choices, outcomes over a whole sequence, real
compute savings). Anything with a dense differentiable loss keeps that loss, because a reward is a
noisier estimate of the same signal. Reasoning is planned in written tokens with the loop as
computation per token (R9 read 2026-10-24: composition holds in written tokens, held-out 2-hop 0.997); L1 read falsified at pilot shape, so a
latent-only reasoning model is not planned.

**R5, 2026-10-25 ([r5_bridge_ladder.md](../measurements/r5_bridge_ladder.md)): the mix and the
reads.** The A-gate ladder on the written-bridge final (arm P's final plus 66M tokens of chain-only
bridge SFT at lr 3e-4, the checkpoint form this section builds on) kept composition in written
tokens (0.997 held out) and lost every other gate: the weakly stored biographies (the tier 1000
residual with them), half the held-out copying, answering and abstention on real questions (SQuAD
EM 0.000, no abstention under any condition; the model writes the bridge template with invented
names) and plain language (clean-slice CE 3.27 to 9.06, the later passes now costing instead of
paying; the suite's headroom +0.019 against +0.062). Consequences, the main thread's verdict,
written into RUN2.md: step 1's mix carries the pretraining slices beside the bridge rows (evidence
QA, the abstention rows, the biographies at their tier structure, the plain-answer form), at a
lower LR than the pilot's continuation (which re-warmed to 3e-4 a trunk whose cosine had ended at
about 6.5e-5); the A gates and the clean-slice CE are read at every save of every stage, with a
kill rule on them; every A gate in the spec is taken on the shipped checkpoint, after 6e and after
6f; the leak reads after post-training are the binding ones (a short post-training removed the
pilot's tier 1000 residual, read as forgetting under a single-format SFT, not as a lever); the
HotpotQA bridge instrument (the bridge decode over retrieved real chunks) and the 6f mix builder
are build items (RUN2.md, Build list 9 and 10). How much of which slice keeps the gates is
unmeasured.

1. **Reasoning SFT.** Short and mixed-length chains of thought, not long R1-style traces (models
   of 3B and under learn short and mixed chains better, arXiv 2502.12143), plus retrieval-grounded
   multi-hop chains with the evidence in the port (HotpotQA, MuSiQue, 2Wiki, the synthetic
   chains). Built by the evidence builders, so every item carries `.ans` and `.evgold`; off-the-shelf
   QA sets without those labels are not used for any stage below. MuSiQue and 2Wiki are new builder
   sources (none exists today). "Supporting chunk visible" is `.evgold` for QA and every chunk with
   `.evhop` above 0 for chains; `merge_evidence_splits.py` carries only `.evgold`, `.cond` and
   `.ans` today, so it is extended to `.evhop` and `.factspan` before any merged RL split. Folded
   into 6e's consolidation SFT where the template allows.
2. **On-policy distillation.** The student samples, a teacher scores every sampled token, the loss
   is the per-token reverse KL (Thinking Machines, October 2025: about a tenth of RL's compute for
   the same score in their report). The signal is dense, so it works where the pass rate is near
   zero, which RL cannot. The teacher must score the student's tokens, and that is the open item
   of this stage. A same-tokenizer teacher is DeepSeek V3 or V4 class: `id_remap.json` (written by
   `prune_vocab.py`) maps its ids onto the pruned 65,536, but the model cannot run locally, an API
   would have to return logprobs on given tokens, and a pruned tokenization is not the canonical
   one in the full vocabulary. The default is therefore a cross-tokenizer variant with a smaller
   self-hosted reasoning teacher, at extra cost; the teacher is chosen in the run spec, and without
   one this stage is skipped and RL starts from SFT. Exit gate: pass@8 at about 15% or more on the target tasks (the
   Parked gate).
3. **RL**, only past that gate. GRPO family with DAPO's fixes (decoupled clip bounds with a higher
   upper one, dynamic sampling that drops prompts solved always or never, a token-level loss) and
   Dr. GRPO's removal of the length and standard-deviation normalization; a KL term with periodic
   reference resets for a long run (ProRL). Three requirements of the looped MoE:
   - RLTT trajectory credit over the passes (arXiv 2602.10520), not credit to the final state only;
     Ouro reports RL after its reasoning SFT brought no significant gain, and RLTT names
     final-state credit as the cause. RLTT needs a trained readout at every pass. The blueprint's
     objective (one CE on the last pass through the coda, the first D minus 3 passes under
     `no_grad`) leaves the per-pass readouts untrained, so RLTT depends on R4's loss placement: if
     the real run trains one CE on the last pass, step 1 adds a per-pass readout term; under
     per-loop CE (today's code) it is there already. R4 was dropped 2026-10-24 (the user), so the
     loss placement is set by the run spec without that arm.
   - Fixed depth: rollouts run at a fixed depth (the convergence exit is exclusive with the cache)
     and RL turns `loop_count_sampling` off, so the training forward runs the rollout's depth.
   - Routing replay (R3, arXiv 2510.11370): the expert choice of every pass at rollout is recorded
     and reused in the training forward; routing mismatch between the two forwards is the known
     cause of MoE RL collapse, and the loop repeats it once per pass. Build: a forced-index
     argument to the router that also fills the dense gate slots.

   Rewards:
   - Outcome on retrieval QA and chains, paid only when the supporting chunk is visible.
   - The three-way abstention reward (TruthRL, arXiv 2509.25760): correct +1, abstain 0, wrong -1;
     an item without its evidence pays for abstaining. **The one RL step that must not fail**
     (decided 2026-10-17, the user): the pilot's arm P abstained on 96% of answerable SQuAD (that read was void, the packer bug; on the fixed packer 2026-10-24 it is 77%, EM 0.05)
     questions with the gold passage attached while its likelihoods and its groundedness head are
     right, and the data lever was left untouched on purpose; this reward is where that behaviour
     is fixed, and the generated-answers block is read on every arm until it is.
   - Evidence conflicts: counterfactual cards, paid for following the visible evidence (the
     behavioural reading of the leak, decided 2026-10-06).
   - A per-query cost once step 4 exists.
4. **Retrieval as an action**, after the append-only buffer and per-pass re-query are built (Parked
   today). The store lookup is a top-k search with no gradient, and whether to re-query is a
   decision: RL learns both from the answer reward (Search-R1, arXiv 2503.09516), with retrieved
   tokens masked from the loss and the per-query cost against query spam (its follow-ups report
   many loosely relevant queries under an outcome reward alone). The selector's ranking inside the
   buffer keeps its selection loss and InfoNCE.

Not RL: MoE balance (a pretraining choice, R8), the selector, the null key, the gates, MTP.
Halting stays Parked: a reward of correctness minus a cost per pass only once a halt skips real
compute. A reward for depth itself is never used; it buys passes that do nothing.

**Prerequisites**, built before any rollout budget (no serving engine runs this architecture):
- Evidence prefill into the cache. The evidence states are fixed per forward and the chunk gate is
  exactly 1 while `freeze_evidence_gate` holds (every config sets it), so each pass's prefix keys
  and values can be written once into `shared_attn`'s cache slot; the asserts against a cache with
  evidence (`gemma4.py`, `moe.py`, `inference.py`) come out. Valid only while the gate is frozen.
- A batched KV cache; today it is single-sequence. `eval_abstention.py` already decodes batched
  without a cache, but it re-encodes the evidence every step, returns no logprobs and is not
  bit-exact through the MoE.
- Rollout outputs: the sampled tokens' logprobs and the per-pass routing record. **Reads** after every stage: A1 closed-book rank (a reward on
answers can pay for recall from the weights, which is the leak with a reward behind it), A3,
pass@k curves up to k 64 (whether RL sharpens or expands, arXiv 2504.13837 against ProRL), the
benchmark suite within noise. **Budget**: unmeasured; the rollout rate sets it, and its line in
the run spec is written after the cache build.

---

## Acceptance (all gates)

Kept results:

- **G0** benchmark harness and noise floor: PASS 2026-08-26. The seed-noise half is still open.
- **G1** IR ablation: FAIL, 0.0004 / 0.0002 nats.
- **G2** sharpening: FAIL, ablation at 0.0002 nats on both arms.
- **G2b** scale fix: FAIL, 0.0002 nats; `g_proj` stalled at RMS 0.0047.
- **G2c** loop input injection: FAIL, +0.0007 / +0.0009 against a 0.01 bar, stopped at 123M tokens.
- **P0** head removal neutral: PASS. **P2** abstention repair: PASS on false abstention, recall
  open.

Live:

- **G3** fixed-target gold gain, gain = CE(none) - CE(cond), at least 1.6 nats (50% of the
  3.23-nat ceiling) on held-out rows; distractors gain at least 0; benchmarks within noise
  (Phase 4).
- **G3b** chunk AUROC (per token, chance 0.5) at least 0.59 per loop; mass/chunk AUROC and
  grounded AUROC (evidence rows) at least 0.674 (Phase 4).
- **G9** the oracle-head reading recorded; tie the head if the factoring costs at least 0.02 nats
  (Phase 4b). The PLE ablation is a separate reading and gates the per-layer and router tables.
- **A1 to A7, L1, L2** as in "What must be proven" (R0b and Phase 5 for the pilot readings, Phase
  6 on the real run).
- **G8** throughput (Phase 6a).
- **G7** the shipped checkpoint (Phase 6e).

Retired: G4 (retriever alignment was planned as an InfoNCE term inside the pilot; the pilot ran
without one, the real-run spec carries no retriever term, decided 2026-10-21, and the model's query
against the store is read by A7's recall half), G5 and G6
(subsumed by A6 and the benchmark evidence path; the beyond-context claim is read as HotpotQA with
64-plus chunks through the port against the best in-prompt packing, at linear cost), the previous
plan's retrieval gates R1 to R5 (A1 to A4 are their measurable forms; the ladder rungs R0 to R9
above are a different, live series).

## Risks

- **Facts reach the weights through the input side.** Masking unsupported spans is necessary and
  not sufficient; later tokens still learn entity-to-attribute links from the input. The
  counterfactual swaps and the tail anonymization are the levers, and their rates are guesses.
  Without them A1 is expected to fail. The levers only work if they are complete: MTP targets carry
  the same span weight, a swap reaches every visible chunk and every later mention, the
  anonymization is re-drawn per epoch, and attribute nouns the tagger misses stay learnable. R0b's
  arm (b) was the input-side leak test and stored nothing at 3 sigma (2026-10-04). Arm (c) leaked
  anyway (2026-10-05): the leak is present at 50M, before its reader copied, and flat after 100M.
  The copy-first warm-up arm showed that the input side and the card path store nothing
  detectable without real-value targets, and that real-value supervision stores tier 1000 facts
  again even with copying in place (0.085, z 6.4 at 300M). Copying first brings tier 100 inside 3
  sigma on both forms; what sets the remaining tier 1000 level (the swap rate, the gold-drop
  documents, residual span loss) is open. The tier 1000 leak is recorded as known (decided
  2026-10-05); the swap rate 0.30 branch (read 2026-10-06) found no large effect of the swap rate on it, and
  the placeholder name arm (read 2026-10-08) falsified the dose lever, so the copy-criterion span
  weight is dropped and no lever against it is left before any 1B-token spend.
- **Small models ignore evidence.** Models of 7B and under have been measured ignoring oracle
  passages 85 to 100% of the time on questions they cannot answer alone. At 120M the pilot may
  come out flat. The matched control and A6 decide, and the 1-hop curve must saturate first.
- **Template learning without transfer.** Synthetic chains can pass while HotpotQA stays flat;
  published second hops fail off template, and BoolQ here is below chance. R5 catches it.
- **The bridge never reaches the query.** At loop 1 the read is one summand under `post_norm`; the
  answer position may not carry the bridge at loop 2. If loops collapse again (`cos(d3, d2)`
  high) the explicit reader-to-query path comes back as an arm.
- **Retrieval mismatch and leakage.** The reader trains on bge's errors and serves on its own;
  near-duplicate continuations inflate the language-model gain while the selector learns topic
  overlap. The candidate refresh in the last 15% of the pilot, the overlap-tier report and the
  same-document exclusion are the mitigations.
- **Context reliance decays.** It rises early and then falls under finetuning on context that
  agrees with the weights. A3 is read at every checkpoint, with a kill on the trend.
- **Coverage.** With the document itself excluded, a fact that appears in one document only is
  never supported: masked, never memorized, never practised. The store must be at least 0.4B
  tokens and its coverage is a build metric.
- **Reader bandwidth.** One read per sublayer in two of three sublayers is still narrower than
  per-layer designs. The dense-layer reads are the prepared arm.
- **Ambiguous pilot null.** 1B tokens at 120M with 10% synthetic may undertrain. The prefix and
  inject interaction and the depth-5 memory is read (R1b, 2026-10-21: fits at batch 4 x 2; the prefix needs a straight-through route). R1b and the saturating 1-hop curve are
  the guards.
- **Benchmark contamination.** Standard val and test splits only; a single suspiciously strong
  benchmark is suspect, not a win; the A1 probe answers are held out of the pretraining text.
- **FP8 divergence.** Validated on the 2B-token A/B before the real run trusts it.
- **Muon stores the tail.** Muon learns rare associations in the feed-forward and value/output
  weights better than Adam (arXiv 2509.26030), which is where a leak is written and what the tier
  100 cell measures. A faster optimizer can raise A1's leak; R7 reads it on the R0b instrument
  before the real run trusts it. Under Muon a matrix with any gradient moves at full rate, so a
  zero-init tensor that stays flat no longer means what the kill rules assumed; the real run has
  no such rules, the micro arms read those lines as descriptive.
- **RL pays for recall from the weights.** A reward on a correct answer given without its evidence
  trains closed-book recall, the leak with a reward behind it. 6f pays outcomes only with the
  supporting chunk visible, pays abstention without it, and reads A1 after every stage.
- **Rollouts are the cost of 6f.** Decode with evidence is uncached today and no serving engine
  runs this architecture, so the rollout rate is unknown and can make 6f's RL stage unaffordable;
  on-policy distillation needs fewer samples per prompt and comes first.
- **Data prep at 100B tokens is a real job**, budgeted in the run spec.
- **Stale numbers.** Every FLOP, parameter and ratio figure in this plan is an estimate until the
  constructor and the builder print it. Budget math keyed to an estimate goes stale silently.

## Literature behind the 2026-10-03 changes

- Ouro, "Scaling Latent Reasoning via Looped Language Models" (arXiv 2510.25741): looped and
  non-looped models store about 2 bits per parameter alike, the loop gain is knowledge
  manipulation, and depth is allocated by an entropy-regularized learned exit. The storage half of
  the loop falsifier and the depth allocation arm.
- Allen-Zhu and Li, Physics of Language Models 3.3 (arXiv 2404.05405): about 1000 exposures reach
  2 bits per parameter; junk data without domain tags cuts capacity. Why tier 1000 is read
  alongside tier 100 on a 95% filler corpus.
- Allen-Zhu and Li, Physics of Language Models 3.1 (arXiv 2309.14316): unaugmented facts are
  memorized but not extractable. Matches arm (a): tier 1000 perfect on the training templates, 0.10
  on a held-out one.
- "Predicting the Emergence of Induction Heads in Language Model Pretraining" (arXiv 2511.16893):
  the formation point is set by batch size x context size in updates, independent of model size,
  and a larger batch gives weaker heads. The batch caveat in "Now", and the reason a reader that
  reuses the model's own attention inherits copying instead of relearning it.
- Ram et al., "In-Context Retrieval-Augmented Language Models" (arXiv 2302.00083): documents
  prepended to an unchanged LM are read. The prompt copy control, and the design the key/value
  read imitates inside the loop.
- Memorization Sinks (arXiv 2507.09937) and the goldfish loss (Hans et al., NeurIPS 2024, arXiv
  2406.10209): alternative levers against memorization, held for the case where the swap and
  anonymization rates do not stop arm (c) from leaking. As of 2026-10-05 the goldfish loss is
  expected to do no better than a dose cut on the biography corpus, since its renders are
  paraphrased; it ranked after the placeholder name and the copy-criterion span weight, and since
  the placeholder arm's null (2026-10-08) a dose cut is not expected to move the tier 1000 level
  at all.
- Added 2026-10-12, for R7. Liu et al., "Muon is Scalable for LLM Training" (Moonlight, arXiv
  2502.16982): weight decay and an update scaled to AdamW's RMS make Muon scale without retuning,
  about 2x compute efficiency at compute optimal, on a 16B MoE. Wang et al., "Muon Outperforms Adam
  in Tail-End Associative Memory Learning" (arXiv 2509.26030, ICLR 2026): the gain sits in the
  value/output and feed-forward weights and comes from learning rare classes better, the reason
  R7 decides on the leak. Kimi K2 (arXiv 2507.20534): attention logits grow under Muon at scale,
  fixed by QK-clip.
- Added 2026-10-12, for 6f, R8 and R9. MobileLLM-R1 (arXiv 2509.24945): a 950M reasoner from
  4.2T pretraining tokens and SFT alone, so at this size the reasoning ceiling is set in
  pretraining. Li et al. (arXiv 2502.12143): small models learn short and mixed chains better than
  long ones. Thinking Machines, "On-Policy Distillation" (October 2025). Yue et al. (arXiv
  2504.13837) and ProRL (arXiv 2505.24864): RL sharpens within the base model's reach, or extends
  it under long stabilized training. DAPO (arXiv 2503.14476) and Dr. GRPO (arXiv 2503.20783).
  RLTT (arXiv 2602.10520) and LoopRPT (arXiv 2603.19714): RL for looped models. Rollout Routing
  Replay (arXiv 2510.11370): MoE RL stability. TruthRL (arXiv 2509.25760), Search-R1 (arXiv
  2503.09516). Loss-free balancing (arXiv 2408.15664).
- Added 2026-10-16, after the pilot read ([review_2026-10-16.md](../review_2026-10-16.md)).
  Limited Memory Language Models (arXiv 2505.15962, ICLR 2026) and Co-LMLM (arXiv 2607.07707):
  the nearest relatives of the pathway, fact values masked from the loss and a lookup from the
  model's own hidden state trained with in-batch InfoNCE; both still measure a residual in the
  weights against the standard model, so the paired reading of A1(b) is the field's own. The
  deletion audit of LMLM (arXiv 2607.00605): FULL / DEL-ON / DEL-OFF. Goldfish loss (arXiv
  2406.10209): a mask-rate lever against memorization. Knowledgeless LMs (arXiv 2607.12831):
  anonymized pretraining pays on grounded tasks and calibration. In-context two-hop phase
  transition (arXiv 2502.13913): the slow phase of random guessing among distractor chains is the
  pilot's chain read. From-scratch two-hop under looping (arXiv 2608.07261, 2604.07822): vanilla
  depth never composes out of distribution, a looped block does after grokking-scale training;
  identity bridge (arXiv 2509.24653): one zero-hop example per bridge entity unlocks it. Ouro
  (arXiv 2510.25741): capacity about 2 bits per parameter looped or not, loops buy manipulation.
  STARS (arXiv 2605.26733): loop performance peaks then collapses without a stability term.
  Over-refusal with sufficient context (arXiv 2603.06198, 2608.27661) and its RL fixes (RSRR,
  GRACE 2601.04525).

## Parked

- **Learned halting.** Only trainable if halting skips real compute; unparks after the exit and KV
  cache exclusion is fixed and a halt decision breaks the loop during training. The learned exit
  gate that raised an exemption question was dropped on the per-exit read (2026-10-06).
- **RL and reasoning training.** Planned as 6f since 2026-10-12; its RL stage still unparks only
  when pass@8 on the target task exceeds about 15%. A looped model wants per-loop latent credit
  (RLTT), not final-state GRPO. LoopRPT (reinforcement during pretraining of a looped model, arXiv
  2603.19714) stays a research arm only.
- **Per-pass re-query and the append-only buffer at serving.** Untrained; after the pilot.
- **Explicit reader-to-query feedback.** Query drift was measured without evidence; revisit only
  if it stays flat with evidence attached.
- **Evidence compression** (chunk embeddings in place of token states). A fallback for the memory
  wall; high compression costs 30 to 40% of capability in the published results.
- **Masked-diffusion trunk.** Unparks only if the budget math lands data-bound by 2x or more.
- **Hierarchical or adaptive softmax.** Changes the normalizer every calibration number assumes.

## What changed from the previous plan

- 2026-10-24 (night): the review's code changes landed (fp32 CE sums, the log-space selection
  loss with `evidence_selection_weighting`, `evidence_contrastive_loops` default `[0]`, the
  groundedness detach under the key/value reader, the row estimator and the 4093-token builder cap
  with the reader unchanged by decision, the evidence pad blocks, the dense-slot GEMM skip, the
  frozen-gate skip, fp32 selection scores under loss-free); held the eval-pass rewrite (6a) and the
  `eval_chains.py` bridge-mode items, which the review's Record paragraph had listed as landed by
  mistake. Built: the two leak candidate flags and their four splits, `inject_val_clean`. Running:
  the three micro retriever arms. The State block, the Open items table (the code row landed, the
  matching row out, the clean slice built, the retriever arms running, the leak arms built and
  queued), a Record paragraph, the R0b ladder row and the query training and repository review
  Decisions rows.
- 2026-10-24 (evening): the eval packer bug and R9's final read
  ([written_bridge_pilot.md](../measurements/written_bridge_pilot.md)). Every left padded decode
  with evidence before the fix read its buffer shifted by the dump width; the A3 generation, A7
  pathway and A4 edit reads of 2026-10-14 and the early bridge reads are void, the A3 and A7
  re-reads on arm P's final taken the same day (A3 weak in generation, EM 0.05 and abstention 0.77
  under gold, likelihood unchanged; A7 oracle-buffer EM 0.09, the model's query 0.001), the A4 flip
  read still unreadable on the model's query route (a bge-routed read needs the eval extended). R9 passes by a wide margin (held-out 2-hop 0.997 against chance 0.191, 0.982 at
  10M, 4-hop at chance past the trained two written steps): the State block, the Open items table
  (the R9 row out; R4 recommended dropped, open; the reasoning clause and 6f rows carry the read),
  two Record paragraphs, the R9 and R4 ladder rows and the reasoning post-training Decisions row.
  The user's decisions the same day: R4 dropped (obsolete on R9, the coda stays unbuilt), R6
  skipped (its mechanism read at micro, the spec uses the masking lever), A4's flip read routed
  through bge (`eval_store.py edit --route bge`, building): the State block (a Decided bullet; next
  the bge flip read, then R5 and the run spec), the Open items table (R4 and R6 closed, the A4 row
  building, an R5 row), the R4 and R6 ladder rows and the readout, loss placement, real run
  ordering, run order and repository review Decisions rows. The bge routed A4 flip read was built
  and read on arm P's final the same day (95 of 2,000 answered before the edit; delete flip 0.90
  +- 0.035, PASS; edit flip 0.13 +- 0.07 on 23 items, FAIL on generation, 0.09 follow with the
  edited chunk attached directly): the State block (nothing running, next R5 and the run spec), the
  Open items table (the A4 row closed, its edit half to 6f with A3), the written bridge record, the
  PILOT.md A4 row and the design page.
- 2026-10-24 (later): the repository review ([review_2026-10-24.md](../review_2026-10-24.md)) and
  the bridge's 30M read. The two "Now" sections restructured into a State block, one Open items
  table mirrored in CLAUDE.md and a dated Record; the records' gate labels renamed to match the
  gate table (Decisions, gate names); a Decisions row for the review (its plan proposals waiting
  on the user, the 4094-token and `inject_val` caveats as facts); the A1 bar and a new A1(b) row
  carry the amended paired reading, with the person-clustered sigma run the same day (within
  10%, the reading stands); the query training row reads the decision first; the loops axiom
  row gains the bridge read, the shape row the E4 result, the depth row R1b's caveat; the
  retired-gates lines corrected (rungs R0 to R9; the pilot had no InfoNCE term); the goal's
  reader is the key/value reader.
- 2026-10-12: Muon proposed for the real run (the user). A Decisions row (Muon on the hidden
  matrices, AdamW for the tied table, the head adapters, the router and every norm, scalar and
  vector), rung R7 (two micro arms against their AdamW twins, then one pilot-shape arm against arm
  P), a Risks entry (Muon learns the tail better, so the leak is the deciding read), and 6d takes
  the optimizer from R7. The pilot arms stay on AdamW.
- 2026-10-12: reasoning post-training (6f) and MoE balancing (the user). 6f: reasoning SFT,
  on-policy distillation, RL past the pass@8 gate with trajectory credit, depth records and
  routing replay, rewards on outcomes with visible evidence, abstention and evidence conflicts,
  retrieval as an action after the re-query build. Rungs R8 (loss-free balancing against the aux
  loss) and R9 (a written bridge on arm P's chains). Two Decisions rows and two Risks entries; the
  RL Parked entry points to 6f. A feasibility review against the code the same day changed: the
  Muon partition became a rule (tables, heads, router and zero-init matrices on AdamW, R7 tests
  today's set only); R7 gained its build items and an AdamW reseed twin; RLTT depends on R4's
  loss placement; the depth record became fixed-depth rollouts; the cache prerequisite became an
  evidence prefill plus a batched cache; the distillation teacher is the open item; R9 is budgeted
  in questions and needs a generate-then-score read.
- 2026-10-09: the pilot spec ([PILOT.md](PILOT.md)). The pilot is cut to what is built: the R0b
  recipe on today's code at 99.1M parameters (width 512, 6 prelude layers, 8 experts, seq 4096,
  key/value reader), a merged corpus from the existing builders (biographies plus filler 60%, QA
  plus web plus replay 25%, synthetic chains 10%, 1B tokens per arm), the copy-first warm-up with
  a switch criterion at every 50M save, swap rate 0.15, and a matched full-CE control. A new 1-hop
  gate (1-hop accuracy at depth 3 at least 3 sigma over chance by the 500M save) is the
  precondition for L1. Deferred: every Phase 5 blueprint item that is not built, and the rungs R4,
  R4b, R6. Not readable in this pilot: A1(a), A6 through the port, L2 at depth 4. Cost measured on
  the probe shape (2026-10-09): 44k to 50k tokens per second at batch 8 x 4096 (20.4 GB) on
  low-evidence rows, 2.7k to 3.9k on `evidence_train` (ratio 3.75) at the 4608 cap (fill 17 to 30%,
  15.0 GB); about 16k on the merged corpus by a two-point fit, 35 to 45 GPU hours for both arms
  (estimate); the chain slice (ratio 4.2) is the heavy one, and its per-document chunk count is the
  lever if one is needed.
- 2026-10-24: arm E4 complete and read ([expert_count_pilot.md](../measurements/expert_count_pilot.md)):
  4 routed experts at top 2 against arm P's 8 (89.6M against 99.1M, the same 84.9M active),
  killed once at about 680M on the user's request and resumed from the 652M save. Nothing
  measurable lost: held-out CE 0.0075 above arm P at the final, copying and the port read equal,
  the 1-hop gate 0.875 against 0.848, composition at chance on both, the leak equal at tier 100
  and smaller at tier 1000 (paired raw rank z -3.8, about one sigma on the prior-controlled
  `delta`); two changes (loss-free) and one seed each on record. The expert count becomes a cost
  choice for shape L until a read at the real run's size; R4b's sublayer half stays open. R9's
  bridge decode smoke (2026-10-23, arm P's final, marker rate 0 as expected) passed end to end and
  R9's SFT launched on the 500k-question bridge split (33.7M prompt tokens per pass, evidence
  ratio 3.26, two passes; it trained to its end the same day at 66.35M tokens, the 30M read is in
  the Record).
- 2026-10-23: run order and two rung cuts (the user, after a cost read). R4b's expert arm first, R9
  next, R6 last, one training run at a time with builds behind default-off flags; R4 staged from
  eight arms to a corner-first read of at most five, gated on R9; R4b's sublayer half named as the
  unbuilt three-sublayer core, its build was open for the user (decided 2026-10-25: one block per pass, not built). Arm E4 (4 experts, loss-free, arm P's
  recipe) launched; R9's bridge build and generate-then-score read built, review pending.
- 2026-10-21: four spec decisions (the user). Loss-free balancing enters the real-run spec in
  place of the aux term (R8). Muon is dropped from the real-run spec, the pilot-shape Muon arm P
  is not run and the real run stays on AdamW (R7 closed). The 100B-token corpus build (6c) runs
  on the rented cluster, not locally. The throughput work (6a) starts only after every other spec
  decision is made, so nothing optimized has to be redone; it moves to the end of the pre-run
  order. 6d states AdamW and loss-free balancing.
  Also decided (the user): the retriever's InfoNCE term enters the pilot-shape recipe at 0.25 and
  is read on one pilot-size arm before the real-run spec commits to it; arm PI launched
  (`launch_pilot_pi.sh`, `config_pilot_infonce.yaml`, loss-free, AdamW, arm P's two stages).
  Arm PI stopped at 87M the same day: held-out CE 0.30 nats behind arm P and growing (0.03 /
  0.11 / 0.30 at steps 890 / 1770 / 2660), store recall@5 0.006 against 0.004, so the term as
  built stays out of the real-run spec; the hidden-state probe and a head-only micro arm next.
  A research round on training the model's own retrieval without the language cost started;
  write-up [review_2026-10-21_retriever.md](../review_2026-10-21_retriever.md) written the same day.
  The query head probe ([query_head_probe.md](../measurements/query_head_probe.md)): no ridge or
  MLP head on the frozen loop 1 state, last token, mean or both, gets past about 7% of bge at k5
  (best 0.055 on `w25e3` against its trained query's 0.059, 0.025 on `w0e3`), so the term's gain
  split between the head and the trunk state, the head-only arm reads what the head alone gets
  from an untouched trunk (about 0.02 to 0.03 predicted), and a tower on detached states is
  capped the same way.
  Two training keys built and reviewed, `evidence_contrastive_loops` and
  `evidence_contrastive_trunk_grad` (0.0 trains only the query head, 1.0 is bit-identical to
  before); the head-only micro arm (`launch_infonce_head.sh`, loop 1, trunk gradient 0) launched.
  The head-only arm read the same day: held-out CE 4.3904 against `w0e3`'s 4.3809, recall@5
  0.003 against `w25e3`'s 0.059, so training only the search head is closed at this size.
  Decided (the user): the separate search tower is held, not built; bge keys the store and
  embeds the question at serving, A7's recall half stays untested, the model's own retrieval is revisited at
  the real run's size with `evidence_contrastive_loops` and `evidence_contrastive_trunk_grad`
  kept as the knob, and the real-run spec carries no retriever term.
  R1b run ([depth_memory_probe.md](../measurements/depth_memory_probe.md), `detach_loops` and
  `scripts/probe_depth_memory.py` built): fits, depth 5 at batch 4 x 2, the detached prefix needs
  a straight-through route to the prelude before use.
- 2026-10-20: the aux loss's masked branch fixed (`compute_aux_loss`): hard counts, mask and the
  probability reduction now run in fp32. In bf16 every slot's count saturated at 256, so on every
  masked aux-balanced run since the token mask (the R0b arms, both pilot arms, the Muon arms) the
  aux value was a constant near 0.056 and the term a floor, not a balancer; the pilot arms'
  routing was the router's alone. Test `test_aux_loss_mask_bf16_counts`; R8's comparison half
  needs an arm (a) rerun under the fixed term (about 85 minutes plus reads). The rerun ran the
  same day (`ckpts/evidence_auxfix_full`, 299.44M tokens in 79.7 minutes): loss-free is better
  on all three bars, the working aux term stores more at tier 100 in distribution (z 14.5) and
  moves a third to a half of the picks onto the dense slots; R8 passes for loss-free.
- 2026-10-19: every chain continuation save read (no transition inside the epoch); the
  balancing rung's build (R8): `training: moe_balancing: aux | loss_free`, the selection bias
  `moe.balance_bias` on `LoopMixtureOfExperts` with its per-step update and per-pass load log,
  `tests/test_loss_free_balancing.py`; the micro arm launched against arm (a). Read the same
  evening ([lossfree_micro.md](../measurements/lossfree_micro.md)): loss and storage pass, the
  MLP load even by its own field, the comparison unread because the logged aux value saturates
  in bf16; the real-run spec choice is the user's.
- 2026-10-18: the optimizer rung's build (R7): `modules/model/optim.py` (Muon on the hidden
  matrices by a name rule, AdamW's arithmetic on the rest, one class), `training: optimizer`,
  `eval_exit.py --pair`, `tests/test_muon_optimizer.py`; the three micro arms queued behind the
  three-epoch InfoNCE pair, which was relaunched clean from the seed after the evening's stop.
  The three-epoch pair read the same afternoon (the twin relaunched once more after a stop at
  85M): store recall@5 0.059 against 0.000, recall@20 0.133 against 0.002, at 0.07 nats of
  held-out language; the term's entry into the pilot-shape recipe is the user's decision
  (recommended at 0.25). The Muon arms ran behind it and were read the same night
  ([muon_micro.md](../measurements/muon_micro.md)): loss and copying bars pass, the leak bar
  passes by the letter (largest cell z 2.6) and is worse by the sign in distribution, Muon
  stores about twice as early on arm (a); the pilot-shape arm is the user's decision.
- 2026-10-17 (evening): the chain continuation of arm P complete and read
  ([chain_continuation_pilot.md](../measurements/chain_continuation_pilot.md)): one epoch of the
  chain split takes 1-hop to 0.99 and leaves every composition cell at chance (held-out paired
  Delta +0.004 +- 0.008); L1 failed at this shape, the "falsified" row applies. The InfoNCE micro
  arm against its twin, one epoch of the QA slice (35M tokens each,
  [infonce_micro.md](../measurements/infonce_micro.md)): store recall@5 0.012 against 0.000,
  recall@20 0.035 against 0.001, language and selection equal; the three-epoch pair launched. The
  model-keyed retriever variant held until it reads.
- 2026-10-17: the pilot's verdicts decided (the user): A1(b) passes with the leak known and the
  criterion amended to the paired reading; L1 failed at this shape pending the chain continuation
  of arm P (launched, one pass over the chain split, about 9 GPU hours); the retriever's in-batch
  InfoNCE built (`evidence_contrastive_loss`, weight 0 by default); A3's behaviour to 6f's RL,
  marked the RL step that must not fail; queue: chain arm, InfoNCE arm, R7, R8, R9. Intermediate
  pilot saves deleted (34 GB), the design page republished in this organization.
- 2026-10-16: research round after the pilot read ([review_2026-10-16.md](../review_2026-10-16.md)):
  LMLM and Co-LMLM as the nearest relatives with the same residual, the chain read as the slow
  phase before an in-context two-hop transition, looping as the condition for from-scratch
  composition, in-batch InfoNCE as the A7 recall fix; seven directions proposed, none decided.
- 2026-10-14: arm C complete at 909.72M (one epoch) and read: the control stores tier 100 at
  0.467 in distribution (z 62) and 0.049 held out (z 12), tier 10 at 0.20; the paired compare
  against arm P reads z -81 / -6.4 at tier 100 (held out on entities only: raw z -3.8, `delta`
  z +7.0). A1(b)'s control half holds; its arm P half fails
  on the held-out form by z 3.2 at a 4x reduction. The ladder's remaining reads done the same
  day: A5 and A6 hold, A3 holds in likelihood and is unreadable in generation (96% abstention),
  A7's recall half fails (the query at recall@5 0.004 against bge 0.829), A7's pathway half and
  the A4 edit unreadable (both taken before the 2026-10-24 packer fix); exits and depth reads
  repeat the micro shape. The verdict rows waited for the user (decided 2026-10-17; the "Run" section of
  [PILOT.md](PILOT.md)).
- 2026-10-11, evening: arm P complete at 860.61M (one epoch) and read at the final save: 1-hop
  0.85 (gate holds), L1 Delta +0.002 +- 0.005 (the "falsified" cell), tier 1000 0.064 / 0.046,
  tier 100 z 3.2 held out and z 2.2 in distribution (the arm P half of A1(b) not met on one
  form), copying intact, A2 and A5 clear on the trainer's numbers. Arm C launched. Verdicts
  pending arm C and the user (the "Run" section of [PILOT.md](PILOT.md)).
- 2026-10-11: arm P read at 530M after a pause at 429M: the 1-hop gate holds at its 500M bar
  (1-hop 0.53, rising from 0.32), L1 flat (every composition cell at chance, paired delta -0.004),
  the tier 1000 leak 0.040 / 0.043 on the two forms with tier 100 flat, held-out copying back to
  0.97. At 631M: 1-hop 0.59, L1 flat, tier 1000 0.041 / 0.028, the lower tiers creeping in
  distribution inside 3 sigma per cell, held-out copying 0.92. At 731M: 1-hop 0.80 with the
  answer under ln K, L1 flat, tier 1000 0.064 / 0.045, and the A1(b) tier 100 cell at z 3.2 in
  distribution (z 2.4 held out), a breach on one form at one read. At 832M: the breach on the
  other form (tier 100 held out z 3.4, in distribution z 1.8), tier 1000 0.068 / 0.046, 1-hop
  0.85, L1 flat; decided at the final save against arm C (the "Run" section of
  [PILOT.md](PILOT.md)). Nothing in the spec changed.
- 2026-10-10, evening: arm P launched at 19:03, ran at 42k to 48k tokens per second (three times
  the fit), was stopped after the 300M save and read: copying complete (gold top-1 1.000 in
  distribution, swapped followed 1.000), closed book at chance in every cell on both forms, the
  1-hop gate passed (0.32 against chance 0.115, z +21) with 2-hop and held-out composition at
  chance. Relaunched on the main split from the 301M save (the "Run" section of
  [PILOT.md](PILOT.md)). Nothing in the spec changed.
- 2026-10-10: the pilot's build list is done (the "Build" section of [PILOT.md](PILOT.md)): the
  split merge tool and its test, the `eval_chains.py` instrument fixes, `config_pilot.yaml` and
  the seed, the biography builds at 4096 and the three merges (910.2M prompt tokens per arm, the
  budget; evidence ratio 0.717), the launchers and the per-save read script. The merged splits
  share one document order (`--order-from`), so the switch from the warm-up to the main split
  resumes by position without a repeat or a gap. Nothing in the spec changed.
- 2026-10-08, later: the chain depth arm (`chains_kv`, 102.66M tokens, 2M chain questions, key/value
  reader from `seed_micro.pt`) read **inconclusive** under its pre-registered criteria
  ([chain_depth_micro.md](../measurements/chain_depth_micro.md)). Every accuracy cell is at chance
  at 50M and final, the 1-hop curve included (0.104 at depth 3 against chance 0.115), so the
  readable precondition fails; held-out Delta +0.006 at sigma 0.013. The model copies an
  answer-type entity from the buffer and does not select which (about 3.3 nats per answer against
  about 1.9 for a uniform pick among the candidates); the selector stayed near uniform and the
  third pass's `loop_scale` collapsed to under 0.01 by 50M. No change to depth or to the pilot's
  chain slice; the pilot's L1 is the deciding read; the pilot spec is next and the ladder below it
  is the pilot. Options for a readable micro read (a larger model or dose, a 1-hop curriculum, a
  selector loss that is not left uniform) are recorded, not planned. Instrument gaps in
  `eval_chains.py` are listed in the record.
- The goal is stated over the pathway and the strong form is replaced by A1 with A3 and A4.
- The learned IR table, the selector's value read and adapters, the chunk gate and the factored
  heads leave the real run. The selector becomes always on; the head is tied.
- Phase 4 is a mechanism check with fixed instruments, not the main POC spend.
- Phase 5 (retriever alignment and the real index) is folded into the pilot: the InfoNCE term,
  the store, the near-duplicate filter and the evidence path in the benchmarks.
- The loop gets a readout stage, one loss on the last pass, a depth schedule to 5 with a detached
  prefix, fresh loop scales, input re-presentation and a union selection loss.
- The from-scratch corpus attaches retrieval to every slice, masks unsupported fact spans, swaps
  entities, anonymizes the tail, caps the QA buffer, and includes synthetic chains.
- The real run targets shape L with a fixed token target and derived hours (about 2.5k H100 hours
  planned), with a context extension phase after it.
- 2026-09-30 revision after the review: A1 becomes a closed-book rank against the matched control
  plus a fact-injection probe, A3 a memorization ratio, A4 goes through the store, A7 is added,
  L1 and L2 are scored by read ablation at fixed depth, R0b is added, and the budget is recomputed
  at about 1.3 GFLOP per token.
- 2026-10-08: the placeholder name arm (`inject_retrieval_kv_cf_p50`, from the warm-up arm's 100M
  save, `--placeholder-rate 0.5`) read: tier 1000 entity `delta` 0.0986 (z 6.7) against the warm-up
  arm's 0.0848, paired +0.011 (z +1.8) pooled over forms, so the pre-registered falsifier (pooled z
  above -2) is met: halving the real-name-with-real-value supervision did not lower the maintained
  level. The held-out form fails the rule at tier 100 (0.0133, z 3.7) and held-out copying degrades
  (swapped followed 0.51 to 0.54 against 0.60 to 0.63). Decided the same day: the copy-criterion
  span weight is dropped, and the pilot's lever list is the copy-first warm-up alone at swap rate
  0.15. The chain depth arm's spec is written
  ([chain_depth_micro.md](../measurements/chain_depth_micro.md)): about 104.7M tokens, one epoch
  of 2M questions, the held-out split rebuilt at 2,000 questions, the per-exit read replaced by the
  full-site cells at depth 1, 2 and 3; splits built, training started 2026-10-08.
- 2026-10-06: the swap rate 0.30 branch (`inject_retrieval_kv_cf30`, from the warm-up arm's 100M
  save) read: tier 1000 entity `delta` 0.083 (z 6.2) against the warm-up arm's 0.085, paired
  -0.002 (z -0.3) in distribution and -0.008 (z -1.4) held out, so the pre-registered criterion is
  not met and the swap rate is not a strong dial for the level that training maintains (no large
  effect; a dose-sized one is under the stated power). Tier 100 is lower on both forms as a
  direction (paired z -1.0 and -1.4) and the rule holds with more margin; copying is unchanged in
  distribution. Decided by the user the same day: the pilot's swap rate stays at 0.15 (the user's
  reasoning: a leak this size matters little for the pilot, and RL planned later can act on small
  leaks; caveat on record: RL is Parked and shapes behaviour at conflicts rather than removing what
  is stored, so for small leaks the goal is read as behavioural at conflicts); the placeholder name
  arm is approved, not started; the learned exit gate arm is dropped and its three open questions
  are moot, and a chain depth arm (L1 at micro scale plus the per-exit read) is approved, not
  started, spec first. The placeholder name arm is designed (ladder step 6: about 40 lines plus 30 of tests, about 64 minutes
  of training from the warm-up arm's 100M save; expected tier 1000 0.043 if the maintained level is
  linear in dose, 0.064 if it saturates, unchanged under the before-copying reading); the
  recommendation is to run it before freezing the pilot's lever list. The seed-side instruments are
  done ([seed_instruments.md](../measurements/seed_instruments.md)): PopQA needs its prior control
  (a raw-rank 7 sigma effect of the 10M finetune is the prior moving), and the counterfactual
  strata must be read by frequency ratio or against a neutral-port baseline; gates A1 and A3 carry
  both notes.
- 2026-10-05: R0b arm (c) ran in full with the key/value reader. It copies completely from 150M on
  (top-1 1.000, a swapped card followed on every item in distribution) and fails the rule on both
  forms (tier 100 0.027, z 4.1 in distribution; 0.013, z 3.6 held out; tier 1000 a leak on both).
  The leak is present at 50M before copying, at arm (a)'s level at tier 100, and does not grow
  after 100M: not a climb. The 0.30 sweep arm was set aside and a copy-first warm-up arm (swap
  rate 1.0 until the reader copies, then the arm (c) split) launched with its criteria fixed
  before the read. Its first phase met both switch criteria (copying at 0.95 to 0.97 top-1, nothing
  stored at 50M or 100M), so the arm (c) store is written by supervised real-value spans, not by
  the input or card path; its 150M read, 50M tokens after the switch, has tier 1000 back at 0.081
  (z 6.0, paired +0.082, z 5.9), which triggers the pre-registered falsifier at tier 1000. Read in
  full at 300M (299.19M tokens, filler CE 3.9106 against 3.8842, gap unexplained): `compare`
  HOLDS through tier 100 on both forms (0.015, z 2.6 in distribution; 0.007, z 2.0 held out),
  tier 1000 leaks on both (0.085, z 6.4; 0.047, z 5.1), the 0.03 expectation failed, and the
  warm-up arm sits under the original arm in every per-form cell without reaching 3 sigma per
  form. The arm (c) leak is read as both: in part written before copying, in part a level that
  training maintains while real values are supervised next to the real name. Decided by the user
  the same day: R0b is accepted on the warm-up arm with the tier 1000 leak known, "no real-value
  fact supervision before the reader copies" enters the pilot as a requirement (the warm-up is
  the better of two arms on one seed, not an optimum), and a swap rate 0.30 second phase branched
  from the warm-up arm's 100M save (`inject_retrieval_kv_cf30`) runs before the pilot spec is
  frozen, with its criteria and its power stated before the read; the placeholder name follows if
  the level does not follow the swap rate. Corrections: the
  held-out hold at matched 100M did not last; copying emerged between 50M and 100M. A per-exit
  read (`eval_exit.py`) finds the second pass's gain spread evenly over tokens and the
  entropy-regularized exit optimum near uniform, so the learned depth allocation arm is held (dropped 2026-10-06), with
  three decisions open (an exemption from the halting rule, the equal-weight control arm, the
  pilot's lack of per-pass exits) and the recommendation to read depth on the chain corpus instead.
- 2026-10-04: R0b arm (b) ran in full and stores nothing at 3 sigma, so the arm (c) leak is most
  likely on its supervised spans and the anonymization sweep arm is dropped (it reaches under 1% of
  the supervised spans that see the real name); the loop reading on arm (a) met the storage
  falsifier's condition at tier 100 (closed-book `delta` grows with depth, almost all in the second
  pass, on the real name only; tier 1000 is saturated after one pass), and the axiom is reworded
  to "passes add computation, not capacity; recalling a weakly stored fact can take more than one
  pass", with leak reads at full depth; `closed_book_rank.py bios --n-loops` reads a trained exit
  depth.
- 2026-10-03: R0b arm (a) read the instrument; the copy control showed the 20M arm (c) smoke was
  read before copying existed; the key/value reader (evidence as leading keys of the shared
  self-attention, prefix rotation) copies at 100M where the cross reader does not at matched
  tokens, and replaces it in the plan; arm (c) leaks at tier 100 at 100M under both readers; the
  pass rule reads tier 1000 in distribution alongside tier 100; the loop axiom gets a storage-half
  falsifier (closed-book rank by loop count on arm (a)) and learned depth allocation is the planned
  loop fix; `eval_abstention.py --counterfactual-likelihood-only` reads abstaining checkpoints.
- 2026-10-01: Phase 4 ran and both arms were killed on the content gain; the pilot's first reader
  arm is a reader per loop or dense prelude reads, without rotary. The graft branch is closed:
  all seven failed gates measured the same graft condition, and no from-scratch result exists yet
  in either direction. Phase 4b and the R0b tooling are built (see "Now"); R0b is the next spend,
  arm (a) first, a 20M smoke of arm (c) as the reader precondition, and the ladder below R0b is
  cut to the budget before the pilot launches. Decisions taken in the build: the micro shape is
  selected by `TINY_LLM_CONFIG` (shape is not inferable from a state dict); the span weight
  travels through `.mask`, so the MTP targets carry it; biography values are fictional and drawn
  uniformly per person, but exposure weighting makes the corpus marginals non-uniform, so the
  closed-book reading is paired against a fresh-name prior control (PopQA likewise against a
  subject-blind control); chain chance is one over the distinct objects of the question's
  final-relation chunks, since the wh-frame names that relation; read ablation cuts the reader
  and the IR expert's external read together at a site; the model-side store index on this
  lineage goes through `key_adapter`, so it is per checkpoint.

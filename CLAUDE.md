# CLAUDE.md

Operational map: where things live, invariants that bite, how to run things, what is next. Prose
lives in [docs/](docs/): [architecture.md](docs/architecture.md), [moe.md](docs/moe.md),
[training.md](docs/training.md), [configuration.md](docs/configuration.md),
[runbook.md](docs/runbook.md), and the latest full review,
[review_2026-10-24.md](docs/review_2026-10-24.md). Every gate is in
[docs/measurements/](docs/measurements/). The plan is [docs/plans/NEXT.md](docs/plans/NEXT.md)
(older notes call it `PLAN.md`).

## Now (keep current)

Mirror of the "Now" section of NEXT.md; update both when the next step changes. The State block
and the Open items table are the same text in both files; the Record below is terser here.
Restructured 2026-10-24 after the repository review.

### State (2026-10-24)

- **Ran and read this evening:** two of the three micro retriever arms the review proposed,
  launched 2026-10-24 about 20:25 in sequence (`bash ckpts/inject/launch_review_arms.sh`, about 40
  minutes each plus the store recall read, on the code in the tree, three passes over
  `evidence_nomany_train` from `seed_micro.pt`): `w0e3b`, the `w0e3` recipe rerun on the reviewed
  code as the twin the other two are read against (`ckpts/inject/config_micro_infonce0_e3.yaml`);
  `l0e3`, the `w25e3` recipe with the contrastive term on the first pass only and the trunk
  gradient on (`config_micro_infonce_l0_e3.yaml`). The third, `seg_e3`, the `w0e3` recipe with the
  selection loss weighted per query segment (`config_micro_segsel_e3.yaml`), runs 2026-10-25. Run
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
  0.0001, 4.6262 against 4.6261; 104.56M tokens, exit 0); `seg_e3` runs 2026-10-25 (its yaml
  parked as `config_micro_segsel_e3.yaml.hold` so the first launcher skipped it; rename it back
  before `bash ckpts/inject/launch_review_arms.sh seg_e3`), then the leak arms and the clean-slice
  read. The card pauses around 22:30 on 2026-10-24 (the user).
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
  code changes land (landed the same evening, uncommitted, tests passing on the GPU, a detached
  Sonnet review with no high or medium finding; the Open items row and the Record), except the
  eval-pass rewrite (deferred to 6a) and the `eval_chains.py` bridge-mode items. The evidence
  reader keeps dropping the 4094-token rows: future evidence arms on the existing builds stay
  matched to the arms already run (the pilot merges keep dropping the 12,240 rows; new evidence
  builds cap a training row at 4093 tokens). Run order: the three micro retriever arms, then the
  two leak arms, then the clean-slice CE re-read, with R5's eval set between them when the card
  is idle.
- **Just read:** this evening, two of the three micro retriever arms
  (`docs/measurements/infonce_micro.md`, the first pass only section): `l0e3`, the contrastive
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
  attached directly, a generation weakness on real questions, not a store failure. Arm E4
  (`docs/measurements/expert_count_pilot.md`: the expert count is a cost choice); the repository
  review (`docs/review_2026-10-24.md`).
- **Next, in order:** `seg_e3` on 2026-10-25 (rename
  `ckpts/inject/config_micro_segsel_e3.yaml.hold` back to `config_micro_segsel_e3.yaml`, then
  `bash ckpts/inject/launch_review_arms.sh seg_e3`; store recall and held-out CE against the fresh
  twin `w0e3b`); then the two leak arms (`bash ckpts/inject/launch_leak_arm.sh m1`, then
  `g1`: the copy-first warm-up recipe on the flagged splits, run directories
  `ckpts/evidence_leak_{m1,g1}`, reads `ckpts/inject/rank_leak_{ARM}_{100M,final}_*` and
  `compare_cf_vs_leak_{ARM}.log` against the warm-up arm); then the clean-slice CE re-read (`bash
  ckpts/inject/read_clean_val.sh`: `eval_exit.py` on the arm P, C and E4 finals on
  `inject_val_clean` and `inject_val`); R5's eval set between them when the card is idle; then the
  run spec.
- **Open for the user:** the goal's reasoning clause reworded on R9 (written tokens); R4b's
  sublayer half; 6f's abstention reward against the A3 re-read.

### Open items

Mirrored in both "Now" sections; a row leaves when it is done.

| item | status | the read that gates it | waiting on |
|---|---|---|---|
| the review's code changes | landed 2026-10-24 (uncommitted; tests pass on the GPU; a detached Sonnet review found no high or medium issue): fp32 CE, top-1 and `p_max` sums in `mtp.py`; the log-space selection loss with `evidence_selection_weighting` (`token` default, `segment`); `evidence_contrastive_loops` default `[0]`; the groundedness head input detached under the key/value reader; the row estimator replaying each reader's rule and a 4093-token training cap in the evidence builder (the reader itself unchanged by decision); the evidence padding cut into 128-token blocks; the dense-slot GEMM skip; the frozen-gate skip; fp32 selection scores under loss-free. Held: the eval-pass rewrite, deferred to 6a; the `eval_chains.py` bridge-mode items (in `--generate-bridge` the latent verdict lines suppressed and the kept 3 minus kept 1 line labelled bridge-internal; the evidence built once per batch) | none | 6a (the eval pass); a free card for the `eval_chains.py` identity check |
| A4 edit, a flip read routed through bge | closed: read 2026-10-24 (`scripts/eval_store.py edit --route bge` on arm P's final, retrieval through the store's bge query keys, the serving configuration; `ckpts/inject/pilot_p_ladder2_store_edit_bge.*`; 95 of 2,000 answered before the edit): delete holds (flip 0.90 +- 0.035, n 72, bar 0.70), edit fails on generation (flip 0.13 +- 0.07, n 23; the edited chunk attached directly followed 0.09); the edit half goes to 6f with A3 | none | nothing |
| R5, the winner, eval only | next, on the chosen shape (the A4 bge read done 2026-10-24); runs between the micro arms when the card is idle | held-out 4-hop by read sites and at D = 4; HotpotQA bridge by read sites; A1 to A7 | the chosen shape; an idle card |
| a clean validation slice (documents neither builder used) | built 2026-10-24, re-read queued: `scripts/prepare_clean_val.py`, `data/prepared_pilot/inject_val_clean` (974 windows, 650,125 tokens; verbatim 64-token overlap with the QA training rows' supervised tokens 0.99% against 13.3% for `inject_val`) | `bash ckpts/inject/read_clean_val.sh` (`eval_exit.py` on the arm P, C and E4 finals on both slices) | the two leak arms |
| a value-to-name closed-book probe | proposed, mode not built | arm P's final | a build |
| R4, the coda corner | closed: dropped 2026-10-24 (the user), obsolete on R9 (composition at 0.997 held out once the intermediate entity is written out); the coda stays unbuilt | none | nothing |
| R4b, the sublayer half | open; the review recommends a cost choice, no arms | none at pilot scale | the user |
| three micro retriever arms (a fresh `w0e3` twin on the reviewed code; the contrastive term on the first pass with the trunk gradient on; per-segment selection weighting) | two read 2026-10-24 (`bash ckpts/inject/launch_review_arms.sh`; `docs/measurements/infonce_micro.md`): the twin `w0e3b` final CE 4.3996, recall@5 0.000; `l0e3` recall@5 / @20 at loop 1 0.049 / 0.122 against `w25e3`'s 0.059 / 0.133 at 0.041 nats against the twin (the every-loop pair 0.074), gate failed, so the cost sits in the trunk's change and the decision of 2026-10-21 stands; `seg_e3` runs 2026-10-25 (rename `config_micro_segsel_e3.yaml.hold` back first) | store recall and held-out CE against `w0e3b` | `seg_e3` |
| two leak candidates (value-first templates; distractor-collision spans) | built, queued: builder flags `--mask-first-name-mention` and `--support-gold-only` (`.mask` only), splits `inject_retrieval_s100a50m1_train`, `s100a50g1`, `s15a50m1`, `s15a50g1` built 2026-10-24; arms `bash ckpts/inject/launch_leak_arm.sh m1` and `g1` | closed-book rank at 100M and at the final against the copy-first warm-up arm (`compare_cf_vs_leak_{ARM}.log`) | the retriever arms |
| R6, pretraining-text facts | closed: skipped 2026-10-24 (the user); its mechanism is read at micro (arm (a) stores plain-text facts, tier 100 `delta` 0.378; arm (b), spans masked, stores nothing; arm C repeats arm (a) at pilot shape) and the run spec uses that lever | none | nothing |
| the goal's reasoning clause reworded | open since 2026-10-17; R9 read 2026-10-24: composes in written tokens, not in the loop | none | the user |
| 6f, reasoning post-training | planned; its reasoning form written tokens (R9, 2026-10-24), with multi-hop rows at the serving depth | the A3 generation re-read (weak, 2026-10-24) and A4's edit half (fails on generation, 2026-10-24); decode with evidence through a KV cache | the real run |
| 6a throughput, 6c corpus on the rented cluster, 6d run spec | after every other spec decision (2026-10-21); 6a carries the deferred eval-pass rewrite | none | the rows above |

**The design page moves with the architecture and the plan.** Any change to the architecture (a
tensor, a module, a loss term, an invariant) or to the plan (NEXT.md, this section, a decision, a
gate, a measured verdict) is also made in `docs/evidence_path_design.html` in the same turn; the
file goes into the same commit line. The repo file is the source. Its online copy,
https://claude.ai/artifact/GaXssrGoBoikpGMy7s8rsf (private, this organization, published
2026-10-17; the older copy in another organization is stale), is republished only when the user
asks (decided 2026-10-11); a new session reads the file.

### Record

In date order where it can be: the R0b bullets (2026-10-02 to 2026-10-08), the running record
(2026-10-05 to 2026-10-24), the two 2026-10-24 entries, the closed list, then reference numbers.
Superseded sentences are in the past tense; nothing is cut.

- Graft lineage, binding: Phase 4 arms A and B killed at 10.08M on content gain (+0.058 / +0.081);
  in-context ceiling on `evidence_fixed` 3.18 nats pooled; R0 failed; the graft branch is closed.
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

As of 2026-10-05
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
1-hop gate before L1; the unbuilt blueprint items and the rungs R4, R4b and R6 are deferred (brought back staged
2026-10-23). On
2026-10-10 its build list was done: the merge tool, the chain instrument fixes, `config_pilot.yaml`
and `ckpts/inject/seed_pilot_grounded.pt`, the biography builds and the three merges in
`data/prepared_pilot` (910.2M prompt tokens per arm, the budget), the launchers and the per-save
read script under `ckpts/inject/`. **Arm P ran 2026-10-10 to 2026-10-11 and is complete at
860.61M tokens, one epoch** (the trainer stops at the epoch's end, so the cosine fitted to the
merge's 910M stopped at LR about 6e-5; 42k to 48k tokens per second; the warm-up to the 300M save,
then `pilot_main_train`; reads at 300, 351, 426, 530, 631, 731, 832M and final, each a stop, read,
relaunch of 26 minutes, `ckpts/inject/pilot_p_{TAG}_*`; the "Run" section of PILOT.md). Final
read: the 1-hop gate holds (0.85 against chance 0.115, gold answer 0.82 nats against ln K 2.20,
from 0.32 at 300M); **L1 in the spec's "falsified" cell** (held-out 2-hop Delta kept 3 minus kept
1 +0.002 +- 0.005, every composition cell at chance at depths 3 and 4 at every read); **the arm P
half of A1(b) not met on one form** (tier 100 entity `delta` 0.0125, z 3.2 held out; 0.010, z 2.2
in distribution; tier 1000 0.064, z 7.2 / 0.046, z 5.5, written within 50M of the switch as at
micro and grown through the cosine's tail; the tier 100 forms swapped sides over the last three
reads, pooled z 4 to 5); copying intact (gold top-1 1.000 in distribution, 0.95 held out, swapped
followed 1.000 / 0.94 to 0.97); A2 and A5 clear on the trainer's numbers (gold gain +4.00, content
+3.97, chunk AUROC 0.78, grounded 0.994). Arm C (launched 2026-10-11, `launch_pilot_c.sh`,
`ckpts/evidence_pilot_c/`, `pilot_c.log`, 51k to 59k tokens per second; reads `read_pilot_save.sh
CKPT c TAG`) at 325M stores the biographies as the micro arm (a) did: entity `delta` 0.299 (z 43)
at tier 100 and 0.390 (z 25) at tier 1000 in distribution, 0.029 / 0.105 held out, about 300x and
9x arm P at the same tokens. **Arm C complete at 909.72M on 2026-10-14** (final `inject_val` CE
3.2195 against arm P's 3.2321): the control's final closed-book read is 0.016 / 0.201 / 0.467 /
0.468 by tier in distribution and -0.006 / 0.005 / 0.049 / 0.118 held out; the paired `compare`
(`ckpts/inject/compare_pilot_c_vs_p.log`) reads z -81 at tier 100 in distribution and z -6.4 held
out (all classes pooled; on entities only held out raw z -3.8, `delta` z +7.0, both with the
control storing more). A1(b)'s control half holds with margin; its arm P half fails on the held-out form at tier
100 (z 3.2) at a 4x reduction (47x in distribution). The ladder's remaining reads
(`ckpts/inject/ladder_pilot.sh`, outputs `pilot_{p,c}_ladder_*`, 2026-10-14): A5 holds (port
gain +4.00 against an in-context ceiling of 2.56 on the same save), A6 holds (13 tasks, within
noise of the control, both at chance on most), A3 holds in likelihood (gold 2.13 against none
5.87, `mr_ll` 0.11) and is unreadable in generation (96% abstention under gold, EM 0.01), A7's
recall half fails (model query recall@5 0.004 against bge 0.829), A7's pathway half unreadable
(oracle EM 2.6%) and the A4 edit unreadable (nothing to flip; the generation, pathway and edit
reads predate the 2026-10-24 packer fix, re-read pending); exit reads on both arms repeat the micro shape (pass 2 +0.07 to +0.08, pass 3 under
+0.01); depth reads: the control recalls tier 100 in pass 2 (0.348 / 0.464 / 0.467), arm P flat.
The verdict rows of the PILOT.md decision table waited for the user (decided 2026-10-17, below). A research round on 2026-10-16
([review_2026-10-16.md](docs/review_2026-10-16.md)) found the nearest relatives (LMLM, Co-LMLM)
reading the same residual against a standard model, the chain read matching the slow phase before
an abrupt in-context two-hop transition, from-scratch two-hop composing only under looping after
grokking-scale training, and Co-LMLM's in-batch InfoNCE on the model's own query as the A7 recall fix;
seven directions were proposed there. **Decided 2026-10-17 (the user): A1(b) passes with the
leak known and its criterion amended to the paired reading (PILOT.md read table); L1 is failed at
this shape unless the chain continuation of arm P shows composition; the retriever fix is built
(`evidence_contrastive_loss`, weight 0 by default, the port bullet below); A3's behaviour goes to
6f's RL, marked the RL step that must not fail; queue: chain arm, InfoNCE micro arm, R7, R8, R9.**
**The chain continuation ran and was read 2026-10-17
([chain_continuation_pilot.md](docs/measurements/chain_continuation_pilot.md)): L1 is failed at
this shape.** Arm P's final for one epoch of `chains_train` alone (`ckpts/evidence_pilot_chains/`,
102.66M tokens in 5.1 hours, batch 4 x 2 after batch 8 spilled, saves every 10M, reads
`ckpts/inject/pilot_ch_{TAG}_chains_*`): held-out answer CE 0.315 to 0.201 in a smooth decay,
1-hop 0.99 at the final, every composition cell at chance at depths 1 to 4 (held-out 2-hop paired
Delta +0.004 +- 0.008, composed answers 1.93 nats against ln K 1.72), loop scales healthy. **The
InfoNCE micro arm against its twin, one epoch of the QA slice
([infonce_micro.md](docs/measurements/infonce_micro.md), `ckpts/inject/launch_infonce.sh`,
35M tokens each): store recall@5 0.012 against 0.000, recall@20 0.035 against 0.001 (bge
0.829), language and selection equal; the term moves recall from zero to small.** **The same
pair at three epochs (`w25e3`, `w0e3`, 104M tokens each, read 2026-10-18): recall@5 0.059
against 0.000, recall@20 0.133 against 0.002 (one sixth of bge), at a cost of 0.07 nats of
held-out language on the arm (4.45 against 4.38, the gap widening with dose), selection 0.01
better, the port read on neither.** **Decided 2026-10-21 (the user): the term enters the
pilot-shape recipe at 0.25, its language cost read on one pilot-size arm before the real-run
spec commits to it.** **Arm PI (`config_pilot_infonce.yaml`: arm P's recipe with the term at
0.25 and loss-free balancing; `ckpts/evidence_pilot_pi`, `ckpts/inject/pilot_pi.log`) was
stopped at 87M the same day on the language bar:** held-out CE 0.03 / 0.11 / 0.30 nats behind
arm P at steps 890 / 1770 / 2660 against the pilot's noise of 0.013, load fields even, store
recall@5 0.006 on the 75M save against arm P's 0.004, the port read lagging arm P's (content
gain +0.43 against +1.76 at step 2660). The term as built (trunk gradient, every loop) stays out
of the real-run spec; next were an offline probe of the hidden state and one head-only micro arm
(trunk gradient stopped, loop 1 only; both read the same day, below), a separate search tower the other direction (held
2026-10-21). A
research round on
training the model's own retrieval without the language cost started the same day (write-up
written the same day, [review_2026-10-21_retriever.md](docs/review_2026-10-21_retriever.md): term at loop 1 only, a stopped trunk gradient, bge query distillation and a key bank as the cheap arms, an offline probe of the hidden state first); the model-keyed retriever variant stays held.
**The query head probe, read 2026-10-21 ([query_head_probe.md](docs/measurements/query_head_probe.md),
`scripts/probe_query_head.py`, eval only), caps a head on the frozen state:** ridge and MLP maps
from the IR expert's loop 1 input onto bge's question vector reach at best recall@5 / @20 0.025 /
0.055 on `w0e3` and 0.055 / 0.122 on `w25e3` (mean and last token concatenated, MLP), under 7% of
bge at k5 at any position or pooling; the term's gain split between the head and the trunk state
(a head on the untouched trunk reaches about 0.025 at k5, on the term-trained trunk about 0.055,
the model's own query 0.059), so the head-only arm reads what the head alone gets from an
untouched trunk, and the trunk's share cost 0.30 nats at pilot shape; a tower on detached states is capped the same way, so a working tower would be a small
bge-style encoder over the prompt text beside the model (held, decided 2026-10-21, below).
**The head-only micro arm, read the same day ([infonce_micro.md](docs/measurements/infonce_micro.md), `bash
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
([depth_memory_probe.md](docs/measurements/depth_memory_probe.md)): depth 5 is trainable at pilot
shape with full gradient at batch 4 x 2 (17.3 GB, about 31 GB at batch 8 over the card, 1.27x
the time per token of depth 3), and the detached prefix (`detach_loops`, built the same day)
fits at 21.5 GB but cuts the prelude off from the language loss until a straight-through route
exists; nothing changes in the real-run spec.** **The Muon rung's three micro arms ran and were read 2026-10-18
([muon_micro.md](docs/measurements/muon_micro.md)): the loss bar passes (Muon 0.07 to 0.15
nats below its AdamW twin on both recipes, z 16 to 25 paired per token, 2 to 4x the one seed
pair's spread of 0.03), copying is no later (both warm-up arms switch between 50M and 100M,
Muon copies held out at 0.90 against 0.67), and the leak passes by the letter (largest cell
z 2.6, tier 1000 in distribution) while worse by the sign (1.6x at tier 1000 and 2.4x at tier
100 in distribution, held out equal); on arm (a) Muon stores the biographies about twice as
early and more at every tier above 1, far outside the seed spread, the wrong direction for the
goal. Decided 2026-10-21 (the user): Muon is out of the real-run spec, the pilot-shape arm is
not run, the real run stays on AdamW.** Every save of the chain continuation is read (2026-10-19, the per-save table in its
record): 1-hop 0.82 to 0.99 in a smooth climb, every composition cell at chance at all eleven
reads, the paired held-out Delta within one sigma throughout; no transition anywhere inside the
epoch. **R8 (loss-free balancing, the Config bullet) run and read 2026-10-19 and 2026-10-20
([lossfree_micro.md](docs/measurements/lossfree_micro.md)): loss-free passes on all three
bars.** Held-out CE 3.9055 against arm (a)'s 3.9037 (paired at exit 3 -0.0055,
z -1.8, inside the 0.029 seed spread), storage inside the two AdamW seeds cell for cell, MLP
load max over mean 1.03 to 1.05 per pass; the logged aux value saturates in bf16 (hard counts
stop at 256) and reads the batch, so arm (a)'s aux term was a floor, not a balancer. Fixed
2026-10-20 (fp32 counts in `compute_aux_loss`, `test_evidence_selector.py`); the comparison half
of the load bar was read on an arm (a) rerun under the fixed term on 2026-10-20 (`bash
ckpts/inject/launch_auxfix.sh`, `ckpts/evidence_auxfix_full`, `auxfix_full.log`, reads
`ckpts/inject/auxfix_full_{TAG}_{rank_none,exit}` paired against arm (a)). **Loss-free is better
on all three bars:** the working aux term ends at CE 3.9142 (paired -0.0076 at exit 3, z -2.3),
puts 0.35 to 0.54 of the picks on the dense slots with the MLP load at 1.08 to 1.26, and stores
the biographies more in distribution (tier 100 z 14.5 against arm (a), z 14.4 against the
reseed, far outside the seed spread). **Decided 2026-10-21 (the user): loss-free balancing
enters the real-run spec.** Decided the same day (the user): the 100B-token corpus build runs on
the rented cluster, not locally, and the throughput work starts only after every other spec
decision is made, so nothing optimized has to be redone.
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
two arms; open for the user: build it, or read one arm of the three-sublayer core against today's
one block. **Arm E4 was launched 2026-10-23 17:09** (read 2026-10-24, below; `bash ckpts/inject/launch_pilot_e4.sh`,
`ckpts/inject/config_pilot_e4.yaml`: `config_pilot.yaml` with `num_mlp_experts: 4` and
`moe_balancing: loss_free`; seed `ckpts/inject/seed_pilot_e4_grounded.pt` from
`init_scratch_seed.py` seed 0 then `migrate_groundedness_head.py`, 89.6M total, 84.9M active, arm
P's active compute; run directory `ckpts/evidence_pilot_e4`, log `ckpts/inject/pilot_e4.log`; arm
P's recipe: the warm-up on `pilot_warm_train` stopped after the 300M save by `stop_after_save.sh e4
'30[0-4]M'`, then `SPLIT=pilot_main_train bash ckpts/inject/launch_pilot_e4.sh`; reads `bash
ckpts/inject/read_pilot_save.sh CKPT e4 TAG`, which loads the e4 yaml): 44k to 48k tokens per
second, peak 26.3 GB, step 0 evals neutral. Read against arm P on held-out CE on `inject_val`,
closed-book rank by tier on both forms, the 1-hop gate and L1, the port read and the load fields;
caveat on record: E4 runs loss-free where arm P ran under the aux term the bf16 saturation had made
a constant, two changes, and at micro loss-free alone moved CE and storage inside the seed spread,
so a difference is read as the expert count with that caveat. **R9 built 2026-10-23** (code by a
subagent, review pending at the time; run 2026-10-24, below): `prepare_chain_data.py --bridge` renders the assistant turn
as "Step 1: e1. Step 2: e2. Answer: ek." (1 hop: "Answer: e1.") with every buffer and sidecar byte
identical to the plain build and `bridge` in the meta; `eval_chains.py --generate-bridge
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
**Arm E4 complete and read 2026-10-24 ([expert_count_pilot.md](docs/measurements/expert_count_pilot.md)):
halving the routed experts from 8 to 4 at top 2 (10M fewer parameters, the same active compute)
costs nothing measurable at this shape and dose.** 858.40M tokens, one epoch (killed once at about
680M on the user's request on 2026-10-23 and resumed from the 652M save, which replays the same
documents; the 675M eval reproduced within 0.001; final `ckpts/evidence_pilot_e4/checkpoint_evidence_final.pt`,
reads `ckpts/inject/pilot_e4_{300M,final}_*`, `compare_pilot_e4_vs_p.log`). Held-out CE 3.2396
against arm P's 3.2321, E4 above at every read by 0.001 to 0.019, under the pilot's noise of 0.013
at every read but 25M and 276M; the switch criterion met at 300M as on arm P; copying equal (gold
top-1 1.000 in distribution, 0.96 to 0.97 held out; swapped followed 1.000 / 0.91 to 0.93); the
1-hop gate 0.875 against 0.848, every composition cell at chance, so L1 fails as on arm P; port
gold gain +4.19, content +3.82, grounded AUROC 0.972 against +4.00, +3.97, 0.994; the leak equal at
tier 100 (entity `delta` 0.017 / 0.012 by form against 0.010 / 0.013, paired z -1.2) and smaller at
tier 1000 (0.053 / 0.041 against 0.064 / 0.046; paired raw rank z -3.8, about one sigma on
`delta`, since E4's prior ranks worse too); MLP load max over mean 1.01 to 1.06 per pass. Caveat
on record: two changes (loss-free against the saturated aux term) and one seed each. For shape L
the expert count is a cost choice, not a quality one, until a read at the real run's size says
otherwise; R4b's sublayer half stays open. **R9's bridge decode smoke ran 2026-10-23** on arm P's
final (8 questions per hop, `ckpts/inject/bridge_smoke.json`, moved 2026-10-24 to
`ckpts/inject/bridge_prefix_bug/`: decoded before the packer fix `af24c70`): the decode runs end to end and
writes entity-like text, marker rate 0 as expected on a model that never saw the form. **R9's SFT
ran 2026-10-24** (`bash ckpts/inject/launch_pilot_bridge.sh`, `ckpts/evidence_pilot_bridge`,
log `ckpts/inject/pilot_bridge.log`; the bridge split 500k questions, 33.7M prompt tokens per pass
at evidence ratio 3.26, two passes; record below).
Intermediate pilot saves deleted
2026-10-17 (kept: both finals, arm P 300M and 530M, arm C 325M). `stop_after_save.sh ARM
PATTERN` drops STOP after a matching save; a stop leaves `STOP` to remove before relaunch.
The plan was rewritten on 2026-09-30 around the goal "facts in the store through the retrieval
pathway, reasoning in the looped trunk"; the
design is `docs/evidence_path_design.html`, the findings are `docs/review_2026-09-29.md`, the R0b
record is [r0b_micro_pilot.md](docs/measurements/r0b_micro_pilot.md).

- **R9, the written bridge, ran 2026-10-24; its 30M read meets the bar** (record in NEXT.md
  "Now"). From arm P's final on the 500k-question bridge split, two passes, target about 72M,
  trained to its end at 17:18 at 66.35M tokens in 184.5 minutes (exit 0; `ckpts/evidence_pilot_bridge`,
  `ckpts/inject/pilot_bridge.log`); reads by `ckpts/inject/read_bridge_save.sh`, the reads decoded
  before `af24c70` moved to `ckpts/inject/bridge_prefix_bug/`. The 30M read
  (`ckpts/inject/pilot_bridge_30M_gen_d3.log`, `_hop4.log`), `chains_eval` at depth 3: one read
  site 0.998 / 0.992 / 0.980 at 1 / 2 / 3 hops (chance 0.115 / 0.185 / 0.206), three sites 0.999 /
  0.994 / 0.988; held-out 2-hop 0.989 at one site, 0.993 at three (chance 0.191), paired kept 3
  minus kept 1 +0.004 +- 0.002; no read site at chance (0.132 / 0.199 / 0.220); marker written on
  every question with a site kept, intermediate hop hit rate 0.98 to 0.99 (the 40M read: three
  sites 1.000 / 0.997 / 0.996, no site 0.137 / 0.192 / 0.192). The 4-hop split at depth 4 reads
  0.176 against 0.210 (z -2.6): the model writes exactly two steps and then "Answer:" on 999 of 1000
  questions (trained on hops 1 to 3), not a decode truncation (bridges of 52 to 88 characters
  against the 48-token cap). The log's `instrument valid: no` and four FAIL lines are the latent
  read's sub-hop rule in a mode it does not fit (their suppression in that mode is held). Bar met at 30M on
  this read (the final read below).
- **The repository review, 2026-10-24** ([review_2026-10-24.md](docs/review_2026-10-24.md)):
  externalization demonstrated at pilot shape against the matched control with the tier 1000
  residual known; reasoning composes in written tokens, not the latent loop, at this shape; the
  retriever and abstention verdicts rest on flawed reads. Facts on record: every generation read
  with evidence before `af24c70` read a shifted buffer (the A3 generation read, the A7 pathway EM,
  the A4 edit flip rate, the pre-fix bridge smoke); arm C alone trained on 12,240 documents, 50.11M
  tokens, of 4094-token QA rows and finished its cosine at the 6.0e-5 floor where the evidence arms
  stopped at 6.55e-5; about 7.4% of `inject_val` tokens appear verbatim as supervised text both arms
  train on, another 6.7% in rows only arm C trains on; logged CE on `inject_val` accumulates in bf16
  (about 0.002); a 2-hop shortcut scores 0.256 against chance 0.185. The person-clustered sigma,
  estimated up to 1.7x, was run 2026-10-24 on the pilot rank JSONs: sigma within 10% (arm P held-out
  tier 100 z 3.2 item and 3.2 clustered; in distribution 2.2 and 2.3; tier 1000 7.2 and 8.0 / 5.5
  and 5.9), the decided reading stands. Landed 2026-10-24 (default behaviour byte identical): the
  splits, fp32 masters and RNG states in the checkpoint, `--switch-split`, the fixed save and eval
  grid, a finished run exiting 0, the `evidence_cu_seqlens` order assert, the `n_valid` syncs
  removed, the
  clustered sigma and paired `delta` lines, the shortcut baseline, the merge tool's over-limit
  report, the training-distractor gold exclusion (next build only), `migrate_groundedness_head.py`
  on pre-2026-10-19 sources. Held then: the rest, landed the same evening but for the eval-pass
  rewrite (next entry). The bridge eval's two items (verdict lines suppressed, evidence built once
  per batch) were first recorded here as landed; `scripts/eval_chains.py` is unchanged since
  `fee6a5c`, so both stay held. Its plan proposals are a NEXT.md Decisions
  row ("repository review"), waiting on the user.
- **The eval packer bug, found and fixed 2026-10-24**
  ([written_bridge_pilot.md](docs/measurements/written_bridge_pilot.md), record in NEXT.md):
  `_pack_evidence_batch` put a row's own evidence before the dump width, unsorted by segment id
  under left padding, so `evidence_cu_seqlens` gave every left padded decode with evidence a
  buffer shifted by the dump width. Void: the A3 generation, A7 pathway and A4 edit reads of
  2026-10-14, the bridge smoke and the early bridge reads (`ckpts/inject/bridge_prefix_bug/`).
  Likelihood reads, the dataset's packing and `inference.py` unaffected. Found on R9's 30M save
  (teacher forced bridge argmax 100% right padded against 69.6% left padded); fixed with a sorted
  layout and an assert, `tests/test_evidence_pack_layout.py`, Sonnet review clean. Re-read on arm
  P's final (`ladder_pilot_redo.sh`, `pilot_p_ladder2_*`): A3 weak in generation (abstention 0.770
  under gold against 0.956, EM 0.046; likelihood identical), A7's pathway gate still fails (model
  query EM 0.001 against oracle 0.088, bge 0.044; the void read's oracle 0.026); the A4 flip read
  still unreadable on the model's query route (1 of 2,000 answered before the edit, flip 0 of 1;
  `pilot_p_ladder2_store_edit.*`), a flip read through bge needs `eval_store.py edit` extended.
- **R9, the written bridge, read 2026-10-24: passes by a wide margin**
  ([written_bridge_pilot.md](docs/measurements/written_bridge_pilot.md)). 66.35M tokens in 184.5
  minutes; held-out bridge CE 4.895 to 0.0004. On the fixed packer, held-out 2-hop 0.982 at 10M,
  0.997 at the final (chance 0.191, z 64.7); final `chains_eval` 0.998 / 0.998 / 0.997 at 1 / 2 / 3
  hops, no read site at chance, marker and hit rates 1.00, paired held-out Delta +0.002 +- 0.001;
  4-hop 0.206 against 0.210 (the model writes the two steps it trained on). Verdict: the model
  composes as soon as the intermediate entity is a token it writes and reads back, on unseen
  compositions, after 20M tokens on the form; the loop never did this in silence at this size. 6f
  reasons in written tokens; R4's corner recommended dropped (open for the user); the real run's
  mix needs multi-hop rows at the serving depth; transfer to real questions and retrieved chunks
  unmeasured.
- **The review's code changes, landed 2026-10-24 (evening)** (uncommitted; tests pass on the GPU; a
  detached Sonnet review found no high or medium issue; decided by the user on the review's
  recommendations). `mtp.py`: the unweighted CE, top-1 and `p_max` sums accumulate in fp32 (the
  logged CE leaves its 1/64-nat grid, `top1_acc` cannot pass 1; `tests/test_chunked_ce_fp32.py`).
  `moe.py`: under loss-free balancing the selection scores are formed in fp32 for the topk (a zero
  bias is bit-identical; at bias 0.30 the bf16 sum picked differently on 2% of tokens); the frozen
  chunk gate is skipped (no selector mass, scatter or multiply by ones per loop; the zero test is
  cached against the scale's `_version`, no per-step sync; forward and key/value reader gradients
  bit-identical, `tests/test_evidence_gate_skip.py`); `ParallelSparseMoELayer.forward` delegates to
  `experts.grouped_mlp_forward` with `null_mask=~mlp_mask` (dense-slot picks sorted after the real
  experts, counted in the same `bincount().tolist()` sync and sliced off before the grouped GEMM;
  exact zeros as before, outputs differ only by bf16 summation order; about 5% of the MoE layer's
  time at micro shape, more at larger expert widths; `tests/test_efficiency_2026_10_24.py`).
  `transformer.py`: evidence padding cut into 128-token blocks (`EVIDENCE_PAD_SEGMENT`) before the
  encoder's `cu_seqlens` (real-token states and logits bit-identical; at pilot shape 9.61 to 7.57 ms
  on a 4608/600/600/600 batch, no change when every row is full); `groundedness_term` detaches the
  head input under the key/value reader (the head still learns, no trunk parameter gets a gradient
  from it; the evidence-only signal of the review's finding 9 is not built); the cross reader's
  rotary gather skipped under the key/value reader; the eval path norms only the last loop when
  `return_hidden` is False. `evidence_selection_loss`: log-space BCE with no clamp on the
  renormalized share, rows with fewer than two visible chunks excluded, and
  `evidence_selection_weighting` (`token` default, `segment`) plumbed through
  `moe.evidence_selection_term(segment_ids=, num_segments=)` from the `pretrain.py` call site
  (`tests/test_evidence_selector.py`). `evidence_contrastive_loops` falls back to `[0]` in
  `config.py` and the three yamls (null still accepted; the recorded arms' gitignored configs under
  `ckpts/inject/` are pinned to null). `sft.py estimate_packed_rows` replays each reader's rule
  (evidence reader `len + mtp <= max_length - 1`, plain reader `<= max_length`;
  `tests/test_packed_rows_estimate.py`): `pilot_main_train` estimates 241,398 rows (about 30.17k
  steps) against the old 258,852 (32,356), matching arm P's observed epoch end at about 30.3k; the
  control's 252,912 unchanged. `prepare_evidence_data.py`: the training cap is 4093
  (`MAX_DOC_TOKENS_TRAIN`), the held-out cap 4094 unchanged. `evidence_dataset.py` logs the
  documents and tokens over the row limit at startup and per worker. The reader itself is unchanged
  (decided 2026-10-24: future evidence arms stay matched to the arms already run on the existing
  builds; the existing pilot merges keep dropping the 12,240 rows). `prepare_injection_data.py`
  gains two flags, default off and byte identical at the default, each changing only `.mask`:
  `--mask-first-name-mention` (the first run of name tokens per document gets mask 0; 1,474,885
  tokens of 300M on the micro builds) and `--support-gold-only` (gold-absent documents count no
  value as supported); built the same day `inject_retrieval_s100a50m1_train`, `s100a50g1`,
  `s15a50m1`, `s15a50g1` (44 of 44 companion files byte identical to the flagless builds;
  `ckpts/inject/build_leak_arms.sh`, log `build_leak_arms.log`). `scripts/prepare_clean_val.py`
  (`tests/test_prepare_clean_val.py`) built `data/prepared_pilot/inject_val_clean`: 974 windows,
  650,125 tokens, from `inject_val`'s windows whose documents the evidence QA builder never used
  (the first 80,672 `ir.bin` documents excluded; phase1 and phase2 kept); verbatim 64-token overlap
  with the QA training rows' supervised tokens 0.99% against 13.3% for `inject_val` under the same
  check (the review's 7.4% was a narrower definition); read by `ckpts/inject/read_clean_val.sh`, not
  yet run. Held: the eval-pass rewrite (deferred to 6a) and the two `eval_chains.py` bridge-mode
  items. Launchers: `ckpts/inject/launch_review_arms.sh [w0e3b|l0e3|seg_e3]` (launched 2026-10-24
  about 20:25, running) and `ckpts/inject/launch_leak_arm.sh m1|g1` (queued after it).

- **The former "Next, in order" list, closed 2026-10-24** (items 1 to 4 done or closed, item 5
  moved to the Open items table; the full ladder with its done steps is in NEXT.md):
  1. Done 2026-10-09 (spec) and 2026-10-17 (verdicts): the pilot spec, with the copy-first warm-up requirement, swap rate 0.15 and no further lever
     against the leak (decided 2026-10-06 by the user: a leak this size matters little for the
     pilot, and RL planned later can act on small leaks; caveat on record: RL is Parked and shapes
     behaviour at conflicts rather than removing what is stored, so for small leaks the goal is read
     as behavioural at conflicts). The chain slice stays 10% at hops 25 / 50 / 25 and the pilot's
     L1 is the deciding read on the loop clause.
  2. Done 2026-10-14: the pilot (Phase 5) on that spec with the key/value reader, ladder cut to
     budget (the cut superseded 2026-10-23 by the run order).
  3. Closed 2026-10-21: rung R7 (proposed 2026-10-12, NEXT.md Decisions "optimizer"): Muon on the hidden matrices
     for the real run, AdamW for `E`, the head adapters, the router and every norm, scalar and
     vector; then two micro arms against their AdamW twins on the R0b recipe, then one
     pilot-shape arm against arm P. Decided on the leak, not the loss. The pilot arms stay AdamW.
     **Built 2026-10-18**: `modules/model/optim.py` (`MuonAdamW`, the partition rule
     `muon_kind`, `build_optimizer`), the `training: optimizer: adamw | muon` key read by both
     trainers, `eval_exit.py --pair TWIN` (paired per-token CE per exit on the same tokens),
     `tests/test_muon_optimizer.py`; the sampled max q·k logit per layer is not built (flash
     varlen exposes no logits; it needs a sampled recompute). Arms: `bash
     ckpts/inject/launch_muon.sh [full|cf|reseed]` (`config_micro_muon.yaml`,
     `config_micro_seed4321.yaml`; run directories `ckpts/evidence_muon_full`,
     `evidence_muon_kv_cf`, `evidence_adamw_full_s4321`; reads `ckpts/inject/muon_{ARM}_{TAG}_*`,
     the exit read paired against the twin's save at the same tokens), **run and read
     2026-10-18** ([muon_micro.md](docs/measurements/muon_micro.md)): loss and copying bars
     pass, the leak bar passes by the letter and is worse by the sign, storage twice as early
     on arm (a). A checkpoint resumes only under its own optimizer. Decided 2026-10-21 (the
     user): Muon dropped, the real run stays on AdamW.
  4. Closed 2026-10-21 for R8 (in the spec), R9 run 2026-10-24, 6f open. Proposed 2026-10-12:
     rung R8 (loss-free MoE balancing against the aux loss, micro; **built
     2026-10-19**: `moe_balancing: loss_free` in the Config bullet below, arm `bash
     ckpts/inject/launch_lossfree.sh` on the arm (a) recipe from `seed_micro.pt` against
     `ckpts/evidence_inject_full`, run directory `ckpts/evidence_lossfree_full`, reads
     `ckpts/inject/lossfree_full_{TAG}_{rank_none,exit}`; bar: CE not worse beyond the paired
     sigma and the seed spread, load at least as even; **run and read 2026-10-19 and
     2026-10-20**: passes for loss-free on all three bars against the fixed aux term, which
     stores more and balances the dense slots; decided 2026-10-21 (the user): loss-free enters
     the real-run spec), rung R9
     (a written bridge on arm P's chains: does reasoning in written tokens compose where the loop
     did not), and NEXT.md 6f, reasoning post-training after the real run (SFT, on-policy
     distillation, RL past the pass@8 gate with credit over passes, depth record and routing
     replay; prerequisite: decode with evidence through a KV cache).
  5. Now the Open items table. Run order decided 2026-10-23 (the user): R4b's expert arm (arm E4, run and read 2026-10-24,
     [expert_count_pilot.md](docs/measurements/expert_count_pilot.md): 4 experts cost nothing
     measurable against 8 at the same active compute), then R9 (built 2026-10-23:
     `prepare_chain_data.py --bridge`, `eval_chains.py --generate-bridge`,
     `build_chains_bridge.sh`, `launch_pilot_bridge.sh`; SFT run and read 2026-10-24,
     [written_bridge_pilot.md](docs/measurements/written_bridge_pilot.md): passes by a wide margin,
     held-out 2-hop 0.997 against chance 0.191, composition in written tokens, not the loop), whose
     read gates R4's corner arm (R4 staged: arm P the baseline cell, the far corner with the
     unbuilt coda first, five arms at most; recommended dropped on R9's read; dropped 2026-10-24 by the user, obsolete on R9), then R6 (skipped 2026-10-24 by the user); R4b's sublayer half needs the unbuilt
     three-sublayer core, its build open for the user.
  Spec written 2026-10-09 ([PILOT.md](docs/plans/PILOT.md)); build list done 2026-10-10 (its
  "Build" section). Launch: `bash ckpts/inject/launch_pilot_p.sh` (warm-up split; after the switch
  criterion holds, `SPLIT=pilot_main_train bash ckpts/inject/launch_pilot_p.sh` resumes the same
  run directory) and `bash ckpts/inject/launch_pilot_c.sh`, each in the background under a Monitor
  watch; reads per save with `bash ckpts/inject/read_pilot_save.sh CKPT p|c|pi|e4 TAG [0|1]`. Arm P's
  own saves timed the read set (no probe save existed). Since 2026-10-24 the switch relaunch
  passes `--switch-split` (a relaunch on another split is refused without it). No change to `modules/`.
- Pilot corpus (2026-10-10, `data/prepared_pilot`, built by `ckpts/inject/build_pilot_bios.sh`,
  `merge_pilot.sh`, `merge_pilot2.sh`, log `merge_pilot.log`): biographies at seq 4096 and 600M
  tokens (974,525 documents, filler repeat 2.18x, bio share 2.3%; store `data/index/pilot_bios`,
  facts, pools and `inject_val` byte identical to the micro build); merges `pilot_warm_train`
  (bios s100a50), `pilot_main_train` (bios s15), `pilot_control_train` (bios full, no evidence
  files): 3,488,744 documents, 910.2M prompt tokens each (bios 599.6M, QA 210.6M at 2 passes,
  chains 100.0M), evidence 652M tokens at ratio 0.717 (chains 419.6M at 4.20, QA 202.7M at 0.96,
  bios 30M at 0.05), one shared document order (`--order-from`), QA and chain subsequences md5
  identical across the three (`md5_by_slice` in `{split}.merge.json`); `evidence_fixed.*` copied
  with its `.src`. `sft.py` fits the cosine to the split, so 910M is the per-arm budget, not the
  yaml's 1B. The merge tool: `python scripts/merge_evidence_splits.py --out-dir D --split S
  --target-tokens N --seed 42 --slice label=dir/split:share[:passes] ... [--no-evidence]
  [--max-chunks-per-doc label=K] [--order-from other.slice]`.
- Pilot shape probe (2026-10-09, `ckpts/inject/config_pilot_probe*.yaml`, logs
  `ckpts/inject/pilot_probe*.log`): 99.1M total, 84.9M active, 249M FLOP/token at seq 4096, 39.4M
  per evidence token. Batch 8 x 4096 x 1 on `inject_retrieval_train`: 44k to 50k tok/s, 20.4 GB;
  batch 4 x 2: 41k to 45k, 12.7 GB. On `evidence_train` (ratio 3.75, cap 4608) rows close on the
  evidence budget at fill 17 to 30% and the rate is 2.7k to 3.9k, GPU at 99%: the evidence ratio
  sets the rate, not the batch; the pilot's merged corpus (ratio about 0.73) fits to about 16k.
- Micro runs: 55k to 60k tok/s, 13.3 GB peak with the card buffer, `--batch-size 1024` for
  `closed_book_rank.py`. The batch move to 32 x 1 changed no tokens per update; if copying ever
  emerges late, the lever is fewer tokens per update (16 x 1). Relaunch a stopped arm with the same
  `-c` and every flag (a different split than the checkpoint recorded needs `--switch-split`,
  since 2026-10-24); a stop leaves `STOP` in the run directory. `ckpts/instrsmoke.log` stays the
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
- **One command per tool call.** Do not chain commands with `&&`, `;` or pipes into `tail`,
  `grep` and the like beyond the `cd ... && source env_init && python ...` wrapper above. Chained
  lines cannot be auto-approved and hide which step failed; independent commands go in separate
  calls (user request 2026-10-10).
- **Every training launch (`sft.py`, `pretrain.py`, `run_training.py`) runs in the background
  under a Monitor watch**, however short. The filter matches failures and progress:
  `Traceback|Error|Killed|OOM|assert|device not ready` plus `Step` / `Tokens/sec`, and for
  `--evidence` also `eval fixed|KILL|kill check`. Tee the watched output to a gitignored `*.log`
  so the eval blocks survive a crash.
  **The watch never owns the process** (2026-10-24: a Monitor kills its command when it expires,
  after 30 minutes at most, and took a launcher and its trainer down at 74M): launch detached
  (`setsid nohup bash launcher.sh > launcher.log 2>&1 < /dev/null & disown`) and watch with
  `tail -n 0 -F` on the log, re-armed on every expiry.
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

- **Delegate the routine work to Opus subagents** (user request 2026-10-19, to keep the main
  model's usage down): log and JSON extraction, measurement notes and tables, the plan-file and
  design-page mirroring of a result, launcher and config files, and code builds against a
  written spec. The main thread keeps the decisions, the final reading of a result, the launches
  under a watch and the handovers. Detached reviews stay on Sonnet (below). Every subagent runs
  caveman full or above (`/caveman:caveman full`, `ultra` for a pure lookup); say so in the
  spawn prompt, the mode is not inherited. Caveman is for the report only: code, comments and
  docs are plain English in the repo's style.
- **Watches wake the main thread once per event**, so a training watch filters to evals, saves,
  failures, the trainer's exit and the reads, never the per-step or per-thousand-step lines
  (user request 2026-10-19); the rate is checked once after launch and on a stall.
- Subagents share one checkout and cannot coordinate, so they never run git that changes the
  working tree or index (`stash`, `reset`, `checkout`, `restore`, `switch`, `clean`, `add`,
  `commit`, ...). Read-only git is fine; baselines come from `git show <rev>:<path>`. The main
  agent may still use them.
- Each task is self-contained and finishes fast. Bundle work that shares a file into one agent and
  name its owned files; split genuinely independent work across parallel agents in one message.
- **Every substantial or architectural change gets a Sonnet subagent review before the commit
  line is handed over** (user request 2026-10-18): a new module, a loss term, an optimizer, a
  tensor or checkpoint format change, a trainer path change. Give the reviewer the diff scope and
  the invariants that bite; fold its findings in; say in the reply that it ran and what it found.
  Small edits, docs and config values are exempt. The reviewer is fully detached: a
  self-contained brief, no conversation history, and told not to read the memory directory, any
  transcript or task output, or anything outside the repo.

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
config_micro.yaml, config_pilot.yaml  the micro (35M) and pilot (99.1M) shapes, picked by TINY_LLM_CONFIG
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
  prepare_clean_val.py       inject_val's filler windows minus those whose document the evidence builder read -> data/prepared_pilot/inject_val_clean
  merge_evidence_splits.py   merges evidence-format splits by token share into one split (the pilot corpus)
  build_store.py      retrieval stores (NQ, TriviaQA, HotpotQA) and edit stores -> data/index/<name>/
  init_scratch_seed.py       random-init seed for a from-scratch sft.py run (TINY_LLM_CONFIG picks the shape)
  migrate_*.py        phase0, ir_reshape (--arm random|warm), loop_inject, evidence_port, groundedness_head
  eval_abstention.py  THE acceptance metric (SQuAD v2); --evidence-port scores gold/none/distractors/mixed/counterfactual
  eval_benchmarks.py  fixed 13-task suite, one scoring path for this model and the peers; popqa is a rank task outside `all`
  closed_book_rank.py likelihood rank of the gold among same-type candidates (bios: by exposure tier; compare: paired)
  eval_chains.py      accuracy by hops and by kept read sites (reader_sites_kept) at fixed depth; --generate-bridge decodes the written bridge, then scores
  eval_exit.py        per-exit CE without evidence: paired gains, oracle and beta optimum bounds, confidence exit vs fixed depths
  eval_store.py       store recall of the model's query vs bge, open-book pathway EM vs oracle, store-edit flip rate
  probe_query_head.py ridge and MLP heads from the frozen IR input onto bge's question vector, scored as store recall
  probe_depth_memory.py peak memory and time of one training step per depth:detach_loops, one child process per setting
  entity_frequency.py exact token-sequence counts in a .bin (the counterfactual strata)
  eval_calibration.py eval_probe.py eval_stage0.py evidence_ceiling_probe.py (--fixed-split: ceiling on evidence_fixed)
  inference.py gradio_app.py prune_vocab.py fetch_tokenizer.py setup.sh onstart.sh run_sft_after_pretrain.sh
modules/model/        transformer gemma4 moe router experts information_retrieval evidence mtp attention kv_cache optim
                      embeddings (rotary) modules (SmallLMHead) utils (EncoderOutput, causal mask)
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
python scripts/eval_benchmarks.py --peer pythia-410m --validate --json-out docs/measurements/benchmarks/peer.pythia-410m.json
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
python scripts/prepare_injection_data.py --out-dir data/prepared_inject --target-tokens 300000000 --filler-phases ir,phase1,phase2 --seq-length 1024 --swap-rate 1.0 --anon-rate 0.5 --gold-drop-rate 0.2 --distractors 3 --seed 42 --arms retrieval --suffix s100a50m1 --mask-first-name-mention --device cuda   # a leak arm's warm-up split; also s15a50m1 at swap 0.15, and --support-gold-only for s100a50g1 and s15a50g1 (all four: bash ckpts/inject/build_leak_arms.sh)
bash ckpts/inject/launch_leak_arm.sh m1|g1   # the copy-first warm-up recipe on a flag's splits, stop at 100M, read, switch, read the final and compare against the warm-up arm
bash ckpts/inject/launch_review_arms.sh [w0e3b|l0e3|seg_e3]   # the micro retriever arms of the 2026-10-24 review, then store recall at loop 1
python scripts/prepare_clean_val.py [--check-against PREFIX --check-baseline data/prepared_pilot/inject_val]   # inject_val_clean: inject_val's windows from documents the evidence builder never used
bash ckpts/inject/read_clean_val.sh   # eval_exit.py on the arm P, C and E4 finals on inject_val_clean and inject_val
TINY_LLM_CONFIG=config_micro.yaml python scripts/sft.py --evidence --reader-kv -c ckpts/inject/seed_micro.pt --run-name inject_retrieval_kv_cf --data-dir data/prepared_inject --train-split inject_retrieval_s100a50_train --val-split inject_val   # warm-up to 100M, then the same line with --train-split inject_retrieval_train --switch-split
TINY_LLM_CONFIG=config_micro.yaml python scripts/closed_book_rank.py bios -c CKPT --form both --evidence none|gold|swapped|prompt --batch-size 1024 [--n-loops N] --json-out ckpts/inject/rank_full.json
python scripts/closed_book_rank.py compare rank_full.json rank_masked.json rank_retrieval.json
TINY_LLM_CONFIG=config_micro.yaml python scripts/eval_exit.py -c CKPT --data-dir data/prepared_inject --split inject_val --max-batches 40 --json-out OUT.json
python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains --splits eval,heldout_tmpl,hop4 --eval-questions-per-hop 1000 --seed 42 --device cpu   # train,val on cuda
python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains --splits train,val --train-questions 2000000 --val-questions 2000 --seed 42 --device cuda
python scripts/eval_chains.py -c CKPT --data-dir data/prepared --splits chains_eval,chains_heldout_tmpl,chains_hop4 --depths 3,4 --sites all --json-out OUT.json
python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains_bridge --splits train,val --train-questions 500000 --val-questions 2000 --seed 42 --device cuda --bridge   # the written bridge form
python scripts/eval_chains.py -c CKPT --data-dir data/prepared --splits chains_eval,chains_heldout_tmpl,chains_hop4 --depths 3,4 --sites all --generate-bridge [--bridge-max-tokens 48] --json-out OUT.json
python scripts/build_store.py --name openqa --sources nq,triviaqa,hotpotqa --max-questions 3000 --chunk-tokens 128 --device cuda --seed 42
python scripts/build_store.py --edit-from openqa --name openqa_edit --edit-fraction 0.5 --neighbour-cos 0.85 --device cpu --seed 42
python scripts/eval_store.py recall|pathway|edit -c CKPT --store data/index/openqa [--edit-store data/index/openqa_edit] --buffer 4 --loop 1 [--route bge] --json-out OUT.json   # edit --route bge: serving time retrieval by the store bge query keys
TINY_LLM_CONFIG=config_micro.yaml python scripts/probe_query_head.py -c CKPT --store data/index/openqa --batch-size 16 --pool last|mean|meanlast --json-out OUT.json
TINY_LLM_CONFIG=config_pilot.yaml python scripts/probe_depth_memory.py -c ckpts/inject/seed_pilot_grounded.pt --settings 3:0,5:0,5:2 --json-out OUT.json   # [--batch-size 4]
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
`test_config_override`, `test_chain_generator`, `test_store`, `test_popqa`, `test_eval_exit`,
`test_evidence_contrastive` (tests 1 to 6; 7 to 11 skip without CUDA and TE), `test_muon_optimizer`,
`test_eval_chains_bridge`, `test_merge_evidence_splits`, `test_dataset_real`, `test_dataset_resume`,
`test_accelerate_batch`, `test_prepare_clean_val` (`test_loss_free_balancing`,
`test_evidence_gate_skip` and `test_efficiency_2026_10_24` need the GPU; `test_eval_chains`,
`test_evidence_pack_layout`, `test_mtp_alignment`, `test_chunked_ce_fp32` and
`test_packed_rows_estimate` run on the CPU but import TE through `eval_abstention.py`, `mtp.py` or
`sft.py`).
`TINY_LLM_CONFIG=path.yaml`
swaps the config yaml for every script (the checkpoint still decides shape and mode on load, so a
micro checkpoint under the default yaml fails the strict load, as it should).

## Config

Tables in [configuration.md](docs/configuration.md).
- `ModelConfig.Params` is splatted into `TinyMoETransformer(**...)`.
- `TrainingConfig` holds **every loss weight** (`lambda_mtp`, `aux_loss_weight`,
  `evidence_selection_weight` + `evidence_selection_weighting` (`token`, default, or `segment`,
  since 2026-10-24), `evidence_contrastive_weight` + `evidence_contrastive_temperature`
  + `evidence_contrastive_loops` (fallback `[0]` since 2026-10-24, null every loop) +
  `evidence_contrastive_trunk_grad`, `groundedness_weight`, `loop_ce_weights` + `loop_ce_subsample`,
  `loop_count_sampling`) and the `optimizer` choice (`adamw`, every run before 2026-10-18, or
  `muon`: `modules/model/optim.py`, Muon on the hidden 2D matrices by the name rule `muon_kind`
  with Nesterov 0.95, five Newton-Schulz steps in bf16 and the update scaled to AdamW's RMS so
  `lr` and `weight_decay` carry over; tables, heads, the router, scalars, vectors and the
  zero-init matrices stay on AdamW with `torch.optim.AdamW`'s arithmetic and state keys, so
  `apply_ir_refresh` still finds `exp_avg`; fused `gate_up` weights are orthogonalized as two
  halves; the momentum buffer stays fp32 across a bf16 resume; a checkpoint resumes only under
  its own optimizer, `check_resume_kind` names both kinds before `load_state_dict`; the step
  loops per tensor with no foreach batching, so time a Muon step at the pilot shape before
  relying on it there, reviewed 2026-10-18) and the `moe_balancing` choice (`aux`, every run
  before 2026-10-19: `aux_loss_weight` times the router's balancing loss; or `loss_free`: no aux
  term, a per-pass selection bias `moe.balance_bias` `[n_loops, num_experts]` on the MLP slots
  moved after each optimizer step by `moe_balance_bias_rate` times the sign of the load error,
  centred over the MLP slots so the update is zero sum and the MLP against dense split stays the
  router's (reviewed 2026-10-19: raw signs under a skewed load drift every MLP bias up against
  the attention and IR slots, whose bias is pinned at zero), `LoopMixtureOfExperts.update_balance_bias`;
  the bias is added to the routing probabilities for the top-k selection only, the weights stay
  the unbiased probabilities, zero is exactly the old router and an absent buffer loads as zero
  through `BALANCE_KEYS` on every load path (the strict resume paths grant this one tolerance
  through `with_balance_default`); the buffer follows the model's bf16 cast, so the update runs
  on an fp32 shadow; counts accumulate on device across micro steps, a pass a sampled depth
  skipped is left alone; `load max/mean` and `dense share` per pass in the step line in every
  balancing mode since 2026-10-20 (the counting is always on, the aux mode passes a zero rate so
  the bias never moves there);
  `tests/test_loss_free_balancing.py`).
  `total_steps = target_tokens // (batch * seq * grad_accum)`.
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
- `num_ir_entries % ir_num_clusters == 0`; capacity `<= ir_dim` (a design rule, not asserted). `ir_num_clusters: 0` is the exact
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
  Resume paths (`utils.load_checkpoint`, `sft.load_sft_checkpoint`) are strict on every learned
  tensor; their one tolerance is `moe.balance_bias` (zero, `with_balance_default`).
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
  Since 2026-10-24 `ParallelSparseMoELayer.forward` delegates to `experts.grouped_mlp_forward`
  with `null_mask=~mlp_mask`: the dense-slot picks (`(index 0, weight 0)`) sort after the real
  experts, are counted in the same `bincount().tolist()` sync and are sliced off before the
  grouped GEMM (exact zeros as before; outputs differ from the old path only by bf16 summation
  order; about 5% of the MoE layer's time at micro shape, more at larger expert widths;
  `tests/test_efficiency_2026_10_24.py`). Under loss-free balancing the selection scores (the
  probabilities plus `balance_bias`) are formed in fp32 for the topk (a zero bias is
  bit-identical; at bias 0.30 the bf16 sum picked differently on 2% of tokens).
- Aux loss is on the router softmax, normalized by `loops_run` (`loops_run - detach_loops`, the
  live passes, under `detach_loops`), one `topk` feeds loss and
  selection. Both trainers pass `token_mask = input_ids != pad` (`None` is bit-identical), so aux
  values before that are not comparable. Selection fraction is the signal, not mean weight.
  The masked branch counts in fp32 since 2026-10-20: bf16 counts saturate at 256, so every aux
  value logged on the mask path before that date is a constant, not a load read, and the aux
  term of those runs balanced nothing.
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
  their own chunks, checked in `tests/test_evidence_port.py` (test 9), not asserted at runtime. `last_memory_mass` (detached, per loop, every token) is
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
  device-side assert). Since 2026-10-24 the padding is cut into 128-token segments
  (`transformer.EVIDENCE_PAD_SEGMENT`; 0 restores one causal segment per row) before the
  encoder's `cu_seqlens`, so a row's trailing pad run is no longer one long causal segment
  (real-token states and logits bit-identical; at pilot shape 9.61 to 7.57 ms on a
  4608/600/600/600 batch, no change when every row is full). Under the key/value reader
  `build_evidence` skips the cross reader's rotary gather (`position_embeddings` None).
- One evidence segment per query segment, paired by position; a document that retrieved nothing
  gets a zero-length segment, never dropped (else later documents read the wrong evidence).
- The reader reads `step_input + evidence_query_bias` every loop, bidirectional, scaled by
  one-init `evidence_loop_scale[loop]`; evidence states are gated per chunk by
  `1 + evidence_gate_scale * sigmoid(selector mass)` before `k_proj`/`v_proj` (zero-init scale is
  exactly neutral). Since 2026-10-24 a frozen zero gate (`freeze_evidence_gate: true`, scale
  exactly 0, every config) is skipped under both readers: no selector mass, scatter or multiply
  by ones per loop; `_evidence_gate_is_identity` caches the zero test against the scale's
  `_version`, so there is no per-step sync; forward and key/value reader gradients bit-identical
  (`tests/test_evidence_gate_skip.py`). IR experts run before the shared seed so this loop's selector gates. No KV
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
- **Eval packer layout** (`eval_abstention.py: _pack_evidence_batch`, every generation path with the
  port): since 2026-10-24 each row's evidence is laid out sorted by segment id, the row's own
  tokens before the dump width under right padding and after it under left padding (the dump is
  then the row's first segment), with a non-decreasing assert on CPU tensors, because
  `evidence_cu_seqlens` counts tokens per segment. Before that date every left padded decode with
  evidence read its buffer shifted by the dump width, so the A3 generation and A7 pathway (and A4
  edit) reads before 2026-10-24 are void; all three were re-read on the fixed packer that day, A4
  on the bge route (`eval_store.py edit --route bge`). `tests/test_evidence_pack_layout.py`.
- `evidence_selection_loss`: BCE per visible chunk vs gold, renormalized over the token's visible
  chunks, averaged over every (loop, IR expert), weight 0.1. Supervised at the positions that
  produce a supervised token (`labels[:, 1:] != -100` shifted to `:-1`), the positions
  `answer_start_positions` and the readouts use. **Asserts gradient checkpointing off**
  (it reads stashed tensors a recompute would produce under `no_grad`). Logged `selection:`.
  Since 2026-10-24 the per-chunk BCE runs in log space with no clamp on the renormalized share (a
  gold chunk under 1e-6 of the visible mass now gets a gradient; the old clamp zeroed it), and rows
  with fewer than two visible chunks are excluded. `evidence_selection_weighting` (`token`,
  default, the mean over supervised predicting positions; `segment`, each query segment averaged
  first from `cu_seqlens`, then the segments) is read at the pretrain.py call site and passed
  through `evidence_selection_term(segment_ids=, num_segments=)`.
- `evidence_contrastive_loss` (added 2026-10-17, the pilot's A7 recall fix): in-batch InfoNCE on the
  selector's unit query at the answer-start positions against every chunk in the batch, the other
  documents' chunks as negatives, positives = the position's own document's gold chunks (joint
  mass when several, rows without one excluded), fixed temperature
  `evidence_contrastive_temperature` 0.07, both directions averaged (Co-LMLM's recipe). The IR
  module stashes `last_memory_query` / `last_adapted_keys` per loop next to the weights;
  `moe.evidence_contrastive_term(chunk_gold, positions, temperature)` averages over (loop, expert).
  Weight `evidence_contrastive_weight`, 0 by default and in every run before 2026-10-17; logged
  `contrastive:` in the step line and the held-out eval when on. Same no-checkpointing rule as the
  selection term. Two keys added 2026-10-21: `evidence_contrastive_loops` (null = every loop, else
  zero-based loop indices the term averages over; loop 0 is the only pass stochastic depth always
  runs, so a listed later loop can be skipped and the term then returns `None` for that step) and
  `evidence_contrastive_trunk_grad` (in [0, 1], the share of the term's gradient reaching the
  trunk through the query's hidden state; 1.0 is bit-identical to before, 0.0 trains only the
  query head: `norm`, `down_proj`, `loop_query_bias`, `key_adapter`). Under 1.0 the IR expert
  stashes its input per loop in `ir_module.query_input_by_loop` (cleared in `begin_forward`) only
  while `ir_module.stash_query_input` is on, which both trainers set once after the model build
  through `moe.set_contrastive_query_stash(True)` when the weight is above 0 and
  `evidence_contrastive_trunk_grad` is under 1.0; the term then
  rebuilds the query at the answer-start rows through `InformationRetrievalExpert.contrastive_query`
  (`scale_gradient`: identity forward, scaled backward; then norm, `down_proj`, `unit_query`) under
  `te.autocast(enabled=False)`, since the row count is not a multiple of 16.
  `evidence_contrastive_loops` defaults to `[0]` since 2026-10-24 (config.py fallback and the
  three yamls): past the first pass the IR query position has already read the evidence through
  the key/value reader, so in-batch InfoNCE there can be solved by matching the chunk content it
  read, and the store eval reads the first pass's query; null (every loop) is still accepted, and
  the recorded arms w0e3 and w25e3 ran with it.
  `tests/test_evidence_contrastive.py` (tests 1 to 6 on the loss alone, GPU-free; tests 7 to 11 need
  CUDA and TE: bit-identity of value and gradient at the defaults, the loop filter, zero trunk
  gradient with the head still learning, half scaling exact, a detached prefix skipped by both terms).
- `GroundednessHead` + `groundedness_loss`: label "gold present AND answerable" from the corpus
  (`.ans`, not the condition; never the model's argmax). Reads `moe.last_reader_output` at
  `pretrain.answer_start_positions` (last prompt token of each supervised span), weight 0.1. **Its
  weight gradient is zero until `o_proj` leaves zero**: an early falling `grounded:` is the bias
  learning the base rate; the held-out AUROC in `[eval]` tells them apart. That holds for the cross
  reader only: under the key/value reader `last_reader_output` is `shared_attn`'s whole output (own
  context plus evidence), nonzero from step 0, so the BCE gradient reaches the trunk's own attention
  and the head can read the label off the prompt context or the base rate (review 2026-10-24,
  finding 9). Since 2026-10-24 `groundedness_term` detaches the head input under the key/value
  reader: the head still learns and no trunk parameter gets a gradient from it, but the head can
  still read the label off the prompt context in the forward; an evidence-only input (the
  prefix-only softmax mass) is not built.
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
  peak; `loop_ce_weights` is then required. `loss_ce` is the final loop's raw CE. Since
  2026-10-24 the unweighted CE, top-1 and `p_max` sums of `_chunked_linear_ce` accumulate in fp32
  (before, the logged and `[eval]` CE sat on a 1/64-nat grid and `top1_acc` could read 1.0078;
  `tests/test_chunked_ce_fp32.py`), and the eval path (`return_hidden` False) norms only the last
  loop's state.
- `return_aux_loss=True` -> `(x, aux_loss)` plus `extra_token_outputs` with MTP. `skip_mtp=True`
  drops it (bit-identical logits); evals and non-drafting inference pass it. `evidence=` takes an
  `EvidenceBatch`.
- `detach_loops=k` (on `TinyMoETransformer.forward` and `LoopMixtureOfExperts.forward`, 0 by
  default and bit-identical): the first `k` passes run under `no_grad` as plain steps before the
  checkpoint branch, their states kept in `hidden_states_all`, their aux terms constants;
  asserted `0 <= k < n_loops`. Under it the prelude decoder gets no gradient from the language
  loss (0 of 26 decoder parameters at `k = 2` on the test fixture, measured with the depth memory probe
  and recorded in `docs/measurements/depth_memory_probe.md`, not asserted by the test), so it is not a training recipe
  until a straight-through route to the first live pass exists. `tests/test_loop_machinery.py`.
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
- `--max-doc-tokens` defaults to 4093 for training builds since 2026-10-24
  (`MAX_DOC_TOKENS_TRAIN` = 4096 - 2 MTP pads - 1, what the evidence reader keeps) and stays 4094
  under `--heldout` (`MAX_DOC_TOKENS_HELDOUT`, byte identical held-out builds). Every build before
  that date, the pilot merges included, carries 4094-token rows the evidence reader drops (12,240
  documents, 50.11M tokens in each pilot split); by decision they stay as built.
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
  two flags added 2026-10-24, default off and byte identical, change only `.mask` in the
  retrieval arm: `--mask-first-name-mention` (mask 0 on the first run of name or placeholder
  tokens per document, suffix `m1`) and `--support-gold-only` (a value counts as supported by the
  gold card only, so gold-absent documents supervise no value span, suffix `g1`);
  `inject_val` is filler only and identical across arms; `scripts/prepare_clean_val.py` cuts
  `inject_val_clean` from the same validation windows minus those whose document the evidence
  builder read) and `prepare_chain_data.py` (fictional
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
- Every parameter gets an fp32 master (bf16 ulp is ~3x the AdamW step at 3e-5); since 2026-10-24
  the masters are saved in the payload (`fp32_masters`, about 0.4 GB at 99M) and restored when
  present, and the depth-sampling and CUDA RNG states are saved and restored; an older checkpoint
  loads as before and reseeds its masters from bf16.
  Since 2026-10-24 (evening) the epoch loop selects the permutation through the prepared loader
  (`select_epoch`): accelerate's `DataLoaderShard.__iter__` calls `dataset.set_epoch(iteration)`
  at the start of every pass and `iteration` counts the passes of the current process from zero,
  so a resume past epoch 0 that set the epoch on the dataset alone trained on epoch 0's order
  again (found on the `w0e3b` twin's resume at epoch 1, killed before a save; no recorded run
  resumed past epoch 0). A fresh run is unchanged; `tests/test_sft_resume_state.py` test 8.
- The SFT cosine floors at `lr_min_factor: 0.05` of lr (pretrain floors at 0.1); the pilot profile
  (`config_pilot.yaml` and the pilot-shape yamls under `ckpts/inject/`) sets `lr_min_factor` 0.1.
- `fresh_lr` group: `--ir` = `is_rebuilt_ir_param or is_fresh_loop_param`; `--evidence` =
  `is_fresh_loop_param` alone (the IR table trains at the trunk's rate by choice).
- Per-conversation loss weighting is off for `SFTConfig`, on for repair and evidence.
  `p_max`/`top1_acc` stay unweighted. The global token counter continues.
- `--data-dir`, `--train-split`, `--val-split` override the profile's data for any profile (logged
  as `data overrides`; a relaunch passes them again). Since 2026-10-24 the checkpoint stores the
  splits and `data_dir`, a relaunch on a different split than the one recorded is refused unless
  `--switch-split` is passed (the warm-up to main switch), the save and eval grid stays fixed in
  tokens across relaunches (arm P's 300, 351, 426, 530M reads sat on the old shifting grid), and a
  finished run exits 0 instead of resuming from `_final.pt`. `estimate_packed_rows` (which sets
  `total_steps` and so the cosine) replays each reader's rule since 2026-10-24: the evidence
  reader keeps a document when `len + num_mtp_tokens <= max_length - 1`, the plain reader when it
  is `<= max_length`; empty documents are skipped. On `pilot_main_train` it estimates 241,398 rows
  (about 30.17k steps) against the old 258,852 (32,356; arm P's epoch ended at about 30.3k), so a
  new evidence arm's cosine reaches its floor where arm P's stopped at 6.55e-5; the control's
  252,912 is unchanged. `EvidenceDataset` logs the documents and tokens over the row limit at
  startup and per worker.
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

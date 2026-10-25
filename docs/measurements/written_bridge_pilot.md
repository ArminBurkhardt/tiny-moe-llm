# The written bridge at pilot shape (2026-10-24): R9, arm P's chains with each hop written before the answer

Rung R9 ([NEXT.md](../plans/NEXT.md), the R9 ladder row; run order decided 2026-10-23 by the
user, R9 after R4b's expert arm, its read gating R4's corner arm). The question: does reasoning in
written tokens compose where the loop did not? Arm P's final save
(`ckpts/evidence_pilot_p/checkpoint_evidence_final.pt`) reads every composition cell at chance,
and so did every save of its chain continuation with lookup at 0.99
([chain_continuation_pilot.md](chain_continuation_pilot.md)). R9 fine-tunes the same save on the
same kind of chains with the intermediate entities written out before the answer ("Step 1: e1.
Step 2: e2. Answer: ek.") and reads it with a generate-then-score mode. The same day an eval-side
packer bug was found and fixed that had voided every decode with evidence attached before it
(below); every read in the tables here was taken after the fix.

R9 passes by a wide margin. The model composes two and three facts as soon as the intermediate
entity is a token it writes and reads back: held-out 2-hop accuracy 0.997 against chance 0.191 at
the final (z 64.7), 0.982 already at the first save (10M), on compositions absent from every
training split; 3-hop 0.997 against 0.206. Read sites past the first pass add nothing once the hop
is written (paired held-out Delta, kept 3 minus kept 1, +0.002 +- 0.001), so the composition lives
in the written tokens, not in the loop. The written depth does not extend past the trained one:
on 4-hop questions the model writes two steps (the most it was trained on) and answers at chance.

## Run

- `bash ckpts/inject/launch_pilot_bridge.sh`, config `ckpts/inject/config_pilot_bridge.yaml` (the
  chain continuation config: batch 4 x 2, lr and fresh lr 3e-4 with the cosine fitted to the
  split, saves and evals every 10M, `fixed_split` off) on `chains_bridge_train` /
  `chains_bridge_val`, two passes, from arm P's final.
- The bridge split, built 2026-10-23 by `bash ckpts/inject/build_chains_bridge.sh` (log
  `ckpts/inject/build_chains_bridge.log`, exit 0): 500,000 train questions, 33.67M prompt tokens
  and 109.8M evidence tokens per pass (ratio 3.26), 12 chunks per question; 2,000 val questions.
  Every buffer and sidecar byte is the plain chain build's; only the assistant turn differs.
- Run directory `ckpts/evidence_pilot_bridge/`, log `ckpts/inject/pilot_bridge.log`. Complete
  2026-10-24 at 17:18: 66.35M tokens over two epochs in 184.5 minutes, exit 0 (the planned two
  passes of 33.67M are 67.3M; the trainer's count is the one recorded). 5.7k to 9.1k tokens per
  second by window, the rate set by the evidence axis as on the chain continuation; peak 16.15 GB,
  flat. Saves at 10M to 60M plus `checkpoint_evidence_final.pt`.
- Reads by `bash ckpts/inject/read_bridge_save.sh CKPT TAG`, outputs
  `ckpts/inject/pilot_bridge_{TAG}_gen_{d3,hop4}.{json,log}`: `eval_chains.py --generate-bridge`
  on 1,000 questions per split and hop count, `chains_eval` and `chains_heldout_tmpl` at depth 3
  with every kept-sites count (0 to 3), `chains_hop4` at depth 4 with every site. The mode decodes
  the bridge greedily up to 48 tokens with the evidence attached, stops at "Answer:", then scores
  the type-matched candidates after the generated prefix through the unchanged scoring path.
  Accuracy is over every question; the marker and hit columns say how many wrote the bridge.

## A note on the splits

464 of the 2,000 val questions and 610 of the 3,000 `chains_eval` questions repeat a train
question's text verbatim (name pool collisions in the generator; the plain chain splits share
this). `chains_heldout_tmpl` has zero overlap with train by construction (its 2-hop compositions
are absent from every seen split), so the held-out cell is the clean one and the seen cells carry
some repeats. The verdict rests on the held-out cell.

## The eval packer bug, found and fixed 2026-10-24

`scripts/eval_abstention.py: _pack_evidence_batch` (used by every generation path with the port
attached) laid each row's evidence tokens out as [the row's own tokens, then the dump width].
`modules/model/evidence.py: evidence_cu_seqlens` counts tokens per query segment and requires the
flattened evidence axis sorted by segment id. A right padded batch has the real segment before
the dump segment, so that layout is sorted. A left padded batch has the pad run as the row's first
segment, so the dump id is below the real id and the layout is not sorted: the count-based
`cu_seqlens` handed the first (S_ev minus n) tokens of each row, the row's own evidence, to the
dump segment and the last n to the real segment. Every left padded decode with evidence attached
therefore read its buffer shifted by the dump width, correct only for the row with the widest
evidence in the batch.

- Affected: every generation with the port attached. That is `eval_abstention.py --evidence-port`
  generation (the pilot's A3 generation read of 2026-10-14: 96% abstention under gold, EM 0.01,
  recorded as "unreadable in generation"), `eval_store.py pathway` (A7's pathway half: oracle EM
  2.6%, recorded as "unreadable"; the A4 edit read decodes through the same path) and the new
  `eval_chains.py --generate-bridge`.
- Not affected: every likelihood read (closed-book rank, the chain candidate scoring, the
  trainers' evals and the port read, the teacher forced calibration), all right padded; the
  dataset's own packing (real first, dump last); `inference.py` (one segment).
- Diagnosis on R9's 30M save: teacher forced argmax over the gold bridge tokens of 32 two-hop val
  rows was 100% under right padding and 69.6% under left padding (the first token of the
  intermediate entity 100% against 47%), while the incremental decode agreed with the left padded
  one-shot forward on 99.85% of tokens, so the decode loop was right and the left padded forward
  was not. After the fix: 100% / 100% / 100%.
- Fix: the packer places a row's own tokens after the dump width when the dump segment id is
  below the real one, and asserts that the flattened segment ids are non decreasing (CPU tensors,
  no device sync); `evidence_cu_seqlens` carries the same assert whenever gradients are off.
  `tests/test_evidence_pack_layout.py` checks both layouts with rows of unequal width, a row with
  nothing attached and a width 1 row, and fails on the old layout. A detached Sonnet review found
  no bugs (two nits and one test gap folded in).
- Void reads, kept under `ckpts/inject/bridge_prefix_bug/`: the bridge decode smoke on arm P's
  final (2026-10-23, `bridge_smoke.json`), the 10M and 20M bridge reads and the 30M val split read
  of the morning of 2026-10-24 (held-out 2-hop 0.61 at 10M with the first hop hit only 41 to 57%
  of the time). The 10M to final reads below are the re-reads on the fixed packer.
- Re-reads on arm P's final with the fixed packer, 2026-10-24: `bash
  ckpts/inject/ladder_pilot_redo.sh`, outputs
  `ckpts/inject/pilot_p_ladder2_{abstention,store_pathway}.*`, the A4 edit read
  (`eval_store.py edit`, `ckpts/inject/pilot_p_ladder2_store_edit.*`) and its bge routed variant
  (`eval_store.py edit --route bge`, `ckpts/inject/pilot_p_ladder2_store_edit_bge.*`).

| re-read on arm P's final (fixed packer) | 2026-10-14 read (void) | re-read 2026-10-24 |
|---|---|---|
| A3 under gold: false abstention on answerable rows | 0.956 | 0.770 |
| A3 under gold: exact match / token F1 | 0.0101 / 0.0156 | 0.0456 / 0.0628 |
| A3 under gold: external memory mass at the last prompt position | 0.0791 | 0.1625 |
| A3 non-abstain under distractors / none | 0.0045 / 0.0000 | 0.0070 / 0.0000 |
| A3 likelihood: answer span CE gold / distractors / none, AUROC | 2.1257 / 6.1133 / 5.8720, 0.458 | identical |
| A3 counterfactual, items answered right with the unswapped chunk | n 6 | n 28: follow 0.429, `mr_ll` 0.026, other 0.571, every stratum "n too small" |
| A3 counterfactual, every eligible item (218 of 2,000), follow by stratum 0 / 1-9 / 10-99 / 100-999 / 1000+ | 0.018 pooled | 0.039 / 0.000 / 0.026 / 0.182 / 0.138 (`mr_ll` 0.02 to 0.22) |
| A7 pathway EM, model query / bge / oracle buffer (n 6,000) | 0.000 / 0.022 / 0.026 (oracle abstain 0.883) | 0.001 / 0.044 / 0.088 (abstain 0.948 / 0.584 / 0.610) |
| A7 pathway, first per-source block (HotpotQA, n 2,000), oracle EM | 0.036 | 0.189 (F1 0.253, abstain 0.169) |
| A4 edit flip rate (model query route) | 0 of 2,000 answered before the edit | 1 of 2,000 answered before the edit: edit n 1, flip 0.000; delete n 0; the edited gold chunk attached directly followed (n 1) |
| A4 edit flip rate (bge route, the serving configuration) | not run | 95 of 2,000 answered before the edit. Delete: n 72, abstained 48, other 17, stuck 7, flip 0.903 +- 0.035 against the bar 0.70, PASS. Edit: n 23, follow 3, other 19, stuck 1, flip 0.130 +- 0.070, FAIL; the edited gold chunk attached directly: n 23, follow 2, other 20, stuck 1, follow 0.087 |

The likelihood numbers are identical to the old read, which confirms that only the generation path
changed. A3 in generation is readable now but weak: under gold the model still abstains on 77% of
answerable rows and answers 5% exactly; the likelihood half of A3 is unchanged and holds. A7's
pathway gate (pathway EM against oracle EM) still fails, 0.001 against 0.088 (z -23.7), because
the model's own query finds nothing (A7's recall half, recall@5 0.004). The oracle route is
readable now (EM 0.09), but the A4 flip read runs on the model's own retrieval route, which finds
nothing, so it stays unreadable (one item answered before the edit); a flip read routed through
bge needed `eval_store.py edit` extended, which was built and read the same day (next paragraph).

The bge routed flip read (A4), 2026-10-24. `scripts/eval_store.py edit --route bge` retrieves the
"before" and "after" buffers by the question's bge query key against the original store's and the
edit store's raw bge keys, the serving configuration, while the model still generates with the
port attached; the default route is unchanged (`tests/test_eval_store_route.py`; a detached Sonnet
review found no bugs). Command: `TINY_LLM_CONFIG=config_pilot.yaml python scripts/eval_store.py
edit -c ckpts/evidence_pilot_p/checkpoint_evidence_final.pt --store data/index/openqa --edit-store
data/index/openqa_edit --buffer 4 --route bge --json-out
ckpts/inject/pilot_p_ladder2_store_edit_bge.json`, log `pilot_p_ladder2_store_edit_bge.log`. On
this route 95 of 2,000 sampled edits were answered correctly before the edit (the model route read
1 of 2,000 the same day). When the fact leaves the store (delete edits, the gold chunk removed),
the answer leaves with it: the model abstains or answers something else and does not fall back to
a memorized answer (stuck 7 of 72), so on the serving route the facts it answers from do live in
the store. When the fact is rewritten (edit edits, the gold chunk carrying a substitute answer),
the model neither keeps the old answer (stuck 1 of 23) nor copies the new one (follow 3 of 23) but
answers something else, and the same holds with the edited chunk handed to it directly (follow
0.09). Following an edited real-text chunk is therefore a generation weakness of arm P on real
questions (exact match 0.05 under gold on SQuAD, the A3 re-read above), not a store failure; on the
synthetic biographies the same model follows a swapped card 100% of the time. The edit half rests
on 23 items. Verdict: A4's delete half holds on the serving route, the edit half fails on
generation quality; both halves were unreadable before the decode fix of the same day and on the
model's own query route, which finds nothing.

## The trainer's series (held-out `chains_bridge_val`, 25 batches, 44,673 supervised tokens)

Teacher forced CE over the assistant turn, that is the written bridge and the answer.

| save | bridge CE | top-1 per token | selection |
|---|---|---|---|
| arm P final (step 0) | 4.895 | 0.535 | 0.453 |
| 10M | 0.0101 | 0.9985 | 0.373 |
| 20M | 0.0054 | 0.9986 | 0.331 |
| 30M | 0.0018 | 0.9998 | 0.309 |
| 40M | 0.0010 | 1.000 | 0.296 |
| 50M | 0.0009 | 0.9998 | 0.287 |
| 60M | 0.0005 | 0.9998 | 0.281 |
| final | 0.0004 | 0.9998 to 1.000 | 0.279 |

The form is learned inside the first 10M tokens and the rest is confidence. On the training rows
the per-loop CE ends at [0.0002, 0.0001, 0.0001], the loop scales go [1.289, 0.566, 0.258] to
[1.27, 0.77, 0.34], and the selector's held-out loss falls 0.45 to 0.28 while its external mass
climbs 0.31 to 0.62 (the written entity names the chunk to read next).

## The reads by save (depth 3, three read sites kept unless marked)

Chance: 1-hop 0.115, 2-hop 0.185, 3-hop 0.206, held-out 2-hop 0.191, 4-hop 0.210. Delta is the
paired held-out 2-hop accuracy, kept 3 minus kept 1.

| save | 1-hop | 2-hop | 3-hop | held out, kept 1 | held out, kept 3 | Delta | 4-hop (depth 4) |
|---|---|---|---|---|---|---|---|
| 10M | 0.983 | 0.977 | 0.973 | 0.971 | 0.982 | +0.011 +- 0.005 | 0.199 |
| 20M | 0.996 | 0.994 | 0.995 | 0.988 | 0.993 | +0.005 +- 0.003 | 0.195 |
| 30M | 0.999 | 0.994 | 0.988 | 0.989 | 0.993 | +0.004 +- 0.002 | 0.176 (z -2.6) |
| 40M | 1.000 | 0.997 | 0.996 | 0.991 | 0.992 | +0.001 +- 0.002 | 0.193 |
| final | 0.998 | 0.998 | 0.997 | 0.995 | 0.997 (z 64.7) | +0.002 +- 0.001 | 0.206 |

With no read site kept every cell is at chance at every save (final 0.136 / 0.198 / 0.220, held out
0.201, z at most +2.2 across the twenty kept-0 cells). The 50M and 60M saves were not read.

## The final read (`ckpts/inject/pilot_bridge_final_gen_{d3,hop4}.*`)

`chains_eval` at depth 3 by kept sites (accuracy, n 1,000 per cell):

| kept | 1-hop (0.115) | 2-hop (0.185) | 3-hop (0.206) | held-out 2-hop (0.191) |
|---|---|---|---|---|
| 0 | 0.136 | 0.198 | 0.220 | 0.201 |
| 1 | 0.998 | 0.997 | 0.998 | 0.995 |
| 2 | 0.998 | 0.998 | 0.997 | 0.997 |
| 3 | 0.998 | 0.998 | 0.997 | 0.997 |

- **The bridge is written and right.** With any site kept the marker rate is 1.000 at every hop
  count and the intermediate entity hit rates are 1.00 (2-hop) and 1.00 / 1.00 (3-hop), held out
  1.00. Accuracy among questions with the bridge fully written is 0.998 to 1.000; the 1 to 4
  questions per cell without it score 0 (one 3-hop question at kept 1 scored 1).
- **The written answer is near certain.** Gold NLL per answer at kept 3: 0.007 nats at 1 hop,
  0.056 at 2 hops (against ln K 1.72), 0.035 at 3 hops (ln K 1.62), held out 0.072 (ln K 1.68).
  Arm P's chain continuation stopped at 1.93 nats at 2 hops, a uniform pick among the candidates.
- **The loop adds nothing on top.** Paired Delta held out +0.002 +- 0.001 (n 1,000), seen +0.001 +-
  0.001; one read site composes 3-hop chains at 0.998. In this mode each decode step is a fresh
  forward over the prompt plus everything written so far, so one read site per step suffices: the
  model looks up hop 1, writes it, and the next forward looks up hop 2 keyed on the written name.
- **4-hop, never trained at that length** (`chains_hop4`, depth 4, every site): 0.206 against chance
  0.210 (z -0.3), marker 1.000, hit rates 1.00 / 0.99 / 0.00 by intermediate hop. The training mix
  has 1 to 3 hops, so at most two written steps, and the model writes two steps and then
  "Answer:", never the third hop. Every save reads the same (hit rate on the third hop 0.00 to
  0.01). A length limit of the training mix, not of composition; the bridges are well inside the
  48-token cap.
- **`instrument valid: no` is stale in this mode.** The line and its FAIL rows are the plain
  mode's check that a cell with fewer read sites than hops sits at chance. With the hops written
  that premise no longer holds (a later decode step reads again), so the line does not apply here
  and is not a failure; the bridge eval suppresses the verdict lines in this mode since the
  repository review of 2026-10-24.

## Against arm P's latent read

Arm P's final and all eleven saves of its chain continuation sit at chance in every composition
cell (held-out paired Delta +0.002 to +0.004, composed answers at a uniform pick among
candidates). The same save, fine-tuned for 10M tokens on the written form, reads held-out 2-hop
0.982. R9's bar (held-out 2-hop 3 sigma above chance with the 1-hop gate kept) is passed at the
first save and at every save after it, at z above 60. The repository review's 2-hop shortcut
baseline (0.256 against the uniform chance of 0.185) is far below every read with a site kept.

## Verdict

**The model composes two and three facts as soon as the intermediate entity is a token it writes
and reads back, on compositions never seen in training, after 20M tokens of fine-tuning on the
form; the loop never did this in silence at this size.** One seed, one dose; the margin (z 64.7
held out, 0.982 at 10M) leaves no doubt about the direction.

What it decides, and what stays open:

- 6f's reasoning form is written tokens; the loop keeps lookup and copying, as computation per
  token.
- R4's corner arm (loss placement through the coda) is no longer needed for the real run and is
  recommended dropped. Open for the user, not decided.
- The real run's training mix needs multi-hop rows at the depth wanted at serving: the written
  depth does not generalize past the trained one (4-hop at chance with two written steps).
- Not measured: transfer to real questions (A3 in generation, now readable and weak on arm P;
  HotpotQA bridge questions) and to
  retrieved rather than handed chunks. The result is on a synthetic task with the gold chunks in
  the buffer.

## Raw files

- Log `ckpts/inject/pilot_bridge.log`; build log `ckpts/inject/build_chains_bridge.log`.
- Reads `ckpts/inject/pilot_bridge_{10M,20M,30M,40M,final}_gen_{d3,hop4}.{json,log}` (fixed packer).
- Void reads `ckpts/inject/bridge_prefix_bug/` (`bridge_smoke.json`,
  `pilot_bridge_{10M,20M}_gen_d3.*`, `pilot_bridge_30M_gen_val.*`).
- Saves `ckpts/evidence_pilot_bridge/` (10M to 60M, final).
- Re-reads on arm P's final: `ckpts/inject/ladder_pilot_redo.sh`,
  `ckpts/inject/pilot_p_ladder2_{abstention,store_pathway}.{json,log}` (read 2026-10-24),
  `ckpts/inject/pilot_p_ladder2_store_edit.{json,log}` (read 2026-10-24).

## Later: the A-gate ladder on this checkpoint (2026-10-25)

R5 read the full A-gate ladder on the bridge final
(`ckpts/evidence_pilot_bridge/checkpoint_evidence_final.pt`) against arm P's final:
[r5_bridge_ladder.md](r5_bridge_ladder.md). The composition verdict above stands, but the chain-only
SFT left no other gate standing: the closed book reads at chance in every cell (the tier 1000
residual gone with the weakly stored biographies), held-out copying fell from 0.95 to 0.40 to 0.47,
SQuAD answering and abstention were replaced by the bridge form or degenerate text (EM 0.000,
abstention 0.000), the pathway read answers nothing on any route, and the clean-slice CE rose from
3.267 to 9.062 at exit 3. In the main thread's verdict this checkpoint is a demonstration that
composition happens in written tokens, not a usable checkpoint; 6f's mix carries the pretraining
slices beside the bridge rows, and every A gate is read on the shipped checkpoint.

# The real run (Phase 6): the spec, drafted and decided 2026-10-25 for the cluster application

Status 2026-10-25: the spec, written the evening every pilot read closed (NEXT.md "Now"), for the
cluster application; its open decisions were decided by the user the same night, every row on the
recommended option. Section order as [PILOT.md][pilot]. Every number names its source note (keys
at the end); a number marked "estimate" is arithmetic on cited figures and stays one until the
constructor, the builder or the first hour of the run prints it. Decided items carry their date
and source; the user's decisions of 2026-10-25 are the first table, and the body, written under
the options that were then decided, is marked "(decided 2026-10-25)" where it depends on them.

Status 2026-10-26: launch path L1 to L5 built (the launch-path table below): the constructor print
replaces the estimates in Model and Cost (330.4M total / 171.1M active / 459M FLOP per token, 118.4M
per evidence token), `config_run2.yaml` and the control, multi-GPU on the evidence trainer, and the
corpus pipeline on the existing builders ([RUN2_corpus.md](RUN2_corpus.md)). Three corpus
decisions of the main thread are rows 11 to 13 of the table below.

Status 2026-10-27: the allocation is MareNostrum 5 (BSC, AI Factories, the ACC partition; the
section "The allocation and the cluster" below), and the user raised the token target to 800B
(row 14): 266.67B unique prompt tokens per split, 3 epochs, on 16 H100, about 3,300 H100 hours at
the expected rate against the 4,500 targeted. Built the same day, uncommitted: the corpus driver at
800B with resumable builders, the Slurm job chain with its watchdog, the PC-side ops tools and the
supervisor, the trainer additions for the chained jobs (Build plan, "Built 2026-10-27"). What the
user does, in order: [RUN2_handover.md](RUN2_handover.md). Where a section below still reads 280B
or 93.33B it is the 2026-10-25 and 2026-10-26 text, kept; the 2026-10-27 notes override it.

## Decisions (the user, 2026-10-25)

Rows 1 to 7 are the main thread's; rows 0, 8 and 9 were raised while drafting; rows 11 to 13 are 2026-10-26 corpus decisions of the main thread; row 14 is the user's of 2026-10-27, which supersedes rows 6 and 11 and the share in row 12. Every row was
decided on the option the draft recommended; the consequence column is the draft's, kept.

| # | decision | decided | consequence of each option |
|---|---|---|---|
| 0 | which shape is shape L | **as configured in `config.yaml`**: width 768, not the blueprint's 1024 (8 prelude layers, 32 experts, 3 loops), the only shape the constructor has printed and the 16B run measured (332,324,717 total / 173.1M active at about 490M FLOP per token with the 8192 x 128 IR table, CONCLUSION.md lines 14 to 16; that matches the decided layout's about 333M / 174M with the 256-entry table) | configured: 383.5M total / 224.2M active, about 502M FLOP per token (CLAUDE.md), the 16B run's 97k tokens per second at 11% MFU on this width ([16B]); 280B tokens cost about 1.1k to 1.3k H100 hours (Cost). The plan's blueprint (width 1024, about 470M / 300M, NEXT.md Decisions "shape"): the shape [6b]'s 2.5k hours for 270 to 290B tokens was computed for, with its coda dropped and its sublayers open (row 5); no constructor print, no measured rate |
| 1 | routed expert count | **keep 32**, decided on the cluster's memory and the hours, not on quality (neither binds) | 32: about 333M total (with the 256-entry selector table, Model), 5.3 GB of training state per GPU, 4.7 GB per save. 16: about 249M total, 4.0 GB of state, 3.5 GB per save; the same active compute and FLOP, so the same hours (arm E4 ran at arm P's rate, 44k to 48k against 42k to 48k tokens per second, [e4]); quality equal at pilot shape (held-out CE 0.007 behind on clean text, [clean]) on one seed with two changes |
| 2 | `evidence_selection_weighting: segment` as the default | **yes**: `segment` is the real run's default | yes: 0.033 nats better on the QA slice and 0.060 on plain text at exit 3 (z +8.1 per token), selection equal, one seed ([infonce]); never read on the leak instrument, so the switch criterion and A1(b) reads carry it. A micro warm-up arm under `segment` (the `cfb` recipe, about 1.5 hours local) would read the leak before launch. No: the run stays on what every recorded arm ran (`token`) |
| 3 | `--support-gold-only` as a data rule | **yes**: carried into the real corpus builder as a data rule, **a build item behind a flag**; scope the biography-style supervised spans; the QA and natural-text analog (a value supported only by a chunk that also carries its subject) is a builder change | yes: held-out swapped-card following 0.67 to 0.79, held-out gold top-1 0.60 to 0.74 against 0.79 to 0.86 (those baselines are the recorded warm-up arm's; the same-code twin `cfb` reads following 0.64 to 0.65 and gold top-1 0.58 to 0.68, [leak]), no language or leak cost; copying at the 100M switch 0.88 to 0.90 against 0.95 to 0.96 for the same-code twin, so the flags slowed copying ([leak]), one seed. No: following stays at about 0.67 held out |
| 4 | the goal's reasoning clause | **reworded** to "facts in the store through the retrieval pathway, reasoning in written tokens, the looped trunk as computation per token"; the original clause, "reasoning in the looped trunk" (2026-09-30), is kept on record as the original goal and the aspiration to revisit after the real run (NEXT.md "The goal") | yes: L1 becomes descriptive and the written bridge read (R9's form) becomes the reasoning gate (Reads and gates); matches the measurement (latent composition at chance at every read, [chain]; written 0.997 held out, [bridge]). No: L1 stays a gate the pilot already failed at this shape |
| 5 | R4b's sublayer half | **today's one block per pass**; the three-sublayer core is not built | one block: no build; one read site per pass, three per forward. Three sublayers: a day of architecture work plus review and two 1B arms ([review]), at a noise floor that read 4 against 8 experts as equal ([e4]); the pilot never showed the loop composing |
| 6 | the token target | (superseded 2026-10-27 by row 14: 800B) **280B tokens**, the hours derived; the application asks for 2.5k H100 hours plus the corpus build line plus a 20% buffer (about 3.0k), the 5k ceiling stated; G8 (2x) turns the same hours into the 400B data cap | the main thread's "280B for 2.5k hours" is [6b]'s figure for the blueprint width. At the configured shape (row 0) 280B costs about 1.1k to 1.3k H100 hours at today's MFU and the 400B cap about 1.6k to 1.9k (Cost), so the ask covers the 400B cap at today's MFU; raising the target to 400B uses that headroom, keeping 280B leaves it for 6a, 6e and the second arm a failed read would need |
| 7 | R5 (the winner's eval set) | **tonight, on the local card, on arm P's final** (8 experts) and the bridge final, launched by the main thread | tonight: needs no decision beyond the shape, about 2 to 4 hours (estimate from the pilot ladder's 45 minutes for twelve reads plus the chain reads, [pilot]), and its margins enter the gate table before the run. On the cluster: costs H100 hours and lands after the application |
| 8 | the A1 control at the real run's size | **a short matched control**: no retrieval, the same stream and shape, to 10B tokens, about 30 H100 hours (about 4 hours on an 8-GPU node if DDP scales linearly) | short control: A1(b)'s paired reading (decided 2026-10-17) and A1(a) read at matched 10B; the final saves read against the prior and the 10B level. No control: A1 rests on the pilot's 47x / 4x reduction ([pilot]) and the 16B checkpoint, unmatched. A full-length control doubles the run |
| 9 | the thin tier 100 margin at micro | **accepted** as the recipe's known state on one seed, stated in the caveats; no second seed of `cfb` | held-out tier 100 at z 3.1 / 3.3 / 3.7 on the three same-recipe arms of 2026-10-25, paired inside noise: `cfb` against the recorded arm z -1.8 (`m1` and `g1` read z -0.9 and -1.5, but they are flagged variants, not same-recipe arms) ([leak]); a second seed says whether the R0b rule's tier 100 half holds on today's code |
| 10 | staircase visibility in the real corpus builder | **not on the launch path 2026-10-26 (moot without per-window retrieval over plain text; revisit for a later run)**; was open for the user, blocking the real builder | the design's staircase (a window sees only the chunks retrieved up to it) has no form under the key/value reader, whose prefix is visible in full to every token of its segment; without a fix a token reads chunks retrieved for later windows (Build list 4). Option A, this draft's recommendation: retrieve once per document from its first 64 tokens and mask that opening's loss (no kernel work). Option B: a windowed prefix mask (kernel work, 6a era) |
| 11 | split size and epochs | (superseded 2026-10-27 by row 14: 266.67B per split, 3 epochs) **93.33B prompt tokens per split, 3 epochs** (280B / 3), decided by the main thread 2026-10-26 as routine | 93.33B x 3: the evidence trainer runs whole epochs (`num_epochs` times the split's rows, it ignores `target_tokens`), so the split must be built at T = 280B / 3 for the run to be 280B. The Corpus section's 100B unique at up to 4 epochs would train 300B at 3 epochs or stop at 2.8 passes only by hand ([RUN2_corpus.md](RUN2_corpus.md) section 1) |
| 12 | chain share of the split | (replaced 2026-10-27 in the 800B plan by an absolute 2.8B per split, `--chain-tokens 2.8e9`: the same 1.4B plain and 1.4B bridge, since a chain prompt token carries about 177 bytes of chunk keys and a share of a 266.67B split would grow the disk with the corpus; [RUN2_corpus.md](RUN2_corpus.md) section 1) **0.03** (`--chain-share`), decided by the main thread 2026-10-26 as routine, pending the user's confirmation | 0.03: 3.7 TB peak disk, about 2 days after the filler download, 28.1M plain and 19.5M bridge questions each seen 3 times. RUN2's 0.10: 6.8 TB peak, 3 to 4 days, 93.6M / 65.0M questions; a share with no rung behind it (RUN2_corpus.md section 4) |
| 13 | chat replay share | **held at the supply limit** (about 0.5B unique non-holdout chat, so the 20% line is unreachable and the share goes to the filler), decided by the main thread 2026-10-26 as routine, pending the user's confirmation | at supply: smoltalk2 binds (phase 2 and ir take 1.5B, the QA replay arm 0.3B), the 18.5 points go to the biography slice. At 20%: needs a chat source that does not exist at this size (RUN2_corpus.md section 1) |
| 14 | the token target on the granted allocation | **800B tokens** (the user, 2026-10-27): 266.67B unique prompt tokens per split x 3 epochs, on 16 H100 (4 nodes of 4 at accumulation 1), for the MareNostrum 5 allocation (5,000 GPU hours applied, 4,500 targeted) | 800B: about 3,300 H100 hours at the expected 67k tokens per second per GPU (about 2,400 at 93k, the corpus's evidence ratio 0.045), about 206 hours of wall time at 16 GPUs, at least 3 chained 72 hour jobs; the floor is about 54k per GPU (about 4,100 hours for the run, the rest of the 4,500 for the corpus build and the control). The split, the chains (absolute 2.8B), the persons (3x the pilot's) and the filler (phase 1 about 260B) are resized in [RUN2_corpus.md](RUN2_corpus.md); 280B (rows 6 and 11) would have used about 1,160 of the 4,500 hours at the same rate |

## Question, and what it decides

Does the pathway recipe that passed at pilot shape (externalization against a matched control, a
copying reader, a port worth more than its in-context ceiling) hold at shape L over 280B tokens
(800B since 2026-10-27, row 14),
with retrieval attached to natural text, and does the trunk it leaves give 6e and 6f a base?

- **Externalization**: A1(a) on natural-text facts (first readable here: the pilot had no
  retrieval on natural text) and A1(b) on injected biographies, against the short control.
- **The port and the selector**: A2, A5, A3 and A4 on the bge route.
- **Language and reasoning**: held-out CE on a clean slice, the benchmark suite, the written bridge
  (row 4), L2. The goal's reasoning clause reads "reasoning in written tokens, the looped trunk as
  computation per token" (decided 2026-10-25); the original clause, "reasoning in the looped
  trunk", is kept as the original goal and the aspiration to revisit after the real run, with L1
  read descriptively until then.

A pass sends the checkpoint to 6e and 6f; the model's own retrieval is revisited on it (decided
2026-10-21). A fail at the copy, port or leak read stops the run before the main spend (Decision
table).

## Scope: the recipe, each component with the rung that admitted it

| component | value | admitted by, margin |
|---|---|---|
| retrieval from token 0, no graft | evidence on every slice | graft lineage 0 for 4 (CLAUDE.md, binding verdicts) |
| reader | key/value read in `shared_attn`, prefix rotation | R0b at 100M: copies (gold top-1 0.92, swapped followed 92%) where the cross reader does not (0.01, 2%) ([r0b]); pilot: top-1 1.000 in distribution, 0.95 held out ([pilot]) |
| copy-first warm-up | no real-value fact supervision before the reader copies; swap rate 1.0 in the warm-up | decided 2026-10-05: the warm-up arm stored nothing before the switch, tier 100 0.015 (z 2.6) at 300M ([r0b]); pilot switch at 300M ([pilot]) |
| swap rate after the switch | 0.15 | decided 2026-10-06: the 0.30 branch moved tier 1000 by -0.002 (z -0.3) ([r0b]) |
| unsupported fact spans | masked (weight 0, MTP targets too); plain-text facts outside the tagger accepted into the weights | R6 skipped 2026-10-24: arm (a) stores plain-text facts (tier 100 0.378), arm (b) masked stores nothing (z at most 1.7) ([r0b]) |
| `--support-gold-only` | on biography spans | row 3 (decided 2026-10-25) |
| balancing | loss-free, rate 0.001, no aux term | R8, decided 2026-10-21: CE 3.9055 against 3.9037 (paired z -1.8), load max over mean 1.03 to 1.05; the fixed aux term stored more (tier 100 z 14.5) ([lossfree]); 1.01 to 1.06 at pilot shape ([e4]) |
| optimizer | AdamW | R7, decided 2026-10-21: Muon stored the biographies about twice as early, leak 1.6x / 2.3x by sign ([muon]) |
| retriever term | none (`evidence_contrastive_weight` 0); bge keys the store and embeds the question at serving | decided 2026-10-21, unchanged 2026-10-24: every-loop term 0.059 recall@5 at 0.074 nats, arm PI 0.30 nats behind at 87M, first pass only 0.049 at 0.041, head only 0.003 ([infonce], [qhead]) |
| selection loss | 0.1, log space, `segment` weighting | the selector reaches chunk AUROC 0.78 at pilot ([pilot]); `segment` row 2 (decided 2026-10-25) |
| groundedness head | 0.1, input detached under the key/value reader | pilot grounded AUROC 0.994 ([pilot]); detach landed 2026-10-24 ([review]) |
| depth | 3 loops, `loop_count_sampling` 0.3, per-loop CE `[0.2, 0.3, 1.0]` at subsample 0.25 | arm P's recipe; R1b: depth 5 trains at 1.27x, no spec change ([depth]); exit gain 1 to 3 about 0.08 nats on every pilot arm ([clean]) |
| reasoning form | chains written out as a bridge, hops 1 to 4 | R9, 2026-10-24: held-out 2-hop 0.997 against chance 0.191 (z 64.7); 4-hop at chance when untrained ([bridge]) |
| shape | configured `config.yaml`, 32 experts, one block per pass | rows 0, 1, 5 (decided 2026-10-25); E4: the count is a cost choice ([e4]) |

**Not in the spec**, because no rung admitted it: the coda (R4 dropped 2026-10-24), the three
sublayer core (row 5), the always-on selector with a null key and the one-hot chunk channel, the
tied token table, the depth draw with a detached prefix (cuts the prelude off the loss, [depth]),
the union selection over passes, Muon, any retriever term, a separate search tower (held
2026-10-21). The plan's Phase 5 blueprint lists them; each is a later arm.

## Model

`config_run2.yaml` (built 2026-10-26, with `config_run2_control.yaml`): `config.yaml`'s model block with the pilot's selector table.

| key | value |
|---|---|
| width, intermediate (dense and MoE) | 768, 2304 |
| prelude layers, heads | 8; 12 x 64, 3 KV heads (`num_heads // 4`, [arch]) |
| experts | 32 MLP experts top 2 (row 1), 1 self and 1 cross attention expert, 1 IR expert |
| IR (the selector) | 256-entry exact-read table as at pilot (`ir_num_clusters` 0, `ir_dim` 384, direct read); `config.yaml`'s 65,536 entries are dropped by decision (NEXT.md Decisions, learned IR table) |
| evidence | encoder over all 8 prelude layers, `--reader-kv`, `freeze_evidence_gate: true` |
| MTP, LM head, PLE | 2 extra tokens, `lm_head_factor` 4, PLE 32 |
| depth, context, vocab | `n_loops` 3, seq 4096, 65,536 |

| layout | total | active | training state per GPU (16 B per parameter) | per save (14 B) |
|---|---|---|---|---|
| `config.yaml` as printed (HEAD; 380.2M / 221.0M / 464M after the 2026-10-26 model set) | 383.5M | 224.2M | 6.1 GB | 5.4 GB |
| 32 experts, 256-entry table (decided 2026-10-25), **printed 2026-10-26** | **330.4M** (estimate was about 333M) | **171.1M** (about 174M) | 5.3 GB | 4.6 GB |
| 16 experts, 256-entry table | about 249M | about 174M | 4.0 GB | 3.5 GB |

**Constructor print, 2026-10-26 (replaces the estimates in this section and in Cost):** the trainer
on `config_run2.yaml` with the key/value reader prints `params: total=330.4M active=171.1M (excl.
emb=101.9M) | forward FLOP/token ~= 459M (body 252M + heads 100M + attn 107M @ seq_len=4096)`; the
evidence encoder is 118.4M FLOP per evidence token over its 8 layers; the token table is 50.3M. The
drop of the IR expert's inner attention under the direct read (`direct_read`, every config) is why
the FLOP line fell from the 502M of the earlier estimates to 459M (the older estimates counted a
dead attention, as the Cost note said). Training state at 330.4M parameters: 5.29 GB per GPU, 4.63
GB per save.

Derivation (estimate, superseded by the print above): one routed expert is 3 x 768 x 2304 = 5.31M parameters, and 383.5M minus
224.2M is exactly 30 idle experts, so the 65,536 x 384 key and value tables (50.3M) sit in both
counts. Training state: bf16 weights and gradients, fp32 masters (every parameter, as `sft.py`),
fp32 AdamW moments; a save drops the gradients. FLOP per token about 502M as printed plus 118M per
evidence token (CLAUDE.md); the table cut changes it by about 1M (the `config.yaml` comment).
Activations, not state, set the peak: the 16B run peaked at 29 GB at batch 8 x 4096 without
evidence ([16B]), and the evidence axis adds about 0.61 MiB per evidence token (CLAUDE.md), about
30 GB at batch 8 and ratio 1.55, so about 60 GB on an 80 GB card (mean estimate); the per-row cap of about 9k gives a worst case of 8 x 9k x 0.61 MiB = 43 GB plus 29 GB, about 72 GB on 80 GB. Neither expert count
binds.

## Corpus

**Launch version, 2026-10-26:** the corpus the run trains on is the pilot recipe at scale on the
existing builders, by `scripts/build_run2_corpus.py` ([RUN2_corpus.md](RUN2_corpus.md)): T = 93.33B
prompt tokens per split, 3 epochs (rows 11 to 13 of the decisions), four slices (biographies with
filler 95.4%, QA 1.6% at its supply, plain chains 1.5%, bridge chains 1.5%), chat replay at supply,
an evidence ratio of about 0.4. The table and bullets below are the design this section was written
for; the new builder features they need are not on the launch path.

**Launch version at 800B, 2026-10-27** (row 14; [RUN2_corpus.md](RUN2_corpus.md), rewritten from the
code, is the source for every number): T = 266.67B prompt tokens per split, 3 epochs; per split bios
about 262.3B (share 0.984), QA 1.53B at its supply, plain chains 1.4B and bridge chains 1.4B (an
absolute 2.8B, not a share); persons `1000:300,100:1500,10:3000,1:6000` (10,800 persons, 486,000
renderings, 3x the pilot's, so about 1.7x lower sigma on the leak read); filler built phase 1 about
260.0B, phase 2 5B, ir 2.5B, capped per source (fineweb widened to all crawls, no repeat); evidence
about 12.0B tokens, a ratio of about 0.045 per prompt token (not the 0.4 of the 93.33B plan). The QA
build is permuted once by a seeded `qa_shuffle` stage before the order (its round robin front-loads
SQuAD and HotpotQA into the first 30% of the build, so the warm and control prefixes would have carried
about 3x the run's share of extractive QA; found by the final corpus review). The warm split is the
first about 28B of the main order and the control the first 10B (no evidence files, the full
biography arm), both byte prefixes in slice order checked by the driver; epoch 0 of the main split
starts with the warm split's documents in the warm split's order (`run2_main_train.order.json`,
`epoch0_head_docs`), so the switch continues the stream rather than repeating it. Disk peak about
3.06 TB, standing about 1.55 TB, in a 4 TB project quota. The tail curriculum (phase T below) is not
built.

100B unique tokens at up to 4 epochs (decided, [6b]); at 280B the run sees each token about 2.8
times. Built on the cluster (decided 2026-10-21) by the pilot builder mode at scale ([6c]), which
does not exist yet (Build list 4). Retrieval on every slice; the abstention condition is the only
evidence-free case.

| slice | main, share | tail, share | evidence (ratio, [6b]) | supervision | admitted by |
|---|---|---|---|---|---|
| web, edu text, code, math | 70% | 55% | top 2 chunks per 256-token window (1.0) | span weights, swaps 0.15, tail anonymization 0.5 | design (NEXT.md Phase 5 corpus); unread at any rung |
| evidence QA (SQuAD v2, HotpotQA, NQ, TriviaQA) | at its source limit, at most 2 passes | the same | gold, mixed, distractors, none (2) | answer or abstain | pilot QA slice ([pilot]) |
| synthetic chains, written bridges | 10% | 25% | 4 to 14 chunks (5) | answer with the bridge written | R9 ([bridge]) |
| chat replay (smoltalk2) | 20% | 20% | per window (1.0) | assistant text | design |
| injected biographies (A1(b)) | under 0.01% | none | store cards | span weights, `--support-gold-only` | R0b, pilot ([r0b], [pilot]) |

- **The QA share cannot be 15%.** The QA sources hold about 10.7M prompt tokens per pass for SQuAD
  and HotpotQA (CLAUDE.md, data prep); with NQ and TriviaQA at 2 passes the slice is well under 1B
  tokens (estimate), under 0.4% of the run. The design's 15% goes to the web slice unless a
  question generator over store chunks is built (not in this spec).
- **The tail** (the last 15%, reasoning weighted as the 16B run's phase 2, [16B]): shares are a
  design choice with no rung; chains hops 1 to 4 at 25 / 25 / 25 / 25 (4-hop is the serving depth
  the eval reads; R9 composed only the hops it trained, [bridge]).
- **Span weights.** Entities, numbers and dates tagged; supported when the span occurs in a visible
  chunk (weight 1, chunk gold), else weight 0; MTP targets carry the same weight. The A1 claim is
  restricted to entity-valued relations (NEXT.md, what must be proven).
- **Warm-up form.** Every supported span carries a substitute its visible chunk also carries (swap
  rate 1.0); QA gold rows and chains supervise real answers from the start (the pilot's QA
  exception: their answers are in the buffer by construction, [pilot]).
- **Store.** Training corpus plus Wikipedia, at least 0.4B tokens, 128-token chunks, MinHash
  deduplicated, raw bge keys, the document itself excluded, the near-duplicate filter (run of 32
  tokens or 8-gram Jaccard 0.3), PopQA and the A1 probe answers held out of the text.
- **Training row cap** 4093 tokens (landed 2026-10-24, so no row the evidence reader drops).
- **Build metrics**, the build fails under them: supported share of tagged spans at least 20%;
  bge top 2 gold recall on QA; near-duplicate drop rate per source; evidence ratio and fill per
  slice; the contamination check against the clean validation slice.

## Schedule

From a fresh seed (`init_scratch_seed.py` under `config_run2.yaml`, then
`migrate_groundedness_head.py`), `--evidence --reader-kv`, one run directory.

**At 800B, 2026-10-27** (row 14; `config_run2.yaml`): W runs on `run2_warm_train` (at most about 28B)
with `--max-epochs-this-split 1`, launched with `STOP_AFTER='4000M'` (the watcher drops STOP after
the first save at or past 4B; whether to read again at 12B is open for the user), then the switch
decision; M is three epochs of `run2_main_train` to 800B; phase T does not exist in the launch
version (the tail curriculum is not built). Every launch passes `--plan-split run2_main_train`, so
the cosine spans the 800B plan from step 0 and the switch changes nothing in it; `warmup_fraction`
0.000655 is 1000 steps of about 1,525,900 (about 1.1k in practice, the plan derived from the packed
row estimate); accumulation 1 at 16 GPUs (the job scripts pass `16 / GPUs`); saves every 1B, the
newest two kept plus the first at or past every 10B (about 80, 370 GB); `[eval]` and `[eval fixed]`
in one pass every 4B (2B cost about 10% at 16 GPUs); the kill at the first fixed eval past 8B (5B
could fire falsely: at 16 GPUs QA is about 20x thinner per update than at the pilot); uploads off.
The control: the first 10B of the stream, one epoch, `lr_min_factor` 1.0 (constant 4e-4 after the
warmup: the main run is still at peak LR at 10B, and an annealed control would store more and bias
the paired leak read), kill off. The control's exposure is open for the user: at 10B it sees about
3.75% of each person's renderings (a tier 1000 person about 37 times), so either probe tiers at
2,667 / 26,667 exposures per pass (the control then sees 100 / 1000) or a denser build of its own.

| phase | tokens | corpus form | ends when |
|---|---|---|---|
| W, warm-up | expected 2B to 5B (estimate) | warm-up form | the switch criterion holds, read every 2B from 2B; stop and decide if unmet by 28B |
| M, main | to 238B (85%) | main form, swap 0.15 | the token count |
| T, tail | to 280B (decided 2026-10-25) | tail shares | the token count; the cosine continues |
| control (row 8) | 10B | the same stream, no evidence, fact spans in the loss | the token count |
| 6e | a few B each | context extension 8k to 16k with retrieval; consolidation SFT | NEXT.md 6e |
| 6f | unmeasured | reasoning SFT with the bridge rows beside the pretraining slices (the 6f mix rule below), at an LR under the pilot continuation's 3e-4 | NEXT.md 6f; the A gates and the clean-slice CE read at every save, the kill rule on them |

- **Switch criterion** as the pilot: gold card top-1 in distribution at least 0.9 at every tier,
  and closed-book entity `delta` within 3 sigma of the prior at every tier on both forms
  ([pilot]). The 2B to 5B expectation: copying emerged between 50M and 100M at micro and before
  300M at pilot ([r0b], [pilot]); natural text adds supported spans in every window, which neither
  had; the run's 0.5M tokens per update is 16x the pilot's, and the micro record names fewer tokens
  per update as the lever if copying comes late (CLAUDE.md, Record). The relaunch passes
  `--switch-split`.
- **Optimizer.** AdamW, lr 4e-4 (the 16B run at this width and batch, stable over 16B tokens,
  [16B]; the [6d] transfer sweep at reduced width is optional and can run on the local card),
  weight decay 0.1 (the micro and pilot value every leak read ran under; the 16B run used 0.02),
  clip 1.0, warmup 1000 steps (0.52B tokens), cosine to 0.1 lr, router noise annealed over 1B
  tokens, BF16; FP8 only after 6a's 2B-token A/B within about 1% ([6a]).
- **Batch.** 524,288 tokens per update (the 16B run's, [16B]): batch 8 x 4096 per GPU, the
  accumulation set by the GPU count. `max_evidence_tokens` about 9k per row (1.5x the mean at
  ratio 1.55, as the pilot's 4608 was against 0.72; estimate; the 1.55 came from 6b's 15% QA share, RUN2's own shares give about 1.49: 1.40 in the main body, 2.0 in the tail), set from the packing line in minute
  one.
- **Cadence.** Save every 1B tokens (about 33 minutes on 8 GPUs at the estimated rate), keep every
  10B and the final; `[eval]` every 1B; `[eval fixed]` every 2B. The fixed pass is about 18% of a
  pilot arm's wall clock at its 25M cadence ([review]); at 2B, a 2 to 4 minute pass per 66
  minutes on 8 GPUs is about 3 to 6% until 6a's rewrite brings it under 30 seconds (estimate).
- **The 6f mix rule** (R5, 2026-10-25, [r5]). 6f's reasoning SFT carries the pretraining slices
  beside the bridge rows: evidence QA, the abstention rows, the biographies at their tier
  structure and the plain-answer form, at a lower LR than the pilot's chain continuation (3e-4
  re-warmed on a trunk whose cosine had ended at about 6.5e-5). The A gates and the clean-slice CE
  are read at every 6f save with a kill rule on them. The single-format alternative is measured:
  the chain-only bridge SFT kept composition (0.997 held out) and lost every other gate.

## Logging from step 0

| signal | where | state |
|---|---|---|
| CE, MTP, selection, groundedness, LR, grad norm, tokens per second | step line | exists |
| load max over mean and dense share per pass; `loop_scale` per pass | step line | exists |
| packing: fill, rows closed by the evidence budget, rows over the cap | minute one and per worker | exists |
| gold and content gain per condition and source; reader gain per loop; chunk AUROC and gold mass per chunk per loop; grounded AUROC | `[eval fixed]` | exists |
| per-exit CE on the clean validation slice | `eval_exit.py` per 10B save | exists (offline) |
| supported-span share of the loss (supervised tokens that are tagged fact spans) | step line | build |
| MFU against the constructor's FLOP line | step line | build |
| copy curve and closed-book `delta` on the biographies | `closed_book_rank.py` per read | exists (offline) |

## Early signals and kill rules

The run has no zero-init tensor to watch; these replace that rule ([6d]).

| signal | read | action |
|---|---|---|
| tokens per second under 0.8x the minute-one rate for an hour | step line | fix now (memory spill or straggler), CLAUDE.md "fix slow runs immediately" |
| loss spike not recovered in 200M tokens, or NaN | step line | roll back to the last save and decide |
| port: gold gain under 0.1 nats or content gain under 0 at the first fixed eval past 5B (8B since 2026-10-27, `config_run2.yaml`) | `[eval fixed]`, the built kill (`kill_tokens` 5B, 8B since 2026-10-27; `kill_min_gain` 0.1) | automatic stop (exit 10; the job chain halts on it with the `KILL:` line); the pilot read +2.02 at 75M ([pilot]) |
| no step line for 15 minutes, no save within the job's limit (49 minutes at 16 GPUs), a loss NaN, infinite or above 20 three times, or a finite loss whose 30-step median sits 1.0 nats above the job's lowest for 5 polls; a crash loop (3 bad jobs in the last 5, 2 early deaths, 2 jobs with no new save) | the Slurm watchdog every 60 s (`scripts/slurm/watchdog.sh`, 2026-10-27) | a stall: SIGTERM, save, the job marked stalled; a divergence: SIGKILL with no save and HALT (never restarted); a crash loop: HALT; every rule in `scripts/slurm/README.md` |
| port: gold gain under 1.6 nats at 10B | `[eval fixed]` | stop and decide |
| copy: switch criterion unmet at 28B | biography reads every 2B | stop and decide (the pilot spec's rule, [pilot]) |
| leak: A1(b) at 10B fails the paired reading against the control at tier 100 on either form, or the run's tier 100 residual above a quarter of the control's | `compare` at matched 10B | stop and decide before the main spend |
| selector: chunk AUROC under 0.59 on any loop at 10B | `[eval fixed]` | read with A5; not a stop |
| context reliance: A3 `mr_ll` rising per stratum on three consecutive reads | A3 likelihood per 10B save | stop and decide (Risks: reliance rises then decays) |
| supported-span share far from the build metric | step line | the builder, not the model; read at 1B |
| third pass `loop_scale` under 0.01 | step line | a defect to fix, never a reason to cut depth (binding rule) |

## Reads and gates

Every closed-book read at full depth. Noise: the CE seed spread is 0.029 at micro on one pair
([lossfree], [muon]); at pilot shape no seed pair exists, the 0.013 often quoted is arm C minus
arm P at the final and was contamination (0.002 on clean text, [clean]); shape L runs one seed, so
a CE difference under 0.03 is not read as an effect. Rank reads use the paired per-item sigma;
the person-clustered sigma is within 10% of it ([pilot]).

**Every A gate, the copy read and the clean-slice CE are also read on the shipped checkpoint,
after 6e and after 6f**, not only at the end of pretraining (R5, 2026-10-25, [r5]): 66M tokens of
chain-only bridge SFT on arm P's final erased the tier 1000 residual and the weakly stored
biographies, halved held-out copying, replaced answering and abstention on real questions with
the bridge form or degenerate text and multiplied the clean-slice CE by about 2.8. The cadence
column below is the main run's; the leak reads after post-training are the binding ones.
Since 2026-10-27 `[eval]` and `[eval fixed]` run every 4B (`config_run2.yaml`), so the "every 2B"
cadences below read every 4B, and the copy read before the switch is taken at the warm-up's 4B stop
(a second read at 12B is open for the user).

| gate | instrument | bar | noise | pilot margin | cadence |
|---|---|---|---|---|---|
| copy | `closed_book_rank.py bios --evidence gold\|swapped` | switch criterion | paired per item | top-1 1.000 in distribution at 300M; swapped followed 1.000 / 0.94 to 0.97 ([pilot]) | every 2B to the switch, then every 10B |
| A1(b) | biographies, `--evidence none --form both`; `compare` against the control | as amended 2026-10-17 (PILOT.md read table), at matched 10B; finals against the prior | arm P held-out tier 100 sigma about 0.004 ([pilot]) | control 47x / 4x above arm P at tier 100 (z -81 / -6.4); arm P held-out tier 100 0.0125 (z 3.2), tier 1000 0.064 / 0.046 ([pilot]) | every 10B |
| A1(a) | PopQA rank, prior controlled, cue relations apart ([seedinstr]) | tail tier worse than the control by 3 sigma | paired about 0.0012 on the seed's non-cue tail ([seedinstr]) | not readable (no natural-text retrieval) | at 10B against the control; finals against the 16B checkpoint, descriptive |
| A2 | `[eval fixed]` gold and content gain | gold at least 1.6 nats, distractors at least 0 | | +4.00 / +3.97 (E4 +4.19 / +3.82) ([pilot], [e4]) | every 2B |
| A5 | chunk AUROC per loop, grounded AUROC | 0.59 per loop; grounded 0.674 | 0.030 (NEXT.md gate table) | 0.78; 0.994 ([pilot]) | every 2B |
| A3 | `eval_abstention.py --evidence-port` gold, counterfactual | memorization ratio at most 0.05, follow at least 0.9 per stratum | binomial | likelihood holds (`mr_ll` 0.11); generation weak, EM 0.05, abstention 0.77 under gold ([pilot]) | likelihood every 10B; generation at the finals; its home is 6f |
| A4 | `eval_store.py edit --route bge` | flip rate at least 0.70 | binomial | delete 0.90 +- 0.035 (n 72), pass; edit 0.13 +- 0.07 (n 23), fails on generation ([bridge]) | finals |
| A7 | `eval_store.py recall`, `pathway` | recorded, not gating (no retriever term, decided 2026-10-21) | | model query recall@5 0.004 against bge 0.829; oracle EM 0.09 ([pilot]) | finals |
| reasoning (row 4) | `eval_chains.py --generate-bridge`, held-out 2-hop and 3-hop, 4-hop | 3 sigma above chance with the 1-hop gate kept, read against the 2-hop shortcut line (`penultimate_fed`, about 0.256 against 1/K 0.185), which alone would pass 3 sigma at z about 5.8 at n 1000; every recorded L1 read sits below it and R9's 0.997 clears it at z about 54 | paired bootstrap | 0.997 against 0.191 (z 64.7) after 20M tokens on the form (the bar was already passed at 10M, 0.982) ([bridge]); the same checkpoint failed every other gate (R5, [r5]) | every 10B |
| L1 | latent read by kept sites | descriptive (row 4) | 0.005 to 0.008 paired | +0.002 +- 0.005 ([pilot]); +0.004 +- 0.008 ([chain]) | finals |
| L2 | HotpotQA bridge CE by read sites; held-out 4-hop by sites | at least 0.05 nats on bridge CE | paired | unread at pilot; exit gain 1 to 3 about 0.08 on clean text ([clean]) | finals (R5, 2026-10-25: held-out 4-hop at chance by read sites on arm P and with the bridge written; the HotpotQA bridge has no instrument, Build list 9, [r5]) |
| language | clean slice CE per exit; 13-task suite against the 16B snapshot and the peers | within noise of or above the 16B run | 0.029 CE; binomial | arm P equal to the control on clean text (0.002) and within two sigma on the suite ([clean], [pilot]) | every 10B; suite at the finals |
| G8 | tokens per second at shape L over an hour, checkpointing and upload live | at least 2x today's | | | 6a, before the launch |

## Cost

**At 800B on MareNostrum 5, 2026-10-27** (row 14; supersedes the 280B lines below, kept as the
2026-10-26 figures). At the expected 67.3k tokens per second per H100 (the line below at RUN2's ratio
1.49 to 1.55): 800e9 / 67.3k = 3,302 H100 hours, about 206 hours of wall time on 16 GPUs, at least 3
chained 72 hour jobs (submit 5 to 6; surplus jobs exit in seconds). At the corpus's own evidence
ratio of about 0.045 (459M + 118.4M x 0.045 = 464M FLOP per token, 97k x 490 / 464 x 0.91 = 93k per
second): about 2,400 hours. The floor is about 54k per GPU (about 4,100 hours, the rest of the 4,500
for the corpus build and the control); the supervisor alerts under it and on projected hours over
4,500. The control: about 27 H100 hours at 104k (about 2 hours at 16 GPUs). The corpus build on the
cluster: about 400 GPU hours (estimate; a whole node is billed for each CPU-bound stage) plus about
444 GB of filler shards pushed in through the transfer nodes ([RUN2_corpus.md](RUN2_corpus.md)
section 4), or on the user's PC (about 3 TB of local disk needed, D: has about 125 GB free; open for
the user). Charging is probably per whole node (unconfirmed, a question for support), so reads run
four to a node.

Application table (H100 hours). Every line is an estimate until the constructor prints the shape
and the first hour prints the rate. Rate at the configured shape with evidence, **recomputed 2026-10-26 from the constructor print**
(459M FLOP per token forward, 118.4M per evidence token): 459M + 118.4M x 1.55 = 643M (1.55 is 6b's
evidence ratio from its 15% QA share; RUN2's own ratio 1.49 gives 635M). At the 16B run's achieved
rate (97k tokens per second at 490M FLOP, 11% MFU, [16B]) the same MFU gives 97k x 490 / 643 = 73.9k
tokens per second, times 0.91 for the evidence path's lower efficiency at pilot shape (arm P 42k to
48k against arm C 51k to 59k at 1.11x the FLOP, [pilot]): about 67k, range 64k to 75k per H100
(the earlier 685M line gave 63k, range 60k to 70k: it counted the dead IR inner attention). Hours
for 280B: 280e9 / 67.3k = 4.16e6 GPU seconds = 1,155 h (1,215 h at 64k, 1,040 h at 75k); for 400B
1,650 h (1,480 to 1,740). The control without evidence at 459M: 97k x 490 / 459 = 104k per second,
10e9 / 104k = 27 h. The blueprint figure is [6b]'s 30k to 32k.

**The corpus built for this run has an evidence ratio of about 0.4, not 1.49 to 1.55**
([RUN2_corpus.md](RUN2_corpus.md) section 6: filler windows carry no evidence under the pilot
recipe; chains and QA carry nearly all of it): 459M + 118.4M x 0.4 = 506M, 97k x 490 / 506 x 0.91 =
85k per second and about 910 h for 280B (range 820 to 950). So the lines below, kept at RUN2's own
ratio, are conservative for the corpus actually built; the first hour of the run prints the real
rate. The smoke at shape L on the 5090 (30 steps, batch 2, accumulation 1, `max_evidence_tokens`
4608, 2026-10-26) read 25.4k to 27.4k tokens per second at 16.5 GB peak, step 0 CE 11.17, step 30
10.96, a save and exit 10 clean; it is a consumer card at batch 2 and sets no cluster figure.

| line | configured shape (row 0) | blueprint width 1024 ([6b]) | basis |
|---|---|---|---|
| main run, 280B tokens, today's MFU | 1,040 to 1,220 (about 910 at the corpus ratio 0.4) | 2,400 to 2,600 | the rates above |
| main run at the 400B cap instead | 1,480 to 1,740 | 3,500 to 3,700 | the same |
| main run after G8 (2x) | half of the line | half of the line | [6a] |
| short control, 10B, no evidence (row 8) | about 27 | about 70 | 97k / about 40k tokens per second ([6b]'s rate without its 1.29 evidence factor) |
| corpus build, GPU | about 25 to 30 | the same | fact tagger about 15, bge window queries about 5, store embedding and ANN a few ([6c]) |
| corpus build, CPU | 1 to 2 node days | the same | tokenization of 100B, about 400M ANN queries, near-duplicate filter, store deduplication ([6c]); unmeasured |
| 6a throughput work | 100 to 150 billed per GPU; about 580 if a full 8-GPU node is held 3 days | the same | profiling and compile on one GPU about 3 days, the FP8 A/B (2 x 2B tokens) about 18, G8 one node hour |
| eval ladder | 30 to 50 | 40 to 70 | the pilot's read set 26 minutes per save and the ladder 45 minutes ([pilot]), at about 2x the FLOP per token, every 10B plus the finals |
| 6e context extension and consolidation | 40 to 60 | 80 to 120 | a few B tokens each ([6e]); a context change is also a corpus rebuild (CPU) |
| 6f | not asked | not asked | the rollout rate is unmeasured ([6f]) |
| ledger, 280B, before buffer | about 1,260 to 1,540 | about 2,700 to 3,050 | sum (6a per GPU): 1,040 + 27 + 25 + 100 + 30 + 40 = 1,262 low, 1,220 + 30 + 30 + 150 + 50 + 60 = 1,540 high |
| **ask (row 6, decided 2026-10-25)** | **2.5k + corpus line + 20%: about 3.0k H100 hours; ceiling 5k** | the same | configured (recomputed 2026-10-26): the ledger is 1.26k to 1.54k and 1.5k to 1.85k with its own 20%, so the ask leaves about 1.2k to 1.5k at 280B, and the 400B cap fits (1.7k to 2.1k, 2.0k to 2.5k with 20%); the 2.5k in the ask is the blueprint-width figure; named contingencies: the 400B cap, a second arm after a failed read, 6a and 6e; blueprint: the ask covers the ledger without its buffer |

- **Wall clock**: on 8 H100s about 6.0 days for 280B at 67k per GPU (280e9 / (8 x 67.3k) = 520k s), 3.0 on 16, assuming linear
  DDP scaling (unmeasured: the multi-GPU evidence trainer was built 2026-10-26, its 2-GPU against 1-GPU check runs in the cluster's first hour).
  At 800B (2026-10-27): about 206 hours on 16 GPUs at 67.3k per GPU.
- **Storage**: 4.6 GB per save at the printed 330.4M (earlier estimate 4.7 GB; configured, 256-entry table: 0.67 GB bf16 weights, 1.33 GB fp32
  masters, 2.67 GB fp32 moments; the masters are saved since 2026-10-24, CLAUDE.md "SFT"); 30
  kept saves about 140 GB (with `keep_every_tokens` 10B and `keep_local_checkpoints` 2 the trainer keeps the newest two plus one per 10B, so about 30 saves of 4.6 GB at the end of the run is an upper bound); 280 uploads at the 1B cadence about 1.3 TB moved. Corpus in today's
  per-occurrence format about 1.5 TB (200 GB prompt, 310 GB evidence tokens at ratio 1.55 (about 1.49 on RUN2's shares), about
  930 GB of fp16 chunk keys at 768 bytes per 128-token chunk occurrence); about 0.5 TB with chunks
  referenced by store id (a builder option). All estimates. The corpus as the launch pipeline builds
  it is another size: 3.7 TB peak and 1.9 TB standing at the 3% chain share, the warm and main
  merges 853 GB each and the control 30 GB ([RUN2_corpus.md](RUN2_corpus.md) section 4).
  At 800B (2026-10-27): the corpus peaks at about 3.06 TB and stands at about 1.55 TB (the main
  merge 1.37 TB, warm 144 GB, control 30 GB; RUN2_corpus.md section 3) in the 4 TB project quota;
  about 80 kept saves of 4.6 GB, about 370 GB, the 10B saves hard linked into `$DATA_ROOT/pinned`;
  uploads off (no node has internet); weights leave as bf16 exports of about 0.67 GB
  (`scripts/export_weights.py`, `scripts/ops/pull_exports.sh`) against the 2 TB transfer allowance,
  counted in a local ledger.

## The allocation and the cluster (2026-10-27)

| item | value |
|---|---|
| system | MareNostrum 5 (BSC), AI Factories, the ACC partition: 4 x H100 64 GB per node, 80 CPUs, 512 GB RAM and a 480 GB node disk at `$TMPDIR`; QOS `acc_ehpc`; exclusive nodes, probably billed per whole node (unconfirmed) |
| grant | 5,000 GPU hours applied, 4,500 targeted; 4 TB project storage; 2 TB transfer in and out |
| limits | 72 hour jobs; no outbound internet on any node, login nodes included (inbound ssh to `alogin1` and `alogin2`, transfers through `transfer1` to `transfer4`); Singularity only |
| run | 16 H100 (4 nodes), 8 x 4096 per GPU at accumulation 1, 524,288 tokens per update; at least 3 chained 72 hour jobs for 800B |
| order | the user's access and day 1, the corpus build, the 10B control, the warm-up with `STOP_AFTER='4000M'`, the switch decision, the run ([RUN2_handover.md](RUN2_handover.md)) |
| runbook | `scripts/slurm/README.md` (Day 0 on the PC, Day 1 on the cluster, the filler through the mirror, the chain rules, supervision, saves); the supervising session's standing instructions `scripts/ops/SUPERVISOR.md` |

Open for the user (2026-10-27): the control's exposure (probe tiers at 2,667 / 26,667 exposures per
pass, or a denser build of its own; Schedule), whether to read the warm-up at 12B in addition to 4B,
and where the corpus is built (the cluster, about 400 GPU hours and about 444 GB of filler shards
pushed in, or the PC, about 3 TB of local disk); and, from before, 6f's abstention reward.

## Build list before launch, in order

1. **Done 2026-10-25: R5** (row 7, [r5]). Arm P's final already carried A1 to A7 (the decode
   reads on the fixed packer) and its 4-hop reads by read sites (at chance at every save); the
   bridge final carried the 4-hop generate read (0.206 against chance 0.210). New: the A-gate
   ladder on the bridge final, the checkpoint form 6f builds on: a demonstration that composition
   happens in written tokens, not a usable checkpoint (the 6f mix rule in Schedule, the
   post-training reads in Reads and gates). The HotpotQA bridge by read sites has no instrument
   (item 9).
2. **Done 2026-10-26 (L4):** `config_run2.yaml`, `config_run2_control.yaml`, the seed `ckpts/inject/seed_run2_grounded.pt` and the constructor print (330.4M / 171.1M / 459M, Model and Cost) are in. Text as written: `config_run2.yaml` and its seed; (launch path, L4) the constructor print (total, active, FLOP per token, per
   evidence token) recorded here, replacing every estimate in Model and Cost. `config.yaml` has the opposite of four values the real run needs, so `config_run2.yaml` must carry these overrides: `weight_decay` 0.1 (`config.yaml` 0.02), `moe_balancing: loss_free` (`config.yaml`: aux), `evidence_selection_weighting: segment` (`config.yaml`: token), `num_ir_entries` 256 (`config.yaml`: 65536).
3. **Done 2026-10-26 (L3), the 2-GPU against 1-GPU check pending on the cluster.** **Multi-GPU on the evidence trainer.** (launch path, L3) `sft.py` is single GPU and `EvidenceDataset` has no
   per-rank sharding today (no rank or process index in either): rank sharding, DDP under
   accelerate, the token tracker across ranks, and the supervisor's lifecycle (preemption exit 20,
   upload, retention, `verify_resume`) on the `--evidence` path; a 2-GPU run of 100 steps against
   one GPU.
4. **The real corpus builder** ([6c]): **Not on the launch path, 2026-10-26: the launch uses the existing builders at scale (L5); the new features here (per-window retrieval over plain text, the fact tagger, the near-duplicate filter, the staircase question) are revisited for a later run.** per-window bge retrieval on natural text, the fact tagger
   and span weights, swaps and the warm-up form, tail anonymization, the near-duplicate filter,
   the store at 0.4B tokens or more, the biographies with tiers counted over the run's epochs,
   `--support-gold-only` (row 3; the QA and natural-text analog, a value supported only by a chunk that also carries its subject, is open: not yet decided in or out of scope), the bridge chains at hops 1 to 4 (`prepare_chain_data.py
   --bridge`), the 4093 cap, the build metrics, interruption safe. **Blocking, unbuilt:** the
   design's staircase visibility (a window sees only the chunks retrieved up to it) has no form
   under the key/value reader, whose prefix is visible in full to every token of its segment
   (`attention.prefix_varlen_attention`); without it a token reads chunks retrieved for later
   windows. Options: retrieve once per document from its first 64 tokens and mask that opening's
   loss (no kernel work; this draft's recommendation), or a windowed prefix mask (kernel work, 6a
   era).
5. Logging builds: (moved to during the run 2026-10-26, B2) the supported-span share and the MFU line.
6. **Done 2026-10-26 (L5):** the smoke at scale 0.0005 ran every stage in 23.7 minutes. A pilot-shape smoke of the real builder, (launch path, L5, under 5 hours local) local, one 1B-token arm on the 5090 (about 7 hours,
   arm P's rate, [pilot]): packing, fill, supported share, copy emergence, the A1(a) direction.
   Raised by this draft; the natural-text retrieval slice is otherwise first read on the cluster.
7. **Done 2026-10-26 (L4):** `config_run2_control.yaml`, `scripts/launch_run2_control.sh`, `scripts/reads/`. The control's config and launcher (launch path, L4) (row 8); the read scripts per save at shape L, timed on the
   first save.
8. 6a, after the spec (moved to during the run 2026-10-26, optional, none required) ([6a], decided 2026-10-21): profile, compile, host syncs, the eval-pass
   rewrite (deferred to 6a 2026-10-24), FP8 A/B; gate G8.
9. **The HotpotQA bridge instrument** (after the main run 2026-10-26, C1) (L2 and the reasoning row on real questions, raised by R5,
   [r5]): the bridge decode of `eval_chains.py --generate-bridge` over retrieved real chunks
   (HotpotQA bridge questions with the bge buffer attached), scored by read sites. Today the decode
   exists only for the synthetic chain splits. Needed by the finals, not by the launch.
10. **The 6f mix builder** (after the main run 2026-10-26, C2) (raised by R5, [r5]): the bridge rows merged with the pretraining slices
   (evidence QA with its abstention rows, the biographies at their tier structure, plain-answer
   QA, plain text) through `merge_evidence_splits.py`, extended to `.evhop` and `.factspan` (NEXT.md
   6f), and a 6f launcher that reads the A gates and the clean-slice CE per save with their kill
   rule. Needed by 6f, not by the launch.

## Build plan with estimates, 2026-10-25, re-cut 2026-10-26 to the launch path

**Hard constraint (the user, 2026-10-25): no local run longer than 5 hours; multi-day runs happen only on the rented cluster.** Every 6a A/B is a micro arm of about 40 minutes and the pilot-shape smoke of the corpus pipeline stays under 5 hours local.

**Re-cut 2026-10-26 (the user).** Cluster access starts 2026-10-28, so the launch path is 2 days of build, and anything not required for a successful run leaves it. The earlier multi-week figure summed optional work (6a throughput, the new corpus builder features, the post-run instruments) with the launch path and is withdrawn. The run uses the pilot recipe at scale on the existing builders. Estimates are build effort for one agent lane (h or d); wall clock is shorter where lanes run in parallel. Every efficiency percentage is an estimate; the token target, the hours and the ask in Cost do not change on this re-cut.

### A. Launch path, 2 days (2026-10-26 and 2026-10-27), required for a successful run

| item | what | estimate | check | blocked on |
|---|---|---|---|---|
| L1 | the model set in the tree verified: the IR inner attention dropped under `direct_read` with its load tolerance, the FLOP line, the `detach_loops` clamp, the balance shadow, the `ir_dim` assert (`experts.py`, `moe.py`, `transformer.py`, `utils.py`, `migrate_*.py`, `tests/test_ir_inner_attn_drop.py`, `test_loop_machinery.py`, `test_loss_free_balancing.py`) | 0.5 h; running 2026-10-26 | tests, a Sonnet review, then commit or restore | nothing |
| L2 | trainer: grid snap at every save and eval site; resume verification with exit 30 on the sft path; `total_steps` stored and warned on change; the groundedness checkpointing tripwire | 2 h; running 2026-10-26 | tests | nothing |
| L3 | multi-GPU on the evidence trainer under accelerate: per-rank sharding in `SFTDataset` and `EvidenceDataset`, DDP kwargs with `find_unused_parameters`, the token tracker and the balance counts all-reduced at the log cadence, kill, STOP and the save decided on rank 0 and broadcast, saves and `run_state` written by rank 0 behind a barrier, the eval and fixed-eval passes sharded by rank with reduced sums, the single-process path bit-identical | 1 d plus a Sonnet review; running 2026-10-26 | the single-process path bit-identical in tests; the 2-GPU against 1-GPU 100-step check happens in the cluster's first hour | L2 for the shared trainer lines |
| L4 | `config_run2.yaml` (`config.yaml` with the decided overrides: `weight_decay` 0.1, `moe_balancing: loss_free`, `evidence_selection_weighting: segment`, `num_ir_entries` 256, the batch and LR from Scope), its seed, the constructor print replacing the estimates in Model and Cost; the control's config and launcher; the per-save read scripts and ladders moved from gitignored `ckpts/inject` into tracked `scripts/reads` | 0.5 d | the print against Model | L1 |
| L5 | the corpus pipeline for the cluster on the existing builders (`prepare_data.py` for the filler mix, `prepare_injection_data.py` for the biographies and the warm-up split, `prepare_evidence_data.py` for QA, `prepare_chain_data.py` for the chains and the bridge rows, `merge_evidence_splits.py` for the three merges at the shares in Scope with `--support-gold-only` on), as one tracked script with a source list sized to the token target, a dry-run mode on a tiny slice, and a pilot-shape smoke under 5 hours local | 1 d; running 2026-10-26 | the dry run; the smoke's packing, fill, supported share and copy emergence | nothing |
| L6 | cluster day 1 (2026-10-28): the 2-GPU against 1-GPU check, the corpus build on cluster CPUs, the 10B control (about 4 hours on a node), the main run's warm-up launch; plus (2026-10-26): the HF token with Nemotron access accepted, the user's confirmation of `--chain-share` (rows 12 and 13), and the first cluster job `build_run2_corpus.py --stage filler` | 1 d on the cluster | the 2-GPU match; the control's first reads | L3, L4, L5, the token, the confirmations (2026-10-27: the cluster is MareNostrum 5 with no internet on any node, so the HF token is used on the PC by `scripts/mirror_sources.py` and the filler reaches the cluster through the transfer nodes; the day 1 list is now `scripts/slurm/README.md` "Day 1" and [RUN2_handover.md](RUN2_handover.md)) |

**L1 to L5 done 2026-10-26** (uncommitted in the tree; the commit line is the main thread's):

- **L1:** the IR expert's inner attention is no longer built when `direct_read` is on (every config); an old checkpoint's `moe.experts.{i}.attn.*` keys of IR experts are dropped on load by `utils.drop_ir_inner_attn` (called from `load_model_state` and the migrate scripts; the strict resume paths stay strict, so a run checkpoint written before today cannot resume, none needs to); `flops_per_token_fwd` no longer counts the IR attention and `evidence_prefix_attn_flops` bills the key/value reader's prefix attention (not wired into a trainer); `detach_loops` clamped to `n_loops - 1` and the sampled depth; the fp32 balance shadow stamped on the buffer's `_version`, `data_ptr` and device; `check_store_embed_dim` asserts `ir_dim == 384` with the port. 24 test files pass; `seed_micro.pt`, the warm-up arm's 100M save and arm P's final load strictly with logits bit-identical to HEAD; a detached Sonnet review found no high or medium issue (two lows folded in). Prints: `config_micro` 34.9M / 31.3M / 61M, `config_pilot` 97.4M / 83.3M / 227M, `config.yaml` 380.2M / 221.0M / 464M.
- **L2:** grid snap at every save and eval site (`next_grid_point`, now in `pretrain.py`), resume verification with exit 30 on the sft path, `total_steps`, `rows_per_epoch` and `world_size` stored in the checkpoint and warned on change, the groundedness gradient-checkpointing tripwire.
- **L3:** `sft.py --evidence` multi-GPU, one process per GPU: interleaved per-rank document shares in `SFTDataset`, lockstep loaders, gradients averaged in fp32 buckets by `pretrain.all_reduce_grads` (no DDP wrap: the LM heads run outside the model forward), one float64 all-reduce per log step for counts, stop requests and loss means, balance counts summed, rank 0 alone writes saves behind barriers, sharded evals merged, `keep_every_tokens`. A detached Sonnet review found two high and two medium findings, all folded in. The single-process path is unchanged; the 2-GPU against 1-GPU check cannot be bit-exact and runs in the cluster's first hour.
- **L4:** `config_run2.yaml`, `config_run2_control.yaml` (differs in split, token budget and warm-up fraction only), the seed, `scripts/launch_run2.sh`, `scripts/launch_run2_control.sh`, `scripts/reads/` (README, `read_run2_save.sh`, `stop_after_save_run2.sh`, verbatim copies of the pilot read scripts), `tests/test_run2_config.py`, `tests/test_run2_launchers.py`. The yaml carries the decided overrides (weight decay 0.1, loss-free, `segment`, 256 entries read exactly, no contrastive term, 32 experts top 2, 3 loops, AdamW, 8 x 2 per GPU, lr 4e-4, warm-up 1000 steps, cosine to 0.1, seq 4096, `max_evidence_tokens` 9216, saves every 1B, evals every 2B (one cadence key exists, RUN2 wanted 1B and 2B), the kill rule, `keep_every_tokens` 10B with 2 kept, `num_epochs` 3); pilot values where this spec was silent. 30-step smoke at shape L on the 5090: 25.4k to 27.4k tokens per second, 16.5 GB peak, step 0 CE 11.17, step 30 10.96, save and exit 10 clean. Print in Model.
- **L5:** `scripts/build_run2_corpus.py` and `tests/test_build_run2_corpus.py`, the sheet [RUN2_corpus.md](RUN2_corpus.md): stages filler, bios_main, bios_warm, qa, chains_plain, chains_bridge, merge_warm, merge_main, merge_control, heldout_qa, heldout_chains, val_clean, store_openqa, store_edit, each idempotent with a marker, `--dry-run`, `--scale`, `--stage`; T = 93.33B per split, chain share 0.03, per-slice targets bios 89.0B, qa 1.53B, plain chains 1.4B, bridge chains 1.4B, control 10B. The smoke at scale 0.0005 ran every stage in 23.7 minutes with `md5_by_slice` agreeing across the warm and main merges. Not run locally: the filler download and the two store builds. The filler at 88.5B is reachable at 1.00x repeat only with fineweb's prefix widened to all crawls (the builder as coded has one crawl, 14.8B; the driver patches it in memory). Missing builder flags are listed in the sheet (section 5): `--order-from-prefix` so the warm merge need not be built at full size, the hop-weight flag, shard flags.

**Not on the launch path, stated explicitly.** The new builder features of the old item 4 (per-window retrieval over plain text, the fact tagger, the near-duplicate filter, the staircase visibility question) are not built for this run: it uses the pilot recipe at scale. The staircase question is moot without per-window retrieval over plain text; its decisions-table row (row 10) stays, marked "not on the launch path, revisit for a later run".

**Built 2026-10-27** (uncommitted in the tree; for MareNostrum 5 and the 800B target, row 14):

- **The corpus driver at 800B** (`scripts/build_run2_corpus.py`, [RUN2_corpus.md](RUN2_corpus.md)): the stages filler, qa, bios_main, val_clean, bios_full, bios_warm, prune_filler, chains_plain, chains_bridge, qa_shuffle, order, merge_main, merge_warm, merge_control, rows, prune_sources, the held-out stages and the stores; the warm and control splits built as prefix merges of the main order; `rows` writes the packed row estimates for 1, 2, 4, 8 and 16 ranks (the trainer exits 2 without the entry for its rank count); every stage over 60 hours resumable after a kill, a `flock` per stage, SIGTERM forwarded with exit 20, a 300 s heartbeat and a `last_child_output` progress file the corpus job's stall check reads; `plan` refuses a disk plan over budget; `check-offline` lists what a stage lacks without internet. The builders made resumable: `prepare_injection_data.py` (state file, `--stop-after-tokens` prefixes), `prepare_chain_data.py` (streaming and resumable; the `seen_compositions` docstring corrected: tuples of 3 hops or more, 4 included, drop the held-out pairs, as the code already did, which matters now that the bridge trains hops 1 to 4), `merge_evidence_splits.py` (resume and the prefix mode). `scripts/mirror_sources.py`: `list`, `stage-small` (tokenizer, bge, the QA and store datasets into a staged `HF_HOME`) and `push` (the filler shards from the PC through a transfer node into the build's inbox, demand driven, in a 120 GB window, under the 2 TB limit). The smoke at scale 0.0005 on the final code: every stage ok, 28.5 minutes, 2.8 GB ([RUN2_corpus.md](RUN2_corpus.md) section 7).
- **The trainer for chained jobs** (`scripts/sft.py`, `modules/data/sft_dataset.py`, `scripts/pretrain.py`): `--plan-split` (the schedule fitted to the main split from step 0), `--max-epochs-this-split` (the warm launch passes 1: save, "warm split exhausted", exit 10), `--grad-accumulation-steps`, `--estimate-rows-only` (CPU), `--keep-stop`; the switch and `epoch0_head_docs` checks (a resume on the same split under another head is refused); the rows cache, version 2, keyed on each index file's size and a sha1 of its first and last megabyte (no mtime, so a copied corpus keeps its cache); a moved `data_dir` with the same split is a plain resume; `all_reduce_grads` leaves a parameter with no gradient on any rank at `grad` None, as on one GPU (one summed flag vector per call); log steps only on sync steps.
- **The Slurm job chain** (`scripts/slurm/`, runbook `scripts/slurm/README.md`): `submit_chain.sh TRAIN|CONTROL|CORPUS N` presubmits N jobs with `afterany` under one chain id, each job runs a precheck, takes a `flock` on the run directory and runs the trainer in the Singularity image (`container/`, built off cluster), one task per GPU (`rank_exec.sh`); a save 20 minutes before the 72 hour limit (exit 20, the next job resumes); the watchdog (Early signals); HALT with the reason on every failure a rerun would repeat (no container runtime, a bad `STOP_AFTER`, unfilled site values, the trainer's exit 1 or 2, the kill rule, exit 30, rank 0 exit 0 without a final); a cancel by hand respected (never resubmitted); `reads.sbatch` (four reads to a node, optional weight export), `corpus.sbatch` (stage groups with a 120 minute stall check), `status.sh` and `status.py` (one JSON, GPU hours from `sacct`).
- **The PC-side ops** (`scripts/ops/`, run in WSL over ssh master sockets the user opens by hand): `push_code.sh`, `push_corpus.sh`, `pull_exports.sh` (sha256 verified, optional private HF upload) with a transfer ledger refusing a push past 2 TB; `supervise.py`, polled every 30 to 60 minutes by a local Claude Code `/loop` under the standing instructions `scripts/ops/SUPERVISOR.md` (resubmits a broken chain at most once an hour, pins the 10B saves, batches the reads, alerts on everything else, never cancels, deletes, switches or restarts after a HALT); `scripts/export_weights.py` (bf16 weights, about 0.67 GB, with a strict load check).
- **Configs** (`config_run2.yaml`, `config_run2_control.yaml`): the Schedule note above.
- **Reviews**: three Opus final reviewers (Slurm and ops, corpus, trainer and contracts) found no high-severity issue left in the corpus and the trainer; the Slurm review's two highs (supervisor actions silently skipped, a finite divergence unseen) and every medium and low were fixed; a detached Sonnet review of the trainer fixes found no high or medium issue. Tests added: `test_epoch0_head_order.py`, `test_mirror_sources.py`, `test_ops_tools.py`, `test_sft_contracts_2026_10_27.py`, `test_slurm_chain.py`. Nothing has run under real Slurm: the day 1 checks (the `module` command in a non-interactive shell, one supervisor action seen in `squeue`, the signal path, the 2-GPU against 1-GPU check stopped after about 100 steps, the first hour's rate against 54k) are the user's, in [RUN2_handover.md](RUN2_handover.md).

### B. During the run, on the cluster or beside it (not blocking the launch)

| item | what | estimate |
|---|---|---|
| B1 | the checkpoint shape record: the yaml-only keys (`top_k`, `evidence_encoder`, `max_seq_len`, heads x head_dim, `ir_read_top_k`) saved in the payload and checked on load | 2 h |
| B2 | the logging builds: the supported-span share, the MFU line, the ev pad share | 1 h |
| B3 | the evidence builder's rng checkpointing and the empty-chunk re-check | 2 h |
| B4 | the remaining eval items: BoolQ majority chance and the 8-task mean, gsm8k numeric EM, the pad-column assert, `eval_store --chunk-order` and the `recall --loop` check, the `eval_exit --pair` token check | 0.5 d |
| B5 | the two reads when a card is free: the copy read with 3 distractors on arm P's final and the warm-up arm's 100M save (about 20 min); A7 pathway and A4 edit with shuffled chunks (about 30 min) | 1 h |
| B6 | the rebuilt chain eval sidecars copied into `data/prepared`: rebuild the three eval splits into a scratch dir, byte-diff every file but `.chains.jsonl` against `data/prepared`, copy only the three `.chains.jsonl` | 1 h |
| B7 | the value-to-name closed-book probe | 0.5 d |
| B8 | a profile of one training step at shape L on the 5090 at batch 2, to replace the efficiency estimates (a local run of minutes) | 1 h |

**6a throughput, optional, none required.** Each non-bit-identical item is a 40-minute micro A/B against its twin; every percentage is an estimate until B8.

| item | what | estimate |
|---|---|---|
| 3.1 | the bit-identical set: the head's chunk-checkpoint skip behind a flag, worker-side `cu_seqlens`, pinned memory, one `tolist` in the log block, fused AdamW | 0.5 d, about 3 to 5% |
| 3.2 | pad tokens routed to the null slot | 3 h, 1 to 3% |
| 3.3 | the flat evidence axis replacing padding to the batch max (the pad share measured first) | 1 to 2 d, 5 to 15% |
| 3.4 | best-fit packing over a lookahead buffer, the estimator replaying it | 1 to 2 d, 8 to 10% |
| 3.5 | the eval-pass rewrite | 0.5 to 1 d, 3 to 6% down to under 1% |
| 3.6 | the fused linear plus cross-entropy head and the subsampled MTP head | 1 to 2 d, 5 to 7% |
| 3.7 | `torch.compile` on the decoder, shared MLP and norms, with `S_ev` bucketed | 1 to 2 d, size unknown until B8 |
| 3.8 | the FP8 A/B (2 x 2B tokens) and the G8 hour | on the cluster |
| 3.9 | the fully unpadded layout, only if the flat axis and packing fall short | 2 to 4 d |
| 3.10 | encoder depth (`evidence_encoder` 4 against 8, 27% of step compute against 13%): a recipe decision, read on an arm | the user's call |

### C. After the main run

| item | what | estimate |
|---|---|---|
| C1 | the HotpotQA bridge instrument (the bridge decode over retrieved real chunks, scored by read sites) | 1 d |
| C2 | the 6f mix builder and launcher with the A-gate and clean-CE kill rule | 1 d |
| C3 | 6f itself | its own plan |
| C4 | the A-gate ladder on the shipped checkpoint | a read |

Totals: the launch path is about 3 days of build across parallel lanes, 2 days wall clock; nothing else gates the launch. Status at the 2026-10-25 hand-over, kept for the record: landed and tested were the `prepare_data.py` sha1 shard seed and required `--manifest-key`, `closed_book_rank.py --distractors K --seed`, the `eval_chains.py` bridge-mode and shortcut items (measured shortcut `penultimate_fed` against 1/K: `chains_eval` 0.256 / 0.185 at 2 hops, 0.294 / 0.206 at 3; `chains_heldout_tmpl` 0.257 / 0.191; `chains_hop4` 0.301 / 0.210) and the docs sweep; the model set was in the tree unverified and the trainer set was restored to HEAD (its fixes are L2).

## Decision table

| copies by 28B | port at 10B | A1(b) at 10B | A3 likelihood | next |
|---|---|---|---|---|
| no | any | not read | any | stop; the copy curve is the finding; the copy recipe at this scale is decided before any more spend |
| yes | gold gain under 1.6 | any | any | stop; the port does not recover evidence at this scale |
| yes | holds | fails at tier 100 on either form | any | stop before the main spend; the leak is understood first |
| yes | holds | the control not above its prior | any | A1(b) unreadable at 10B; continue, read the control's level again at its end |
| yes | holds | holds | decays on three reads | stop and decide (context reliance) |
| yes | holds | holds | holds | run to the target; finals read; 6e, then 6f; the model's own retrieval revisited on this checkpoint |

A2 to A7 and the language rows failing with the rows above passing are recorded per gate, each a
named target of 6e or 6f (A3 and A4's edit half to 6f's abstention and conflict rewards).

## Known residuals and caveats the application states

- **The tier 1000 leak**: entity `delta` 0.070 to 0.099 at micro (z 5.0 to 6.7) and 0.041 to 0.064
  at pilot shape, known since 2026-10-05, accepted 2026-10-06 ([r0b], [leak], [pilot]); five
  levers tried (the warm-up itself, swap rate, placeholders, the first name mention, collision
  spans), none moved it by 3 sigma. Removed by 66M tokens of chain SFT (R5, [r5]: entity `delta`
  at tier 1000 -0.000 (z 0.0) in distribution and 0.006 (z 0.7) held out on the bridge final,
  every entity cell at chance), so the binding leak read is after post-training.
- **The tier 100 margin is thin at micro on today's code**: held out z 3.1 / 3.3 / 3.7 on the three
  same-recipe arms of 2026-10-25, one seed each ([leak]); accepted 2026-10-25 as the recipe's known
  state on one seed, no second seed (row 9).
- **The retriever gate fails**: the model's own query reaches recall@5 0.004 against bge 0.829 at
  pilot ([pilot]) and at most 0.059 (about 7% of bge) with the term at micro ([infonce]); serving
  uses bge.
- **Abstention is weak in generation** (A3): EM 0.05 and 77% abstention under gold at pilot; the
  likelihoods are right ([pilot]); 6f's three-way reward is the step that must not fail. After
  chain-only bridge SFT the model abstains on nothing and answers nothing (EM 0.000, [r5]).
- **The eval-pass rewrite is deferred to 6a** (about 18% of a pilot arm's wall clock, [review]).
- **The throughput start is the 16B run's 11% MFU** ([16B]); every hour in Cost scales with it.

## Limitations

- One seed at shape L; the seed sigma is unmeasured at pilot and at L.
- The optional segment-weighting warm-up arm on the leak instrument (the `cfb` recipe under `segment`, about 1.5 h local) is not scheduled; `segment` has never been read on the leak instrument.
- The only measured post-training is single-format: the bridge final's ladder (R5, [r5]), 66M
  tokens of chain-only SFT at 3e-4 on arm P's final, kept composition in written tokens (0.997
  held out) and erased the biographies, half the held-out copying, answering and abstention on
  real questions and plain language (clean-slice CE 3.27 to 9.06). 6f's mixed recipe is unread at
  any rung.
- Retrieval on natural text, the fact tagger and A1(a) are unread at any rung; the smoke (Build 6)
  is the only read before the cluster.
- The selector is the pilot's routed IR expert with a learned 256-entry table, not the blueprint's
  always-on selector; the selector results transfer as measured at pilot shape only.
- 524,288 tokens per update is 16x the pilot's; copying emergence in tokens may move.
- `segment` weighting and `--support-gold-only` were read on one seed each and never together.
- No counterfactual or partial-hop QA condition, no MuSiQue or 2Wiki source; A6 through the port
  has no evidence path in `eval_benchmarks.py`.
- The tail shares, the chain share at scale (10% is 28B tokens of templates against the pilot's
  0.1B) and the chat replay's source size have no rung behind them.

## Sources

[pilot]: PILOT.md
[r0b]: ../measurements/r0b_micro_pilot.md
[e4]: ../measurements/expert_count_pilot.md
[bridge]: ../measurements/written_bridge_pilot.md
[r5]: ../measurements/r5_bridge_ladder.md
[chain]: ../measurements/chain_continuation_pilot.md
[lossfree]: ../measurements/lossfree_micro.md
[muon]: ../measurements/muon_micro.md
[infonce]: ../measurements/infonce_micro.md
[leak]: ../measurements/leak_candidates_micro.md
[clean]: ../measurements/clean_validation_slice.md
[depth]: ../measurements/depth_memory_probe.md
[qhead]: ../measurements/query_head_probe.md
[seedinstr]: ../measurements/seed_instruments.md
[review]: ../review_2026-10-24.md
[arch]: ../architecture.md
[16B]: ../CONCLUSION.md
[6a]: NEXT.md#6a-throughput-first
[6b]: NEXT.md#6b-budget
[6c]: NEXT.md#6c-the-corpus-at-100b-tokens
[6d]: NEXT.md#6d-the-run-spec-docsplansrun2md
[6e]: NEXT.md#6e-after-the-main-run
[6f]: NEXT.md#6f-reasoning-post-training

Keys used above: [pilot] PILOT.md, [r0b] r0b_micro_pilot.md, [e4] expert_count_pilot.md, [bridge]
written_bridge_pilot.md (the A4 bge read is recorded there and in NEXT.md), [r5]
r5_bridge_ladder.md (the A-gate ladder on the bridge final), [chain]
chain_continuation_pilot.md, [lossfree], [muon], [infonce], [leak] leak_candidates_micro.md
(including the `cfb` twin), [clean] clean_validation_slice.md, [depth] depth_memory_probe.md,
[qhead] query_head_probe.md, [seedinstr] seed_instruments.md, [review] review_2026-10-24.md, [arch]
architecture.md, [16B] CONCLUSION.md, [6a] to [6f] the NEXT.md Phase 6 sections. CLAUDE.md is cited
by section name.

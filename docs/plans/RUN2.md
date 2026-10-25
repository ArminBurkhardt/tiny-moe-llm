# The real run (Phase 6): the spec, drafted and decided 2026-10-25 for the cluster application

Status 2026-10-25: the spec, written the evening every pilot read closed (NEXT.md "Now"), for the
cluster application; its open decisions were decided by the user the same night, every row on the
recommended option. Section order as [PILOT.md][pilot]. Every number names its source note (keys
at the end); a number marked "estimate" is arithmetic on cited figures and stays one until the
constructor, the builder or the first hour of the run prints it. Decided items carry their date
and source; the user's decisions of 2026-10-25 are the first table, and the body, written under
the options that were then decided, is marked "(decided 2026-10-25)" where it depends on them.

## Decisions (the user, 2026-10-25)

Rows 1 to 7 are the main thread's; rows 0, 8 and 9 were raised while drafting. Every row was
decided on the option the draft recommended; the consequence column is the draft's, kept.

| # | decision | decided | consequence of each option |
|---|---|---|---|
| 0 | which shape is shape L | **as configured in `config.yaml`**: width 768, not the blueprint's 1024 (8 prelude layers, 32 experts, 3 loops), the only shape the constructor has printed and the 16B run measured | configured: 383.5M total / 224.2M active, about 502M FLOP per token (CLAUDE.md), the 16B run's 97k tokens per second at 11% MFU on this width ([16B]); 280B tokens cost about 1.1k to 1.3k H100 hours (Cost). The plan's blueprint (width 1024, about 470M / 300M, NEXT.md Decisions "shape"): the shape [6b]'s 2.5k hours for 270 to 290B tokens was computed for, with its coda dropped and its sublayers open (row 5); no constructor print, no measured rate |
| 1 | routed expert count | **keep 32**, decided on the cluster's memory and the hours, not on quality (neither binds) | 32: about 333M total (with the 256-entry selector table, Model), 5.3 GB of training state per GPU, 4.7 GB per save. 16: about 249M total, 4.0 GB of state, 3.5 GB per save; the same active compute and FLOP, so the same hours (arm E4 ran at arm P's rate, 44k to 48k against 42k to 48k tokens per second, [e4]); quality equal at pilot shape (held-out CE 0.007 behind on clean text, [clean]) on one seed with two changes |
| 2 | `evidence_selection_weighting: segment` as the default | **yes**: `segment` is the real run's default | yes: 0.033 nats better on the QA slice and 0.060 on plain text at exit 3 (z +8.1 per token), selection equal, one seed ([infonce]); never read on the leak instrument, so the switch criterion and A1(b) reads carry it. A micro warm-up arm under `segment` (the `cfb` recipe, about 1.5 hours local) would read the leak before launch. No: the run stays on what every recorded arm ran (`token`) |
| 3 | `--support-gold-only` as a data rule | **yes**: carried into the real corpus builder as a data rule, **a build item behind a flag**; scope the biography-style supervised spans; the QA and natural-text analog (a value supported only by a chunk that also carries its subject) is a builder change | yes: held-out swapped-card following 0.67 to 0.79, held-out gold top-1 0.60 to 0.74 against 0.79 to 0.86, no language or leak cost; copying at the 100M switch 0.88 to 0.90 against 0.95 to 0.96 for the same-code twin, so the flags slowed copying ([leak]), one seed. No: following stays at about 0.67 held out |
| 4 | the goal's reasoning clause | **reworded** to "facts in the store through the retrieval pathway, reasoning in written tokens, the looped trunk as computation per token"; the original clause, "reasoning in the looped trunk" (2026-09-30), is kept on record as the original goal and the aspiration to revisit after the real run (NEXT.md "The goal") | yes: L1 becomes descriptive and the written bridge read (R9's form) becomes the reasoning gate (Reads and gates); matches the measurement (latent composition at chance at every read, [chain]; written 0.997 held out, [bridge]). No: L1 stays a gate the pilot already failed at this shape |
| 5 | R4b's sublayer half | **today's one block per pass**; the three-sublayer core is not built | one block: no build; one read site per pass, three per forward. Three sublayers: a day of architecture work plus review and two 1B arms ([review]), at a noise floor that read 4 against 8 experts as equal ([e4]); the pilot never showed the loop composing |
| 6 | the token target | **280B tokens**, the hours derived; the application asks for 2.5k H100 hours plus the corpus build line plus a 20% buffer (about 3.0k), the 5k ceiling stated; G8 (2x) turns the same hours into the 400B data cap | the main thread's "280B for 2.5k hours" is [6b]'s figure for the blueprint width. At the configured shape (row 0) 280B costs about 1.1k to 1.3k H100 hours at today's MFU and the 400B cap about 1.6k to 1.9k (Cost), so the ask covers the 400B cap at today's MFU; raising the target to 400B uses that headroom, keeping 280B leaves it for 6a, 6e and the second arm a failed read would need |
| 7 | R5 (the winner's eval set) | **tonight, on the local card, on arm P's final** (8 experts) and the bridge final, launched by the main thread | tonight: needs no decision beyond the shape, about 2 to 4 hours (estimate from the pilot ladder's 45 minutes for twelve reads plus the chain reads, [pilot]), and its margins enter the gate table before the run. On the cluster: costs H100 hours and lands after the application |
| 8 | the A1 control at the real run's size | **a short matched control**: no retrieval, the same stream and shape, to 10B tokens, about 30 H100 hours (about 4 hours on an 8-GPU node if DDP scales linearly) | short control: A1(b)'s paired reading (decided 2026-10-17) and A1(a) read at matched 10B; the final saves read against the prior and the 10B level. No control: A1 rests on the pilot's 47x / 4x reduction ([pilot]) and the 16B checkpoint, unmatched. A full-length control doubles the run |
| 9 | the thin tier 100 margin at micro | **accepted** as the recipe's known state on one seed, stated in the caveats; no second seed of `cfb` | held-out tier 100 at z 3.1 / 3.3 / 3.7 on the three same-recipe arms of 2026-10-25, paired inside noise against the recorded arm (z -1.8) ([leak]); a second seed says whether the R0b rule's tier 100 half holds on today's code |

## Question, and what it decides

Does the pathway recipe that passed at pilot shape (externalization against a matched control, a
copying reader, a port worth more than its in-context ceiling) hold at shape L over 280B tokens,
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
| optimizer | AdamW | R7, decided 2026-10-21: Muon stored the biographies about twice as early, leak 1.6x / 2.4x by sign ([muon]) |
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

`config_run2.yaml` (to build): `config.yaml`'s model block with the pilot's selector table.

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
| `config.yaml` as printed | 383.5M | 224.2M | 6.1 GB | 5.4 GB |
| 32 experts, 256-entry table (decided 2026-10-25) | about 333M | about 174M | 5.3 GB | 4.7 GB |
| 16 experts, 256-entry table | about 249M | about 174M | 4.0 GB | 3.5 GB |

Derivation (estimate): one routed expert is 3 x 768 x 2304 = 5.31M parameters, and 383.5M minus
224.2M is exactly 30 idle experts, so the 65,536 x 384 key and value tables (50.3M) sit in both
counts. Training state: bf16 weights and gradients, fp32 masters (every parameter, as `sft.py`),
fp32 AdamW moments; a save drops the gradients. FLOP per token about 502M as printed plus 118M per
evidence token (CLAUDE.md); the table cut changes it by about 1M (the `config.yaml` comment).
Activations, not state, set the peak: the 16B run peaked at 29 GB at batch 8 x 4096 without
evidence ([16B]), and the evidence axis adds about 0.61 MiB per evidence token (CLAUDE.md), about
30 GB at batch 8 and ratio 1.55, so about 60 GB on an 80 GB card (estimate). Neither expert count
binds.

## Corpus

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
  ratio 1.55, as the pilot's 4608 was against 0.72; estimate), set from the packing line in minute
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
| port: gold gain under 0.1 nats or content gain under 0 at the first fixed eval past 5B | `[eval fixed]`, the built kill (`kill_tokens` 5B, `kill_min_gain` 0.1) | automatic stop (exit 10); the pilot read +2.02 at 75M ([pilot]) |
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
| reasoning (row 4) | `eval_chains.py --generate-bridge`, held-out 2-hop and 3-hop, 4-hop | 3 sigma above chance with the 1-hop gate kept | paired bootstrap | 0.997 against 0.191 (z 64.7) after 20M tokens on the form ([bridge]); the same checkpoint failed every other gate (R5, [r5]) | every 10B |
| L1 | latent read by kept sites | descriptive (row 4) | 0.005 to 0.008 paired | +0.002 +- 0.005 ([pilot]); +0.004 +- 0.008 ([chain]) | finals |
| L2 | HotpotQA bridge CE by read sites; held-out 4-hop by sites | at least 0.05 nats on bridge CE | paired | unread at pilot; exit gain 1 to 3 about 0.08 on clean text ([clean]) | finals (R5, 2026-10-25: held-out 4-hop at chance by read sites on arm P and with the bridge written; the HotpotQA bridge has no instrument, Build list 9, [r5]) |
| language | clean slice CE per exit; 13-task suite against the 16B snapshot and the peers | within noise of or above the 16B run | 0.029 CE; binomial | arm P equal to the control on clean text (0.002) and within two sigma on the suite ([clean], [pilot]) | every 10B; suite at the finals |
| G8 | tokens per second at shape L over an hour, checkpointing and upload live | at least 2x today's | | | 6a, before the launch |

## Cost

Application table (H100 hours). Every line is an estimate until the constructor prints the shape
and the first hour prints the rate. Rate at the configured shape with evidence: 502M + 118M x 1.55
= 685M FLOP per token forward; at the 16B run's achieved rate (97k tokens per second at 490M,
[16B]) that is 69k tokens per second, times 0.91 for the evidence path's lower efficiency at pilot
shape (arm P 42k to 48k against arm C 51k to 59k at 1.11x the FLOP, [pilot]): about 63k, range 60k
to 70k per H100. The blueprint figure is [6b]'s 30k to 32k.

| line | configured shape (row 0) | blueprint width 1024 ([6b]) | basis |
|---|---|---|---|
| main run, 280B tokens, today's MFU | 1,100 to 1,300 | 2,400 to 2,600 | the rates above |
| main run at the 400B cap instead | 1,600 to 1,900 | 3,500 to 3,700 | the same |
| main run after G8 (2x) | half of the line | half of the line | [6a] |
| short control, 10B, no evidence (row 8) | about 30 | about 70 | 97k / about 40k tokens per second ([6b]'s rate without its 1.29 evidence factor) |
| corpus build, GPU | about 25 to 30 | the same | fact tagger about 15, bge window queries about 5, store embedding and ANN a few ([6c]) |
| corpus build, CPU | 1 to 2 node days | the same | tokenization of 100B, about 400M ANN queries, near-duplicate filter, store deduplication ([6c]); unmeasured |
| 6a throughput work | 100 to 150 billed per GPU; about 580 if a full 8-GPU node is held 3 days | the same | profiling and compile on one GPU about 3 days, the FP8 A/B (2 x 2B tokens) about 18, G8 one node hour |
| eval ladder | 30 to 50 | 40 to 70 | the pilot's read set 26 minutes per save and the ladder 45 minutes ([pilot]), at about 2x the FLOP per token, every 10B plus the finals |
| 6e context extension and consolidation | 40 to 60 | 80 to 120 | a few B tokens each ([6e]); a context change is also a corpus rebuild (CPU) |
| 6f | not asked | not asked | the rollout rate is unmeasured ([6f]) |
| ledger, 280B, before buffer | about 1,350 to 1,650 | about 2,700 to 3,050 | sum (6a per GPU) |
| **ask (row 6, decided 2026-10-25)** | **2.5k + corpus line + 20%: about 3.0k H100 hours; ceiling 5k** | the same | configured: the ledger with its own 20% leaves about 1.0k to 1.4k at 280B, and the 400B cap fits (2.2k to 2.6k with 20%); blueprint: the ask covers the ledger without its buffer |

- **Wall clock**: on 8 H100s about 6.4 days for 280B at 63k per GPU, 3.2 on 16, assuming linear
  DDP scaling (unmeasured: the evidence trainer is single GPU today, Build list 3).
- **Storage**: 4.7 GB per save (configured, 256-entry table: 0.67 GB bf16 weights, 1.33 GB fp32
  masters, 2.67 GB fp32 moments; the masters are saved since 2026-10-24, CLAUDE.md "SFT"); 30
  kept saves about 140 GB; 280 uploads at the 1B cadence about 1.3 TB moved. Corpus in today's
  per-occurrence format about 1.5 TB (200 GB prompt, 310 GB evidence tokens at ratio 1.55, about
  930 GB of fp16 chunk keys at 768 bytes per 128-token chunk occurrence); about 0.5 TB with chunks
  referenced by store id (a builder option). All estimates.

## Build list before launch, in order

1. **Done 2026-10-25: R5** (row 7, [r5]). Arm P's final already carried A1 to A7 (the decode
   reads on the fixed packer) and its 4-hop reads by read sites (at chance at every save); the
   bridge final carried the 4-hop generate read (0.206 against chance 0.210). New: the A-gate
   ladder on the bridge final, the checkpoint form 6f builds on: a demonstration that composition
   happens in written tokens, not a usable checkpoint (the 6f mix rule in Schedule, the
   post-training reads in Reads and gates). The HotpotQA bridge by read sites has no instrument
   (item 9).
2. `config_run2.yaml` and its seed; the constructor print (total, active, FLOP per token, per
   evidence token) recorded here, replacing every estimate in Model and Cost.
3. **Multi-GPU on the evidence trainer.** `sft.py` is single GPU and `EvidenceDataset` has no
   per-rank sharding today (no rank or process index in either): rank sharding, DDP under
   accelerate, the token tracker across ranks, and the supervisor's lifecycle (preemption exit 20,
   upload, retention, `verify_resume`) on the `--evidence` path; a 2-GPU run of 100 steps against
   one GPU.
4. **The real corpus builder** ([6c]): per-window bge retrieval on natural text, the fact tagger
   and span weights, swaps and the warm-up form, tail anonymization, the near-duplicate filter,
   the store at 0.4B tokens or more, the biographies with tiers counted over the run's epochs,
   `--support-gold-only` (row 3), the bridge chains at hops 1 to 4 (`prepare_chain_data.py
   --bridge`), the 4093 cap, the build metrics, interruption safe. **Blocking, unbuilt:** the
   design's staircase visibility (a window sees only the chunks retrieved up to it) has no form
   under the key/value reader, whose prefix is visible in full to every token of its segment
   (`attention.prefix_varlen_attention`); without it a token reads chunks retrieved for later
   windows. Options: retrieve once per document from its first 64 tokens and mask that opening's
   loss (no kernel work; this draft's recommendation), or a windowed prefix mask (kernel work, 6a
   era).
5. Logging builds: the supported-span share and the MFU line.
6. A pilot-shape smoke of the real builder, local, one 1B-token arm on the 5090 (about 7 hours,
   arm P's rate, [pilot]): packing, fill, supported share, copy emergence, the A1(a) direction.
   Raised by this draft; the natural-text retrieval slice is otherwise first read on the cluster.
7. The control's config and launcher (row 8); the read scripts per save at shape L, timed on the
   first save.
8. 6a, after the spec ([6a], decided 2026-10-21): profile, compile, host syncs, the eval-pass
   rewrite (deferred to 6a 2026-10-24), FP8 A/B; gate G8.
9. **The HotpotQA bridge instrument** (L2 and the reasoning row on real questions, raised by R5,
   [r5]): the bridge decode of `eval_chains.py --generate-bridge` over retrieved real chunks
   (HotpotQA bridge questions with the bge buffer attached), scored by read sites. Today the decode
   exists only for the synthetic chain splits. Needed by the finals, not by the launch.
10. **The 6f mix builder** (raised by R5, [r5]): the bridge rows merged with the pretraining slices
   (evidence QA with its abstention rows, the biographies at their tier structure, plain-answer
   QA, plain text) through `merge_evidence_splits.py`, extended to `.evhop` and `.factspan` (NEXT.md
   6f), and a 6f launcher that reads the A gates and the clean-slice CE per save with their kill
   rule. Needed by 6f, not by the launch.

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

- **The tier 1000 leak**: entity `delta` 0.07 to 0.085 at micro (z 5.0 to 6.4) and 0.046 to 0.064
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

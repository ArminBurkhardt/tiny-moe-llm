# R5 at pilot shape (2026-10-25): the A-gate ladder on the written-bridge final

Rung R5 ([NEXT.md](../plans/NEXT.md), the R5 ladder row: "the winner, eval only": held-out 4-hop
by read sites and at D = 4, HotpotQA bridge by read sites, A1 to A7), decided for this evening on
the local card ([RUN2.md](../plans/RUN2.md), decisions row 7). The verdict below is the main
thread's.

## Purpose

R5 had three parts, and two of them were already on record when it was launched:

- **Arm P's final carries the complete ladder.** A1 to A7 were read 2026-10-14
  ([PILOT.md](../plans/PILOT.md) "Run", `ckpts/inject/ladder_pilot.sh`, outputs
  `ckpts/inject/pilot_{p,c}_ladder_*`); the decode reads (A3 generation, A7 pathway, A4 edit) were
  redone on the fixed eval packer 2026-10-24 (`ckpts/inject/ladder_pilot_redo.sh`,
  `pilot_p_ladder2_*`), and the A4 flip on the bge route the same day
  ([written_bridge_pilot.md](written_bridge_pilot.md)). Its 4-hop reads at depths 3 and 4 by read
  sites sit at chance at every save (final 0.18 to 0.19 against chance 0.210,
  `ckpts/inject/pilot_p_final_chains_hop4.log`).
- **The bridge final carries the 4-hop generate read** (`ckpts/inject/pilot_bridge_final_gen_hop4.*`,
  2026-10-24): 0.206 against chance 0.210, the model writes only the two steps it was trained on.
- **"HotpotQA bridge by read sites" has no instrument.** The bridge decode
  (`eval_chains.py --generate-bridge`) exists only for the synthetic chain splits; a bridge decode
  over retrieved real chunks is a build item (RUN2.md, Build list), not a read.

The new content run tonight is the A-gate ladder on the written-bridge final,
`ckpts/evidence_pilot_bridge/checkpoint_evidence_final.pt`: arm P's final plus 66.35M tokens of
chain-only bridge SFT (R9, [written_bridge_pilot.md](written_bridge_pilot.md)). That is the
checkpoint form 6f builds on (a pretrained trunk, then reasoning SFT in written tokens), so the
read asks whether the SFT keeps the gates the pretrained trunk passed.

## Setup

- Script `ckpts/inject/r5_bridge_ladder.sh` (launched 18:57 detached; one process per read, each
  skipped on relaunch when its json exists; `TINY_LLM_CONFIG=config_pilot.yaml`), log
  `ckpts/inject/r5_bridge_ladder.log`. Every read exit 0: closed book 293 s, gold 263 s, swapped
  287 s, ceiling 119 s, abstention 400 s, pathway 1162 s, bge edit 185 s, clean exit 83 s, and the
  benchmarks 442 s (complete 19:51).
- The reads and their commands are arm P's ladder, unchanged: `closed_book_rank.py bios --form both
  --evidence none|gold|swapped --batch-size 512` on the pilot biographies;
  `evidence_ceiling_probe.py --fixed-split evidence_fixed --max-questions 2000`;
  `eval_abstention.py --evidence-port --evidence-condition gold,counterfactual,distractors,none
  --max-examples 2000 --batch-size 16`; `eval_store.py pathway` and `eval_store.py edit --route
  bge` on `data/index/openqa` (buffer 4, loop 1); `eval_exit.py` on `inject_val_clean` (40 batches
  of 4, no evidence); `eval_benchmarks.py` (the 13-task suite).
- The SFT that made the checkpoint (`ckpts/inject/config_pilot_bridge.yaml`): batch 4 x 2, lr and
  fresh lr 3e-4 with a 2% warm-up and the cosine fitted to the split, floor 0.1; arm P's own cosine
  had ended at about 6.5e-5, so the SFT re-warmed the trunk to about 5x its final rate, on chain
  questions alone (no biographies, no QA, no plain text, no abstention rows) for two passes.

## Every gate, arm P's final against the bridge final

Arm P's numbers are its final save's (closed book, gold and swapped 2026-10-11,
`pilot_p_final_rank_*`; ceiling and benchmarks 2026-10-14, `pilot_p_ladder_*`; abstention, pathway
and the bge edit on the fixed packer 2026-10-24, `pilot_p_ladder2_*`; the clean exit read
2026-10-25, `pilot_p_exit_inject_val_clean.*`, the same 40 batches).

| gate | read | arm P final | bridge final | what the gap means |
|---|---|---|---|---|
| A1(b), closed book | entity `delta` in distribution, tier 100 / 1000 | 0.0096 (z 2.2) / 0.0639 (z 7.2) | 0.0039 (z 0.9) / -0.0001 (z 0.0) | the weakly stored biographies, the tier 1000 residual with them, are gone |
| A1(b), closed book | entity `delta` held out, tier 100 / 1000 | 0.0125 (z 3.2) / 0.0458 (z 5.5) | 0.0003 (z 0.1) / 0.0063 (z 0.7) | the same on the held-out form; every entity cell at chance (largest other cell: dates in distribution, tier 1000 0.029, z 2.3) |
| copy, gold card | entity top-1 in distribution by tier 1 / 10 / 100 / 1000 | 1.000 at every tier and class | 0.755 / 0.757 / 0.755 / 0.797 (normalized rank 0.006 to 0.009) | the port still ranks the carded value near the top, but picks it a quarter less often |
| copy, gold card | entity top-1 held out | 0.942 to 0.967 | 0.403 / 0.408 / 0.416 / 0.467 (rank 0.042 to 0.048) | copying halved on the unseen template |
| copy, gold card | dates / nouns top-1, in distribution | 1.000 / 1.000 | 0.35 to 0.41 / 0.97 to 0.99 | the loss is concentrated on dates; majors still copy |
| copy, swapped card | followed in distribution / held out | 1.000 / 0.943 to 0.968 | 0.710 to 0.734 / 0.496 to 0.525 | the substitute is followed less, `mr_ll` 0.03 to 0.06 (arm P 0.000 / under 0.005) |
| A2 / A5 in likelihood form | in-context answer-span CE on `evidence_fixed`, gold / mixed / distractors / none | 3.49 / 4.10 / 6.20 / 6.05 | 12.33 / 12.41 / 14.14 / 15.41 | absolute levels dominated by the answer format: the model expects "Step 1:" before any answer |
| A2 / A5 in likelihood form | ceiling (none minus gold) / content (distractors minus gold) | 2.56 / 2.71 | 3.08 / 1.81 | the evidence still lowers the answer's CE by about as much; the distractors cost less than before, so the content signal is weaker |
| A3, generation | 2,000 SQuAD v2 questions under gold: EM / token F1 / false abstention | 0.046 / 0.063 / 0.770 | 0.000 / 0.008 / 0.000 | it never abstains and never answers in the plain form |
| A3, generation | non-abstain rate under distractors / none | 0.007 / 0.000 | 1.000 / 1.000 | abstention erased: every evidence-free question gets an invented answer |
| A3, likelihood | real answer's span CE, gold / distractors / none; gold minus none | 2.13 / 6.11 / 5.87; +3.75 | 11.69 / 14.44 / 15.20; +3.51 | the likelihood still prefers the answer with the evidence attached, at a level shifted by the format |
| A3, counterfactual | follow, every eligible item (n 218) / `mr_ll` | 0.087 / 0.114 | 0.000 / 0.328 | nothing followed (nothing is answered in the plain form); the likelihood leans further toward the original answer |
| A3 / A5 | external memory mass at the last prompt position, mean / AUROC against native unanswerable | 0.16 / 0.46 | 0.52 / 0.51 | the selector puts three times the mass on the store and still separates nothing |
| A7, pathway | EM, model query / bge / oracle / none (n 6,000) | 0.001 / 0.044 / 0.088 / 0.000 | 0.000 / 0.000 / 0.000 / 0.000 (F1 at most 0.013) | the oracle buffer no longer helps: no route answers |
| A4, edit on the bge route | items answered before the edit (of 2,000); delete / edit flip | 95; 0.90 +- 0.035 (n 72) / 0.13 +- 0.07 (n 23) | 0; nothing to flip | unreadable: nothing is answered before the edit |
| language | clean slice CE at exits 1 / 2 / 3 (`inject_val_clean`, 588,155 tokens) | 3.348 / 3.274 / 3.267 | 8.687 / 8.974 / 9.062 | plain language destroyed: 2.8x the CE at exit 3 |
| language | token top-1 at exits 1 / 2 / 3; depth gain exit 1 to 3 | 0.413 / 0.420 / 0.420; +0.081 | 0.059 / 0.058 / 0.058; -0.375 (z -64.4 clustered) | depth now hurts; the most confident decile at exit 1 (p_max 0.996) sits at CE 14.4 |
| language | the 13-task suite, mean multiple-choice headroom over chance | +0.0615 over 9 tasks | +0.0191 over 9 tasks (the control +0.0725) | most of the suite's small headroom gone; the cloze tasks hit hardest (table below) |
| reasoning (R9's form) | held-out 2-hop with the bridge written (`chains_heldout_tmpl`, depth 3) | at chance (latent, 0.193 to 0.195) | 0.997 (z 64.7, 2026-10-24) | R9's verdict stands: composition in written tokens |
| L2 / 4-hop | held-out 4-hop by read sites, depths 3 and 4 | 0.18 to 0.19 against 0.210 | 0.206 against 0.210 (generate mode, two steps written) | both at chance; the bridge does not extend past the trained depth |
| L2 / HotpotQA bridge | by read sites | no instrument | no instrument | a build item |

The 13-task suite (`eval_benchmarks.py`, the same tasks and n as arm P's ladder; score, then the
headroom over chance):

| task | metric | chance | arm P final | control final | bridge final |
|---|---|---|---|---|---|
| hellaswag | acc_norm | 0.25 | 0.2650 | 0.2630 | 0.2651 |
| arc_easy | acc_norm | 0.25 | 0.3695 | 0.3590 | 0.2967 |
| arc_challenge | acc_norm | 0.25 | 0.2133 | 0.2159 | 0.2449 |
| piqa | acc_norm | 0.50 | 0.5745 | 0.5653 | 0.5272 |
| winogrande | acc | 0.50 | 0.5051 | 0.5075 | 0.4886 |
| openbookqa | acc_norm | 0.25 | 0.2400 | 0.2640 | 0.2700 |
| sciq | acc | 0.25 | 0.5990 | 0.5660 | 0.4510 |
| boolq | acc | 0.50 | 0.4098 | 0.4881 | 0.3810 |
| lambada_openai | acc | 0.00 | 0.1151 | 0.1100 | 0.0019 |
| mmlu | acc | 0.25 | 0.2441 | 0.2301 | 0.2562 |
| triviaqa | em | 0.00 | 0.0000 | 0.0000 | 0.0000 |
| nq_open | em | 0.00 | 0.0000 | 0.0000 | 0.0000 |
| gsm8k | em | 0.00 | 0.0140 | 0.0170 | 0.0010 |
| mean multiple-choice headroom (9 tasks) | | | +0.0615 | +0.0725 | +0.0191 |

The table lists scores; the three mean rows are normalized headroom, not score minus chance. Restatement of the BoolQ chance (2026-10-25): BoolQ chance was set at 0.50 while the majority class is about 0.62 (arm P 0.4098, control 0.4881, bridge 0.3810 against all-no 0.378), so on the 8-task mean arm P reads +0.0917, the control +0.0845 and the bridge +0.0513 (9-task +0.0615 / +0.0725 / +0.0191); A6's "within noise of the control" holds on both (z about 1.2 and 0.76), the sign of the control's lead flips, and R5's suite gap is about half, not a third; the verdict stands on the other gates.

The cloze and continuation tasks fall furthest: lambada from 0.115 to 0.002, sciq 0.599 to 0.451,
arc_easy 0.370 to 0.297, piqa 0.575 to 0.527; the task perplexities explode (hellaswag about
15,700 against 49, boolq about 1.0M against 386). The cells that rose (arc_challenge, openbookqa,
mmlu) were at or under chance on arm P and stay within about 0.03 of chance, so they read as noise
around a model that no longer scores text, not as a gain. The generation tasks stay at zero and
gsm8k falls from 0.014 to 0.001; the first generations are degenerate repeats ("University of of
of of ...", "B: B: B: B:", "Themo.").

## What the model writes

Under gold, every one of the 2,000 SQuAD completions starts with the bridge form: 1,181 with
"Step 1:" and 819 with "Answer:" (under none 1,223 and 777). The first completions under gold:

- `Step 1: 1: 1: 1: 1: 1: 1: 1: 1: 1:`
- `Answer: The Basisines.`
- `Step 1: Sustiigraphers Dive Biomcigginghers. Answer: Geesthraphers Dive Biomgomebraginghers.`
- `Step 1: Sirthusardin ofthe The Crest ofardin Loth Loth Loth ...`
- `Answer: Demntheififififififif...`

Under distractors: `Step 1: Rundotho. Answer: Fargio.`, `Step 1: Feldrath. Answer:besstem.`;
under none: `Answer: University of Soulndndndnd...`. The text is the bridge template filled with
degenerate repeats or invented entity-like strings in the style of the fictional chain names, never
an abstention phrase and never a span of the attached passage. Arm P under gold abstained on 77% of
answerable rows and answered 5% exactly; the bridge model abstains on none and answers none.

## Verdict (the main thread's)

**The chain-only bridge SFT at the chain continuation's recipe (66M tokens, the LR and schedule of
`config_pilot_bridge.yaml`) is a demonstration that composition happens in written tokens, not a
usable checkpoint.** R9's verdict stands (held-out 2-hop 0.997 against chance 0.191). But the same
SFT erased the weakly stored biographies (the tier 1000 residual with them), halved the port's
copying on the held-out form (gold top-1 0.95 to about 0.41, swapped followed 0.94 to 0.97 to about
0.51), replaced answering and abstention on real questions with the bridge form or degenerate
text (EM 0.000, abstention 0.000 under every condition, no route answering on the pathway read),
and multiplied the clean-slice CE by about 2.8 (3.267 to 9.062 at exit 3), with depth now costing
0.375 nats instead of paying 0.08.

Consequences for the spec (written into [RUN2.md](../plans/RUN2.md) and NEXT.md 6f):

- **6f's mix carries the pretraining slices beside the bridge rows**: evidence QA, the abstention
  rows, the biographies at their tier structure and the plain-answer form, at a lower LR than the
  pilot's continuation (3e-4 against arm P's final 6.5e-5), with the A gates and the clean-slice CE
  read at every save of 6f and the kill rule on them.
- **Every A gate in the spec is taken on the shipped checkpoint**, after 6e and 6f, not only at the
  end of pretraining.
- **The pilot's tier 1000 residual is a property of the pretrained trunk that a short
  post-training removes**, so the leak reads after post-training are the binding ones. Read here
  as forgetting under a single-format SFT, not as a lever: the same SFT removed the copying and the
  language it would have to keep.
- **"HotpotQA bridge by read sites" needs an instrument** (the bridge decode over retrieved real
  chunks) in the build list.

## Open

- What dose of which slices keeps the gates under bridge SFT (the 6f mix) is unmeasured; the
  rule above is a requirement, not a read.
- Whether the copying and plain-language loss recovers with a short plain-text continuation (a
  forgetting check) is unread.
- The bridge model's 50M and 60M saves and earlier saves were not read on the ladder; the
  degradation curve over the SFT is unknown (the trainer's held-out bridge CE was learned inside
  10M).
- One seed, one dose, one LR.

## Raw files

- Script and log: `ckpts/inject/r5_bridge_ladder.sh`, `ckpts/inject/r5_bridge_ladder.log`.
- Reads: `ckpts/inject/pilot_bridge_ladder_{rank_none,rank_gold,rank_swapped,ceiling_fixed,abstention,store_pathway,store_edit_bge,exit_clean,bench}.{json,log}`.
- Arm P comparators: `ckpts/inject/pilot_p_final_rank_{none,gold,swapped}.*`,
  `pilot_p_ladder_{ceiling_fixed,bench}.*`, `pilot_p_ladder2_{abstention,store_pathway,store_edit_bge}.*`,
  `pilot_p_exit_inject_val_clean.*`, `pilot_p_final_chains_hop4.*`; the control's suite
  `pilot_c_ladder_bench.*`.
- The checkpoint: `ckpts/evidence_pilot_bridge/checkpoint_evidence_final.pt`; its 4-hop read
  `ckpts/inject/pilot_bridge_final_gen_hop4.*`.

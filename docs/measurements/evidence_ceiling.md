# What evidence is worth before the port exists

Gate G3 asks for a gold-vs-no-evidence CE gap of ~0.3 nats on the answer span, read through a port
that costs 300–500M tokens to train. That gap has a ceiling, and the ceiling is measurable today for
free: hand the model the same gold passage through **in-context attention** — the pathway it already
has and was finetuned on — and read the same quantity. The port is trying to reach what in-context
reading already delivers. If in-context evidence were not worth 0.3 nats on this checkpoint, the
gate would be measuring the wrong thing rather than the model failing, and the spend would be
unjustified before it started.

Every arm before this one failed. This is the first number in the sequence that came back clearly
positive, and the reason it did is structural rather than lucky.

## 1. The reading

[`scripts/evidence_ceiling_probe.py`](../../scripts/evidence_ceiling_probe.py), on the answerable half of the standard SQuAD v2 slice, 1,500
questions, `checkpoint_repair_final.pt`. Three conditions over the same rows in the same order; CE
restricted to the answer span, teacher-forced.

| condition | CE | ppl | top-1 | mean `p_max` |
|---|---|---|---|---|
| gold passage | **1.6401** | 5.16 | 0.7084 | 0.7465 |
| no passage | 4.8707 | 130.41 | 0.2997 | 0.4722 |
| another row's passage | 5.4987 | 244.37 | 0.2834 | 0.4896 |

- **gold vs. no evidence: +3.2306 nats.** G3 wants 0.3 through the port. The ceiling is ten times
  the bar.
- **gold vs. distractor: +3.8586 nats.** This is the selection signal, and it is larger than the
  reading signal.

The row set is fixed by the gold condition's length check and reused verbatim across all three
conditions. Letting each condition drop its own over-long rows would score the three arms on
different questions, which is the one way this comparison can go quietly wrong. CEs come from
`eval_abstention.teacher_forced_calibration` by import, so they are the same quantity that script
reports rather than a second implementation of it.

## 2. Why this arm is shaped differently from the ones that failed

The IR table arms (`ir_reshape.md`, `ir_sharpening.md`, `ir_scale_fix.md`) all converged on one
result: zeroing the read costs ~0.0002 nats across three key inits and two table widths, trained or
not. The table sharpened; the model still did not use what it retrieved.

That was never really a bug in the mechanism. A learned table trained on the same corpus as the
trunk can only offer content the trunk already holds, so a working retrieval of it buys nothing and
the measured zero is the correct answer. Evidence is the opposite case: the gold passage carries
content this model demonstrably lacks, and the size of that gap is exactly the 3.2 nats above.

So the previous arms' failure and this arm's headroom are the same fact seen twice. **Phase 4 has a
ceiling to reach that arms A–D never had.**

## 3. The unplanned finding: a distractor is worse than nothing

A passage from another question costs **0.628 nats more than supplying no passage at all** (5.4987
against 4.8707), and drops top-1 below the no-evidence condition too. Note that `p_max` goes *up*
slightly in the distractor condition — the model is not merely uninformed by the irrelevant passage,
it is somewhat more confident while being more wrong.

The model reads whatever it is handed, uncritically. Two consequences, both of which change what
Phase 4 is for rather than merely adding detail to it:

- **The gold-among-distractors and distractors-only conditions are load-bearing, not thoroughness.**
  Selection is a defect the training has to fix, not a refinement layered on top of reading. A port
  trained only on gold evidence would make the model strictly worse the moment a real retriever
  handed it something irrelevant, which is what a real retriever mostly does.
- **It sharpens G3b.** The groundedness signal has to separate "retrieved something relevant" from
  "retrieved something". Those are currently indistinguishable to this model — and the confidence
  scalar moves the wrong way across them, so nothing existing supplies it.

## 4. Scope

This is the in-context ceiling, measured through attention over a passage in the prompt. It is not a
measurement of the port, which did not exist when it was taken. It bounds what the port can be worth
and says the bound is generous; it says nothing about how much of that bound the port will reach.

The probe is read-only and reuses `eval_abstention`'s loader, renderer, slice and scorer by import.
Raw output in `ckpts/repair/evidence_ceiling.log` (gitignored, as eval logs are).

## 5. The ceiling on `evidence_fixed` (2026-10-02)

The same probe on the held-out kill split, so every later arm's port gain has an in-context
reference on the same questions: `evidence_ceiling_probe.py --fixed-split evidence_fixed
--max-questions 2000`, seed `checkpoint_repair_final_irrandom_evidence_grounded.pt`, answer-span CE
with the real answer as the target under every condition, split by the `.src` sidecar.

| source | gold | mixed | distractors | none | ceiling | content | questions |
|---|---|---|---|---|---|---|---|
| hotpot_qa | 1.5252 | 1.7512 | 4.7686 | 4.5857 | 3.0604 | 3.2434 | 1,118 |
| squad_v2 | 1.6061 | 1.8652 | 5.4485 | 4.9131 | 3.3070 | 3.8424 | 882 |
| all | 1.5639 | 1.8058 | 5.0942 | 4.7425 | **3.1785** | **3.5303** | 2,000 |

ceiling = CE(none) - CE(gold); content = CE(distractors) - CE(gold).

- The pooled ceiling, 3.18 nats, agrees with the 3.23 in section 1 on a different slice and a
  different checkpoint. The number is a property of the data, not of the slice.
- In context, selection is nearly solved: gold among distractors (`mixed`) costs only 0.24 nats over
  gold alone. A distractor alone costs 0.35 nats over nothing, about half the 0.63 in section 3.
- Against this reference the killed 10M arms reached roughly a sixth of the ceiling on gold gain
  (arm A 0.545, arm B 0.571) and about 2% of it on content gain (0.058 and 0.081). The arms' fixed
  pass scored its own batch cap, not these 2,000 rows, so the ratios are approximate. Every later
  run reads its gain against this file through `sft.py --ceiling-json`, on the same rows.

The JSON is `docs/measurements/ceiling_fixed.json`. It is gitignored like every `*.json`, so this
table is the tracked record; regenerate the file with the command above before any arm needs it.

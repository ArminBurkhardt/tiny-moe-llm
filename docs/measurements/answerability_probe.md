# The answerability probe: what the trunk knows about unanswerable questions (2026-08-26)

Two checkpoints of abstention work established what the *policy* cannot do: tell an answerable
question from an unanswerable one. The corpus ratio slides the operating point along a fixed curve
(precision read at ~0.578 six times while everything else moved), and `p_max` carries no usable
discrimination — AUROC 0.462–0.478 across three checkpoints, with a sign that is not even stable
across a 2,000-question slice. What that record never said is **why**, and the two possible reasons
put the burden in completely different places:

- **the trunk does not represent answerability** → only new information can create the signal, and
  no readout trick, threshold or loss reweighting will. Retrieval-grounded evidence (Phase 4) is
  then the only principled candidate and carries the whole gate.
- **the trunk represents it but the policy cannot read it out** → a thresholded probe is a
  shipping mechanism on its own (tunable operating point, minutes to refit), and Phase 6's
  preference pass has something real to amplify rather than something to invent.

A linear probe separates them, and a linear probe is the right instrument precisely because it is a
**lower bound**: it finds only what is linearly decodable, so a pass is strong evidence the
information is there and a failure is weak evidence that it isn't.

## What was run

`scripts/eval_probe.py`, on three checkpoints, at identical flags:

```bash
python scripts/eval_probe.py -c ckpts/trained/checkpoint_phase2_final_phase0.pt \
  --json-out docs/measurements/probe_phase2_final.json
python scripts/eval_probe.py -c ckpts/trained/checkpoint_sft_final_phase0.pt \
  --json-out docs/measurements/probe_sft_final.json
python scripts/eval_probe.py -c ckpts/repair/checkpoint_repair_final.pt \
  --json-out docs/measurements/probe_repair_055.json
```

Defaults throughout: 10,000 `squad_v2/train` questions to fit (2,000 of them held out to choose the
L2 penalty), the standard 2,000-question `squad_v2/validation` slice to read, `--batch-size 16`,
`--max-prompt-tokens 1024`, seed 1234. The rendering path is `eval_abstention.build_records` by
import, so the prompts are byte-identical to the ones every abstention number in this project is
measured on and the eval slice is *the same slice* — same seeded shuffle, same 1,013/987 split.

Features are read at the **last prompt position**: the state the model would decode its first answer
token from, not an average over the passage. Three sets, fitted separately so the answer is
attributable:

| set | features |
|---|---|
| `scalars` | `p_max`, predictive entropy, top-1/top-2 log-probability margin |
| `hidden` | the final loop's post-norm hidden state, 768-d |
| `hidden+scalars` | both — the plan's stated feature set |

Positive class is **unanswerable**, matching the direction `eval_abstention.py` reports
`AUROC(1 - p_max → unanswerable)` in. The L2 strength is chosen on the held-out train questions and
never on the eval slice; the chosen model is then refit on all 10,000.

## The result

AUROC on the standard slice, all three checkpoints:

| | | pretrained trunk<br>`phase2_final_phase0` | SFT<br>`sft_final_phase0` | repair @ 0.55 |
|---|---|---|---|---|
| **`hidden`** | fit / dev / **slice** | 0.733 / 0.651 / **0.5804** | 0.727 / 0.649 / **0.5850** | 0.726 / 0.630 / **0.5839** |
| **`hidden+scalars`** | slice | **0.5807** | **0.5846** | **0.5837** |
| `scalars` | slice | 0.4761 | 0.5398 | 0.5219 |
| unfitted `1 - p_max` | slice | 0.5259 | 0.4793 | 0.5016 |
| unfitted entropy | slice | 0.5480 | 0.4541 | 0.5013 |
| unfitted `-margin` | slice | 0.4868 | 0.5342 | 0.5238 |

Three readings, in order of how much they change the plan.

**1. The signal exists, it is weak, and it is a property of the pretrained trunk.** 0.5804 / 0.5850
/ 0.5839 — the three checkpoints agree to within **0.005**, an order of magnitude inside the ±0.030
slice noise on an AUROC ([noise_floor.md](noise_floor.md)). A 709M-token SFT and a 49M-token
abstention repair, between them, moved this number by nothing. Whatever answerability information a
linear read can extract was already there when pretraining ended, and neither finetune added to it
or destroyed it.

That is the same finding as "the corpus ratio slides the operating point along a fixed curve and
does not bend it", now visible one level down: the finetunes were **re-reading a fixed
representation**, not changing it. The two observations are consistent by construction, and together
they say the data lever is exhausted for the same reason in both cases.

**2. `p_max` and its friends carry nothing, confirmed a fourth time.** Every unfitted scalar lands
in 0.454–0.548 — straddling chance, with no consistent sign across checkpoints. The fitted `scalars`
probe does no better (0.476–0.540): three free numbers, three orders of magnitude of regularization
searched, and it still cannot separate the classes. This is now measured on four checkpoints by two
different methods and should not be re-litigated.

**3. Neither of the plan's branches fires cleanly, and 0.58 is the honest answer.** The plan wrote
≈0.5 → "not in the representation" and ≥0.7 → "a readout problem". The measurement landed between
them: the hidden state beats the *best* free scalar on its own checkpoint by 0.032 (pretrained),
0.051 (SFT) and 0.060 (repair) — every one of them outside the ±0.030 noise, and the strongest
comparison is on the shipped checkpoint. The signal is real and the policy is not using it. But 0.58
is nowhere near a discriminator anything could ship on alone.

The consistent **fit 0.73 → dev 0.65 → slice 0.58** ladder on all three checkpoints is worth naming
because it bounds how much of the gap is fixable. Fit-to-dev is ordinary probe overfitting on 768
features. Dev-to-slice is a train→validation shift on top of it, and SQuAD v2's validation
unanswerable questions were written adversarially against a different annotation round, so a probe
fitted on train questions transfers into a harder distribution. More train questions or a stronger
probe would recover part of the first gap and none of the second; neither turns 0.58 into 0.65.

## What a threshold on it would actually buy

The repair checkpoint's shipped policy, on this exact slice, abstains at precision **0.5782**,
recall **0.2152**, false-abstention **0.1611**. The `hidden` probe's threshold sweep on the same
questions:

| flag rate | precision | recall | false abstention |
|---|---|---|---|
| 0.50 | 0.5570 | 0.5499 | 0.4488 |
| 0.30 | 0.5967 | 0.3534 | 0.2452 |
| 0.20 | 0.6350 | 0.2507 | 0.1479 |
| **0.15** | **0.7067** | **0.2093** | **0.0892** |
| 0.10 | 0.7500 | 0.1481 | 0.0507 |
| 0.05 | 0.7700 | 0.0760 | 0.0233 |

**At matched recall (0.209 vs the policy's 0.215) the probe is 0.13 more precise and abstains
falsely at half the rate** — 5x the ±0.026 binomial floor on precision and well past the ±0.012 on
false abstention. So the trunk's weak signal is not merely above chance, it is *better ordered than
the shipped policy's own decisions* at the operating point that policy sits on.

Three things keep that from being a result to ship on:

- **It is not the same decision.** The policy emits an answer; the probe only decides whether to
  abstain. A deployed combination — abstain when the probe fires, otherwise emit the model's answer
  with its own abstentions suppressed — is a different mechanism whose answerable-half EM is not
  measured here. That combination is a Phase 6 question, not a Phase 1b finding.
- **The curve is still not bent where it matters.** G7 wants precision ≥ 0.65 *at recall ≥ 0.5*.
  The probe's precision at recall 0.55 is 0.557, barely over the slice's 0.507 base rate — a model
  that flagged half the questions at random would score close to it. Everything it buys, it buys by
  abstaining less, which is sliding along a curve again — a better curve than the policy's, but the
  same shape.
- **It is fitted per task.** 10,000 in-domain questions with labels produced this; nothing says it
  transfers off SQuAD, and the abstention claim is not a SQuAD claim.

## Caveats that belong next to the number

- **The probe is fitted on data the checkpoints trained on.** `prepare_sft_data.py` builds the SFT
  and repair corpora from `squad_v2/train`, so these questions are not novel to the model. That
  cannot leak into the reported AUROC — it is read on held-out validation questions — but it does
  mean the fit set is a distribution the model has seen, which flatters the fit half of the ladder
  and makes the low slice number the more robust of the two.
- **`1 - p_max` reads 0.5016 here against 0.4775 in [noise_floor.md](noise_floor.md), and they are
  different quantities.** This script's `p_max` is the readout at the last *prompt* position; the
  abstention eval's is `p_max` averaged over the tokens the model actually *generated*. Both are
  within their own noise of chance, which is the only thing either was ever used to say.
- **A linear probe is a lower bound**, as intended. A nonlinear probe scoring higher would be
  interesting but would not change the plan: it would still be a readout of a representation that no
  finetune moved, and Phase 4's job is to add information rather than to decode harder.
- **Batch size is part of the measurement.** `ParallelSparseMoELayer` tiles its grouped GEMM by
  per-expert row counts computed over every token in the batch, padding included, so features
  measured at a different `--batch-size` differ at the ~0.5–1% level.

## What this decides

**Phase 4 keeps the burden.** The discrimination has to come from information the trunk does not
currently have, because three checkpoints' worth of finetuning demonstrably did not put any there.
Gate G3b stands as written.

**G3b's reference number changes, and it should.** "Beat `p_max`'s 0.462–0.478" was never a
demanding bar — `p_max` is at chance. The honest bar is that the external-vs-parametric mass
signal must beat **0.584**, what a free linear read of the *existing* trunk already gives, before
"the evidence pathway supplies discrimination" means anything. G3b's stated 0.65 threshold clears
that by 0.066, so the gate's number needs no change — only its comparison, which is now a real
baseline rather than a chance-level one.

**Phase 6 has something small and real to amplify.** The preference pass is no longer being asked to
invent a distinction the model has no representation of; it is being asked to make the policy use
one that measurably exists. That is a materially easier thing to ask, and it is why the probe was
worth running before either phase started.

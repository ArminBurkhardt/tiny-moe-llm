# The benchmark suite: harness validation and the frozen peer numbers (2026-08-23)

CE on a held-out slice of the training corpus is a health check, not a quality claim. It cannot see
a capability regression that leaves average CE flat, and it says nothing about where a 332M model
trained on 16B tokens actually sits in its size class. `scripts/eval_benchmarks.py` is the
instrument that can, and this is the record of building it and proving it works.

## The suite

Thirteen tasks, one scoring path for this model and for every peer.

| task | split | n | metric | chance |
|---|---|---|---|---|
| HellaSwag | validation | 10,042 | acc_norm | 0.25 |
| ARC-Easy | test | 2,376 | acc_norm | 0.25 |
| ARC-Challenge | test | 1,172 | acc_norm | 0.25 |
| PIQA | validation | 1,838 | acc_norm | 0.50 |
| WinoGrande (xl) | validation | 1,267 | acc | 0.50 |
| OpenBookQA (main) | test | 500 | acc_norm | 0.25 |
| SciQ | test | 1,000 | acc | 0.25 |
| BoolQ | validation | 3,270 | acc | 0.50 |
| LAMBADA (OpenAI) | test | 5,153 | acc (+ ppl) | 0 |
| MMLU (all, 0-shot) | test | 14,042 | acc | 0.25 |
| TriviaQA (rc.nocontext) | validation | 1,000 of 17,944 | EM | 0 |
| NQ-open | validation | 1,000 of 3,610 | EM | 0 |
| GSM8K (main, 5-shot) | test | 1,000 of 1,319 | EM | 0 |

Multiple choice is scored by conditional log-likelihood of each option given the prompt, reduced
two ways: `acc` on the summed log-probability and `acc_norm` on the same divided by the option's
length **in bytes**. Each task declares which the published convention makes its headline. The
generative three are greedy, cut at the first stop string, scored by exact match after SQuAD-style
normalization (imported from `eval_abstention.py`, so both evals normalize identically).

**Frozen flags:** `--batch-size 32 --max-context 1024 --gen-limit 1000 --seed 1234`, peers in fp32
with TF32 matmuls, this model in bf16. Multiple choice runs the full split; the three generative
tasks are capped at 1,000 questions by a seeded shuffle-then-truncate, so every model sees the same
subsample. Batch size is part of the measurement — the MoE's grouped GEMM tiles by the batch's
per-expert row counts.

Prompt formats are the EleutherAI harness's, task by task, which is what makes the validation below
a real check rather than a self-consistency test.

## Gate G0, harness half: PASS

Pythia-410m scored against its own published zero-shot numbers, from
[the Pythia repo's `410m_step143000.json`](https://raw.githubusercontent.com/EleutherAI/pythia/main/evals/pythia-v1/pythia-410m/zero-shot/410m_step143000.json).
Those were produced by the EleutherAI harness on these splits with these prompt formats, so they
test this file rather than testing Pythia.

| task | metric | published | measured | delta (points) |
|---|---|---|---|---|
| arc_easy | acc | 0.5210 | 0.5189 | −0.21 |
| arc_easy | acc_norm | 0.4579 | 0.4575 | −0.04 |
| arc_challenge | acc | 0.2133 | 0.2142 | +0.09 |
| arc_challenge | acc_norm | 0.2432 | 0.2432 | −0.00 |
| piqa | acc | 0.6676 | 0.6670 | −0.06 |
| piqa | acc_norm | 0.6714 | 0.6719 | +0.05 |
| winogrande | acc | 0.5367 | 0.5328 | −0.39 |
| sciq | acc | 0.8110 | 0.8120 | +0.10 |
| sciq | acc_norm | 0.7210 | 0.7220 | +0.10 |
| lambada_openai | acc | 0.5162 | 0.5164 | +0.02 |
| lambada_openai | ppl (per document) | 10.828 | 10.780 | −0.05 |

**Eleven of eleven inside 0.4 points**, against a 1.5-point tolerance. Run it again with
`--peer pythia-410m --validate`.

The published file carries no HellaSwag, OpenBookQA or BoolQ entry, so those three tasks have no
anchor. Two of them corroborate anyway: gpt2-medium comes out at HellaSwag acc 0.3327 / acc_norm
0.3926 and BoolQ acc 0.5875, which is where that model is generally reported. OpenBookQA is the one
task with no independent confirmation at all, and it is also the smallest split (500 questions,
±1.8 points of binomial error) — treat a movement there with more suspicion than the others.

### Two conventions the first attempt got wrong

Both were caught by the validation, which is the argument for having one:

- **Byte-length normalization excludes the space that joins the option to the prompt.** Scoring
  `" Paris"` but normalizing by `len(" Paris")` rather than `len("Paris")` moved SciQ's `acc_norm`
  by **2.0 points** and ARC-Easy's by 0.7. One byte matters most where the answers are shortest,
  which is exactly where the tasks with short spans live.
- **LAMBADA's published perplexity is per document, not per token.** The target is a whole word,
  which the GPT-NeoX tokenizer splits into ~1.37 tokens on average, so the per-token figure reads
  5.68 against a published 10.83 while accuracy agrees to 0.02 points. Both are now reported:
  `ppl` per token, `ppl_doc` per document.

Neither would have been visible without a published number to check against.

## The frozen peer numbers

Peers are measured **once** and reused via `--compare`; each run is written to
`docs/measurements/benchmarks/peer.<name>.json` (untracked — `.gitignore` swallows `*.json` — so
regenerate with `--peer all --json-out docs/measurements/benchmarks/peer.json`).

| task | metric | chance | gpt2-medium | pythia-410m | smollm2-360m | qwen2.5-0.5b |
|---|---|---|---|---|---|---|
| hellaswag | acc_norm | 0.25 | 0.3926 | 0.4058 | **0.5617** | 0.5221 |
| arc_easy | acc_norm | 0.25 | 0.4360 | 0.4575 | **0.6806** | 0.5863 |
| arc_challenge | acc_norm | 0.25 | 0.2500 | 0.2432 | **0.3831** | 0.3242 |
| piqa | acc_norm | 0.50 | 0.6638 | 0.6719 | **0.7193** | 0.6997 |
| winogrande | acc | 0.50 | 0.5320 | 0.5328 | **0.5896** | 0.5635 |
| openbookqa | acc_norm | 0.25 | 0.3020 | 0.2940 | **0.3820** | 0.3520 |
| sciq | acc | 0.25 | 0.7690 | 0.8120 | 0.9120 | **0.9310** |
| boolq | acc | 0.50 | 0.5875 | 0.6012 | **0.6205** | 0.6177 |
| lambada_openai | acc | 0 | 0.4304 | 0.5164 | **0.5376** | 0.5255 |
| mmlu | acc | 0.25 | 0.2293 | 0.2322 | 0.2631 | **0.4771** |
| triviaqa | EM | 0 | 0.0300 | 0.0210 | **0.2190** | 0.0660 |
| nq_open | EM | 0 | 0.0070 | 0.0000 | **0.0520** | 0.0260 |
| gsm8k | EM | 0 | 0.0180 | 0.0180 | 0.0430 | **0.3710** |
| **mean MC headroom over chance** | | | **+0.193** | **+0.208** | **+0.345** | **+0.335** |

| model | source | params | pretraining tokens |
|---|---|---|---|
| gpt2-medium | `openai-community/gpt2-medium` | 355M | ~10B (WebText, not officially published) |
| pythia-410m | `EleutherAI/pythia-410m` | 405M | 300B |
| smollm2-360m | `HuggingFaceTB/SmolLM2-360M` | 362M | 4T |
| qwen2.5-0.5b | `Qwen/Qwen2.5-0.5B` | 494M | 18T |

"headroom over chance" is `(score − chance) / (1 − chance)`, the share of the *available* range
above guessing that the model covered. At this size several tasks sit within noise of chance, where
a raw accuracy column reads like a real number and a headroom column reads like the 0 it is —
ARC-Challenge at 0.25 and MMLU at 0.23 are two models scoring nothing, not two models scoring 24%.

Three readings worth keeping:

- **The token axis is visible and it is not a straight line.** gpt2-medium at ~10B and pythia-410m
  at 300B are 1.5 headroom points apart across nine tasks — a 30x data increase buying almost
  nothing on this suite — while SmolLM2-360M at 4T is 14 points clear of both at *fewer* parameters
  than Pythia. That gap is data curation, not scale: FineWeb-Edu/DCLM against the Pile.
- **MMLU separates knowledge from everything else.** Three of the four peers are at chance
  (0.229–0.263). Qwen2.5-0.5B at 0.477 is the only model in the class that has the knowledge, and
  it took 18T tokens. This is the "did knowledge arrive" axis for the real run, and it is honest
  about how expensive arriving is.
- **SmolLM2-360M's TriviaQA 0.219 is the number the evidence thesis is aimed at.** Closed-book
  recall is what 4T tokens buys and what this project intends to supply at eval time instead. It is
  the reference the corpus-attached delta gets measured against.

Two cross-checks on the peer numbers themselves: Qwen2.5-0.5B's MMLU 0.477 matches Qwen's published
47.5 for the base model, and SmolLM2-360M's ARC average of 0.532 matches the 53.0 on its card. Its
card's HellaSwag 54.5 against our 56.2 and its MMLU 35.8 against our 26.3 are format differences,
not disagreements — the card uses lighteval with a cloze MMLU, this suite uses the harness's
lettered-option MMLU.

## Shakedown of the local model path

One checkpoint through all thirteen tasks, to prove the local scoring and generation paths work end
to end rather than only on the four tasks the smoke tests covered. `checkpoint_repair_final.pt` (the
0.55 repair finetune), raw completion format, same flags. **This is not the three-checkpoint
snapshot** — that is still to be recorded — but the numbers are real and reported here so they are
not measured twice.

| task | metric | this model | headroom | gpt2-medium | pythia-410m | smollm2-360m | qwen2.5-0.5b |
|---|---|---|---|---|---|---|---|
| hellaswag | acc_norm | 0.2729 | +0.031 | 0.3926 | 0.4058 | 0.5617 | 0.5221 |
| arc_easy | acc_norm | 0.3830 | +0.177 | 0.4360 | 0.4575 | 0.6806 | 0.5863 |
| arc_challenge | acc_norm | 0.2321 | −0.024 | 0.2500 | 0.2432 | 0.3831 | 0.3242 |
| piqa | acc_norm | 0.5718 | +0.144 | 0.6638 | 0.6719 | 0.7193 | 0.6997 |
| winogrande | acc | 0.5217 | +0.043 | 0.5320 | 0.5328 | 0.5896 | 0.5635 |
| openbookqa | acc_norm | 0.2740 | +0.032 | 0.3020 | 0.2940 | 0.3820 | 0.3520 |
| sciq | acc | 0.6560 | +0.541 | 0.7690 | 0.8120 | 0.9120 | 0.9310 |
| boolq | acc | 0.4190 | −0.162 | 0.5875 | 0.6012 | 0.6205 | 0.6177 |
| lambada_openai | acc | 0.1989 | — | 0.4304 | 0.5164 | 0.5376 | 0.5255 |
| mmlu | acc | 0.2285 | −0.029 | 0.2293 | 0.2322 | 0.2631 | 0.4771 |
| triviaqa | EM | 0.0140 | — | 0.0300 | 0.0210 | 0.2190 | 0.0660 |
| nq_open | EM | 0.0020 | — | 0.0070 | 0.0000 | 0.0520 | 0.0260 |
| gsm8k | EM | 0.0090 | — | 0.0180 | 0.0180 | 0.0430 | 0.3710 |
| **mean MC headroom** | | **+0.084** | | +0.193 | +0.208 | +0.345 | +0.335 |

The harness works on this model: every task ran, `n` matches the peers' exactly, and the shape of
the result is what a 16B-token model should look like — real signal on SciQ (+0.54 headroom),
ARC-Easy and PIQA, at chance on ARC-Challenge and MMLU. It sits below gpt2-medium overall, which is
the honest starting position and the reason the plan's second claim is "on-trend for its token
budget" rather than "competitive".

**BoolQ at 0.4190 is below chance and is the one number here that is a finding rather than a floor.**
A two-option task can only land there by systematically preferring one option, and `acc_norm` on the
same scores reads 0.5920 — so the model prefers `" no"` on log-probability while byte normalization
(`" no"` is shorter than `" yes"`) flips most of it back. Worth revisiting once the trunk moves; it
is the kind of yes/no answer-policy bias that average CE cannot see, which is the entire argument
for having this suite.

Cost on the local model: **36 minutes**, of which **GSM8K alone was 24**. Generation here re-runs the
full prefix at every decode step (`kv_cache.py` is single-sequence and the batched path does not use
it), so a 5-shot prompt decoded 256 tokens deep is quadratic in exactly the wrong place. The MTP head
also runs on every one of those forward passes and its output is discarded. Both are known and
scheduled; until then, use `--gen-limit` for anything iterative.

## Isolation from the corpora

Benchmark shards go to `data/benchmarks/` and peer weights to `ckpts/peers/`. The script **refuses**
a cache directory that resolves inside `data/prepared*`, `data/datasets` or `data/archives`: both
corpus builders delete each source shard the moment they have appended it, and `archive_corpus.py`
packs whole split directories, so a benchmark file landing in either place is at best noise inside
an archive of something else and at worst a deleted download. `eval_abstention.py`'s SQuAD v2
validation scratch directory moved out of `data/prepared/` for the same reason.

Neither directory is tracked (`data/benchmarks` was added to `.gitignore`; `ckpts/` already was).

## Cost

Six to nine minutes per peer for the whole suite on the 5090 — 148k scored continuations plus 3,000
generations — at 5.7–10.7 GB peak.

The first implementation took ~25 minutes for HellaSwag alone at 16% GPU utilization. Two changes
fixed it, both worth remembering for anything else that scores this many short sequences:

- **Batch the tokenizer.** One `encode` call per string is a Python-side round trip per string; the
  fast tokenizers thread a whole list in Rust. HellaSwag repeats each context across four options,
  so deduplicating contexts before the batched call removes three quarters of that work outright.
- **Assemble each batch in numpy and cross to the device once.** Filling a padded batch row by row
  with `torch.tensor(...).to(device)` is a separate small transfer per row, and at these sequence
  lengths those dominate the forward pass they feed.

Together: HellaSwag from ~25 minutes to 61 seconds.

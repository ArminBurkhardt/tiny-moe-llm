# The three-checkpoint baseline snapshot (2026-08-26)

The instrument was built and validated in [benchmark_suite.md](benchmark_suite.md); this is the
reading every later phase diffs against. Three checkpoints, thirteen tasks, one scoring path, the
frozen flags: `--batch-size 32 --max-context 1024 --gen-limit 1000 --seed 1234`, bf16, raw
completion format.

| column | checkpoint | what it is |
|---|---|---|
| **pretrained** | `checkpoint_phase2_final_phase0.pt` | the 16B-token trunk, Phase-0 migrated |
| **SFT** | `checkpoint_sft_final_phase0.pt` | + 709M tokens of SFT, Phase-0 migrated |
| **repair @ 0.55** | `ckpts/repair/checkpoint_repair_final.pt` | + the 49M-token abstention repair |

```bash
python scripts/eval_benchmarks.py -c <ckpt> \
  --json-out docs/measurements/benchmarks/<name>.json \
  --compare docs/measurements/benchmarks/peer.*.json
```

## The snapshot

Headline metric per task, peers as measured once and frozen. `headroom` is the repair column's
share of the range above chance, `(score − chance) / (1 − chance)`.

| task | metric | chance | **pretrained** | **SFT** | **repair** | headroom | gpt2-medium | pythia-410m | smollm2-360m | qwen2.5-0.5b |
|---|---|---|---|---|---|---|---|---|---|---|
| hellaswag | acc_norm | 0.25 | 0.2826 | 0.2758 | 0.2729 | +0.031 | 0.3926 | 0.4058 | 0.5617 | 0.5221 |
| arc_easy | acc_norm | 0.25 | 0.3809 | 0.3788 | 0.3830 | +0.177 | 0.4360 | 0.4575 | 0.6806 | 0.5863 |
| arc_challenge | acc_norm | 0.25 | 0.2261 | 0.2244 | 0.2321 | −0.024 | 0.2500 | 0.2432 | 0.3831 | 0.3242 |
| piqa | acc_norm | 0.50 | 0.5789 | 0.5734 | 0.5718 | +0.144 | 0.6638 | 0.6719 | 0.7193 | 0.6997 |
| winogrande | acc | 0.50 | 0.5185 | 0.5004 | 0.5217 | +0.043 | 0.5320 | 0.5328 | 0.5896 | 0.5635 |
| openbookqa | acc_norm | 0.25 | 0.2720 | 0.2720 | 0.2740 | +0.032 | 0.3020 | 0.2940 | 0.3820 | 0.3520 |
| sciq | acc | 0.25 | 0.6050 | 0.6440 | 0.6560 | +0.541 | 0.7690 | 0.8120 | 0.9120 | 0.9310 |
| boolq | acc | 0.50 | 0.4599 | 0.4416 | 0.4190 | −0.162 | 0.5875 | 0.6012 | 0.6205 | 0.6177 |
| lambada_openai | acc | 0 | 0.1644 | 0.1799 | 0.1989 | — | 0.4304 | 0.5164 | 0.5376 | 0.5255 |
| mmlu | acc | 0.25 | 0.2430 | 0.2283 | 0.2285 | −0.029 | 0.2293 | 0.2322 | 0.2631 | 0.4771 |
| triviaqa | EM | 0 | 0.0000 | 0.0050 | 0.0140 | — | 0.0300 | 0.0210 | 0.2190 | 0.0660 |
| nq_open | EM | 0 | 0.0000 | 0.0010 | 0.0020 | — | 0.0070 | 0.0000 | 0.0520 | 0.0260 |
| gsm8k | EM | 0 | 0.0180 | 0.0050 | 0.0090 | — | 0.0180 | 0.0180 | 0.0430 | 0.3710 |
| **mean MC headroom** | | | **+0.088** | **+0.081** | **+0.084** | | +0.193 | +0.208 | +0.345 | +0.335 |

## What it says

**1. Post-training moved the trunk's benchmark position by nothing.** Mean MC headroom reads
0.088 / 0.081 / 0.084 across the three — a 0.007 spread over 758M tokens of finetuning, and the
repair column is *above* the SFT one. Every individual task that moves does so by two points or
less except SciQ (+5.1 from pretrained to repair) and BoolQ (−4.1). This is the reassuring half of
what the suite was built for: the two narrow finetunes that reshaped the abstention policy did not
cost general capability. It is also the sobering half — they did not buy any either, which is what
"a policy change on a fixed representation" looks like from the outside, and matches
[answerability_probe.md](answerability_probe.md) finding the same thing at the representation level.

**2. Where the model sits in its class: below gpt2-medium, on-trend for its tokens.** +0.084 mean
headroom against gpt2-medium's +0.193 at ~10B tokens, Pythia-410M's +0.208 at 300B, SmolLM2-360M's
+0.345 at 4T. The claim the plan makes is "on-trend for its token budget", not "competitive", and
this is the number that has to be defended: at 16B tokens on a seven-source mix the model is
roughly half of gpt2-medium's headroom. Two mitigations that the table itself supports — SciQ at
+0.54 and ARC-Easy at +0.18 are real signal, not noise, so the architecture is learning something
retrievable — and one that it does not: nothing here separates "architecture wastes capacity" from
"hasn't seen the tokens" on its own. That separation is what the Phase 3–6 ablations are for, each
against its own arm at matched compute.

**3. MMLU and closed-book QA are at zero, exactly as designed.** MMLU 0.229–0.243 is chance on all
three columns and on three of four peers; only Qwen2.5-0.5B at 18T tokens has the knowledge.
TriviaQA 0.000 → 0.014 and NQ-open 0.000 → 0.002 are the "knowledge did not arrive" axis in its
starkest form, and they are the exact numbers Gate G5's corpus-attached delta gets measured
against. SmolLM2's TriviaQA 0.219 is what 4T tokens of closed-book recall buys and what the
evidence pathway intends to supply at eval time instead — the gap the whole thesis is aimed at.

**4. BoolQ is the one number that is a finding rather than a floor, and it got worse.** 0.4599 →
0.4416 → 0.4190 on `acc`, monotonically down across post-training, all three below the 0.50 chance
of a two-option task. `acc_norm` on the identical scores reads 0.6162 / 0.6131 / 0.5920, so the
model prefers `" no"` on raw log-probability and byte normalization (`" no"` is shorter than
`" yes"`) flips most of it back. That is a yes/no answer-policy bias, it deepened under exactly the
finetunes that were teaching the model when to decline to answer, and average CE cannot see it.
Worth re-reading after Phase 4, whose no-evidence condition trains a different decline behaviour.

**5. GSM8K stays at ~0 and the RL unpark stays parked.** 0.018 / 0.005 / 0.009 greedy EM, against
the ~15% pass@8 threshold Step 16 unparks on. The standing measurement exists now; it says nothing
has changed.

## The MTP skip, confirmed end to end

The repair column reproduces the
[2026-08-23 shakedown](benchmark_suite.md#shakedown-of-the-local-model-path) **to four decimals on
all thirteen tasks**. That run predates [NEXT.md](../plans/NEXT.md)'s 1b.4 (`skip_mtp`), this one
follows it, and the two agree exactly — which is what `tests/test_mtp_skip.py` asserts at unit scale,
now confirmed on the real model over 148k scored continuations and 3,000 generations.

What changed is the cost:

| repair checkpoint, identical numbers | 2026-08-23 (MTP head running) | 2026-08-26 (`skip_mtp`) |
|---|---|---|
| whole suite | 36.0 min | **7.3 min** |
| GSM8K alone | 24.0 min | **4.7 min** |
| peak memory | 3.97 GB | 3.97 GB |

Memory is unchanged — the head's activations were never the constraint. The saving is pure compute
on the generative tasks, where the head ran over the whole prefix at every decode step and its
output was discarded.

Across the three columns the suite costs 11.6 / 9.1 / 7.3 min, of which GSM8K is 9.1 / 6.5 / 4.7.
That spread is generation length, not scoring: the pretrained checkpoint runs to the 256-token cap
far more often, visible in its GSM8K samples repeating one sentence until the budget runs out.
Post-training taught it to stop, which is worth noting as the one capability the finetunes clearly
did add and that no accuracy column here scores.

## Reproducing

Payloads are `docs/measurements/benchmarks/snapshot_{phase2_final,sft_final,repair_055}.json`
(untracked — `.gitignore` swallows `*.json`). Regenerate with the command at the top; the peer files
are frozen and must not be re-measured.

**Batch size is part of the measurement.** `ParallelSparseMoELayer` tiles its grouped GEMM by
per-expert row counts computed over every token in the batch, padding included, so a run at a
different `--batch-size` is not comparable to this table.

One operational note from producing it: a full-suite run segfaulted once, mid-GSM8K, on a box whose
WSL instance had already restarted itself once that session. It did not reproduce — GSM8K alone on
the same checkpoint ran clean, and all three re-runs completed — so it is recorded here as
environment flakiness rather than a defect. Run the three checkpoints as independent invocations
(`;`, not `&&`) so one crash cannot cost the other two.

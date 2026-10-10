# Chain depth arm at micro scale: the spec (written 2026-10-08, before the build) and the read

Approved 2026-10-06 (NEXT.md ladder step 8; Decisions, "loops (axiom)"); everything down to "What
this arm cannot read" was fixed before the build, and the run and its read (2026-10-08,
**inconclusive**: the readable precondition fails) are in the last section. Splits built 2026-10-08. The first launch at `batch_size` 32 spilled into
shared memory in minute one (32 GB, 10k tokens/s), as the memory paragraph allowed for; the run
uses the fallback, `batch_size` 16 with `grad_accumulation_steps` 2 (the same tokens per update),
in `ckpts/inject/config_micro_chains.yaml`. Numbers come from the code and from the chain eval splits on disk
(`data/prepared/chains_{eval,heldout_tmpl,hop4}`, built 2026-10-02); where the code gives no number,
the bound the criterion needs is stated instead.

## Question, and what it decides for the pilot

Reasoning is to live in the looped trunk, through the read every loop. In the micro shape
`read_sites_per_loop = 1` (`moe.py`): depth D has D sequential read sites, one per pass, and
`reader_sites_kept = j` cuts the key/value reader and the IR external read in every pass after the
j-th. A 2-hop question needs two sequential reads, so with one site it sits at chance. L1 at micro
scale: read sites past the first pass add at least 5 points on held-out 2-hop at D = 3, read as
`Delta = acc(chains_heldout_tmpl, 2-hop, D 3, kept 3) - acc(same questions, D 3, kept 1)`.

- **Pass**: the loop clause holds at micro scale; the pilot spec keeps the chain slice (10%, hops
  25 / 50 / 25) and reads L1 again at its shape, where two sites per pass change "past the first pass".
- **Falsified**: per Decisions the loop clause is wrong at this scale. Looping stays a requirement;
  the pilot spec records the clause as falsified at micro and makes the pilot's L1 (sublayer sites
  against passes) the deciding read instead of carrying the clause as an assumption.
- **Inconclusive**: L1 stays unmeasured at micro, no change to depth or the chain slice; a larger
  budget is a separate decision. An invalid instrument is a generator leak, and the pilot's chain
  slice uses the same generator, so it is fixed before the pilot either way.

## Splits

| split (seed 42) | questions | hops | tokens / q | evidence tokens / q | chunks / q | candidates | chance |
|---|---|---|---|---|---|---|---|
| `chains_eval` | 3,000 | 1 / 2 / 3 | 48.2 / 52.8 / 55.6 | 191.7 / 218.3 / 248.4 | 10.9 / 12.0 / 13.2 | 9.44 / 5.78 / 5.28 | 0.115 / 0.185 / 0.206 |
| `chains_heldout_tmpl` | 1,000 | 2 | 49.2 | 220.0 | 12.0 | 5.49 | 0.191 |
| `chains_hop4` | 1,000 | 4 | 58.6 | 266.9 | 14.4 | 5.14 | 0.210 |

Held out (`chains.split_compositions(42)`): 3 of 14 typed 2-tuples, founder then alma_mater,
birthplace or employer; 3-hop training tuples drop any tuple holding one (15 of 22 kept); 380 hop4
questions contain one. Two builds, WSL, repo root:

```
python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains --splits heldout_tmpl --eval-questions-per-hop 2000 --seed 42 --device cpu --overwrite
python scripts/prepare_chain_data.py --out-dir data/prepared --prefix chains --splits train,val --train-questions 2000000 --val-questions 2000 --seed 42 --device cuda
```

- Held-out rebuild to 2,000 for power (Reads). The draw is seeded by seed and split name, so its
  first 1,000 draws are today's questions, reshuffled; nothing reads the split now. CPU, minutes.
- Train: hops 25 / 50 / 25 over seen compositions, 6 to 14 distractors, entities fresh per question,
  so no fact recurs and nothing can be stored. Per question, hop-weighted from `chains_eval`: 52.4
  tokens (7.2 supervised: answer and EOS), 219 evidence tokens in 12.0 chunks, evidence ratio 4.2
  (the injection corpus 0.10). No filler: the builder writes chain rows only and nothing mixes them
  with prose, so the arm learns a closed synthetic language from random init. 2,000,000 questions:
  about 104.7M tokens, 24.0M chunks; val 2,000.
- Disk 10.9 KB per question (measured on `chains_eval`, 7.7 KB of it fp16 keys): 21.8 GB (264 GB free
  on D:). The builder holds all questions before writing, 4.4 KB each measured: about 9 GB of the
  31 GB WSL. Sampling about 4k questions/s (one core, measured under tracemalloc), then 24.0M unique
  chunks embedded at an unmeasured rate: expect 30 to 60 minutes. It logs only at the end of a split;
  progress is the size of `chains_train.evkey` (768 bytes per chunk).

## Training

WSL, after `source env_init`, background under Monitor, filter
`Traceback|Error|Killed|OOM|assert|device not ready|Step|Tokens/sec|\[eval\]|packing`:

```
TINY_LLM_CONFIG=config_micro.yaml python scripts/sft.py --evidence --reader-kv -c ckpts/inject/seed_micro.pt --run-name chains_kv --data-dir data/prepared --train-split chains_train --val-split chains_val 2>&1 | tee ckpts/inject/chains_kv.log
```

- **Token target about 104.7M, one epoch of 2M questions.** `sft.py` fits the cosine to one epoch's
  packed rows (`num_epochs: 1`); no flag sets tokens, so the split is the budget. R0b's key/value
  reader had not copied at 50M and copied at 100M, on about 650k supported spans in 300M tokens; here
  every question is a supervised read, so 2M give three times R0b's copy targets (1.5M at two or more
  hops) in a third of its tokens. An extension is a larger rebuild (its first 2M questions equal
  these) run from the seed, not a relaunch.
- Packing: rows close on the 3,072 evidence cap at 13 to 14 questions, about 707 tokens per 1,023-slot
  row (fill about 70%; R0b 73 to 82%), 22.6k tokens per step, about 4,600 steps (R0b 9,300).
- Wall time 32 minutes at 55k tok/s, a rate measured at evidence ratio 0.10; with 4 evidence tokens
  per prompt token through the 4-layer encoder, expect 30k to 55k tok/s, 32 to 58 minutes.
- Memory: R0b's 13.3 GB used 2 to 4% of the evidence cap; this corpus fills it (98k evidence tokens
  per step against about 3k). The full shape's 0.61 MiB per evidence token scaled by width and depth
  (256 / 768, 4 / 8 layers) gives 0.1 MiB, a peak near 20 GB, unmeasured; a spill shows as tok/s
  collapsing, not an OOM. Fallback fixed now: a copy of `config_micro.yaml` with
  `evidence.batch_size: 16` (half the evidence per step) passed as `TINY_LLM_CONFIG`.
- `[eval fixed]` and the kill stay off (`fixed_split: ""`, `kill_tokens: 0`): the fixed pass reads
  prose a chain-only model cannot read, and chain rows carry only `mixed`. The `[eval]` pass on
  `chains_val` (every 25M, 14.4k answer tokens) is the training curve.
- Saves every 50M: 50M, 100M, final in `ckpts/evidence_chains_kv/`, 352 MB each; 50M and final are read.

## Reads

Per read save (`<tag>` 50M, final), `TINY_LLM_CONFIG=config_micro.yaml`, each teed to its `.log`:

```
python scripts/eval_chains.py -c CKPT --data-dir data/prepared --splits chains_eval,chains_heldout_tmpl,chains_hop4 --depths 1,2,3 --sites all --max-questions 2000 --batch-size 256 --json-out ckpts/inject/chains_kv_<tag>_d123.json
python scripts/eval_chains.py -c CKPT --data-dir data/prepared --splits chains_hop4 --depths 3,4 --sites all --max-questions 2000 --batch-size 256 --json-out ckpts/inject/chains_kv_<tag>_hop4.json
```

- The first carries L1, validity and depth over the trained exits (`loop_count_sampling` 0.3 trains
  exits 1 and 2). A forward at `n_loops = D` equals the exit after D passes, so its full-site cells
  are the per-exit read on the chain task, in accuracy. The second is depth past training: 4-hop is
  answerable only at D = 4; its "not readable" line is by construction (no 1-hop cells), ignored.
- `--batch-size 256` (16 is launch bound): about 56k evidence tokens per forward, no backward, peak
  unmeasured; about 376k question-candidate rows per save, 5 to 15 minutes, unmeasured.
- **`eval_exit.py` cannot read this arm**: it refuses every split with a `.ev` file, and the
  evidence-free splits (`inject_val`, the `data/prepared` val bins) are prose this arm never saw.
  Not run; the full-site cells replace it.
- **Power.** `eval_chains.py` writes per-cell accuracy and binomial sigma, no per-question record, so
  Delta is unpaired, `sigma = sqrt(s3^2 + s1^2)`; kept 1 sits at chance, where picks are near
  independent of the full-site ones, so this is close to paired. At chance 0.191 and kept 3 five
  points above: n 1,000 gives sigma 0.018, 5 points at 2.7 sigma; n 2,000 gives 0.013, 3.9 sigma
  (3.2 at the worst case, both cells at 0.5). Hence the rebuild.

## Criteria, fixed before the read

At the final save; 50M is read the same way for the dose. Preconditions, from the D 1 to 3 read:
valid (every cell with kept < hops, 32 cells, within 3 null sigma of chance; a false alarm somewhere
has about 8% probability, so a lone cell under z 4 is checked at the other save before it is called
a leak); readable (`chains_eval` 1-hop at D 3, kept 1 to 3, at least 3 null sigma over chance: 0.146
against 0.115); composes in distribution (`chains_eval` 2-hop at D 3, kept 3 minus kept 1, at least
3 sigma: about 0.055 at n 1,000).

- **Pass**: Delta at least 0.05 and Delta / sigma at least 3.
- **Falsified**: preconditions met and Delta + 2 sigma under 0.05 (Delta under about 0.024): seen
  pairings compose and the sites past the first pass do not carry it to held-out ones.
- **Inconclusive**: everything else, a failed precondition included.
- **Kill**: an invalid instrument at 50M (a data leak; nothing downstream reads), or throughput or
  memory failing in minute one (relaunch on the fallback). Not readable at 50M is no kill: R0b's
  key/value reader had not copied at 50M either.

Secondary, with sigma, no verdict: 1-hop and 3-hop at D 3 by kept (3-hop at chance through kept 2);
seen against held-out 2-hop at D 3, kept 3; hop4 at D 4, kept 4, against chance, its answer CE by
kept and D 4 against D 3 (the L2 shape; depth past 3 degrades on plain text); full sites by depth,
1-hop D 1 against D 3, 2-hop D 2 against D 3; 50M against final; per-loop `[eval]` CE, `loop_scale`.

## Cost

| item | device | time | disk |
|---|---|---|---|
| held-out rebuild | CPU | minutes | 22 MB |
| train, val build | cuda (bge-small, a few GB) | 30 to 60 min, unmeasured | 21.8 GB; 9 GB host RAM |
| training | cuda | 32 to 58 min | 1.1 GB (3 saves) |
| reads, 2 saves | cuda | 10 to 30 min, unmeasured | small |
| total | | 1.3 to 2.6 hours | about 23 GB |

The embedder fits beside any micro trainer, so the build can run during the placeholder name arm. A
batch-1024 read beside the R0b trainer peaked about 17.9 GB of 32; the chain trainer's peak (near
20 GB, estimated) is unknown, so nothing shares the card with it until minute one shows it, and a
read runs beside it only if both peaks sum under about 30 GB.

## What this arm cannot read

Sites are one per pass here, so "past the first pass" means passes 2 and 3; the pilot reads in
sublayers 1 and 3, and whether passes or sublayers compose there is the pilot's own L1. The held-out
test is narrow (all three pairings start with founder) and the corpus has no prose, so the arm says
nothing about chains inside a prose mix or HotpotQA bridges (L2). A null at 35M parameters and 105M
tokens speaks to that capacity and dose, hence a failed precondition is inconclusive, not falsifying.

## Read 2026-10-08: inconclusive, the 1-hop curve is not readable

### The run

- **Splits.** `chains_heldout_tmpl` rebuilt at 2,000 questions. `chains_train`: 2,000,000
  questions, 104,663,146 prompt tokens, 439,093,388 evidence tokens in 24,057,982 chunks (ratio
  4.20, 12.0 chunks per question); `chains_val` 2,000 questions (104,683 tokens). 21 GB on disk;
  the build took about 55 minutes on cuda beside the placeholder name arm's trainer.
- **Training.** `sft.py --evidence --reader-kv` from `seed_micro.pt`, run `chains_kv`, saves in
  `ckpts/evidence_chains_kv/` at 50M, 100M and final. The first launch on `config_micro.yaml`
  (batch 32) spilled into shared memory in minute one (31.9 GB, 3k to 10k tokens/s,
  `ckpts/inject/chains_kv_spill_batch32.log`); stopped at step 10, its run directory removed, and
  relaunched on `ckpts/inject/config_micro_chains.yaml` (batch 16, grad accumulation 2, the same
  32k tokens per update): 13.6 GB peak by the trainer's own line, about 33k tokens/s, 102.66M
  tokens in 51.5 minutes (`ckpts/inject/chains_kv.log`). Packing fill 70 to 73%, every row closed
  by the evidence budget (96 to 97% of the 3,072 cap), as the spec predicted.
- **Reads.** Both `eval_chains.py` lines of the Reads section on the 50M and final saves,
  `ckpts/inject/chains_kv_{50M,final}_{d123,hop4}.{log,json}`.

### Trainer curve

`[eval]` on `chains_val`: answer tokens and EOS, teacher forced, 14,647 supervised tokens (the
whole split).

| tokens | CE | top-1 | `selection:` BCE |
|---|---|---|---|
| 0 | 11.13 | 0.000 | 0.495 |
| 25M | 2.259 | 0.461 | 0.457 |
| 50M | 0.628 | 0.849 | 0.446 |
| 75M | 0.501 | 0.865 | 0.442 |
| 100M | 0.455 | 0.871 | 0.441 |
| final, 102.66M | 0.452 | 0.870 | 0.440 |

The IR selector's entropy stayed at 0.994 of uniform over 256 (`IR E/ln256` at the end): it never
separated the gold chunks. `loop_scale` went from [0.578, 0.578, 0.578] to [1.34, 0.48, 0.07]:
the third pass's scale fell below 0.01 by 50M (0.007 at step 2990) and recovered only to 0.07,
while the first pass's rose.

### Reads

Accuracy among the type-matched candidates, binomial sigma, z against chance; n 1,000 per hop on
`chains_eval` and `chains_hop4`, 2,000 on `chains_heldout_tmpl`.

| cell | final | 50M | chance |
|---|---|---|---|
| `chains_eval` 1-hop, D 3, kept 0 (no read) | 0.107 (z -0.8) | 0.127 (z +1.2) | 0.115 |
| `chains_eval` 1-hop, D 3, kept 1 / 2 / 3 | 0.102 / 0.106 / 0.104 (z -1.1 at kept 3) | 0.113 / 0.119 / 0.118 (z 0.3) | 0.115 |
| `chains_eval` 2-hop, D 3, kept 1 / 3 | 0.193 / 0.187 | 0.179 / 0.177 | 0.185 |
| `chains_eval` 3-hop, D 3, kept 3 | 0.194 | 0.216 | 0.206 |
| `chains_heldout_tmpl` 2-hop, D 3, kept 1 / 3 | 0.188 / 0.194 | 0.185 / 0.184 | 0.191 |
| L1 Delta (held out, kept 3 minus kept 1) | +0.006, sigma 0.013 (z 0.5) | -0.001 | |
| `chains_hop4`, D 3 kept 3 / D 4 kept 4 | 0.195 / 0.188 (z -1.7) | 0.224 / 0.223 | 0.210 |

Validity holds: none of the 32 cells with fewer kept sites than hops is outside 3 sigma of chance,
at either save. The script's "instrument valid: no" line comes from the readable check, not from a
leak: "1-hop curve not readable: 6 cells under 3 sigma above chance" (the readable bar was 0.146).
Composes in distribution fails too (2-hop kept 3 minus kept 1, -0.006 final).

Answer CE per token by kept sites on the 4-hop split:

| | j = 0 | j = 1 | j = 2 | j = 3 | full sites at D 1 / 2 / 3 |
|---|---|---|---|---|---|
| final, D 3 | 3.824 | 0.488 | 0.468 | 0.468 | 0.485 / 0.468 / 0.468 |
| 50M, D 3 | 3.434 | 0.658 | 0.640 | 0.639 | 0.656 / 0.639 / 0.639 |

At D 1 with no site (j = 0) the final save reads 4.147. D 4 against D 3 on the full sites is 0.468
against 0.468 (50M 0.640 against 0.639): depth past training changes nothing here, since there is
nothing to compose.

### Diagnosis

An Opus read of `eval_chains.py` and the numbers found no bug: the scorer uses the training chat
template and answer tokenization, the gold is always among the candidates, the batch size does not
change scores, and rows are independent. The per-token answer CE and the candidate accuracy do not
contradict each other. An answer is about 7.3 tokens with EOS, so 0.455 per token is about 3.3
nats per answer, while a uniform pick among the candidates costs ln K, 2.24 / 1.76 / 1.66 nats at
K 9.4 / 5.8 / 5.3 by hops (about 1.9 hop-weighted): the gold gets less than a chance share. The
model copies an answer-type entity from the buffer (without a read site the answer costs about 28
nats; with one, 0.49 per token) and, under teacher forcing, copies the rest of the name perfectly
after one wrong decision at the first distinctive token, which is what top-1 0.87 per token means.
It does not select which entity: not even 1-hop lookup was learned, at 35M parameters and 105M
tokens with 2M questions. The read sites do carry content (kept 0 against kept 1 is 3.3 nats per
token), so the reader works as a copier; what fails is the choice.

### Verdict

Under the criteria fixed before the read: the readable precondition fails (1-hop at D 3 is not 3
sigma over chance at any kept count), so the arm is **inconclusive**, neither a pass nor a
falsification, and L1 stays unmeasured at micro scale. The L1 numbers alone (Delta + 2 sigma =
0.032, under 0.05) would have met the falsified bound; they do not count, because a model that
cannot do 1-hop lookup cannot show whether later sites compose. Not a kill: the sub-hop cells sit
at chance (vacuously, since nothing is above chance) and the instrument is intact. The dose
reading: 50M and final read the same, and the trainer curve had flattened by 75M, so another 50M
of the same corpus would not change it.

### Readings (not established)

- The chain task at micro scale is a selection task the key/value read does not solve by itself.
  The R0b biography copy worked because the document named the person and the card carried the
  same name, a surface match the read can key on; here the question names one entity among 12
  chunks of the same shape, the gold chunk has to be picked by its subject, and the selector stayed
  uniform (`selection:` 0.495 to 0.440).
- The third pass's scale collapsing to 0.007 and recovering only to 0.07 is the "weak later loop"
  case the rule says to fix, not cut. On a task where no pass composes, the trainer had no reason to
  use the third pass; it says nothing about a task where one does.

### Consequences for the plan

Per the "Inconclusive" clause: no change to depth or to the pilot's chain slice (10%, hops 25 /
50 / 25). The pilot's own L1 (sublayer sites against passes, at the pilot's shape and dose) is the
deciding read, which is already the pilot spec's job; R2's rule ("a 1-hop curve must saturate
before any null is read") applies there as here. What would make a micro read possible is a
separate decision, recorded as an option, not a next step: a larger model or dose, a curriculum
that teaches 1-hop lookup first, or a selector loss that is not left uniform. The next step stays
the pilot spec.

### Instrument gaps (worth fixing before the next chain arm)

Closed 2026-10-10 in `eval_chains.py` (`tests/test_eval_chains.py`): per-question records under
`records` in the JSON, gold NLL against ln K for every split and hop, the NLL at the first token
where the candidates diverge, a paired bootstrap delta between the fewest and the most kept sites
(`readings.paired_delta`), and the `*` marker in its own column. The list below is the record of
what was missing when this arm was read.

- The JSON holds per-cell accuracy only: no per-question record (candidate scores, gold rank,
  margin), so Delta cannot be paired and a diagnosis like the one above has to be argued from
  aggregates.
- Answer CE by kept sites is printed for the 4-hop split only; it is wanted for every split and hop.
- In the printed table the `*` marker glues onto the previous column (`-0.1*0.178`).
- Wanted: per-answer gold NLL against ln K, and the CE at the first token where the candidates
  diverge, which is the token that carries the selection.

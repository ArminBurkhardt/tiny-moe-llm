# tiny-moe-llm

An experimental **~383M-parameter** language model exploring a **looped, sparsely-routed
mixture-of-experts** architecture on top of a dense Gemma4-style backbone, with an **evidence port**
that lets retrieved text reach the model through a side channel instead of the prompt.

A dense decoder feeds a *single* MoE block that is applied `n_loops` times, so depth is recurrence
rather than parameters. 383M total / 224M active per token, of which a 50.3M-parameter
information-retrieval table is a lookup rather than a matmul.

One 16B-token run exists, pretrained on a rented H100 and fine-tuned locally, then carried through
a sequence of measured finetunes. [docs/CONCLUSION.md](docs/CONCLUSION.md) is the write-up of the
pretraining run and [docs/measurements/](docs/measurements/) holds every gate since — read them
before believing anything in this list. The plan is [docs/plans/NEXT.md](docs/plans/NEXT.md); its
"Now" section is what happens next.

The following techniques were used/implemented:

- **Dense Gemma4-style blocks** — GQA, RoPE, RMSNorm, per-layer embeddings (PLE)
- **Looped MoE** — one MoE block applied for `n_loops` iterations, rerouting tokens each pass
  (LoopLM-style recurrence), with routing conditioned on the loop index so consecutive loops do not
  all pick the same experts
- **Heterogeneous experts** — self-attention, cross-attention, information-retrieval and MLP
  experts share one router, plus always-on shared MLP/attention experts outside the router pool
- **Evidence port** — retrieved chunks are encoded by the model's own dense decoder and read at
  every loop by an always-on cross-attention *reader*, while the IR expert acts as a *selector*
  scoring the chunks' external embeddings under the same softmax as its learned table. With no
  evidence attached the forward is bit-identical to the model without the port
- **Per-loop CE supervision** — every loop's readout is supervised, not just the last, so an
  early-exit policy has something to exit *to*
- **Stochastic loop depth** — a fraction of steps train at a reduced depth, making the loop count a
  real runtime choice at inference
- **Parameter-free convergence exit** — inference stops looping once the last position's readout
  stops moving. No learned gate, no loss term, nothing that can saturate
- **Multi-token prediction (MTP)** — auxiliary heads predict several future tokens per step, reused
  at inference as self-speculative drafting
- **KV-cached decoding** — one cache slot per decoder layer and per (loop, attention expert) pair
- **Document packing** — multiple documents per sequence with block-diagonal causal attention via
  flash-attn varlen
- **Low-precision training** — optional FP8 / NVFP4 via NVIDIA Transformer Engine

There is no identity expert. There is also no longer a learned halt head or correctness head: both
were tried, both failed structurally rather than by mistuning, and both were removed. See
[docs/moe.md](docs/moe.md) for what replaced them and [docs/CONCLUSION.md](docs/CONCLUSION.md) for
the measurements that decided it.

<details>
<summary><b>Model flow</b> — one pass, end to end</summary>

383M total, 224M active per token, ~502M forward FLOP/token at `seq_len=4096`, plus ~118M FLOP per
*evidence* token when a buffer is attached (the decoder run once over the evidence).

```
tokens [B,S] = 4×4096, document-packed (uint16)          evidence [B,S_ev] (optional)
  + document_ids ──► cu_seqlens ──► varlen flash-attn     + chunk ids ──► per-chunk cu_seqlens,
        │                                                 │             positions restart per chunk
        ▼                                                 ▼
embed_tokens (65536 × 768) + per-layer embeddings (32/layer)   same table, same decoder ─┐
        │                                                                                │
        ▼                                                                                │
┌─ DENSE DECODER — Gemma4-style, 8 layers, runs ONCE ──────┐                             │
│  GQA (12 heads, head_dim 64) + RoPE + RMSNorm(TE)        │   evidence states [B,S_ev,768] ◄┘
│  SwiGLU MLP (2304) · layer_scalar gain per layer         │   (encoded once, reused every loop)
└──────────────────────────────────────────────────────────┘        │
        │  h [B,S,768], RMS ≈ 1                                     │
        ▼                                                           │
╔═ MoE BLOCK — ONE module, applied n_loops = 3 times ═══════════════╪═══════╗
║                                                                   │       ║
║  router: 35 experts, top_k = 2                                    │       ║
║    + loop_router_bias(sinusoidal loop index)                      │       ║
║    + exploration noise × 0.3 (annealed → 0)                       │       ║
║                                                                   │       ║
║  accumulator seeded ALWAYS-ON, outside the router:                │       ║
║    shared_mlp (SwiGLU 2304) + shared_attn                         │       ║
║    + evidence_loop_scale[k] · shared_evidence(q + loop bias) ◄────┤       ║
║        reader: cross-attn over evidence states, per-chunk gate    │       ║
║        from the selector's chunk mass, o_proj zero-init           │       ║
║                                                                   │       ║
║  [0] SelfAttn  [1] CrossAttn  [2] IR ◄── selector: chunk_keys ────┘       ║
║   └──── run UNCONDITIONALLY ────┘         [3..34] MLP ×32 (grouped GEMM) ║
║        (routing only scales output)                                       ║
║                                                                           ║
║  h = h + loop_scale[k] · dropout(post_norm(Σ))                            ║
║       loop_scale ≈ [0.6, 0.3, 0.1] on the migrated checkpoints            ║
╚═══════════════════════════════════════════════════════════════════════════╝
        │  norm() applied at EVERY loop → [3,B,S,768]
        ▼
lm_head (factored ÷4) ──► per-loop CE, weighted
                          (non-final loops subsampled to 0.25)
MTP heads (+2 tokens) ──► chunked checkpointed CE on the final loop
```

Depth is recurrence, not parameters: the MoE block's weights appear once in the parameter count and
run three times. The non-MLP experts run for every token regardless of routing — attention has to
see the whole sequence — so the top-k mask only scales their output; sparsity buys compute for the
32 MLP experts alone. Everything on the evidence side is absent when no buffer is attached, and the
reader's output projection is zero-initialized, so a freshly ported checkpoint scores identically to
its source with or without a corpus.

</details>

<details>
<summary><b>The information-retrieval expert</b> — selector over a learned table and an external store</summary>

```
h ─► RMSNorm ─► down_proj 768→384 ─► + loop_query_bias(k) ─► F.normalize ─► query
                                                                          │
   learned table, two stage read:   256 centroids → top-8 clusters        │
                                    → 8×256 = 2048 keys scored exactly    │
                                    → global top-32                       │
   external store (evidence):       key_adapter(bge chunk vectors) ───────┤ one softmax
                                    scaled by exp(log_memory_scale)       │ over the union
                                                                          ▼
                                    external mass = Σ external weights  (the G3b signal)
                                                                          │
   values: unit-normalized rows / value_adapter(chunk vectors) ─► retrieved_y [.,384]
                                                                          │
   g_proj 384→384 ─► up_proj 384→768 ─► direct_gate 768→768 (zero-init) ─► expert output
```

The two stage read is what makes a 65536-entry table affordable: an exact read would cost ~101M
FLOP/token/loop against this path's 2.58M, in a ~502M-FLOP model. Clusters are exactly equal in size
so candidate scoring is one `bmm` with no ragged gather and no host sync.

**The learned table does not carry content, and three arms established why.** Zeroing its read costs
0.0002 nats of held-out CE across three key inits and two table widths, trained or not
([ir_sharpening.md](docs/measurements/ir_sharpening.md),
[ir_scale_fix.md](docs/measurements/ir_scale_fix.md)): a table trained on the trunk's own corpus can
only offer content the trunk already holds, so nothing pays for opening the valve. Its size is
frozen out of the real run spec. The external store is the opposite case — a gold passage carries
content the model provably lacks, worth **+3.23 nats** in context
([evidence_ceiling.md](docs/measurements/evidence_ceiling.md)) — and the same module reads it under
the same softmax, so "how much of the read went external" is a measurable groundedness signal.

The expert's original output stage, an inner attention over every position's own read, averaged a
document's reads over its whole prefix (replacing the read by its batch mean cost 0.0000 nats); the
`direct_gate` path writes each token's own read instead and is the default. The old stage stays
loadable for the A/B.

</details>

<details>
<summary><b>Inference</b> — caching, drafting, evidence, and the depth policy</summary>

`scripts/inference.py` is the reference path and `scripts/gradio_app.py` imports `stream_generate`
from it, so the CLI and the UI cannot drift.

| flag | what it does |
|---|---|
| *(default)* | KV-cached decode. Exact, not approximate: every attention call is causal, so a past token's output at a given depth never changes when later tokens are appended |
| `--no-kv-cache` | the slow full-prefix reference path |
| `--evidence FILE` | attach a buffer (JSON list of chunks, or one chunk per line) through the port for the whole session. Needs a checkpoint carrying the port |
| `--num-mtp-tokens N` | self-speculative drafting off the same step's final hidden state, greedily accepted with **no** rejection sampling — trades quality for forward passes. Default 0 |
| `--converge-tol T` | the convergence exit (below). Forces the KV cache **off** |
| `-n / --temperature / -p` | length, sampling temperature, prompt |

**The KV cache** holds one slot per dense decoder layer, plus one per `(loop, non-MLP expert)` pair
*and* per `(loop, shared_attn)` — the MoE block is the same weights re-applied over an evolving
hidden state, not three independent layers, so each loop needs its own cache. Single-sequence only.
The evidence reader has no cache slot: it recomputes its cross attention over the fixed evidence
states every step, which is correct but slower, and under caching its per-step shape differs, so
greedy decodes with evidence can diverge from the uncached path after a few tokens (reduction
order, not RNG). Use `--no-kv-cache` when exact reproducibility with evidence matters.

**The depth policy is parameter-free.** After each loop the model reads out the **last position
only** and stops when the top-1 token is unchanged *and* its log-probability moved by less than
`converge_tol`. Three things make it work:

- It reads the **readout**, not `‖Δh‖`. `loop_scale` still injects a sizeable hidden delta on the
  last loop while the prediction is already stationary, so a hidden-state criterion never fires.
- It is asserted **inference-only** — a short `hidden_states_all` would silently break per-loop CE.
- It is asserted **mutually exclusive with the KV cache**: an exited loop appends no K/V for that
  token, so a later full-depth step would attend over a cache with a hole in it. Filling those
  cheaply is real plumbing through every attention expert, and is unimplemented.

Pick the threshold from `scripts/eval_calibration.py`'s per-transition table. Measured on the 16B
checkpoints: loop 1→2 top-1 agreement ~0.82 with mean `|Δ log p|` ~0.21, loop 2→3 ~0.94 / ~0.07.

Streaming re-decodes the full generated id sequence each step and yields only the new suffix — a
lone step's tokens can decode differently out of context, because of subword and space merges.

</details>

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/plans/NEXT.md](docs/plans/NEXT.md) | **The plan**, with a "Now" section at the top: what is done, what is next, every gate and its result |
| [docs/runbook.md](docs/runbook.md) | **Start here for a real run**: what to run, how to stop it, what is normal, what to do when it is not |
| [docs/CONCLUSION.md](docs/CONCLUSION.md) | What the 16B-token run actually produced, including the failures |
| [docs/measurements/](docs/measurements/) | One record per gate: Phase 0 migration, Stage 0 diagnostics, abstention repair, the benchmark suite and snapshot, the answerability probe, the three IR arms, the loop injection arm, the evidence ceiling |
| [docs/review_2026-09-18.md](docs/review_2026-09-18.md) | The pre-Phase-4 codebase review against the retrieval literature, and what it changed |
| [docs/architecture.md](docs/architecture.md) | End-to-end model architecture and data flow, evidence path included |
| [docs/moe.md](docs/moe.md) | Looped MoE, routing, expert types, the IR selector, the evidence reader, depth policy |
| [docs/training.md](docs/training.md) | Pretraining pipeline, the finetune profiles, data packing, losses, precision, checkpointing, resume |
| [docs/configuration.md](docs/configuration.md) | Full `config.yaml` reference |
| [docs/looped-transformers.md](docs/looped-transformers.md) | Published looped-transformer work and where this model agrees or disagrees |
| [CLAUDE.md](CLAUDE.md) | Operational map: invariants, gotchas, what lives where, what is next |

## Running anything

**Everything runs under WSL, from the repo root, after `source env_init`.** The dev box is Windows;
CUDA, flash-attn and Transformer Engine all live in the WSL Ubuntu install. `env_init` is
gitignored — it sets `CUDA_HOME`, the library paths, `PYTORCH_CUDA_ALLOC_CONF`, and activates
`venv/`. From PowerShell that is:

```powershell
wsl bash -lc "cd /mnt/d/AI/llm/dev/worth_a_try/new/tiny-llm && source env_init && python scripts/inference.py --help"
```

Nothing works without it, and nothing works from a different working directory either — `config.py`
opens `config.yaml` by relative path. The rented box is the one exception: `scripts/setup.sh`
installs into the NGC image's system python and there is no `env_init` to source there.

## Quick start

```bash
# 1. environment, HF token, tokenizer, preflight checks
bash scripts/setup.sh --hf-token hf_xxx

# 2. build the pre-tokenized corpus (hours; resumes itself if interrupted)
python scripts/prepare_data.py

# 3. train both phases, restarting through preemptions
python scripts/run_training.py
```

Or a single phase directly:

```bash
python scripts/pretrain.py --phase phase1
```

On a local box you can skip `setup.sh` and use `pip install -r requirements.txt`, but Transformer
Engine and flash-attn need CUDA builds matched to your GPU — see the
[TE installation guide](https://github.com/NVIDIA/TransformerEngine#installation). flash-attn is
optional (attention falls back to a slower SDPA path); **Transformer Engine is not** — nothing
under `modules/model/` imports without it.

Inference against a checkpoint:

```bash
python scripts/inference.py -c ckpts/repair/checkpoint_repair_final.pt -p "Once upon a time" -n 200
python scripts/inference.py -c ckpts/repair/checkpoint_repair_final_irrandom_evidence.pt \
    --evidence chunks.json -p "Who wrote it?"        # a buffer through the port
python scripts/gradio_app.py                        # the same generation path behind a UI
```

### The finetunes, in the order they were run

All local, on the 5090, in BF16, through one script (`scripts/sft.py`) and one `train_step`:

```bash
python scripts/sft.py --from-hub                                      # SFT off the pretrained trunk
python scripts/prepare_sft_data.py --profile repair                   # the abstention repair corpus
python scripts/sft.py --repair -c ckpts/trained/checkpoint_sft_final_phase0.pt
python scripts/prepare_data.py --phases ir --ir-tokens 210000000 --val-tokens 2000000 --manifest-key ir_prep
python scripts/migrate_ir_reshape.py -c CKPT --arm random             # rebuild the IR table
python scripts/sft.py --ir -c CKPT_irrandom.pt                        # sharpen it (G2 / G2b)
python scripts/migrate_evidence_port.py -c CKPT                       # add the evidence port
python scripts/prepare_evidence_data.py --target-tokens 150000000     # the oracle-evidence corpus
python scripts/sft.py --evidence -c ckpts/repair/checkpoint_repair_final_irrandom_evidence.pt
python scripts/eval_abstention.py -c CKPT --evidence-port --evidence-condition gold,none,distractors,mixed
```

### Migrating an old checkpoint

Checkpoints written before the halt and correctness heads were removed do not load: they carry two
tensors the model no longer has, and a `loop_scale` that was learned underneath a per-token gate.
Fold and strip them first — the script measures the gate on real data rather than assuming it:

```bash
python scripts/migrate_phase0.py -c ckpts/trained/checkpoint_sft_final.pt
```

The result is a finetune **seed** (its optimizer state is dropped on purpose), and it is
behaviourally identical to the original: final-loop CE moved 3.7564 → 3.7604 and top-1 0.3644 →
0.3628 on a fixed held-out slice. The later migrations (`migrate_ir_reshape.py`,
`migrate_loop_inject.py`, `migrate_evidence_port.py`) follow the same rule: every added tensor is
zero- or identity-initialized so the migrated checkpoint scores identically to its source, and the
checkpoint — never `config.yaml` — decides which of these a model carries when it is loaded.

## Repository layout

```
config.py / config.yaml        model + training hyperparameters, and the four finetune profiles
utils.py                       logger, paths, tokenizer/repo constants, checkpoint save/load,
                                 the state-dict-driven model params
scripts/
  setup.sh                     one-shot box setup: deps, token, tokenizer, preflight
  onstart.sh                   vast.ai onstart hook: clone, setup, launch the supervisor
  run_training.py              supervisor: phase 1 -> phase 2, restarts through preemptions
  run_sft_after_pretrain.sh    unattended pretrain -> SFT -> abstention eval chain
  pretrain.py                  the training loop; train_step is shared with every finetune
  prepare_data.py              builds phase1/phase2 .bin/.idx from the Hub source mix;
                                 --phases ir builds the IR sharpening corpus
  prepare_sft_data.py          builds sft_train/sft_val .bin/.idx/.mask; --profile repair
  prepare_evidence_data.py     builds the five-condition oracle-evidence corpus (+ .ev/.evkey/.evgold/.cond)
  archive_corpus.py            pack/list/restore a prepared split as one .tar.gz
  fetch_tokenizer.py           downloads the pruned 65536-token tokenizer
  sft.py                       the post-training entry point: --repair, --ir, --evidence profiles
  migrate_phase0.py            folds the deleted halt gate into loop_scale, strips both old heads
  migrate_ir_reshape.py        rebuilds the IR table at 65536 x 384 (--arm random | warm)
  migrate_loop_inject.py       adds the zero-init loop input injection (arm D, failed G2c)
  migrate_evidence_port.py     adds the reader and the selector's adapters
  inference.py                 greedy/top-k sampling CLI, KV-cached, MTP drafting, --evidence
  gradio_app.py                browser UI over the same generation path
  prune_vocab.py               one-shot 129280 -> 65536 vocab prune
  eval_calibration.py          p_max calibration, early-exit curve, loop-convergence statistics
  eval_abstention.py           SQuAD v2 abstention precision/recall + calibration; --evidence-port
  eval_benchmarks.py           the fixed 13-task suite, one scoring path for this model and 4 peers
  eval_probe.py                linear answerability probe on the final loop's hidden state
  eval_stage0.py               IR retrieval entropy + ablation, query drift, loop dynamics
  evidence_ceiling_probe.py    in-context gold / none / distractor CE on the answer span
modules/model/
  transformer.py               TinyMoETransformer + the evidence encoder + the convergence exit
  gemma4.py                    dense Gemma4-style decoder
  moe.py                       LoopMixtureOfExperts (+ shared experts, evidence reader, loop bias)
  router.py                    router + load-balancing aux loss
  experts.py                   self/cross-attention + the information-retrieval expert
  information_retrieval.py     learned key/value table, two stage read, external store, selection loss
  evidence.py                  EvidenceBatch, chunk positions/segments, reader gate, groundedness head
  mtp.py                       multi-token-prediction head + chunked LM-head loss
  attention.py                 document-packed (varlen) causal attention, separate K segments
  kv_cache.py                  incremental decode cache, one slot per layer and per (loop, expert)
  modules.py                   factored LM head
  embeddings.py                rotary position embeddings
modules/data/dataset.py        mmap flat-file dataset with document packing
modules/data/sft_dataset.py    mmap SFT dataset: explicit loss mask, per-epoch shuffle
modules/data/evidence_dataset.py  SFTDataset plus a second, evidence token stream per row
modules/data/chat.py           chat template + token-level loss masking
modules/data/abstention.py     the closed set of abstention/hedge phrasings
modules/runtime/               unattended-run machinery (no GPU/TE dependency)
  checkpoints.py               naming, retention, latest-VALID resume, resume verification
  hf_sync.py                   background uploader to the Hugging Face Hub
  control.py                   STOP sentinel + SIGTERM/SIGUSR1 handling, exit-code contract
  status.py                    status.json writer + ETA arithmetic
tests/                         plain assert scripts, no pytest
```

## Configuration snapshot

The default [config.yaml](config.yaml) produces a 383M-parameter model, 224M active per token:

| | |
|---|---|
| hidden / intermediate | 768 / 2304 |
| layers / heads / head_dim | 8 / 12 / 64 |
| MLP / attn / IR experts | 32 / 1 / 1 |
| top-k / loops | 2 / 3 |
| IR table: entries × dim | 65536 × 384 |
| IR read: clusters / probed / top-k | 256 / 8 / 32 |
| IR output stage | direct read (`ir_direct_read: true`) |
| evidence encoder | the full dense decoder (`evidence_encoder: true`) |
| MTP extra tokens | 2 |
| vocab / context length | 65536 / 4096 |
| token budget | 16B (phase 1: 85%, phase 2: the anneal) |

The 16B-token run predates the IR reshape and the port and was trained at 332M / 173M with an
`8192 × 128` exact-read table. Whether a checkpoint carries the reshaped table, the loop injection,
the port or the direct read is inferred from its state dict, never from the yaml.

## Credit

Borrows heavily from [Gemma4](https://huggingface.co/google/gemma-4-31b-it).

**Papers & research**
- [LoopLM](https://arxiv.org/abs/2510.25741)
- [Nemotron-3-Super technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [Multi-token Prediction](https://arxiv.org/pdf/2404.19737)
- The retrieval and memory literature the evidence port was reviewed against is listed at the end
  of [docs/review_2026-09-18.md](docs/review_2026-09-18.md) (RETRO, Fusion-in-Decoder, Memory
  Layers at Scale, LMLM, and others)

**Datasets** — pretraining
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) / [DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) / [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)
- [Stack-Edu (Common Pile)](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered)
- [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) (gated)
- [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
- [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) (phase 2)

**Datasets** — post-training and evidence
- [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2), [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
  (distractor setting), [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) as the external chunk embedder

**Benchmarks**
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro), and the thirteen-task log-likelihood /
  generative suite in [docs/measurements/benchmark_suite.md](docs/measurements/benchmark_suite.md)

## Design notes

- Token counts may be slightly inflated during training (on the order of tens of tokens per batch).
- The tokenizer is a 65536-token prune of DeepSeek-V4-Pro's, so the corpus fits in `uint16`. It
  lives at [ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536](https://huggingface.co/ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536)
  and `scripts/fetch_tokenizer.py` pulls it — `ckpts/` is gitignored, so a fresh clone has none.
- `pad_token_id == eos_token_id` and id 0 is BOS, which is why the embedding table has no
  `padding_idx` (setting one froze BOS at zero).
- The corpus under `data/prepared/` on the dev box is an outdated local stand-in, not the corpus the
  16B-token run used. Absolute numbers measured against it are not comparable to
  [docs/CONCLUSION.md](docs/CONCLUSION.md); before/after deltas on the same slice are.

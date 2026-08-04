# tiny-moe-llm

An experimental **332M-parameter** language model exploring a **looped, sparsely-routed
mixture-of-experts** architecture on top of a dense Gemma4-style backbone.

A dense decoder feeds a *single* MoE block that is applied `n_loops` times, so depth is recurrence
rather than parameters. 332M total / 173M active per token.

The following techniques were used/implemented:

- **Dense Gemma4-style blocks** — GQA, RoPE, RMSNorm, per-layer embeddings (PLE)
- **Looped MoE** — one MoE block applied for `n_loops` iterations, rerouting tokens each pass
  (LoopLM-style recurrence), with routing conditioned on the loop index so consecutive loops do not
  all pick the same experts
- **Heterogeneous experts** — self-attention, cross-attention, information-retrieval and MLP
  experts share one router, plus always-on shared MLP/attention experts outside the router pool
- **Halt head** — a per-loop `p_halt` gates how much each loop is allowed to change a token, a
  compute-allocation signal rather than a correctness one
- **Correctness head** — a separate `p_correct` asks "is this specific prediction right", which
  comes apart from `p_halt` on confident hallucinations (provisional, see [PLAN.md](PLAN.md))
- **Per-loop CE supervision** — every loop's readout is supervised, not just the last, so an
  early-exit policy has something to exit *to*
- **Stochastic loop depth** — a fraction of steps train at a reduced depth, making the loop count a
  real runtime choice at inference
- **Multi-token prediction (MTP)** — auxiliary heads predict several future tokens per step
- **Document packing** — multiple documents per sequence with block-diagonal causal attention via
  flash-attn varlen
- **Low-precision training** — optional FP8 / NVFP4 via NVIDIA Transformer Engine

There is no identity expert; it was removed in favour of the halt head.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/runbook.md](docs/runbook.md) | **Start here for a real run**: what to run, how to stop it, what is normal, what to do when it is not |
| [docs/architecture.md](docs/architecture.md) | End-to-end model architecture and data flow |
| [docs/moe.md](docs/moe.md) | Looped MoE, routing, expert types, halt head |
| [docs/training.md](docs/training.md) | Training pipeline, data packing, MTP loss, precision, checkpointing, resume |
| [docs/configuration.md](docs/configuration.md) | Full `config.yaml` reference |
| [CLAUDE.md](CLAUDE.md) | Operational map: invariants, gotchas, what lives where |

## Quick start

Every command runs from the repo root — `config.py` opens `config.yaml` by relative path.

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
python scripts/inference.py -c ckpts/training/checkpoint_phase1_final.pt -p "Once upon a time" -n 200
```

## Repository layout

```
config.py / config.yaml        model + training hyperparameters
utils.py                       logger, paths, tokenizer/repo constants, checkpoint save/load
scripts/
  setup.sh                     one-shot box setup: deps, token, tokenizer, preflight
  onstart.sh                   vast.ai onstart hook: clone, setup, launch the supervisor
  run_training.py              supervisor: phase 1 -> phase 2, restarts through preemptions
  pretrain.py                  the training loop
  prepare_data.py              builds phase1/phase2 .bin/.idx from the Hub source mix
  fetch_tokenizer.py           downloads the pruned 65536-token tokenizer
  inference.py                 greedy/top-k sampling CLI
  prune_vocab.py               one-shot 129280 -> 65536 vocab prune
  eval_calibration.py          ECE / abstention AUROC for the correctness head
modules/model/
  transformer.py               TinyMoETransformer (top-level model)
  gemma4.py                    dense Gemma4-style decoder
  moe.py                       LoopMixtureOfExperts + sparse grouped-GEMM MLP experts
  router.py                    router + load-balancing aux loss
  experts.py                   self/cross-attention + information-retrieval experts
  information_retrieval.py     learned key/value retrieval module
  mtp.py                       multi-token-prediction head + chunked LM-head loss
  attention.py                 document-packed (varlen) causal attention
  modules.py                   factored LM head
  embeddings.py                rotary position embeddings
modules/data/dataset.py        mmap flat-file dataset with document packing
modules/runtime/               unattended-run machinery (no GPU/TE dependency)
  checkpoints.py               naming, retention, latest-VALID resume, resume verification
  hf_sync.py                   background uploader to the Hugging Face Hub
  control.py                   STOP sentinel + SIGTERM/SIGUSR1 handling, exit-code contract
  status.py                    status.json writer + ETA arithmetic
tests/                         plain assert scripts, no pytest
```

## Configuration snapshot

The default [config.yaml](config.yaml) produces a 332M-parameter model, 173M active per token:

| | |
|---|---|
| hidden / intermediate | 768 / 2304 |
| layers / heads / head_dim | 8 / 12 / 64 |
| MLP / attn / IR experts | 32 / 1 / 1 |
| top-k / loops | 2 / 3 |
| MTP extra tokens | 2 |
| vocab / context length | 65536 / 4096 |
| token budget | 29.9B (phase 1: 25.4B, phase 2: 4.5B anneal) |

## Credit

Borrows heavily from [Gemma4](https://huggingface.co/google/gemma-4-31b-it).

**Papers & research**
- [LoopLM](https://arxiv.org/abs/2510.25741)
- [Nemotron-3-Super technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [Multi-token Prediction](https://arxiv.org/pdf/2404.19737)

**Datasets** — pretraining
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) / [DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) / [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)
- [Stack-Edu (Common Pile)](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered)
- [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) (gated)
- [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
- [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) (phase 2)

**Benchmarks**
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)

## Design notes

- Token counts may be slightly inflated during training (on the order of tens of tokens per batch).
- The tokenizer is a 65536-token prune of DeepSeek-V4-Pro's, so the corpus fits in `uint16`. It
  lives at [ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536](https://huggingface.co/ikeafisch4/DeepSeek-V4-Pro-tokenizer-65536)
  and `scripts/fetch_tokenizer.py` pulls it — `ckpts/` is gitignored, so a fresh clone has none.
- `pad_token_id == eos_token_id` and id 0 is BOS, which is why the embedding table has no
  `padding_idx` (setting one froze BOS at zero).

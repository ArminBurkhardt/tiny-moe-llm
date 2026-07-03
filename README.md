# tiny-moe-llm

An experimental ~243M-parameter language model exploring a **looped, sparsely-routed mixture-of-experts** architecture on top of a dense Gemma-style backbone.

The following techniques were used/implemented:

- **Dense Gemma4-style blocks** - GQA, RoPE, RMSNorm, per-layer embeddings (PLE)
- **Looped MoE** - a single MoE block is applied for `n_loops` iterations, rerouting tokens each pass (LoopLM-style recurrence)
- **Heterogeneous experts** - self-attention, cross-attention, information-retrieval, and MLP experts share one router, plus an **identity expert** that lets a token exit routing early
- **Multi-token prediction (MTP)** - auxiliary heads predict several future tokens per step
- **Document packing** - multiple documents per sequence with block-diagonal causal attention via flash-attn varlen.
- **Low-precision training** - optional FP8 / NVFP4 via NVIDIA Transformer Engine

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/architecture.md](docs/architecture.md) | End-to-end model architecture and data flow |
| [docs/moe.md](docs/moe.md) | Looped MoE, routing, expert types, identity skew |
| [docs/training.md](docs/training.md) | Training pipeline, data packing, MTP loss, precision, resume |
| [docs/configuration.md](docs/configuration.md) | Full `config.yaml` reference |

## Quick start

```bash
pip install -r requirements.txt   # feel free to skip TE installation

# provide a data_config.json (see docs/training.md) and a tokenizer under ckpts/
python scripts/pretrain.py
```

Transformer Engine and flash-attn require a CUDA build matched to the GPU - refer to the
[TE installation guide](https://github.com/NVIDIA/TransformerEngine#installation). flash-attn is
optional, without it attention falls back to a slower SDPA path

## Repository layout

```
config.py / config.yaml        model + training hyperparameters
scripts/pretrain.py            pretraining entry point
modules/model/
  transformer.py               TinyMoETransformer (top-level model)
  gemma4.py                    dense Gemma4-style decoder
  moe.py                       LoopMixtureOfExperts + sparse grouped-GEMM MLP experts
  router.py                    router + load-balancing aux loss
  experts.py                   self/cross-attention + information-retrieval experts
  information_retrieval.py     learned key/value retrieval module
  mtp.py                       multi-token-prediction head + loss
  attention.py                 document-packed (varlen) causal attention
  modules.py                   factored LM head
  embeddings.py                rotary position embeddings
modules/data/dataset.py        streaming, document-packing dataset
```

## Configuration snapshot

The default `config.yaml` produces roughly a 243M-parameter model:

| | |
|---|---|
| hidden / intermediate | 512 / 2048 |
| layers / heads / head_dim | 5 / 8 / 64 |
| MLP / attn / IR experts | 36 / 1 / 1 (+ identity) |
| top-k / loops | 2 / 4 |
| MTP extra tokens | 2 |
| context length | 4096 |

## Credit

Borrows heavily from [Gemma4](https://huggingface.co/google/gemma-4-31b-it).

**Papers & research**
- [LoopLM](https://arxiv.org/abs/2510.25741)
- [Nemotron-3-Super technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [Multi-token Prediction](https://arxiv.org/pdf/2404.19737)

**Datasets** - pretraining
- [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/tree/main/data/ultrafineweb_en_v1_4)
- [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)

**Datasets** - reasoning / SFT
- [KIMI-K2.5-550000x](https://huggingface.co/datasets/ianncity/KIMI-K2.5-550000x)
- [Claude-Sonnet-4.6-Reasoning-1100x](https://huggingface.co/datasets/TeichAI/Claude-Sonnet-4.6-Reasoning-1100x)
- [Claude-Opus-4.6-10000x](https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x)
- Fable 5 reasoning traces

**Benchmarks**
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)

## Design notes

- token counts may be slightly inflated during training (in the order of tens of tokens per batch)
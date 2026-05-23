# tiny-moe-llm

tiny-moe-llm is an experimental research project exploring a small, modular mixture-of-experts (MoE) language model architecture

At a high level, the codebase combines:
- Gemma4 style dense transformer layers,
- dynamically routed experts.


## Credit
Borrows heavily from [Gemma4](https://huggingface.co/google/gemma-4-31b-it)

Training guidance and not datasets because they reject individuals who are not affiliated with an institution:
- [Nemotron-3-Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)

Papers and research:
- [LoopLM](https://arxiv.org/abs/2510.25741) 
- [Nemotron-3-Super](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [Multi-token Prediction](https://arxiv.org/pdf/2404.19737)

Datasets:
- [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/tree/main/data/ultrafineweb_en_v1_4)
- [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)


Reasoning Datasets:
- [KIMI-K2.5-550000x](https://huggingface.co/datasets/ianncity/KIMI-K2.5-550000x)
- [Claude-Sonnet-4.6-Reasoning-1100x](https://huggingface.co/datasets/TeichAI/Claude-Sonnet-4.6-Reasoning-1100x)
- [Claude-Opus-4.6-10000x](https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x)


Benchmarks:
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)

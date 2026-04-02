# tiny-moe-llm

tiny-moe-llm is an experimental research project exploring a small, modular mixture-of-experts (MoE) language-model architecture.

At a high level, the codebase combines:
- an encoder-backed latent representation,
- dynamically routed experts,
- and an invertible-style decoding path.

The repository is structured for fast iteration on architecture ideas, including routing behavior, expert lifecycle (addition/pruning), and data pipelines for experimentation.

For implementation details and current project status, see the documentation in `/docs`.




## Credit
Borrows heavily from [Gemma3]

Datasets:
- [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/tree/main/data/ultrafineweb_en_v1_4/CC-MAIN-2020-40)


Reasoning Datasets:
- [KIMI-K2.5-550000x](https://huggingface.co/datasets/ianncity/KIMI-K2.5-550000x)
- [Claude-Sonnet-4.6-Reasoning-1100x](https://huggingface.co/datasets/TeichAI/Claude-Sonnet-4.6-Reasoning-1100x)
- [Claude-Opus-4.6-10000x](https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x)


Benchmarks:
- [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro)

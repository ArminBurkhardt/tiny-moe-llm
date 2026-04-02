# tiny-moe-llm

tiny-moe-llm is an experimental research project exploring a small, modular mixture-of-experts (MoE) language-model architecture.

At a high level, the codebase combines:
- an encoder-backed latent representation,
- dynamically routed experts,
- and an invertible-style decoding path.

The repository is structured for fast iteration on architecture ideas, including routing behavior, expert lifecycle (addition/pruning), and data pipelines for experimentation.

For implementation details and current project status, see the documentation in `/docs`.

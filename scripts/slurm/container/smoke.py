"""Smoke test of the training image, run inside it from the repo root (build.sh and the day 1 checklist call it).

    singularity exec --nv --bind $REPO run2.sif python scripts/slurm/container/smoke.py

Prints the versions and imports the repo's model module, which pulls in transformer_engine and flash_attn at module
scope. Exit 0 when everything imports; a GPU is not required for the imports, but the last check is skipped without one.
"""
import os
import sys

sys.path.insert(0, os.getcwd())


def main():
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
    import transformer_engine
    print("transformer_engine", transformer_engine.__version__)
    import flash_attn
    print("flash_attn", flash_attn.__version__)
    import accelerate
    import transformers
    print("transformers", transformers.__version__, "accelerate", accelerate.__version__)
    import modules.model.transformer  # noqa: F401
    print("repo modules.model: import ok")
    import scripts.pretrain  # noqa: F401
    print("scripts.pretrain: import ok (the TE and flash-attn API check of scripts/setup.sh)")
    if torch.cuda.is_available():
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        print("cuda matmul ok", float((x @ x).float().abs().mean()) > 0)
        import torch.distributed as dist
        print("nccl available", dist.is_nccl_available())


if __name__ == "__main__":
    main()

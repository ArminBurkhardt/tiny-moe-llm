"""Write a from-scratch seed checkpoint for the fact injection runs.

All three arms start from this one file, so they share their initialization exactly and differ by the
data they train on. The model is built from the active yaml (select the micro shape with
``TINY_LLM_CONFIG=config_micro.yaml``) with the evidence port present and no groundedness head and
no loop injection. Arms that never attach evidence leave the port bit-identically inert, so its
tensors only see weight decay; the port exists in all three so the parameter count matches.

The payload is what ``scripts/sft.py -c`` reads from a seed: ``model_state_dict`` (shape, port and IR
mode are inferred from it), ``token_count`` 0 and a phase label. No optimizer state.

Usage:

    TINY_LLM_CONFIG=config_micro.yaml python scripts/init_scratch_seed.py \\
        --out ckpts/inject/seed_micro.pt --seed 0

Args:
    --out: where to write the seed, relative to the repo root unless absolute.
    --seed: the torch seed of the initialization.
    --overwrite: replace an existing file; without it an existing seed is never touched, since it is
        what makes a run reproducible.
    --device: where the model is built; transformer_engine modules allocate on CUDA.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import ModelConfig
from modules.model.transformer import TinyMoETransformer
from utils import BASE_DIR, BF16, logger


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)
    if os.path.exists(out) and not args.overwrite:
        raise SystemExit(f"{out} already exists. Pass --overwrite to replace it; an existing seed is "
                         f"what makes the arms of an earlier run comparable.")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    params = {**ModelConfig.Params, "evidence_port": True, "groundedness_head": False,
              "loop_inject": False}
    torch.manual_seed(args.seed)
    model = TinyMoETransformer(**params).to(args.device).to(BF16)
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)

    assert model.moe.shared_evidence is not None, "the port was not built"
    assert float(model.moe.shared_evidence.attn.o_proj.weight.float().abs().max()) == 0.0, \
        "the reader's output projection must start at zero"
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for n, p in model.named_parameters() if "embed_tokens" in n)
    logger.info(f"seed {args.seed}: {total / 1e6:.1f}M parameters ({embedding / 1e6:.1f}M in the token "
                f"table), {len(state)} tensors")

    # written beside the target and renamed: a kill mid-write must not leave a truncated seed
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        torch.save({"model_state_dict": state, "token_count": 0, "phase": "scratch_init",
                    "model_params": params, "init_seed": args.seed}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()

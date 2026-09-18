"""Add the evidence port to a checkpoint and keep everything else untouched.

The port is two halves that arrive together. The **reader** is an always-on cross attention over
the retrieved evidence tokens, seeded into the loop's accumulator beside ``shared_mlp``/
``shared_attn``; the **selector** is three tensors inside the IR module that let the learned table
read an external chunk store under the same softmax it reads itself.

**The reader's ``o_proj`` is zero, so the migrated checkpoint scores identically to its source --
with a corpus attached or without one.** That is a stronger guarantee than the usual zero-init
pattern: structural absence already covers the no-corpus case, and the zero covers the case that a
corpus IS attached on step 0, which is what makes the replay fraction of the finetune genuinely
protect the trunk instead of training the port. ``tests/test_evidence_port.py`` asserts both
bit-identically rather than to a tolerance.

The **selector is not** zero, deliberately, and this is the one place the migration is not neutral.
Its adapters are square orthogonal rotations (``ir_dim`` was widened to the embedder's native width
for exactly this), and its scores join the table's under one softmax -- which is the whole point,
because two independently normalized reads summed have no notion of which store won and therefore
no split to measure. A zero there would make the external half contribute a zero vector, teaching
"ignore evidence" first. The cost is bounded and known: zeroing the entire IR read is worth 0.0002
nats on this trunk (``docs/measurements/ir_scale_fix.md``), so diluting it cannot cost more than
that, and it only applies when a corpus is attached at all.

Which tensors are FRESH is a real distinction here, because the port's new tensors share a module
with the IR key/value table -- which carries a full sharpening run and would be wrecked by a
from-scratch rate. ``moe.is_fresh_loop_param`` matches the adapters and the source scale by name for
that reason, and not the ``ir_module`` subtree.

Optimizer state is dropped: AdamW's moments are indexed by param-group position and this adds
tensors, so they cannot be paired back up. The output is a finetune seed, like every other migration
here. Run from the repo root:

    python scripts/migrate_evidence_port.py -c ckpts/ir_c/checkpoint_ir_final.pt
"""
import os
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch

from modules.model.moe import is_fresh_loop_param
from modules.model.transformer import TinyMoETransformer
from config import ModelConfig
from utils import BASE_DIR, BF16, logger, model_params_for_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="the checkpoint to add the port to; it becomes the run's control")
    parser.add_argument("--output", "-o", default=None,
                        help="defaults to <checkpoint stem>_evidence.pt next to the source")
    parser.add_argument("--device", default="cpu",
                        help="nothing here needs a GPU -- the tensors are written by init")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output instead of refusing")
    args = parser.parse_args()

    src = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(BASE_DIR, args.checkpoint)
    out_path = args.output or f"{os.path.splitext(src)[0]}_evidence.pt"
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(
            f"{out_path} already exists. Pass -o to write a new seed alongside it, or --force to "
            f"replace it -- an existing seed is what makes an earlier run reproducible."
        )

    payload = torch.load(src, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)

    # the source decides its own shape, IR table included; this only flips the port on, so the run
    # differs from its control in exactly the port's tensors
    params = model_params_for_state_dict(state, ModelConfig.Params)
    if params.get("evidence_port"):
        raise SystemExit(
            f"{os.path.basename(src)} already carries the evidence port -- nothing to migrate."
        )
    params["evidence_port"] = True

    model = TinyMoETransformer(**params).to(args.device).to(BF16)
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)

    fresh = model.state_dict()
    added = sorted(set(fresh) - set(state))
    missing = sorted(set(state) - set(fresh))
    if missing:
        raise SystemExit(
            f"the source carries {len(missing)} tensors this model has no slot for, e.g. "
            f"{missing[:3]} -- refusing rather than dropping them."
        )
    # every added tensor must belong to the port. Anything else absent would load as its random init
    # and train as if it were the source's, which is the silent failure strict loading prevents.
    stray = [k for k in added if not (
        "shared_evidence." in k or ".key_adapter." in k or ".value_adapter." in k
        or k.endswith("log_memory_scale")
    )]
    if stray:
        raise SystemExit(f"these added tensors are not part of the port: {stray}")
    if not added:
        raise SystemExit("nothing was added -- the port did not get built")

    model.load_state_dict(state, strict=False)  # strict=False only for the port tensors, above
    with torch.no_grad():
        # re-zeroed after the load rather than trusted from __init__: load_state_dict with
        # strict=False leaves absent tensors at their init, and "the init happens to be zero" is a
        # fact about another file. The neutrality guarantee is asserted here, where it is claimed.
        model.moe.shared_evidence.attn.o_proj.weight.zero_()
    model.eval()

    n_fresh = sum(p.numel() for name, p in model.named_parameters() if is_fresh_loop_param(name))
    reader = sum(p.numel() for name, p in model.named_parameters() if "shared_evidence." in name)

    payload["model_state_dict"] = model.state_dict()
    for key in ("optimizer_state_dict", "scheduler_state_dict"):
        payload.pop(key, None)
    payload["evidence_port_migration"] = {
        "source": os.path.basename(src),
        "added": added,
        "reader_params": int(reader),
        "fresh_params": int(n_fresh),
    }

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)

    logger.info(f"wrote {out_path}")
    print(f"\n=== evidence port: {os.path.basename(src)} -> {os.path.basename(out_path)} ===")
    print(f"  added:          {len(added)} tensors, {reader / 1e6:.2f}M in the reader")
    print(f"  neutrality:     reader o_proj is zero -- this checkpoint scores exactly as its source,")
    print(f"                  with a corpus attached or without one")
    print(f"  NOT neutral:    the selector's adapters are orthogonal, not zero (see the docstring);")
    print(f"                  bounded by the 0.0002 nats the whole IR read is worth on this trunk")
    print(f"  fresh LR group: {n_fresh / 1e6:.2f}M parameters, and NOT the IR table")
    print(f"  optimizer/scheduler state dropped -- finetune seed, not a resume point")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()

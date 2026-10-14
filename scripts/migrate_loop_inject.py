"""Add the per-loop input injection to a checkpoint, at zero, and keep everything else untouched.

Every loop currently reads only the residual stream, which by loop 3 is the decoder's output plus
two of the block's own updates. The diagnostics say the later loops behave like that is a problem:
consecutive loops repeat 60% of their expert selections, ``cos(q2, q3) = 0.977``, and each loop's
update points largely where the previous one pointed. The published looped transformers all hand
the block's input back to every iteration in some form; this model is the one that does not.

So every loop's router and experts read ``hidden_states + inject(e)``, where ``e`` is what the
dense decoder handed the block. The residual update is unchanged -- ``hidden_states + loop_scale[k]
* delta`` still adds to the stream ``lm_head`` reads -- so the injection reaches a readout only
through what the experts compute from it.

**``inject`` is zero-init, so the migrated checkpoint scores identically to its source.** Same
neutrality pattern as ``g_proj`` and ``loop_router_bias``, and the same warning applies: a
zero-init tensor that never leaves zero is a run that measured nothing, so ``scripts/sft.py`` logs
``|inject|rms`` at every log step. It goes into the fresh-parameter LR group
(``is_fresh_loop_param``) because at the trunk's 1e-5 against an otherwise converged model it would
not move.

Unlike ``scripts/migrate_ir_reshape.py`` this rebuilds nothing, so the optimizer state *could* in
principle be carried -- but AdamW's moments are indexed by param-group position and this adds a
tensor, so they cannot be paired back up. The output is a finetune seed, like the other two
migrations.

Seed it from the same migrated checkpoint the no-injection run started from, not from that run's
output: the two then differ in exactly this tensor at matched tokens, and the existing run is the
control for free. Run from the repo root:

    python scripts/migrate_loop_inject.py -c ckpts/trained/checkpoint_phase2_final_phase0_irc.pt
"""
import os
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch

from modules.model.transformer import TinyMoETransformer
from config import ModelConfig
from utils import BASE_DIR, BF16, logger, model_params_for_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="the checkpoint to add the injection to; it becomes the arm's control")
    parser.add_argument("--output", "-o", default=None,
                        help="defaults to <checkpoint stem>_inject.pt next to the source")
    parser.add_argument("--device", default="cpu",
                        help="nothing here needs a GPU -- the tensor is written as zeros")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output instead of refusing")
    args = parser.parse_args()

    src = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(BASE_DIR, args.checkpoint)
    out_path = args.output or f"{os.path.splitext(src)[0]}_inject.pt"
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(
            f"{out_path} already exists. Pass -o to write a new seed alongside it, or --force to "
            f"replace it -- an existing seed is what makes an earlier run reproducible."
        )

    payload = torch.load(src, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)

    # the source decides its own shape (IR table included); this only flips the injection on, so
    # the arm differs from its control in exactly one tensor
    params = model_params_for_state_dict(state, ModelConfig.Params)
    if params.get("loop_inject"):
        raise SystemExit(
            f"{os.path.basename(src)} already carries moe.inject.weight -- nothing to migrate."
        )
    params["loop_inject"] = True

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
    if added != ["moe.inject.weight"]:
        # anything else absent would be loaded as its random init and trained as if it were the
        # source's, which is exactly the silent failure strict loading exists to prevent
        raise SystemExit(f"expected to add only moe.inject.weight, got {added}")

    model.load_state_dict(state, strict=False)  # strict=False only for moe.inject.weight, above
    with torch.no_grad():
        model.moe.inject.weight.zero_()
    model.eval()

    payload["model_state_dict"] = model.state_dict()
    # a new tensor shifts every later param group's index, so the saved moments no longer describe
    # the parameters they would be paired with. Same contract as the other two migrations.
    for key in ("optimizer_state_dict", "scheduler_state_dict"):
        payload.pop(key, None)
    payload["loop_inject_migration"] = {
        "source": os.path.basename(src),
        "hidden_size": int(model.moe.inject.weight.shape[0]),
        "params": int(model.moe.inject.weight.numel()),
    }

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)

    logger.info(f"wrote {out_path}")
    print(f"\n=== loop injection: {os.path.basename(src)} -> {os.path.basename(out_path)} ===")
    print(f"  added:          moe.inject.weight {tuple(model.moe.inject.weight.shape)}, "
          f"{model.moe.inject.weight.numel() / 1e6:.2f}M parameters, zero")
    print(f"  neutrality:     zero weight, bias free -- this checkpoint scores exactly as its source")
    print(f"  trains at:      the fresh-parameter LR (is_fresh_loop_param), logged as |inject|rms")
    print(f"  optimizer/scheduler state dropped -- finetune seed, not a resume point")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()

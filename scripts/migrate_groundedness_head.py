"""Add the groundedness readout to a checkpoint that already carries the evidence port.

The head is three tensors -- an RMSNorm gain and a zero-init ``[hidden_size, 1]`` projection with
its bias -- reading the evidence reader's own output. **Nothing in ``forward()`` calls it**, so the
migrated checkpoint scores identically to its source by construction rather than by a zero: the
logits do not pass through this module at all, with a corpus attached or without one. The zero
still matters for the first training steps, where it makes the readout start at p = 0.5 for every
row instead of at an arbitrary opinion.

Its own migration rather than a flag on ``migrate_evidence_port.py``: the port's seed already
exists and is the run's control, and re-running that script on a checkpoint that carries the port
is refused (correctly -- it would rebuild the selector's adapters and invalidate the comparison).

What the head is FOR is the half of the retrieval gate the mass split cannot answer. The external
mass says how much of the read went to the corpus; it cannot say whether what came back grounds an
answer, and SQuAD v2's adversarial unanswerables are built precisely so that a relevant-looking
passage is present and the answer still is not in it. The label is external -- "a gold chunk is
present AND the row is answerable", both from the corpus's own ``.evgold``/``.ans`` sidecars --
which is what keeps this from collapsing onto ``p_max`` the way the deleted correctness head did:
that one was trained against ``lm_head``'s own argmax, so reproducing ``p_max`` was the reachable
optimum by construction, and that is what it learned.

The head trains at the from-scratch rate (``moe.is_fresh_loop_param`` matches it), like every other
tensor a migration adds at its init.

Optimizer state is dropped: AdamW's moments are indexed by param-group position and this adds
tensors, so they cannot be paired back up. The output is a finetune seed, like every other
migration here. Run from the repo root:

    python scripts/migrate_groundedness_head.py -c ckpts/repair/checkpoint_repair_final_irrandom_evidence.pt
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
                        help="the checkpoint to add the head to; it must already carry the port")
    parser.add_argument("--output", "-o", default=None,
                        help="defaults to <checkpoint stem>_grounded.pt next to the source")
    parser.add_argument("--device", default="cpu",
                        help="nothing here needs a GPU -- the tensors are written by init")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output instead of refusing")
    args = parser.parse_args()

    src = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(BASE_DIR, args.checkpoint)
    out_path = args.output or f"{os.path.splitext(src)[0]}_grounded.pt"
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(
            f"{out_path} already exists. Pass -o to write a new seed alongside it, or --force to "
            f"replace it -- an existing seed is what makes an earlier run reproducible."
        )

    payload = torch.load(src, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)

    # the source decides its own shape; this only flips the head on, so the run differs from its
    # control in exactly the head's three tensors
    params = model_params_for_state_dict(state, ModelConfig.Params)
    if not params.get("evidence_port"):
        raise SystemExit(
            f"{os.path.basename(src)} has no evidence port -- the head reads the reader's output, "
            f"so there would be nothing for it to read. Run scripts/migrate_evidence_port.py first."
        )
    if params.get("groundedness_head"):
        raise SystemExit(
            f"{os.path.basename(src)} already carries the groundedness head -- nothing to migrate."
        )
    params["groundedness_head"] = True

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
    # every added tensor must belong to the head. Anything else absent would load as its random
    # init and train as if it were the source's, which is the silent failure strict loading prevents
    stray = [k for k in added if not k.startswith("groundedness_head.")]
    if stray:
        raise SystemExit(f"these added tensors are not part of the head: {stray}")
    if not added:
        raise SystemExit("nothing was added -- the head did not get built")

    model.load_state_dict(state, strict=False)  # strict=False only for the head tensors, above
    with torch.no_grad():
        # re-zeroed after the load rather than trusted from __init__, same as the port migration
        # does for the reader: load_state_dict with strict=False leaves absent tensors at their
        # init, and "the init happens to be zero" is a fact about another file
        model.groundedness_head.out_proj.weight.zero_()
        model.groundedness_head.out_proj.bias.zero_()
    model.eval()

    n_fresh = sum(p.numel() for name, p in model.named_parameters() if is_fresh_loop_param(name))
    head = sum(p.numel() for name, p in model.named_parameters()
               if name.startswith("groundedness_head."))

    payload["model_state_dict"] = model.state_dict()
    for key in ("optimizer_state_dict", "scheduler_state_dict"):
        payload.pop(key, None)
    payload["groundedness_head_migration"] = {
        "source": os.path.basename(src),
        "added": added,
        "head_params": int(head),
        "fresh_params": int(n_fresh),
    }

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)

    logger.info(f"wrote {out_path}")
    print(f"\n=== groundedness head: {os.path.basename(src)} -> {os.path.basename(out_path)} ===")
    print(f"  added:          {len(added)} tensors, {head / 1e3:.1f}K parameters")
    print(f"  neutrality:     nothing in forward() reads the head, so this checkpoint scores")
    print(f"                  exactly as its source; the zero-init output starts it at p = 0.5")
    print(f"  fresh LR group: {n_fresh / 1e6:.2f}M parameters (is_fresh_loop_param)")
    print(f"  needs:          a corpus with .evgold and .ans, and groundedness_weight > 0")
    print(f"  optimizer/scheduler state dropped -- finetune seed, not a resume point")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()

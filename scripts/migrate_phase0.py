"""Migrate a pre-Phase-0 checkpoint: fold the halt gate into ``loop_scale``, strip both heads.

Every checkpoint written before the halt and correctness heads were deleted carries
``moe.halt_proj.*`` and ``correct_proj.*``, and -- much more importantly -- a ``loop_scale`` that
was learned *underneath* a per-token gate:

    h <- h + (1 - p_halt) * loop_scale[k] * delta

``p_halt`` saturated near 1, so ``loop_scale`` grew to compensate (the 16B-token run ended at
``[1.73, 1.81, 1.32]``). Dropping the gate naively multiplies every loop's delta by ~1/(1 - p_halt)
and the checkpoint stops working. The fix is to fold the gate's measured per-loop mean into the
gain that survives:

    loop_scale_new[k] = loop_scale_old[k] * mean_k(1 - p_halt)

**Measure it, never assume it.** The training log only ever recorded ``p_halt`` averaged over all
loops, and the per-loop spread is large: on the real SFT checkpoint the mean gate is
``[0.290, 0.134, 0.084]``, not the flat ~0.22 that scalar implies. Folding a flat value would leave
loop 1 roughly 25% too weak and loop 3 roughly 2.6x too strong. So this script measures the gate on
real data by default, and ``--gate`` exists only for reproducing a specific migration.

**How the measurement stays exact without keeping the old code alive.** The post-Phase-0
``forward_step`` returns ``h_out = h + loop_scale * delta`` and dropout is off in eval, so
``h_out - h`` is exactly the ungated update. The old dynamics are therefore reproduced by applying
the gate outside the module:

    p_halt = sigmoid(halt_proj(h))          # halt_proj read out of the old state dict
    h <- h + (1 - p_halt) * (h_out - h)

No legacy branch survives in ``modules/``, and the numbers are the real ones rather than a replay
of a simplified model.

Run from the repo root:

    python scripts/migrate_phase0.py -c ckpts/trained/checkpoint_sft_final.pt
    python scripts/migrate_phase0.py -c ckpts/trained/checkpoint_phase2_final.pt --split phase2

Then check the result with ``scripts/eval_calibration.py`` on the same slice, before and after --
per-loop CE and top-1 must be unchanged within noise. That is the whole acceptance criterion for
this migration: it is meant to change nothing behaviourally, only to remove dead machinery.
"""
import os
import sys
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
from transformers import AutoTokenizer

from modules.model.transformer import TinyMoETransformer
from modules.model.attention import cu_seqlens_from_doc_ids
from modules.data.dataset import Dataset
from config import ModelConfig, TrainingConfig
from utils import BASE_DIR, BF16, TOKENIZER_DIR, logger

# state-dict entries that no longer have a home in the model
DROPPED_PREFIXES = ("moe.halt_proj.", "correct_proj.")


def load_legacy(path: str, device: str):
    """Build the post-Phase-0 model from a pre-Phase-0 checkpoint.

    Returns:
        ``(model, payload, halt_weight, halt_bias)``. ``payload`` is the full checkpoint dict so
        the caller can rewrite it in place; the halt tensors are returned separately because the
        model has nowhere to put them and the measurement below needs them.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload)

    missing = [k for k in ("moe.halt_proj.weight", "moe.halt_proj.bias") if k not in state]
    if missing:
        raise SystemExit(
            f"{os.path.basename(path)} has no halt head ({', '.join(missing)}) -- it is already "
            "migrated, or was never a pre-Phase-0 checkpoint. Nothing to do."
        )

    halt_weight = state["moe.halt_proj.weight"].to(device).float()   # [1, H]
    halt_bias = state["moe.halt_proj.bias"].to(device).float()       # [1]
    kept = {k: v for k, v in state.items() if not k.startswith(DROPPED_PREFIXES)}
    dropped = sorted(set(state) - set(kept))

    model = TinyMoETransformer(**ModelConfig.Params).to(device).to(BF16)
    model.set_checkpointing(False, False)
    model.delayed_mtp_loss(True)
    # strict: everything the current architecture wants must be present. The only tolerated
    # difference is the two deleted heads, which were filtered out above by name -- so an
    # unexpectedly missing tensor still fails loudly here rather than being silently random.
    model.load_state_dict(kept, strict=True)
    model.eval()
    logger.info(f"dropped {len(dropped)} head tensors: {', '.join(dropped)}")
    return model, payload, halt_weight, halt_bias


@torch.no_grad()
def measure_gate(model, halt_weight, halt_bias, dataset, pad_token_id, max_batches, device):
    """Per-loop mean of ``(1 - p_halt)`` over non-pad tokens, under the ORIGINAL gated dynamics.

    Reconstructs the pre-Phase-0 forward exactly (see this module's docstring) by re-gating the
    ungated per-loop delta, so the model that produces the statistic is the model the checkpoint
    was actually trained as -- not the ungated one, whose hidden states would drift further from
    the truth at every loop.

    Args:
        model: the post-Phase-0 model, already carrying the checkpoint's weights.
        halt_weight: ``[1, hidden]`` from the old state dict.
        halt_bias: ``[1]`` from the old state dict.
        dataset: an iterable of packed training batches.
        pad_token_id: excluded from the mean, matching how the trainer counted tokens.
        max_batches: cap on the pass.
        device: where to run.

    Returns:
        ``(gate_mean, gate_std, n_tokens)`` -- lists of length ``n_loops`` plus the token count.
    """
    moe = model.moe
    n_loops = moe.n_loops
    gate_sum = torch.zeros(n_loops, dtype=torch.float64, device=device)
    gate_sq = torch.zeros(n_loops, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)

    for i, batch in enumerate(dataset):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        document_ids = batch["document_ids"].to(device)
        cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
        valid = (input_ids != pad_token_id).float()

        hidden = model.gemma_decoder(input_ids, cu_seqlens, max_seqlen).last_hidden_state
        other = model._moe_ple(input_ids)
        position_embeddings = moe.rotary_emb(hidden, seq_len=hidden.shape[1])
        moe.expert_tracker.begin_forward(n_loops)

        for loop in range(n_loops):
            # p_halt came from the state entering the loop, before that loop's update
            p_halt = torch.sigmoid(
                torch.nn.functional.linear(hidden.float(), halt_weight, halt_bias)
            )                                                  # [B, S, 1]
            gate = 1.0 - p_halt
            updated, _ = moe.forward_step(
                hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, other=other,
                position_embeddings=position_embeddings, loop_idx=loop,
            )
            # (h_out - h) is exactly loop_scale * delta, so re-gating it here reproduces the
            # original update without any legacy code path in the model
            hidden = hidden + (gate.to(hidden.dtype) * (updated - hidden))

            g = gate.squeeze(-1).float()
            gate_sum[loop] += (g * valid).sum().double()
            gate_sq[loop] += (g * g * valid).sum().double()
        count += valid.sum().double()

    if count.item() == 0:
        raise SystemExit("measured zero non-pad tokens -- check --data-dir/--split/--max-batches")
    mean = gate_sum / count
    std = (gate_sq / count - mean * mean).clamp(min=0).sqrt()
    return mean.cpu().tolist(), std.cpu().tolist(), int(count.item())


def main():
    parser = argparse.ArgumentParser(description="fold the halt gate into loop_scale and strip both heads")
    parser.add_argument("--checkpoint", "-c", required=True, help="pre-Phase-0 checkpoint to migrate")
    parser.add_argument("--out", "-o", default=None,
                        help="output path (default: <input>_phase0.pt, never overwrites the input)")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, TrainingConfig.data_dir))
    parser.add_argument("--split", default=TrainingConfig.phase,
                        help="which {split}.bin/.idx to measure the gate on")
    parser.add_argument("--start-doc-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=40,
                        help="batches to measure over; the gate is a mean of a low-variance "
                             "quantity, so a few hundred thousand tokens is plenty")
    parser.add_argument("--gate", default=None,
                        help="comma-separated per-loop (1 - p_halt) to fold INSTEAD of measuring "
                             "(e.g. '0.290,0.134,0.084'). Only for reproducing a past migration")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_path = args.out or (os.path.splitext(args.checkpoint)[0] + "_phase0.pt")
    if os.path.abspath(out_path) == os.path.abspath(args.checkpoint):
        raise SystemExit("refusing to overwrite the input checkpoint -- pass a different --out")

    logger.info(f"Loading {args.checkpoint}")
    model, payload, halt_weight, halt_bias = load_legacy(args.checkpoint, args.device)
    old_loop_scale = model.moe.loop_scale.detach().float().cpu().tolist()

    if args.gate is not None:
        gate = [float(x) for x in args.gate.split(",")]
        if len(gate) != len(old_loop_scale):
            raise SystemExit(f"--gate needs {len(old_loop_scale)} values, got {len(gate)}")
        std, n_tokens = [float("nan")] * len(gate), 0
        logger.warning(f"using the supplied gate {gate} instead of measuring it")
    else:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        dataset = Dataset(
            data_dir=args.data_dir, tokenizer=tokenizer, batch_size=args.batch_size,
            max_length=ModelConfig.Params["max_seq_len"], split=args.split,
            num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
            start_doc_idx=args.start_doc_idx,
        )
        logger.info(f"Measuring the halt gate over <= {args.max_batches} batches of {args.split}")
        gate, std, n_tokens = measure_gate(
            model, halt_weight, halt_bias, dataset, tokenizer.pad_token_id,
            args.max_batches, args.device,
        )

    new_loop_scale = [s * g for s, g in zip(old_loop_scale, gate)]
    with torch.no_grad():
        model.moe.loop_scale.copy_(
            torch.tensor(new_loop_scale, dtype=model.moe.loop_scale.dtype, device=args.device)
        )

    payload["model_state_dict"] = model.state_dict()
    # the optimizer state is indexed by param-group POSITION, and two tensors just left the model,
    # so every moment after the removed heads would be paired with the wrong parameter on load.
    # A migrated checkpoint is a finetune starting point, not a resume point -- drop it and say so.
    for key in ("optimizer_state_dict", "scheduler_state_dict"):
        payload.pop(key, None)
    payload["phase0_migration"] = {
        "source": os.path.basename(args.checkpoint),
        "gate": gate,
        "gate_std": std,
        "gate_tokens": n_tokens,
        "gate_split": None if args.gate is not None else args.split,
        "loop_scale_before": old_loop_scale,
        "loop_scale_after": new_loop_scale,
    }

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)

    def fmt(values):
        return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"

    print(f"\n=== Phase 0 migration: {os.path.basename(args.checkpoint)} ===")
    if n_tokens:
        print(f"  measured on {n_tokens:,} non-pad tokens of {args.split}")
        print(f"  mean (1 - p_halt): {fmt(gate)}   std {fmt(std)}")
    else:
        print(f"  gate supplied on the command line: {fmt(gate)}")
    print(f"  loop_scale before:  {fmt(old_loop_scale)}")
    print(f"  loop_scale after:   {fmt(new_loop_scale)}")
    print(f"  optimizer/scheduler state dropped -- this is a finetune seed, not a resume point")
    print(f"  wrote {out_path}")
    print("\n  Now run scripts/eval_calibration.py on this checkpoint with the SAME "
          "--start-doc-idx/--max-batches/--batch-size you ran on the original;")
    print("  per-loop CE and top-1 must match within noise, or the fold constant is wrong.")


if __name__ == "__main__":
    main()

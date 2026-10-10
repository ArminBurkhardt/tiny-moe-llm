"""Does the per-loop router bias do the job it exists for: making consecutive loops route apart?

Read-only. The bias is zero-init and trained, so it has a norm; a norm is not evidence that it
changed any routing decision. Two things are measured, live and with the bias zeroed at eval:

  * per-loop CE -- what the mechanism is worth to the readout;
  * consecutive-loop expert overlap -- the share of a token's top-k that loop k repeats from
    loop k-1, which is the quantity the mechanism was added to lower.

If zeroing it moves neither, it is not doing that job, and any later design that copies its shape
(a loop-conditioned bias on some other pathway) cannot assume the shape works.
"""
import os
import sys
import argparse

sys.path.insert(0, os.getcwd())

import torch
from transformers import AutoTokenizer

from modules.data.dataset import Dataset
from modules.model.attention import cu_seqlens_from_doc_ids
from config import ModelConfig
from utils import TOKENIZER_DIR
from scripts.eval_calibration import chunked_eval, load_model


class RouteTap:
    """record each loop's top-k expert indices by wrapping route(), which returns them already."""

    def __init__(self, moe):
        self.moe = moe
        self.indices = []
        self._orig = moe.route

        def wrapped(hidden_states, temperature=1.0, loop_idx=0):
            scores, idx, aux = self._orig(hidden_states, temperature=temperature, loop_idx=loop_idx)
            self.indices.append(idx.detach())
            return scores, idx, aux

        moe.route = wrapped

    def close(self):
        self.moe.route = self._orig


def overlap_per_transition(indices, valid):
    """mean share of loop k's selected experts that loop k-1 also selected, over valid tokens."""
    out = []
    k = indices[0].shape[-1]
    for loop in range(1, len(indices)):
        prev = indices[loop - 1][:, :-1, :]
        cur = indices[loop][:, :-1, :]
        # [B, S-1, k, k] equality, any() over the previous loop's slots: set overlap, not position
        same = (cur.unsqueeze(-1) == prev.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1) / k
        out.append(float((same * valid).sum() / valid.sum().clamp_min(1)))
    return out


@torch.no_grad()
def measure(model, dataset, device, max_batches, max_loops):
    moe = model.moe
    ce_sum = [0.0] * max_loops
    ce_n = [0] * max_loops
    overlap = [0.0] * max(max_loops - 1, 1)
    batches = 0

    for batch in dataset:
        if batches >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        doc = batch["document_ids"].to(device)
        labels = batch["labels"].to(device)
        cu, max_seqlen = cu_seqlens_from_doc_ids(doc)
        main_labels = labels[:, 1:].contiguous().view(-1)
        valid = (labels[:, 1:] != -100).float()

        tap = RouteTap(moe)
        hidden = model(input_ids=ids, cu_seqlens=cu, max_seqlen=max_seqlen,
                       return_hidden=True, n_loops=max_loops, skip_mtp=True)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        tap.close()

        for loop in range(hidden.size(0)):
            h = hidden[loop, :, :-1, :].contiguous().view(-1, hidden.size(-1))
            s, n, _ = chunked_eval(model.lm_head, h, main_labels)
            ce_sum[loop] += s
            ce_n[loop] += n

        for i, v in enumerate(overlap_per_transition(tap.indices, valid)):
            overlap[i] += v
        batches += 1

    ce = [ce_sum[i] / max(ce_n[i], 1) for i in range(max_loops)]
    return ce, [v / max(batches, 1) for v in overlap], batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--max-loops", type=int, default=ModelConfig.Params["n_loops"])
    ap.add_argument("--start-doc-idx", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model, _ = load_model(args.checkpoint, args.device)

    def fresh_dataset():
        return Dataset(data_dir=args.data_dir, tokenizer=tok, batch_size=args.batch_size,
                       max_length=ModelConfig.Params["max_seq_len"], split=args.phase,
                       num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
                       start_doc_idx=args.start_doc_idx)

    bias = model.moe.loop_router_bias.weight
    print(f"\n  loop_router_bias: rms={bias.float().pow(2).mean().sqrt():.6f} "
          f"max={bias.float().abs().max():.6f}")

    ce_live, ov_live, n = measure(model, fresh_dataset(), args.device, args.max_batches, args.max_loops)

    # in-place on a leaf that requires grad is refused, and no_grad only covers the forward
    saved = bias.detach().clone()
    with torch.no_grad():
        bias.zero_()
    ce_dead, ov_dead, _ = measure(model, fresh_dataset(), args.device, args.max_batches, args.max_loops)
    with torch.no_grad():
        bias.copy_(saved)

    print(f"\n=== loop router bias ablation ({n} batches) ===")
    print(f"  {'loop':<8}{'CE live':>12}{'CE bias=0':>12}{'dCE':>10}")
    for i in range(args.max_loops):
        print(f"  {i+1:<8}{ce_live[i]:>12.4f}{ce_dead[i]:>12.4f}{ce_dead[i]-ce_live[i]:>10.4f}")
    print(f"\n  {'transition':<14}{'overlap live':>14}{'overlap bias=0':>16}{'delta':>10}")
    for i in range(len(ov_live)):
        print(f"  loop {i+1} -> {i+2:<5}{ov_live[i]:>14.4f}{ov_dead[i]:>16.4f}"
              f"{ov_dead[i]-ov_live[i]:>10.4f}")
    print("\n  overlap = share of a token's top-k that the previous loop also selected;"
          "\n  the bias exists to push this DOWN, so zeroing it should push it UP")


if __name__ == "__main__":
    main()

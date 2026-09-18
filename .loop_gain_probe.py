"""Per-loop CE gains and update repetition, for two checkpoints at matched tokens.

The question the injection was added to answer is whether later loops stop repeating the earlier
ones. That is three numbers, and this reads all three on the same slice for a checkpoint and its
control:

  * **per-loop CE and the loop-to-loop gain** -- `CE(k-1) - CE(k)`. Loop 3's gain is the headline:
    the control's was 0.012 nats, and a mechanism that gives later loops something new to do has to
    move it.
  * **cos(d_k, d_k-1)** -- how far each loop's update points where the previous one pointed. Down is
    the direction that means "stopped repeating".
  * **expert overlap** -- the share of a token's top-k that the previous loop also selected, the
    same quantity the loop bias moves.

Read-only, and deliberately not a rewrite of the Stage 0 script: it reuses that script's probe so
the numbers are the same quantities, measured the same way, on the same slice.
"""
import os
import sys
import argparse

sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modules.data.dataset import Dataset
from modules.model.attention import cu_seqlens_from_doc_ids
from config import ModelConfig
from utils import TOKENIZER_DIR
from scripts.eval_calibration import chunked_eval, load_model


class Tap:
    """per-loop router selections and pre-norm hidden states, via the paths the model already runs."""

    def __init__(self, model):
        self.moe = model.moe
        self.indices = []
        self.hidden = []
        self._route = self.moe.route
        self._step = self.moe.forward_step

        def routed(hidden_states, temperature=1.0, loop_idx=0):
            scores, idx, aux = self._route(hidden_states, temperature=temperature, loop_idx=loop_idx)
            self.indices.append(idx.detach())
            return scores, idx, aux

        def stepped(hidden_states, *a, **k):
            out = self._step(hidden_states, *a, **k)
            # [0] is the UPDATED stream; the input is what the previous loop returned, so storing
            # the output of every loop plus the block's input gives every delta
            if not self.hidden:
                self.hidden.append(hidden_states.detach())
            self.hidden.append(out[0].detach())
            return out

        self.moe.route = routed
        self.moe.forward_step = stepped

    def close(self):
        self.moe.route = self._route
        self.moe.forward_step = self._step


@torch.no_grad()
def measure(model, dataset, device, max_batches, n_loops):
    ce_sum = [0.0] * n_loops
    ce_n = [0] * n_loops
    delta_cos = [0.0] * n_loops
    overlap = [0.0] * n_loops
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

        tap = Tap(model)
        hidden = model(input_ids=ids, cu_seqlens=cu, max_seqlen=max_seqlen,
                       return_hidden=True, n_loops=n_loops, skip_mtp=True)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        tap.close()

        for loop in range(hidden.size(0)):
            h = hidden[loop, :, :-1, :].contiguous().view(-1, hidden.size(-1))
            s, n, _ = chunked_eval(model.lm_head, h, main_labels)
            ce_sum[loop] += s
            ce_n[loop] += n

        prev_delta = None
        k = tap.indices[0].shape[-1]
        for loop in range(n_loops):
            delta = (tap.hidden[loop + 1] - tap.hidden[loop])[:, :-1, :].float()
            if prev_delta is not None:
                cos = F.cosine_similarity(delta, prev_delta, dim=-1)
                delta_cos[loop] += float((cos * valid).sum() / valid.sum().clamp_min(1))
                prev = tap.indices[loop - 1][:, :-1, :]
                cur = tap.indices[loop][:, :-1, :]
                same = (cur.unsqueeze(-1) == prev.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1) / k
                overlap[loop] += float((same * valid).sum() / valid.sum().clamp_min(1))
            prev_delta = delta
        batches += 1

    ce = [ce_sum[i] / max(ce_n[i], 1) for i in range(n_loops)]
    return {
        "ce": ce,
        "gain": [None] + [ce[i - 1] - ce[i] for i in range(1, n_loops)],
        "delta_cos": [None] + [delta_cos[i] / max(batches, 1) for i in range(1, n_loops)],
        "overlap": [None] + [overlap[i] / max(batches, 1) for i in range(1, n_loops)],
        "batches": batches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", required=True, help="the arm")
    ap.add_argument("--control", required=True, help="the same run without the change, matched tokens")
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--n-loops", type=int, default=ModelConfig.Params["n_loops"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    def run(path):
        model, _ = load_model(path, args.device)
        ds = Dataset(data_dir=args.data_dir, tokenizer=tok, batch_size=args.batch_size,
                     max_length=ModelConfig.Params["max_seq_len"], split=args.phase,
                     num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"], start_doc_idx=0)
        out = measure(model, ds, args.device, args.max_batches, args.n_loops)
        inject = getattr(model.moe, "inject", None)
        out["inject_rms"] = (
            float(inject.weight.detach().float().pow(2).mean().sqrt()) if inject is not None else None
        )
        del model
        torch.cuda.empty_cache()
        return out

    ctl = run(args.control)
    arm = run(args.checkpoint)

    def fmt(v, nd=4):
        return "  n/a  " if v is None else f"{v:.{nd}f}"

    print(f"\n=== arm vs control, {arm['batches']} batches ===")
    print(f"  control: {os.path.basename(args.control)}")
    print(f"  arm:     {os.path.basename(args.checkpoint)}")
    print(f"  |inject|rms: control {fmt(ctl['inject_rms'], 6)}  arm {fmt(arm['inject_rms'], 6)}")
    print(f"\n  {'loop':<6}{'CE ctl':>10}{'CE arm':>10}{'dCE':>9}"
          f"{'gain ctl':>11}{'gain arm':>11}{'d gain':>9}")
    for i in range(args.n_loops):
        d_gain = (None if arm["gain"][i] is None else arm["gain"][i] - ctl["gain"][i])
        print(f"  {i+1:<6}{ctl['ce'][i]:>10.4f}{arm['ce'][i]:>10.4f}"
              f"{arm['ce'][i]-ctl['ce'][i]:>9.4f}"
              f"{fmt(ctl['gain'][i]):>11}{fmt(arm['gain'][i]):>11}{fmt(d_gain):>9}")
    print(f"\n  {'transition':<14}{'cos ctl':>10}{'cos arm':>10}{'delta':>9}"
          f"{'ovl ctl':>10}{'ovl arm':>10}{'delta':>9}")
    for i in range(1, args.n_loops):
        print(f"  loop {i} -> {i+1:<5}{fmt(ctl['delta_cos'][i]):>10}{fmt(arm['delta_cos'][i]):>10}"
              f"{fmt(arm['delta_cos'][i]-ctl['delta_cos'][i]):>9}"
              f"{fmt(ctl['overlap'][i]):>10}{fmt(arm['overlap'][i]):>10}"
              f"{fmt(arm['overlap'][i]-ctl['overlap'][i]):>9}")
    print("\n  the change is meant to RAISE the last loop's gain and LOWER cos/overlap")


if __name__ == "__main__":
    main()

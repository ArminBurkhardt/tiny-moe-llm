"""Stage 0 diagnostics -- the five measurements docs/plans/NEXT.md's Phase 1 conditions the whole
IR/evidence redesign on. No training, no gradients, one script, one held-out slice.

Everything here is read off the *existing* checkpoint, because every design decision downstream
(reshape the IR table, re-init it, condition the query on the loop index, push depth past 3) is
currently justified by a number nobody has actually measured:

  1. **IR retrieval entropy** -- how much of the 8192-entry table a token's read actually touches.
     ``ln 8192 = 9.010`` nats is the "uniform mixture, i.e. the table addresses nothing" ceiling.
  2. **IR ablation -> Gate G1** -- held-out CE with the retrieved read zeroed, and with it replaced
     by its own batch mean. The second variant is the one that settles the risk NEXT.md names: a
     *constant* read is a bias term the router happens to like, not a retrieval pathway. Zeroing the
     read zeroes the expert's output exactly -- its attention takes K and V from the read alone and
     ``o_proj`` has no bias, so V == 0 makes the whole expert contribute nothing.
  3. **Query drift** -- ``cos(down_proj(h_k), down_proj(h_j))`` across loops. If consecutive loops
     issue the same query, re-executed retrieval cannot make later loops differ, and Phase 5c's
     loop-conditioned query bias is mandatory rather than optional.
  4. **Loop-to-loop dynamics** -- top-1 flip rate, ``||dh||/||h||`` and ``cos(d_k, d_{k-1})``. Says
     whether a loop is idle or churning, in the readout *and* in the residual stream.
  5. **Oracle minimum sufficient depth** -- per token, the earliest loop whose top-1 matches the
     label. Read it as a floor on today's checkpoint, not a depth recommendation: it measures what
     depth buys on plain LM continuation *before* any of the mechanisms that are supposed to give
     later loops something new to do exist (evidence buffer, loop-conditioned query, novelty
     pressure). Most next-token predictions are decided by loop 1 in any looped LM; that is a
     statement about the task, not about the ceiling on depth.

Also printed, for the same reason: **per-loop CE past the trained loop count**. ``--max-loops 6``
runs the recurrence deeper than it was trained. Loop ``k``'s hidden state does not depend on how
many loops follow it, so this is free for loops 1..3 and only costs the extra ones -- and it is the
baseline the Phase 5c depth curriculum has to beat.

Held-out slice: same convention as ``scripts/eval_calibration.py``. An SFT checkpoint's own
``global_offset`` indexes a different corpus, so pass ``--start-doc-idx`` explicitly for those.
"""
import os
import sys
import math
import argparse

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

from modules.model.attention import cu_seqlens_from_doc_ids
from modules.model.experts import InformationRetrievalExpert
from modules.data.dataset import Dataset
from config import ModelConfig, TrainingConfig
from utils import BASE_DIR, logger, TOKENIZER_DIR
from scripts.eval_calibration import chunked_eval, find_latest_checkpoint, load_model

# chunk for the [tokens, num_ir_entries] retrieval softmax. 8192 tokens x 8192 entries in fp32 is
# ~270MB transient, next to nothing on the eval box, and keeps the stats exact rather than binned.
IR_CHUNK = 8192

G1_THRESHOLD = 0.02  # nats of held-out CE the IR read must be worth (NEXT.md Gate G1)


class IRProbe:
    """Forward hooks over the single IR expert: capture its per-loop query/read, and ablate it.

    Everything lives in the script rather than in ``modules/model/`` on purpose -- a diagnostic
    must not be able to change the thing it measures. The hooks read module outputs the model
    already computes; the ablation replaces one of them on the way out.

    Args:
        moe: the ``LoopMixtureOfExperts`` block (``model.moe``).
    """

    def __init__(self, moe):
        self.moe = moe
        ir_indices = [i for i, e in enumerate(moe.experts) if isinstance(e, InformationRetrievalExpert)]
        assert len(ir_indices) == 1, f"expected exactly one IR expert, found {len(ir_indices)}"
        self.expert_index = ir_indices[0]
        self.expert = moe.experts[self.expert_index]

        self.mode = "full"          # full | zero | mean
        self.queries = []           # per loop, [B, S, ir_dim] -- down_proj(norm(h))
        self.reads = []             # per loop, [B, S, ir_dim] -- the IR module's output
        self.router_logits = []     # per loop, [B, S, num_experts] -- pre loop-bias, pre softmax
        self.moe_input = None       # the decoder output the first loop starts from
        self.hidden_all = None      # [loops_run, B, S, H] RAW residual stream (pre self.norm)
        self._handles = []

    def attach(self):
        e = self.expert
        self._handles = [
            e.down_proj.register_forward_hook(lambda m, i, o: self.queries.append(o.detach())),
            e.ir_module.register_forward_hook(lambda m, i, o: self.reads.append(o.detach())),
            e.up_proj.register_forward_hook(self._ablate),
            self.moe.router.register_forward_hook(lambda m, i, o: self.router_logits.append(o.detach())),
            self.moe.register_forward_hook(self._capture_moe),
        ]
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self, mode: str = "full"):
        self.mode = mode
        self.queries, self.reads, self.router_logits = [], [], []
        self.moe_input, self.hidden_all = None, None

    def _ablate(self, module, inputs, output):
        if self.mode == "zero":
            return torch.zeros_like(output)
        if self.mode == "mean":
            # the read, made token-independent: same magnitude and direction the expert already
            # sees on average, with all per-token content removed. contiguous() because the
            # downstream k_proj/v_proj are TE Linears, not view-friendly.
            return output.mean(dim=(0, 1), keepdim=True).expand_as(output).contiguous()
        return output

    def _capture_moe(self, module, inputs, output):
        self.moe_input = inputs[0].detach()
        self.hidden_all = output[2].detach()   # (hidden, aux_loss, hidden_states_all)

    def routed_ir_weight(self, loop_idx: int):
        """Recompute the routing weight the IR slot actually received on this loop.

        Exact, not approximate: ``Router``'s exploration noise is gated on ``self.training`` and the
        model is in eval mode, so replaying ``route()``'s softmax/top-k over the captured logits
        reproduces the selection bit for bit.

        Returns:
            (mean routed weight over all tokens, fraction of tokens that selected the IR slot).
        """
        logits = self.router_logits[loop_idx].float() + self.moe.loop_bias(loop_idx).float()
        scores = F.softmax(logits / self.moe.temperature, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.moe.top_k, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        selected = topk_indices == self.expert_index
        weight = (topk_scores * selected).sum(-1)
        return float(weight.mean()), float(selected.any(-1).float().mean())


def retrieval_stats(ir_module, query: torch.Tensor, top_m: int = 32):
    """Entropy and concentration of the IR softmax, recomputed from the captured query.

    Recomputed rather than captured because ``InformationRetrievalModule`` only returns the weights
    when asked, and asking would mean editing the model to measure it. The arithmetic here is the
    module's own (cosine similarity against L2-normalized keys, divided by ``temperature``), just
    done in fp32 and chunked.

    Args:
        ir_module: the ``InformationRetrievalModule``.
        query: ``down_proj`` output, [B, S, ir_dim].
        top_m: how many entries the "top-m mass" column sums.

    Returns:
        dict of scalar floats: entropy (nats), max weight, top-m mass.
    """
    z = F.normalize(ir_module.z_keys.float(), p=2, dim=-1)
    q = query.reshape(-1, query.shape[-1]).float()
    ent, wmax, mass, n = 0.0, 0.0, 0.0, 0
    for start in range(0, q.shape[0], IR_CHUNK):
        qq = F.normalize(q[start:start + IR_CHUNK], p=2, dim=-1)
        w = torch.softmax(qq @ z.t() / ir_module.temperature, dim=-1)
        ent += float(-(w * w.clamp_min(1e-12).log()).sum(-1).sum())
        top = w.topk(top_m, dim=-1).values
        wmax += float(top[:, 0].sum())
        mass += float(top.sum())
        n += qq.shape[0]
    return {"entropy": ent / n, "max_weight": wmax / n, "top_m_mass": mass / n, "tokens": n}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """mean of a [B, S] quantity over the supervised positions only."""
    total = mask.sum()
    if total == 0:
        return float("nan")
    return float((values * mask).sum() / total)


@torch.no_grad()
def collect(model, dataset, device, max_batches, max_loops, n_loops_cfg, top_m):
    """Run the whole diagnostic over the held-out slice.

    Three forward passes per batch: the baseline (at ``max_loops``, which also yields the deeper
    early-exit curve), then the zero- and mean-ablated IR read at the trained ``n_loops``. Loop
    ``k``'s hidden state is independent of how many loops run after it, so the baseline's loop-k CE
    is directly comparable to an ablation run at a shallower total depth.
    """
    probe = IRProbe(model.moe).attach()
    acc = {
        "ce_sum": [0.0] * max_loops, "ce_n": [0] * max_loops,
        "ce_sum_zero": [0.0] * n_loops_cfg, "ce_n_zero": [0] * n_loops_cfg,
        "ce_sum_mean": [0.0] * n_loops_cfg, "ce_n_mean": [0] * n_loops_cfg,
        # index k holds "loop k+1 vs loop k"; index 0 is undefined
        "agree": [0.0] * max_loops, "logprob_gap": [0.0] * max_loops,
        "delta_ratio": [0.0] * max_loops, "delta_cos": [0.0] * max_loops,
        "ir_entropy": [0.0] * max_loops, "ir_max_w": [0.0] * max_loops, "ir_top_m": [0.0] * max_loops,
        "read_dispersion": [0.0] * max_loops,
        "ir_route_weight": [0.0] * max_loops, "ir_route_rate": [0.0] * max_loops,
        # oracle depth: index k = tokens first predicted correctly at loop k+1, plus a "never"
        # bucket. "correct" is the plain per-loop top-1 count, so the regression rate at any depth
        # is just (cumulative first-correct) - (correct there) -- no separate counter needed, and it
        # can be read at the TRAINED depth rather than only at whatever --max-loops was set to.
        "first_correct": [0] * max_loops, "never_correct": 0,
        "correct": [0] * max_loops, "total": 0,
        "query_cos": np.zeros((max_loops, max_loops)),
        "batches": 0,
    }

    for batch in dataset:
        if max_batches is not None and acc["batches"] >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        document_ids = batch["document_ids"].to(device)
        labels = batch["labels"].to(device)
        cu_seqlens, max_seqlen = cu_seqlens_from_doc_ids(document_ids)
        main_labels = labels[:, 1:].contiguous().view(-1)
        valid = (labels[:, 1:] != -100)                      # [B, S-1]
        valid_f = valid.float()

        probe.reset("full")
        hidden_all = model(input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                           return_hidden=True, n_loops=max_loops, skip_mtp=True)
        if isinstance(hidden_all, tuple):
            hidden_all = hidden_all[0]                       # kept: without skip_mtp it is a tuple
        loops_run = hidden_all.size(0)

        # ---- readout per loop: CE, top-1, convergence, oracle depth ----
        prev, correct_at = None, []
        for loop in range(loops_run):
            h = hidden_all[loop, :, :-1, :].contiguous().view(-1, hidden_all.size(-1))
            ce_sum, n_valid, extra = chunked_eval(model.lm_head, h, main_labels)
            acc["ce_sum"][loop] += ce_sum
            acc["ce_n"][loop] += n_valid
            correct_at.append(((extra["argmax"] == main_labels) & (main_labels != -100)).view(valid.shape))
            if prev is not None:
                acc["agree"][loop] += float((extra["argmax"] == prev["argmax"]).float().sum())
                acc["logprob_gap"][loop] += float((extra["logprob_top"] - prev["logprob_top"]).abs().sum())
            prev = extra

        # ---- residual-stream dynamics: is a loop moving h at all, and in a new direction? ----
        raw = probe.hidden_all                                # [loops_run, B, S, H], pre-norm
        prev_delta = None
        for loop in range(loops_run):
            base = probe.moe_input if loop == 0 else raw[loop - 1]
            delta = (raw[loop] - base)[:, :-1, :].float()
            h_norm = raw[loop][:, :-1, :].float().norm(dim=-1)
            acc["delta_ratio"][loop] += _masked_mean(delta.norm(dim=-1) / h_norm.clamp_min(1e-6), valid_f)
            if prev_delta is not None:
                acc["delta_cos"][loop] += _masked_mean(F.cosine_similarity(delta, prev_delta, dim=-1), valid_f)
            prev_delta = delta

        # ---- IR expert: what it retrieves, and how the router weights it ----
        for loop in range(loops_run):
            stats = retrieval_stats(probe.expert.ir_module, probe.queries[loop], top_m)
            acc["ir_entropy"][loop] += stats["entropy"]
            acc["ir_max_w"][loop] += stats["max_weight"]
            acc["ir_top_m"][loop] += stats["top_m_mass"]
            # how far a token's read strays from the mean read: the bias-term test in vector form
            read = probe.reads[loop].float().reshape(-1, probe.reads[loop].shape[-1])
            mean_read = read.mean(dim=0, keepdim=True)
            acc["read_dispersion"][loop] += float(
                (read - mean_read).norm(dim=-1).mean() / mean_read.norm().clamp_min(1e-6)
            )
            w, rate = probe.routed_ir_weight(loop)
            acc["ir_route_weight"][loop] += w
            acc["ir_route_rate"][loop] += rate

        # ---- query drift: does loop k re-issue loop j's query? ----
        for i in range(loops_run):
            qi = F.normalize(probe.queries[i].float(), p=2, dim=-1)
            for j in range(i, loops_run):
                qj = F.normalize(probe.queries[j].float(), p=2, dim=-1)
                acc["query_cos"][i, j] += float((qi * qj).sum(-1).mean())

        # ---- oracle minimum sufficient depth ----
        stacked = torch.stack(correct_at, dim=0)              # [loops_run, B, S-1]
        ever = stacked.any(dim=0)
        # first True along the loop axis; argmax on a bool picks the first maximum
        first = stacked.float().argmax(dim=0)
        for loop in range(loops_run):
            acc["first_correct"][loop] += int(((first == loop) & ever & valid).sum())
            acc["correct"][loop] += int((stacked[loop] & valid).sum())
        acc["never_correct"] += int((~ever & valid).sum())
        acc["total"] += int(valid.sum())

        # ---- Gate G1: the same slice with the IR read ablated ----
        for mode, key in (("zero", "zero"), ("mean", "mean")):
            probe.reset(mode)
            ablated = model(input_ids=input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                            return_hidden=True, n_loops=n_loops_cfg, skip_mtp=True)
            if isinstance(ablated, tuple):
                ablated = ablated[0]
            for loop in range(ablated.size(0)):
                h = ablated[loop, :, :-1, :].contiguous().view(-1, ablated.size(-1))
                ce_sum, n_valid, _ = chunked_eval(model.lm_head, h, main_labels)
                acc[f"ce_sum_{key}"][loop] += ce_sum
                acc[f"ce_n_{key}"][loop] += n_valid

        acc["batches"] += 1
        if acc["batches"] % 10 == 0:
            logger.info(f"[eval_stage0] processed {acc['batches']} batches ({acc['total']:,} supervised tokens)")

    probe.detach()
    return acc


def report(acc, n_loops_cfg, num_ir_entries, top_m):
    b = max(acc["batches"], 1)
    tokens = max(acc["total"], 1)
    loops_run = len(acc["ce_sum"])
    per_loop_ce = [s / max(c, 1) for s, c in zip(acc["ce_sum"], acc["ce_n"])]
    ce_zero = [s / max(c, 1) for s, c in zip(acc["ce_sum_zero"], acc["ce_n_zero"])]
    ce_mean = [s / max(c, 1) for s, c in zip(acc["ce_sum_mean"], acc["ce_n_mean"])]
    max_entropy = math.log(num_ir_entries)

    print(f"\n=== 1. IR retrieval entropy (max = ln {num_ir_entries} = {max_entropy:.3f} nats) ===")
    print(f"  {'loop':<6} {'entropy':>9} {'frac of max':>12} {'max weight':>12} {f'top-{top_m} mass':>13} {'read dispersion':>16}")
    for loop in range(loops_run):
        ent = acc["ir_entropy"][loop] / b
        print(f"  {loop + 1:<6} {ent:>9.4f} {ent / max_entropy:>12.4f} {acc['ir_max_w'][loop] / b:>12.6f} "
              f"{acc['ir_top_m'][loop] / b:>13.6f} {acc['read_dispersion'][loop] / b:>16.4f}")
    print(f"  uniform reference: max weight {1.0 / num_ir_entries:.6f}, top-{top_m} mass "
          f"{top_m / num_ir_entries:.6f}")
    print("  read dispersion = mean ||y_t - y_bar|| / ||y_bar||; ~0 means every token reads the same vector")

    print("\n=== 2. IR ablation (Gate G1) ===")
    print(f"  {'loop':<6} {'CE full':>10} {'CE read=0':>11} {'dCE':>9} {'CE read=mean':>14} {'dCE':>9}")
    for loop in range(n_loops_cfg):
        print(f"  {loop + 1:<6} {per_loop_ce[loop]:>10.4f} {ce_zero[loop]:>11.4f} "
              f"{ce_zero[loop] - per_loop_ce[loop]:>9.4f} {ce_mean[loop]:>14.4f} "
              f"{ce_mean[loop] - per_loop_ce[loop]:>9.4f}")
    d_zero = ce_zero[n_loops_cfg - 1] - per_loop_ce[n_loops_cfg - 1]
    d_mean = ce_mean[n_loops_cfg - 1] - per_loop_ce[n_loops_cfg - 1]
    print(f"  GATE G1 (final-loop dCE with the read zeroed > {G1_THRESHOLD}): "
          f"{d_zero:.4f} -> {'PASS' if d_zero > G1_THRESHOLD else 'FAIL'}")
    print(f"  content share: dCE(zero) - dCE(mean) = {d_zero - d_mean:.4f} nats is what the read's "
          "per-token CONTENT is worth;")
    print(f"  the remaining {d_mean:.4f} nats is worth a constant vector, i.e. a bias term.")
    print(f"\n  {'loop':<6} {'IR routed weight':>18} {'IR selection rate':>19}")
    for loop in range(loops_run):
        print(f"  {loop + 1:<6} {acc['ir_route_weight'][loop] / b:>18.4f} {acc['ir_route_rate'][loop] / b:>19.4f}")

    print("\n=== 3. Query drift: cos(down_proj(h_i), down_proj(h_j)) ===")
    print("  " + " " * 8 + "".join(f"{f'loop {j + 1}':>10}" for j in range(loops_run)))
    for i in range(loops_run):
        row = "".join(f"{acc['query_cos'][i, j] / b:>10.4f}" if j >= i else " " * 10 for j in range(loops_run))
        print(f"  loop {i + 1:<3}{row}")

    print("\n=== 4. Loop-to-loop dynamics ===")
    print(f"  {'transition':<18} {'top-1 flip':>11} {'mean |dlogp|':>13} {'||dh||/||h||':>13} {'cos(d_k,d_k-1)':>16}")
    for loop in range(loops_run):
        label = "decoder -> loop 1" if loop == 0 else f"loop {loop} -> {loop + 1}"
        flip = "n/a" if loop == 0 else f"{1.0 - acc['agree'][loop] / tokens:>11.4f}"
        gap = "n/a" if loop == 0 else f"{acc['logprob_gap'][loop] / tokens:>13.4f}"
        cos = "n/a" if loop == 0 else f"{acc['delta_cos'][loop] / b:>16.4f}"
        print(f"  {label:<18} {flip:>11} {gap:>13} {acc['delta_ratio'][loop] / b:>13.4f} {cos:>16}")

    print("\n=== 5. Oracle minimum sufficient depth ===")
    print(f"  {'loop':<6} {'first correct':>14} {'oracle cum.':>12} {'top-1 acc':>10} {'regressed':>10}")
    cum, oracle_cum = 0, []
    for loop in range(loops_run):
        cum += acc["first_correct"][loop]
        oracle_cum.append(cum)
        print(f"  {loop + 1:<6} {acc['first_correct'][loop] / tokens:>14.4f} {cum / tokens:>12.4f} "
              f"{acc['correct'][loop] / tokens:>10.4f} {(cum - acc['correct'][loop]) / tokens:>10.4f}")
    print(f"  never correct at any depth: {acc['never_correct'] / tokens:.4f}")
    trained = n_loops_cfg - 1
    print(f"  at the trained depth (loop {n_loops_cfg}): oracle {oracle_cum[trained] / tokens:.4f} vs. actual "
          f"{acc['correct'][trained] / tokens:.4f} -- headroom "
          f"{(oracle_cum[trained] - acc['correct'][trained]) / tokens:.4f}")
    print("  'regressed' = right at some earlier loop, wrong at this one. measured on plain LM")
    print("  continuation with no evidence pathway attached -- a floor on today's checkpoint, not a")
    print("  recommended depth.")

    print(f"\n=== 6. Per-loop CE, including past the trained depth (n_loops={n_loops_cfg}) ===")
    for loop in range(loops_run):
        marker = "" if loop < n_loops_cfg else "   <- untrained depth"
        print(f"  loop {loop + 1}: CE={per_loop_ce[loop]:.4f}  ppl={math.exp(per_loop_ce[loop]):.3f}{marker}")

    print("\n=== summary line ===")
    print(f"  tokens={tokens:,}  entropy/max={acc['ir_entropy'][n_loops_cfg - 1] / b / max_entropy:.4f}  "
          f"dCE(zero)={d_zero:.4f}  dCE(mean)={d_mean:.4f}  "
          f"cos(q1,q{n_loops_cfg})={acc['query_cos'][0, n_loops_cfg - 1] / b:.4f}  "
          f"oracle_headroom={(oracle_cum[trained] - acc['correct'][trained]) / tokens:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Stage 0 diagnostics (NEXT.md Phase 1)")
    parser.add_argument("--checkpoint", "-c", default=find_latest_checkpoint(os.path.join(BASE_DIR, "ckpts", "training")))
    parser.add_argument("--tokenizer", "-t", default=TOKENIZER_DIR)
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, TrainingConfig.data_dir))
    parser.add_argument("--phase", default=TrainingConfig.phase)
    parser.add_argument("--start-doc-idx", type=int, default=None,
                        help="override the checkpoint's own global_offset (an SFT checkpoint's indexes another corpus)")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.Batch_size)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-loops", type=int, default=ModelConfig.Params["n_loops"],
                        help="run the recurrence this deep; loops past n_loops are the untrained-depth probe")
    parser.add_argument("--top-m", type=int, default=32, help="how many IR entries the top-m mass column sums")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint is None:
        raise SystemExit("No checkpoint found in ckpts/training and none passed via --checkpoint")
    n_loops_cfg = ModelConfig.Params["n_loops"]
    if args.max_loops < n_loops_cfg:
        raise SystemExit(f"--max-loops ({args.max_loops}) must be >= the trained n_loops ({n_loops_cfg})")

    logger.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model, checkpoint_offset = load_model(args.checkpoint, args.device)
    start_doc_idx = args.start_doc_idx if args.start_doc_idx is not None else checkpoint_offset
    logger.info(f"Held-out slice starts at doc {start_doc_idx:,} (checkpoint global_offset={checkpoint_offset:,})")

    dataset = Dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=ModelConfig.Params["max_seq_len"],
        split=args.phase,
        num_mtp_tokens=ModelConfig.Params["mtp_num_extra_tokens"],
        start_doc_idx=start_doc_idx,
    )
    if start_doc_idx >= dataset.num_docs:
        raise SystemExit(f"start_doc_idx ({start_doc_idx:,}) >= num_docs ({dataset.num_docs:,}) in {args.phase}")

    acc = collect(model, dataset, args.device, args.max_batches, args.max_loops, n_loops_cfg, args.top_m)
    if acc["total"] == 0:
        raise SystemExit("No supervised tokens collected -- check --start-doc-idx/--max-batches.")
    logger.info(f"Evaluated {acc['batches']} batches, {acc['total']:,} supervised tokens")
    report(acc, n_loops_cfg, ModelConfig.Params["num_ir_entries"], args.top_m)


if __name__ == "__main__":
    main()

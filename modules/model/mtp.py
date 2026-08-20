from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from modules.model.gemma4 import Gemma4MLP, GemmaRMSNorm as RMSNorm
from modules.model.modules import SmallLMHead
import transformer_engine.pytorch as te
from transformer_engine.pytorch import checkpoint

# token chunk size for the chunked LM head cross entropy (see _chunked_linear_ce)
CE_CHUNK_SIZE = 8192 # 2048


class MTPHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_extra_tokens: int = 3, dropout: float = 0.1, lm_head_factor: int = 16):
        super().__init__()
        intermediate_size = int(hidden_size * num_extra_tokens * 1.5)
        
        self.gate = te.Linear(hidden_size, intermediate_size, bias=False)
        self.up = te.Linear(hidden_size, intermediate_size, bias=False)
        self.down = te.Linear(int(hidden_size * 1.5), hidden_size // 2, bias=False)
        
        self.norm = RMSNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.SiLU()
        
        self.lm_head = SmallLMHead(hidden_size // 2, vocab_size, factor=lm_head_factor)
        
        self.num_extra_tokens = num_extra_tokens
        self.late_token_loss = True  # whether to compute loss for the last few tokens that only have MTP supervision

    def forward(self, x: torch.Tensor):
        gate = self.gate(x)  # [batch_size, seq_len, intermediate_size]
        up = self.up(x)      # [batch_size, seq_len, intermediate_size]
        mid = self.act_fn(gate) * up
        mid = self.dropout(mid)
        
        mid = mid.view(mid.size(0), mid.size(1), self.num_extra_tokens, -1)  # [batch_size, seq_len, num_extra_tokens, hidden_size * 1.5]
        out = self.down(mid)  # [batch_size, seq_len, num_extra_tokens, hidden_size // 2]
        
        if self.late_token_loss:
            # return the hidden states instead of computing all logits at once to save vram footprint
            return out 
        else:
            return [self.lm_head(out[:, :, i, :]) for i in range(self.num_extra_tokens)]


def pad_for_low_fp(tensor: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """Pads the input tensor on the sequence dimension to be a multiple of `multiple` for better low-precision performance."""
    seq_len = tensor.size(0)
    pad_len = (multiple - (seq_len % multiple)) % multiple
    if pad_len > 0:
        padding = torch.zeros(pad_len, tensor.size(1), device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)
    return tensor

def unpad(tensor: torch.Tensor, original_seq_len: int) -> torch.Tensor:
    """Removes padding from the input tensor to restore the original sequence length."""
    return tensor[:original_seq_len].contiguous()

def _safe_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """cross entropy that returns a graph-connected zero instead of NaN when every label is -100"""
    if (labels != -100).any():
        return F.cross_entropy(logits, labels)
    return logits.sum() * 0.0


def _chunked_linear_ce(lm_head: nn.Module, hidden: torch.Tensor, labels: torch.Tensor,
                       chunk_size: int = CE_CHUNK_SIZE,
                       collect_metrics: bool = False,
                       weights: torch.Tensor = None):
    """mean cross entropy of lm_head(hidden) [T, H] vs labels [T], without ever materializing the full [T, vocab] logits.

    tokens go through in chunks, each projection checkpointed so its logits are freed after the forward and recomputed in backward.
    peak logit memory is chunk_size * vocab instead of T * vocab
    the logits dominate activation memory here. otherwise equivalent to a normal F.cross_entropy(lm_head(hidden), labels, ignore_index=-100).

    ``weights`` (optional, ``[T]``) makes it a *weighted* mean, ``sum(w * ce) / sum(w)`` over the
    valid tokens -- SFT passes one weight per supervised token so that every conversation counts
    the same regardless of answer length (see ``modules/data/sft_dataset.py``). None keeps the plain
    per-token mean, which is what pretraining uses and what every previously reported CE number is.

    when collect_metrics is set, additionally returns a dict with mean p_max/top1_acc over the same
    valid tokens -- reusing this chunk's already materialized logits under no_grad, so it's a few
    cheap reductions, not a second forward pass. Those stay **unweighted** even when ``weights`` is
    given: they are the reported confidence signal, compared across pretraining, SFT and
    eval_calibration.py, and a per-conversation weighting would silently redefine them. ``p_max`` is
    the model's confidence signal everywhere downstream now that the learned correctness head is
    gone: it beat ``p_correct`` on ECE and AUROC on the real checkpoint, and ``1 - p_correct``
    scored below chance at flagging unanswerable questions (docs/CONCLUSION.md).
    """
    valid = labels != -100
    n_valid = valid.sum()
    if n_valid == 0:
        # keep the head in the autograd graph so DDP still sees its grads
        z = lm_head(hidden[:1]).sum() * 0.0
        if collect_metrics:
            zm = z.new_zeros(())
            return z, {"p_max": zm, "top1_acc": zm}
        return z

    if weights is None:
        denominator = n_valid
    else:
        # masked by `valid` rather than trusting the caller's zeros, so a weight left on an ignored
        # position can't inflate the denominator. clamped because a division by an all-zero weight
        # vector would poison the whole step with NaNs -- it cannot happen with the dataset's
        # weights (every supervised token gets 1/n > 0) but the failure mode is silent.
        denominator = torch.clamp((weights * valid).sum(), min=1e-6)

    def _chunk_loss(h: torch.Tensor, l: torch.Tensor, w: torch.Tensor = None):
        h0 = h.size(0)
        logits = unpad(lm_head(pad_for_low_fp(h)), h0)
        if w is None:
            # sum, not mean: divided by the global denominator below
            ce = F.cross_entropy(logits, l, ignore_index=-100, reduction="sum")
        else:
            # reduction="none" already returns 0 at ignore_index positions, so the dot product
            # needs no extra masking. the [chunk] fp32 vector is negligible next to the logits.
            ce = (F.cross_entropy(logits, l, ignore_index=-100, reduction="none") * w).sum()
        if not collect_metrics:
            return ce
        valid_chunk = (l != -100).to(logits.dtype)
        with torch.no_grad():
            max_logit, argmax = logits.max(-1)
            # p_max == softmax(logits).max() == 1 / sum_j exp(l_j - l_max). computed this
            # way to avoid an fp32 copy of the [chunk, vocab] logits: at chunk_size=8192 /
            # vocab=65536 a `.float().softmax(-1)` is two ~2GB transients, allocated on
            # every step AND again on the checkpoint recompute. the exp() temporary here
            # stays in the logits' own dtype and the sum accumulates in fp32.
            p_max_sum = (
                (1.0 / (logits - max_logit.unsqueeze(-1)).exp().sum(-1, dtype=torch.float32))
                * valid_chunk
            ).sum()
            top1_sum = ((argmax == l).to(logits.dtype) * valid_chunk).sum()
        return ce, p_max_sum, top1_sum

    loss_sum = hidden.new_zeros(())
    if collect_metrics:
        p_max_sum = hidden.new_zeros(())
        top1_sum = hidden.new_zeros(())
    T = hidden.size(0)
    for start in range(0, T, chunk_size):
        # the weights chunk is appended only when there is one: passing None through
        # transformer_engine's checkpoint would make it a non-tensor argument to a recomputed
        # function, which is exactly the shape of thing that breaks quietly under recompute
        args = (hidden[start:start + chunk_size], labels[start:start + chunk_size])
        if weights is not None:
            args = args + (weights[start:start + chunk_size],)
        if torch.is_grad_enabled() and args[0].requires_grad:
            out = checkpoint(_chunk_loss, *args, use_reentrant=False)
        else:
            out = _chunk_loss(*args)
        if collect_metrics:
            ce, pm, t1 = out
            p_max_sum = p_max_sum + pm
            top1_sum = top1_sum + t1
        else:
            ce = out
        loss_sum = loss_sum + ce

    if collect_metrics:
        # note the two denominators: the loss is weighted (if asked), the metrics never are
        return loss_sum / denominator, {"p_max": p_max_sum / n_valid, "top1_acc": top1_sum / n_valid}
    return loss_sum / denominator


def compute_mtp_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mtp_outputs: torch.Tensor = None,
    lm_head: nn.Module = None,
    lambda_mtp: float = 0.1,
    main_lm_head: nn.Module = None,
    pad_mask: torch.Tensor = None,
    loop_ce_weights: list = None,
    return_metrics: bool = False,
    loop_ce_subsample: float = 1.0,
    loss_weights: torch.Tensor = None,
):
    """
    Args:
        loop_ce_subsample (float, optional): fraction of token positions to supervise on the
            NON-final loops (the final loop is always supervised in full). The LM head is the
            single most expensive GEMM in the model and per-loop CE runs it once per loop, so at
            n_loops=3 two thirds of that cost buys the low-weight intermediate readouts. Those
            readouts are a regularizer, not the main objective, and a CE mean over a uniform token
            subsample is an unbiased estimate of the full mean -- so the ``loop_ce_weights``
            semantics are unchanged, only the variance goes up. 1.0 disables subsampling.
        loss_weights (torch.Tensor, optional): ``[B, S]`` per-token loss weights, aligned with
            ``targets`` (so each term shifts them exactly as it shifts the labels). Turns every CE
            term -- per-loop and MTP alike -- into a weighted mean. SFT passes
            ``1 / supervised_tokens_in_this_conversation`` here so a 6-token refusal stops
            out-earning a multi-sentence answer (NEXT.md Phase 2's fix #3). None reproduces the
            plain per-token mean exactly, which is what pretraining runs.
    """
    metrics = {} if return_metrics else None
    if main_lm_head is not None:
        # outputs: [n_loops, B, S, H] per-loop post-norm hidden states (PLAN.md Step 4a). Project +
        # CE per loop, each internally chunked, so logits for more than one loop/chunk are never
        # live at once. Without per-loop supervision, intermediate hidden states are only ever
        # optimized as inputs to the next loop, never as something lm_head can read.
        assert loop_ce_weights is not None and len(loop_ce_weights) == outputs.size(0), (
            f"loop_ce_weights must have exactly one weight per loop, got {loop_ce_weights} "
            f"for {outputs.size(0)} loops"
        )
        main_labels = targets[:, 1:].contiguous().view(-1)
        main_weights = (
            None if loss_weights is None else loss_weights[:, 1:].contiguous().view(-1)
        )
        loss_ce = None
        loss = outputs.new_zeros(())
        n_loops = len(loop_ce_weights)
        per_loop_ce = [] if return_metrics else None

        # token subsample for the non-final loops (see loop_ce_subsample in the docstring).
        # indices are built in the flat [B*S] space rather than [B*(S-1)] so each loop's hidden
        # slice can be gathered straight out of its contiguous [B, S, H] plane -- indexing the
        # [:, :-1, :] view instead would force a full [B*(S-1), H] copy first, which is most of
        # what the subsample is trying to avoid.
        S, H = outputs.size(2), outputs.size(-1)
        sub_flat, sub_labels, sub_weights = None, None, None
        if 0.0 < loop_ce_subsample < 1.0 and n_loops > 1:
            n_pos = main_labels.numel()
            k = max(1, int(round(n_pos * loop_ce_subsample)))
            sel = torch.randperm(n_pos, device=outputs.device)[:k]
            sub_flat = (sel // (S - 1)) * S + (sel % (S - 1))
            sub_labels = main_labels.index_select(0, sel)
            # weighted case: sum(w*ce)/sum(w) over a uniform subsample is a ratio estimator of the
            # full weighted mean rather than the strictly unbiased estimator the unweighted case
            # gets. Same regularizer-not-objective argument applies (these are the low-weight
            # intermediate readouts), so the extra bias is not worth running the model's largest
            # GEMM at full width to avoid.
            if main_weights is not None:
                sub_weights = main_weights.index_select(0, sel)

        for loop, weight in enumerate(loop_ce_weights):
            if sub_flat is not None and loop != n_loops - 1:
                hidden = outputs[loop].reshape(-1, H).index_select(0, sub_flat)
                labels_for_loop = sub_labels
                weights_for_loop = sub_weights
            else:
                hidden = outputs[loop, :, :-1, :].contiguous().view(-1, H)
                labels_for_loop = main_labels
                weights_for_loop = main_weights
            # p_max / top-1 are read off the FINAL loop only: they are the reported confidence
            # signal, and reading them at an intermediate depth would describe a readout no
            # downstream consumer (abstention, calibration, generation) ever uses.
            if return_metrics and loop == n_loops - 1:
                loop_ce, head_metrics = _chunked_linear_ce(
                    main_lm_head, hidden, labels_for_loop, collect_metrics=True,
                    weights=weights_for_loop,
                )
                metrics.update(head_metrics)
            else:
                loop_ce = _chunked_linear_ce(
                    main_lm_head, hidden, labels_for_loop, weights=weights_for_loop,
                )
            loss = loss + weight * loop_ce
            loss_ce = loop_ce  # last iteration is the final loop's raw (unweighted) CE, for logging
            if return_metrics:
                per_loop_ce.append(loop_ce.detach())
        if return_metrics:
            metrics["per_loop_ce"] = per_loop_ce
            metrics.setdefault("p_max", None)
            metrics.setdefault("top1_acc", None)
    else:
        # main loss: targets shifted by 1 relative to inputs (outputs are already logits)
        main_logits = outputs[:, :-1, :].contiguous()
        main_labels = targets[:, 1:].contiguous()
        loss_ce = _safe_cross_entropy(main_logits.view(-1, main_logits.size(-1)), main_labels.view(-1))
        loss = loss_ce

    if mtp_outputs is not None and lm_head is not None:
        # mtp_outputs shape: [batch_size, seq_len, num_extra_tokens, hidden_size // 2]
        num_extra_tokens = mtp_outputs.size(2)
        # auxiliary MTP losses: shift targets further for each head
        for i in range(num_extra_tokens):
            shift = i + 2
            # slice hidden states before lm_head projection to save memory
            hidden = mtp_outputs[:, :-shift, i, :].contiguous()
            aux_labels = targets[:, shift:].clone().contiguous()
            
            if pad_mask is not None:
                source_is_pad = pad_mask[:, :-shift]
                aux_labels[source_is_pad] = -100

            # chunked LM head + CE, bounds the per head logit memory. the weights shift with the
            # labels (both index the token being predicted), and _chunked_linear_ce masks them by
            # `labels != -100` itself, so the pad zeroing above needs no mirror here.
            aux_weights = (
                None if loss_weights is None
                else loss_weights[:, shift:].contiguous().view(-1)
            )
            hidden = hidden.view(-1, hidden.size(-1))
            aux_loss = _chunked_linear_ce(lm_head, hidden, aux_labels.view(-1), weights=aux_weights)
            
            loss = loss + lambda_mtp * aux_loss

    if return_metrics:
        return loss, loss_ce.detach(), metrics
    return loss, loss_ce.detach()


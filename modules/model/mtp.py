from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from modules.model.gemma4 import Gemma4MLP, GemmaRMSNorm as RMSNorm
from modules.model.modules import SmallLMHead
import transformer_engine.pytorch as te
from transformer_engine.pytorch import checkpoint

# token chunk size for the chunked LM head cross entropy (see _chunked_linear_ce)
CE_CHUNK_SIZE = 2048


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
                       correct_proj: nn.Module = None):
    """mean cross entropy of lm_head(hidden) [T, H] vs labels [T], without ever materializing the full [T, vocab] logits.

    tokens go through in chunks, each projection checkpointed so its logits are freed after the forward and recomputed in backward.
    peak logit memory is chunk_size * vocab instead of T * vocab
    the logits dominate activation memory here. otherwise equivalent to a normal F.cross_entropy(lm_head(hidden), labels, ignore_index=-100).

    when correct_proj is given (PLAN.md Step 4b), also returns the mean correctness head BCE loss
    for the same tokens, reusing each chunk's already-live logits for the free is_correct target
    instead of a second lm_head(hidden) pass. Returns (ce_loss, conf_loss) then, else just ce_loss.
    """
    n_valid = (labels != -100).sum()
    if n_valid == 0:
        # keep the head(s) in the autograd graph so DDP still sees their grads
        z = lm_head(hidden[:1]).sum() * 0.0
        return (z, correct_proj(hidden[:1]).sum() * 0.0) if correct_proj is not None else z

    if correct_proj is not None:
        def _chunk_loss(h: torch.Tensor, l: torch.Tensor):
            h0 = h.size(0)
            logits = unpad(lm_head(pad_for_low_fp(h)), h0)
            ce = F.cross_entropy(logits, l, ignore_index=-100, reduction="sum")
            valid = (l != -100).to(logits.dtype)
            with torch.no_grad():
                # free target: no labels, no extra forward pass needed. must stay no_grad -- a
                # differentiable target here would leak gradient back into the LM logits/lm_head
                # through the "correct" label itself, on top of correct_proj's own gradient.
                is_correct = (logits.argmax(-1) == l).float()
            correct_logit = correct_proj(h).squeeze(-1)
            conf = F.binary_cross_entropy_with_logits(correct_logit, is_correct, reduction="none")
            return ce, (conf * valid).sum()
    else:
        def _chunk_loss(h: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
            h0 = h.size(0)
            logits = unpad(lm_head(pad_for_low_fp(h)), h0)
            # sum, not mean: divided by the global valid token count below
            return F.cross_entropy(logits, l, ignore_index=-100, reduction="sum")

    loss_sum = hidden.new_zeros(())
    conf_sum = hidden.new_zeros(()) if correct_proj is not None else None
    T = hidden.size(0)
    for start in range(0, T, chunk_size):
        h_chunk = hidden[start:start + chunk_size]
        l_chunk = labels[start:start + chunk_size]
        if torch.is_grad_enabled() and h_chunk.requires_grad:
            out = checkpoint(_chunk_loss, h_chunk, l_chunk, use_reentrant=False)
        else:
            out = _chunk_loss(h_chunk, l_chunk)
        if correct_proj is not None:
            ce, conf = out
            conf_sum = conf_sum + conf
        else:
            ce = out
        loss_sum = loss_sum + ce

    if correct_proj is not None:
        return loss_sum / n_valid, conf_sum / n_valid
    return loss_sum / n_valid


def compute_mtp_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mtp_outputs: torch.Tensor = None,
    lm_head: nn.Module = None,
    lambda_mtp: float = 0.1,
    main_lm_head: nn.Module = None,
    pad_mask: torch.Tensor = None,
    loop_ce_weights: list = None,
    correct_proj: nn.Module = None,
    lambda_conf: float = 0.0,
):
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
        loss_ce = None
        loss = outputs.new_zeros(())
        n_loops = len(loop_ce_weights)
        for loop, weight in enumerate(loop_ce_weights):
            hidden = outputs[loop, :, :-1, :].contiguous().view(-1, outputs.size(-1))
            # correctness head (PLAN.md Step 4b) reads only the final loop's hidden states --
            # p_halt asks "is more compute useful", this asks "is this prediction correct", and
            # they come apart on confident hallucinations, so it's deliberately not per-loop.
            if loop == n_loops - 1 and correct_proj is not None:
                loop_ce, conf_loss = _chunked_linear_ce(main_lm_head, hidden, main_labels, correct_proj=correct_proj)
                loss = loss + lambda_conf * conf_loss
            else:
                loop_ce = _chunked_linear_ce(main_lm_head, hidden, main_labels)
            loss = loss + weight * loop_ce
            loss_ce = loop_ce  # last iteration is the final loop's raw (unweighted) CE, for logging
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

            # chunked LM head + CE, bounds the per head logit memory
            hidden = hidden.view(-1, hidden.size(-1))
            aux_loss = _chunked_linear_ce(lm_head, hidden, aux_labels.view(-1))
            
            loss = loss + lambda_mtp * aux_loss
            
    return loss, loss_ce.detach()


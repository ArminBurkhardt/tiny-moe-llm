from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from modules.model.gemma4 import Gemma4MLP, GemmaRMSNorm as RMSNorm
from modules.model.modules import SmallLMHead
import transformer_engine.pytorch as te


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


def compute_mtp_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mtp_outputs: torch.Tensor = None,
    lm_head: nn.Module = None,
    lambda_mtp: float = 0.1,
    main_lm_head: nn.Module = None,
    pad_mask: torch.Tensor = None,
):
    if main_lm_head is not None:
        hidden = outputs[:, :-1, :].contiguous()
        hidden = hidden.view(-1, hidden.size(-1))
        hidden_0 = hidden.size(0)
        hidden = pad_for_low_fp(hidden)
        main_logits = main_lm_head(hidden)
        main_logits = unpad(main_logits, hidden_0)
        main_labels = targets[:, 1:].contiguous()
        loss_ce = _safe_cross_entropy(main_logits, main_labels.view(-1))
    else:
        # main loss: targets shifted by 1 relative to inputs
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
                
            # apply LM head on flattened tensor to avoid large intermediate buffers
            hidden = hidden.view(-1, hidden.size(-1))
            hidden_0 = hidden.size(0)
            hidden = pad_for_low_fp(hidden)
            aux_logits = lm_head(hidden)
            aux_logits = unpad(aux_logits, hidden_0)
            aux_loss = _safe_cross_entropy(aux_logits, aux_labels.view(-1))
            
            loss = loss + lambda_mtp * aux_loss
            
    return loss, loss_ce.detach()


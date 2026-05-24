import torch
from dataclasses import dataclass


@dataclass
class EncoderOutput:
    """encoder output container

    Attributes:
        last_hidden_state: hidden states from the last transformer layer,
            shape ``[batch, seq_len, hidden_size]``
        hidden_states: all intermediate hidden states (including the embedding
            layer at index 0)
    """

    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


def create_causal_attention_mask(seq_len: int, dtype: torch.dtype = torch.bool, device: torch.device = None) -> torch.Tensor:
    """creates a causal attention mask of shape [1, 1, seq_len, seq_len]"""
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=dtype, device=device)).unsqueeze(0).unsqueeze(0)
    return mask

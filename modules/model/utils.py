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

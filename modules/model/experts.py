import torch
from torch import nn
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm, Gemma4TextAttention as GroupedQueryAttention
from modules.model.embeddings import RotaryPositionEmbeddingsFrequency as RoPEFreq

   
class SelfAttention(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(input_size)
        self.attn = GroupedQueryAttention(
            hidden_size=input_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=input_size // num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x_norm = self.norm(x)
        attn_output = self.attn(
            hidden_states=x_norm, 
            attention_mask=attn_mask,
        )
        return self.dropout(attn_output)



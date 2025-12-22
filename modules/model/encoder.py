import torch 
from torch import nn

# LLM Encoder Module

class Encoder(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, ffn_dim: int, num_hidden_layers: int, dropout_rate: float):
        super(Encoder, self).__init__()
        self.config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "ffn_dim": ffn_dim,
            "num_hidden_layers": num_hidden_layers,
            "dropout_rate": dropout_rate
        }
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout_rate,
                activation='gelu'
            ) for _ in range(num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, input_ids, attention_mask=None):
        x = input_ids
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)
        x = self.norm(x)
        return x

## TODO replace with Gemma3 https://huggingface.co/google/gemma-3-1b-it

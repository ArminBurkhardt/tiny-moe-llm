import torch
from torch import nn
import transformer_engine.pytorch as te
from modules.model.gemma4 import GemmaRMSNorm as RMSNorm, Gemma4TextAttention as GroupedQueryAttention
from modules.model.information_retrieval import InformationRetrievalModule

   
class SelfAttention(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.1, num_heads: int = 8, num_kv_heads: int = 4):
        super().__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(input_size)
        self.attn = GroupedQueryAttention(
            hidden_size=input_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=input_size // num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor = None, max_seqlen: int = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] = None, kv_cache=None) -> torch.Tensor:
        x_norm = self.norm(x)
        attn_output = self.attn(
            hidden_states=x_norm,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
        )
        return self.dropout(attn_output)


class CrossAttention(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.1, num_heads: int = 8, num_kv_heads: int = 4):
        super().__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(input_size)
        self.attn = GroupedQueryAttention(
            hidden_size=input_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=input_size // num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, other: torch.Tensor, cu_seqlens: torch.Tensor = None, max_seqlen: int = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] = None, kv_cache=None) -> torch.Tensor:
        x_norm = self.norm(x)
        attn_output = self.attn(
            hidden_states=x_norm,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            other_states=other,
            kv_cache=kv_cache,
        )
        return self.dropout(attn_output)

class InformationRetrievalExpert(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        num_entries: int,
        ir_dim: int,
        num_heads: int = 8, 
        num_kv_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(input_size)
        self.attn = GroupedQueryAttention(
            hidden_size=input_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=input_size // num_heads,
            dropout=dropout,
        )
        
        # operate on the per-head dimension for more efficient retrieval
        self.ir_module = InformationRetrievalModule(
            num_entries=num_entries,
            latent_dim=ir_dim,
            output_dim=ir_dim,
            temperature=1.0,
            use_min_dist=False,
            residual=residual,
        )
        self.down_proj = te.Linear(input_size, ir_dim, bias=False)
        self.up_proj = te.Linear(ir_dim, input_size, bias=False)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor = None, max_seqlen: int = None, position_embeddings: tuple[torch.Tensor, torch.Tensor] = None, kv_cache=None) -> torch.Tensor:
        x_norm = self.norm(x)

        down = self.down_proj(x_norm)
        ir_output = self.ir_module(down)
        information = self.up_proj(ir_output)

        attn_output = self.attn(
            hidden_states=x_norm,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            other_states=information,
            kv_cache=kv_cache,
        )
        return self.dropout(attn_output)


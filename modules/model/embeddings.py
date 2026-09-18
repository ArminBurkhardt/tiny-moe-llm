import torch
from torch import nn
from torch.nn import functional as F
import math

class RotaryPositionEmbeddings(nn.Module):
    """Rotary Position Embeddings for Transformer models"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        """
        Args:
            dim: The feature dimension to rotate (D)
            max_position_embeddings: The maximum sequence length
            base: The base for the inverse frequency
        """
        super().__init__()
        self.rope_freqs = RotaryPositionEmbeddingsFrequency(
            dim=dim, 
            max_position_embeddings=max_position_embeddings, 
            base=base
        )
        
    def forward(self, x: torch.Tensor):
        cos, sin = self.rope_freqs(x, seq_len=x.shape[-2])
        return (x * cos) + (rotate_half(x) * sin)
        

class RotaryPositionEmbeddingsFrequency(nn.Module):
    """Rotary Position Embeddings for Grouped Query Attention"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        """
        Args:
            dim: the feature dimension to rotate (D)
            max_position_embeddings: maximum sequence length
            base: the base for the inverse frequency
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # calculate the inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # precompute the cos and sin matrices
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # freqs shape [seq_len, dim // 2]
        freqs = torch.outer(t, self.inv_freq)
        
        # concat to match the head dimension [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # cache cos and sin. reshaped for broadcasting [1, 1, seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x shape: [batch_size, num_heads, seq_len, head_dim]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def slice(self, start: int, length: int, dtype: torch.dtype):
        """cos/sin for absolute positions ``[start, start + length)`` -- for KV-cached decoding,
        where the new tokens being processed do not start at position 0."""
        return (
            self.cos_cached[:, :, start:start + length, ...].to(dtype=dtype),
            self.sin_cached[:, :, start:start + length, ...].to(dtype=dtype),
        )

    def gather(self, position_ids: torch.Tensor, dtype: torch.dtype):
        """cos/sin at ARBITRARY per token positions ``[B, S]`` -> ``[B, 1, S, dim]``.

        ``slice`` covers every case where positions are one contiguous run. Retrieved evidence is
        not such a case: each chunk gets a position basis restarting at 0, so the packed evidence
        axis carries positions like ``[0,1,2,0,1,0,1,2,3]`` and has to be indexed rather than
        sliced. Within-chunk order survives (a copied span still reads left to right); cross-chunk
        geometry is absent, which is correct -- two chunks from different documents have no relative
        position, and giving them one would invent an ordering the retriever never meant.
        """
        # clamped because this is an unchecked gather and an out of range index here does not raise:
        # it trips a device side assert, which is reported asynchronously and leaves the process
        # spinning against a dead CUDA context with no traceback and no error in the log. A position
        # past the cache means a chunk longer than the model's whole context, which the corpus
        # builder does not produce -- so the clamp never fires on real data, and when something does
        # reach it a wrong position is a far cheaper failure than a run that hangs until someone
        # notices the GPU is idle.
        limit = self.cos_cached.shape[2] - 1
        position_ids = position_ids.clamp(0, limit)
        cos = self.cos_cached[0, 0].to(dtype=dtype)[position_ids]   # [B, S, dim]
        sin = self.sin_cached[0, 0].to(dtype=dtype)[position_ids]
        return cos.unsqueeze(1), sin.unsqueeze(1)                   # broadcast over heads

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(x, cos, sin):
    """``apply_rotary_pos_emb`` for one side only.

    Cross attention over evidence rotates its keys on a different position basis than its queries
    (see ``RotaryPositionEmbeddingsFrequency.gather``), so the two sides cannot share one call.
    """
    return (x * cos) + (rotate_half(x) * sin)


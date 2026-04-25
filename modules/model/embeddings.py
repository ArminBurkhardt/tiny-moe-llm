import torch
from torch import nn
from modules.model.modules import LinearAttention

class PerLayerEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(PerLayerEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.rope = RoPE(embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.attn = LinearAttention(embedding_dim, embedding_dim)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token indices of shape ``[Batch, Seq]``.
        """
        embed = self.embedding(input_ids)
        embed = self.rope(embed)
        embed = self.attn(embed)
        return embed


class RoPE(nn.Module):
    """Rotary Position Embeddings"""
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        """
        Args:
            dim: The feature dimension to rotate (D)
            max_position_embeddings: The maximum sequence length
            base: The base for the inverse frequency
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute the inverse frequency
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq) # [S, D//2]
        
        # Duplicate for the rotate_half logic
        emb = torch.cat((freqs, freqs), dim=-1) # [S, D]
        
        # Register as [1, S, D] for broadcasting across Batch
        self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)

    def forward(self, x):
        # Input x: [Batch, Seq_len, Dim]
        batch, seq_len, dim = x.shape

        # Slice the cache to match current sequence length
        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]

        # Apply rotation
        return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionEmbeddingsForAttention(nn.Module):
    """Rotary Position Embeddings for Grouped Query Attention"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        """
        Args:
            dim: The feature dimension to rotate (D)
            max_position_embeddings: The maximum sequence length
            base: The base for the inverse frequency
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Calculate the inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute the cosine and sine matrices
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # freqs shape: [seq_len, dim // 2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Concatenate to match the head dimension: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin, reshaped for broadcasting: [1, 1, seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x shape: [batch_size, num_heads, seq_len, head_dim]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

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
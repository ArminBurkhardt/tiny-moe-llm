import torch
from torch import nn
import torch.nn.functional as F
from modules.model.embeddings import RotaryPositionEmbeddingsForAttention, apply_rotary_pos_emb

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, max_position_embeddings=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # RoPE instantiation
        self.rotary_emb = RotaryPositionEmbeddingsForAttention(self.head_dim, max_position_embeddings)

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeats the key/value states to match the number of query heads.
        Input shape:  [batch, num_kv_heads, seq_len, head_dim]
        Output shape: [batch, num_heads, seq_len, head_dim]
        """
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        
        # Expand and reshape to duplicate the KV heads
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(self, hidden_states, attention_mask=None, use_causal_mask=True):
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. Reshape for attention math: [batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Get RoPE frequencies and apply them to Q and K
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 4. Repeat K and V to match the number of Q heads
        key_states = self.repeat_kv(key_states, self.num_groups)
        value_states = self.repeat_kv(value_states, self.num_groups)

        # 5. Scaled Dot-Product Attention
        attn_mask = None
        is_causal_arg = False

        if attention_mask is not None:
            # combine causal mask with padding mask
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device).tril()
            padding_mask = attention_mask.to(dtype=torch.bool).view(batch_size, 1, 1, seq_len)
            if use_causal_mask:
                attn_mask = causal_mask.view(1, 1, seq_len, seq_len) & padding_mask
            else:
                attn_mask = padding_mask
        else:
            is_causal_arg = use_causal_mask

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            is_causal=is_causal_arg
        )

        # 6. Reshape back to [batch_size, seq_len, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # 7. Final output projection
        output = self.o_proj(attn_output)

        return output






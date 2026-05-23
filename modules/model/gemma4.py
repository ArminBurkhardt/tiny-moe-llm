import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.model.embeddings import RotaryPositionEmbeddingsFrequency, apply_rotary_pos_emb
from modules.model.utils import EncoderOutput

# adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4
# https://github.com/huggingface/blog/blob/main/gemma4.md#overview-of-capabilities-and-architecture 
# following the dense architecture

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor):
        mean_square = x.pow(2).mean(-1, keepdim=True)
        norm_x = x / torch.sqrt(mean_square + self.eps)
        return (norm_x * self.weight).type_as(x)

class Gemma4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # intermediate size is 4x hidden size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        return self.down_proj(self.act_fn(gate) * up)


class Gemma4TextAttention(nn.Module):
    # implements Grouped Query Attention with separate projection matrices for keys and values, and support for rotary positional embeddings
    def __init__(
        self, 
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.dropout_p = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # repeat KV heads for Grouped Query Attention
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attn_mask=attention_mask, 
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        return self.o_proj(attn_output)

class Gemma4TextDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        ple_size: int | None = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = Gemma4TextAttention(
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.mlp = Gemma4MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )
        self.input_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_scalar = nn.Parameter(torch.ones(1))
        
        if ple_size is not None:
            self.ple_proj = nn.Linear(ple_size, hidden_size, bias=False)
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.post_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        else:
            self.ple_proj = None
            self.gate_proj = None
            self.post_feedforward_layernorm = None
            

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        per_layer_embeddings: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.self_attn(hidden_states, attention_mask, position_embeddings)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # do per layer embeddings
        if self.post_feedforward_layernorm is not None:
            residual = hidden_states
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            ple_emb = self.ple_proj(per_layer_embeddings)
            gate = self.gate_proj(hidden_states)
            hidden_states = residual + (F.sigmoid(gate) * ple_emb)

        return hidden_states * self.layer_scalar

class Gemma4TextModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_hidden_layers: int,
        rms_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        rope_theta: float = 100000.0,
        dropout: float = 0.0,
        per_layer_embeddings_size: int | None = None,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        
        if per_layer_embeddings_size is not None:
            self.ple = nn.Embedding(
                vocab_size, 
                per_layer_embeddings_size * num_hidden_layers, 
                pad_token_id
            )
        else:
            self.ple = None
        
        self.layers = nn.ModuleList([Gemma4TextDecoderLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            dropout=dropout,
            ple_size=per_layer_embeddings_size
        ) for i in range(num_hidden_layers)])
        self.norm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)
        
        self.rotary_emb = RotaryPositionEmbeddingsFrequency(
            dim=head_dim, 
            max_position_embeddings=max_position_embeddings,
            base=rope_theta
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden_states = self.embed_tokens(input_ids)
        
        # scale embeddings by sqrt(hidden_size)
        hidden_states = hidden_states * (self.hidden_size**0.5)
        hidden_states = self.dropout(hidden_states)
        
        position_embeddings = self.rotary_emb(hidden_states, seq_len=input_ids.shape[1])
        
        if self.ple is not None:
            ple_emb = self.ple(input_ids)
            ple_emb = ple_emb.view(
                input_ids.shape[0], 
                input_ids.shape[1], 
                -1, 
                ple_emb.shape[-1] // len(self.layers)
            ).transpose(1, 2)
        else:
            ple_emb = None

        layers_outputs = []
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, 
                attention_mask, 
                position_embeddings, 
                per_layer_embeddings=ple_emb[:, i] if ple_emb is not None else None
            )
            layers_outputs.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        return EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=layers_outputs
        )

class Gemma4ForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        intermediate_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_hidden_layers: int,
        rms_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        rope_theta: float = 100000.0,
        dropout: float = 0.0,
        per_layer_embeddings_size: int | None = None,
    ):
        super().__init__()
        self.model = Gemma4TextModel(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            rms_norm_eps=rms_norm_eps,
            pad_token_id=pad_token_id,
            rope_theta=rope_theta,
            dropout=dropout,
            per_layer_embeddings_size=per_layer_embeddings_size
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden_states = self.model(input_ids, attention_mask).last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits





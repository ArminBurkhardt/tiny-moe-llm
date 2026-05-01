import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.model.embeddings import RotaryPositionEmbeddingsForAttention, apply_rotary_pos_emb
from modules.model.encoder import EncoderOutput

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Gemma uses (x / rms) * (1 + weight)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return x_norm * (1.0 + self.weight)

class Gemma4GEGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # GEGLU activation
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)

class Gemma4MoE(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 128, num_active_experts: int = 8, num_shared_experts: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.num_shared_experts = num_shared_experts
        
        # Shared expert
        self.shared_experts = nn.ModuleList([
            Gemma4GEGLU(hidden_size, intermediate_size) for _ in range(num_shared_experts)
        ])
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            Gemma4GEGLU(hidden_size, intermediate_size) for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Shared expert forward
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)
        
        # Routed experts forward
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_active_experts, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        final_out = torch.zeros_like(x_flat)
        
        # This is a naive implementation; highly optimized ops exist in vLLM/Triton
        for i in range(self.num_active_experts):
            expert_indices = selected_experts[:, i]
            expert_weights = routing_weights[:, i]
            
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    # Compute only for routed tokens
                    selected_x = x_flat[mask]
                    expert_out = self.routed_experts[expert_idx](selected_x)
                    final_out[mask] += expert_out * expert_weights[mask].unsqueeze(-1)
                    
        final_out += shared_out
        
        return final_out.view(batch_size, seq_len, hidden_size)

class Gemma4Attention(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        num_kv_heads: int, 
        head_dim: int,
        max_position_embeddings: int = 256000,
        sliding_window: int = 4096
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # QK-Norm
        self.q_norm = GemmaRMSNorm(num_heads * head_dim)
        self.k_norm = GemmaRMSNorm(num_kv_heads * head_dim)

        # RoPE
        self.rotary_emb = RotaryPositionEmbeddingsForAttention(head_dim, max_position_embeddings)

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(self, hidden_states, attention_mask=None, use_sliding_window=False):
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = self.repeat_kv(key_states, self.num_groups)
        value_states = self.repeat_kv(value_states, self.num_groups)

        # Setup sliding window if applicable
        attn_mask = None
        is_causal_arg = False

        if attention_mask is not None:
            # attention_mask is usually (batch, seq_len) where 1/True is keep, 0/False is ignore.
            # SDPA expects True = KEEP. We need to combine causal (and optionally sliding window) with padding mask.
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device).tril()
            padding_mask = attention_mask.to(dtype=torch.bool).view(batch_size, 1, 1, seq_len)
            attn_mask = causal_mask.view(1, 1, seq_len, seq_len) & padding_mask

            if use_sliding_window and self.sliding_window is not None:
                window_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device)
                window_mask = torch.tril(window_mask)
                window_mask = torch.triu(window_mask, diagonal=-self.sliding_window)
                attn_mask = attn_mask & window_mask.view(1, 1, seq_len, seq_len)
        elif use_sliding_window and self.sliding_window is not None:
            # Create causal sliding window mask
            attn_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device)
            attn_mask = torch.tril(attn_mask)
            attn_mask = torch.triu(attn_mask, diagonal=-self.sliding_window)
        else:
            is_causal_arg = True

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            is_causal=is_causal_arg
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        return self.o_proj(attn_output)

class Gemma4Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_rmsnorm_1 = GemmaRMSNorm(config["hidden_size"])
        self.pre_rmsnorm_2 = GemmaRMSNorm(config["hidden_size"])
        self.post_rmsnorm_1 = GemmaRMSNorm(config["hidden_size"])
        self.post_rmsnorm_2 = GemmaRMSNorm(config["hidden_size"])
        
        self.attn = Gemma4Attention(
            hidden_size=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"],
            max_position_embeddings=config["max_position_embeddings"],
            sliding_window=config["sliding_window"]
        )
        
        self.moe = Gemma4MoE(
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_experts=config["num_experts"],
            num_active_experts=config["num_active_experts"],
            num_shared_experts=config["num_shared_experts"]
        )

    def forward(self, x, attention_mask=None, use_sliding_window=False):
        # Attention path
        residual = x
        x = self.pre_rmsnorm_1(x)
        x = self.attn(x, attention_mask=attention_mask, use_sliding_window=use_sliding_window)
        x = self.post_rmsnorm_1(x)
        x = residual + x
        
        # MoE path
        residual = x
        x = self.pre_rmsnorm_2(x)
        x = self.moe(x)
        x = self.post_rmsnorm_2(x)
        x = residual + x
        
        return x

class Gemma4Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            # Default Gemma 4 26B-A4B Config
            config = {
                "vocab_size": 262144,
                "hidden_size": 2816,
                "intermediate_size": 704,
                "num_hidden_layers": 30,
                "num_attention_heads": 16, # Assume standard heuristics or 16 for 2816 / 128 (head dim)?
                "num_key_value_heads": 8,
                "head_dim": 256,
                "max_position_embeddings": 256000,
                "sliding_window": 4096,
                "num_experts": 128,
                "num_active_experts": 8,
                "num_shared_experts": 1
            }
            # Adjust if heads don't divide
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]
            if config["hidden_size"] % config["num_attention_heads"] != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.config = config
        
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        
        self.layers = nn.ModuleList([
            Gemma4Block(config) for _ in range(config["num_hidden_layers"])
        ])
        
        self.norm = GemmaRMSNorm(config["hidden_size"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # Shared embeddings
        self.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, return_hidden_states=False) -> tuple[torch.Tensor, EncoderOutput | None]:
        hidden_states = self.embed_tokens(input_ids)
        
        # Scaled embeddings following Gemma design
        hidden_states = hidden_states * math.sqrt(self.config["hidden_size"])
        
        if return_hidden_states:
            all_hidden_states = []
        
        # 5:1 local:global sliding window
        for i, layer in enumerate(self.layers):
            # Every 6th layer is global (0, 1, 2, 3, 4 are local; 5 is global)
            use_sliding_window = ((i % 6) != 5)
            hidden_states = layer(hidden_states, attention_mask=attention_mask, use_sliding_window=use_sliding_window)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if return_hidden_states:
            return logits, EncoderOutput(last_hidden_state=hidden_states, hidden_states=tuple(all_hidden_states))
        return logits

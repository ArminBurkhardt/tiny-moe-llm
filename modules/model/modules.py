import torch
from torch import nn



class LinearAttention(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.q = nn.Linear(input_size, output_size)
        self.k = nn.Linear(input_size, output_size)
        self.v = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor, other: torch.Tensor = None) -> torch.Tensor:
        if other is None:
            other = x
        
        Q = self.q(x)
        K = self.k(other)
        V = self.v(other)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.output_size ** 0.5)
        attn_weights = self.softmax(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        return output



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)



    def forward(self, hidden_states: torch.Tensor, other: torch.Tensor = None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        if other is None:
            other = hidden_states

        # 1. Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(other)
        value_states = self.v_proj(other)

        # 2. Reshape for attention math: [batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        attn_output = torch.matmul(query_states, key_states.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_output += attention_mask

        attn_weights = torch.softmax(attn_output, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        # 4. Reshape back and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output
        
        




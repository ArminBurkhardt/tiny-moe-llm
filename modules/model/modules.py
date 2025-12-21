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







import torch
from torch import nn
from modules.model import SolvableLinear, ParameterizedSigmoid


class ExpertModule(nn.Module):
    """An expert module consisting of an invertible linear layer followed by a parameterized sigmoid activation."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = SolvableLinear(input_size, output_size)
        self.activation = ParameterizedSigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x





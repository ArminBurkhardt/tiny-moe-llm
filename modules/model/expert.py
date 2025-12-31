import torch
from torch import nn
from modules.model.linear import SolvableLinear
from modules.model.activations import InvertibleActivation


class ExpertModule(nn.Module):
    """An expert module consisting of an invertible linear layer followed by a parameterized sigmoid activation."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = SolvableLinear(input_size, output_size)
        self.activation = InvertibleActivation()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x

    def solve_for_batch(self, x: torch.Tensor, y: torch.Tensor, l2: float = 1e-4):
        """
        Solves for the linear layer weights such that the expert output approximates y.
        y = activation(linear(x)) => linear(x) = activation_inv(y)
        """
        with torch.no_grad():
            # Invert the activation to get the target for the linear layer
            y_pre_act = self.activation.inverse(y)
        
        # Solve the linear layer
        self.linear.solve_from_batch(x, y_pre_act, l2=l2)






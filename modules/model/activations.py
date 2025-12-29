import torch
from torch import nn
from typing import Callable
from utils import logger, InvertibleModule

class ParameterizedSigmoid:    
    @staticmethod
    def f(a: float, b: float) -> Callable[[torch.Tensor], torch.Tensor]:
        """Creates a parameterized nonlinear function.

        Args:
            a (float): x -> inf, f(x) -> a
            b (float): x -> -inf, f(x) -> -b

        Returns:
            Callable[[float], float]: A function that computes f(x).
        """
        assert b > 0, "Parameter 'b' must be positive."
        def nonlinear_function(x: torch.Tensor) -> torch.Tensor:
            return -(a*torch.exp(x) -1) / (1/-b - torch.exp(x))
        return nonlinear_function


    @staticmethod
    def f_inv(a: float, b: float) -> Callable[[torch.Tensor], torch.Tensor]:
        """Creates the inverse to the parameterized nonlinear function.

        Args:
            a (float): x -> inf, f(x) -> a
            b (float): x -> -inf, f(x) -> -b

        Returns:
            Callable[[float], float]: A function that computes f^{-1}(y).
        """
        assert b > 0, "Parameter 'b' must be positive."
        def nonlinear_function(y: torch.Tensor) -> torch.Tensor:
            assert torch.all((y < a) & (y > -b)), f"Input y must be in the range (-{b}, {a}). Found y with min {y.min().item()} and max {y.max().item()}."
            return torch.log((1 + y/b) / 
                            (a - y))
        return nonlinear_function

class InvertibleActivation(nn.Module, InvertibleModule):
    """An invertible activation function using parameterized sigmoid."""

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(InvertibleActivation, self).__init__()
        self.a = a
        self.b = b
        self.forward_func = ParameterizedSigmoid.f(a, b)
        self.inverse_func = ParameterizedSigmoid.f_inv(a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in [torch.float64]:
            logger.warning("InvertibleActivation received a low precision input. FP32 may lead to extreme numerical inaccuracies downstream.")
        return self.forward_func(x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.dtype not in [torch.float64]:
            logger.warning("InvertibleActivation received a low precision input. Inverse will likely produce an unexpected result. Please use float64 for accurate inversion.")
        return self.inverse_func(y)
    
    def auto_inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.inverse(y)


class InvertibleLeakyReLUActivation(nn.Module, InvertibleModule):
    def __init__(self, negative_slope: float = 0.01):
        super(InvertibleLeakyReLUActivation, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.negative_slope * x)
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.where(y >= 0, y, y / self.negative_slope)
    
    def auto_inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.inverse(y)


class ShiftActivation(nn.Module, InvertibleModule):
    def __init__(self, shift: float, activation: nn.Module = None):
        super(ShiftActivation, self).__init__()
        self.shift = shift
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation is not None:
            x = self.activation(x)
        return x + self.shift
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = y - self.shift
        if self.activation is not None:
            y = self.activation.auto_inverse(y)
        return y
    
    def auto_inverse(self, y: torch.Tensor) -> torch.Tensor:
        return self.inverse(y)


def test_invertible_activation():
    activation = InvertibleActivation(a=1.0, b=1.0)
    x = torch.linspace(-10, 10, steps=100, dtype=torch.float64)
    y = activation(x)
    try:
        x_reconstructed = activation.inverse(y)
    except AssertionError as e:
        raise AssertionError("Input to inverse function is out of valid range.", y) from e
    assert torch.allclose(x, x_reconstructed, atol=1e-5), "InvertibleActivation failed the invertibility test."




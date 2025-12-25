import torch
from torch import nn
from typing import Callable
from utils import logger

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
            assert torch.all((y < a) & (y > -b)), "Input y must be in the range (-b, a)."
            return torch.log((1 + y/b) / 
                            (a - y))
        return nonlinear_function

class InvertibleActivation(nn.Module):
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




def test_invertible_activation():
    activation = InvertibleActivation(a=1.0, b=1.0)
    x = torch.linspace(-10, 10, steps=100, dtype=torch.float64)
    y = activation(x)
    try:
        x_reconstructed = activation.inverse(y)
    except AssertionError as e:
        raise AssertionError("Input to inverse function is out of valid range.", y) from e
    assert torch.allclose(x, x_reconstructed, atol=1e-5), "InvertibleActivation failed the invertibility test."




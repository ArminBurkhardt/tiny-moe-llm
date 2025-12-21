import torch
from typing import Callable


class ParameterizedSigmoid(torch.nn.Module):
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
            return (-a*torch.exp(x) -1) / (1/b - torch.exp(x))
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



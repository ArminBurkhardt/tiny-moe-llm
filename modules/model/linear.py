import torch
from torch import nn


class SolvableLinear(nn.Module):
    """Linear layer that can be solved from a batch via (regularized) least squares.

    Notes:
        - Inversion is only possible when ``input_size == output_size`` and the
          solved weight matrix is full-rank. Otherwise ``inverse`` will raise.
        - ``solve_from_batch`` overwrites the layer weights/bias using a closed-form
          solution (normal equations with L2 regularization) on the provided batch.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self._cached_inv = None  # lazily computed inverse when square and full-rank
        self.grad_enabled = False
        
        
    def enable_grad(self, enabled: bool = True):
        """Enable or disable gradient tracking for this layer's parameters."""
        for param in self.parameters():
            param.requires_grad = enabled
        self.grad_enabled = enabled

    def disable_grad(self):
        """Disable gradient tracking for this layer's parameters."""
        self.enable_grad(False)

    @property
    def is_square(self) -> bool:
        return self.input_size == self.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_enabled:
            return self.linear(x)
        else:
            with torch.no_grad():
                return self.linear(x)

    def solve_from_batch(self, x: torch.Tensor, y: torch.Tensor, l2: float = 1e-4):
        """Solve weights and bias from input/output batches.

        Args:
            x: Tensor of shape (batch, input_size)
            y: Tensor of shape (batch, output_size)
            l2: Small L2 regularization term for stability
        """
        if x.dim() != 2 or y.dim() != 2:
            raise ValueError("x and y must be 2D: (batch, feature)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Batch sizes of x and y must match")
        if x.shape[1] != self.input_size or y.shape[1] != self.output_size:
            raise ValueError("Input/output dimensions do not match layer configuration")

        device = x.device
        dtype = x.dtype
        ones = torch.ones((x.shape[0], 1), device=device, dtype=dtype)
        design = torch.cat([x, ones], dim=1)  # (batch, input_size + 1)

        # Normal equations with Tikhonov regularization for stability
        gram = design.T @ design
        reg = l2 * torch.eye(gram.shape[0], device=device, dtype=dtype)
        rhs = design.T @ y

        try:
            theta = torch.linalg.solve(gram + reg, rhs)  # (input_size+1, output_size)
        except torch.linalg.LinAlgError:
            theta = torch.linalg.pinv(gram + reg) @ rhs

        weight = theta[:-1].T  # (output_size, input_size)
        bias = theta[-1]       # (output_size,)

        with torch.no_grad():
            self.linear.weight.data.copy_(weight)
            self.linear.bias.data.copy_(bias)
            self._cached_inv = None  # reset cached inverse

        return weight, bias

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Compute x from y when the solved weight matrix is invertible and square."""
        if not self.is_square:
            raise ValueError("Inverse is only defined for square layers (input_size == output_size)")

        weight = self.linear.weight
        bias = self.linear.bias

        if self._cached_inv is None:
            try:
                self._cached_inv = torch.linalg.inv(weight)
            except torch.linalg.LinAlgError as exc:
                raise ValueError("Weight matrix is not invertible; cannot compute inverse") from exc

        # y = x @ W^T + b  =>  x = (y - b) @ (W^T)^{-1} = (y - b) @ W^{-T}
        with torch.no_grad():
            x = (y - bias) @ self._cached_inv.T
        return x




    def approx_linear_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Compute approximate x from y using the pseudo-inverse of the weight matrix."""
        weight = self.linear.weight
        bias = self.linear.bias

        weight_pinv = torch.linalg.pinv(weight)

        # y = x @ W^T + b  =>  x ≈ (y - b) @ (W^T)^{+} = (y - b) @ W^{+T}
        with torch.no_grad():
            x_approx = (y - bias) @ weight_pinv.T
        return x_approx








def test_solvable_linear():
    layer = SolvableLinear(3, 3)
    x = torch.randn(100, 3)
    true_weight = torch.tensor([[2.0, -1.0, 0.5],
                                [0.0, 1.5, -0.5],
                                [-1.0, 0.0, 1.0]])
    true_bias = torch.tensor([0.5, -1.0, 2.0])
    y = x @ true_weight.T + true_bias

    layer.solve_from_batch(x, y, l2=1e-4)

    y_pred = layer(x)
    assert torch.allclose(y_pred, y, atol=1e-3), "Forward pass does not match expected output"

    x_recovered = layer.inverse(y)
    assert torch.allclose(x_recovered, x, atol=1e-3), "Inverse pass does not recover input"

    print("SolvableLinear tests passed.")






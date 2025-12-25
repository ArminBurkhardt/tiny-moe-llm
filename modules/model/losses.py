import torch
from torch import nn

class MatrixInvertabilityLoss(nn.Module):
    """Loss that encourages a batch of square matrices to be invertible

    Args:
        epsilon: Small constant added to the diagonal for numerical stability.
    """

    def __init__(self, epsilon: float = 1e-6, method: str = "non_square_pinverse"):
        super().__init__()
        self.epsilon = epsilon
        self.method = method


    def forward(self, matrices: torch.Tensor) -> torch.Tensor:
        """Compute the invertibility loss.

        Args:
            matrices: Tensor of shape [..., N, N] containing square matrices.

        Returns:
            Scalar tensor representing the average invertibility loss.
        """
        if self.method == "determinant":
            return self.determinant_method(matrices)
        elif self.method == "pinverse":
            return self.pinverse_method(matrices)
        elif self.method == "non_square_pinverse":
            return self.non_square_pinverse_method(matrices)
        else:
            raise ValueError(f"Unknown method '{self.method}' for MatrixInvertabilityLoss.")



    def determinant_method(self, matrices: torch.Tensor) -> torch.Tensor:
        """Compute the invertibility loss.
        
        This is done by minimizing the negative log-determinant of each matrix,
        which encourages the determinant to be large in magnitude (away from zero).

        Args:
            matrices: Tensor of shape [..., N, N] containing square matrices.

        Returns:
            Scalar tensor representing the average invertibility loss.
        """
        # Add epsilon to the diagonal for numerical stability
        eye = torch.eye(matrices.size(-1), device=matrices.device, dtype=matrices.dtype)
        matrices_stable = matrices + self.epsilon * eye

        # Compute log-determinant
        sign, logabsdet = torch.linalg.slogdet(matrices_stable)

        # We want to maximize log-determinant, so minimize negative log-determinant
        loss = -logabsdet

        # Average over all matrices in the batch
        return loss.mean()


    def pinverse_method(self, matrices: torch.Tensor) -> torch.Tensor:
        """Compute the invertibility loss using the pseudo-inverse method.

        Args:
            matrices: Tensor of shape [..., N, N] containing square matrices.

        Returns:
            Scalar tensor representing the average invertibility loss.
        """
        # Compute pseudo-inverse
        pinv_matrices = torch.linalg.pinv(matrices)

        # Compute product of matrix and its pseudo-inverse
        identity_approx = torch.matmul(matrices, pinv_matrices)

        # Compute deviation from identity matrix
        eye = torch.eye(matrices.size(-1), device=matrices.device, dtype=matrices.dtype)
        deviation = identity_approx - eye

        # Compute Frobenius norm of the deviation
        loss = torch.norm(deviation, dim=(-2, -1), p='fro')

        # Average over all matrices in the batch
        return loss.mean()

    def non_square_pinverse_method(self, matrices: torch.Tensor) -> torch.Tensor:
        """Compute the invertibility loss for non-square matrices using the pseudo-inverse method.

        Args:
            matrices: Tensor of shape [..., M, N] containing non-square matrices.
        Returns:
            Scalar tensor representing the average invertibility loss.
        """
        # Compute pseudo-inverse
        pinv_matrices = torch.linalg.pinv(matrices)

        # Compute product of matrix and its pseudo-inverse
        identity_approx = torch.matmul(matrices, pinv_matrices)

        # Compute deviation from identity matrix
        eye = torch.eye(identity_approx.size(-1), device=matrices.device, dtype=matrices.dtype)
        deviation = identity_approx - eye

        # Compute Frobenius norm of the deviation
        loss = torch.norm(deviation, dim=(-2, -1), p='fro')

        # Average over all matrices in the batch
        return loss.mean()

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon}, method='{self.method}'"


# https://docs.pytorch.org/docs/stable/generated/torch.linalg.pinv.html#torch.linalg.pinv 


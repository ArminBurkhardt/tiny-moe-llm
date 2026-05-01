import torch 
from torch import nn
from modules.model.linear import InvertibleLinear
from modules.model.activations import InvertibleActivation, ShiftActivation
from utils import logger, FP64, InvertibleModule


class InvertibleLinearAttention(nn.Module, InvertibleModule):
    def __init__(self, input_size, output_size, activation: nn.Module = nn.Softmax(dim=-1), dtype=FP64):
        super(InvertibleLinearAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.q = InvertibleLinear(input_size, output_size, dtype=dtype)
        self.k = InvertibleLinear(input_size, output_size, dtype=dtype) # has .inverse() for square matrices and .approx_linear_inverse() for non-square
        self.v = InvertibleLinear(input_size, output_size, dtype=dtype)
        
        if not hasattr(activation, 'inverse'):
            logger.warning("Provided activation does not have an inverse method. Unable to invert attention weights.")
        
        self.activation = activation
        self._warned = False
        
    def forward(self, x: torch.Tensor, other: torch.Tensor = None) -> torch.Tensor:
        if other is None:
            other = x
        
        Q = self.q(x)
        K = self.k(other)
        V = self.v(other)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.output_size ** 0.5)
        attn_weights = self.activation(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        return output
        
    @property
    def is_square(self) -> bool:
        return self.input_size == self.output_size    
    
    def inverse(self, output: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """
        Attempts to invert the attention mechanism to recover input 'x'.
        
        IMPORTANT constraints for invertibility:
        1. 'other' (Context) must be provided. We cannot invert Self-Attention 
           algebraically because K and V depend on the unknown x.
        2. The activation function must be invertible (standard Softmax is NOT bijective).
        3. Sequence length must match output_size. If seq_len != output_size the
           Q @ K^T contraction discards dimensions and recovery becomes
           least-squares only.
        """
        
        # 1. Reconstruct Key and Value from the known context 'other'
        # We need these to isolate the Query.
        K = self.k(other) # [Batch, Seq_S, Dim], K = other @ W_k + b_k
        V = self.v(other) # [Batch, Seq_S, Dim], V = other @ W_v + b_v

        seq_len = other.size(-2)
        if (seq_len != self.output_size) and not self._warned:
            logger.warning(
                "InvertableLinearAttention: seq_len (%d) != output_size (%d); "
                "inverse will be a least-squares projection, not an exact recovery.",
                seq_len,
                self.output_size,
            )
            self._warned = True
        
        # 2. Solve for Attention Weights (W)
        # Equation: Output = W @ V
        # We need to compute W = Output @ pinv(V)
        # Note: We use pinverse (Pseudo-Inverse) because V might not be square.
        # If V is tall (Seq > Dim), information is lost and recovery is approximate.
        
        calc_dtype = torch.float64 if V.dtype == torch.float64 else torch.float32

        # Try to use solve if square for better precision
        if self.is_square and seq_len == self.output_size:
            # W @ V = Output => V.T @ W.T = Output.T => W.T = solve(V.T, Output.T)
            try:
               attn_weights = torch.linalg.solve(V.transpose(-2, -1).to(calc_dtype), output.transpose(-2, -1).to(calc_dtype)).transpose(-2, -1).to(V.dtype)
            except torch.linalg.LinAlgError:
               V_pinv = torch.linalg.pinv(V.to(calc_dtype)).to(V.dtype) # [Batch, Dim, Seq_S]
               attn_weights = torch.matmul(output, V_pinv) # [Batch, Seq_N, Seq_S]
        else:
            V_pinv = torch.linalg.pinv(V.to(calc_dtype)).to(V.dtype) # [Batch, Dim, Seq_S]
            attn_weights = torch.matmul(output, V_pinv) # [Batch, Seq_N, Seq_S]
        
        # 3. Inverse Activation
        # Equation: W = Activation(Scores) -> Scores = Activation_Inv(W)
        if hasattr(self.activation, 'inverse'):
            attn_scores = self.activation.inverse(attn_weights)
        else:
            # Fallback/Error if activation is standard softmax (which has no inverse)
            raise RuntimeError("Activation function has no .inverse() method. Cannot invert attention weights.")

        # 4. Solve for Query (Q)
        # Equation: Scores = (Q @ K.T) / sqrt(d)
        # Scores * sqrt(d) = Q @ K.T
        # Q = (Scores * sqrt(d)) @ pinv(K.T)
        scale = self.output_size ** 0.5
        scaled_scores = attn_scores * scale
        
        K_T = K.transpose(-2, -1) # [Batch, Dim, Seq_S]
        
        if self.is_square and seq_len == self.output_size:
            # Q @ K.T = scaled_scores => K @ Q.T = scaled_scores.T => Q.T = solve(K, scaled_scores.T)
            try:
                Q = torch.linalg.solve(K.to(calc_dtype), scaled_scores.transpose(-2, -1).to(calc_dtype)).transpose(-2, -1).to(K.dtype)
            except torch.linalg.LinAlgError:
                K_T_pinv = torch.linalg.pinv(K_T.to(calc_dtype)).to(K.dtype) # [Batch, Seq_S, Dim]
                Q = torch.matmul(scaled_scores, K_T_pinv) # [Batch, Seq_N, Dim]
        else:
            K_T_pinv = torch.linalg.pinv(K_T.to(calc_dtype)).to(K.dtype) # [Batch, Seq_S, Dim]
            Q = torch.matmul(scaled_scores, K_T_pinv) # [Batch, Seq_N, Dim]
        
        # 5. Recover x from Q
        # Equation: Q = x @ W_q + bias
        if self.is_square:
            x = self.q.inverse(Q)
        else:
            x = self.q.approx_linear_inverse(Q)
             
        return x

    def auto_inverse(self, output: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        return self.inverse(output, other)


def test_invertible_linear_attention():
    batch_size = 2
    seq_len = 8
    input_size = 8
    output_size = 8
    
    shifted_activation = ShiftActivation(shift=0.1, activation=InvertibleActivation(a=0.9, b=0.1))

    attention = InvertibleLinearAttention(input_size, output_size, activation=shifted_activation, dtype=FP64)

    torch.manual_seed(0)

    x = torch.randn(batch_size, seq_len, input_size, dtype=FP64)
    other = torch.randn(batch_size, seq_len, input_size, dtype=FP64)

    output = attention(x, other)
    recovered_x = attention.inverse(output, other)

    diff = torch.abs(x - recovered_x)
    #logger.info("Original x: %s", x)
    #logger.info("Recovered x: %s", recovered_x)
    logger.info("Difference (mean/ max): %s / %s", diff.mean().item(), diff.max().item())
    assert torch.allclose(x, recovered_x, atol=1e-6), "Recovered x diverges; check invertibility constraints"
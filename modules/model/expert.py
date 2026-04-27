import torch
from torch import nn
from modules.model.linear import SolvableLinear
from utils import SolvableModule
from modules.model.activations import InvertibleActivation
from modules.model.embeddings import PerLayerEmbedding


class ExpertModule(nn.Module, SolvableModule):
    """An expert module consisting of an invertible linear layer followed by a parameterized sigmoid activation."""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = SolvableLinear(input_size, output_size)
        self.activation = InvertibleActivation()
        self.completed = False
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return x

    def solve_from_batch(self, x: torch.Tensor, y: torch.Tensor, l2: float = 1e-5, **kwargs):
        """
        Solves for the linear layer weights such that the expert output approximates y.
        y = activation(linear(x)) => linear(x) = activation_inv(y)
        """
        with torch.no_grad():
            # Invert the activation to get the target for the linear layer
            y_pre_act = self.activation.inverse(y)
        
        # Solve the linear layer
        self.linear.auto_solve(x, y_pre_act, l2=l2)
        self.completed = True
        
    def consolidate(self, force: bool = False, disable_grad: bool = True, dtype=torch.float32):
        """Consolidate completed expert by converting to a **non-invertable** module. Frees up memory and allows for more efficient inference. 
        
        If `force` is `True`, consolidates even if not completed."""
        
        # TODO: use in training to convert fp64 -> fp32 after solving, to save memory and speed up inference. can still have grad enabled
        
        if self.completed or force:
            # Create a new linear layer with the same weights but no gradient tracking
            new_linear = nn.Linear(self.linear.input_size, self.linear.output_size, bias=self.linear.bias is not None)
            with torch.no_grad():
                new_linear.weight.copy_(self.linear.linear.weight.to(dtype))
                if self.linear.bias is not None:
                    new_linear.bias.copy_(self.linear.linear.bias.to(dtype))
            
            self.linear = new_linear
            
            self.enable_grad(enabled=not disable_grad)
            
    def enable_grad(self, enabled = False):
        """Enable gradient tracking for this expert's parameters."""
        if hasattr(self.linear, 'disable_grad'):
            self.linear.enable_grad(enabled)
        else:
            for param in self.linear.parameters():
                param.requires_grad = enabled
                
    def disable_grad(self):
        """Disable gradient tracking for this expert's parameters."""
        self.enable_grad(False)
        
    def reset(self):
        """Reset all parameters"""
        self.linear.reset()
        self.completed = False

class ExpertModuleWithSkip(ExpertModule):
    """An expert module with a skip connection, pre-LayerNorm, and dropout.

    The forward pass applies pre-normalization before the linear transformation,
    then adds dropout to the activated output before the residual addition:

        output = x + dropout(activation(linear(norm(x))))

    This follows the pre-norm residual style used in modern transformer architectures
    and improves training stability.
    """

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        super().__init__(input_size, output_size)
        # Ensure the linear layer maps input to output size for the skip connection
        assert input_size == output_size, "Input and output sizes must match for skip connection."
        # Pre-norm stabilises the distribution entering the linear layer
        self.norm = nn.LayerNorm(input_size)
        # Dropout regularises the expert's contribution before it is added back
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_norm = self.norm(x)
        x_linear = self.linear(x_norm)
        x_activated = self.activation(x_linear)
        return x + self.dropout(x_activated)  # Skip connection adds input to regularised output

    def solve_from_batch(self, x: torch.Tensor, y: torch.Tensor, l2: float = 1e-5, **kwargs):
        """Solves for the linear layer weights such that the expert output approximates y.

        Accounts for the pre-norm layer:
          y = x + activation(linear(norm(x)))
          => linear(norm(x)) = activation_inv(y - x)
        Dropout is excluded from the closed-form solve (equivalent to p=0 during solving).
        """
        with torch.no_grad():
            x_norm = self.norm(x)
            y_pre_act = self.activation.inverse(y - x)

        self.linear.auto_solve(x_norm, y_pre_act, l2=l2)
        self.completed = True
        
    def reset(self):
        """Reset all parameters, including the LayerNorm."""
        super().reset()
        self.norm.reset_parameters()


class ExpertModuleWithSkipAndEmbedding(ExpertModuleWithSkip):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1, num_embeddings: int = None):
        super().__init__(input_size, output_size, dropout)
        self.embedding = PerLayerEmbedding(num_embeddings=num_embeddings, embedding_dim=input_size)
        
    def forward(self, x: torch.Tensor, input_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape ``[Batch, Seq, InputSize]``.
            input_ids: Token indices of shape ``[Batch, Seq]``.
        """
        embed = self.embedding(input_ids)
        x_norm = self.norm(x + embed)
        x_linear = self.linear(x_norm)
        x_activated = self.activation(x_linear)
        return x + self.dropout(x_activated)

    def solve_from_batch(self, x: torch.Tensor, y: torch.Tensor, input_ids: torch.Tensor, l2: float = 1e-5, **kwargs):
        """Solves for the linear layer weights such that the expert output approximates y.

        Accounts for the pre-norm layer:
          y = x + activation(linear(norm(x)))
          => linear(norm(x)) = activation_inv(y - x)
        Dropout is excluded from the closed-form solve (equivalent to p=0 during solving).
        """
        with torch.no_grad():
            embed = self.embedding(input_ids)
            x_norm = self.norm(x + embed)
            y_pre_act = self.activation.inverse(y - x)

        self.linear.auto_solve(x_norm, y_pre_act, l2=l2)
        self.completed = True
        
    def reset(self):
        """Reset all parameters, including the LayerNorm and embedding."""
        super().reset()
        self.embedding.reset()



class SelfAttentionExpert(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_size)
        self.attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_norm = self.norm(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        return x + self.dropout(attn_output)
    
class CrossAttentionExpert(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_size)
        self.attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        attn_output, _ = self.attn(x_norm, context, context)
        return x + self.dropout(attn_output)



import torch 
import math
from torch import nn
from modules.model.linear import InvertibleLinear, GroupedInvertibleLinear
from modules.model.invertible_modules import InvertibleLinearAttention
from modules.model.activations import InvertibleActivation, InvertibleLeakyReLUActivation, ShiftActivation
from utils import logger, FP64


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dtype=FP64):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        shifted_activation = ShiftActivation(shift=0.1, activation=InvertibleActivation(a=0.9, b=0.1))
        
        self.layers = nn.ModuleList([
            InvertibleLinear(hidden_size, hidden_size, dtype=dtype),
            InvertibleLeakyReLUActivation(),
            InvertibleLinear(hidden_size, hidden_size, dtype=dtype),
            InvertibleActivation(a=1, b=1),
            InvertibleLinearAttention(hidden_size, hidden_size, activation=shifted_activation, dtype=dtype),
            InvertibleActivation(a=1, b=1)
        ])
        
        assert output_size >= hidden_size, "Output size must be greater than or equal to hidden size for grouped invertible linear layer."
        self.lm_head = GroupedInvertibleLinear(
            input_size=hidden_size, 
            output_size=output_size, 
            num_groups=math.gcd(hidden_size, output_size), 
            dtype=dtype
        )


    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if isinstance(layer, InvertibleLinearAttention):
                x = layer(x, other=context)
            else:
                x = layer(x)
            #print(f"After layer {i} ({layer.__class__.__name__}): range {x.min().item()} to {x.max().item()}")
        return x
    
    def inverse(self, output: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        y = output
        for i, layer in enumerate(reversed(self.layers)):
            if isinstance(layer, InvertibleLinearAttention):
                y = layer.inverse(y, other=context)
            else:
                y = layer.auto_inverse(y)
            #print(f"After inverse layer {len(self.layers)-1 - i} ({layer.__class__.__name__}): range {y.min().item()} to {y.max().item()}")
        return y


def test_decoder():
    batch_size = 2
    seq_len = 8
    hidden_size = 8
    output_size = 16         # eeeeeeehhhhhhhhhhh, this makes it very situationally invertible

    decoder = Decoder(hidden_size, output_size)
    
    errors = []
    
    torch.manual_seed(0)
    
    for i in range(64):        
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=FP64)
        context = torch.randn(batch_size, seq_len, hidden_size, dtype=FP64)

        output = decoder(x, context)
        reconstructed_x = decoder.inverse(output, context)

        assert torch.allclose(x, reconstructed_x, atol=1e-3), "Decoder inversion failed!"
        errors.append(torch.abs(x - reconstructed_x).max().item())
    
    logger.info("Decoder test passed successfully, with inversion error of %s", max(errors))






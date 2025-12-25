import torch 
from torch import nn
from modules.model.linear import SolvableLinear


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = SolvableLinear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden_states):
        outputs = self.linear(hidden_states)
        outputs = self.softmax(outputs)
        return outputs

    def inverse(self, outputs):
        # Inverse operation to map outputs back to hidden states
        pseudo_hidden = self.linear.approx_linear_inverse(outputs)
        return pseudo_hidden


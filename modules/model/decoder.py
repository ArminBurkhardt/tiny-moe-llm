import torch 
from torch import nn


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden_states):
        outputs = self.linear(hidden_states)
        outputs = self.softmax(outputs)
        return outputs

    def inverse(self, outputs):
        # Inverse operation to map outputs back to hidden states
        # This is a placeholder implementation and may not be accurate
        pseudo_hidden = torch.matmul(outputs, torch.pinverse(self.linear.weight))
        return pseudo_hidden


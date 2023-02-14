import torch
import torch.nn as nn


class RNN(nn.Module):
    # what's and the difference of 'input_size', hidden_size', 'output_size'?
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # why there 'dim=1'?
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # so hidden on have two dimensions?
        return torch.zeros(1, self.hidden_size)

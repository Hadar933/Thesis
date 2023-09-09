from typing import Literal

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, type: Literal['lstm', 'gru'],
                 input_dim: int, output_dim: int,
                 hidden_dim: int = 16, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False):
        """
        an RNN implementation from which we extract the last output and propagated it via a fully connected layer
        :param type: the rnn model of interest
        :param input_dim: the number of input features
        :param hidden_dim: rnn hidden size
        :param num_layers: number of stacked rnn blocks
        :param output_dim: the output of the fully connected layer
        :param dropout: dropout percentage in [0,1], to be applied between RNN blocks (relevant for num_layers > 1)
        :param bidirectional: iff True
        """
        super(RNN, self).__init__()
        self.name = type.upper()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_dim = output_dim
        self.rnn = getattr(nn, self.name)(input_dim, hidden_dim, num_layers, dropout=dropout,
                                             bidirectional=bidirectional,
                                             batch_first=True)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.unsqueeze(1)


if __name__ == '__main__':
    net = RNN('gru', 10, 11, 2, 3, 0.1, True)
    x = torch.rand(64, 20, 10)
    y = net(x)
    z = 'blabla'

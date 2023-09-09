from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, history_size: int, output_dim: int, hidden_dims_list: List[int]):
        """
        an MLP implementation that performs (B,H,F) -> (B,H*F) -> MLP -> (B,Out_dim)
        :param input_dim: the number of features in the input, required to calculate the MLP input dim
        :param history_size: feature window, essentially, required to calculate the MLP input dim
        :param output_dim: output dim for multi-target prediction
        :param hidden_dims_list: the dimensions of the hidden layers only (without input and output)
        """
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_dim * history_size, hidden_dims_list[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims_list) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims_list[i], hidden_dims_list[i + 1]))
            self.hidden_layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_dims_list[-1], output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

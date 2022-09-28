import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from ctgan.synthesizers.ctgan import Discriminator as CTGANDiscriminator


class _Layer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0, act: str = 'relu'):
        super().__init__()
        activations = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU
        }
        self.act = activations[act]()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return self.linear(self.dropout(self.act(x)))


class MLP(nn.Module):
    """Fundamental MLP structure."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[Tuple[int, ...]] = (100, 100),
                 dropout: float = 0, act: str = 'relu'):
        """
        **Args**:

        - `in_dim` (`int`): Input dimension.
        - `out_dim` (`int`): Output dimension.
        - `hidden_dim` (`Optional[Tuple[int, ...]]`): Hidden dimensions. Default is (100, 100).
        - `dropout` (`float`): Dropout rate. Default is 0.
        - `act` (`str`): Activation function. Can be "relu", "sigmoid", "tanh", "gelu", "leaky_relu".
        """
        super().__init__()
        input_dim = (in_dim,) + hidden_dim
        output_dim = hidden_dim + (out_dim,)
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(input_dim[0], output_dim[0]))
        for i, o in zip(input_dim[1:], output_dim[1:]):
            self.hidden.append(_Layer(i, o, dropout, act))

    def forward(self, x: Tensor) -> Tensor:
        """"""
        for layer in self.hidden:
            x = layer(x)
        return x


class Discriminator(CTGANDiscriminator):
    def forward(self, input_: torch.Tensor):
        x, y = input_.size()
        size = math.ceil(x / self.pac) * self.pac
        placeholder = torch.zeros(size, y).to(input_.device)
        placeholder[:x, :] = input_
        res = super().forward(placeholder)
        return res[:x]

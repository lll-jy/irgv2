import math
from typing import Tuple, Optional, Union, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ctgan.synthesizers.ctgan import Discriminator as CTGANDiscriminator


class _Layer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, act: str = 'relu'):
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
                 dropout: float = 0.1, act: str = 'relu'):
        """
        **Args**:

        - `in_dim` (`int`): Input dimension.
        - `out_dim` (`int`): Output dimension.
        - `hidden_dim` (`Optional[Tuple[int, ...]]`): Hidden dimensions. Default is (100, 100).
        - `dropout` (`float`): Dropout rate. Default is 0.1.
        - `act` (`str`): Activation function. Can be "relu", "sigmoid", "tanh", "gelu", "leaky_relu".
        """
        super().__init__()
        input_dim = (in_dim,) + tuple(hidden_dim)
        output_dim = tuple(hidden_dim) + (out_dim,)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_: torch.Tensor):
        x, y = input_.size()
        size = math.ceil(x / self.pac) * self.pac
        placeholder = torch.zeros(size, y).to(input_.device)
        placeholder[:x, :] = input_
        res = super().forward(placeholder)
        res = self.sigmoid(res)
        return res[:x]

    @staticmethod
    def _reshape(tensor, pac):
        x, y = tensor.size()[0], tensor.size()[1:]
        size = math.ceil(x / pac) * pac
        placeholder = torch.zeros(size, *y).to(tensor.device)
        placeholder[:x] = tensor
        return placeholder

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        real_data, fake_data = self._reshape(real_data, pac), self._reshape(fake_data, pac)
        return super().calc_gradient_penalty(real_data, fake_data, device, pac, lambda_)


class LinearAutoEncoder(nn.Module):
    """
    Simple auto-encoder with both encoder and decoder linear, and a sigmoid layer between encoder and decoder.
    """
    def __init__(self, context_dim: int, full_dim: int, encoded_dim: Optional[int] = None):
        """
        **Args**:

        - `context_dime` (`int`): Dimensions of the known part (as context).
        - `full_dim` (`int`): Original data dimension.
        - `encoded_dim` (`Optional[int]`): Encoded data dimension.
          If not provided, this will be 10xceil(ln(context_dim)).
        """
        super().__init__()
        if encoded_dim is None:
            encoded_dim = 10 * math.ceil(np.log(context_dim))
        self.encoder = nn.Linear(context_dim, encoded_dim)
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(encoded_dim, context_dim + full_dim)
        self._encoded_dim = encoded_dim

    @property
    def encoded_dim(self) -> int:
        """Encoded vector dimension."""
        return self._encoded_dim

    def forward(self, x: Tensor, mode: Literal['enc', 'dec', 'recon']) -> Tensor:
        if mode == 'recon':
            return self.decoder(self.sigmoid(self.encoder(x)))
        if mode == 'enc':
            return self.sigmoid(self.encoder(x))
        if mode == 'dec':
            return self.decoder(x)
        raise NotImplementedError(f'Unrecognized mode {mode}.')


class CNNDiscriminator(nn.Module):
    """
    Discriminator for GAN that takes in multiple rows as input and passes through a CNN to predict the realness of data.
    We implement here only a 2-layer CNN.
    """
    def __init__(self, row_width: int, num_samples: int, out_channels: int = 16, intermediate_channels: int = 6,
                 kernel_size: Union[int, Tuple[int, int]] = 5, **kwargs):
        """
        **Args**:

        - `row_width` (`int`): Number of dimensions per row in the table. That is, width of input.
        - `num_samples` (`int`): Number of samples per set of input.
        - `out_channels` (`int`): Output channels of the second convolutional layer. Default is 16.
        - `intermediate_channels` (`int`): Output channels of the first convolutional layer. Default is 6.
        - `kernel_size` (`Union[int, Tuple[int, int]]`): Kernel size of both convolutional layers. Default is 5.
        - `kwargs`: Other arguments to `MLP`.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, intermediate_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(intermediate_channels, out_channels, kernel_size)
        h, w = num_samples // 2 - 2, row_width // 2 - 2
        h, w = h // 2 - 2, w // 2 - 2
        self.fc = MLP(out_channels * h * w, 1, **kwargs)

    def forward(self, x: Tensor):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


class TimeGanNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_dim: int = 40, n_layers: int = 1,
                 activation: str = 'sigmoid', rnn_type: str = 'gru'):
        super().__init__()
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        else:
            raise NotImplementedError(f'RNN {rnn_type} is not implemented.')
        self.fc = nn.Linear(hidden_dim, output_size)

        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'none':
            self.act = nn.Identity()
        else:
            raise NotImplementedError(f'Activation {activation} is not implemented.')

        self._is_lstm = rnn_type == 'lstm'
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        batch_size = x.shape[0]
        if self._is_lstm:
            h0 = torch.zeros(self._n_layers, batch_size, self._hidden_dim, device=x.device, dtype=torch.float32)
            c0 = torch.zeros(self._n_layers, batch_size, self._hidden_dim, device=x.device, dtype=torch.float32)
            hidden = h0, c0
        else:
            hidden = torch.zeros(self._n_layers, batch_size, self._hidden_dim, device=x.device, dtype=torch.float32)
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self._hidden_dim)
        out = self.fc(out)
        out = self.act(out)
        return out, hidden


class SequenceEmbedding(nn.Module):
    """Adapted from https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html"""
    def __init__(self, vocab_size: int, embedding_dim: int = 50, context_size: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embeds = self.embeddings(x).view(x.shape[0], -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)

    def decode(self, emb: Tensor) -> Tensor:
        pass


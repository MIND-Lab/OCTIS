"""PyTorch class for feed-forward inference network."""

from collections import OrderedDict
from torch import nn
import torch
import numpy as np

class InferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, bert_size, output_size, hidden_sizes,
                 activation='gelu', dropout=0.2):
        """
        Initialize InferenceNetwork.

        Args
            bert_size : int, dimension of BERT input
            output_size : int, dimension of output
            hidden_sizes : tuple, length = n_layers
            activation : string, default 'gelu'
            dropout : float, default 0.2
        """
        super(InferenceNetwork, self).__init__()
        assert isinstance(bert_size, int), "input_size must by type int."
        assert isinstance(output_size, int) or isinstance(output_size, np.int64), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu', 'gelu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'tanh'," \
            " 'leakyrelu', 'rrelu', 'elu', 'selu', or 'gelu'."
        assert dropout >= 0, "dropout must be >= 0."

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'rrelu':
            self.activation = nn.RReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()

        self.adapt_bert = nn.Linear(bert_size, hidden_sizes[0])
        self.dropout = nn.Dropout(p=dropout)

        self.hiddens = nn.Sequential(OrderedDict([
            (f"l_{i}", nn.Sequential(nn.Linear(h_in, h_out), self.activation, nn.Dropout(p=dropout)))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.output = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x_bert):
        """Forward pass."""
        x_bert = self.adapt_bert(x_bert)
        x = self.activation(x_bert)
        x = self.dropout(x)
        x = self.hiddens(x)
        out = self.output(x)

        return out

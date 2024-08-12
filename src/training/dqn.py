from typing import Tuple
import torch
from src.environment.board import Board
from src.environment.tile import Tile, TileType
from torch import nn


class DQN(nn.Module):
    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        input_channels: int,
        n_actions: int,
        n_hidden_conv_layers: int = 1,
        n_hidden_conv_channels: int = 32,
    ):
        super(DQN, self).__init__()
        layers = self._conv_block(input_channels, n_hidden_conv_channels)

        # hidden conv layers, input size stays the same
        for _ in range(n_hidden_conv_layers):
            layers.extend(
                self._conv_block(n_hidden_conv_channels, n_hidden_conv_channels)
            )

        # collapse channels with 1x1 conv
        layers.extend(
            self._conv_block(n_hidden_conv_channels, 1, kernel_size=1, padding=0)
        )

        # fully connected layer to predict over actions
        layers.extend(
            [
                nn.Flatten(),
                nn.Linear(n_hidden_conv_channels * n_rows * n_columns, n_actions),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def _conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
        ]

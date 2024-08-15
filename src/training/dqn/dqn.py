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
        kernel_size: int,
        input_channels: int,
        n_actions: int,
        n_hidden_conv_layers: int = 1,
        n_hidden_conv_channels: int = 32,
        hidden_conv_padding: int = 1,
    ):
        super(DQN, self).__init__()
        self.hidden_layers = self._conv_block(
            input_channels, n_hidden_conv_channels, kernel_size, hidden_conv_padding
        )
        # first conv layer may change input size
        n_output_columns = n_columns + 2 * hidden_conv_padding - kernel_size + 1
        n_output_rows = n_rows + 2 * hidden_conv_padding - kernel_size + 1

        # hidden conv layers, input size stays the same
        for _ in range(n_hidden_conv_layers):
            self.hidden_layers.extend(
                self._conv_block(
                    n_hidden_conv_channels,
                    n_hidden_conv_channels,
                    kernel_size,
                    hidden_conv_padding,
                )
            )
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        # collapse channels with 1x1 conv
        self.final_conv_block = nn.Sequential(
            *self._conv_block(n_hidden_conv_channels, 1, kernel_size=1, padding=0)
        )

        # fully connected layer to predict over actions
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_output_rows * n_output_columns, n_actions),
        )

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.final_conv_block(x)
        return self.output_head(x)

    def _conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
        ]

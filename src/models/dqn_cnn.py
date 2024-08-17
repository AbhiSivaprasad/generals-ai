import torch

from torch import nn

from src.models.utils import ConvNeXtBlock, LayerNorm


class DQN(nn.Module):
    """
    The Deep Q-Network model. This model is a convolutional neural network that takes in an image of the game state and outputs a Q-value for each action.
    Input: (batch_size, n_rows, n_cols, input_channels) the grid value, see src/environment/environment.py for more details
    Output: (batch_size, num_actions) the q-values
    """
    def __init__(
        self, 
        input_channels, 
        num_actions, 
        n_rows, 
        n_cols,
        spatial_dropout=0.0,
        encoder_dim=64,
        encoder_depth=2,
        block_dim=64,        
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_dim, kernel_size=1, padding="same"), 
            *[
                nn.Sequential(
                    nn.Conv2d(encoder_dim, encoder_dim, kernel_size=1, padding="same"), 
                    LayerNorm(encoder_dim, eps=1e-6, data_format="channels_first"),
                    nn.Dropout2d(p=spatial_dropout)
                )
                for _ in range(encoder_depth)
            ],
            nn.Conv2d(encoder_dim, block_dim, kernel_size=3, padding="same"),
            LayerNorm(encoder_dim, eps=1e-6, data_format="channels_first"),
        )   
        self.blocks = nn.Sequential(
            ConvNeXtBlock(block_dim),
            nn.Conv2d(block_dim, block_dim // 2, kernel_size=3, padding=1, stride=2),
            nn.Dropout2d(p=spatial_dropout),
            ConvNeXtBlock(block_dim // 2),
            nn.Conv2d(block_dim // 2, block_dim // 4, kernel_size=3, padding=1),
            LayerNorm(block_dim // 4, eps=1e-6, data_format="channels_first"),
        )
        half_ceil = lambda x: (x + 1) // 2
        fc_dim = half_ceil(n_rows) * half_ceil(n_cols) * (block_dim // 4)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_dim, fc_dim),
            nn.Linear(fc_dim, fc_dim),
            nn.GELU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Linear(fc_dim, num_actions)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.fc_layers(x)
        return x
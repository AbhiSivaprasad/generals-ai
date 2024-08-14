import torch

from torch import nn

from src.models.utils import ConvNeXtBlock, ParallelSum


class DQN(nn.Module):
    def __init__(
        self, 
        input_channels, 
        num_actions, 
        n_rows, 
        n_cols,
        spatial_dropout=0.0,
        encoder_dim=64,
        encoder_depth=3,
        block_dim=64,
        n_blocks=1
        
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(input_channels, encoder_dim, kernel_size=1, padding="same"), 
                    nn.GELU(),
                    # nn.Dropout2d(p=spatial_dropout)
                )
                for _ in range(encoder_depth)
            ],
            nn.Conv2d(encoder_dim, block_dim, kernel_size=3, padding="same")
        )   
        self.blocks = ParallelSum(
            *[ConvNeXtBlock(block_dim) for _ in range(n_blocks)],
            
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_rows * n_cols * block_dim, n_rows * n_cols * block_dim),
            nn.GELU(),
            nn.Linear(n_rows * n_cols * block_dim, num_actions)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.fc_layers(x)
        return x
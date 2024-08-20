from pathlib import Path
import torch
from torch import nn


class FCNetwork(nn.Module):
    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        n_actions: int,
        n_input_channels: int,
        n_hidden_dim: int = 128,
        n_hidden_layers: int = 3,
    ):
        super(FCNetwork, self).__init__()
        # store model architecture
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_hidden_dim = n_hidden_dim
        self.n_hidden_layers = n_hidden_layers

        self.input_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_rows * n_columns * n_input_channels, n_hidden_dim),
            nn.ReLU(),
        )
        self.hidden_layers = []
        for _ in range(n_hidden_layers):
            self.hidden_layers.extend(
                [
                    nn.Linear(n_hidden_dim, n_hidden_dim),
                    nn.ReLU(),
                ]
            )
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        self.output_head = nn.Linear(n_hidden_dim, n_actions)

    def forward(self, x):
        x = self.input_layers(x)
        x = self.hidden_layers(x)
        return self.output_head(x)

    def save_checkpoint(self, checkpoint_dir: Path, step: int):
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pth"

        checkpoint = {
            "step": step,
            "model_state_dict": self.state_dict(),
            "model_architecture": {
                "n_rows": self.n_rows,
                "n_columns": self.n_columns,
                "n_actions": self.n_actions,
                "n_input_channels": self.n_input_channels,
                "n_hidden_dim": self.n_hidden_dim,
                "n_hidden_layers": self.n_hidden_layers,
                "n_actions": self.n_actions,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path, device="cpu"):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        architecture = checkpoint["model_architecture"]
        model = cls(**architecture)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model

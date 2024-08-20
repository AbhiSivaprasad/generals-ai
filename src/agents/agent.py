from abc import ABC, abstractmethod
from typing import Optional

import torch


class Agent(ABC):
    def __init__(self, player_index) -> None:
        self.player_index = player_index

    @abstractmethod
    def move(self, state: torch.Tensor, env) -> int:
        """
        An agent returns an integer index representing the action it wants to take
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the agent's state for a new game
        """
        pass

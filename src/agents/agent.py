from abc import ABC, abstractmethod
from typing import Optional

import torch


class Agent(ABC):
    def __init__(self, player_index) -> None:
        self.player_index = player_index

    @abstractmethod
    def move(self, state: torch.Tensor, env) -> Optional[int]:
        """
        An agent returns None if there are no legal moves or if the agent wishes to wait
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the agent's state for a new game
        """
        pass

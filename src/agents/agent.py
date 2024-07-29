from abc import ABC, abstractmethod
from typing import Optional

from src.environment.action import Action


class Agent(ABC):
    def __init__(self, player_index) -> None:
        self.player_index = player_index

    @abstractmethod
    def move(self, state) -> Optional[Action]:
        """
        An agent returns None if there are no legal moves or if the agent wishes to wait
        """
        pass

from abc import ABC, abstractmethod
from typing import Optional

from src.environment.action import Action
from src.environment.gamestate import GameState


class Agent(ABC):
    player_index: int
    
    def __init__(self, player_index) -> None:
        self.player_index = player_index

    @abstractmethod
    def move(self, state: GameState) -> Optional[Action]:
        """
        An agent returns None if there are no legal moves or if the agent wishes to wait
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset method. Good place to reset random seeds, agentstate variables, etc.
        """
        pass
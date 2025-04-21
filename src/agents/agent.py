from abc import ABC, abstractmethod
from typing import Optional

from src.environment.action import Action
from src.environment.gamestate import GameState


class Agent(object):
    player_index: int
    
    def __init__(self, player_index, *args, **kwargs) -> None:
        self.player_index = player_index

    def move(self, state: GameState) -> Optional[Action]:
        """
        An agent returns an integer index representing the action it wants to take
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the agent's state for a new game
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs) -> None:
        """
        Reset method. Good place to reset random seeds, agentstate variables, etc.
        """
        raise NotImplementedError
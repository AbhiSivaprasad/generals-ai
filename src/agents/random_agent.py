import numpy as np

from src.agents.agent import Agent
from src.environment.action import Action, Direction
from src.environment.gamestate import GameState


class RandomAgent(Agent):
    rng: np.random.Generator
    
    def __init__(self, player_index, seed=0, *args, **kwargs):
        self.reset(seed)
        super().__init__(player_index, *args, **kwargs)
    
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        
    def move(self, gamestate: GameState) -> Action:
        valid_actions = gamestate.board.get_valid_actions(self.player_index)
        if len(valid_actions) > 0:
            ind = self.rng.integers(len(valid_actions), endpoint=True)
            return valid_actions[ind] if ind < len(valid_actions) else None
        else:
            return None

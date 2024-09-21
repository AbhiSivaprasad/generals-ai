import numpy as np

import torch

from src.agents.agent import Agent
from src.environment.action import Action
from src.environment.gamestate import GameState


class RandomAgent(Agent):
    rng: np.random.Generator
    
    def __init__(self, player_index, seed, *args, **kwargs):
        super().__init__(player_index, *args, **kwargs)
        self.reset(seed)
    
    def reset(self, seed, *args, **kwargs) -> None:
        self.rng = np.random.default_rng(seed)
        
    def move(self, gamestate: GameState) -> Action:
        valid_actions = gamestate.board.get_valid_actions(self.player_index)
        if len(valid_actions) > 0:
            ind = self.rng.integers(len(valid_actions), endpoint=True)
            return valid_actions[ind] if ind < len(valid_actions) else None
        else:
            action = Action(do_nothing=True).to_index(env.unwrapped.board_x_size)
        info = {"best_action": action}
        return action, info

    def reset(self, seed=None):
        pass

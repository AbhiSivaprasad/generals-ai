from typing import Optional

import numpy as np
from src.agents.agent import Agent
from src.agents.utils.agent_wrapper import AgentWrapper
from src.environment.action import Action
from src.environment.gamestate import GameState


class EpsilonRandomAgent(AgentWrapper):
    rng: np.random.Generator
    epsilon: float
    
    def __init__(self, agent: Agent, epsilon: float, seed: int, *args, **kwargs):
        super().__init__(agent, *args, **kwargs)
        self.epsilon = epsilon
        self.reset(seed)
        
    def move(self, state: GameState) -> Optional[Action]:
        move = self.agent.move(state)
        if self.rng.random() < self.epsilon:
            valid_actions = state.board.get_valid_actions(self.player_index)
            if len(valid_actions) > 0:
                return self.rng.choice(valid_actions)
            else:
                return None
        return move

    def reset(self, *args, seed = None, **kwargs) -> None:
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

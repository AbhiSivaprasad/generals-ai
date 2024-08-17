from src.agents.agent import Agent
from src.environment.action import Action
from src.environment.gamestate import GameState
import traceback
import sys

class AgentWrapper(Agent):
    agent: Agent = None
    
    def __init__(self, agent, *args, **kwargs):
        super().__init__(agent.player_index, *args, **kwargs)
        self.agent = agent

    def __getattr__(self, name):
        if self.agent is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.agent, name)
    
    def move(self, state: GameState) -> Action | None:
        return self.agent.move(state)
    
    def reset(self, *args, **kwargs) -> None:
        self.agent.reset(*args, **kwargs)
    
    @property
    def unwrapped(self):
        if self.agent is None:
            return None
        if not hasattr(self.agent, "unwrapped"):
            return self.agent
        return self.agent.unwrapped
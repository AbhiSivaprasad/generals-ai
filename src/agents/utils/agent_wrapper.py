from typing import Optional
from src.agents.agent import Agent
from src.environment.action import Action
from src.environment.gamestate import GameState
import traceback
import sys

class AgentWrapper(Agent):
    agent: Optional[Agent] = None # Optional because wrappers can have their own behavior too.
    
    def __init__(self, agent, *args, **kwargs):
        self.agent = agent

    def __getattr__(self, name):
        if self.agent is None:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.agent, name)
    
    def move(self, state: GameState) -> Action | None:
        return self.agent.move(state)
    
    def reset(self, *args, **kwargs) -> None:
        if self.agent is not None:
            self.agent.reset(*args, **kwargs)
    
    @property
    def unwrapped(self):
        if self.agent is None:
            return None
        if not hasattr(self.agent, "unwrapped"):
            return self.agent
        return self.agent.unwrapped
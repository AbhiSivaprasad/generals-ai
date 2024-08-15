from src.agents.agent import Agent
from src.environment.action import Action
from src.environment.gamestate import GameState

class AgentWrapper(Agent):
    agent: Agent
    
    def __init__(self, agent):
        self.agent = agent

    def __getattr__(self, name):
        return getattr(self.agent, name)
    
    def move(self, state: GameState) -> Action | None:
        return self.agent.move(state)
    
    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self.agent.reset(*args, **kwargs)
    
    @property
    def unwrapped(self):
        if self.agent is None:
            return None
        if not hasattr(self.agent, "unwrapped"):
            return self.agent
        return self.agent.unwrapped
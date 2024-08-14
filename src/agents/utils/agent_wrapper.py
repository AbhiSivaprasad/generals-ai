from src.agents.agent import Agent

class AgentWrapper(Agent):
    agent: Agent
    
    def __init__(self, agent):
        self.agent = agent

    def __getattr__(self, name):
        return getattr(self.agent, name)
    
    @property
    def unwrapped(self):
        if self.agent is None:
            return None
        if not hasattr(self.agent, "unwrapped"):
            return self.agent
        return self.agent.unwrapped
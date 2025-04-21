from typing import Callable, Dict, Tuple

from src.agents.agent import Agent
from src.agents.utils.agent_wrapper import AgentWrapper
from src.environment.gamestate import ObsType
from src.utils.replay_buffer import Experience


class ObservationReceivingInterface(object):
    def observe(self, experience: Experience) -> None:
        raise NotImplementedError
    
class ObservationReceiving(AgentWrapper, ObservationReceivingInterface):
    handler: Callable[[Experience], None]
    
    def __init__(self, agent: Agent, observation_handler: Callable[[Experience], None], *args, **kwargs):
        super().__init__(agent, *args, **kwargs)
        self.handler = observation_handler
        
    def observe(self, experience: Experience) -> None:
        self.handler(experience)
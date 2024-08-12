from typing import Optional
from src.agents.agent import Agent
from src.environment.action import Action
from src.environment.gamestate import GameState


class GymAgent(Agent):
    """
    Wrapper pass-thru class to interface the `Agent` class with the `gymnasium` framework's environments and agents.
    Allows the gymnasium environment to work with a `step(a: Action) -> s': State` interface rather than an environment-driven `step(s: State) -> a: Action` interface.
    """
    action: Action = None
    
    def __init__(self):
        self.action = None

    def set_action(self, action):
        self.action = action

    def move(self, state: GameState) -> Optional[Action]:
        return self.action

    def reset(self):
        self.action = None
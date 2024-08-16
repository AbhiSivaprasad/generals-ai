import math
import torch
import random
from src.agents.agent import Agent
import numpy as np

from src.environment.environment import GeneralsEnvironment


class CuriousGeorgeAgent(Agent):
    def __init__(
        self,
        policy_net: torch.nn.Module,
        eps_start: float = None,
        eps_end: float = None,
        eps_decay: float = None,
        train: bool = True,
    ):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.train = train
        self.steps = 0
        self.policy_net = policy_net

    def move(self, state: torch.Tensor, env: GeneralsEnvironment):
        sample = random.random()
        self.steps += 1
        if self.train:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1.0 * self.steps / self.eps_decay
            )
        else:
            eps_threshold = 0
        if sample > eps_threshold:
            with torch.no_grad():
                # argmax returns the indices of the maximum values along the specified dimension
                return self.policy_net(state).argmax(dim=1).squeeze()
        else:
            return torch.tensor(
                [[list(env.action_spaces.values())[0].sample()]],
                device=state.device,
                dtype=torch.long,
            ).squeeze()

    def reset(self):
        self.steps = 0

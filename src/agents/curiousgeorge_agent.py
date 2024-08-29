import math
import torch
import random
from src.agents.agent import Agent
import numpy as np

from src.environment.environment import GeneralsEnvironment
from src.utils.scheduler import ConstantHyperParameterSchedule, HyperParameterSchedule


class CuriousGeorgeAgent(Agent):
    def __init__(
        self,
        policy_net: torch.nn.Module,
        epsilon_schedule: HyperParameterSchedule = ConstantHyperParameterSchedule(0),
    ):
        self.steps = 0
        self.policy_net = policy_net
        self.epsilon_schedule = epsilon_schedule
        self.epsilon = self.epsilon_schedule.get(0)

    def move(self, state: torch.Tensor, env: GeneralsEnvironment):
        sample = random.random()
        self.steps += 1
        self.epsilon = self.epsilon_schedule.get(self.steps)

        # argmax returns the indices of the maximum values along the specified dimension
        best_action = self.policy_net(state.unsqueeze(0)).argmax(dim=1).squeeze().item()
        info = {"best_action": best_action}
        if sample > self.epsilon:
            with torch.no_grad():
                return best_action, info
        else:
            return (
                torch.tensor(
                    [[list(env.action_spaces.values())[0].sample()]],
                    device=state.device,
                    dtype=torch.long,
                )
                .squeeze()
                .item()
            ), info

    def reset(self):
        pass

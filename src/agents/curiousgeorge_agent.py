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
        episilon_schedule: HyperParameterSchedule = ConstantHyperParameterSchedule(0),
    ):
        self.steps = 0
        self.policy_net = policy_net
        self.epsion_schedule = episilon_schedule

    def move(self, state: torch.Tensor, env: GeneralsEnvironment):
        sample = random.random()
        self.steps += 1
        eps_threshold = self.episilon_schedule.get(self.steps)
        if sample > eps_threshold:
            with torch.no_grad():
                # argmax returns the indices of the maximum values along the specified dimension
                return self.policy_net(state).argmax(dim=1).squeeze().item()
        else:
            return (
                torch.tensor(
                    [[list(env.action_spaces.values())[0].sample()]],
                    device=state.device,
                    dtype=torch.long,
                )
                .squeeze()
                .item()
            )

    def reset(self):
        pass

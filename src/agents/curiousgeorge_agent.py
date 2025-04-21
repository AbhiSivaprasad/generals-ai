import math
import torch
import random
from src.agents.agent import Agent
import numpy as np

from src.environment.action import Action, Direction
from src.environment.environment import GeneralsEnvironment
from src.utils.scheduler import ConstantHyperParameterSchedule, HyperParameterSchedule


class CuriousGeorgeAgent(Agent):
    def __init__(
        self,
        player_index: int,
        policy_net: torch.nn.Module,
        epsilon_schedule: HyperParameterSchedule = ConstantHyperParameterSchedule(0),
    ):
        super().__init__(player_index)
        self.steps = 0
        self.policy_net = policy_net
        self.epsilon_schedule = epsilon_schedule
        self.epsilon = self.epsilon_schedule.get(0)

    def move(self, state: torch.Tensor, env: GeneralsEnvironment):
        sample = random.random()
        self.steps += 1
        self.epsilon = self.epsilon_schedule.get(self.steps)
        
        info = {}

        # argmax returns the indices of the maximum values along the specified dimension
        with torch.no_grad():
            best_action = self.policy_net(state.unsqueeze(0)).argmax(dim=1).squeeze().item()
            info["best_action"] = best_action
            
        if sample > self.epsilon:
            info["random"] = 0
            action = best_action
        else:
            info["random"] = 1
            
            board = env.unwrapped.game_master.board
            valid_actions = []
            for i in range(board.num_rows):
                for j in range(board.num_cols):
                    for direction in Direction:
                        action = Action(startx=j, starty=i, direction=direction)
                        if board.is_action_valid(action, self.player_index):
                            valid_actions.append(action)
            if len(valid_actions) > 0:
                action = random.choice(valid_actions).to_index(env.unwrapped.board_x_size)
            else:
                action = 0
                
                
        return action, info

    def reset(self):
        pass

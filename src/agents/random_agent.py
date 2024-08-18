import random

import torch

from src.agents.agent import Agent
from src.environment.action import Action, Direction
from src.environment.environment import GeneralsEnvironment


class RandomAgent(Agent):
    def move(self, state: torch.Tensor, env: GeneralsEnvironment):
        board = env.unwrapped.game_master.board
        valid_actions = []
        for i in range(board.num_rows):
            for j in range(board.num_cols):
                for direction in Direction:
                    action = Action(startx=i, starty=j, direction=direction)
                    if board.is_action_valid(action, self.player_index):
                        valid_actions.append(action)

        if len(valid_actions) > 0:
            action = random.choice(valid_actions)
            return action.to_index(env.unwrapped.board_x_size)
        else:
            return Action(do_nothing=True).to_index(env.unwrapped.board_x_size)

    def reset(self, seed=None):
        pass

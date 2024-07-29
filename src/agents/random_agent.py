import random

from src.agents.agent import Agent
from src.environment.action import Action, Direction


class RandomAgent(Agent):
    def move(self, board):
        valid_actions = []
        for i in range(board.num_rows):
            for j in range(board.num_cols):
                for direction in Direction:
                    action = Action(startx=i, starty=j, direction=direction)
                    if board.is_action_valid(action, self.player_index):
                        valid_actions.append(action)

        if len(valid_actions) > 0:
            return random.choice(valid_actions)
        else:
            return None

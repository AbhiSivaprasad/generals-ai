from typing import Dict
from pettingzoo import ParallelEnv
from src.environment.environment import GeneralsEnvironment


class ProbeFourEnvironment(GeneralsEnvironment):
    def _get_rewards(self):
        win_loss_reward = self.get_main_rewards()
        win_loss_reward = {
            agent_index: 10.0 * win_loss_reward[agent_index] for agent_index in range(len(self.agents))
        }
        # auxiliary_land_reward = self.get_auxiliary_land_reward()
        auxiliary_legal_move_rewards = self.get_auxiliary_legal_move_reward()
        auxiliary_legal_move_rewards = {
            agent_index: -(1 - int(is_legal)) for agent_index, is_legal in auxiliary_legal_move_rewards.items()
        }
        total_rewards = {
            agent_index:
                win_loss_reward[agent_index] +
                # self.auxiliary_reward_weight * (auxiliary_land_reward[agent_index]) +
                self.auxiliary_reward_weight * (auxiliary_legal_move_rewards[agent_index])
            for agent_index in range(len(self.agents))
        }
        return total_rewards

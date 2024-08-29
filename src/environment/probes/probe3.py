from pettingzoo import ParallelEnv
from src.environment.environment import GeneralsEnvironment


class ProbeThreeEnvironment(GeneralsEnvironment):
    def _get_rewards(self):
        total_rewards = {
            agent_index: self.auxiliary_reward_weight
            * (
                # auxiliary_legal_move_rewards[agent_index]
                self.get_auxiliary_land_reward()[agent_index]
            )
            for agent_index in range(len(self.agents))
        }
        return total_rewards

from pettingzoo import ParallelEnv
from src.environment.environment import GeneralsEnvironment


class ProbeThreeEnvironment(GeneralsEnvironment):
    def _get_rewards(self):
        auxiliary_land_reward = self.get_auxiliary_land_reward()
        total_rewards = {
            agent_index: self.auxiliary_reward_weight
            * (auxiliary_land_reward[agent_index])
            for agent_index in range(len(self.agents))
        }
        print(total_rewards)
        return total_rewards

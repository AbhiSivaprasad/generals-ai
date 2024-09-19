from typing import Dict
from src.environment.environment import GeneralsEnvironment


class ProbeOneEnvironment(GeneralsEnvironment):
    def _get_rewards(self, auxiliary_legal_move_rewards: Dict[int, int]):
        total_rewards = {
            agent_index: self.auxiliary_reward_weight
            * auxiliary_legal_move_rewards[agent_index]
            for agent_index in range(len(self.agents))
        }
        return total_rewards

    def _get_terminations(self):
        return {agent_index: False for agent_index in range(len(self.agents))}

    def _get_truncations(self):
        return {agent_index: True for agent_index in range(len(self.agents))}

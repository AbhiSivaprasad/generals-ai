from src.environment.environment import GeneralsEnvironment


class ProbeZeroEnvironment(GeneralsEnvironment):
    """
    ProbeZeroEnvironment tests a single state with a constant reward = 1
    """

    def _get_rewards(self):
        return {agent_index: 1 for agent_index in range(len(self.agents))}

    def _get_terminations(self):
        return {agent_index: True for agent_index in range(len(self.agents))}

    def _get_truncations(self):
        return {agent_index: False for agent_index in range(len(self.agents))}

from collections import defaultdict
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from typing import Dict, List

from src.agents.agent import Agent
from src.environment.action import Action
from src.environment.board_generator import generate_board_state
from src.environment.game_master import GameMaster
from src.environment.logger import Logger
from src.training.input import (
    convert_state_to_array,
    get_input_channel_dimension_size,
)


class ProbeZeroEnvironment(ParallelEnv):
    """
    ProbeZeroEnvironment tests a single state with a constant reward = 1
    """

    metadata = {"name": "generals_v0"}

    def __init__(
        self,
        agents: List[Agent],
        max_turns: int = 1000,
        board_x_size: int = 3,
        board_y_size: int = 3,
        mountain_probability: float = 0.0,
        city_probability: float = 0.0,
        use_fog_of_war: bool = False,
        auxiliary_reward_weight: float = 0.01,
    ):
        self.agents = agents
        self.max_turns = max_turns
        self.board_x_size = board_x_size
        self.board_y_size = board_y_size
        self.mountain_probability = mountain_probability
        self.city_probability = city_probability
        self.use_fog_of_war = use_fog_of_war
        self.n_step = 0

        # Define action and observation spaces
        self.action_spaces = {
            agent_index: spaces.Discrete(board_x_size * board_y_size * 4)
            for agent_index in range(len(agents))
        }
        self.observation_spaces = {
            agent_index: spaces.Box(
                low=0,
                high=np.inf,
                shape=(
                    get_input_channel_dimension_size(self.use_fog_of_war),
                    board_x_size,
                    board_y_size,
                ),
                dtype=np.int32,
            )
            for agent_index in range(len(agents))
        }

    def reset(self, seed=None, options=None, logger: Logger = None):
        self.n_step = 0
        board = generate_board_state(
            self.board_x_size,
            self.board_y_size,
            mountain_probability=self.mountain_probability,
            city_probability=self.city_probability,
        )
        self.game_master = GameMaster(
            board,
            players=self.agents,
            logger=logger,
            max_turns=self.max_turns,
        )
        # reset agent states
        for agent in self.agents:
            agent.reset()
        observations = self._get_observations()
        infos = {agent_index: {} for agent_index in range(len(self.agents))}
        self.initial_observations = observations
        return observations, infos

    def step(self, actions):
        # Convert actions to the format expected by game_master
        game_actions = [
            Action.from_index(actions[agent_index], self.board_x_size)
            for agent_index in range(len(self.agents))
        ]

        # Execute one tick of the game
        self.game_master.step(game_actions)
        self.n_step += 1

        observations = self._get_observations()
        rewards = self._get_rewards()
        terminations = self._get_terminations()
        truncations = self._get_truncations()
        infos = self._get_infos(game_actions)
        return observations, rewards, terminations, truncations, infos

    def _get_observations(self):
        return self.initial_observations

    def _get_rewards(self):
        return {agent_index: 1 for agent_index in range(len(self.agents))}

    def _get_terminations(self):
        return {agent_index: True for agent_index in range(len(self.agents))}

    def _get_truncations(self):
        return {agent_index: False for agent_index in range(len(self.agents))}

    def _get_infos(self, game_actions):
        are_game_actions_legal = {
            agent_index: self.game_master.board.is_action_valid(
                game_actions[agent_index], agent_index
            )
            for agent_index in range(len(self.agents))
        }
        infos = self._merge_info_dicts(is_game_action_legal=are_game_actions_legal)
        return infos

    def _merge_info_dicts(self, **info_dicts):
        infos = defaultdict(dict)
        for info_type, info_dict in info_dicts.items():
            for agent_index, info in info_dict.items():
                infos[agent_index][info_type] = info
        return infos

    def render(self):
        pass

    def close(self):
        pass

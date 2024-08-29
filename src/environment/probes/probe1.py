from collections import defaultdict
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from typing import Dict, List

from src.agents.agent import Agent
from src.environment.action import Action
from src.environment.board_generator import generate_board_state
from src.environment.environment import GeneralsEnvironment
from src.environment.game_master import GameMaster
from src.environment.logger import Logger
from src.training.input import (
    convert_state_to_array,
    get_input_channel_dimension_size,
)


class ProbeOneEnvironment(GeneralsEnvironment):
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
        normal_tile_increment_frequency: int = 50,
    ):
        self.agents = agents
        self.max_turns = max_turns
        self.board_x_size = board_x_size
        self.board_y_size = board_y_size
        self.mountain_probability = mountain_probability
        self.city_probability = city_probability
        self.use_fog_of_war = use_fog_of_war
        self.n_step = 0
        self.auxiliary_reward_weight = auxiliary_reward_weight
        self.normal_tile_increment_frequency = normal_tile_increment_frequency

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
            normal_tile_increment_frequency=self.normal_tile_increment_frequency,
        )
        self.previous_agent_scores = {
            agent_index: 0 for agent_index in range(len(self.agents))
        }

        # reset agent states
        for agent in self.agents:
            agent.reset()

        observations = self._get_observations()
        infos = {agent_index: {} for agent_index in range(len(self.agents))}
        return observations, infos

    def step(self, actions, agent_infos={}):
        self.agent_infos = agent_infos
        
        player_dict_to_list = lambda player_dict: [
            player_dict[agent_index] for agent_index in range(len(self.agents))
        ]

        # when serializing observations, serialize them channels-last instead of channels-first i.e. (C, H, W) -> (H, W, C)
        self.game_master.logger.log_info(
            "obs_tensor",
            [
                a.transpose(1, 2, 0).tolist()
                for a in player_dict_to_list(self._get_observations())
            ],
            self.n_step,
        )

        # Convert actions to the format expected by game_master
        game_actions = [
            Action.from_index(actions[agent_index], self.board_x_size)
            for agent_index in range(len(self.agents))
        ]

        # save wehther moves were legal before the board is updated
        auxiliary_legal_move_rewards = self.get_auxiliary_legal_move_reward(
            game_actions
        )

        # Execute one tick of the game
        self.game_master.step(game_actions)
        self.n_step += 1

        observations = self._get_observations()
        rewards = self._get_rewards(auxiliary_legal_move_rewards)
        terminations = self._get_terminations()
        truncations = self._get_truncations()
        infos = self._get_infos()

        self.game_master.logger.log_info(
            "rewards", player_dict_to_list(rewards), self.n_step - 1
        )
        self.game_master.logger.log_info(
            "action_indices", player_dict_to_list(actions), self.n_step - 1
        )
        self.game_master.logger.log_info(
            "actions", [vars(a) for a in game_actions], self.n_step - 1
        )
        self.game_master.logger.log_info(
            "agent_infos", player_dict_to_list(infos), self.n_step - 1
        )

        return observations, rewards, terminations, truncations, infos

    def _get_observations(self):
        observation = convert_state_to_array(
            self.game_master.board,
            len(self.agents),
            fog_of_war=self.use_fog_of_war,
        )
        return {
            agent_index: observation[agent_index]
            for agent_index in range(len(self.agents))
        }

    def _get_rewards(self, auxiliary_legal_move_rewards: Dict[int, int]):
        total_rewards = {
            agent_index: self.auxiliary_reward_weight
            * auxiliary_legal_move_rewards[agent_index]
            for agent_index in range(len(self.agents))
        }
        return total_rewards

    def get_main_rewards(self):
        winning_agent_index = self.game_master.board.terminal_status()
        main_rewards = {}
        for agent_index in range(len(self.agents)):
            if winning_agent_index == -1:
                main_rewards[agent_index] = 0
            else:
                main_rewards[agent_index] = (
                    1 if agent_index == winning_agent_index else -1
                )
        return main_rewards

    def get_auxiliary_legal_move_reward(self, game_actions: List[Action]):
        return {
            i: int(
                self.game_master.board.is_action_valid(action=action, player_index=i)
            )
            for i, action in enumerate(game_actions)
        }

    def _get_terminations(self):
        return {agent_index: False for agent_index in range(len(self.agents))}

    def _get_truncations(self):
        return {agent_index: True for agent_index in range(len(self.agents))}

    def _get_infos(self):
        return {agent_index: self.agent_infos[agent_index] for agent_index in range(len(self.agents))}

    def render(self):
        pass

    def close(self):
        pass

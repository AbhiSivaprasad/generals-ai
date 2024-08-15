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
    convert_action_index_to_action,
    convert_state_to_array,
    get_input_channel_dimension_size,
)


class GeneralsEnvironment(ParallelEnv):
    metadata = {"name": "generals_v0"}

    def __init__(
        self,
        players: List[Agent],
        max_turns: int = 1000,
        board_x_size: int = 3,
        board_y_size: int = 3,
        mountain_probability: float = 0.0,
        city_probability: float = 0.0,
        use_fog_of_war: bool = False,
        auxiliary_reward_weight: float = 0.01,
    ):
        self.players = players
        self.agent_name_by_player_index = [
            f"player_{i}" for i in range(len(self.players))
        ]
        self.player_index_by_agent_name = {
            agent_name: i
            for i, agent_name in enumerate(self.agent_name_by_player_index)
        }
        self.max_turns = max_turns
        self.board_x_size = board_x_size
        self.board_y_size = board_y_size
        self.mountain_probability = mountain_probability
        self.city_probability = city_probability
        self.use_fog_of_war = use_fog_of_war
        self.n_step = 0
        self.auxiliary_reward_weight = auxiliary_reward_weight

        # Define action and observation spaces
        self.action_spaces = {
            agent_name: spaces.Discrete(board_x_size * board_y_size * 4)
            for agent_name in self.agent_name_by_player_index
        }
        self.observation_spaces = {
            agent_name: spaces.Box(
                low=0,
                high=np.inf,
                shape=(
                    get_input_channel_dimension_size(self.use_fog_of_war),
                    board_x_size,
                    board_y_size,
                ),
                dtype=np.int32,
            )
            for agent_name in self.agent_name_by_player_index
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
            players=self.players,
            logger=logger,
            max_turns=self.max_turns,
        )
        self.previous_player_scores = {
            agent_name: 0 for agent_name in self.agent_name_by_player_index
        }

        # reset player states
        for player in self.players:
            player.reset()

        observations = self._get_observations()
        infos = {agent_name: {} for agent_name in self.agent_name_by_player_index}
        return observations, infos

    def step(self, actions):
        # Convert actions to the format expected by game_master
        game_actions = [
            convert_action_index_to_action(actions[agent_name], self.board_x_size)
            for agent_name in self.agent_name_by_player_index
        ]
        are_game_actions_legal = {
            agent_name: self.game_master.board.is_action_valid(
                game_actions[player_index], player_index
            )
            for player_index, agent_name in enumerate(self.agent_name_by_player_index)
        }

        # Execute one tick of the game
        self.game_master.step(game_actions)
        self.n_step += 1

        observations = self._get_observations()
        rewards = self._get_rewards()
        terminations = self._get_terminations()
        truncations = self._get_truncations()
        infos = self._get_infos(is_game_action_legal=are_game_actions_legal)
        return observations, rewards, terminations, truncations, infos

    def _get_observations(self):
        return {
            agent_name: convert_state_to_array(
                self.game_master.board,
                len(self.players),
                fog_of_war=self.use_fog_of_war,
            )[i]
            for i, agent_name in enumerate(self.agent_name_by_player_index)
        }

    def _get_rewards(self):
        # reward component 1: change in difference between agent's score and other agent's score
        # reward component 2: win/loss
        # final reward is component 2 + 0.01 * component 1
        player_scores = {
            agent_name: self.game_master.board.get_player_score(i)
            for i, agent_name in enumerate(self.agent_name_by_player_index)
        }
        player_score_changes = {
            agent_name: player_scores[agent_name]
            - self.previous_player_scores[agent_name]
            for agent_name in self.agent_name_by_player_index
        }
        total_player_score_changes = sum(player_score_changes.values())
        auxiliary_rewards = {
            agent_name: 2 * player_score_change - total_player_score_changes
            for agent_name, player_score_change in player_score_changes.items()
        }

        main_rewards = self._get_main_rewards()
        total_rewards = {
            agent_name: main_rewards[agent_name]
            + self.auxiliary_reward_weight * auxiliary_rewards[agent_name]
            for agent_name in self.agent_name_by_player_index
        }
        self.previous_player_scores = player_scores
        return total_rewards

    def _get_main_rewards(self):
        winning_player_index = self.game_master.board.terminal_status()
        main_rewards = {}
        for i, agent_name in enumerate(self.agent_name_by_player_index):
            if winning_player_index == -1:
                main_rewards[agent_name] = 0
            else:
                main_rewards[agent_name] = 1 if i == winning_player_index else -1
        return main_rewards

    def _get_terminations(self):
        game_over = self.game_master.board.terminal_status() != -1
        return {agent_name: game_over for agent_name in self.agent_name_by_player_index}

    def _get_truncations(self):
        truncated = self.n_step >= self.max_turns
        return {agent_name: truncated for agent_name in self.agent_name_by_player_index}

    def _get_infos(self, **kwargs):
        infos = defaultdict(dict)
        for info_type, info_dict in kwargs.items():
            for agent_name, info in info_dict.items():
                infos[agent_name][info_type] = info
        return infos

    def render(self):
        pass

    def close(self):
        pass

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
    ):
        self.players = players
        self.agent_names = [f"player_{i}" for i in range(len(self.players))]
        self.max_turns = max_turns
        self.board_x_size = board_x_size
        self.board_y_size = board_y_size
        self.mountain_probability = mountain_probability
        self.city_probability = city_probability
        self.use_fog_of_war = use_fog_of_war
        self.n_step = 0

        # Define action and observation spaces
        self.action_spaces = {
            agent_name: spaces.Discrete(board_x_size * board_y_size * 4)
            for agent_name in self.agent_names
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
            for agent_name in self.agent_names
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
        self.previous_player_scores = {agent_name: 0 for agent_name in self.agent_names}

        # reset player states
        for player in self.players:
            player.reset()

        observations = self._get_observations()
        infos = {agent_name: {} for agent_name in self.agent_names}
        return observations, infos

    def step(self, actions):
        # Convert actions to the format expected by game_master
        game_actions = [
            convert_action_index_to_action(actions[agent_name], self.board_x_size)
            for agent_name in self.agent_names
        ]

        # Execute one tick of the game
        self.game_master.step(game_actions)
        self.n_step += 1

        observations = self._get_observations()
        rewards = self._get_rewards()
        terminations = self._get_terminations()
        truncations = self._get_truncations()
        infos = {agent_name: {} for agent_name in self.agent_names}
        return observations, rewards, terminations, truncations, infos

    def _get_observations(self):
        return {
            agent_name: convert_state_to_array(
                self.game_master.board,
                len(self.players),
                fog_of_war=self.use_fog_of_war,
            )[i]
            for i, agent_name in enumerate(self.agent_names)
        }

    def _get_rewards(self):
        # reward component 1: change in difference between agent's score and other agent's score
        # reward component 2: win/loss
        # final reward is component 2 + 0.01 * component 1
        player_scores = {
            agent_name: self.game_master.board.get_player_score(i)
            for i, agent_name in enumerate(self.agent_names)
        }
        rewards = {
            agent_name: player_scores[agent_name]
            - self.previous_player_scores[agent_name]
            for agent_name in self.agent_names
        }
        self.previous_player_scores = player_scores
        return rewards

    def _get_terminations(self):
        game_over = self.game_master.board.terminal_status() != -1
        return {agent_name: game_over for agent_name in self.agent_names}

    def _get_truncations(self):
        truncated = self.n_step >= self.max_turns
        return {agent_name: truncated for agent_name in self.agent_names}

    def render(self):
        pass

    def close(self):
        pass

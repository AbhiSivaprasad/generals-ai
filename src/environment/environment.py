from typing import Dict, List

import gymnasium
import numpy as np
from src.agents.agent import Agent
from src.agents.random_agent import RandomAgent
from src.environment.action import Action
from src.environment.board_generator import generate_board_state
from src.environment.game_master import GameMaster
from src.training.input import convert_state_to_array, get_input_channel_dimension_size


class GeneralsEnvironment(gymnasium.Env):
    game_master: GameMaster

    def __init__(
        self,
        players: List[Agent],
        max_turns: int = 1000,
        board_x_size: int = 3,
        board_y_size: int = 3,
        mountain_probabilitiy: float = 0.0,
        city_probabilitiy: float = 0.0,
        use_fog_of_war: bool = False,
    ) -> None:
        super().__init__()
        self.players = players
        self.max_turns = max_turns
        self.board_x_size = board_x_size
        self.board_y_size = board_y_size
        self.mountain_probabilitiy = mountain_probabilitiy
        self.city_probabilitiy = city_probabilitiy
        self.use_fog_of_war = use_fog_of_war

        # spaces
        self.action_space = gymnasium.spaces.Discrete(board_x_size * board_y_size * 4)
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=np.inf,
            shape=(
                len(self.players),
                get_input_channel_dimension_size(self.use_fog_of_war),
                board_x_size,
                board_y_size,
            ),
            dtype=np.int32,
        )

    def step(self, actions: List[Action]):
        # execute one tick of the game
        self.game_master.step(actions)

        # return the new state, reward, terminal status, and info dict
        observation = convert_state_to_array(
            self.game_master.board, len(self.players), fog_of_war=self.use_fog_of_war
        )
        rewards = [
            self.game_master.board.get_player_score(i) for i in range(len(self.players))
        ]
        done = self.board.terminal_status() != -1
        info = {}
        return (observation, rewards, done, info)

    def reset(self, seed=None, options: Dict = {}):
        super().reset(seed=seed)

        # generate new board
        board = generate_board_state(
            self.board_x_size,
            self.board_y_size,
            mountain_probability=self.mountain_probabilitiy,
            city_probability=self.city_probabilitiy,
        )
        self.game_master = GameMaster(
            board,
            players=self.players,
            logger=None,
            max_turns=self.max_turns,
        )
        initial_state = convert_state_to_array(
            self.game_master.board, len(self.players), self.use_fog_of_war
        )
        info = {}
        return initial_state, info

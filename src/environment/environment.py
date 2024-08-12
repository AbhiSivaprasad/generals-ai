from typing import Dict, Optional, Tuple
import numpy as np
from  itertools import product as cartesian

import gymnasium.core as gym
from gymnasium.spaces import Space, \
    Tuple as TupleSpace, \
    MultiBinary as MultiBinarySpace, \
    Discrete as DiscreteSpace, \
    Box as BoxSpace
    
from src.agents.gym_agent import GymAgent
from src.environment import board_generator, game_master
from src.agents.agent import Agent
from src.environment.action import Action
from src.environment.gamestate import GameState
from src.environment.logger import Logger
from src.environment.tile import TileType


MAX_SIZE = [15, 15]

ObsType = np.ndarray
ActType = Action


class GeneralsEnvironment(gym.Env):
    """
    Single-player gymnasium environment wrapper for Generals.io.
    """
    action_space: Space
    observation_space: Space
    
    agent: GymAgent = GymAgent()
    opponent: Agent
    
    game: game_master.GameMaster
    
    def __init__(
        self, 
        num_rows: int,
        num_cols: int,
        mountain_probability: float = 0,
        city_probability: float = 0,
        min_ratio_of_generals_distance_to_board_side: float = 2 / 3,
    ):
        super().__init__()
        
        self.action_space = DiscreteSpace(MAX_SIZE[0] * MAX_SIZE[1] * 4 + 1) # 15 x 15 x 4 + 1 (none action)
        self.observation_space = TupleSpace(DiscreteSpace(MAX_SIZE[0]), DiscreteSpace(MAX_SIZE[1]), BoxSpace(low=0, dtype=int), DiscreteSpace(4), MultiBinarySpace(2)) \
            # [army, 1-hot general, 1-hot city, 1-hot mountain, 1-hot in-bounds, 0/1 is_mine, 0/1 visible]
        
        self.num_rows, self.num_cols = num_rows, num_cols
        self.mountain_probability, self.city_probability = mountain_probability, city_probability
        self.min_ratio_of_generals_distance_to_board_side = min_ratio_of_generals_distance_to_board_side
                
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[ObsType, Dict]:
        '''
        Resets the environment to its initial state.
        '''
        super().reset(seed=seed, options=options)
        
        self.opponent = options["opponent"]
        
        assert self.opponent is not None and isinstance(self.opponent, Agent), "Generals env requires `opponent` option to environment!"
        
        board = board_generator.generate_board_state(
            self.num_rows, 
            self.num_cols, 
            self.mountain_probability, 
            self.city_probability, 
            self.min_ratio_of_generals_distance_to_board_side
        )
        
        self.game = game_master.GameMaster(board, [self.agent, self.opponent], logger=Logger())
        
        return self.game_state_to_observation(self.game.state), {}
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        '''
        Execute an action and return the new state, reward, done flag, and additional info.
        The behaviour of this function depends primarily on the dynamics of the underlying
        environment.
        '''
        reward = 0.0
        self.agent.set_action(action)
        legal_move = self.game.state.board.is_action_valid(action, 0)
        reward += (1 - legal_move) * -1.0 # penalize illegal moves / invalid actions
        new_game_state = self.game.step()
        new_obs = self.game_state_to_observation(new_game_state)

        terminated = self.game.state.board.terminal_status() > -1
        if terminated:
            agentWon = self.game.state.board.terminal_status() == 0
            reward += 1.0 if agentWon else -1.0
        
        return (new_obs, reward, terminated, False, {"opponent": self.opponent, "legal_move": legal_move, "game_state": self.game.state})
    
    def game_state_to_observation(self, state: GameState) -> ObsType:
        # [army, 1-hot general, 1-hot city, 1-hot mountain, 1-hot in-bounds, 0/1 is_mine, 0/1 visible
        obs = np.zeros((MAX_SIZE[0], MAX_SIZE[1], 7), dtype=np.float32)
        board_r, board_c = len(state.board.grid), len(state.board.grid[0])
        for r, c in cartesian(range(MAX_SIZE[0]), range(MAX_SIZE[1])):
            if 0 <= r < board_r and 0 <= c < board_c:
                tile = state.board.grid[r][c]
                obs[r, c, 4] = 1
                if tile.player_visibilities[0]: # GymAgent is index 0 by design
                    obs[r, c, 0] = tile.army
                    obs[r, c, 1] = 1 if tile.type == TileType.GENERAL else 0
                    obs[r, c, 2] = 1 if tile.type == TileType.CITY else 0
                    obs[r, c, 3] = 1 if tile.type == TileType.MOUNTAIN else 0
                    obs[r, c, 5] = (tile.player_index == 0)
                    obs[r, c, 6] = 1
        return obs

            
            
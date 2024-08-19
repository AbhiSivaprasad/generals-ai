from typing import Dict, Optional, Tuple
import numpy as np
from copy import deepcopy

import gymnasium.core as gym
from gymnasium.spaces import Space, \
    Tuple as TupleSpace, \
    MultiBinary as MultiBinarySpace, \
    Discrete as DiscreteSpace, \
    Box as BoxSpace
    
from src.agents.utils.gym_agent import GymAgent
from src.environment import MAX_SIZE, board_generator, game_master
from src.agents.agent import Agent
from src.environment.action import Action
from src.environment.logger import Logger
from src.environment import ObsType, ActType



class GeneralsEnvironment(gym.Env):
    """
    Single-player gymnasium environment wrapper for Generals.io.
    """
    action_space: Space
    observation_space: Space
    
    agent: GymAgent
    opponent: Agent
    
    game: game_master.GameMaster
    
    num_rows: int
    num_cols: int
    mountain_probability: float
    city_probability: float
    min_ratio_of_generals_distance_to_board_side: float
    
    def __init__(
        self, 
        n_rows: int,
        n_cols: int,
        mountain_probability: float = 0,
        city_probability: float = 0,
        min_ratio_of_generals_distance_to_board_side: float = 2 / 3,
        seed: int = 0,
        agent: Optional[Agent] = None,
        opponent: Optional[Agent] = None
    ):
        super().__init__()
        
        self.action_space = DiscreteSpace(MAX_SIZE[0] * MAX_SIZE[1] * 4 + 1) # 15 x 15 x 4 + 1 (none action)
        self.observation_space = DiscreteSpace(1) # placeholder
        # self.observation_space = TupleSpace(DiscreteSpace(MAX_SIZE[0]), DiscreteSpace(MAX_SIZE[1]), BoxSpace(low=0, dtype=int), DiscreteSpace(4), MultiBinarySpace(2)) \
            # [army, 1-hot general, 1-hot city, 1-hot mountain, 1-hot in-bounds, 0/1 is_mine, 0/1 visible]
        
        self.num_rows, self.num_cols = n_rows, n_cols
        self.mountain_probability, self.city_probability = mountain_probability, city_probability
        self.min_ratio_of_generals_distance_to_board_side = min_ratio_of_generals_distance_to_board_side
        
        self.agent = GymAgent(player_index=0) if agent is None else GymAgent(agent=agent)
        self.opponent = opponent
    
    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[ObsType, Dict]:
        '''
        Resets the environment to its initial state.
        '''
        super().reset(seed=seed, options=options)
        
        assert self.opponent is not None and isinstance(self.opponent, Agent), "Generals env requires `opponent` option to environment!"
        
        self.agent.reset(seed=seed)
        self.opponent.reset(seed=seed)
        
        board = board_generator.generate_board_state(
            self.num_rows, 
            self.num_cols, 
            self.mountain_probability, 
            self.city_probability, 
            self.min_ratio_of_generals_distance_to_board_side,
            self.np_random
        )
        self.game = game_master.GameMaster(board, [self.agent, self.opponent], logger=Logger())
        
        return self.game.state.to_observation(self.agent.player_index), {}
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        '''
        Execute an action and return the new state, reward, done flag, and additional info.
        The behaviour of this function depends primarily on the dynamics of the underlying
        environment.
        '''
        self._prev_state = deepcopy(self.game.state)
        
        action_idx = action
        action = Action.from_space_sample(action_idx, self.num_rows, self.num_cols)
        self.agent.set_action(action)
        
        reward = 0.0
        legal_move = self.game.state.board.is_action_valid(action, self.agent.player_index)
        reward += (1 - legal_move) * -1.0 # penalize illegal moves / invalid actions
        
        new_game_state = self.game.step()
        new_obs = new_game_state.to_observation(self.agent.player_index)

        terminated = self.game.state.board.terminal_status() > -1
        if terminated:
            agentWon = self.game.state.board.terminal_status() == self.agent.player_index
            multiplier = 1.0 if agentWon else -1.0
            reward += (multiplier * (0.1/(1.0 - self.agent.gamma)))
            
        info = {"opponent": self.opponent, "legal_move": legal_move, "game_state": self.game.state, "prev_state": self._prev_state}
        
        self.game.logger.log_info("obs", self._prev_state.to_observation(self.agent.player_index), self._prev_state.turn)
        self.game.logger.log_info("reward", reward, self._prev_state.turn)
        self.game.logger.log_info("action_idx", action_idx, self._prev_state.turn)
        self.game.logger.log_info("action", str(action), self._prev_state.turn)
        self.game.logger.log_info("info", info, self._prev_state.turn)
        
        return (new_obs, reward, terminated, False, info)
    
    def write(self, path: str):
        '''
        Write the environment to a file.
        '''
        self.game.logger.write(path)

            
from gymnasium.envs.registration import register

register(
     id="generals-v0",
     entry_point=GeneralsEnvironment,
     max_episode_steps=500,
)
<<<<<<< HEAD
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


class GeneralsEnvironment(ParallelEnv):
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

    def step(self, actions, action_infos=None):
        pre_action_observation = self._get_observations()
        self.action_infos = action_infos

        # convert actions to the format expected by game_master
        game_actions = [
            Action.from_index(actions[agent_index], self.board_x_size)
            for agent_index in range(len(self.agents))
        ]

        # save wehther moves were legal before the board is updated
        self.are_agent_actions_legal = self.check_agent_actions_legal(game_actions)

        # execute one tick of the game
        self.game_master.step(game_actions)

        # new environment information
        observations = self._get_observations()
        rewards = self._get_rewards()
        terminations = self._get_terminations()
        truncations = self._get_truncations()
        infos = self._get_infos()

        self._log(pre_action_observation, rewards, actions, game_actions, infos)
        self.n_step += 1

        return observations, rewards, terminations, truncations, infos

    def _get_observations(self):
        observation = convert_state_to_array(
            self.game_master,
            len(self.agents),
            fog_of_war=self.use_fog_of_war,
        )
        return {
            agent_index: observation[agent_index]
            for agent_index in range(len(self.agents))
        }

    def _get_rewards(self):
        # reward component 1a: change in difference between agent's score and other agent's score
        # reward component 1b: whether action was legal
        # reward component 2: win/loss
        # final reward is component 2 + constant * component 1
        auxiliary_land_rewards = self.get_auxiliary_land_reward()
        auxiliary_legal_move_rewards = self.get_auxiliary_legal_move_reward()
        main_rewards = self.get_main_rewards()
        total_rewards = {
            agent_index: main_rewards[agent_index]
            + self.auxiliary_reward_weight
            * (
                auxiliary_land_rewards[agent_index]
                + auxiliary_legal_move_rewards[agent_index]
            )
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

    def check_agent_actions_legal(self, game_actions: List[Action]):
        return {
            i: self.game_master.board.is_action_valid(action=action, player_index=i)
            for i, action in enumerate(game_actions)
        }

    def get_auxiliary_legal_move_reward(self):
        return {
            i: int(is_legal) for i, is_legal in self.are_agent_actions_legal.items()
        }

    def get_auxiliary_land_reward(self):
        agent_scores = {
            agent_index: self.game_master.board.get_player_score(agent_index)
            for agent_index in range(len(self.agents))
        }
        agent_score_changes = {
            agent_index: agent_scores[agent_index]
            - self.previous_agent_scores[agent_index]
            for agent_index in range(len(self.agents))
        }
        total_agent_score_changes = sum(agent_score_changes.values())
        auxiliary_rewards = {
            agent_index: 2 * agent_score_change - total_agent_score_changes
            for agent_index, agent_score_change in agent_score_changes.items()
        }
        self.previous_agent_scores = agent_scores
        return auxiliary_rewards

    def _get_terminations(self):
        game_over = self.game_master.board.terminal_status() != -1
        return {agent_index: game_over for agent_index in range(len(self.agents))}

    def _get_truncations(self):
        truncated = self.n_step >= self.max_turns
        return {agent_index: truncated for agent_index in range(len(self.agents))}

    def _get_infos(self):
        merged_info_for_agent = lambda agent_index: (
            self.action_infos[agent_index] if self.action_infos is not None else {}
        )
        return {
            agent_index: merged_info_for_agent(agent_index)
            for agent_index in range(len(self.agents))
        }

    def _log(self, pre_action_observation, rewards, actions, game_actions, infos):
        player_dict_to_list = lambda player_dict: [
            player_dict[agent_index] for agent_index in range(len(self.agents))
        ]
        # when serializing observations, serialize them channels-last instead of channels-first i.e. (C, H, W) -> (H, W, C)
        self.game_master.logger.log_info(
            "obs_tensor",
            [
                a.transpose(1, 2, 0).tolist()
                for a in player_dict_to_list(pre_action_observation)
            ],
            self.n_step,
        )
        self.game_master.logger.log_info(
            "rewards", player_dict_to_list(rewards), self.n_step
        )
        self.game_master.logger.log_info(
            "action_indices", player_dict_to_list(actions), self.n_step
        )
        self.game_master.logger.log_info(
            "actions", [vars(a) for a in game_actions], self.n_step
        )
        self.game_master.logger.log_info(
            "agent_infos", player_dict_to_list(infos), self.n_step
        )

    def render(self):
        pass

    def close(self):
        pass
=======
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
from src.environment.gamestate import GameState
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
        
        self.reset(seed=seed)
    
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
        
        return self.game.state.to_observation(self.agent.player_index, fog_of_war=False), {}
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        self._prev_state = GameState(self.game.state.board, deepcopy(self.game.state.scores), self.game.state.turn, self.game.state.terminal_status)
        
        action_idx = action
        action = Action.from_space_sample(action_idx, self.num_rows, self.num_cols)
        self.agent.set_action(action)
        
        reward = 0.0
        legal_move = self.game.state.board.is_action_valid(action, self.agent.player_index)
        reward += (1 - legal_move) * -1.0 # penalize illegal moves / invalid actions
        
        new_game_state = self.game.step()
        new_obs = new_game_state.to_observation(self.agent.player_index, fog_of_war=False)

        terminated = self.game.state.board.terminal_status() > -1
        if terminated:
            agentWon = self.game.state.board.terminal_status() == self.agent.player_index
            win_loss_reward = 10.0 if agentWon else -10.0
            reward += win_loss_reward
            
        info = {"opponent": self.opponent, "legal_move": legal_move, "game_state": self.game.state, "prev_state": self._prev_state}
        
        self.game.logger.log_info("obs", self._prev_state.to_observation(self.agent.player_index)[1].tolist(), self._prev_state.turn)
        self.game.logger.log_info("reward", float(reward), self._prev_state.turn)
        self.game.logger.log_info("action_idx", int(action_idx), self._prev_state.turn)
        self.game.logger.log_info("action", str(action), self._prev_state.turn)
        self.game.logger.log_info("info", {k: str(v) for (k, v) in info.items()}, self._prev_state.turn)
        
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
>>>>>>> origin/ds/main

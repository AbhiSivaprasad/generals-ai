from gymnasium import Space
import numpy as np
from tqdm import tqdm

from typing import List, Optional
import gymnasium as gym

from src.agents.agent import Agent
from src.agents.utils.agent_wrapper import AgentWrapper
from src.environment.action import Action
from src.environment import ActType, ObsType
from src.environment.gamestate import GameState
from src.utils.replay_buffer import Experience


class GymAgent(AgentWrapper):
    """
    Pass-through **wrapper** class to interface the `Agent` class with the `gymnasium` framework's environments and agents.
    Allows the gymnasium environment to work with a `step(a: Action) -> newstate: State` method rather than an environment-driven `step(s: State) -> a: Action` interface.
    When given an `agent: Agent`, the `GymAgent` wrapper will use the given agent's `move()` logic to make moves; without an agent, the `GymAgent` will simply act as a pass-through for enacting actions passed to the environment's `step()` method.
    """
    action: Action = None
    env: gym.Env
    
    observation_space: Space
    action_space: Space
    
    gamma: float
    
    def __init__(self, agent: Optional[Agent] = None, env: Optional[gym.Env] = None, gamma: float = 0.99, *args, **kwargs):
        super().__init__(agent)
        self.set_action(None)
        self.set_env(env)
        if agent is not None:
            self.set_agent(agent)
        self.gamma = gamma
        
    def set_action(self, action: Action):
        self.action = action
    
    def set_env(self, env):
        self.env = env
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
    
    def set_agent(self, agent: Agent):
        self.agent = agent

    def move(self, state: GameState) -> Optional[Action]:
        return self.action

    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self.action = None
        if self.agent is not None:
            self.agent.reset(*args, **kwargs)
    
    def get_action(self, obs: ObsType) -> ActType:
        assert self.agent is not None, "Need an agent to act!"
        if isinstance(self.agent, GymAgent):
            return self.agent.get_action(obs)
        s = GameState.from_observation(obs, self.player_index)
        return Action.to_space_sample(self.agent.move(s), s.board.num_rows, s.board.num_cols)
    
    def run_episode(self, seed=None) -> List[float]:
        '''
        Simulates one episode of interaction, agent learns as appropriate
        Inputs:
            seed : Seed for the random number generator
        Outputs:
            The rewards obtained during the episode
        '''
        assert self.env is not None, "Need an environment to run an episode!"
        assert self.agent is not None, "Need an agent to run an episode!"

        rewards = []
        obs, _ = self.env.reset()
        self.reset(seed=seed)
        self.agent.reset(seed=seed)
        done = False
        while not done:
            act = self.get_action(obs)
            self.set_action(act)
            (new_obs, reward, terminated, truncated, info) = self.env.step(act) # this will set self.action
            done = terminated or truncated
            experience: Experience = (obs, act, reward, new_obs, done)
            if hasattr(self.agent, "observe"):
                self.agent.observe(experience)
            rewards.append(reward)
            obs = new_obs
        return rewards
    
    def run_episodes(self, seed, n_runs=500):
        '''
        Run a batch of episodes, and return the total reward obtained per episode
        Inputs:
            n_runs : The number of episodes to simulate
        Outputs:
            The discounted sum of rewards obtained for each episode
        '''
        all_rewards = []
        powers = np.random.default_rng(seed).integers(2, 15, n_runs)
        offsets = np.random.default_rng(seed).integers(0, 32, n_runs)
        seeds = np.power(2, powers) + offsets
        for idx, seed in enumerate(seeds):
            rewards = self.run_episode(int(seed))
            episode_len = len(rewards)
            rewards = np.array(rewards, dtype=np.float32) * np.power(self.gamma, np.arange(episode_len))
            all_rewards.append(int(rewards.sum()))
        return all_rewards
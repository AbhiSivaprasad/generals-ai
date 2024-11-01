from typing import Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import register
import numpy as np
from torch import nn

from src.environment import ActType, ObsType

class ProbeOne(gym.Env):
    """One action, zero observation, one timestep long, +1 reward every timestep."""
    
    def __init__(self, n_rows, n_cols, n_channels):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)
        self.shape = (n_rows, n_cols, n_channels)

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[ObsType, Dict]:
        return (0, np.zeros(self.shape)), {}

    def step(self, act: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        return (1, np.zeros(self.shape)), 1.0, True, False, {}

    def write(self, path: str) -> None:
        '''
        Write the environment to a file.
        '''
        pass
    
class ProbeTwo(gym.Env):
    """One action, random +1/-1 observation, one timestep long, obs-dependent +1/-1 reward every time"""
    obs_alpha: float
    
    def __init__(self, n_rows, n_cols, n_channels):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(2)
        self.shape = (n_rows, n_cols, n_channels)

    def reset(self, *args, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[ObsType, Dict]:
        super().reset(*args, seed=seed, options=options)
        self.obs_alpha = 1.0 if self.np_random.random() < 0.5 else -1.0
        return (0, self.obs_alpha * np.ones(self.shape)), {}

    def step(self, act: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        return (1, self.obs_alpha * np.ones(self.shape)), float(self.obs_alpha), True, False, {}

    def write(self, path: str) -> None:
        '''
        Write the environment to a file.
        '''
        pass
    
class ProbeThree(gym.Env):
    """One action, zero-then-one observation, two timesteps long, +1 reward at the end"""
    
    def __init__(self, n_rows, n_cols, n_channels):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(2)
        self.shape = (n_rows, n_cols, n_channels)
        self.timestep = 0

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[ObsType, Dict]:
        self.timestep = 0
        return (0, np.zeros(self.shape)), {}

    def step(self, act: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        self.timestep += 1
        if self.timestep == 1:
            return (1, np.ones(self.shape)), 0.0, False, False, {}
        else:
            return (2, np.ones(self.shape)), 1.0, True, False, {}

    def write(self, path: str) -> None:
        '''
        Write the environment to a file.
        '''
        pass
    
class ProbeFour(gym.Env):
    """Two actions, zero observation, one timestep long, action-dependent +1/-1 reward"""
    
    def __init__(self, n_rows, n_cols, n_channels):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(1)
        self.shape = (n_rows, n_cols, n_channels)

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[ObsType, Dict]:
        return (0, np.zeros(self.shape)), {}

    def step(self, act: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        if act == 0:
            return (0, np.zeros(self.shape)), -1.0, True, False, {}
        elif act == 1:
            return (0, np.zeros(self.shape)), 1.0, True, False, {}
        else:
            raise ValueError("Invalid action")

    def write(self, path: str) -> None:
        '''
        Write the environment to a file.
        '''
        pass
    
class ProbeFive(gym.Env):
    """Two actions, random 0/1 observation, one timestep long, action-and-obs dependent +1/-1 reward"""
    
    def __init__(self, n_rows, n_cols, n_channels):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.shape = (n_rows, n_cols, n_channels)

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[ObsType, Dict]:
        self.obs_alpha = 1.0 if self.np_random.random() < 0.5 else 0.0
        return (0, self.obs_alpha * np.ones(self.shape)), {}

    def step(self, act: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        reward = 1.0 if abs(float(act) - self.obs_alpha) < 1e-3 else -1.0
        self.obs_alpha = 1.0 if self.np_random.random() < 0.5 else 0.0
        return (1, self.obs_alpha * np.ones(self.shape)), reward, True, False, {}

    def write(self, path: str) -> None:
        '''
        Write the environment to a file.
        '''
        pass
    
class ProbeSix(gym.Env):
    """Two actions, random 0/1 observation, two timesteps long, action-and-obs dependent +1/-1 reward"""
    
    def __init__(self, n_rows, n_cols, n_channels):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.shape = (n_rows, n_cols, n_channels)
        self.timestep = 0

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[ObsType, Dict]:
        super().reset(seed=seed)
        self.timestep = 0
        self.obs_alpha = 1.0 if self.np_random.random() < 0.5 else 0.0
        return (0, self.obs_alpha * np.ones(self.shape)), {}

    def step(self, act: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        self.timestep += 1
        reward = 1.0 if abs(float(act) - self.obs_alpha) < 1e-3 else -1.0
        self.obs_alpha = 1.0 if self.np_random.random() < 0.5 else 0.0
        done = self.timestep >= 2
        return (self.timestep, self.obs_alpha * np.ones(self.shape)), reward, done, False, {}

    def write(self, path: str) -> None:
        '''
        Write the environment to a file.
        '''
        pass

register(
     id="probe1",
     entry_point=ProbeOne,
)

register(
     id="probe2",
     entry_point=ProbeTwo,
)

register(
     id="probe3",
     entry_point=ProbeThree,
)

register(
     id="probe4",
     entry_point=ProbeFour,
)

register(
     id="probe5",
     entry_point=ProbeFive,
)

register(
     id="probe6",
     entry_point=ProbeSix,
)





class ProbeDQN(nn.Module):
    """
    DQN for probe envs. Much smaller.
    """
    def __init__(
        self, 
        input_channels, 
        num_actions, 
        n_rows, 
        n_cols,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        simple_dim = 16
        self.simple_net = nn.Sequential(
            nn.Conv2d(input_channels, simple_dim, 3, padding=1),
            nn.MaxPool2d(n_rows, n_cols),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(simple_dim, 4 * simple_dim),
            nn.GELU(),
            nn.Linear(4 * simple_dim, simple_dim),
            nn.Dropout(0.25),
            nn.Linear(simple_dim, simple_dim),
            nn.Linear(simple_dim, num_actions),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # x = self.encoder(x)
        # x = self.blocks(x)
        # x = self.fc_layers(x)
        x = self.simple_net(x)
        return x
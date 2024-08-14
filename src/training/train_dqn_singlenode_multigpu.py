import copy
import time
from typing import Dict
import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
import gc
from tqdm import trange
import random

from src.agents.agent import Agent
from src.agents.human_exe_agent import HumanExeAgent
from src.agents.q_greedy_agent import DQNAgent
from src.agents.utils.observation_receiver import ObservationReceiving
from src.models.dqn_cnn import DQN
from src.utils.replay_buffer import RayReplayBuffer, Experience

from src.agents.utils.gym_agent import GymAgent
from src.environment.environment import GeneralsEnvironment

from math import prod as product

ray.init()


class DQNTrainingConfig(object):
    learning_rate: float = None
    batch_size: int = None
    memory_capacity: int = None
    num_steps: int = None
    gamma: float = None
    wait_buffer_size: int = None
    input_channels: int = None
    use_wandb: bool = None
    
    def __init__(self, config_file="default.yml", **kwargs):
        self.load_config(config_file)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        for key, value in config.items():
            try:
                setattr(self, key, value)
            except Exception as e:
                print(f"Error setting {key} to {value}: {e}")

    def save_config(self, config_file):
        config = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith('_')
        }
        with open(config_file, 'w') as file:
            yaml.dump(config, file)

    def __str__(self):
        config = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith('_')
        }
        return f'DQNTrainingConfig({config})'

@ray.remote(num_gpus=0)
class SharedServer:
    target_net_state_dict: Dict = None
    training_steps: int = 0
    
    def __init__(self, target_net_state_dict: Dict):
        self.target_net_state_dict = target_net_state_dict
        self.training_steps = 0
        
    def get_target_state_dict(self):
        return self.target_net_state_dict
    
    def set_target_state_dict(self, dict: Dict):
        self.target_net_state_dict = dict
    
    def increment_training_steps(self):
        self.training_steps += 1
        return self.training_steps

@ray.remote
class EnvRunner:
    buffer: ray.ObjectRef[RayReplayBuffer]
    agent: GymAgent
    
    def _handle_observation(self, experience: Experience):
        self.buffer.add.remote(experience)
    
    def __init__(self, config: DQNTrainingConfig, agent: Agent, opponent: Agent, server: ray.ObjectRef[SharedServer], buffer: ray.ObjectRef[RayReplayBuffer]):
        self.config = config
        self.server = server
        self.buffer = buffer
        agent = ObservationReceiving(agent, observation_handler=self._handle_observation)
        self.env = gym.make("generals-v0", agent=agent, opponent=opponent, seed=config.seed, n_rows=config.n_rows, n_cols=config.n_cols)
        self.agent: GymAgent = self.env.agent
    
    def run(self):
        rng = np.random.default_rng(self.config.seed)
        self.env.reset(seed=self.config.seed)
        while True:
            agent_seed = rng.integers(-2**29, 2**29)
            if isinstance(self.agent.unwrapped, DQNAgent):
                dqn_agent: DQNAgent = self.agent.unwrapped
                model = dqn_agent.model.cuda()
                state_dict = ray.get(self.server.get_target_state_dict.remote())
                if state_dict is not None:
                    model.load_state_dict(state_dict)
                    dqn_agent.update_model(model.cuda())
                    gc.collect()
                    torch.cuda.empty_cache()
            self.env.reset()
            self.agent.run_episodes(seed=agent_seed, n_runs=1000)

# Define training loop
def train(config: DQNTrainingConfig, server: ray.ObjectRef[SharedServer], buffer: ray.ObjectRef[RayReplayBuffer]):
    start_time = time.time()   
    target_net = DQN(config.input_channels, config.num_actions, config.n_rows, config.n_cols).cuda()
    target_net.load_state_dict(ray.get(server.get_target_state_dict.remote()))
    target_net.eval()
    
    policy_net = DQN(config.input_channels, config.num_actions, config.n_rows, config.n_cols).cuda().train()
    optimizer = optim.AdamW(policy_net.parameters(), lr=config.learning_rate)
    
    print("Waiting for buffer to fill...")
    size = ray.get(buffer.size.remote())
    while size < config.wait_buffer_size:
        print(f"Buffer size: {size}")
        time.sleep(5)
        size = ray.get(buffer.size.remote())
    
    losses = []
    print("Starting training...")
    for step in trange(config.num_steps):
        data = ray.get(buffer.sample.remote(config.batch_size))
        s_t, a_t, r_t_1, d_t_1, s_t_1 = data

        with torch.inference_mode():
            target_max = target_net(s_t_1).max(-1).values
            
        predicted_q_vals = policy_net(s_t)[range(config.batch_size), a_t]

        td_error = r_t_1 + config.gamma * target_max * (1 - d_t_1.float()) - predicted_q_vals
        loss = td_error.pow(2).mean()
        losses.append(loss.item())
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if config.use_wandb:
            wandb.log({
                "td_loss": loss,
                "q_values": predicted_q_vals.mean().item(),
                "SPS": int(step / (time.time() - start_time))
            }, step=step)

if __name__ == "__main__":
    config: DQNTrainingConfig = DQNTrainingConfig()
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    
    buffer = RayReplayBuffer.remote(config.memory_capacity)
    
    tgt_net = DQN(config.input_channels, config.num_actions, config.n_rows, config.n_cols).cpu()
    server = SharedServer.remote(tgt_net.state_dict())
    
    configs = [copy.deepcopy(config) for _ in range(1000)]
    for i, c in enumerate(configs):
        c.seed = 2**(i % 29) + 2 * i + 1
    env_runners = [EnvRunner.remote(config=random.sample(configs), agent=HumanExeAgent(0), opponent=HumanExeAgent(1), server=server, buffer=buffer) for _ in range(50)]
    env_runners.extend([EnvRunner.remote(config=random.sample(configs), agent=DQNAgent(0, tgt_net, "cpu"), opponent=HumanExeAgent(1), server=server, buffer=buffer) for _ in range(100)])

    # Initialize optimizer and replay memory
    train(config=config, server=server, buffer=buffer)
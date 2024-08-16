import os
import sys
sys.dont_write_bytecode = True

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
from src.agents.random_agent import RandomAgent
from src.agents.utils.observation_receiver import ObservationReceiving
from src.environment.environment import GeneralsEnvironment
from src.models.dqn_cnn import DQN
from src.utils.replay_buffer import RayReplayBuffer, Experience
from src.agents.utils.gym_agent import GymAgent

import argparse


class DQNTrainingConfig(object):
    # agent / env
    seed: int = None
    num_actions: int = None
    input_channels: int = None
    n_rows: int = None
    n_cols: int = None
    
    # training
    memory_capacity: int = None
    wait_buffer_size: int = None
    num_steps: int = None
    learning_rate: float = None
    batch_size: int = None
    gamma: float = None
    
    use_wandb: bool = None
    
    def __init__(self, config_file=None, **kwargs):
        if config_file is not None:
            self.load_config(config_file)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            
            for key, value in config.items():
                try:
                    setattr(self, key, value)
                except Exception as e:
                    print(f"Error setting {key} to {value}: {e}")
        except:
            pass

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

@ray.remote(num_cpus=1, memory=10 * 1024**3, max_restarts=0, max_task_retries=0)
class SharedServer:
    target_net: torch.nn.Module = None
    training_steps: int = 0
    
    def __init__(self, target_net: torch.nn.Module):
        self.target_net = target_net
        self.training_steps = 0
        
    def get_target_net(self):
        return self.target_net
    
    def set_target_net(self, net: torch.nn.Module):
        self.target_net = net.cpu()
    
    def increment_training_steps(self):
        self.training_steps += 1
        return self.training_steps

@ray.remote(num_cpus=1)
class EnvRunner:
    buffer: ray.ObjectRef
    server: ray.ObjectRef
    agent: GymAgent
    env: GeneralsEnvironment
    
    _local_buffer: list[Experience] = []
    
    def _handle_observation(self, experience: Experience):
        self._local_buffer.append(experience)
        if len(self._local_buffer) >= 50:
            self.buffer.add.remote(self._local_buffer)
            self._local_buffer = []
            gc.collect()
    
    def __init__(self, config: DQNTrainingConfig, agent: Dict | Agent, opponent: Dict | Agent, server: ray.ObjectRef, buffer: ray.ObjectRef):
        self.config = config
        self.server = server
        self.buffer = buffer
        if isinstance(agent, Dict) and agent["type"].lower() == "humanexe":
            agent = HumanExeAgent(0, **agent)
        if isinstance(opponent, Dict) and opponent["type"].lower() == "humanexe":
            opponent = HumanExeAgent(1, **opponent)
        agent = ObservationReceiving(agent, observation_handler=self._handle_observation)
        self.env = gym.make("generals-v0", agent=agent, opponent=opponent, seed=config.seed, n_rows=config.n_rows, n_cols=config.n_cols)
        self.agent: GymAgent = self.env.agent
    
    def run(self):
        rng = np.random.default_rng(self.config.seed)
        self.env.reset(seed=self.config.seed)
        while True:
            agent_seed = rng.integers(0, 2**29)
            if isinstance(self.agent.unwrapped, DQNAgent):
                dqn_agent: DQNAgent = self.agent.unwrapped
                model: torch.nn.Module = ray.get(self.server.get_target_net.remote())
                model.cuda()
                dqn_agent.update_model(model.cuda())
                gc.collect()
                torch.cuda.empty_cache()    
            self.env.reset()
            print("Running episode...")
            self.agent.run_episodes(seed=agent_seed, n_runs=100)

# Define training loop
# @ray.remote(num_cpus=1, num_gpus=4)
def train(config: DQNTrainingConfig, server: ray.ObjectRef, buffer: ray.ObjectRef):
    start_time = time.time()   
    target_net: torch.nn.Module = ray.get(server.get_target_net.remote())
    target_net.cuda().eval()
    
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
    for train_step in trange(config.num_steps, miniters=20):
        print("Training step:", train_step)
        data = ray.get(buffer.sample.remote(config.batch_size))
        experience_batch  = tuple([list(t) for t in zip(*data)])
        s_t, a_t, r_t_1, s_t_1, d_t_1 = experience_batch
        # print("[INFO] Sampled experience batch: ", experience_batch)
        
        # decouple (turn, grid) ObsType
        s_t = [np.array(grid) for (_, grid) in s_t]
        s_t_1 = [np.array(grid) for (_, grid) in s_t_1]
        
        with torch.inference_mode():
            s_t_1 = torch.tensor(s_t_1, dtype=torch.float32).cuda()
            target_max = torch.max(target_net(s_t_1), dim=-1).values
        
        s_t = torch.tensor(s_t, dtype=torch.float32).cuda()
        a_t = [int(a) for a in a_t]
        r_t_1 = torch.tensor(r_t_1, dtype=torch.float32).cuda()
        d_t_1 = torch.tensor(d_t_1, dtype=torch.float32).cuda()
        
        predicted_q_vals = policy_net(s_t)[:config.batch_size, a_t]

        td_error = r_t_1 + config.gamma * target_max * (1 - d_t_1) - predicted_q_vals
        loss = td_error.pow(2).mean()
        losses.append(loss.item())
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if config.use_wandb:
            wandb.log({
                "td_loss": loss,
                "q_values": predicted_q_vals.mean().item(),
                "SPS": int(train_step / (time.time() - start_time))
            }, step=train_step)

if __name__ == "__main__":
    ray.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=False, help="Path to a training config YAML file.")
    args = parser.parse_args()

    config_file = args.config_file
    
    config: DQNTrainingConfig = DQNTrainingConfig(config_file=config_file)
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    
    print("Starting ray cluster...")
    time.sleep(5)
    
    print("Cluster resources: ", ray.cluster_resources())
    
    print("Initializing buffer and target network params...")
    buffer = RayReplayBuffer.remote(config.memory_capacity, config.seed)
    
    tgt_net = DQN(config.input_channels, config.num_actions, config.n_rows, config.n_cols).cpu()
    server = SharedServer.remote(tgt_net)
    del tgt_net
    
    
    print("initial states:")
    print("Buffer size: ", ray.get(buffer.size.remote()))
    print("Target network: ", list(ray.get(server.get_target_net.remote()).state_dict().keys())[:5])
    
    configs = [copy.deepcopy(config) for _ in range(1000)]
    for i, c in enumerate(configs):
        c.seed = 2**(i % 29) + 2 * i + 1
    
    env_runners = [EnvRunner.remote(config=c, agent={"type": "humanexe"}, opponent=RandomAgent(1, c.seed), server=server, buffer=buffer) for c in random.sample(configs, 10)]
    env_runners.extend([EnvRunner.remote(config=c, agent={"type": "humanexe"}, opponent={"type": "humanexe"}, server=server, buffer=buffer) for c in random.sample(configs, 10)])
    # env_runners.extend([EnvRunner.options(num_gpus=0.05).remote(config=c, agent=DQNAgent(0, None, None), opponent={"type": "humanexe"}, server=server, buffer=buffer) for c in random.sample(configs, 1)])

    for runner in env_runners:
        runner.run.remote()
        
    # Initialize optimizer and replay memory
    # train_task = train.remote(config=config, server=server, buffer=buffer)
    # ray.wait([train_task], num_returns=1)

    train(config=config, server=server, buffer=buffer)


    print("Finished training script.")
    
    
    for runner in env_runners:
        ray.kill(runner)
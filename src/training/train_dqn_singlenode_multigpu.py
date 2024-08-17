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
from torchsummary import summary


from src.agents.agent import Agent
from src.agents.human_exe_agent import HumanExeAgent
from src.agents.q_greedy_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.agents.utils.observation_receiver import ObservationReceiving
from src.environment.environment import GeneralsEnvironment
from src.models.dqn_cnn import DQN
from src.utils.replay_buffer import RayReplayBuffer, Experience
from src.agents.utils.gym_agent import GymAgent
from src.agents.utils.epsilon_random_agent import EpsilonRandomAgent


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
    target_update_freq: int = None
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
        with open(config_file, 'w') as file:
            config = {
                key: value
                for key, value in vars(self).items()
                if not key.startswith('_')
            }
            yaml.dump(config, file)

    def __str__(self):
        config = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith('_')
        }
        return f'DQNTrainingConfig({config})'

@ray.remote(num_cpus=1, memory=2 * 1024**3, max_restarts=0, max_task_retries=0)
class SharedServer:
    target_net: Dict = None
    training_steps: int = 0
    
    def __init__(self, target_net: Dict):
        self.target_net = target_net
        self.training_steps = 0
        
    def get_target_net(self):
        return self.target_net
    
    def set_target_net(self, net: Dict):
        self.target_net = net
    
    def increment_training_steps(self):
        self.training_steps += 1
        return self.training_steps
    
    def get_training_steps(self):
        return self.training_steps

@ray.remote(num_cpus=0.1)
class EnvRunner:
    buffer: ray.ObjectRef
    server: ray.ObjectRef
    agent: GymAgent
    env: GeneralsEnvironment
    
    _local_buffer: list[Experience] = []
    
    def _handle_observation(self, experience: Experience):
        self._local_buffer.append(experience)
        if len(self._local_buffer) >= 500:
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
        
    def _fetch_updates(self):
        if isinstance(self.agent.unwrapped, DQNAgent):
            dqn_agent: DQNAgent = self.agent.unwrapped
            model: torch.nn.Module = DQN(config.input_channels, config.num_actions, config.n_rows, config.n_cols).cuda().eval()
            model.load_state_dict(ray.get(self.server.get_target_net.remote()))
            dqn_agent.update_model(model, device=torch.device("cuda"))
            gc.collect()
            torch.cuda.empty_cache() 
    
    def run(self):
        rng = np.random.default_rng(self.config.seed)
        self.env.reset(seed=self.config.seed)
        episodes = 0
        self._fetch_updates()
        last_train_steps = 0
        while True:
            train_steps = ray.get(self.server.get_training_steps.remote())
            if train_steps - last_train_steps >= self.config.target_update_freq:
                self._fetch_updates()
                last_train_steps = train_steps
            agent_seed = rng.integers(0, 2**29)   
            self.env.reset()
            print("[INFO] Running episode ", episodes + 1)
            batchsize = 25
            self.agent.run_episodes(seed=agent_seed, n_runs=batchsize)
            episodes += batchsize

# Define training loop
# @ray.remote(num_cpus=1, num_gpus=0.1, memory=1*1024**3)
def train(config: DQNTrainingConfig, server: ray.ObjectRef, buffer: ray.ObjectRef):
    start_time = time.time()   
    target_net: torch.nn.Module = DQN(config.input_channels, config.num_actions, config.n_rows, config.n_cols).cuda().eval()
    target_net.load_state_dict(ray.get(server.get_target_net.remote()))
    
    policy_net = DQN(config.input_channels, config.num_actions, config.n_rows, config.n_cols).cuda().train()
    policy_net.load_state_dict(target_net.state_dict())
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=config.learning_rate)
    
    val_env_rand: GeneralsEnvironment = gym.make("generals-v0", agent=DQNAgent(0, policy_net.cuda(), None), opponent=RandomAgent(1, config.seed + (config.seed % 13) + 2 ** (config.seed % 19)), seed=config.seed, n_rows=config.n_rows, n_cols=config.n_cols)
    val_env_humanexe: GeneralsEnvironment = gym.make("generals-v0", agent=DQNAgent(0, policy_net.cuda(), None), opponent=HumanExeAgent(1), seed=config.seed, n_rows=config.n_rows, n_cols=config.n_cols)
    
    print("Waiting for buffer to fill...")
    size = ray.get(buffer.size.remote())
    while size < config.wait_buffer_size:
        print(f"Buffer size: {size}")
        time.sleep(5)
        size = ray.get(buffer.size.remote())
        
    if config.use_wandb:
        run = wandb.init(
            project=config.wandb_project,
            config=config.__dict__,
        )
    
    losses = []
    print("Starting training...")
    for train_step in trange(config.num_steps, miniters=20):
        print("Training step:", train_step, ray.get(server.increment_training_steps.remote()))
        
        data = ray.get(buffer.sample.remote(config.batch_size))
        experience_batch  = tuple([list(t) for t in zip(*data)])
        s_t, a_t, r_t_1, s_t_1, d_t_1 = experience_batch
        # print("[INFO] Sampled experience batch: ", experience_batch)
        
        # decouple (turn, grid) ObsType
        s_t = np.array([np.array(grid) for (_, grid) in s_t])
        s_t_1 = np.array([np.array(grid) for (_, grid) in s_t_1])
        
        with torch.inference_mode():
            s_t_1 = torch.tensor(s_t_1, dtype=torch.float32).cuda()
            # double DQN
            target_net_max_action = torch.argmax(target_net(s_t_1), dim=-1)
            policy_net_target_q_value: torch.Tensor = policy_net(s_t_1)[:config.batch_size, target_net_max_action]
            
        
        s_t = torch.tensor(s_t, dtype=torch.float32).cuda()
        a_t = [int(a) for a in a_t]
        r_t_1 = torch.tensor(r_t_1, dtype=torch.float32).cuda()
        d_t_1 = torch.tensor(d_t_1, dtype=torch.float32).cuda()
        
        predicted_q_vals: torch.Tensor = policy_net(s_t)[:config.batch_size, a_t]

        td_error = r_t_1 + config.gamma * policy_net_target_q_value * (1 - d_t_1) - predicted_q_vals
        loss = td_error.pow(2).mean()
        losses.append(loss.item())
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if config.use_wandb:
            grad_norm = torch.norm(torch.nn.utils.parameters_to_vector(policy_net.parameters()), 2.0)
            grad_max = torch.max(torch.nn.utils.parameters_to_vector(policy_net.parameters()))
            wandb.log({
                "loss": loss.item(),
                "q_values": predicted_q_vals.max().item(),
                "SPS": int(train_step / (time.time() - start_time)),
                "grad_norm": grad_norm.item(),
                "grad_max": grad_max.item()
            }, step=train_step)
            
        if (train_step + 1) % config.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            server.set_target_net.remote(target_net.cpu().state_dict())
            target_net.cuda()
            print("[INFO] Updated target network.")
        
        if (train_step + 1) % (config.num_steps // 100) == 0:
            print("[INFO] Running validation episodes...")
            val_env_rand.reset()
            val_env_humanexe.reset()
            reward_rand = np.array(val_env_rand.agent.run_episodes(seed=config.seed, n_runs=2)).mean()
            reward_humanexe = np.array(val_env_humanexe.agent.run_episodes(seed=config.seed, n_runs=2)).mean()
            print("[INFO] Validation rewards: ", reward_rand, "--", reward_humanexe)
            
            if config.use_wandb:
                wandb.log({
                    "val_rewards_rand": reward_rand,
                    "val_rewards_humanexe": reward_humanexe
                }, step=train_step)
            
    
    if config.use_wandb:
        run.finish()

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
    
    print("[DEBUG] Starting ray cluster...")    
    print("[DEBUG] Cluster resources: ", ray.cluster_resources())
    time.sleep(5)
    
    print("[DEBUG] Initializing buffer and target network params...")
    buffer = RayReplayBuffer.remote(config.memory_capacity, config.seed)
    
    tgt_net = DQN(config.input_channels, config.num_actions, config.n_rows, config.n_cols).cpu()
    server = SharedServer.remote(tgt_net.state_dict())
    
    print("[INFO] DQN network:")
    summary(tgt_net, (config.n_rows, config.n_cols, config.input_channels), device="cpu")
    del tgt_net
    
    time.sleep(5)
    

    print("[INFO] Initial conditions...")
    print("[INFO] Buffer size: ", ray.get(buffer.size.remote()))
    print("[INFO] Target network: ", list(ray.get(server.get_target_net.remote()).keys())[:5])
    
    configs = [copy.deepcopy(config) for _ in range(1000)]
    for i, c in enumerate(configs):
        c.seed = 2**(i % 29) + 2 * i + 1
    
    env_runners = []
    env_runners.extend([EnvRunner.remote(config=c, agent={"type": "humanexe"}, opponent=RandomAgent(1, c.seed), server=server, buffer=buffer) for c in random.sample(configs, 20)])
    time.sleep(10)
    env_runners.extend([EnvRunner.remote(config=c, agent={"type": "humanexe"}, opponent={"type": "humanexe"}, server=server, buffer=buffer) for c in random.sample(configs, 10)])
    time.sleep(10)
    env_runners.extend([EnvRunner.options(num_gpus=0.05).remote(config=c, agent=EpsilonRandomAgent(DQNAgent(0, None, None), 0.3, c.seed), opponent={"type": "humanexe"}, server=server, buffer=buffer) for c in random.sample(configs, 10)])
    time.sleep(10)
    env_runners.extend([EnvRunner.options(num_gpus=0.05).remote(config=c, agent=EpsilonRandomAgent(DQNAgent(0, None, None), 0.3, c.seed + 2**7 - 1), opponent=RandomAgent(1, c.seed), server=server, buffer=buffer) for c in random.sample(configs, 30)])
    
    
    run_tasks = [runner.run.remote() for runner in env_runners]
    
    # ray.wait(env_runners, num_returns=10)
    # ray.get(run_tasks)
    
    # train_task = train.remote(config=config, server=server, buffer=buffer)
    # ray.wait([train_task], num_returns=1)

    train(config=config, server=server, buffer=buffer)

    print("Finished training script.")
    
    
    for runner in env_runners:
        ray.kill(runner)
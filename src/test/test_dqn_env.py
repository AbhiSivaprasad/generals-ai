from pathlib import Path
import unittest

import gymnasium as gym
import torch

from src.agents.q_greedy_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.environment.environment import GeneralsEnvironment
import argparse

from src.models.dqn_cnn import DQN


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Path to the output file")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint")
    args = parser.parse_args()

    output_path = args.output
    ckpt_path = args.checkpoint
    
    dqn: DQN = torch.load(ckpt_path)
    dqn.eval().cuda()
    
    env: GeneralsEnvironment = gym.make("generals-v0", agent=DQNAgent(0, dqn, None), opponent=RandomAgent(1, 31), seed=31, n_rows=15, n_cols=15)
    env.agent.set_env(env)
    
    env.reset(seed=31)
    
    reward = float(env.agent.run_episode(11))
    
    print("Collected reward:", reward)
    
    replay_path = Path(output_path)
    env.write(replay_path)

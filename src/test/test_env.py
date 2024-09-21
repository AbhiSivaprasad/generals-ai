from pathlib import Path
import unittest

import gymnasium as gym

from src.agents.random_agent import RandomAgent
from src.environment.environment import GeneralsEnvironment
import argparse


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Path to the output file")
    args = parser.parse_args()

    output_path = args.output
    
    env: GeneralsEnvironment = gym.make("generals-v0", agent=RandomAgent(0, 3), opponent=RandomAgent(1, 31), seed=31, n_rows=15, n_cols=15)
    env.agent.set_env(env)
    
    print(env._max_episode_steps)
    env.reset()
    reward = float(env.agent.run_episode(11))
    
    print("Collected reward:", reward)
    
    replay_path = Path(output_path)
    env.write(replay_path)

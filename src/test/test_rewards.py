from pathlib import Path
import unittest

import gymnasium as gym
import numpy as np

from src.agents.random_agent import RandomAgent
from src.environment.action import Action
from src.environment.environment import GeneralsEnvironment
import argparse

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Path to the output file")
    parser.add_argument("--seed", type=int, help="Seed.")
    args = parser.parse_args()

    output_path = args.output
    seed = args.seed if args.seed else 0
    
    seeder = np.random.default_rng(seed)
    
    env: GeneralsEnvironment = gym.make("generals-v0", opponent=RandomAgent(1, 31), seed=int(seeder.integers(0, 2**29)), n_rows=15, n_cols=15)
    env.agent.set_env(env)
    
    for _ in range(10):
        env.reset(seed=int(seeder.integers(0, 2**29)))
        
        generals = env.game.state.board.generals
        generals = [(tile.x, tile.y) for tile in generals]
        
        print(generals)
        
        g1 = Action(generals[0][0], generals[0][1], 1)
        g2 = Action(generals[1][0], generals[1][1], 3)
        
        # wait for 10 ticks
        for _ in range(10):
            obs, reward, term, trunc, info = env.step(0)
            assert reward >= 0.0, "Null move should not have negative reward: " + str(reward)
        
        obs, reward, term, trunc, info = env.step(Action.to_space_sample(g1, env.game.state.board.num_rows, env.game.state.board.num_cols))
        assert reward >= 0.0, "Legal move should not have negative reward: " + str(reward)
        
        obs, reward, term, trunc, info = env.step(Action.to_space_sample(g2, env.game.state.board.num_rows, env.game.state.board.num_cols))
        assert reward < 0.0, "Illegal move should have negative reward: " + str(reward)
        
        if output_path is not None:
            replay_path = Path(output_path)
            env.write(replay_path)

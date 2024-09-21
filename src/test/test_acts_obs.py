from pathlib import Path
import unittest

import gymnasium as gym

from src.agents.random_agent import RandomAgent
from src.environment.action import Action, Direction
from src.environment.environment import GeneralsEnvironment
import argparse


if __name__ == "__main__":
    env: GeneralsEnvironment = gym.make("generals-v0", agent=RandomAgent(0, 3), opponent=RandomAgent(1, 31), seed=33, n_rows=3, n_cols=3)
    env.agent.set_env(env)
    
    for i in range(env.game.state.board.num_rows * env.game.state.board.num_cols * 4 + 1):
        print(Action.from_space_sample(i, env.game.state.board.num_rows, env.game.state.board.num_cols))
    
    print()
    print()
    
    for r in range(env.game.state.board.num_rows):
        for c in range(env.game.state.board.num_cols):
            for i in range(4):
                print(Action.to_space_sample(Action(c, r, Direction(i)), env.game.state.board.num_rows, env.game.state.board.num_cols))
    
    (turn, obs), info = env.reset()
    
    print("Game board: ", env.game.state.board.grid)
    print("Observation: ", obs)
    a = env.agent.get_action((turn, obs))
    (turn, obs), reward, term, trunc, info = env.step(a)
    print("Action: ", env.agent.action)
    print("Action #: ", Action.to_space_sample(env.agent.action, env.game.state.board.num_rows, env.game.state.board.num_cols))
    print("Action back: ", Action.from_space_sample(a, env.game.state.board.num_rows, env.game.state.board.num_cols))
    print("---------------------")
    print()

    print("Game board: ", env.game.state.board.grid)
    print("Observation: ", obs)
    a = env.agent.get_action((turn, obs))
    (turn, obs), reward, term, trunc, info = env.step(a)
    print("Action: ", env.agent.action)
    print("Action #: ", Action.to_space_sample(env.agent.action, env.game.state.board.num_rows, env.game.state.board.num_cols))
    print("Action back: ", Action.from_space_sample(a, env.game.state.board.num_rows, env.game.state.board.num_cols))
    print("---------------------")
    print()
    
    print("Game board: ", env.game.state.board.grid)
    print("Observation: ", obs)
    
    # reward = float(env.agent.run_episode(11))
    
    # print("Collected reward:", reward)
    
    # replay_path = Path(output_path)
    # env.write(replay_path)

from src.agents.random_agent import RandomAgent
from src.environment.environment import GeneralsEnvironment
import argparse

from src.environment.logger import Logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    output_file_path = args.output
    
    env = GeneralsEnvironment(
        agent=RandomAgent(0, 0),
        opponent=RandomAgent(1, 0),
        board_x_size=2,
        board_y_size=2,
    )
    
    obs, info = env.reset(logger=Logger())
    done = False
    while not done:
        obs, reward, term, trunc, info = env.step()
        done = term or trunc
    
    env.game_master.logger.write(output_file_path)
    
    
    
    
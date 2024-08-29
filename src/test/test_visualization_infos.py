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
        agents=[
            RandomAgent(0),
            RandomAgent(1)
        ],
        board_x_size=2,
        board_y_size=2,
        auxiliary_reward_weight=0.1,
    )
    obs, info = env.reset(logger=Logger())
    done = False
    term = False
    trunc = False
    while not done:
        obs, reward, term, trunc, info = env.step(actions=[a.move(obs, env)[0] for a in env.agents])
        term = list(term.values())[0]
        trunc = list(trunc.values())[0]
        done = term or trunc
    
    env.game_master.logger.write(output_file_path)
    
    
    
    
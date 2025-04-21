import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


from src.agents.human_exe_agent import HumanExeAgent
from src.agents.q_greedy_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.environment.board_generator import generate_board_state
from src.environment.game_master import GameMaster
from src.environment.logger import Logger
from src.models.dqn_cnn import DQN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Seed.")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to checkpoint")
    args = parser.parse_args()

    seed = args.seed if args.seed else 0
    
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    board = generate_board_state(15, 15, mountain_probability=0.0, city_probability=0.0)
    print(board.grid)
    
    if args.checkpoint:
        dqn: DQN = torch.load(args.checkpoint)
    else:
        dqn = DQN(7, 901, board.num_rows, board.num_cols)
    
    dqn = dqn.cuda().eval()
    
    dqn_agent = DQNAgent(0, dqn, None)
    
    logger = Logger()
    game_master = GameMaster(
        board, players=[dqn_agent, RandomAgent(1, 2**12 + 13 - seed)], logger=logger, max_turns=500
    )
    game_master.play()

    replays_dir = Path("resources/replays")
    replays_dir.mkdir(parents=True, exist_ok=True)

    replay_path = replays_dir / "test_dqn_replay.json"
    logger.write(replay_path)

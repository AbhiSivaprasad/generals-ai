import json
import os
from pathlib import Path
import random


from src.agents.human_exe_agent import HumanExeAgent
from src.agents.random_agent import RandomAgent
from src.environment.board_generator import generate_board_state
from src.environment.game_master import GameMaster
from src.environment.logger import Logger


if __name__ == "__main__":
    random.seed(1)
    board = generate_board_state(15, 15, mountain_probability=0.0, city_probability=0.0)
    print(board.grid)
    
    logger = Logger()
    game_master = GameMaster(
        board, players=[HumanExeAgent(0), RandomAgent(1, 0)], logger=logger, max_turns=500
    )
    game_master.play()

    replays_dir = Path("resources/replays")
    replays_dir.mkdir(parents=True, exist_ok=True)

    replay_path = replays_dir / "humanexe_random_replay.txt"
    logger.write(replay_path)

    # read json file replay_path
    with open(replay_path, "r") as f:
        data = json.load(f)

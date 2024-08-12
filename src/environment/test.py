import json
import os
from pathlib import Path

from src.agents.random_agent import RandomAgent
from src.environment.board_generator import generate_board_state
from src.environment.game_master import GameMaster
from src.environment.logger import Logger



logger = Logger()
board = generate_board_state(15, 15, mountain_probability=0.2, city_probability=0.03)
game_master = GameMaster(
    board, players=[RandomAgent(0), RandomAgent(1)], logger=logger, max_turns=1000
)
game_master.play()

replays_dir = Path("resources/replays")
replays_dir.mkdir(parents=True, exist_ok=True)
logger.write(replays_dir / "test_replay.txt")

replays_dir = Path("resources/replays")
replay_path = replays_dir / "test_replay.txt"

# read json file replay_path
with open(replay_path, "r") as f:
    data = json.load(f)

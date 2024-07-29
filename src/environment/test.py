import os
from pathlib import Path

from src.agents.random_agent import RandomAgent
from src.environment.board_generator import generate_board_state
from src.environment.game_master import GameMaster
from src.environment.logger import Logger


board = generate_board_state(15, 15)

logger = Logger()
game_master = GameMaster(
    board, players=[RandomAgent(0), RandomAgent(1)], logger=logger, max_turns=1000
)
game_master.play()

replays_dir = Path("resources/replays")
replays_dir.mkdir(parents=True, exist_ok=True)
logger.write(replays_dir / "test_replay.txt")

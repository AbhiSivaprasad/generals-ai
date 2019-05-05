import sys
sys.path.insert(0, '../../')

import json

from src.board_generator import BoardGenerator
from src.game_master import GameMaster
from src.graphics.board import Board
from src.logger import Logger
from src.players.random_player import RandomPlayer


for i in range(1):
    bg = BoardGenerator()
    board = bg.generate_board_state(18, 18)

    print(board.serialize())

    logger = Logger(num_players=2)

    game_master = GameMaster(board, players=[RandomPlayer(), RandomPlayer()], logger=logger)
    game_master.play()

    with open("../../resources/replays/temp2.txt", "w") as f:
        f.write(json.dumps(logger.output()))

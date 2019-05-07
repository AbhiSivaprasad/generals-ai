import sys
sys.path.insert(0, '../../')

import json
import sys
sys.path.insert(0, '../../')

from src.board_generator import BoardGenerator
from src.game_master import GameMaster
from src.graphics.board import Board
from src.logger import Logger
from src.players.random_player import RandomPlayer
from src.players.deep_general import DeepGeneral
from src.graphics.generals import Generals

for i in range(1):

    # bg = BoardGenerator()
    # board = bg.generate_board_state(15, 15)

    # print(board.serialize())

    logger = Logger(num_players=2)

    player = DeepGeneral()

    g = Generals(userid='cpsc663_bot_uid', 
                 # username='[Bot] head honcho', # already set
                 mode='private', 
                 gameid='CPSC663_test_bot_lobby',
                 force_start=True
                )

    for update in g.get_updates():


      print("update:", update)

      startx, starty, destx, desty = player.move(board)
      g.move(startx, starty, destx, desty)

      # get position of your general
      # pi = update['player_index']
      # y, x = update['generals'][pi]

      # move units from general to arbitrary square
      # for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      #     if (0 <= y+dy < update['rows'] and 0 <= x+dx < update['cols']
      #             and update['tile_grid'][y+dy][x+dx] != generals.MOUNTAIN):
      #         g.move(y, x, y+dy, x+dx)
      #         break


    # game_master = GameMaster(board, players=[RandomPlayer(), RandomPlayer()], logger=logger)
    # game_master.play()

    with open("../../resources/replays/temp2.txt", "w") as f:
        f.write(json.dumps(logger.output()))

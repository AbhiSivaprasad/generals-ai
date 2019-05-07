import json
import sys
sys.path.insert(0, '../../')

from src.board_generator import BoardGenerator
# from src.game_master import GameMaster
from src.graphics.tile import Tile
from src.graphics.board import Board
from src.logger import Logger
from src.players.random_player import RandomPlayer
# from src.players.deep_general import DeepGeneral
from src.graphics.generals import Generals
from src.graphics.constants import *

for i in range(1):

    # bg = BoardGenerator()
    # board = bg.generate_board_state(15, 15)

    # print(board.serialize())

    logger = Logger(num_players=2)

    # player = DeepGeneral()
    player = RandomPlayer()

    g = Generals(userid='cpsc663_bot_uid', 
                 # username='[Bot] head honcho', # already set
                 mode='private', 
                 gameid='CPSC663_test_bot_lobby',
                 force_start=True
                )

    for update in g.get_updates():


      print("update:", update)
      row = update['rows']
      col = update['cols']
      pi = update['player_index']
      gen_y, gen_x = update['generals'][pi]
      army_grid = update['army_grid']
      tile_grid = update['tile_grid']
      generals = update['generals']
      cities = update['cities']

      board = Board(rows=row, cols=col, player_index=None)

      grid = [  # 2D List of Tile Objects
              [Tile(board, x, y) for x in range(col)]
              for y in range(row)
            ]

      print(len(grid), len(grid[0]), row, col)

      # place terrain
      for x in range(row):
        for y in range(col):
          tile = grid[x][y]

          tile.type = tile_grid[x][y]
          tile.army = army_grid[x][y]

          if x is gen_x and y is gen_y:
            tile.is_general = True
            board.generals[0] = tile

          if (x,y) in cities:
            tile.is_city = True
            board.cities.append(tile)
          

      board.set_grid(grid)

      startx, starty, destx, desty = player.move(board)
      g.move(startx, starty, destx, desty)


    # game_master = GameMaster(board, players=[RandomPlayer(), RandomPlayer()], logger=logger)
    # game_master.play()

    with open("../../resources/replays/temp2.txt", "w") as f:
        f.write(json.dumps(logger.output()))

import json
import sys
sys.path.insert(0, '../../')

from src.board_generator import BoardGenerator
from src.move import Move
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

    # game_master = GameMaster(board, players=[RandomPlayer(), RandomPlayer()], logger=logger)
    # game_master.play()

    logger = Logger(num_players=2)

    # player = DeepGeneral()
    player = RandomPlayer()

    g = Generals(userid='cpsc663_bot_uid', 
                 # username='[Bot] head honcho', # already set
                 mode='private', 
                 gameid='CPSC663_test_bot_lobby',
                 force_start=True,
                 col=20,  row=18
                )

    for update in g.get_updates():
      if update['complete']:
        print(update['result'])
      elif not update['bad_board']:
        board = update['board']

        move = player.move(board)
        print(move)

        g.move(move.startx, move.starty, move.destx, move.desty)
      
 

    with open("../../resources/replays/temp2.txt", "w") as f:
        f.write(json.dumps(logger.output()))

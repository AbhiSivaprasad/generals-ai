# import threading
# import time
# from src.graphics.map import Map
#
# # create valid Map object
# from src.graphics.viewer import GeneralsViewer
#
# # start_data:
# # {
# # 	'playerIndex': int,
# # 	'usernames': [
# # 		str ordered by player index
# # 	]
# # }
# # data:
# # {
# # 	'stars': [int ordered by playerIndex],
# # 	'scores': [
# # 		{
# # 			'i': int playerIndex,
# # 			'dead': boolean is player dead or alive,
# # 			'total': int,
# # 			'tiles': int
# # 		}
# # 	],
# # 	'generals': [int generalsIndex ordered by playerIndex]
# # }
#
# map = Map(start_data={
#     'usernames': [
#         'Player 1',
#         'Player 2'
#     ], 'swamps': [
#
#     ],
#     'playerIndex': 1
# }, data={
#     'turn': 5,
#     'stars': [40, 50],
#     'scores': [
#         {
#             'i': 0,
#             'dead': False,
#             'total': 120,
#             'tiles': 15
#         },
#         {
#             'i': 1,
#             'dead': False,
#             'total': 85,
#             'tiles': 40
#         }
#     ],
#     'generals': [1, 8],
#     'map_diff': [0, 20,
#                  3, 3,
#                  0, 5, 1, 0, 1, 0, 2, 2, 0,
#                  -4, 0, 0, -3, 1, -2, 0, 0, -1],
#     'cities_diff': [0, 3, 0, 4, 7]
# })
#
# gv = GeneralsViewer("window", moveEvent=None)
#
#
# def fun():
#     gv.mainViewerLoop() # mainThread
#
#
# def _create_thread(f):
#     t = threading.Thread(target=f)
#     t.daemon = True
#     t.start()
#
#
# _create_thread(fun)
# gv.updateGrid(map)
# map2 = map.update(data={
#     'turn': 6,
#     'stars': [40, 50],
#     'scores': [
#         {
#             'i': 0,
#             'dead': False,
#             'total': 120,
#             'tiles': 15
#         },
#         {
#             'i': 1,
#             'dead': False,
#             'total': 85,
#             'tiles': 40
#         }
#     ],
#     'generals': [1, 8],
#     'map_diff': [20],
#     'cities_diff': [3]
# })
# gv.updateGrid(map2)
# time.sleep(200000)
import json

from src.board_generator import BoardGenerator
from src.game_master import GameMaster
from src.graphics.board import Board
from src.logger import Logger
from src.players.random_player import RandomPlayer


for i in range(1):
    bg = BoardGenerator()
    board = bg.generate_board_state(15, 15)

    print(board.serialize())

    logger = Logger(num_players=2)

    game_master = GameMaster(board, players=[RandomPlayer(), RandomPlayer()], logger=logger)
    game_master.play()

    with open("../../resources/replays/temp.txt", "w") as f:
        f.write(json.dumps(logger.output()))

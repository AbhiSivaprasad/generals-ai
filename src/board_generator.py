import random as rand
import math

from src.graphics.board import Board
from src.graphics.constants import TILE_EMPTY, TILE_MOUNTAIN
from src.graphics.tile import Tile


class BoardGenerator:
    """
    responsible for automatically creating board configurations
    """
    def generate_board_state(self, row, col):
        """
        contains logic to create board state of certain size

        :param row: number of rows in board
        :param col: number of cols in board
        :return:
        """
        mountain_probability = rand.random() * 0.05 + 0.15  # random choice in interval [0.20, 0.25]
        city_probability = rand.random() * 0.02 + 0.05     # random choice in interval [0.07, 0.09]

        # place generals
        min_distance = math.ceil((2 / 3) * min(row, col))  # generals shouldn't be placed too close

        while True:
            p1_general_position = self._random_position(row, col)
            p2_general_position = self._random_position(row, col)

            if self._get_distance(p1_general_position, p2_general_position) >= min_distance:
                break

        # generate random boards until one is valid
        board = Board(rows=row, cols=col, player_index=None)
        grid = [  # 2D List of Tile Objects
            [Tile(self, x, y) for x in range(col)]
            for y in range(row)
        ]

        while True:
            # place terrain
            for x in range(row):
                for y in range(col):
                    tile = grid[x][y]

                    if (x, y) == p1_general_position:
                        # player one's general
                        tile.type = 0            # player index
                        tile.army = 1            # initialize one troop on general
                        tile.is_general = True
                        board.generals[0] = tile
                    elif (x, y) == p2_general_position:
                        # player two's general
                        tile.type = 1            # player index
                        tile.army = 1            # initialize one troop on general
                        tile.is_general = True
                        board.generals[1] = tile
                    elif rand.random() < city_probability:
                        # tile is a city
                        tile.type = TILE_EMPTY
                        tile.army = self._random_city_size()
                        tile.is_city = True
                        board.cities.append(tile)
                    elif rand.random() < mountain_probability:
                        # tile is a mountain
                        tile.type = TILE_MOUNTAIN
                    else:
                        # tile is empty
                        tile.type = TILE_EMPTY

            # set board grid
            board.set_grid(grid)

            # validate board
            if board.generals[0].path_to(board.generals[1]):
                # path between generals exists so map is valid
                break

            # grid was not valid so reset
            grid = [
                [Tile(self, x, y) for x in range(col)]
                for y in range(row)
            ]

        return board

    @staticmethod
    def _random_city_size():
        return rand.randint(40, 50)

    @staticmethod
    def _random_position(row, col):
        return rand.randint(0, row - 1), rand.randint(0, col - 1)

    @staticmethod
    def _get_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
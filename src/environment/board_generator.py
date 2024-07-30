import random as rand
import math

from src.environment.board import Board
from src.environment.tile import Tile, TileType


MIN_CITY_SIZE = 40
MAX_CITY_SIZE = 50


def generate_board_state(
    num_rows: int,
    num_cols: int,
    mountain_probability: float = 0,
    city_probability: float = 0,
    min_ratio_of_generals_distance_to_board_side: float = 2.0 / 3,
) -> Board:
    """Create a new game board with the given parameters"""

    # generate random boards until one is valid
    while True:
        board = generate_candidate_board_state(
            num_rows=num_rows,
            num_cols=num_cols,
            mountain_probability=mountain_probability,
            city_probability=city_probability,
            min_ratio_of_generals_distance_to_board_side=min_ratio_of_generals_distance_to_board_side,
        )

        # validate board
        if board.path_exists_between(board.generals[0], board.generals[1]):
            # path between generals exists so map is valid
            break

    return board


def generate_candidate_board_state(
    num_rows: int,
    num_cols: int,
    mountain_probability: float = 0,
    city_probability: float = 0,
    min_ratio_of_generals_distance_to_board_side: float = 2 / 3,
):
    # generals shouldn't be placed too close to each other
    min_distance = math.ceil(
        min_ratio_of_generals_distance_to_board_side * min(num_rows, num_cols)
    )

    board = Board(num_rows=num_rows, num_cols=num_cols, player_index=None)
    while True:
        p1_general_position = _random_position(num_rows, num_cols)
        p2_general_position = _random_position(num_rows, num_cols)

        if _get_distance(p1_general_position, p2_general_position) >= min_distance:
            break

    grid = [[Tile(board, x, y) for x in range(num_cols)] for y in range(num_rows)]

    # place terrain
    for x in range(num_rows):
        for y in range(num_cols):
            tile = grid[x][y]

            if (x, y) == p1_general_position:
                tile.type = TileType.GENERAL
                tile.player_index = 0
                tile.army = 1  # initialize with one troop on general
                board.generals[0] = tile
            elif (x, y) == p2_general_position:
                tile.type = TileType.GENERAL
                tile.player_index = 1
                tile.army = 1
                board.generals[1] = tile
            elif rand.random() < city_probability:
                tile.type = TileType.CITY
                tile.army = _random_city_size()
                board.cities.append(tile)
            elif rand.random() < mountain_probability:
                tile.type = TileType.MOUNTAIN
            else:
                tile.type = TileType.NORMAL
                tile.army = 0

    # set board grid
    board.set_grid(grid)

    # add vision to generals
    board.add_vision(board.generals[0], 0)
    board.add_vision(board.generals[1], 1)
    return board


def _random_city_size():
    return rand.randint(MIN_CITY_SIZE, MAX_CITY_SIZE)


def _random_position(row, col):
    return rand.randint(0, row - 1), rand.randint(0, col - 1)


def _get_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


if __name__ == "__main__":
    board = generate_board_state(15, 15)
    print(board.serialize())
    print(board.generals)
    print(board.cities)
    print(board.is_valid_position(0, 0))
    print(board.path_exists_between(board.generals[0], board.generals[1]))

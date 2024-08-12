from typing import Optional
import numpy as np
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
    rng: np.random.Generator = np.random.default_rng(0)
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
            rng=rng
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
    rng: np.random.Generator = np.random.default_rng(0)
):
    # generals shouldn't be placed too close to each other
    min_distance = math.ceil(
        min_ratio_of_generals_distance_to_board_side * min(num_rows, num_cols)
    )

    board = Board(num_rows=num_rows, num_cols=num_cols)
    while True:
        p1_general_position = _random_position(num_rows, num_cols, rng)
        p2_general_position = _random_position(num_rows, num_cols, rng)

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
            elif rng.random() < city_probability:
                tile.type = TileType.CITY
                tile.army = _random_city_size()
                board.cities.append(tile)
            elif rng.random() < mountain_probability:
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


def _random_city_size(rng: Optional[np.random.Generator]):
    if rng:
        return rng.integers(MIN_CITY_SIZE, MAX_CITY_SIZE, endpoint=True)
    return np.random.randint(MIN_CITY_SIZE, MAX_CITY_SIZE)


def _random_position(row, col, rng: Optional[np.random.Generator]):
    if rng:
        return rng.integers(0, row), rng.integers(0, col)
    return np.random.randint(0, row), np.random.randint(0, col)


def _get_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


if __name__ == "__main__":
    board = generate_board_state(15, 15)
    print(board.serialize())
    print(board.generals)
    print(board.cities)
    print(board.is_valid_position(0, 0))
    print(board.path_exists_between(board.generals[0], board.generals[1]))

from collections import deque
from typing import List, Set
from src.environment.action import Action, convert_direction_to_vector
from src.environment.tile import Tile, TileType


class Board:
    """
    Board a representation of the board with necessary helper methods
    """

    def __init__(self, num_rows: int, num_cols: int, player_index: int):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cities: List[Tile] = []
        self.generals: List[Tile] = [None, None]
        self.grid: List[List[Tile]] = None

    # Public Helper Methods
    def set_grid(self, grid):
        self.grid = grid
        self._set_neighbors()

    def terminal_status(self):
        """
        check if game is over
        :return: if unfinished return -1 else return player index of victory
        """
        if self.generals[1].player_index == 0:  # player 0 captured player 1's general
            return 0
        elif self.generals[0].player_index == 1:  # player 1 captured player 0's general
            return 1
        else:
            return -1

    def is_valid_position(self, x: int, y: int):
        return (
            0 <= y < self.num_rows
            and 0 <= x < self.num_cols
            and self.grid[y][x].type != TileType.MOUNTAIN
        )

    def path_exists_between(self, source: Tile, dest: Tile, include_cities=False):
        frontier = deque()
        frontier.append(source)
        visited = set()

        while len(frontier) > 0:
            current = frontier.popleft()

            if current == dest:  # Found Destination
                return True

            for next in current.neighbors(include_cities=include_cities):
                if next not in visited:
                    frontier.append(next)
                    visited.add(next)

        return False

    def serialize(self):
        """
        return dictionary containing board state ready to be json dumped
        :param board: Board instance
        :return: dict
        """
        return [[tile.serialize() for tile in row] for row in self.grid]

    def is_action_valid(self, action: Action, player_index: int):
        """
        Check if player's action is valid

        0. waiting is a valid action
        1. start tile is within bounds and not a mountain
        2. destination tile is within bounds and not a mountain
        3. player owns start tile
        4. more than one troop on start tile
        """
        # check if agent decides to wait
        if action.do_nothing:
            return True

        # check that start tile is within bounds and not a mountain
        if not self.is_valid_position(action.startx, action.starty):
            return False

        start_tile = self.grid[action.starty][action.startx]

        # check that destination tile is within bounds and not a mountain
        destination_tile = self._get_destination_tile(start_tile, action)
        if destination_tile is None:
            return False

        print('tile owner is', start_tile.player_index, 'player index is', player_index)
        # check that player owns start tile
        if start_tile.player_index != player_index:
            return False

        # check for more than one troop on start tile
        if start_tile.army is None or start_tile.army <= 1:
            return False

        return True

    def add_troops_to_board(self):
        """increment all troops on captured cities or generals"""
        # only increment troops on even turns
        if self.turn % 2 == 1:
            return

        for i in range(self.board.num_rows):
            for j in range(self.board.num_cols):
                tile = self.board.grid[i][j]

                # increment generals and captured cities every 2 turns
                # increment player's land every 50 turns
                if (
                    tile.type == TileType.GENERAL
                    or (tile.type == TileType.CITY and tile.player_index is not None)
                    or (tile.type == TileType.NORMAL and self.turn % (25 * 2) == 0)
                ):
                    tile.army += 1

    def surrounding_tiles(self, tile: Tile):
        tiles = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == dy == 0:
                    continue

                if (
                    0 <= tile.y + dy <= self.num_rows - 1
                    and 0 <= tile.x + dx <= self.num_cols - 1
                ):
                    tiles.append(self.grid[tile.y + dy][tile.x + dx])
        return tiles

    def add_vision(self, tile: Tile, player_index: int):
        """
        add vision for player to all tiles surrounding provided tile
        """
        updated_tiles = []
        for tile in self.surrounding_tiles(tile):
            if not tile.player_visibilities[player_index]:
                tile.player_visibilities[player_index] = True
                updated_tiles.append(tile)
        return updated_tiles

    def update_vision_from_captured_tile(self, tile: Tile, player_index: int):
        """
        given a view and a tile remove vision produced by the given tile
        """
        # the player should not own the tile if the vision the tile produces is removed
        updated_tiles = []
        for tile in self.surrounding_tiles(tile):
            old_vision = tile.player_visibilities[player_index]
            self.update_vision(tile, player_index)
            if old_vision != tile.player_visibilities[player_index]:
                updated_tiles.append(tile)  # tile vision changed
        return updated_tiles

    def update_vision(self, tile: Tile, player_index: int):
        """
        given a tile, compute the vision of the tile
        """
        has_vision = False
        for tile in self.surrounding_tiles(tile):
            if tile.player_index == player_index:
                # found a surrounding tile owned by the player, so add vision
                has_vision = True

        tile.player_visibilities[player_index] = has_vision

    def get_player_score(self, player_index: int) -> int:
        score = 0
        for row in self.grid:
            for tile in row:
                if tile.player_index == player_index:
                    score += tile.army
        return score

    def _get_destination_tile(self, start_tile: Tile, action: Action):
        direction_vector = convert_direction_to_vector(action.direction)
        dest_x = start_tile.x + direction_vector[0]
        dest_y = start_tile.y + direction_vector[1]
        if not self.is_valid_position(dest_x, dest_y):
            return None
        else:
            return self.grid[dest_y][dest_x]

    def _set_neighbors(self):
        for x in range(self.num_cols):
            for y in range(self.num_rows):
                self.grid[y][x].set_neighbors(self)

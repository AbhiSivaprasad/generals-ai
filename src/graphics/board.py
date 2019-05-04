from src.graphics.constants import TILE_MOUNTAIN
from src.graphics.tile import Tile


class Board:
    """
    Board a representation of the board with necessary helper methods
    """
    def __init__(self, rows, cols, player_index):
        self.rows = rows                  # Integer Number Grid Rows
        self.cols = cols                  # Integer Number Grid Cols
        self.player_index = player_index
        self.grid = None                  # 2D List of Tile Objects: grid[x][y] = Tile object at position (x, y)
        self.cities = []                  # List of City Tiles
        self.generals = [None, None]      # List of Generals (x, y)
        self.tiles = []
        self.legal_moves = set()          # Set of Move objects representing legal actions
                                          # Will be updated by GameMaster throughout the course of the game

    # Public Helper Methods
    def set_grid(self, grid):
        self.grid = grid
        self._set_neighbors()

    def terminal_status(self):
        """
        check if game is over
        :return: if unfinished return -1 else return player index of victory
        """
        # only game master should check terminal status which has full vision
        assert(self.generals[0] is not None and self.generals[1] is not None)

        if self.generals[1].type == 0:    # player 0 captured player 1's general
            return 0
        elif self.generals[0].type == 1:  # player 1 captured player 0's general
            return 1
        else:
            return -1

    def is_valid_position(self, x, y):
        return 0 <= y < self.rows and 0 <= x < self.cols and self.grid[y][x].type != TILE_MOUNTAIN

    def serialize(self):
        """
        return dictionary containing board state ready to be json dumped
        :param board: Board instance
        :return: dict
        """
        return [
            [{
                'type': tile.type,
                'army': tile.army,
                'isCity': tile.is_city,
                'isGeneral': tile.is_general
            } for tile in row]
            for row in self.grid
        ]

    def remove_city(self, tile):
        if not tile.is_city:
            return

        for city in self.cities:
            if (city.x, city.y) == (tile.x, tile.y):
                self.cities.remove(city)

    # Private Methods
    def _set_neighbors(self):
        for x in range(self.cols):
            for y in range(self.rows):
                self.grid[y][x].set_neighbors(self)

from enum import Enum
from queue import Queue
from typing import List, Optional


DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


class TileType(Enum):
    NORMAL = "normal"
    MOUNTAIN = "mountain"
    GENERAL = "general"
    CITY = "city"


class Tile(object):
    x: int
    y: int
    type: TileType
    player_visibilities: Optional[List[bool]]
    player_index: Optional[int]
    army: int
    
    def __init__(
        self,
        board,
        x: int,
        y: int,
        type: Optional[TileType] = None,
        player_visibilities: Optional[List[bool]] = None,
        player_index: Optional[int] = None,
        army: int = 0,
    ):
        # tile properties
        self.x = x
        self.y = y
        self.type = type
        self.player_visibilities = player_visibilities
        self.player_index = player_index
        self.army = army

        if self.player_visibilities is None:
            self.player_visibilities = [False, False]

        # convenience pointer to board
        self._board = board

    def __repr__(self):
        return "(%d,%d) %s (%d)" % (self.x, self.y, str(self.type), self.army)

    def serialize(self):
        return {
            "x": self.x,
            "y": self.y,
            "type": self.type.value,
            "army": self.army,
            "player_index": self.player_index,
            "player_visibilities": self.player_visibilities.copy(),
        }

    def set_neighbors(self, board):
        self._board = board
        self._set_neighbors()

    def neighbors(self, include_cities=True):
        neighbors = []
        for tile in self._neighbors:
            if tile.type != TileType.CITY or (
                include_cities and tile.type == TileType.CITY
            ):
                neighbors.append(tile)
        return neighbors

    def _set_neighbors(self):
        x = self.x
        y = self.y

        neighbors = []
        for dy, dx in DIRECTIONS:
            if self._board.is_valid_position(x + dx, y + dy):
                tile = self._board.grid[y + dy][x + dx]
                neighbors.append(tile)

        self._neighbors = neighbors
        return neighbors

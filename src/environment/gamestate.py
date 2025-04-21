from dataclasses import dataclass
from typing import List, Tuple
from itertools import product as cartesian


import numpy as np

from src.environment import MAX_SIZE, ObsType
from src.environment.board import Board
from src.environment.tile import Tile, TileType

@dataclass
class GameState(object):
    board: Board
    land_counts: List[int] # not functional / not in use yet
    army_counts: List[int] # not functional / not in use yet
    turn: int
    terminal_status: int = -1
    
    def to_observation(self, player_index: int, fog_of_war: bool = True) -> ObsType:
        state = self
        # [army, 1-hot general, 1-hot city, 1-hot mountain, 1-hot in-bounds, 0/1 is_mine, 0/1 visible
        board_r, board_c = len(state.board.grid), len(state.board.grid[0])
        obs = np.zeros((board_r, board_c, 7), dtype=np.float32)
        for r, c in cartesian(range(board_r), range(board_c)):
            if 0 <= r < board_r and 0 <= c < board_c:
                tile = state.board.grid[r][c]
                obs[r, c, 4] = 1
                if tile.player_visibilities[player_index] or not fog_of_war:
                    obs[r, c, 0] = tile.army
                    obs[r, c, 1] = 1 if tile.type == TileType.GENERAL else 0
                    obs[r, c, 2] = 1 if tile.type == TileType.CITY else 0
                    obs[r, c, 3] = 1 if tile.type == TileType.MOUNTAIN else 0
                    obs[r, c, 5] = (tile.player_index == player_index)
                    obs[r, c, 6] = 1
        return self.turn, obs
    
    @classmethod
    def from_observation(cls, obs: ObsType, player_index: int) -> "GameState":
        turn, grid = obs
        board = Board(grid.shape[0], grid.shape[1])
        tile_grid = [[Tile(board, x, y) for x in range(grid.shape[1])] for y in range(grid.shape[0])]
        board.set_grid(tile_grid)
        

        # each tile vector is [army, 0/1 general, 0/1 city, 0/1 mountain, 0/1 in-bounds, 0/1 player_id, 0/1 visible]
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                tile = board.grid[r][c]
                tile.army = int(grid[r, c, 0])
                tile.type = TileType.GENERAL if grid[r, c, 1] else TileType.CITY if grid[r, c, 2] else TileType.MOUNTAIN if grid[r, c, 3] else TileType.NORMAL
                tile.player_index = player_index if bool(int(grid[r, c, 5])) else 1 - player_index
                if tile.type == TileType.GENERAL:
                    board.generals[tile.player_index] = tile
                if tile.type == TileType.CITY:
                    board.cities.append(tile)
                tile.player_visibilities[player_index] = bool(int(grid[r, c, 6]))
        
        terminal_status = board.terminal_status()

        return cls(board, [], turn, terminal_status)
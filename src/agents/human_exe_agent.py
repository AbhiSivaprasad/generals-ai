import numpy as np

from typing import Optional
import traceback

import src.agents.human_exe_cleaned

from src.agents.agent import Agent
from src.environment.board import Board
from src.environment.gamestate import GameState
from src.environment.tile import Tile, TileType

from src.environment.action import Action, Direction, convert_vector_to_direction

from src.agents.human_exe_cleaned.bot_ek0x45 import EklipZBot as HumanExeBot
from src.agents.human_exe_cleaned.base.client.map import MapBase as HumanExeMap, Score
import src.agents.human_exe_cleaned.base.client.tile as HumanExeTile


# utils
def get_exe_tile(tile: Tile, player_index: int, board: Board) -> HumanExeTile:
    x, y, army = tile.x, tile.y, tile.army
    army = 0 if army is None else army
    exe_tile = HumanExeTile.Tile(x, y, army=army, tileIndex=y * board.num_cols + x)
    exe_tile._player = tile.player_index if tile.player_index is not None else -1
    
    make_exe_updates(exe_tile, tile, player_index)
    
    return exe_tile

def make_exe_updates(exe_tile: HumanExeTile, tile: Tile, player_index: int):
    exe_tile.army = 0 if tile.army is None else tile.army
    exe_tile._player = tile.player_index if tile.player_index is not None else -1
    
    if tile.type == TileType.CITY:
        if tile.player_visibilities[player_index]:
            exe_tile.isCity = True
            exe_tile.tile = HumanExeTile.TILE_EMPTY if tile.player_index is None else tile.player_index
        else:
            exe_tile.tile = HumanExeTile.TILE_OBSTACLE
    elif tile.type == TileType.GENERAL:
        if tile.player_visibilities[player_index]:
            exe_tile.isGeneral = True
            exe_tile.tile = tile.player_index
        else:
            exe_tile.tile = HumanExeTile.TILE_FOG
    elif tile.type == TileType.MOUNTAIN:
        if tile.player_visibilities[player_index]:
            exe_tile.tile = HumanExeTile.TILE_MOUNTAIN
            exe_tile.isMountain = True
        else:
            exe_tile.tile = HumanExeTile.TILE_OBSTACLE
    elif tile.type == TileType.NORMAL:
        if tile.player_visibilities[player_index]:
            exe_tile.tile = HumanExeTile.TILE_EMPTY if tile.player_index is None else tile.player_index
        else:
            exe_tile.tile = HumanExeTile.TILE_FOG
    else:
        print("[ERROR] unknown tile type")


def get_map_ready(map: HumanExeMap):
    map.update_turn(map.turn)
    for x in range(map.cols):
        for y in range(map.rows):
            realTile = map.grid[y][x]

            hasVision = realTile.player == map.player_index or any(
                filter(lambda tile: tile.player == map.player_index, realTile.adjacents))

            if hasVision:
                map.update_visible_tile(realTile.x, realTile.y, realTile.tile, realTile.army, realTile.isCity, realTile.isGeneral)
                if realTile.isMountain:
                    map.grid[realTile.y][realTile.x].tile = HumanExeTile.TILE_MOUNTAIN
                    map.grid[realTile.y][realTile.x].isMountain = True
            else:
                if realTile.isCity or realTile.isMountain:
                    tile = map.GetTile(realTile.x, realTile.y)
                    tile.isMountain = False
                    tile.tile = HumanExeTile.TILE_OBSTACLE
                    tile.army = 0
                    tile.isCity = False
                    # tile.discovered = realTile.discovered
                    map.update_visible_tile(realTile.x, realTile.y, HumanExeTile.TILE_OBSTACLE, tile_army=0, is_city=False, is_general=False)
                else:
                    map.update_visible_tile(realTile.x, realTile.y, HumanExeTile.TILE_FOG, tile_army=0, is_city=False, is_general=False)

    map.update_scores([Score(0, 1, 1, 0), Score(1, 1, 1, 0)])
    map.update(bypassDeltas=True)


class HumanExeAgent(Agent):  
    humanexe: HumanExeBot
    exemap: HumanExeMap
    
    def __init__(self, player_index, *args, **kwargs):
        self.humanexe = HumanExeBot()
        super().__init__(player_index, *args, **kwargs)
    
    def reset(self):
        self.humanexe = HumanExeBot()
        
    def move(self, state: GameState) -> Optional[Action]:
        board = state.board
        
        if not self.humanexe.isInitialized:
            exe_tiles = [[get_exe_tile(tile, self.player_index, board) for tile in row] for row in board.grid]
            self.exemap = HumanExeMap(self.player_index, teams=None, user_names=["p1", "p2"], turn=(state.turn + 1), map_grid_y_x=exe_tiles, replay_url="")
            general_x, general_y = board.generals[self.player_index].x, board.generals[self.player_index].y
            self.exemap.generals[self.player_index] = self.exemap.grid[general_y][general_x]
            self.humanexe.initialize_from_map_for_first_time(self.exemap)
            self.humanexe.city_expand_plan = None
            self.humanexe.timings = None
            self.humanexe.curPath = None
            self.humanexe.cached_scrims.clear()
            self.humanexe._expansion_value_matrix = None
            self.humanexe.targetingArmy = None
            self.humanexe.armyTracker.lastTurn = -1
            
            get_map_ready(self.exemap)
            
        
        self.exemap.update_turn(state.turn + 1)

        # send updates for tiles they lost vision of
        # send updates for tiles they can see
        tiles = list(self.exemap.get_all_tiles())
        for tile in tiles:
            new_tile_board: Tile = board.grid[tile.y][tile.x]
            make_exe_updates(tile, new_tile_board, self.player_index)
            
            # the way the game client works, it always 'updates' every tile on the players map even if it didn't get a server update, that's why the deltas were ghosting in the sim
            # if tile in self.tiles_updated_this_cycle:
            # print(type(tile))
            playerHasVision = (not tile.isNeutral and tile.player == self.player_index) or any(filter(lambda adj: not adj.isNeutral and adj.player == self.player_index, tile.adjacents))
            if playerHasVision:
                # print(f"player has vision of {tile.x}, {tile.y}, updating")
                self.exemap.update_visible_tile(tile.x, tile.y, tile.tile, tile.army, tile.isCity, tile.isGeneral)
            else:
                tileVal = HumanExeTile.TILE_FOG
                if (tile.isMountain or tile.isCity) and not tile.isGeneral:
                    tileVal = HumanExeTile.TILE_OBSTACLE
                self.exemap.update_visible_tile(tile.x, tile.y, tileVal, tile_army=0, is_city=False, is_general=False)

        own_tiles = len([tile for tile in tiles if tile.player == self.player_index])
        own_armies = sum([tile.army for tile in tiles if tile.player == self.player_index])
        their_tiles = len([tile for tile in tiles if tile.player == 1 - self.player_index])
        their_armies = sum([tile.army for tile in tiles if tile.player == 1 - self.player_index])
        
        scores = [None] * 2
        scores[self.player_index] = Score(self.player_index, own_armies, own_tiles, 0)
        scores[1 - self.player_index] = Score(1-self.player_index, their_armies, their_tiles, 0)
        
        self.exemap.update_scores(scores)
        self.exemap.update()
        
        # print(f"turn: {state.turn}")
        
        with self.humanexe.perf_timer.begin_move(self.exemap.turn) as moveTimer:
            try:
                move = self.humanexe.find_move()
                if move is not None:
                    # print("move: ", move.__str__() + (" ## trying a split" if move.move_half else ""))
                    dx = move.dest.x - move.source.x
                    dy = move.dest.y - move.source.y
                    dir: Direction = convert_vector_to_direction((dx, dy))
                    action = Action(move.source.x, move.source.y, dir)
                    return action
            except Exception as e:
                # print("move execution failed: ", traceback.print_tb(e.__traceback__), e)
                pass
            
        
        

import random as rand

import math
from typing import List

from src.environment.board import Board
from src.environment.logger import Logger
from src.environment.tile import Tile, TileType
from src.environment.action import Action, convert_direction_to_vector


class GameMaster:
    """
    handles game state
    """

    def __init__(self, board: Board, players, max_turns=None, logger: Logger = None):
        self.board = board
        self.board.player_index = None
        self.logger = logger
        self.players = players
        self.turn = 0
        self.max_turns = max_turns

        if self.logger is not None:
            # log initial board configuration
            self.logger.init(self.board)

    def play(self):
        """
        conduct game between players on given board
        :return: index of winning player or -1 if max turns reached
        """
        while self.board.terminal_status() == -1 and self.turn < self.max_turns:
            actions = [player.move(self.board) for player in self.players]
            self.step(actions)

        return self.board.terminal_status()

    def step(self, actions: List[Action]):
        """
        conduct one turn of the game
        """
        # each player outputs a move given their view
        for moving_player_index, action in list(enumerate(actions)):
            if action.do_nothing:
                continue

            # check for validity of action
            if not self.board.is_action_valid(action, moving_player_index):
                continue

            # update game board with player's action
            self.update_game_state(action)

        # game logic to add troops to generals, cities, and land on specific ticks
        self.add_troops_to_board()
        self.turn += 1

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
                if tile.type == TileType.GENERAL or (
                    tile.player_index is not None
                    and (
                        tile.type == TileType.CITY
                        or (tile.type == TileType.NORMAL and self.turn % (25 * 2) == 0)
                    )
                ):
                    tile.army += 1
                    self._log(tile)

    def update_game_state(self, action: Action):
        """
        updates the game state given a player's action
        :param move:
        """
        updated_tiles = []
        start_tile = self.board.grid[action.starty][action.startx]
        dest_tile = self.board._get_destination_tile(start_tile, action)

        if dest_tile.player_index == start_tile.player_index:
            # player moving into his own tile
            dest_tile.army += start_tile.army - 1
        else:
            # player moving into unclaimed tile or opponent's tile
            if dest_tile.army < start_tile.army - 1:
                # attack is successful--destination cell is captured
                old_dest_player_index = dest_tile.player_index
                print('DEST TILE PLAYER INDEX', dest_tile.player_index)
                dest_tile.player_index = start_tile.player_index
                dest_tile.army = (start_tile.army - 1) - dest_tile.army

                # update vision from captured tile
                if old_dest_player_index is not None:
                    # tile was captured
                    updated_tiles.extend(
                        self.board.update_vision_from_captured_tile(
                            dest_tile, old_dest_player_index
                        )
                    )

                # add vision to newly owned tile
                updated_tiles.extend(
                    self.board.add_vision(dest_tile, start_tile.player_index)
                )
            else:
                # attack is unsuccessful--destination cell is not fully captured
                dest_tile.army = dest_tile.army - (start_tile.army - 1)

        # every legal action moves all but one troop away from start tile
        start_tile.army = 1

        # log tile changes
        updated_tiles.extend([start_tile, dest_tile])

        # log updated tiles
        for tile in updated_tiles:
            self._log(tile)

    def _log(self, tile):
        if self.logger is not None:
            self.logger.log(tile, self.turn)

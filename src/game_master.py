import random as rand

import math

from src.graphics.board import Board
from src.graphics.constants import TILE_EMPTY, TILE_FOG, TILE_MOUNTAIN, TILE_OBSTACLE, DIRECTIONS
from src.graphics.tile import Tile
from src.move import Move
from src.training.model import convert_move, convert_board


class GameMaster():
    """
    handles game state
    """
    def __init__(self, board, players, logger=None):
        self.board = board
        self.board.player_index = None
        self.logger = logger
        self.players = players
        self.turn = 0
        self.no_action = False
        self.views = [self.create_board_view(0),
                      self.create_board_view(1)]

        if self.logger is not None:
            # log initial board configuration
            self.logger.init(self.board, self.views)

    def play(self, trainer=None):
        """
        conduct game between players on given board
        :return:
        """
        while True:
            
            self.no_action = False

            #print("turn: {}".format(self.turn))
            if self.board.terminal_status() != -1:
                # game is over
                winner = self.board.terminal_status()
                ## NEW CODE
                ## NEW CODE
                # Tells trainer to calculate rewards and add to real memory
                if (trainer != None):
                    trainer.convert_temp_memory(winner)
                    print("Illegal Moves: " + str(self.players[0].illegal_moves))
                    print("Legal Moves: " + str(self.players[0].legal_moves))
                ## END
                return winner

            # each player outputs a move given their view
            moves = [player.move(view) if len(list(view.legal_moves)) > 0 else None
                     for player, view
                     in zip(self.players, self.views)]
            
            if (moves[0] is None or moves[1] is None):
                self.no_action = True

            ## NEW CODE
            ## NEW CODE
            # Creates the current states/actions to be passed to the trainer
            if (trainer != None and not self.no_action):
                states = [convert_board(view) for view in self.views]
                actions = [convert_move(move) for move in moves]
            ## END

            for moving_player_index, move in list(enumerate(moves)):
                if move is None or move not in self.views[moving_player_index].legal_moves:
                    # no valid moves OR a previous move this turn by opponent has rendered the player's move invalid
                    continue

                # update game board
                self.patch_move(move)

                # update views by copying start and destination tiles of the move from the game board
                # then add vision to the destination tile if the player captures it
                for view in self.views:
                    view_start_tile, view_dest_tile = view.grid[move.starty][move.startx], \
                                                      view.grid[move.desty][move.destx]
                    game_start_tile, game_dest_tile = self.board.grid[move.starty][move.startx], \
                                                      self.board.grid[move.desty][move.destx]

                    old_dest_type = view_dest_tile.type
                    old_dest_army = view_dest_tile.army

                    if view_start_tile.type != TILE_OBSTACLE and view_start_tile.type != TILE_FOG:
                        # only update the view if it has visibility of that tile
                        view_start_tile.copy(game_start_tile)
                        self._log(view_start_tile, view.player_index)

                    if view_dest_tile.type != TILE_OBSTACLE and view_dest_tile.type != TILE_FOG:
                        # only update the view if it has visibility of that tile
                        view_dest_tile.copy(game_dest_tile)
                        self._log(view_dest_tile, view.player_index)

                    # add vision around the destination cell if cell is captured
                    # remove vision around the destination cell if cell is lost
                    if old_dest_type != view_dest_tile.type:  # the destination cell type has changed
                        if view_dest_tile.type == view.player_index:  # captured cell
                            self.add_vision(view, view_dest_tile)
                        elif old_dest_type == view.player_index:  # lost cell
                            self.remove_vision(view, view_dest_tile)

                    # player was able to move troops from dest tile
                    had_legal_actions = old_dest_type == view.player_index and old_dest_army > 1

                    # player is still able to move troops from dest tile
                    has_legal_actions = view_dest_tile.type == view.player_index and view_dest_tile.army > 1

                    if had_legal_actions and not has_legal_actions:
                        # if player no longer has legal actions from the dest tile then remove them
                        self.remove_legal_actions_from_tile(view_dest_tile, view.player_index)
                    elif has_legal_actions and not had_legal_actions:
                        # if player now has legal actions from the dest tile then add them
                        self.add_legal_actions_from_tile(view_dest_tile, view.player_index)

                    # when moving all but one troop from start cell to dest cell,
                    # the start cell only has one troop so no longer valid to move from start cell into other cells
                    if view_start_tile.type == view.player_index:
                        self.remove_legal_actions_from_tile(view_start_tile, view.player_index)

            if self.turn % 2 == 0:
                # increment all troops on game board
                self.tick(self.board)

                # increment all troops for each view
                for view in self.views:
                    self.tick(view)

            ## NEW CODE
            ## NEW CODE
            if (trainer != None and not self.no_action):
                # Creates the next states after move has been taken
                next_states = [convert_board(view) for view in self.views]
                # Adds the SAS' to the temporary memory and trains for our player
                for i in range(len(self.players)):
                    trainer.step(states[i], actions[i], next_states[i], i, self.board.terminal_status() != -1)
            ## END

            self.turn += 1

    def tick(self, board):
        # TODO: align turn number
        for i in range(board.rows):
            for j in range(board.cols):
                tile = board.grid[i][j]

                if(tile.is_general  # player's general
                   or (tile.is_city and tile.type >= 0)  # player's city
                   or (tile.type >= 0 and self.turn % (25 * 2) == 0)):  # player's land increments once every 25 seconds
                                                                        # there are two turns per tick
                    tile.army += 1

                    # note if game board is passed in then board.player_index is None so trivially will not enter
                    if tile.army == 2 and board.player_index == tile.type:
                        # the tile now has two troops so moves from this tile are now legal
                        self.add_legal_actions_from_tile(tile, tile.type)

    def add_vision(self, view, tile):
        """
        given a view and a tile supply vision to all 8 tiles around the given tile
        i.e. make any fog tiles in the view disappear and replace with the game board's tile
        :param view: Board instance representing the player's view
        :param tile: Tile
        :return:
        """
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if not (0 <= tile.y + dy <= view.rows - 1 and 0 <= tile.x + dx <= view.cols - 1):
                    # out of bounds of grid
                    continue

                game_tile = self.board.grid[tile.y + dy][tile.x + dx]
                view_tile = view.grid[tile.y + dy][tile.x + dx]

                # no need to mark tile as "remembered opponent is here" since now have direct vision
                view_tile.memory = -1  

                if dx == dy == 0:
                    continue

                if view_tile.type == TILE_OBSTACLE or view_tile.type == TILE_FOG:
                    # currently have no vision of tile so copy game tile into view tile
                    view_tile.copy(game_tile)
                    self._log(view_tile, view.player_index)

    def remove_vision(self, view, tile):
        """
        given a view and a tile remove vision produced by the given tile
        i.e. make any of the 8 surrounding tiles in the view which are visible only due to the given tile and replace
        with appropriate fog tiles
        :param view: Board instance representing the player's view
        :param tile: Tile
        :return:
        """
        # the player should not own the tile if the vision the tile produces is removed
        assert(tile.type != view.player_index)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if not (0 <= tile.y + dy <= view.rows - 1 and 0 <= tile.x + dx <= view.cols - 1):
                    # out of bounds of grid
                    continue

                view_tile = view.grid[tile.y + dy][tile.x + dx]
                view_tile_type = view_tile.type
                if not self._has_vision(view, view_tile):
                    if view_tile.type >= 0:
                        # retain memory that opponent is there
                        view_tile.memory = view_tile.type

                    # turn tile into a fog tile
                    if view_tile.type >= 0 or (view_tile.type == TILE_EMPTY and not view_tile.is_city):
                        view_tile.type = TILE_FOG
                    elif view_tile.type == TILE_EMPTY and view_tile.is_city:
                        view_tile.type = TILE_OBSTACLE

                    if view_tile.type != view_tile_type:
                        # view tile has changed so update log
                        self._log(view_tile, view.player_index)

    def _has_vision(self, view, tile):
        """
        given a view and a tile, check if the tile has any of the 8 surrounding tiles owned by the player. That is,
        should the given tile be visible
        :param view: Board representing player's view
        :param tile: Tile
        :return: boolean should the tile be visible
        """
        if tile.type == view.player_index:
            # player owns tile so trivially has vision of it
            return True

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == dy == 0:
                    continue

                if not (0 <= tile.y + dy <= view.rows - 1 and 0 <= tile.x + dx <= view.cols - 1):
                    # out of bounds of grid
                    continue

                if view.grid[tile.y + dy][tile.x + dx].type == view.player_index:
                    # player owns a surrounding tile so should have vision over the given tile
                    return True

        # player owns no surrounding tiles nor the tile itself so should not have vision over it
        return False

    def patch_move(self, move):
        """
        updates the game board and the legal actions for each player
        :param move:
        :return:
        """
        start_tile = self.board.grid[move.starty][move.startx]
        dest_tile = self.board.grid[move.desty][move.destx]

        # must have army > 1 for it to be a valid move
        assert(start_tile.army > 1)

        if dest_tile.type == TILE_EMPTY:
            if dest_tile.is_city:
                # trying to capture a city
                self.attack(start_tile, dest_tile)
            else:
                # moving into empty tile
                dest_tile.type = start_tile.type
                dest_tile.army = start_tile.army - 1
        elif dest_tile.type >= 0 and dest_tile.type == start_tile.type:
            # player moving into his own tile
            dest_tile.army += start_tile.army - 1
        elif dest_tile.type >= 0 and dest_tile.type != start_tile.type:
            # player attacking opponent's tile
            self.attack(start_tile, dest_tile)

        # every legal action moves all but one troop away from start tile
        # if attack was called then this update has already occurred
        start_tile.army = 1

        # log tile changes
        self._log(start_tile, None)
        self._log(dest_tile, None)

    def attack(self, start_tile, dest_tile):
        if dest_tile.army < start_tile.army - 1:
            # attack is successful--destination cell is captured
            dest_tile.type = start_tile.type
            dest_tile.army = (start_tile.army - 1) - dest_tile.army
        else:
            # attack is unsuccessful--destination cell is not fully captured
            dest_tile.army = dest_tile.army - (start_tile.army - 1)

        start_tile.army = 1

    def create_board_view(self, player_index):
        """
        create a new board instance representing player view

        :param player_index: index of player to create view for
        :return: view for player
        """
        grid = [  # 2D List of Tile Objects
            [Tile(self, x, y) for x in range(self.board.cols)]
            for y in range(self.board.rows)
        ]

        # completely fog the board
        for x in range(self.board.cols):
            for y in range(self.board.rows):
                player_view_tile = grid[y][x]
                game_master_tile = self.board.grid[y][x]

                # game master tiles from game master's board has full vision so no fog or fog obstacle tile types
                if game_master_tile.type == TILE_MOUNTAIN or game_master_tile.is_city:
                    player_view_tile.type = TILE_OBSTACLE
                elif game_master_tile.type == TILE_EMPTY:
                    player_view_tile.type = TILE_FOG

        # add the player's general to grid
        general = self.board.generals[player_index]
        grid[general.y][general.x].type = player_index
        grid[general.y][general.x].army = 1
        grid[general.y][general.x].is_general = True

        # create player's view
        view = Board(self.board.rows, self.board.cols, player_index)
        view.set_grid(grid)

        # update view's list of known generals--this is the player's own general
        view.generals[player_index] = grid[general.y][general.x]

        # player's view starts with vision around general
        self.add_vision(view, view.generals[player_index])

        return view

    def add_legal_actions_from_tile(self, tile, player_index):
        """
        add all legal actions from tile for a given player index--that is, add them into the corresponding player view

        :param tile: Tile
        :param player_index: identifies player whose view's legal actions should be updated
        :return:
        """
        if tile.army <= 1:
            # cannot move troops without at least 2
            return

        for dx, dy in DIRECTIONS:
            # iterate through all four adjacent tiles to the given tile
            if self.board.is_valid_position(tile.x + dx, tile.y + dy):
                # the neighboring tile is not a mountain so we have found a valid move
                self.views[player_index].legal_moves.add(
                    Move(startx=tile.x,
                         starty=tile.y,
                         destx=tile.x + dx,
                         desty=tile.y + dy)
                )

    def remove_legal_actions_from_tile(self, tile, player_index):
        """
        remove all legal actions from tile for a given player index--that is, remove them from the corresponding player view
        :param tile: Tile
        :param player_index: identifies player whose view's legal actions should be updated
        :return:
        """
        if tile.army > 1 and tile.type == player_index:
            # it is legal to move troops from tile
            return

        for dx, dy in DIRECTIONS:
            if self.board.is_valid_position(tile.x + dx, tile.y + dy):
                move = Move(startx=tile.x, starty=tile.y, destx=tile.x + dx, desty=tile.y + dy)

                # the move is legal so must have been in the legal moves
                assert(move in self.views[player_index].legal_moves)

                # remove the move from the set of legal moves
                self.views[player_index].legal_moves.remove(move)

    def _log(self, tile, player_index):
        if self.logger is not None:
            self.logger.log(tile, self.turn, player_index)

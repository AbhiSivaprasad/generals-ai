class Logger:
    def __init__(self, num_players):
        self.game_board_log = BoardLog()
        self.view_logs = [BoardLog() for _ in range(num_players)]

    def init(self, board, views):
        """
        save serialized versions of board and views
        """
        self.game_board_log.serialized_board = board.serialize()

        for view_log, view in zip(self.view_logs, views):
            view_log.serialized_board = view.serialize()

    def log(self, tile, turn, player_index):
        # if player index is null then update is for game board otherwise update player's view log
        if player_index is None:
            update_log = self.game_board_log
        else:
            update_log = self.view_logs[player_index]

        log_length = len(update_log.serialized_board_diffs)
        if turn == log_length - 1:
            update_log.serialized_board_diffs[turn].serialized_tiles.append(tile.serialize())
        elif turn >= log_length:
            # extend the log with empty moves for turns where player didn't move
            update_log.serialized_board_diffs.extend([BoardDiff()] * (turn - log_length))

            # new turn so increase the length of the log
            update_log.serialized_board_diffs.append(BoardDiff(serialized_tiles=[tile.serialize()]))
        else:
            raise Exception("turn must be on the last line of log or starting new line")

    def flush(self):
        """
        in case the player chose not to make moves at the end make sure logs for views are of same length
        :return:
        """
        # game board is fully updated so use as total length
        num_moves = len(self.game_board_log.serialized_board_diffs)

        for view_log in self.view_logs:
            view_log.serialized_board_diffs.extend(
                [BoardDiff()] * (num_moves - len(view_log.serialized_board_diffs))
            )

    def output(self):
        # flush log
        self.flush()

        return {
            'gameLog': self.game_board_log.output(),
            'viewLogs': [view_log.output() for view_log in self.view_logs],
            'numViews': len(self.view_logs)
        }


class BoardLog:
    def __init__(self, serialized_board=None, serialized_board_diffs=None):
        self.serialized_board = [] if serialized_board is None else serialized_board
        self.serialized_board_diffs = [] if serialized_board_diffs is None else serialized_board_diffs

    def output(self):
        return {
            'initialBoard': self.serialized_board,
            'boardDiffs': [serialized_board_diff.serialized_tiles
                           for serialized_board_diff
                           in self.serialized_board_diffs]
        }


class BoardDiff:
    def __init__(self, serialized_tiles=None):
        self.serialized_tiles = [] if serialized_tiles is None else serialized_tiles
import json
from pathlib import Path
from typing import Any, Dict, List
from src.environment.board import Board
from src.environment.tile import Tile


class Logger:
    def __init__(
        self,
        serialized_board: Dict = None,
        serialized_board_diffs: List[List[Dict]] = None,
    ):
        self.serialized_board = serialized_board
        self.serialized_board_diffs = (
            [] if serialized_board_diffs is None else serialized_board_diffs
        )
        self.infos = []

    def init(self, board: Board):
        self.serialized_board = board.serialize()
        self.infos = []

    def log(self, tile: Tile, turn: int):
        if turn >= len(self.serialized_board_diffs):
            self.serialized_board_diffs.extend(
                [[] for _ in range(turn - len(self.serialized_board_diffs) + 1)]
            )
        elif turn < len(self.serialized_board_diffs) - 1:
            raise ValueError(
                f"Turn {turn} is already logged. Cannot log turn {turn} again."
            )

        self.serialized_board_diffs[turn].append(tile.serialize())
    
    def log_info(self, key: str, val: Any, turn: int):
        if turn >= len(self.infos):
            self.infos.extend([{} for _ in range(turn - len(self.infos) + 1)])
        self.infos[turn][key] = val
      
    def log_info_dict(self, info: Dict, turn: int):
        if turn >= len(self.infos):
            self.infos.extend([{} for _ in range(turn - len(self.infos) + 1)])
        self.infos[turn] = (info)

    def write(self, path: Path):
        with open(path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "initialBoard": self.serialized_board,
                        "boardDiffs": self.serialized_board_diffs,
                        "infos": self.infos,
                    }
                )
            )

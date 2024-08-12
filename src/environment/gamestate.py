from dataclasses import dataclass
from typing import List
from src.environment.board import Board

@dataclass
class GameState:
    terminal_status: int = -1
    board: Board
    scores: List[int] # not functional / not in use yet
    turn: int
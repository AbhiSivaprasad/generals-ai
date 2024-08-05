from dataclasses import dataclass
from typing import List
from src.environment.board import Board

@dataclass
class GameState:
    board: Board
    scores: List[int]
    turn: int
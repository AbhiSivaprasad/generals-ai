from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class Direction(int, Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

def get_direction_from_str(direction_str: str) -> Direction:
    if direction_str == 'up':
        return Direction.UP
    elif direction_str == 'down':
        return Direction.DOWN
    elif direction_str == 'left':
        return Direction.LEFT
    elif direction_str == 'right':
        return Direction.RIGHT


@dataclass
class Action:
    startx: int
    starty: int
    direction: Direction
    do_nothing: bool = False
    
    @classmethod
    def to_space_sample(cls, action: Optional["Action"], num_rows: int, num_col: int) -> int:
        if action is None:
            return 0
        return np.ravel_multi_index((action.starty, action.startx, action.direction.value), (num_rows, num_col, 4)) + 1
    
    @classmethod
    def from_space_sample(cls, sample: int, num_rows: int, num_col: int) -> "Action":
        # 0 is None action
        if sample == 0:
            return None
        # sample - 1 is the unraveled index
        sample = sample - 1
        y, x, dir = np.unravel_index(sample, (num_rows, num_col, 4))
        direction = Direction(dir)
        return cls(x, y, direction)

    
    def serialize(self) -> dict:
        return {
            'do_nothing': self.do_nothing,
            'startx': self.startx,
            'starty': self.starty,
            'direction': self.direction.name if self.direction else None
        }


def convert_direction_to_vector(direction: Direction) -> Tuple[int, int]:
    if direction == Direction.UP:
        return 0, -1
    elif direction == Direction.DOWN:
        return 0, 1
    elif direction == Direction.LEFT:
        return -1, 0
    elif direction == Direction.RIGHT:
        return 1, 0

def convert_vector_to_direction(vector: Tuple[int, int]) -> Direction:
    if vector == (0, -1):
        return Direction.UP
    elif vector == (0, 1):
        return Direction.DOWN
    elif vector == (-1, 0):
        return Direction.LEFT
    elif vector == (1, 0):
        return Direction.RIGHT
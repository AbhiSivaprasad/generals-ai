from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass
class Action:
    startx: int
    starty: int
    direction: Direction
    
    def to_space_sample(self, num_rows: int, num_col: int) -> int:
        return np.ravel_multi_index((self.starty, self.startx, self.direction.value), (num_rows, num_col, 4)) + 1
    
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


def convert_direction_to_vector(direction: Direction) -> Tuple[int, int]:
    if direction == Direction.UP:
        return 0, 1
    elif direction == Direction.DOWN:
        return 0, -1
    elif direction == Direction.LEFT:
        return -1, 0
    elif direction == Direction.RIGHT:
        return 1, 0

def convert_vector_to_direction(vector: Tuple[int, int]) -> Direction:
    if vector == (0, 1):
        return Direction.UP
    elif vector == (0, -1):
        return Direction.DOWN
    elif vector == (-1, 0):
        return Direction.LEFT
    elif vector == (1, 0):
        return Direction.RIGHT
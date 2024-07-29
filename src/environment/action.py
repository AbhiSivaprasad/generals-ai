from dataclasses import dataclass
from enum import Enum
from typing import Tuple


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


def convert_direction_to_vector(direction: Direction) -> Tuple[int, int]:
    if direction == Direction.UP:
        return 0, 1
    elif direction == Direction.DOWN:
        return 0, -1
    elif direction == Direction.LEFT:
        return -1, 0
    elif direction == Direction.RIGHT:
        return 1, 0

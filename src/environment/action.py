from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass
class Action:
    """
    If do_nothing is True, the agent decides to wait.
    If do_nothing is False, the agent decides to move, and startx, starty, and direction are set.
    """

    do_nothing: bool = False
    startx: Optional[int] = None
    starty: Optional[int] = None
    direction: Optional[Direction] = None

    def to_index(self, n_columns: int):
        if self.do_nothing:
            return 0
        else:
            return (
                1 + self.startx * 4 + self.starty * n_columns * 4 + self.direction.value
            )

    @staticmethod
    def from_index(action_index: int, n_columns: int) -> Optional["Action"]:
        if action_index == 0:
            # agent decides to wait
            return Action(do_nothing=True, startx=None, starty=None, direction=None)

        # agent decides to move
        action_index -= 1
        x = (action_index // 4) % n_columns
        y = action_index // (4 * n_columns)
        direction = action_index % 4
        return Action(startx=x, starty=y, direction=Direction(direction))


def convert_direction_to_vector(direction: Direction) -> Tuple[int, int]:
    if direction == Direction.UP:
        return 0, 1
    elif direction == Direction.DOWN:
        return 0, -1
    elif direction == Direction.LEFT:
        return -1, 0
    elif direction == Direction.RIGHT:
        return 1, 0

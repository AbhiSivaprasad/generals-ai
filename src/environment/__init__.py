from typing import Tuple
from src.environment.action import Action

import numpy as np

MAX_SIZE = [15, 15]
ActType = Action
ObsType = Tuple[int, np.ndarray] # turn, grid

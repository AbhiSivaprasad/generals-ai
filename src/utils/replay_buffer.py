from abc import ABC, abstractmethod
from typing import Tuple, List

import ray

from src.environment import ObsType, ActType

from ray.rllib.utils.replay_buffers import replay_buffer as ray_replay_buffer

Experience = Tuple[ObsType, ActType, float, ObsType, bool]

class ReplayBuffer(ABC):
    @abstractmethod
    def add(self, samples: List[Experience]):
        pass

    @abstractmethod
    def sample(self, size: int):
        pass
    
    @abstractmethod
    def size(self) -> int:
        pass

@ray.remote
class RayReplayBuffer(ReplayBuffer):
    ray_buffer: ray_replay_buffer.ReplayBuffer
    
    def __init__(self, capacity: int):
        self.ray_buffer = ray_replay_buffer.ReplayBuffer(capacity=capacity, storage_unit=ray_replay_buffer.StorageUnit.FRAGMENTS)
    
    def add(self, samples: List[Experience]):
        sample_dict = {
            "s0": [sample[0] for sample in samples],
            "a": [sample[1] for sample in samples],
            "r": [sample[2] for sample in samples],
            "s1": [sample[3] for sample in samples],
            "d": [sample[4] for sample in samples],
        }
        self.ray_buffer.add_batch(sample_dict)

    def sample(self, size: int) -> List[Experience]:
        batch: ray_replay_buffer.SampleBatch = self.ray_buffer.sample(size)
        list: List[Experience] = [(batch["s0"][i], batch["a"][i], batch["r"][i], batch["s1"][i], batch["d"][i]) for i in range(size)]
        return list
    
    def size(self) -> int:
        return len(self.ray_buffer)
    
    
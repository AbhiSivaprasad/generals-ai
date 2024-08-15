from abc import ABC, abstractmethod
import sys
import threading
from typing import Tuple, List

import numpy as np
import ray
from src.environment import ObsType, ActType

from torchrl.data import ReplayBuffer as TorchReplayBuffer, ListStorage

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

@ray.remote(num_cpus=1, memory=3*(1024**4))
class RayReplayBuffer(ReplayBuffer):
    buffer: List[Experience]
    capacity: int
    
    rng: np.random.Generator
    lock: threading.Lock
    
    def __init__(self, capacity: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.capacity = capacity
        self.buffer = []
        self.lock = threading.Lock()
    
    def add(self, samples: List[Experience]):
        self.lock.acquire()
        self.buffer.extend(samples)
        self.buffer = self.buffer[-self.capacity:]
        self.lock.release()
        # print(f"[INFO] Added {len(samples)} samples to buffer.")
        # print(f"[INFO] Buffer size: {len(self.buffer)}")
        # print(f"[INFO] Buffer size (bytes): {sys.getsizeof(self.buffer)}")

    def sample(self, size: int) -> List[Experience]:
        success = self.lock.acquire()
        if success:
            idx = self.rng.integers(0, len(self.buffer), size)
            batch = [self.buffer[i] for i in idx]
            self.lock.release()
            return batch
        print("[ERROR] Failed to acquire lock on buffer.")
        return []
    
    def size(self) -> int:
        success = self.lock.acquire()
        if success:
            size = len(self.buffer)
            self.lock.release()
            return size
        print("[ERROR] Failed to acquire lock on buffer.")
        return -1
    
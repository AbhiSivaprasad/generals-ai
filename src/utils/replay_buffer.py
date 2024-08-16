from abc import ABC, abstractmethod
import gc
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

@ray.remote(num_cpus=1, memory=1*(1024**4), max_restarts=0, max_task_retries=0)
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
        s = self.lock.acquire(timeout=10)
        if s:
            if len(self.buffer) + len(samples) > self.capacity:
                self.buffer = self.buffer[len(samples):]
            self.buffer.extend(samples)
            l = len(self.buffer)
            print(f"[INFO] Buffer size: {l}")
            self.lock.release()
            gc.collect()
        # print(f"[INFO] Added {len(samples)} samples to buffer.")
        # print(f"[INFO] Buffer size: {len(self.buffer)}")
        # print(f"[INFO] Buffer size (bytes): {sys.getsizeof(self)}")

    def sample(self, size: int) -> List[Experience]:
        success = self.lock.acquire(timeout=10)
        if success:
            idx = self.rng.integers(0, len(self.buffer), size)
            batch = [self.buffer[i] for i in idx]
            self.lock.release()
            return batch
        print("[ERROR] Failed to acquire lock on buffer.")
        return []
    
    def size(self) -> int:
        success = self.lock.acquire(timeout=10)
        if success:
            size = len(self.buffer)
            self.lock.release()
            return size
        print("[ERROR] Failed to acquire lock on buffer.")
        return -1
    
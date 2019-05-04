import random

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_samples(self, samples):
        random.shuffle(samples)
        for sample in samples:
            self.add_sample(sample)

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        return random.sample(self._samples, min(no_samples, len(self._samples))
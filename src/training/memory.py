import random
import params
import pickle
import bz2

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []
        self.load_memory()

    def low_size(self):
        return len(self._samples) < params.MIN_MEMORY

    def add_samples(self, samples):
        random.shuffle(samples)
        for sample in samples:
            self.add_sample(sample)

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        return random.sample(self._samples, min(no_samples, len(self._samples)))

    def save_memory(self, name='memory.pkl'):
        with bz2.BZ2File(name, 'wb') as f:
            pickle.dump(self._samples, f)

    def load_memory(self, name='memory.pkl'):
        try:
            with bz2.BZ2File(name, 'rb') as f:
                self._samples = pickle.load(f)
            print("Memory file loaded: " + name)
            print()
        except:
            print("No memory file found: " + name)
            print()

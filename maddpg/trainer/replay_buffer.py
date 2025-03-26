# replay_buffer.py
import random
import numpy as np

class ReplayBuffer:

    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.storage = []
        self.ptr = 0

    def add(self, states, actions, rewards, next_states, done):
        data = (states, actions, rewards, next_states, done)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size=64):
        batch = random.sample(self.storage, batch_size)
        # Unpack
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.storage)

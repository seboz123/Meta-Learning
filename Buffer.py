import numpy as np
from typing import Tuple

class Buffer:
    def __init__(self, obs: np.ndarray = None, acts: np.ndarray = None, rews: np.ndarray = None, n_obs: np.ndarray = None, done: np.ndarray = None, action_space: Tuple = None):
        self.observations = obs
        self.actions = acts
        self.rewards = rews
        self.next_observations = n_obs
        self.done = done
        self.action_space = action_space

    def __len__(self):
        return len(self.rewards)

    def sample_batch(self, batch_size: int = 128):
        indx = np.arange(len(self))
        np.random.shuffle(indx)
        indx = indx[:batch_size]
        return Buffer(self.observations[indx], self.actions[indx], self.rewards[indx], self.next_observations[indx], self.done[indx], self.action_space)
        # return [self.observations[indx], self.actions[indx], self.rewards[indx], self.next_observations[indx], self.done[indx]]
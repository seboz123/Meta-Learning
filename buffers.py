import numpy as np
from typing import Tuple

class SACBuffer:
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
        return SACBuffer(self.observations[indx], self.actions[indx], self.rewards[indx], self.next_observations[indx], self.done[indx], self.action_space)

class PPOBuffer:
    def __init__(self, buffer_size: int, action_space: Tuple = None, obs: np.ndarray = None, acts: np.ndarray = None, rews: np.ndarray = None,
        n_obs: np.ndarray = None, done: np.ndarray = None, values: np.ndarray = None, entropies: np.ndarray = None,
                 act_log_probs: np.ndarray = None, advantages: np.ndarray = None):
        self.max_buffer_size = buffer_size
        self.action_space = action_space

        self.observations = obs
        self.actions = acts
        self.rewards = rews
        self.next_observations = n_obs
        self.done = done
        self.values = values
        self.entropies = entropies
        self.act_log_probs = act_log_probs
        self.advantages = advantages

    def __len__(self):
        return len(self.rewards)

    def sample_batch(self, batch_size: int = 128):
        indx = np.arange(len(self))
        np.random.shuffle(indx)
        indx = indx[:batch_size]
        # return PPOBuffer(self.observations[indx], self.actions[indx], self.rewards[indx], self.next_observations[indx], self.done[indx], self.action_space)

    def store(self, obs: np.ndarray = None, acts: np.ndarray = None, rews: np.ndarray = None,
        n_obs: np.ndarray = None, done: np.ndarray = None, values: np.ndarray = None, entropies: np.ndarray = None,
                 act_log_probs: np.ndarray = None, advantages: np.ndarray = None):
        if self.rewards is not None:
            self.observations = np.vstack((self.observations, obs))
            self.actions = np.vstack((self.actions, acts))
            self.rewards = np.append(self.rewards, rews)
            self.next_observations = np.vstack((self.next_observations, n_obs))
            self.done = np.append(self.done, done)
            self.values = np.append(self.values, values)
            self.entropies = np.append(self.entropies, entropies)
            self.act_log_probs = np.vstack((self.act_log_probs, act_log_probs))
            self.advantages = np.append(self.advantages, advantages)
        else:
            self.observations = obs
            self.actions = acts
            self.rewards = rews
            self.next_observations = n_obs
            self.done = done
            self.values = values
            self.entropies = entropies
            self.act_log_probs = act_log_probs
            self.advantages = advantages

    def split_into_batches(self, batch_size: int = 512):
        indx = np.arange(len(self))
        np.random.shuffle(indx)

        self.observations = self.observations[indx]
        self.actions = self.actions[indx]
        self.rewards = self.rewards[indx]
        self.next_observations = self.next_observations[indx]
        self.done = self.done[indx]
        self.values = self.values[indx]
        self.entropies = self.entropies[indx]
        self.act_log_probs = self.act_log_probs[indx]
        self.advantages = self.advantages[indx]

        batches = []
        for size in range(0, len(self), batch_size):
            if size + batch_size <= len(self):
                batch = PPOBuffer(buffer_size=batch_size, action_space=self.action_space, obs=self.observations[size: size + batch_size],
                                  acts=self.actions[size: size + batch_size], rews=self.rewards[size: size + batch_size],
                                  n_obs=self.next_observations[size: size + batch_size], done=self.done[size: size + batch_size],
                                  values=self.values[size: size + batch_size], advantages=self.advantages[size: size + batch_size],
                                  act_log_probs=self.act_log_probs[size: size + batch_size],entropies=self.entropies[size: size + batch_size])
            else:
                batch = PPOBuffer(buffer_size=len(self)-size, action_space=self.action_space,
                                  obs=self.observations[size:],
                                  acts=self.actions[size:],
                                  rews=self.rewards[size:],
                                  n_obs=self.next_observations[size:],
                                  done=self.done[size:],
                                  values=self.values[size:],
                                  advantages=self.advantages[size:],
                                  act_log_probs=self.act_log_probs[size:],
                                  entropies=self.entropies[size:])
            batches.append(batch)
        return batches
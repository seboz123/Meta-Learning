import collections
import numpy as np
import random

from typing import Dict, List, Tuple, Deque
from segment_tree import MinSegmentTree, SumSegmentTree


class SACBuffer:
    def __init__(self, max_buffer_size: int, obs: np.ndarray = None, acts: np.ndarray = None, rews: np.ndarray = None, n_obs: np.ndarray = None, done: np.ndarray = None, action_space: Tuple = None):
        self.max_buffer_size = max_buffer_size

        self.observations = obs
        self.actions = acts
        self.rewards = rews
        self.next_observations = n_obs
        self.done = done
        self.action_space = action_space

    def __len__(self):
        if self.rewards is not None:
            return len(self.rewards)
        else:
            return 0


    def remove_old_obs(self, remove_percentage: float):
        start_indx = int(len(self) * remove_percentage)
        self.observations = self.observations[start_indx:]
        self.actions = self.actions[start_indx:]
        self.rewards = self.rewards[start_indx:]
        self.next_observations = self.next_observations[start_indx:]
        self.done = self.done[start_indx:]

    def sample_batch(self, batch_size: int = 128):
        indx = np.arange(len(self))
        np.random.shuffle(indx)
        indx = indx[:batch_size]
        return SACBuffer(self.max_buffer_size, self.observations[indx], self.actions[indx], self.rewards[indx], self.next_observations[indx], self.done[indx], self.action_space)

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



class DQNBuffer:
    """A simple numpy replay buffer."""

    def __init__(
            self,
            obs_dim: int,
            size: int,
            action_dim: int = 1,
            batch_size: int = 32,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32) if action_dim == 1 else np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.use_act_prob = False
        self.act_prob_buf = np.zeros([size], dtype=np.float32) if action_dim == 1 else np.zeros([size, action_dim], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0


        # for N-step Learning
        self.n_step_buffer = collections.deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
            act_prob: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = np.array(obs).flatten()
        self.next_obs_buf[self.ptr] = np.array(next_obs).flatten()
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        if act_prob is not None:
            self.use_act_prob = True
            self.act_prob_buf[self.ptr] = act_prob
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        if self.use_act_prob:
            return dict(
                obs=self.obs_buf[idxs],
                next_obs=self.next_obs_buf[idxs],
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs],
                done=self.done_buf[idxs],
                act_prob=self.act_prob_buf[idxs],
                # for N-step Learning
                indices=idxs,
            )
        else:
            return dict(
                obs=self.obs_buf[idxs],
                next_obs=self.next_obs_buf[idxs],
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs],
                done=self.done_buf[idxs],
                # for N-step Learning
                indices=idxs,
            )

    def sample_batch_from_idxs(
            self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def _get_n_step_info(
            self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size


## Replay Buffer

class PrioritizedDQNBuffer(DQNBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
            self,
            obs_dim: int,
            size: int,
            action_dim: int = 1,
            batch_size: int = 32,
            alpha: float = 0.7,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedDQNBuffer, self).__init__(
            obs_dim, size, action_dim, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, batch_size: int, beta: float = 0.5) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
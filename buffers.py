import collections
import numpy as np
import random

from typing import Dict, List, Tuple, Deque
from segment_tree import MinSegmentTree, SumSegmentTree

# Implementation of Buffers
# SAC and PPO are obsolete and not used in this Project

class SACBuffer:
    def __init__(self, max_buffer_size: int, obs_space: int, action_space: Tuple):
        self.max_buffer_size = max_buffer_size
        self.ptr = 0
        self.action_space = action_space
        self.obs_space = obs_space

        self.observations = np.zeros((max_buffer_size, obs_space))
        self.actions = np.zeros((max_buffer_size, len(action_space)))
        self.rewards = np.zeros((max_buffer_size, 2))
        self.next_observations = np.zeros((max_buffer_size, obs_space))
        self.done = np.zeros((max_buffer_size))

    def store(self, obs: np.ndarray, acts: np.ndarray, rews: np.ndarray,
              n_obs: np.ndarray, done: np.ndarray):
        num_experiences = len(done)
        self.observations[self.ptr:self.ptr + num_experiences, :] = obs
        self.actions[self.ptr:self.ptr + num_experiences, :] = acts
        self.rewards[self.ptr:self.ptr + num_experiences, :] = rews
        self.next_observations[self.ptr:self.ptr + num_experiences, :] = n_obs
        self.done[self.ptr:self.ptr + num_experiences] = done

        self.ptr += num_experiences

    def __len__(self):
        return self.ptr

    def remove_old_obs(self, remove_percentage: float):
        start_indx = int(len(self) * remove_percentage)
        self.ptr = self.ptr - start_indx
        self.observations = np.pad(self.observations[start_indx:], [(0, start_indx), (0,0)], mode='constant', constant_values=0)
        self.actions = np.pad(self.actions[start_indx:], [(0, start_indx), (0,0)], mode='constant', constant_values=0)
        self.rewards = np.pad(self.rewards[start_indx:], [(0, start_indx), (0,0)], mode='constant', constant_values=0)
        self.next_observations = np.pad(self.next_observations[start_indx:], [(0, start_indx), (0,0)], mode='constant', constant_values=0)
        self.done = np.pad(self.done[start_indx:], [(0, start_indx)], mode='constant', constant_values=0)

    def sample_batch(self, batch_size: int):
        indx = np.arange(len(self))
        np.random.shuffle(indx)
        indx = indx[:batch_size]
        batch = SACBuffer(batch_size, self.obs_space, self.action_space)
        batch.store(self.observations[indx], self.actions[indx], self.rewards[indx],
                         self.next_observations[indx], self.done[indx])

        return batch

class PPOBuffer:
    def __init__(self, buffer_size: int, action_space: Tuple, obs_size: int):
        self.max_buffer_size = buffer_size
        self.action_space = action_space
        self.obs_space = obs_size
        self.ptr = 0

        self.observations = np.zeros((buffer_size, obs_size))
        self.actions = np.zeros((buffer_size, len(action_space)))
        self.next_observations = np.zeros((buffer_size, obs_size))
        self.done = np.zeros(buffer_size)
        self.entropies = np.zeros((buffer_size, len(action_space)))
        self.act_log_probs = np.zeros((buffer_size, len(action_space)))

        self.rewards = np.zeros((buffer_size, 2))
        self.values = np.zeros((buffer_size, 2))
        self.advantages = np.zeros((buffer_size, 2))
        self.returns = np.zeros((buffer_size, 2))

        self.global_advantages = np.zeros(buffer_size)
        self.global_returns = np.zeros(buffer_size)

    def __len__(self):
        return self.ptr

    def store(self, obs: np.ndarray, acts: np.ndarray, rews: np.ndarray,
              n_obs: np.ndarray, done: np.ndarray,
              entropies: np.ndarray,
              act_log_probs: np.ndarray, value_estimates,
              returns, advantages, global_returns, global_advantages):

        num_experiences = len(done)
        self.observations[self.ptr:self.ptr + num_experiences, :] = obs
        self.actions[self.ptr:self.ptr + num_experiences, :] = acts
        self.rewards[self.ptr:self.ptr + num_experiences, :] = rews
        self.next_observations[self.ptr:self.ptr + num_experiences, :] = n_obs
        self.done[self.ptr:self.ptr + num_experiences] = done
        self.entropies[self.ptr:self.ptr + num_experiences, :] = entropies
        self.act_log_probs[self.ptr:self.ptr + num_experiences, :] = act_log_probs

        self.values[self.ptr:self.ptr + num_experiences, :] = value_estimates
        self.advantages[self.ptr:self.ptr + num_experiences, :] = advantages
        self.returns[self.ptr:self.ptr + num_experiences, :] = returns

        self.global_returns[self.ptr:self.ptr + num_experiences] = global_returns
        self.global_advantages[self.ptr:self.ptr + num_experiences] = global_advantages


        self.ptr += num_experiences

    def split_into_batches(self, batch_size: int = 512):
        indx = np.arange(len(self))
        np.random.shuffle(indx)

        observations = self.observations[indx]
        actions = self.actions[indx]
        next_observations = self.next_observations[indx]
        done = self.done[indx]
        entropies = self.entropies[indx]
        act_log_probs = self.act_log_probs[indx]


        rewards = self.rewards[indx]
        values = self.values[indx]
        advantages = self.advantages[indx]
        returns = self.returns[indx]

        global_advantages = self.global_advantages[indx]
        global_returns = self.global_returns[indx]


        batches = []
        for size in range(0, len(self), batch_size):
            if size + batch_size <= len(self):
                batch = PPOBuffer(batch_size, action_space=self.action_space, obs_size=self.obs_space)
                batch.store(obs=observations[size: size + batch_size],
                            acts=actions[size: size + batch_size], rews=rewards[size: size + batch_size],
                            n_obs=next_observations[size: size + batch_size], done=done[size: size + batch_size],
                            advantages=advantages[size: size + batch_size], value_estimates=values[size: size + batch_size],
                            act_log_probs=act_log_probs[size: size + batch_size], returns=returns[size: size + batch_size],
                            global_advantages=global_advantages[size: size + batch_size], global_returns=global_returns[size: size + batch_size],
                            entropies=entropies[size: size + batch_size])
            else:
                batch = PPOBuffer(len(self) - size, action_space=self.action_space, obs_size=self.obs_space)
                batch.store(obs=observations[size:],
                            acts=actions[size:],
                            rews=rewards[size:],
                            n_obs=next_observations[size:],
                            done=done[size:],
                            advantages=advantages[size:],
                            act_log_probs=act_log_probs[size:],
                            entropies=entropies[size:],
                            value_estimates=values[size:],
                            global_returns=global_returns[size:],
                            global_advantages=global_advantages[size:],
                            returns=returns[size:])

            batches.append(batch)
        return batches


class DQNBuffer:
    """A simple numpy replay buffer."""

    def __init__(
            self,
            obs_dim: int,
            size: int,
            action_dim: int = 1,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32) if action_dim == 1 else np.zeros([size, action_dim],
                                                                                            dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.use_act_prob = False
        self.act_prob_buf = np.zeros([size], dtype=np.float32) if action_dim == 1 else np.zeros([size, action_dim],
                                                                                                dtype=np.float32)
        self.max_size = size
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
            alpha: float = 0.7,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedDQNBuffer, self).__init__(
            obs_dim, size, action_dim, n_step, gamma
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

        indices = self._sample_proportional(batch_size)

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

    def _sample_proportional(self, batch_size: int) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
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

import torch
import torch.nn as nn
from typing import List
import numpy as np
from Swish import Swish
from torch.distributions import Categorical
from utils import torch_from_np


class ActorCriticPolicy(nn.Module):
    class SharedActorCritic(nn.Module):
        def __init__(self, state_dim, act_dim, hidden_size, device: str = 'cpu'):
            nn.Module.__init__(self)
            self.device = device
            self.body = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.actions_out = nn.Sequential(
                [nn.Sequential(nn.Linear(hidden_size, shape), nn.Softmax(dim=-1)) for shape in
                 act_dim])
            self.value_out = nn.Linear(hidden_size, 1)

        def forward(self, obs):
            obs = torch_from_np(obs, self.device)
            hidden = self.body(obs)
            actions = self.actions_out(hidden)
            value = self.value_out(hidden)
            return actions, value

    class Critic(nn.Module):
        def __init__(self, state_dim, hidden_size, device: str = 'cpu'):
            nn.Module.__init__(self)
            self.device = device

            self.value_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

        def forward(self, obs: np.ndarray):
            obs_tensor = torch_from_np(obs, self.device)
            value = self.value_layer(obs_tensor)
            return value

    class Actor(nn.Module):
        def __init__(self, state_dim, act_dim, hidden_size, device: str = 'cpu'):
            nn.Module.__init__(self)
            self.device = device

            self.hidden_state = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.actions_out = [nn.Sequential(nn.Linear(hidden_size, shape), nn.Softmax(dim=-1)) for shape in
                                act_dim]
            self.actions_out = nn.Sequential(*self.actions_out)

        def forward(self, obs: np.ndarray):
            obs_tensor = torch_from_np(obs, self.device)
            hidden_state = self.hidden_state(obs_tensor)
            action_prob = []
            dists = []
            for layer in self.actions_out:
                action_out = layer(hidden_state)
                action_prob.append(action_out)
                dist = Categorical(action_out)
                dists.append(dist)

            return dists

    def __init__(self, state_dim, action_dim, hidden_size: int = 256, shared_actor_critic: bool = False,
                 device: str = 'cpu'):
        super(ActorCriticPolicy, self).__init__()
        self.device = device
        if not shared_actor_critic:
            self.critic = self.Critic(state_dim, hidden_size, device).to(device)
            self.actor = self.Actor(state_dim, action_dim, hidden_size, device).to(device)
        else:
            self.actor_critic = self.SharedActorCritic(state_dim, action_dim, hidden_size).to(device)

    def get_probs_and_entropies(self, acts: torch.FloatTensor, dists: List[torch.distributions.Categorical]):
        cumulated_log_probs = torch.zeros([acts.shape[0]]).to(self.device)
        entropies = torch.zeros([acts.shape[0]]).to(self.device)
        for i, dist in enumerate(dists):
            cumulated_log_probs = torch.add(cumulated_log_probs, dist.log_prob(acts[:, i]))
            entropies = torch.add(entropies, dist.entropy())

        all_log_probs = torch.cat([torch.log(dist.probs) for dist in dists], dim=-1)
        return cumulated_log_probs, entropies, all_log_probs


class ValueNetwork(nn.Module):
    """
    Class for encoding the input to hidden_size vector with num_layers
    Input: [Obs_dim*batch_size]
    Output: [Act_dim*batch_size] value for every action
    """

    def __init__(self, obs_dim, act_dim, hidden_size, num_layers, device: str = 'cpu'):
        super(ValueNetwork, self).__init__()
        self.device = device
        layers = [nn.Linear(obs_dim, hidden_size)]
        layers.append(Swish())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(Swish())
        layers.append(nn.Linear(hidden_size, act_dim))
        self.seq_layers = torch.nn.Sequential(*layers)

    def forward(self, obs: np.ndarray):
        obs_tensor = torch_from_np(obs, device=self.device)
        return self.seq_layers(obs_tensor)


class PolicyValueNetwork(nn.Module):
    """
    Contains q1 and q2 Network
    Input: obs
    Output: Value(act_size)
    """

    def __init__(self, obs_dim, act_dim, hidden_size, hidden_layers, device: str = 'cpu'):
        super(PolicyValueNetwork, self).__init__()
        self.device = device
        self.values_out = sum(act_dim)

        self.q1 = ValueNetwork(obs_dim, self.values_out, hidden_size, hidden_layers, self.device).to(self.device)
        self.q2 = ValueNetwork(obs_dim, self.values_out, hidden_size, hidden_layers, self.device).to(self.device)

    def forward(self, obs: np.ndarray) -> (torch.Tensor, torch.Tensor):
        q1_out = self.q1(obs)
        q2_out = self.q2(obs)
        return q1_out, q2_out

import torch
import torch.nn as nn
from typing import List
import numpy as np
from Swish import Swish
from torch.distributions import Categorical
from utils import torch_from_np


class ActorCriticPolicy(nn.Module):
    class SharedActorCritic(nn.Module):
        def __init__(self, state_dim, act_dim, hidden_size, num_hidden_layers):
            nn.Module.__init__(self)
            body = [nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh())]
            for i in range(num_hidden_layers - 1):
                body.append(nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                ))
            self.body = nn.Sequential(*body)
            self.actions_out = nn.Sequential(
                *[nn.Sequential(nn.Linear(hidden_size, shape), nn.Softmax(dim=-1)) for shape in
                  act_dim])
            self.value_out = nn.Linear(hidden_size, 1)

        def forward(self, obs: torch.Tensor):
            hidden = self.body(obs)
            dists = []
            for layer in self.actions_out:
                action_out = layer(hidden)
                dist = Categorical(action_out)
                dists.append(dist)
            value = self.value_out(hidden)
            return dists, value

    class Critic(nn.Module):
        def __init__(self, state_dim, hidden_size, num_hidden_layers):
            nn.Module.__init__(self)

            hidden_layers = [nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
            )]
            for i in range(num_hidden_layers - 1):
                hidden_layers.append(nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                ))
            out_layer = nn.Linear(hidden_size, 1)

            self.value_layer = nn.Sequential(
                *hidden_layers, out_layer
            )

        def forward(self, obs: torch.Tensor):
            assert type(obs) == torch.Tensor
            value = self.value_layer(obs)
            return value

    class Actor(nn.Module):
        def __init__(self, state_dim, act_dim, hidden_size, num_hidden_layers):
            nn.Module.__init__(self)

            hidden_layers = [nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
            )]
            for i in range(num_hidden_layers - 1):
                hidden_layers.append(nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                ))
            self.hidden_layers = nn.Sequential(*hidden_layers)
            self.actions_layer = [nn.Sequential(nn.Linear(hidden_size, shape), nn.Softmax(dim=-1)) for shape in
                                  act_dim]
            self.actions_out = nn.Sequential(*self.actions_layer)

        def forward(self, obs: torch.Tensor):
            assert type(obs) == torch.Tensor
            hidden_state = self.hidden_layers(obs)
            action_prob = []
            dists = []
            for layer in self.actions_out:
                action_out = layer(hidden_state)
                action_prob.append(action_out)
                dist = Categorical(action_out)
                dists.append(dist)

            return dists

    def __init__(self, state_dim, action_dim, hidden_size: int = 256, num_hidden_layers: int = 2,
                 shared_actor_critic: bool = False):
        super(ActorCriticPolicy, self).__init__()
        if not shared_actor_critic:
            self.critic = self.Critic(state_dim, hidden_size, num_hidden_layers)
            self.actor = self.Actor(state_dim, action_dim, hidden_size, num_hidden_layers)
        else:
            self.policy = self.SharedActorCritic(state_dim, action_dim, hidden_size, num_hidden_layers)


class ValueNetwork(nn.Module):
    """
    Class for encoding the input to hidden_size vector with num_layers
    Input: [Obs_dim*batch_size]
    Output: [Act_dim*batch_size] value for every action
    """

    def __init__(self, obs_dim, act_dim, hidden_size, num_layers):
        super(ValueNetwork, self).__init__()
        layers = [nn.Linear(obs_dim, hidden_size), Swish()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(Swish())
        layers.append(nn.Linear(hidden_size, act_dim))
        self.seq_layers = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor):
        return self.seq_layers(obs)


class PolicyValueNetwork(nn.Module):
    """
    Contains q1 and q2 Network
    Input: obs
    Output: Value(act_size)
    """

    def __init__(self, obs_dim, act_dim, hidden_size, hidden_layers):
        super(PolicyValueNetwork, self).__init__()
        # self.device = device
        values_out = sum(act_dim)

        self.q1 = ValueNetwork(obs_dim, values_out, hidden_size, hidden_layers)
        self.q2 = ValueNetwork(obs_dim, values_out, hidden_size, hidden_layers)

    def forward(self, obs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        q1_out = self.q1(obs)
        q2_out = self.q2(obs)
        return q1_out, q2_out

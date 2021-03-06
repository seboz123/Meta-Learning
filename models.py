import torch
import torch.nn as nn
import math
from utils import Swish
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

# Contains Implementations of Neural Networks used for PPO/SAC/Rainbow

# Actor-Critic Policy Implementation for PPO/SAC
# Not used in Current Version
# class ActorCriticPolicy(nn.Module):
#     class SharedActorCritic(nn.Module):
#         def __init__(self, state_dim, act_dim, hidden_size, num_hidden_layers, enable_curiosity: bool):
#             nn.Module.__init__(self)
#             self.enable_curiosity = enable_curiosity
#             body = [nn.Sequential(
#                 nn.Linear(state_dim, hidden_size),
#                 Swish())]
#             for i in range(num_hidden_layers - 1):
#                 body.append(nn.Sequential(
#                     nn.Linear(hidden_size, hidden_size),
#                     Swish(),
#                 ))
#             self.body = nn.Sequential(*body)
#             self.actions_out = nn.Sequential(
#                 *[nn.Sequential(nn.Linear(hidden_size, shape), nn.Softmax(dim=-1)) for shape in
#                   act_dim])
#             self.value_out = nn.Linear(hidden_size, 1)
#             if enable_curiosity:
#                 self.curiosity_value = nn.Linear(hidden_size, 1)
#
#         def forward(self, obs: torch.Tensor):
#             hidden = self.body(obs)
#             dists = []
#             actions = []
#             for layer in self.actions_out:
#                 action_probs = layer(hidden)
#                 dist = Categorical(action_probs)
#                 dists.append(dist)
#                 actions.append(dist.sample())
#             return dists, actions
#
#         def critic_pass(self, obs: torch.Tensor):
#             hidden = self.body(obs)
#             values = [self.value_out(hidden).squeeze()]
#             if self.enable_curiosity:
#                 values.append(self.curiosity_value(hidden).squeeze())
#             return values
#
#     class Critic(nn.Module):
#         def __init__(self, state_dim, hidden_size, num_hidden_layers, enable_curiosity: bool):
#             nn.Module.__init__(self)
#
#             self.enable_curiosity = enable_curiosity
#
#             hidden_layers = [nn.Sequential(
#                 nn.Linear(state_dim, hidden_size),
#                 Swish(),
#             )]
#             for i in range(num_hidden_layers - 1):
#                 hidden_layers.append(nn.Sequential(
#                     nn.Linear(hidden_size, hidden_size),
#                     Swish(),
#                 ))
#
#
#             self.hidden_layer = nn.Sequential(
#                 *hidden_layers)
#             if enable_curiosity:
#                 self.curiosity_layer = nn.Linear(hidden_size, 1)
#             self.out_layer = nn.Linear(hidden_size, 1)
#
#
#         def forward(self, obs: torch.Tensor):
#             hidden = self.hidden_layer(obs)
#             values = [self.out_layer(hidden).squeeze()]
#             if self.enable_curiosity:
#                 values.append(self.curiosity_layer(hidden))
#             return values
#
#     class Actor(nn.Module):
#         def __init__(self, state_dim, act_dim, hidden_size, num_hidden_layers):
#             nn.Module.__init__(self)
#
#             hidden_layers = [nn.Sequential(
#                 nn.Linear(state_dim, hidden_size),
#                 Swish(),
#             )]
#             for i in range(num_hidden_layers - 1):
#                 hidden_layers.append(nn.Sequential(
#                     nn.Linear(hidden_size, hidden_size),
#                     Swish(),
#                 ))
#             self.hidden_layers = nn.Sequential(*hidden_layers)
#             self.actions_layer = [nn.Sequential(nn.Linear(hidden_size, shape), nn.Softmax(dim=-1)) for shape in
#                                   act_dim]
#             self.actions_out = nn.Sequential(*self.actions_layer)
#
#         def forward(self, obs: torch.Tensor):
#             assert type(obs) == torch.Tensor
#             hidden_state = self.hidden_layers(obs)
#             action_prob = []
#             dists = []
#             for layer in self.actions_out:
#                 action_out = layer(hidden_state)
#                 action_prob.append(action_out)
#                 dist = Categorical(action_out)
#                 dists.append(dist)
#
#             return dists
#
#     def __init__(self, state_dim, action_dim, hyperparameters: {}, shared_actor_critic: bool):
#         super(ActorCriticPolicy, self).__init__()
#         if not shared_actor_critic:
#             self.critic = self.Critic(state_dim, hyperparameters['layer_size'], hyperparameters['hidden_layers'], hyperparameters['enable_curiosity'])
#             self.actor = self.Actor(state_dim, action_dim, hyperparameters['layer_size'], hyperparameters['hidden_layers'])
#         else:
#             self.policy = self.SharedActorCritic(state_dim, action_dim, hyperparameters['layer_size'], hyperparameters['hidden_layers'], hyperparameters['enable_curiosity'])
#
#
# class ValueNetwork(nn.Module):
#     """
#     Class for encoding the input to hidden_size vector with num_layers
#     Input: [Obs_dim*batch_size]
#     Output: [Act_dim*batch_size] value for every action
#     """
#
#     def __init__(self, obs_dim, act_dim, hidden_size, num_layers, enable_curiosity: bool):
#         super(ValueNetwork, self).__init__()
#         self.enable_curiosity = enable_curiosity
#         layers = [nn.Linear(obs_dim, hidden_size), Swish()]
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(hidden_size, hidden_size))
#             layers.append(Swish())
#         self.hidden_layers = torch.nn.Sequential(*layers)
#         self.out_layer = nn.Linear(hidden_size, act_dim)
#         if enable_curiosity:
#             self.curiosity_layer = nn.Linear(hidden_size, act_dim)
#
#
#     def forward(self, obs: torch.Tensor):
#         hidden = self.hidden_layers(obs)
#         values = [self.out_layer(hidden).squeeze()]
#         if self.enable_curiosity:
#             values.append(self.curiosity_layer(hidden).squeeze())
#         return values
#
# class PolicyValueNetwork(nn.Module):
#     """
#     Contains q1 and q2 Network
#     Input: obs
#     Output: Value(act_size)
#     """
#
#     def __init__(self, obs_dim, act_dim, hidden_size, hidden_layers, enable_curiosity: bool):
#         super(PolicyValueNetwork, self).__init__()
#         values_out = sum(act_dim)
#
#         self.q1 = ValueNetwork(obs_dim, values_out, hidden_size, hidden_layers, enable_curiosity)
#         self.q2 = ValueNetwork(obs_dim, values_out, hidden_size, hidden_layers, enable_curiosity)
#
#     def forward(self, obs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         q1_out = self.q1(obs)
#         q2_out = self.q2(obs)
#         return q1_out, q2_out

# class TorchNetworks:
#     def __init__(self, hyperparameters: {}, obs_dim, act_dim, device):
#         hidden_size = hyperparameters['layer_size']
#         num_hidden_layers = hyperparameters['hidden_layers']
#         init_entropy_coeff = hyperparameters['init_coeff']
#         adaptive_ent_coeff = hyperparameters['adaptive_coeff']
#
#         self.device = device
#         self.discrete_target_entropy_scale = 0.2
#         self.init_entropy_coeff = init_entropy_coeff
#
#         ### Actor Critic Policy Network ####
#         self.policy_network = ActorCriticPolicy(obs_dim, act_dim, hyperparameters, False).to(device)  # Pi
#         ### Value Network of Policy ######
#         ### Contains 2x Q-Networks ######
#         self.value_network = PolicyValueNetwork(obs_dim, act_dim, hidden_size, num_hidden_layers,
#                                                 hyperparameters['enable_curiosity']).to(
#             device)  # Q
#         ### Target Value Network ####
#         self.target_network = ValueNetwork(obs_dim, 1, hidden_size, num_hidden_layers,
#                                            hyperparameters['enable_curiosity']).to(device)  # V
#
#         self.soft_update(self.policy_network, self.target_network, 1.0)
#         self.use_adaptive_entropy_coeff = adaptive_ent_coeff
#         if adaptive_ent_coeff:
#             self.log_entropy_coeff = torch.nn.Parameter(
#                 torch.log(torch.as_tensor([self.init_entropy_coeff] * len(act_dim))).to(device),
#                 requires_grad=True
#             ).to(device)
#         else:
#             self.log_entropy_coeff = torch.log(torch.as_tensor([self.init_entropy_coeff] * len(act_dim))).to(device)
#
#         self.target_entropy = [self.discrete_target_entropy_scale * np.log(i).astype(np.float32) for i in act_dim]
#
#     def soft_update(self, source: nn.Module, target: nn.Module, tau: float):
#         for source_param, target_param in zip(source.parameters(), target.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


# Implementation of Noise-Layers for Rainbow
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())

# Implementation of DeepQ Network for Rainbow
class DeepQNetwork(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_size: int,
            num_hidden_layers: int,
            atom_size: int,
            support: torch.Tensor,
            enable_curiosity: bool
    ):
        """Initialization."""
        super(DeepQNetwork, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        feature_layers = [nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            Swish(),
        )]
        for i in range(num_hidden_layers - 1):
            feature_layers.append(nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            Swish()))
        self.feature_layer = nn.Sequential(*feature_layers)

        # set advantage layer
        self.advantage_hidden_layer = nn.Sequential(NoisyLinear(hidden_size, hidden_size), Swish())
        self.advantage_layer = NoisyLinear(hidden_size, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = nn.Sequential(NoisyLinear(hidden_size, hidden_size), Swish())
        self.value_layer = NoisyLinear(hidden_size, atom_size)


        if enable_curiosity:
            self.curiosity_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def get_curiosity_value(self, x: torch.Tensor):
        feature = self.feature_layer(x)
        value_hidden = self.value_hidden_layer(feature)
        value = self.curiosity_layer(value_hidden)
        return value

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = self.advantage_hidden_layer(feature)
        val_hid = self.value_hidden_layer(feature)

        advantage = self.advantage_layer(adv_hid).reshape(
            -1, self.out_dim, self.atom_size
        )
        value_o = self.value_layer(val_hid)

        value = value_o.reshape(-1, 1, self.atom_size)

        q_atoms = value + advantage - torch.mean(advantage, dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist_out = dist.clamp(min=1e-3)  # for avoiding nans

        return dist_out

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()



import torch
import torch.nn as nn
import numpy as np
from typing import List
import gym.spaces
import itertools

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

# Contains Helper Functions for Different Modules

# Convert Numpy to Torch
def torch_from_np(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    return torch.as_tensor(np.asanyarray(array)).to(device)

# Branching for Q-Values of PPO/SAC
# Not used right now
def condense_q_stream(q_out: torch.Tensor, actions: torch.Tensor, action_space, enable_curiosity: bool) -> torch.Tensor:
    condensed_qs = []
    one_hot_actions = actions_to_onehot(actions, action_space)
    k = 2 if enable_curiosity else 1
    for i in range(k):
        branched_q1 = break_into_branches(q_out[:, :, i], action_space)
        only_qs = torch.stack([torch.sum(act_branch * q_branch, dim=1, keepdim=True) for act_branch, q_branch in
                     zip(one_hot_actions, branched_q1)])
        cond_q = torch.mean(only_qs, dim=0)
        condensed_qs.append(cond_q)

    return condensed_qs

# Get probabilites and entropies of sampled actions
def get_probs_and_entropies(acts: torch.FloatTensor, dists: List[torch.distributions.Categorical], device):
    cumulated_log_probs = torch.zeros([acts.shape[0]]).to(device)
    entropies = torch.zeros([acts.shape[0]]).to(device)
    for i, dist in enumerate(dists):
        cumulated_log_probs = torch.add(cumulated_log_probs, dist.log_prob(acts[:, i]))
        entropies = torch.add(entropies, dist.entropy())

    all_log_probs = torch.cat([torch.log(dist.probs) for dist in dists], dim=-1)
    return cumulated_log_probs, entropies, all_log_probs

# Encode Actions to One-Hot Encoding (MultiDiscrete to Discrete Action Space)
# Used in Rainbow Module
def actions_to_onehot(
        discrete_actions: torch.Tensor, action_size: List[int]
) -> List[torch.Tensor]:
    """
    Takes a tensor of discrete actions and turns it into a List of onehot encoding for each
    action.
    :param discrete_actions: Actions in integer form.
    :param action_size: List of branch sizes. Should be of same size as discrete_actions'
    last dimension.
    :return: List of one-hot tensors, one representing each branch.
    """
    onehot_branches = [
        torch.nn.functional.one_hot(_act.T, action_size[i]).float()
        for i, _act in enumerate(discrete_actions.long().T)
    ]

    return onehot_branches

def break_into_branches(
    concatenated_logits: torch.Tensor, action_size: List[int]
) -> List[torch.Tensor]:
    """
    Takes a concatenated set of logits that represent multiple discrete action branches
    and breaks it up into one Tensor per branch.
    :param concatenated_logits: Tensor that represents the concatenated action branches
    :param action_size: List of ints containing the number of possible actions for each branch.
    :return: A List of Tensors containing one tensor per branch.
    """
    action_idx = [0] + list(np.cumsum(action_size))
    branched_logits = [
        concatenated_logits[:, action_idx[i] : action_idx[i + 1]]
        for i in range(len(action_size))
    ]
    return branched_logits

# Definition of Swish activation function
def swish(input, beta: float = 0.8):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(beta*input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class Swish(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input, beta: float = 1):
        '''
        Forward pass of the function.
        '''
        return swish(input, beta) # simply apply already implemented SiLU

# Action Flattener for Rainbow Implementation
# Flattens multidiscrete actions to discrete actions and backwards
class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        """
        Initialize the flattener.
        :param branched_action_space: A List containing the sizes of each branch of the action
        space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
        """
        self.action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self.action_shape)
        self.action_space = gym.spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        """
        Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
        Each key in the Dict maps to one unique set of branched actions, and each value
        contains the List of branched actions.
        """
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        # Dict should be faster than List for large action spaces
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        """
        Convert a scalar discrete action into a unique set of branched actions.
        :param: action: A scalar value representing one of the discrete actions.
        :return: The List containing the branched actions.
        """
        return self.action_lookup[action]

# Initialize a UnityEnvironment with the defined Environment Parameters
# Used for Rainbow Meta-Learning


def init_unity_env(env_path: str, maze_rows: int, maze_cols: int, maze_seed: int, random_agent: int, random_target: int, difficulty: int, agent_rot: int,
                   agent_x: int, agent_z: int, target_x: int, target_z: int, enable_heatmap: bool, enable_sight_cone: bool, base_port: int) -> UnityEnvironment:

    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration_parameters(time_scale=10.0)
    env_parameters_channel = EnvironmentParametersChannel()
    env_parameters_channel.set_float_parameter("maze_seed", float(maze_seed))
    env_parameters_channel.set_float_parameter("maze_rows", float(maze_rows))
    env_parameters_channel.set_float_parameter("maze_cols", float(maze_cols))
    env_parameters_channel.set_float_parameter("target_x", float(target_x))
    env_parameters_channel.set_float_parameter("target_z", float(target_z))
    env_parameters_channel.set_float_parameter("agent_x", float(agent_x))
    env_parameters_channel.set_float_parameter("agent_z", float(agent_z))
    env_parameters_channel.set_float_parameter("random_agent", float(random_agent))
    env_parameters_channel.set_float_parameter("random_target", float(random_target))
    env_parameters_channel.set_float_parameter("enable_heatmap", float(enable_heatmap))
    env_parameters_channel.set_float_parameter("enable_sight_cone", float(enable_sight_cone))
    env_parameters_channel.set_float_parameter("agent_rot", float(agent_rot))
    env_parameters_channel.set_float_parameter("difficulty", float(difficulty))

    env = UnityEnvironment(file_name=env_path,
                           base_port=base_port, timeout_wait=120,
                           no_graphics=False, seed=0,
                           side_channels=[engine_configuration_channel, env_parameters_channel])

    return env

# Function to step the environment
# Supports Multi-Agent Environements

def step_env(env: UnityEnvironment, actions: np.array):
    agents_transitions = {}
    for brain in env.behavior_specs:
        actions = np.resize(actions,
                            (len(env.get_steps(brain)[0]), len(env.behavior_specs[brain].discrete_action_branches)))
        env.set_actions(brain, actions)
        env.step()
        decision_steps, terminal_steps = env.get_steps(brain)

        for agent_id_decisions in decision_steps:
            agents_transitions[agent_id_decisions] = [decision_steps[agent_id_decisions].obs,
                                                      decision_steps[agent_id_decisions].reward, False]

        for agent_id_terminated in terminal_steps:
            agents_transitions[agent_id_terminated] = [terminal_steps[agent_id_terminated].obs,
                                                       terminal_steps[agent_id_terminated].reward, True]

    return agents_transitions
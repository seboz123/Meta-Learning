import torch
import numpy as np
from typing import List

def torch_from_np(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    return torch.as_tensor(np.asanyarray(array)).to(device)

def condense_q_stream(q_out: torch.Tensor, actions: torch.Tensor, action_space) -> torch.Tensor:
    one_hot_actions = actions_to_onehot(actions, action_space)
    branched_q1 = break_into_branches(q_out, action_space)
    only_qs = torch.stack([torch.sum(act_branch * q_branch, dim=1, keepdim=True) for act_branch, q_branch in
                 zip(one_hot_actions, branched_q1)])
    condensed_q = torch.mean(only_qs, dim=0)

    return condensed_q

def get_probs_and_entropies(acts: torch.FloatTensor, dists: List[torch.distributions.Categorical], device):
    # print(acts.shape[0])
    # print('Reserved:', round(torch.cuda.memory_reserved() / 1023, 1), 'MB')
    # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1023, 1), 'MB')
    # print('Cached: ', round(torch.cuda.memory_cached(0) / 1023, 1), 'MB')
    if device == 'cuda':
        test_tensor = torch.FloatTensor([0]).to(device)
        cumulated_log_probs = torch.zeros([acts.shape[0]]).to('cuda')
        entropies = torch.zeros([acts.shape[0]]).to('cuda')
    else:
        cumulated_log_probs = torch.zeros([acts.shape[0]])
        entropies = torch.zeros([acts.shape[0]])
    for i, dist in enumerate(dists):
        cumulated_log_probs = torch.add(cumulated_log_probs, dist.log_prob(acts[:, i]))
        entropies = torch.add(entropies, dist.entropy())

    all_log_probs = torch.cat([torch.log(dist.probs) for dist in dists], dim=-1)
    return cumulated_log_probs, entropies, all_log_probs


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
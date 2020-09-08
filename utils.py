import torch
import numpy as np
from typing import List

def torch_from_np(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    return torch.as_tensor(np.asanyarray(array)).to(device)


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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import Swish


class CuriosityModule():
    def __init__(self, obs_size: int, enc_size: int, enc_layers: int, device: torch.device, action_flattener,
                 learning_rate: int = 0.001):
        self.device = device

        self.encoderModel = VectorEncoder(obs_size, enc_size, enc_layers).to(device)

        self.forwardModel = ForwardModel(enc_size, sum(action_flattener.action_shape)).to(device)
        self.inverseModel = InverseModel(enc_size, action_flattener.action_shape).to(device)

        # self.flattener = action_flattener
        parameters = list(self.forwardModel.parameters()) + list(self.inverseModel.parameters()) + list(
            self.encoderModel.parameters())
        self.optimizer = optim.Adam(parameters, lr=learning_rate)

        exp_schedule = lambda epoch: 0.99 ** epoch

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=exp_schedule)

    def calc_loss(self, memory_n, indices) -> torch.Tensor:
        samples = memory_n.sample_batch_from_idxs(indices)
        action_shape = self.flattener.action_shape
        actions = []
        for action in samples['acts']:
            action = self.flattener.lookup_action(int(action))
            tmp_actions = []
            for i, shape in enumerate(action_shape):
                for k in range(shape):
                    if (k == action[i]):
                        tmp_actions.append(1)
                    else:
                        tmp_actions.append(0)
            actions.append(tmp_actions)
        acts = torch.Tensor(actions).to(dtype=torch.int64, device=self.device)
        obs = torch.FloatTensor(samples['obs']).to(dtype=torch.float32, device=self.device)
        obs = self.encoderModel(obs)
        next_obs = torch.FloatTensor(samples['next_obs']).to(dtype=torch.float32, device=self.device)
        next_obs = self.encoderModel(next_obs)

        # Encode obs and next_obs
        # Predict states and actions
        pred_states = self.forwardModel(obs, acts)
        pred_acts = self.inverseModel(obs, next_obs)
        # Calculate MSE and Cross-Entropy-Error
        mean_squared_loss = nn.MSELoss()
        forward_loss = mean_squared_loss(pred_states, next_obs)

        cross_entropy_loss = -torch.log(pred_acts + 1e-10) * acts
        inverse_loss = torch.mean(torch.sum(cross_entropy_loss, dim=1))

        return forward_loss, inverse_loss


class VectorEncoder(nn.Module):
    def __init__(
            self,
            obs_size: int,
            enc_size: int,
            num_layers: int,
    ):
        super(VectorEncoder, self).__init__()
        layers = [nn.Sequential(nn.Linear(obs_size, enc_size), Swish())]
        for _ in range(num_layers - 1):
            layers.append(nn.Sequential(nn.Linear(enc_size, enc_size), Swish()))
        self.encoder = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)


class ForwardModel(nn.Module):
    def __init__(
            self,
            enc_size: int,
            act_size: int,
            num_layers: int
    ):
        super(ForwardModel, self).__init__()
        layers = [nn.Sequential(
            nn.Linear(enc_size + act_size, enc_size),
            Swish()
        )]
        for _ in range(num_layers - 1):
            layers.append(nn.Sequential(
                nn.Linear(enc_size, enc_size),
                Swish()
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([obs, act.to(dtype=torch.float32)], dim=1)
        enc_state = self.layers(combined)
        return enc_state


class InverseModel(nn.Module):
    def __init__(
            self,
            enc_size: int,
            act_size: list,
            hidden_size: int,
            hidden_layers: int,
    ):
        """Initialization."""
        super(InverseModel, self).__init__()

        layers = [nn.Sequential(
            nn.Linear(enc_size * 2, hidden_size),
            Swish()
        )]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Sequential(nn.Linear(enc_size * 2, hidden_size),
                                        Swish()))
        self.hidden_layers = nn.Sequential(*layers)
        self.out_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_size, shape), nn.Softmax()) for shape in act_size])

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([obs, next_obs], dim=1)
        hidden = self.hidden_layers(combined)
        pred_action = torch.cat([layer(hidden) for layer in self.out_layers], dim=1)
        return pred_action

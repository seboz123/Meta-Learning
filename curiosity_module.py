import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import Swish

class CuriosityModule():
    def __init__(self, obs_size: int, enc_size: int, enc_layers: int, device: torch.device, action_flattener, learning_rate: int = 0.001):
        self.device = device

        self.encoderModel = VectorEncoder(obs_size, enc_size, enc_layers).to(device)

        self.forwardModel = ForwardModel(enc_size, sum(action_flattener.action_shape)).to(device)
        self.inverseModel = InverseModel(enc_size, action_flattener.action_shape, device).to(device)

        # self.flattener = action_flattener
        parameters = list(self.forwardModel.parameters()) + list(self.inverseModel.parameters()) + list(self.encoderModel.parameters())
        self.optimizer = optim.Adam(parameters, lr=learning_rate)
        linear_schedule = lambda epoch: max((1 - epoch / max_scheduler_steps), 0.0001)
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_schedule)
    def calc_loss(self, memory_n, indices) -> torch.Tensor:
        samples = memory_n.sample_batch_from_idxs(indices)
        action_shape = self.flattener.action_shape
        actions = []
        for action in samples['acts']:
            action = self.flattener.lookup_action(int(action))
            tmp_actions = []
            for i, shape in enumerate(action_shape):
                for k in range(shape):
                    if(k == action[i]):
                        tmp_actions.append(1)
                    else:
                        tmp_actions.append(0)
            actions.append(tmp_actions)
        acts = torch.Tensor(actions).to(dtype=torch.int64,device=self.device)
        obs = torch.FloatTensor(samples['obs']).to(dtype=torch.float32,device=self.device)
        obs = self.encoderModel.forward(obs)
        next_obs = torch.FloatTensor(samples['next_obs']).to(dtype=torch.float32,device=self.device)
        next_obs = self.encoderModel.forward(next_obs)

        # Encode obs and next_obs
        # Predict states and actions
        pred_states = self.forwardModel.forward(obs, acts)
        pred_acts = self.inverseModel.forward(obs, next_obs)
        # Calculate MSE and Cross-Entropy-Error
        mean_squared_loss = nn.MSELoss()
        forward_loss = mean_squared_loss(pred_states, next_obs)

        cross_entropy_loss = -torch.log(pred_acts + 1e-10)*acts
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
        layerIn = nn.Sequential(nn.Linear(obs_size, enc_size), Swish())
        layers = []
        for _ in range(num_layers-1):
            layers.append(nn.Sequential(nn.Linear(enc_size, enc_size), Swish())
        self.encoder = nn.Sequential(layerIn, *layers)

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
        self.layerIn = nn.Sequential(
            nn.Linear(enc_size+act_size, enc_size),
        )

        self.layerOut = nn.Sequential(
            nn.Linear(256, enc_size)
        )


    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([obs, act.to(dtype=torch.float32)], dim=1)
        features = self.layerIn(combined)
        features = swish(features, beta=1)
        pred_state = self.layerOut(features)
        return pred_state

class InverseModel(nn.Module):
    def __init__(
            self,
            enc_size: int,
            act_size: list,
            device: str
    ):
        """Initialization."""
        super(InverseModel, self).__init__()
        self.device = device

        self.hidden = nn.Sequential(
            nn.Linear(enc_size*2, 256),
        )

        self.out_layers = [nn.Sequential(nn.Linear(256, shape), nn.Softmax()).to(self.device) for shape in act_size]


    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([obs, next_obs], dim=1)
        hidden = self.hidden(combined)
        hidden = swish(hidden, 1.0)
        pred_action = torch.cat([layer(hidden) for layer in self.out_layers], dim=1)
        return pred_action

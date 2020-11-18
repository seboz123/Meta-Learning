import torch
import torch.nn as nn
import torch.optim as optim

from utils import Swish, torch_from_np, actions_to_onehot

# Implementation of Curiosity Module

class CuriosityModule():
    def __init__(self, obs_size: int, enc_size: int, hidden_layers: int, hidden_size: int,device: torch.device,
                 learning_rate: int = 0.001, action_shape=None, action_flattener=None):
        assert action_shape is not None or action_flattener is not None
        self.device = device
        self.action_shape = action_shape
        self.action_flattener = action_flattener

        self.encoderModel = VectorEncoder(obs_size, enc_size, hidden_layers).to(device)

        if action_shape is not None:
            self.forwardModel = ForwardModel(enc_size=enc_size, act_size=len(action_shape), hidden_size=hidden_size, hidden_layers=hidden_layers).to(device)
            self.inverseModel = InverseModel(enc_size=enc_size, act_size=action_shape, hidden_size=hidden_size, hidden_layers=hidden_layers).to(device)

        self.networks = [self.encoderModel, self.forwardModel, self.inverseModel]

        # self.flattener = action_flattener
        parameters = list(self.forwardModel.parameters()) + list(self.inverseModel.parameters()) + list(
            self.encoderModel.parameters())
        self.optimizer = optim.Adam(parameters, lr=learning_rate)

        exp_schedule = lambda epoch: 0.99 ** epoch

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=exp_schedule)

    def get_networks(self):
        return self.networks


    def calc_loss_rainbow(self, samples, action_flattener, action_dim: list) -> torch.Tensor:
        actions = [action_flattener.lookup_action(int(action)) for action in samples['acts']]
        acts = torch.Tensor(actions).to(dtype=torch.int64, device=self.device)
        true_actions = torch.cat(actions_to_onehot(acts, action_dim), dim=1)

        obs = torch_from_np(samples['obs'], self.device)
        next_obs = torch_from_np(samples['next_obs'], self.device)

        enc_obs = self.encoderModel(obs)
        enc_next_obs = self.encoderModel(next_obs)
        # Encode obs and next_obs
        # Predict states and actions
        pred_states = self.forwardModel(enc_obs, acts)
        pred_acts = self.inverseModel(enc_obs, enc_next_obs)

        # Calculate MSE and Cross-Entropy-Error
        mean_squared_loss = nn.MSELoss()
        forward_loss = mean_squared_loss(pred_states, enc_next_obs)

        cross_entropy_loss = -torch.log(pred_acts + 1e-10) * true_actions
        inverse_loss = torch.mean(torch.sum(cross_entropy_loss, dim=1))

        return forward_loss, inverse_loss

    def calc_loss_ppo_sac(self, batch):
        actions = torch_from_np(batch.actions, self.device).to(dtype=torch.float32)
        obs = torch_from_np(batch.observations, self.device).to(dtype=torch.float32)
        next_obs = torch_from_np(batch.next_observations, self.device).to(dtype=torch.float32)

        encoded_obs = self.encoderModel(obs).to(dtype=torch.float32)
        encoded_next_obs = self.encoderModel(next_obs).to(dtype=torch.float32)

        pred_states = self.forwardModel(encoded_obs, actions)

        pred_acts = self.inverseModel(encoded_obs, encoded_next_obs)

        # Calculate MSE and Cross-Entropy-Error
        mean_squared_loss = nn.MSELoss()


        if self.action_shape is not None:
            one_hot = actions_to_onehot(actions,self.action_shape)
        true_actions = torch.cat(one_hot, dim=1)

        cross_entropy_loss = -torch.log(pred_acts + 1e-10) * true_actions
        inverse_loss = torch.mean(torch.sum(cross_entropy_loss, dim=1))

        forward_loss = mean_squared_loss(pred_states, encoded_next_obs)

        return forward_loss, inverse_loss

    def evaluate(self, obs, acts, next_obs):
        with torch.no_grad():
            obs = torch_from_np(obs, self.device)
            actions = torch_from_np(acts, self.device)
            next_obs = torch_from_np(next_obs, self.device)

            encoded_obs = self.encoderModel(obs)
            predicted_next_obs = self.forwardModel(encoded_obs, actions)
            encoded_next_obs = self.encoderModel(next_obs)

            squared_error = torch.sum(0.5 * (predicted_next_obs - encoded_next_obs)**2, dim=1)
            return squared_error




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
            hidden_size: int,
            hidden_layers: int
    ):
        super(ForwardModel, self).__init__()
        layers = [nn.Sequential(
            nn.Linear(enc_size + act_size, hidden_size),
            Swish()
        )]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                Swish()
            ))
        out_layer = nn.Linear(hidden_size, enc_size)
        self.layers = nn.Sequential(*layers, out_layer)

    def forward(self, enc_state: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([enc_state, act], dim=1)
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
            layers.append(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        Swish()))
        self.hidden_layers = nn.Sequential(*layers)
        self.out_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_size, shape), nn.Softmax()) for shape in act_size])

    def forward(self, encoded_obs: torch.Tensor, encoded_next_obs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([encoded_obs, encoded_next_obs], dim=1)
        hidden = self.hidden_layers(combined)
        actions = []
        for layer in self.out_layers:
            actions.append(layer(hidden))
        pred_action = torch.cat([branch for branch in actions], dim=1)
        return pred_action

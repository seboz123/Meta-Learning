import numpy as np
import time
from typing import Tuple
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from buffers import SACBuffer
from models import ActorCriticPolicy, PolicyValueNetwork, ValueNetwork
from curiosity_module import CuriosityModule
from utils import torch_from_np, break_into_branches, condense_q_stream, get_probs_and_entropies, init_unity_env
from env_utils import step_env

flatten = lambda l: [item for sublist in l for item in sublist]


class TorchNetworks:
    def __init__(self, hyperparameters: {}, obs_dim, act_dim, device):
        hidden_size = hyperparameters['layer_size']
        num_hidden_layers = hyperparameters['hidden_layers']
        init_entropy_coeff = hyperparameters['init_coeff']
        adaptive_ent_coeff = hyperparameters['adaptive_coeff']

        self.device = device
        self.discrete_target_entropy_scale = 0.2
        self.init_entropy_coeff = init_entropy_coeff

        ### Actor Critic Policy Network ####
        self.policy_network = ActorCriticPolicy(obs_dim, act_dim, hyperparameters, False).to(device)  # Pi
        ### Value Network of Policy ######
        ### Contains 2x Q-Networks ######
        self.value_network = PolicyValueNetwork(obs_dim, act_dim, hidden_size, num_hidden_layers, hyperparameters['enable_curiosity']).to(
            device)  # Q
        ### Target Value Network ####
        self.target_network = ValueNetwork(obs_dim, 1, hidden_size, num_hidden_layers, hyperparameters['enable_curiosity']).to(device)  # V

        self.soft_update(self.policy_network, self.target_network, 1.0)
        self.use_adaptive_entropy_coeff = adaptive_ent_coeff
        if adaptive_ent_coeff:
            self.log_entropy_coeff = torch.nn.Parameter(
                torch.log(torch.as_tensor([self.init_entropy_coeff] * len(act_dim))).to(device),
                requires_grad=True
            ).to(device)
        else:
            self.log_entropy_coeff = torch.log(torch.as_tensor([self.init_entropy_coeff] * len(act_dim))).to(device)

        self.target_entropy = [self.discrete_target_entropy_scale * np.log(i).astype(np.float32) for i in act_dim]

        # self.policy_params = list(self.policy_network.parameters())
        # self.value_params = list(self.value_network.parameters())

    def soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


class SAC_Meta_Learner:
    def __init__(
            self,
            device: str,
            writer: SummaryWriter,
            is_meta_learning: bool
    ):
        self.device = device
        print("Using: " + str(self.device))
        self.writer = writer
        self.meta_step = 0
        self.step = 0
        self.policy: ActorCriticPolicy
        self.obs_space: Tuple
        self.action_space: Tuple
        self.enable_curiosity = False
        self.curiosity = None
        self.is_meta_learning = is_meta_learning

    def get_default_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['Algorithm'] = "SAC"

        hyperparameters['logging_period'] = 2000

        hyperparameters['enable_curiosity'] = True
        hyperparameters['curiosity_lambda'] = 10    # Weight factor of curiosity loss
        hyperparameters['curiosity_beta'] = 0.2     # Factor for using more of forward loss or more of inverse loss
        hyperparameters['curiosity_enc_size'] = 32  # Encoding size of curiosity_module
        hyperparameters['curiosity_layers'] = 2     # Layers of Curiosity Modules
        hyperparameters['curiosity_units'] = 128    # Number of hidden units for curiosity modules

        hyperparameters['max_steps'] = 1000000
        hyperparameters['learning_rate'] = 0.0001   # Typical range: 0.00001 - 0.001
        hyperparameters['batch_size'] = 512         # Typical range: 32-512
        hyperparameters['hidden_layers'] = 2
        hyperparameters['layer_size'] = 256
        hyperparameters['time_horizon'] = 64
        hyperparameters['gamma'] = 0.99
        hyperparameters['decay_lr'] = True

        hyperparameters['buffer_size'] = 30000  # Replay buffer size

        hyperparameters[
            'init_coeff'] = 0.5  # Typical range: 0.05 - 0.5 Decrease for less exploration but faster convergence
        hyperparameters['tau'] = 0.005  # Typical range: 0.005 - 0.01 decrease for stability
        hyperparameters['steps_per_update'] = 12  # Typical range: 1 - 20 -> Equal to number of agents in scene
        hyperparameters['adaptive_coeff'] = True  # Whether entropy coeff should be learned

        return hyperparameters

    def set_env_and_detect_spaces(self, env, task):
        env.reset()
        self.env = env
        self.task = task
        result = 0
        for brain_name in self.env.behavior_specs:
            # Set up flattened action space
            branches = self.env.behavior_specs[brain_name].discrete_action_branches
            for shape in self.env.behavior_specs[brain_name].observation_shapes:
                if (len(shape) == 1):
                    result += shape[0]
        print("Space detected successfully.")
        print("Observation space detected as {}, Action space detected as: {}".format(result, branches))

        self.obs_space = result
        self.action_space = branches
        self.env.reset()

    def init_networks_and_optimizers(self, hyperparameters: dict):

        adaptive_coeff = hyperparameters['adaptive_coeff']

        max_scheduler_steps = hyperparameters['max_steps']
        learning_rate = hyperparameters['learning_rate']

        obs_dim = self.obs_space
        action_dim = self.action_space

        self.networks = TorchNetworks(hyperparameters, obs_dim, action_dim, device=self.device)
        self.policy = self.networks.policy_network

        value_parameters = list(self.networks.value_network.parameters()) + list(self.policy.critic.parameters())
        policy_parameters = list(self.policy.actor.parameters())

        self.all_parameters = value_parameters + policy_parameters

        self.policy_optimizer = optim.Adam(policy_parameters, lr=learning_rate)

        self.value_optimizer = optim.Adam(value_parameters, lr=learning_rate)
        linear_schedule = lambda epoch: max((1 - epoch / max_scheduler_steps), 1e-6)

        if adaptive_coeff:
            self.entropy_optimizer = optim.Adam([self.networks.log_entropy_coeff], lr=learning_rate)
            self.learning_rate_scheduler_e = optim.lr_scheduler.LambdaLR(self.entropy_optimizer,
                                                                         lr_lambda=linear_schedule)
        self.learning_rate_scheduler_p = optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda=linear_schedule)
        self.learning_rate_scheduler_v = optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda=linear_schedule)

        if hyperparameters['enable_curiosity']:
            self.enable_curiosity = True
            self.curiosity = CuriosityModule(obs_size=obs_dim, enc_size=hyperparameters['curiosity_enc_size'], hidden_layers=hyperparameters['curiosity_layers'],
                                             hidden_size=hyperparameters['curiosity_units'], learning_rate=0.003, device=self.device, action_shape=action_dim)
            print("Enabled curiosity module")

    def get_networks_and_parameters(self) -> dict:
        networks_and_parameters = {}
        networks_and_parameters['networks'] = []
        networks_and_parameters['networks'].append(self.networks.value_network)
        networks_and_parameters['networks'].append(self.networks.policy_network)
        # networks_and_parameters['networks'].append(self.networks.target_network)
        if self.enable_curiosity:
            for network in self.curiosity.get_networks():
                networks_and_parameters['networks'].append(network)

        networks_and_parameters['parameters'] = []
        if self.networks.use_adaptive_entropy_coeff:
            networks_and_parameters['parameters'].append(self.networks.log_entropy_coeff)
        if self.curiosity is not None:
            for network in self.curiosity.get_networks():
                networks_and_parameters['networks'].append(network)

        return networks_and_parameters

    def calc_q_loss(self, q1_out, q2_out, target_values, dones, rewards, gamma):
        q1_losses = []
        q2_losses = []
        if self.enable_curiosity:
            for i in range(2):
                q1_out[i] = q1_out[i].squeeze()
                q2_out[i] = q2_out[i].squeeze()
            with torch.no_grad():
                q_backup = rewards[:, i] + (1 - dones) * gamma * target_values[:, i]
            q1_tmp = 0.5 * torch.mean(torch.nn.functional.mse_loss(q_backup, q1_out[i]))
            q2_tmp = 0.5 * torch.mean(torch.nn.functional.mse_loss(q_backup, q2_out[i]))
            q1_losses.append(q1_tmp)
            q2_losses.append(q2_tmp)

        else:
            for i in range(1):
                q1_out[i] = q1_out[i].squeeze()
                q2_out[i] = q2_out[i].squeeze()
            with torch.no_grad():
                q_backup = rewards[:, i] + (1 - dones) * gamma * target_values[i]
            q1_tmp = 0.5 * torch.mean(torch.nn.functional.mse_loss(q_backup, q1_out[i]))
            q2_tmp = 0.5 * torch.mean(torch.nn.functional.mse_loss(q_backup, q2_out[i]))
            q1_losses.append(q1_tmp)
            q2_losses.append(q2_tmp)
        q1_loss = torch.mean(torch.stack(q1_losses))
        q2_loss = torch.mean(torch.stack(q2_losses))




        return q1_loss, q2_loss

    def calc_value_loss(self, log_probs, values, q1p_out, q2p_out, action_space):
        with torch.no_grad():
            _ent_coeff = torch.exp(self.networks.log_entropy_coeff)

        minimal_policy_qs = []
        action_probs = log_probs.exp()
        for q1, q2, value in zip(q1p_out, q2p_out, values):
            _branched_q1_policy_out = break_into_branches(q1 * action_probs, action_space)
            _branched_q2_policy_out = break_into_branches(q2 * action_probs, action_space)

            _q1_policy_mean = torch.mean(
                torch.stack(
                    [torch.sum(_br, dim=1, keepdim=True) for _br in _branched_q1_policy_out]
                )
            )
            _q2_policy_mean = torch.mean(
                torch.stack(
                    [torch.sum(_br, dim=1, keepdim=True) for _br in _branched_q2_policy_out]
                )
            )
            minimal_policy_qs.append(torch.min(_q1_policy_mean, _q2_policy_mean))

        value_losses = []
        branched_per_action_ent = break_into_branches(log_probs * log_probs.exp(), action_space)
        branched_ent_bonus = torch.stack(
            [
                torch.sum(_ent_coeff[i] * _lp, dim=1, keepdim=True)
                for i, _lp in enumerate(branched_per_action_ent)
            ]
        )
        for value, minimal_qs in zip(values, minimal_policy_qs):
            with torch.no_grad():
                v_backup = minimal_qs - torch.mean(branched_ent_bonus, axis=0)
                value_loss = 0.5 * torch.mean(torch.nn.functional.mse_loss(value.squeeze(), v_backup.squeeze()))

                value_losses.append(value_loss)

        value_loss = torch.mean(torch.stack(value_losses))
        return value_loss

    def calc_policy_loss(self, log_probs, q1p_out, action_space):
        _ent_coeff = torch.exp(self.networks.log_entropy_coeff)

        mean_q1 = torch.mean(torch.stack(list(q1p_out)), axis = 0)

        action_probs = torch.exp(log_probs)

        branched_per_act_entropy = break_into_branches(log_probs * action_probs, action_space)
        branched_q = break_into_branches(mean_q1 * action_probs, action_space)

        branched_policy_loss = torch.stack(
            [
                torch.sum(_ent_coeff[i] * _lp - _qt, dim=1, keepdim=True)
                for i, (_lp, _qt) in enumerate(zip(branched_per_act_entropy, branched_q))
            ]
        )
        batch_policy_loss = branched_policy_loss.squeeze()
        policy_loss = torch.mean(batch_policy_loss)

        return policy_loss

    def calc_entropy_loss(self, log_probs, action_space):
        with torch.no_grad():
            branched_per_act_entropy = break_into_branches(log_probs * log_probs.exp(), action_space)

            branched_ent_sums = torch.stack([
                torch.sum(_lp, axis=1, keepdim=True) + _te
                for _lp, _te in zip(branched_per_act_entropy, self.networks.target_entropy)
            ], axis=1)
            branched_ent_sums = torch.squeeze(branched_ent_sums)
        entropy_loss = -1 * torch.mean(torch.mean(self.networks.log_entropy_coeff * branched_ent_sums, axis=1), axis=0)
        return entropy_loss

    def calc_losses(self, batch: SACBuffer, hyperparameters: {}):
        losses = {}
        gamma = hyperparameters['gamma']
        observations = torch_from_np(batch.observations, self.device).to(dtype=torch.float32)
        next_observations = torch_from_np(batch.next_observations, self.device).to(dtype=torch.float32)
        rewards = torch_from_np(batch.rewards, self.device)
        dones = torch_from_np(batch.done, self.device)

        dists = self.policy.actor(observations)
        sampled_values = self.policy.critic(observations)


        actions = []
        for dist in dists:
            action = dist.sample()
            actions.append(action)
        actions = torch.stack(actions).transpose(0, 1)

        _, entropies, all_log_probs = get_probs_and_entropies(actions, dists, device=self.device)
        with torch.no_grad():
            q1_policy_out, q2_policy_out = self.networks.value_network(observations)

        with torch.no_grad():
            target_values_tmp = self.networks.target_network(next_observations)
            target_values = torch.stack(target_values_tmp, -1)

        q1_tmp, q2_tmp = self.networks.value_network(observations)
        q1_out = torch.stack(q1_tmp, -1)
        q2_out = torch.stack(q2_tmp, -1)

        condensed_q1 = condense_q_stream(q1_out, actions, batch.action_space, hyperparameters['enable_curiosity'])
        condensed_q2 = condense_q_stream(q2_out, actions, batch.action_space, hyperparameters['enable_curiosity'])

        value_loss = self.calc_value_loss(all_log_probs, sampled_values, q1_policy_out, q2_policy_out,
                                          batch.action_space)

        policy_loss = self.calc_policy_loss(all_log_probs, q1_policy_out, batch.action_space)

        entropy_loss = self.calc_entropy_loss(all_log_probs, batch.action_space)

        q1_loss, q2_loss = self.calc_q_loss(condensed_q1, condensed_q2, target_values, dones, rewards, gamma=gamma)

        # Update the target network

        total_value_loss = value_loss + q1_loss + q2_loss

        losses['Policy Loss'] = policy_loss.item()
        losses['Value Loss'] = value_loss.item()
        losses['Entropy Loss'] = entropy_loss.item()
        losses['Q1 Loss'] = q1_loss.item()
        losses['Q2 Loss'] = q2_loss.item()

        curiosity_loss = None
        if self.enable_curiosity:
            f_loss, i_loss = self.curiosity.calc_loss_ppo_sac(batch)
            curiosity_loss = hyperparameters['curiosity_lambda'] * (
                    hyperparameters['curiosity_beta'] * f_loss + (1 - hyperparameters['curiosity_beta']) * i_loss)
            losses['Curiosity Forward Loss'] = f_loss.item()
            losses['Curiosity Inverse Loss'] = i_loss.item()

        return policy_loss, total_value_loss, entropy_loss, losses, curiosity_loss

    def generate_trajectories_and_fill_buffer(self, buffer, time_horizon: int = 512) -> {}:
        env = self.env

        # Create transition dict
        transitions = {}
        trajectory_lengths = []  # length of agent trajectories, maxlen = timehorizon/done

        episode_lengths = []
        cumulative_rewards = []

        first_iteration = True
        buffer_finished = False
        appended_steps = 0

        if len(buffer) > 0:
            buffer.remove_old_obs(remove_percentage=0.2)

        while not buffer_finished:
            if first_iteration:
                # This part is to set the inital actions for the agents
                for brain in env.behavior_specs:
                    decision_steps, terminal_steps = env.get_steps(brain)
                    num_agents = len(decision_steps)
                    if (num_agents == 0):
                        env.reset()
                        decision_steps, terminal_steps = env.get_steps(brain)
                        num_agents = len(decision_steps)
                    agent_current_step = [0 for _ in range(num_agents)]
                    current_added_reward = [0 for _ in range(num_agents)]

                    actions = np.zeros((num_agents, len(self.action_space)))
                    print("Brain :" + str(brain) + " with " + str(num_agents) + " agents detected.")
                    agent_ptr = [0 for _ in range(num_agents)]  # Create pointer for agent transitions

                    for i in range(num_agents):
                        transitions[i] = {
                            'obs_buf': np.zeros([time_horizon, self.obs_space], dtype=np.float32),
                            'n_obs_buf': np.zeros([time_horizon, self.obs_space], dtype=np.float32),
                            'acts_buf': np.zeros([time_horizon, len(self.action_space)], dtype=np.float32),
                            'rews_buf': np.zeros([time_horizon, 2], dtype=np.float32),
                            'done_buf': np.zeros([time_horizon], dtype=np.float32)}

                    for agent_id_decisions in decision_steps:
                        init_state = decision_steps[agent_id_decisions].obs
                        init_state = flatten(init_state)
                        with torch.no_grad():
                            init_state = np.array(init_state)
                            init_state_tensor = torch_from_np(init_state, device=self.device)
                            dists = self.policy.actor(init_state_tensor)
                            action = [dist.sample() for dist in dists]
                            action = [action_branch.detach().cpu().numpy() for action_branch in action]
                            actions[agent_id_decisions] = action

                        transitions[agent_id_decisions]['obs_buf'][agent_ptr[agent_id_decisions]] = init_state
                        transitions[agent_id_decisions]['acts_buf'][agent_ptr[agent_id_decisions]] = action

                    for agent_id_terminated in terminal_steps:
                        init_state = terminal_steps[agent_id_terminated].obs
                        init_state = flatten(init_state)
                        with torch.no_grad():
                            init_state = np.array(init_state)
                            init_state_tensor = torch_from_np(init_state, device=self.device)
                            dists = self.policy.actor(init_state_tensor)
                            action = [dist.sample() for dist in dists]
                            action = [action_branch.detach().cpu().numpy() for action_branch in action]

                        actions[agent_id_terminated] = action
                        transitions[agent_id_terminated]['obs_buf'][agent_ptr[agent_id_terminated]] = init_state
                        transitions[agent_id_terminated]['acts_buf'][agent_ptr[agent_id_terminated]] = action

            # Step environment
            next_experiences = step_env(self.env, np.array(actions))
            # Create action vector to store actions
            actions = np.zeros((num_agents, len(self.action_space)))

            for agent_id in next_experiences:

                reward = next_experiences[agent_id][1]  # Reward
                next_obs = flatten(next_experiences[agent_id][0])  # Next_obs
                done = next_experiences[agent_id][2]  # Done


                # Store trajectory of every Agent {Agent0:[obs,act,rew,n_obs,done,obs,act,rew,n_obs,done,.... Agent1: ....}
                transitions[agent_id]['rews_buf'][agent_ptr[agent_id], 0] = reward
                transitions[agent_id]['n_obs_buf'][agent_ptr[agent_id]] = next_obs
                transitions[agent_id]['done_buf'][agent_ptr[agent_id]] = done

                current_added_reward[agent_id] += reward
                agent_current_step[agent_id] += 1

                if done or agent_ptr[agent_id] == time_horizon - 1:
                    # If the corresponding agent is done or trajectory is max length, get the next action
                    with torch.no_grad():
                        next_obs_tmp = np.array(next_obs)
                        next_obs_tensor = torch_from_np(next_obs_tmp, device=self.device)
                        dists = self.policy.actor(next_obs_tensor)
                        action = [dist.sample() for dist in dists]
                        next_action = [action_branch.detach().cpu().numpy() for action_branch in action]
                        actions[agent_id] = action

                    actions[agent_id] = next_action

                    if agent_ptr[agent_id] == time_horizon - 1:
                        with torch.no_grad():
                            next_obs_tmp = np.array(next_obs)
                            next_obs_tensor = torch_from_np(next_obs_tmp, device=self.device)
                            value_estimate = self.policy.critic(next_obs_tensor).detach().cpu().numpy()
                            transitions[agent_id]['rews_buf'][agent_ptr[agent_id], 0] = value_estimate

                    if done:
                        episode_lengths.append(agent_current_step[agent_id])
                        cumulative_rewards.append(current_added_reward[agent_id])
                        current_added_reward[agent_id] = 0
                        agent_current_step[agent_id] = 0

                    trajectory_lengths.append(agent_ptr[agent_id] + 1)
                    if agent_ptr[agent_id] + 1 + len(buffer) > buffer.max_buffer_size:
                        buffer_finished = True
                        break
                    if self.enable_curiosity:
                        transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1, 1] = self.curiosity.evaluate(obs=transitions[agent_id]['obs_buf'][:agent_ptr[agent_id] + 1],
                                                acts=transitions[agent_id]['acts_buf'][:agent_ptr[agent_id] + 1],
                                                next_obs=transitions[agent_id]['n_obs_buf'][:agent_ptr[agent_id] + 1])


                    appended_steps += len(transitions[agent_id]['done_buf'][:agent_ptr[agent_id] + 1])

                    buffer.store(transitions[agent_id]['obs_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['acts_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['n_obs_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['done_buf'][:agent_ptr[agent_id] + 1])


                    transitions[agent_id]['obs_buf'] = np.zeros(
                        [time_horizon, self.obs_space], dtype=np.float32)
                    transitions[agent_id]['acts_buf'] = np.zeros([time_horizon, len(self.action_space)],
                                                                 dtype=np.float32)
                    transitions[agent_id]['rews_buf'] = np.zeros([time_horizon, 2],
                                                                 dtype=np.float32)
                    transitions[agent_id]['n_obs_buf'] = np.zeros(
                        [time_horizon, self.obs_space], dtype=np.float32)
                    transitions[agent_id]['done_buf'] = np.zeros([time_horizon],
                                                                 dtype=np.float32)
                    agent_ptr[agent_id] = 0

                else:  # If the corresponding agent is not done, continue
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs

                    with torch.no_grad():
                        obs = np.array(next_obs)
                        obs_tensor = torch_from_np(obs, device=self.device)
                        dists = self.policy.actor(obs_tensor)
                        action = [dist.sample() for dist in dists]
                        next_action = [action_branch.detach().cpu().numpy() for action_branch in action]
                        actions[agent_id_decisions] = action

                    transitions[agent_id]['acts_buf'][agent_ptr[agent_id]] = next_action
                    actions[agent_id] = next_action

            first_iteration = False

        # If this update step has generated new finished episodes, report them
        if len(cumulative_rewards) > 0:
            print("Mean Cumulative Reward: {} at step {}".format(np.mean(cumulative_rewards), self.step))
            print("Mean Episode Length: {} at step {}".format(np.mean(trajectory_lengths), self.step))
            if self.is_meta_learning:
                self.writer.add_scalar(
                    'Task: ' + str(self.task) + r"/Meta Step: " + str(self.meta_step) + r"/Cumulative Reward",
                    np.mean(cumulative_rewards), self.step)
                self.writer.add_scalar(
                    'Task: ' + str(self.task) + r"/Meta Step: " + str(self.meta_step) + r"/Mean Episode Length",
                    np.mean(episode_lengths), self.step)
            else:
                self.writer.add_scalar(
                    "Environment/Cumulative Reward", np.mean(cumulative_rewards), self.step)
                self.writer.add_scalar("Environment/Episode Length",
                                       np.mean(episode_lengths), self.step)

        return appended_steps, np.mean(cumulative_rewards), np.mean(trajectory_lengths)

    def close_env(self):
        self.env.close()

    def train(self, hyperparameters: dict):
        self.writer.add_text("Hyperparameters", str(hyperparameters))
        print("Started run with following hyperparameters:")
        print("Started SAC training with {} steps to take".format(hyperparameters['max_steps']))
        for key in hyperparameters:
            print("{:<25s} {:<20s}".format(key, str(hyperparameters[key])))
        max_steps = hyperparameters['max_steps']
        batch_size = hyperparameters['batch_size']
        time_horizon = hyperparameters['time_horizon']
        buffer_size = hyperparameters['buffer_size']

        steps_per_update = hyperparameters['steps_per_update']
        tau = hyperparameters['tau']

        replay_buffer = SACBuffer(max_buffer_size=buffer_size, obs_space=self.obs_space, action_space=self.action_space)

        mean_rewards = []
        mean_episode_lengths = []

        value_losses, policy_losses, entropy_losses, q1_losses, q2_losses = [], [], [], [], []

        self.step = 0
        logging_steps = 0

        while self.step < max_steps:

            steps_taken, mean_reward, mean_episode_length = self.generate_trajectories_and_fill_buffer(
                buffer=replay_buffer, time_horizon=time_horizon)

            if self.step > logging_steps*hyperparameters['logging_period']:
                if self.is_meta_learning:
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(
                            self.meta_step) + r"/SAC Cumulative Reward",
                        np.mean(mean_rewards), self.step)
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(
                            self.meta_step) + r"/SAC Mean Episode Lengths",
                        np.mean(mean_episode_lengths), self.step)
                else:
                    self.writer.add_scalar(
                        "Environment/Cumulative Reward", np.mean(mean_rewards), self.step)
                    self.writer.add_scalar("Environment/Episode Length",
                                           np.mean(mean_episode_lengths), self.step)
                mean_rewards.clear()
                mean_episode_lengths.clear()

            print("Buffer filled, {} steps taken".format(steps_taken))
            mean_rewards.append(mean_reward)
            mean_episode_lengths.append(mean_episode_length)

            self.step += steps_taken
            update_steps = 0
            frame_start = time.time()
            while update_steps * steps_per_update < steps_taken:
                update_steps += 1
                print("Updating. Step {} of {}".format(update_steps*steps_per_update, steps_taken))
                batch = replay_buffer.sample_batch(batch_size=batch_size)
                p_loss, total_v_loss, e_loss,losses, curiosity_loss = self.calc_losses(batch, hyperparameters)
                value_losses.append(losses['Value Loss'])
                policy_losses.append(losses['Policy Loss'])
                entropy_losses.append(losses['Entropy Loss'])
                q1_losses.append(losses['Q1 Loss'])
                q2_losses.append(losses['Q2 Loss'])

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                if hyperparameters['adaptive_coeff']:
                    self.entropy_optimizer.zero_grad()
                    loss_start = time.time()
                    e_loss.backward()
                p_loss.backward()
                total_v_loss.backward()
                print("Loss calc took: {}".format(time.time()-loss_start))

                self.entropy_optimizer.step()

                if self.enable_curiosity:
                    self.curiosity.optimizer.zero_grad()
                    curiosity_loss.backward()
                    for network in self.curiosity.get_networks():
                        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                    self.curiosity.optimizer.step()
                    self.curiosity.learning_rate_scheduler.step()

                torch.nn.utils.clip_grad_norm_(self.all_parameters, 10.0)

                self.policy_optimizer.step()
                self.value_optimizer.step()

                if (hyperparameters['decay_lr']):
                    self.learning_rate_scheduler_p.step(epoch=self.step)
                    self.learning_rate_scheduler_v.step(epoch=self.step)
                    if hyperparameters['adaptive_coeff']:
                        self.learning_rate_scheduler_e.step(epoch=self.step)

                if update_steps % 10 == 0:
                    self.networks.soft_update(self.networks.policy_network.critic,
                                              self.networks.target_network, tau=tau)

            if self.step > logging_steps*hyperparameters['logging_period']:
                logging_steps += 1

                if self.is_meta_learning:
                    self.writer.add_scalar('Meta Learning Parameters/Learning Rate',
                                           self.learning_rate_scheduler_v.get_lr()[0], self.meta_step)
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(self.meta_step) + r"/SAC Policy Loss",
                        np.mean(policy_losses), self.step)
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(self.meta_step) + r"/SAC Value Loss",
                        np.mean(value_losses), self.step)
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(self.meta_step) + r"/SAC Entropy Loss",
                        np.mean(entropy_losses), self.step)
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(self.meta_step) + r"/SAC Q1 Loss",
                        np.mean(q1_losses), self.step)
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(self.meta_step) + r"/SAC Q2 Loss",
                        np.mean(q2_losses), self.step)
                else:
                    self.writer.add_scalar("Policy/Learning Rate", self.learning_rate_scheduler_v.get_lr()[0], self.step)
                    self.writer.add_scalar("Losses/Entropy Loss",
                        np.mean(entropy_losses), self.step)
                    self.writer.add_scalar("Losses/Policy Loss",
                        np.mean(policy_losses), self.step)
                    self.writer.add_scalar(
                        "Losses/Value Loss",
                        np.mean(value_losses), self.step)
                    self.writer.add_scalar(
                        "Losses/Q1 Loss",
                        np.mean(q1_losses), self.step)
                    self.writer.add_scalar(
                        "Losses/Q2 Loss",
                        np.mean(q2_losses), self.step)

                value_losses.clear()
                policy_losses.clear()
                entropy_losses.clear()
                q1_losses.clear()
                q2_losses.clear()


            frame_end = time.time()
            print("Current Update rate = {} updates per second".format(steps_taken / (frame_end - frame_start)))


if __name__ == '__main__':

    run_id = "results/sac_1"
    writer = SummaryWriter(run_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    sac_module = SAC_Meta_Learner(device, writer=writer, is_meta_learning=False)

    env = init_unity_env("mMaze.app", maze_rows=3, maze_cols=3, maze_seed=0, random_agent=0, random_target=0,
                         agent_x=0, agent_z=0, target_x=0, target_z=1, base_port=4000)

    ########################## Hyperparameters for train Run ############################
    #####################################################################################

    training_parameters = sac_module.get_default_hyperparameters()

    training_parameters['Algorithm'] = "SAC"
    training_parameters['enable_curiosity'] = False
    training_parameters['run_id'] = run_id

    training_parameters['max_steps'] = 1000000
    training_parameters['buffer_size'] = 5000  # Replay buffer size
    training_parameters['learning_rate'] = 0.0001  # Typical range: 0.00001 - 0.001
    training_parameters['batch_size'] = 512  # Typical range: 32-512
    training_parameters['hidden_layers'] = 2
    training_parameters['layer_size'] = 128
    training_parameters['time_horizon'] = 512
    training_parameters['gamma'] = 0.99
    training_parameters['decay_lr'] = True

    training_parameters[
        'init_coeff'] = 0.5  # Typical range: 0.05 - 0.5 Decrease for less exploration but faster convergence
    training_parameters['tau'] = 0.005  # Typical range: 0.005 - 0.01 decrease for stability
    training_parameters['steps_per_update'] = 10  # Typical range: 1 - 20 -> Equal to number of agents in scene
    training_parameters['adaptive_coeff'] = True  # Whether entropy coeff should be learned

    sac_module.set_env_and_detect_spaces(env, task=0)

    sac_module.init_networks_and_optimizers(training_parameters)
    sac_module.train(training_parameters)

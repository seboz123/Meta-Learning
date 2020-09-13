import numpy as np
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from buffers import SACBuffer
from models import ActorCriticPolicy, PolicyValueNetwork, ValueNetwork
from utils import torch_from_np, break_into_branches, condense_q_stream, get_probs_and_entropies, init_unity_env
from env_utils import step_env

flatten = lambda l: [item for sublist in l for item in sublist]


class TorchNetworks:
    def __init__(self, obs_dim, act_dim, hidden_size, num_hidden_layers, init_entropy_coeff, adaptive_ent_coeff, device):
        self.device = device
        self.discrete_target_entropy_scale = 0.2
        self.init_entropy_coeff = init_entropy_coeff
        self.policy_network = ActorCriticPolicy(obs_dim, act_dim, 256).to(device)  # Pi

        self.value_network = PolicyValueNetwork(obs_dim, act_dim, hidden_size, num_hidden_layers).to(
            device)  # Q

        self.target_network = ValueNetwork(obs_dim, 1, hidden_size, num_hidden_layers).to(device)  # V

        self.soft_update(self.policy_network, self.target_network, 1.0)
        self.use_adaptive_entropy_coeff = adaptive_ent_coeff
        if adaptive_ent_coeff:
            self.log_entropy_coeff = torch.nn.Parameter(
                torch.log(torch.as_tensor([self.init_entropy_coeff] * len(act_dim))).to(device),
                requires_grad=True
            ).to(device)
        else:
            self.log_entropy_coeff = self.init_entropy_coeff
        self.target_entropy = [self.discrete_target_entropy_scale * np.log(i).astype(np.float32) for i in act_dim]

        # self.policy_params = list(self.policy_network.parameters())
        # self.value_params = list(self.value_network.parameters())

    def soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


class SAC_Meta_Learner:
    """
    """

    def __init__(
            self,
            device: str,
            writer: SummaryWriter = None,
    ):
        self.device = device
        print("Using: " + str(self.device))
        self.writer = writer
        self.meta_step = 0
        self.step = 0
        self.policy: ActorCriticPolicy
        self.obs_space: Tuple
        self.action_space: Tuple

    def get_default_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['Algorithm'] = "SAC"

        hyperparameters['max_steps'] = 1000000
        hyperparameters['learning_rate'] = 0.0001  # Typical range: 0.00001 - 0.001
        hyperparameters['batch_size'] = 512  # Typical range: 32-512
        hyperparameters['hidden_layers'] = 2
        hyperparameters['layer_size'] = 256
        hyperparameters['time_horizon'] = 64
        hyperparameters['gamma'] = 0.99
        hyperparameters['decay_lr'] = True


        hyperparameters['buffer_size'] = 30000  # Replay buffer size

        hyperparameters['init_coeff'] = 0.5  # Typical range: 0.05 - 0.5 Decrease for less exploration but faster convergence
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

    def get_networks_and_parameters(self) -> dict:
        networks_and_parameters = {}
        networks_and_parameters['networks'] = []
        networks_and_parameters['networks'].append(self.networks.value_network)
        networks_and_parameters['networks'].append(self.networks.policy_network)
        networks_and_parameters['networks'].append(self.networks.target_network)
        networks_and_parameters['parameters'] = []
        if self.networks.use_adaptive_entropy_coeff:
            networks_and_parameters['parameters'].append(self.networks.log_entropy_coeff)

        return networks_and_parameters

    def init_networks_and_optimizers(self, hyperparameters: dict):

        init_ent_coeff = hyperparameters['init_coeff']
        adaptive_coeff = hyperparameters['adaptive_coeff']
        hidden_size = hyperparameters['layer_size']
        hidden_layers = hyperparameters['hidden_layers']
        max_scheduler_steps = hyperparameters['max_steps']
        learning_rate = hyperparameters['learning_rate']

        obs_dim = self.obs_space
        action_dim = self.action_space
        self.networks = TorchNetworks(obs_dim, action_dim, hidden_size, hidden_layers, init_entropy_coeff=init_ent_coeff, adaptive_ent_coeff=adaptive_coeff, device=self.device)
        self.policy = self.networks.policy_network

        value_parameters = list(self.networks.value_network.parameters()) + list(self.policy.critic.parameters())
        policy_parameters = list(self.policy.actor.parameters())

        self.all_parameters = value_parameters + policy_parameters

        self.policy_optimizer = optim.Adam(policy_parameters, lr=learning_rate)
        self.value_optimizer = optim.Adam(value_parameters, lr=learning_rate)
        self.entropy_optimizer = optim.Adam([self.networks.log_entropy_coeff], lr=learning_rate)

        linear_schedule = lambda epoch: max((1 - epoch / max_scheduler_steps), 1e-6)

        self.learning_rate_scheduler_p = optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda=linear_schedule)
        self.learning_rate_scheduler_v = optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda=linear_schedule)
        self.learning_rate_scheduler_e = optim.lr_scheduler.LambdaLR(self.entropy_optimizer, lr_lambda=linear_schedule)


    def calc_q_loss(self, q1_out, q2_out, target_values, dones, rewards, gamma):
        q1_out = q1_out.squeeze()
        q2_out = q2_out.squeeze()
        target_values = target_values.squeeze()
        with torch.no_grad():
            q_backup = rewards + (1 - dones) * gamma * target_values
        q1_loss = 0.5 * torch.mean(torch.nn.functional.mse_loss(q_backup, q1_out))
        q2_loss = 0.5 * torch.mean(torch.nn.functional.mse_loss(q_backup, q2_out))

        return q1_loss, q2_loss

    def calc_value_loss(self, log_probs, values, q1p_out, q2p_out, action_space):
        with torch.no_grad():
            _ent_coeff = torch.exp(self.networks.log_entropy_coeff)

        action_probs = log_probs.exp()
        _branched_q1_policy_out = break_into_branches(q1p_out * action_probs, action_space)
        _branched_q2_policy_out = break_into_branches(q2p_out * action_probs, action_space)

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
        minimal_policy_qs = torch.min(_q1_policy_mean, _q2_policy_mean)
        branched_per_action_ent = break_into_branches(log_probs * log_probs.exp(), action_space)
        branched_ent_bonus = torch.stack(
            [
                torch.sum(_ent_coeff[i] * _lp, dim=1, keepdim=True)
                for i, _lp in enumerate(branched_per_action_ent)
            ]
        )
        with torch.no_grad():
            v_backup = minimal_policy_qs - torch.mean(branched_ent_bonus, axis=0)

        value_loss = 0.5 * torch.mean(torch.nn.functional.mse_loss(values.squeeze(), v_backup.squeeze()))

        return value_loss

    def calc_policy_loss(self, log_probs, q1p_out, action_space):
        _ent_coeff = torch.exp(self.networks.log_entropy_coeff)

        mean_q1 = torch.mean(torch.stack(list(q1p_out)))

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

    def calc_losses(self, batch: SACBuffer, gamma):
        observations = torch_from_np(batch.observations, self.device)
        next_observations = torch_from_np(batch.next_observations, self.device)
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
            target_values = self.networks.target_network(next_observations)

        q1_out, q2_out = self.networks.value_network(observations)

        condensed_q1 = condense_q_stream(q1_out, actions, batch.action_space)
        condensed_q2 = condense_q_stream(q2_out, actions, batch.action_space)

        value_loss = self.calc_value_loss(all_log_probs, sampled_values, q1_policy_out, q2_policy_out,
                                          batch.action_space)

        policy_loss = self.calc_policy_loss(all_log_probs, q1_policy_out, batch.action_space)

        entropy_loss = self.calc_entropy_loss(all_log_probs, batch.action_space)
        q1_loss, q2_loss = self.calc_q_loss(condensed_q1, condensed_q2, target_values, dones, rewards, gamma=gamma)

        # Update the target network

        return policy_loss, value_loss, entropy_loss, q1_loss, q2_loss

    def generate_trajectories_and_fill_buffer(self, buffer, time_horizon) -> {}:
        env = self.env

        # Create transition dict
        transitions = {}
        trajectory_lengths = []  # length of agent trajectories, maxlen = timehorizon/done

        episode_lengths = []
        cumulative_rewards = []

        appended_steps = 0

        first_iteration = True
        buffer_finished = False

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
                            'rews_buf': np.zeros([time_horizon], dtype=np.float32),
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

                current_added_reward[agent_id] += reward

                # Store trajectory of every Agent {Agent0:[obs,act,rew,n_obs,done,obs,act,rew,n_obs,done,.... Agent1: ....}
                transitions[agent_id]['rews_buf'][agent_ptr[agent_id]] = reward
                transitions[agent_id]['n_obs_buf'][agent_ptr[agent_id]] = next_obs
                transitions[agent_id]['done_buf'][agent_ptr[agent_id]] = done

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
                            transitions[agent_id]['rews_buf'][agent_ptr[agent_id]] = value_estimate


                    if done:
                        episode_lengths.append(agent_current_step[agent_id])
                        agent_current_step[agent_id] = 0
                        cumulative_rewards.append(current_added_reward[agent_id])
                        current_added_reward[agent_id] = 0

                    trajectory_lengths.append(agent_ptr[agent_id] + 1)

                    for i in range(agent_ptr[agent_id] + 1):
                        # Store all experiences of the agent in the buffer
                        appended_steps += 1

                        if buffer.observations is None:
                            buffer.observations = transitions[agent_id]['obs_buf'][i]
                            buffer.actions = transitions[agent_id]['acts_buf'][i]
                            buffer.rewards = transitions[agent_id]['rews_buf'][i]
                            buffer.next_observations = transitions[agent_id]['n_obs_buf'][i]
                            buffer.done = transitions[agent_id]['done_buf'][i]
                        else:
                            buffer.observations = np.vstack((buffer.observations, transitions[agent_id]['obs_buf'][i]))
                            buffer.actions = np.vstack((buffer.actions, transitions[agent_id]['acts_buf'][i]))
                            buffer.rewards = np.append(buffer.rewards, transitions[agent_id]['rews_buf'][i])
                            buffer.next_observations = np.vstack(
                                (buffer.next_observations, transitions[agent_id]['n_obs_buf'][i]))
                            buffer.done = np.append(buffer.done, transitions[agent_id]['done_buf'][i])

                    if len(buffer) >= buffer.max_buffer_size:
                        length_counter = len(buffer) - buffer.max_buffer_size
                        buffer.observations = buffer.observations[length_counter:]
                        buffer.rewards = buffer.rewards[length_counter:]
                        buffer.actions = buffer.actions[length_counter:]
                        buffer.next_observations = buffer.next_observations[length_counter:]
                        buffer.done = buffer.done[length_counter:]
                        buffer_finished = True
                        break

                    transitions[agent_id]['obs_buf'] = np.zeros(
                        [time_horizon, self.obs_space], dtype=np.float32)
                    transitions[agent_id]['acts_buf'] = np.zeros([time_horizon, len(self.action_space)],
                                                                 dtype=np.float32)
                    transitions[agent_id]['rews_buf'] = np.zeros([time_horizon],
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

            self.writer.add_scalars('task_' + str(self.task) + r"\SAC Cumulative Reward",
                                    {r'\meta_step_' + str(self.meta_step): np.mean(cumulative_rewards)}, self.step)
            self.writer.add_scalars('task_' + str(self.task) + r'\SAC Mean Episode Length',
                                    {r'\meta_step_' + str(self.meta_step): np.mean(episode_lengths)}, self.step)
        return buffer, appended_steps, np.mean(cumulative_rewards), np.mean(trajectory_lengths)

    def close_env(self):
        self.env.close()

    def train(self, hyperparameters: dict):
        max_steps = hyperparameters['max_steps']
        batch_size = hyperparameters['batch_size']
        time_horizon = hyperparameters['time_horizon']
        gamma = hyperparameters['gamma']
        buffer_size = hyperparameters['buffer_size']

        steps_per_update = hyperparameters['steps_per_update']
        tau = hyperparameters['tau']

        print("Started SAC training with {} steps to take".format(hyperparameters['max_steps']))

        replay_buffer = SACBuffer(max_buffer_size=buffer_size, action_space=self.action_space)

        mean_rewards = []
        mean_episode_lengths = []
        self.step = 0

        while self.step < max_steps:
            buffer, steps_taken, mean_reward, mean_episode_length = self.generate_trajectories_and_fill_buffer(buffer=replay_buffer, time_horizon=time_horizon)
            mean_rewards.append(mean_reward)
            mean_episode_lengths.append(mean_episode_length)

            self.step += steps_taken
            update_steps = 0
            frame_start = time.time()
            value_losses = []
            policy_losses = []
            entropy_losses = []
            q1_losses = []
            q2_losses = []
            while update_steps * steps_per_update < steps_taken:
                batch = buffer.sample_batch(batch_size=batch_size)
                p_loss, v_loss, e_loss, q1_loss, q2_loss = self.calc_losses(batch, gamma)
                value_losses.append(v_loss.item())
                policy_losses.append(p_loss.item())
                entropy_losses.append(e_loss.item())
                q1_losses.append(q1_loss.item())
                q2_losses.append(q2_loss.item())
                total_value_loss = q1_loss + q2_loss + v_loss

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                self.entropy_optimizer.zero_grad()

                p_loss.backward()
                total_value_loss.backward()
                e_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.all_parameters, 10.0)

                self.policy_optimizer.step()
                self.value_optimizer.step()
                self.entropy_optimizer.step()

                if (hyperparameters['decay_lr']):
                    self.learning_rate_scheduler_e.step(epoch=self.step)
                    self.learning_rate_scheduler_p.step(epoch=self.step)
                    self.learning_rate_scheduler_v.step(epoch=self.step)

                self.networks.soft_update(self.networks.policy_network.critic,
                                                self.networks.target_network, tau=tau)

                update_steps += 1

            self.writer.add_scalars('task_' + str(self.task) + r"\SAC Entropy Loss",
                                    {r'\meta_step_'+str(self.meta_step): np.mean(entropy_losses)}, self.step)
            self.writer.add_scalars('task_' + str(self.task) + r'\SAC Policy Loss',
                                   {r'\meta_step_' + str(self.meta_step) : np.mean(policy_losses)}, self.step)
            self.writer.add_scalars('task_' + str(self.task) + r'\SAC Value Loss',
                                   {r'\meta_step_' + str(self.meta_step) : np.mean(value_losses)}, self.step)
            self.writer.add_scalars('task_' + str(self.task) + r'\SAC Q1 Loss',
                                    {r'\meta_step_' + str(self.meta_step): np.mean(q1_losses)}, self.step)
            self.writer.add_scalars('task_' + str(self.task) + r'\SAC Q2 Loss',
                                    {r'\meta_step_' + str(self.meta_step): np.mean(q2_losses)}, self.step)

            frame_end = time.time()
            print("Current Update rate = {} updates per second".format(steps_taken / (frame_end - frame_start)))

        return np.mean(mean_rewards), np.mean(mean_episode_lengths)

if __name__ == '__main__':

    run_id = "results/sac_1"
    writer = SummaryWriter(run_id)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    sac_module = SAC_Meta_Learner(device, writer=writer)

    env = init_unity_env("mMaze/RLProject.exe", maze_rows=3, maze_cols=3, maze_seed=0, random_agent=0, random_target=0, agent_x=0, agent_z=0, target_x=2, target_z=2)

    ########################## Hyperparameters for train Run ############################
    #####################################################################################

    training_parameters = {}

    training_parameters['Algorithm'] = "SAC"
    training_parameters['run_id'] = run_id


    training_parameters['max_steps'] = 1000000
    training_parameters['buffer_size'] = 2000 # Replay buffer size
    training_parameters['learning_rate'] = 0.0001 # Typical range: 0.00001 - 0.001
    training_parameters['batch_size'] = 512 # Typical range: 32-512
    training_parameters['hidden_layers'] = 2
    training_parameters['layer_size'] = 256
    training_parameters['time_horizon'] = 64
    training_parameters['gamma'] = 0.99
    training_parameters['decay_lr'] = True


    training_parameters['init_coeff'] = 0.5 # Typical range: 0.05 - 0.5 Decrease for less exploration but faster convergence
    training_parameters['tau'] = 0.005 # Typical range: 0.005 - 0.01 decrease for stability
    training_parameters['steps_per_update'] = 1 # Typical range: 1 - 20 -> Equal to number of agents in scene
    training_parameters['adaptive_coeff'] = True # Whether entropy coeff should be learned

    writer.add_text("training_parameters", str(training_parameters))
    print("Started run with following hyperparameters:")
    for key in training_parameters:
        print("{:<25s} {:<20s}".format(key, str(training_parameters[key])))

    sac_module.set_env_and_detect_spaces(env, task=0)

    sac_module.init_networks_and_optimizers(training_parameters)
    sac_module.train(training_parameters)



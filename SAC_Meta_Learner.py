import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from Buffer import Buffer
from networks import ActorCriticPolicy, PolicyValueNetwork, ValueNetwork
from utils import torch_from_np, actions_to_onehot, break_into_branches

flatten = lambda l: [item for sublist in l for item in sublist]


class Optimizer:
    def __init__(self, obs_dim, act_dim, hidden_size, num_hidden_layers):
        act_len = sum(act_dim)
        self.discrete_target_entropy_scale = 0.2
        self.init_entropy_coeff = 0.2 # Typical range: 0.05 - 0.5

        self.policy_network = ActorCriticPolicy(obs_dim, act_dim, 256) # Pi

        self.value_network = PolicyValueNetwork(obs_dim, act_dim, hidden_size, num_hidden_layers) # Q

        self.target_network = ValueNetwork(obs_dim, act_len, hidden_size, num_hidden_layers) # V

        self.soft_update(self.policy_network, self.target_network, 1.0)

        self.log_entropy_coeff = torch.nn.Parameter(
            torch.log(torch.as_tensor([self.init_entropy_coeff] * len(act_dim))),
            requires_grad = True
        )
        self.target_entropy = [self.discrete_target_entropy_scale * np.log(i).astype(np.float32) for i in act_dim]

        self.policy_params = list(self.policy_network.parameters())
        self.value_params = list(self.value_network.parameters())

        learning_rate = 0.002

#       self.policy_optimizer = torch.optim.Adam(self.policy_params, lr=learning_rate)
#       self.value_optimizer = torch.optim.Adam(self.value_params, lr=learning_rate)
#       self.entropy_optimizer = torch.optim.Adam(self.log_entropy_coeff, lr=learning_rate)

    def soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

class SAC_Meta_Learner:
    """
    """

    def __init__(
            self,
            device: torch.device,
            writer: SummaryWriter = None,
    ):
        self.device = device
        print("Using: " + str(self.device))
        self.writer = writer
        self.meta_step = 0
        self.policy: ActorCriticPolicy


    def init_network_and_optim(self, hidden_size: int = 256, learning_rate: float = 0.0003):
        obs_dim = self.obs_space
        action_dim = self.action_space
        print("Obs space detected as: " + str(self.obs_space))
        print("Action space detected as: " + str(self.action_space))
        self.networks = Optimizer(obs_dim, action_dim, 256, 2)
        self.policy = self.networks.policy_network

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        lambda2 = lambda epoch: 0.95**epoch
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda2)

    def set_environment(self, env):
        env.reset()
        self.env = env
        self.task = 0
        result = 0
        for brain_name in self.env.behavior_specs:
            # Set up flattened action space
            branches = self.env.behavior_specs[brain_name].discrete_action_branches
            for shape in self.env.behavior_specs[brain_name].observation_shapes:
                if (len(shape) == 1):
                    result += shape[0]
        self.obs_space = result
        self.action_space = branches
        self.env.reset()

    def discount_rewards(self, r, gamma=0.99, value_next=0.0):
        """
        Computes discounted sum of future rewards for use in updating value estimate.
        :param r: List of rewards.
        :param gamma: Discount factor.
        :param value_next: T+1 value estimate for returns calculation.
        :return: discounted sum of future rewards as list.
        """
        discounted_r = np.zeros_like(r)
        running_add = value_next
        for t in reversed(range(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def step_env(self, env: UnityEnvironment, actions: np.array):
        agents_transitions = {}
        for brain in env.behavior_specs:
            actions = np.resize(actions,
                                (len(env.get_steps(brain)[0]), len(env.behavior_specs[brain].discrete_action_branches)))
            self.env.set_actions(brain, actions)
            self.env.step()
            decision_steps, terminal_steps = env.get_steps(brain)

            for agent_id_decisions in decision_steps:
                agents_transitions[agent_id_decisions] = [decision_steps[agent_id_decisions].obs,
                                                          decision_steps[agent_id_decisions].reward, False]

            for agent_id_terminated in terminal_steps:
                agents_transitions[agent_id_terminated] = [terminal_steps[agent_id_terminated].obs,
                                                           terminal_steps[agent_id_terminated].reward, True]

        return agents_transitions

    def calc_q_loss(self, q1_out, q2_out, target_values, done, rewards):
        return -1, -1
        for rew in enumerate(rewards):
            q_backup.apppend()
        q_1_loss = 0.5 * torch.mean(torch.nn.functional.mse_loss(q_backup, q1_out))
        q_2_loss = 0.5 * torch.mean(torch.nn.functional.mse_loss(q_backup, q2_out))

        return q_1_loss, q_2_loss

    def calc_value_loss(self, log_probs, values, q1p_out, q2p_out, ):
        with torch.no_grad():
            _ent_coeff = torch.exp(self.networks.log_entropy_coeff)

    def calc_policy_loss(self, log_probs, q1p_out, action_space):
        return -1
        _ent_coeff = torch.exp(self.networks.log_entropy_coeff)
        mean_q1 = torch.mean(torch.stack(list(q1p_out)))

        action_probs = torch.exp(log_probs)
        branched_per_act_entropy = break_into_branches(log_probs * action_probs, action_space)
        branched_q = break_into_branches(mean_q1 * action_probs)

    def calc_entropy_loss(self, log_probs, action_space):
        with torch.no_grad():
            branched_per_act_entropy = break_into_branches(log_probs * log_probs.exp(), action_space)

            target_current_diff_branched = torch.stack([
                torch.sum(_lp, axis=1, keepdim=True) + _te
                for _lp, _te in zip(branched_per_act_entropy, self.networks.target_entropy)
            ], axis=1)

            target_current_diff = torch.squeeze(target_current_diff_branched, axis=2)
            print(target_current_diff)
            print(self.networks.log_entropy_coeff)
            entropy_loss = -1 * torch.mean(self.networks.log_entropy_coeff * target_current_diff, axis=1)
        print("Entropy Loss: ")
        print(entropy_loss)
        entropy_loss.backward()
        return entropy_loss


    def update(self, batch: Buffer):
        observations = torch_from_np(batch.observations, self.device)
        next_observations = torch_from_np(batch.next_observations, self.device)
        rewards = torch_from_np(batch.rewards, self.device)
        dones = torch_from_np(batch.done, self.device)

        dists = self.policy.actor.forward(observations)
        sampled_values = self.policy.critic.forward(observations)
        actions = []
        for dist in dists:
            action = dist.sample()
            actions.append(action)
        actions = torch.stack(actions).transpose(0, 1) # From 3x batch_size to batch_size x 3 (2,2,0),(1,2,0),....
        one_hot_actions = actions_to_onehot(actions, batch.action_space) # List with every element referring to a branch
        # For every branch-> action gets one hot encoded
        # Example -> (2,1,2) gets [tensor(0,0,1),tensor(0,1,0),tensor(0,0,1)]

        _, entropies, all_log_probs = self.policy.get_probs_and_entropies(actions, dists)
        with torch.no_grad():
            q1_policy_out, q2_policy_out = self.networks.value_network.forward(observations)

        q1_out, q2_out = self.networks.value_network.forward(observations)

        with torch.no_grad():
            target_values = self.networks.target_network.forward(next_observations)

        branched_q1 = break_into_branches(q1_out, batch.action_space)
        branched_q2 = break_into_branches(q2_out, batch.action_space)

        condensed_q1 = torch.stack([torch.sum(act_branch*q_branch, dim=1, keepdim=True) for act_branch, q_branch in zip(one_hot_actions, branched_q1)])
        condensed_q2 = torch.stack([torch.sum(act_branch*q_branch, dim=1, keepdim=True) for act_branch, q_branch in zip(one_hot_actions, branched_q2)])

        q1_loss, q2_loss = self.calc_q_loss(condensed_q1, condensed_q2, target_values, dones, rewards)
        value_loss = self.calc_value_loss(all_log_probs, sampled_values, q1_policy_out, q2_policy_out)

        policy_loss = self.calc_policy_loss(all_log_probs, q1_policy_out, batch.action_space)

        entropy_loss = self.calc_entropy_loss(all_log_probs, batch.action_space)
        # total_value_loss = q1_loss + q2_loss + value_loss

        # Update the target network

        tau = 0.3
        self.networks.soft_update(self.policy.critic, self.networks.target_network, tau)

    def generate_trajectories(self, buffer, max_buffer_size, time_horizon=256) -> {}:
        env = self.env

        # Create transition dict
        transitions = {}
        rewards = []
        trajectory_lengths = [] # length of agent trajectories, maxlen = timehorizon/done
        episode_lengths = []

        appended_steps = 0

        first_iteration = True
        buffer_finished = False

        while not buffer_finished:
            if first_iteration:
                # This part is to set the inital actions for the agents
                for brain in env.behavior_specs:
                    decision_steps, terminal_steps = env.get_steps(brain)
                    num_agents = len(decision_steps)
                    if(num_agents == 0):
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
                            'act_log_prob_buf': np.zeros([time_horizon, len(self.action_space)],
                                                         dtype=np.float32),
                            'entropies_buf': np.zeros([time_horizon, len(self.action_space)],
                                                         dtype=np.float32),
                            'values_buf': np.zeros([time_horizon], dtype=np.float32),
                            'rews_buf': np.zeros([time_horizon], dtype=np.float32),
                            'done_buf': np.zeros([time_horizon], dtype=np.float32)}

                    for agent_id_decisions in decision_steps:
                        init_state = decision_steps[agent_id_decisions].obs
                        init_state = flatten(init_state)
                        with torch.no_grad():
                            init_state = np.array(init_state)
                            dists = self.policy.actor.forward(init_state)
                            value = self.policy.critic.forward(init_state)
                            action = [dist.sample() for dist in dists]
                            entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                            action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in zip(dists, action)]
                            action =[action_branch.detach().cpu().numpy() for action_branch in action]
                            actions[agent_id_decisions] = action
                            value = value.detach().cpu().numpy()

                        transitions[agent_id_decisions]['obs_buf'][agent_ptr[agent_id_decisions]] = init_state
                        transitions[agent_id_decisions]['acts_buf'][agent_ptr[agent_id_decisions]] = action
                        transitions[agent_id_decisions]['act_log_prob_buf'][agent_ptr[agent_id_decisions]] = action_log_probs
                        transitions[agent_id_decisions]['values_buf'][agent_ptr[agent_id_decisions]] = value
                        transitions[agent_id_decisions]['entropies_buf'][agent_ptr[agent_id_decisions]] = entropies

                    for agent_id_terminated in terminal_steps:
                        init_state = terminal_steps[agent_id_terminated].obs
                        init_state = flatten(init_state)
                        with torch.no_grad():
                            init_state = np.array(init_state)
                            dists = self.policy.actor.forward(init_state)
                            value = self.policy.critic.forward(init_state)
                            entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                            action = [dist.sample() for dist in dists]
                            action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in
                                                zip(dists, action)]
                            action = [action_branch.detach().cpu().numpy() for action_branch in action]
                            value = value.detach().cpu().numpy()

                        actions[agent_id_terminated] = action
                        transitions[agent_id_terminated]['obs_buf'][agent_ptr[agent_id_terminated]] = init_state
                        transitions[agent_id_terminated]['acts_buf'][agent_ptr[agent_id_terminated]] = action
                        transitions[agent_id_terminated]['values_buf'][agent_ptr[agent_id_terminated]] = value
                        transitions[agent_id_terminated]['act_log_prob_buf'][agent_ptr[agent_id_terminated]] = action_log_probs
                        transitions[agent_id_terminated]['entropies_buf'][agent_ptr[agent_id_terminated]] = entropies

            # Step environment
            next_experiences = self.step_env(self.env, np.array(actions))
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
                    # If the corresponding agent is done or trajectory is max length
                    with torch.no_grad():
                        next_obs_tmp = np.array(next_obs)
                        dists = self.policy.actor.forward(next_obs_tmp)
                        value = self.policy.critic.forward(next_obs_tmp)

                        action = [dist.sample() for dist in dists]
                        entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                        action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in
                                            zip(dists, action)]
                        next_action = [action_branch.detach().cpu().numpy() for action_branch in action]
                        actions[agent_id_decisions] = action
                        value = value.detach().cpu().numpy()

                    transitions[agent_id]['act_log_prob_buf'][agent_ptr[agent_id]] = action_log_probs
                    transitions[agent_id]['values_buf'][agent_ptr[agent_id]] = value
                    transitions[agent_id]['entropies_buf'][agent_ptr[agent_id]] = entropies

                    actions[agent_id] = next_action

                    if done:
                        episode_lengths.append(agent_current_step[agent_id])
                        agent_current_step[agent_id] = 0
                        rewards.append(current_added_reward[agent_id])
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
                            buffer.next_observations = np.vstack((buffer.next_observations, transitions[agent_id]['n_obs_buf'][i]))
                            buffer.done = np.append(buffer.done, transitions[agent_id]['done_buf'][i])


                    if(len(buffer) >= max_buffer_size):
                        length_counter = len(buffer) - max_buffer_size
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
                    transitions[agent_id]['act_log_prob_buf'] = np.zeros(
                        [time_horizon, len(self.action_space)],
                        dtype=np.float32)
                    transitions[agent_id]['entropies_buf'] = np.zeros(
                        [time_horizon, len(self.action_space)],
                        dtype=np.float32)


                    agent_ptr[agent_id] = 0

                else:  # If the corresponding agent is not done, continue
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs

                    with torch.no_grad():
                        obs = np.array(next_obs)
                        dists = self.policy.actor.forward(obs)
                        value = self.policy.critic.forward(obs)
                        action = [dist.sample() for dist in dists]
                        entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                        action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in
                                            zip(dists, action)]
                        next_action = [action_branch.detach().cpu().numpy() for action_branch in action]
                        actions[agent_id_decisions] = action
                        value = value.detach().cpu().numpy()

                    transitions[agent_id]['entropies_buf'][agent_ptr[agent_id]] = entropies
                    transitions[agent_id]['acts_buf'][agent_ptr[agent_id]] = next_action
                    transitions[agent_id]['values_buf'][agent_ptr[agent_id]] = value
                    transitions[agent_id]['act_log_prob_buf'][agent_ptr[agent_id]] = action_log_probs
                    actions[agent_id] = next_action

            first_iteration = False
        print("Current mean Reward: " + str(np.mean(rewards)))
        print("Current mean Episode Length: " + str(np.mean(episode_lengths)))

        self.writer.add_scalar('Task: ' + str(self.task) + '/Cumulative Reward', np.mean(rewards), self.meta_step)

        self.writer.add_scalar('Task: ' + str(self.task) + '/Mean Episode Length', np.mean(episode_lengths),
                               self.meta_step)

        return buffer, appended_steps


writer = SummaryWriter("C:/Users/Sebastian/Desktop/RLUnity/Training/results" + r"/sac_test_2")

sac_module = SAC_Meta_Learner('cpu', writer=writer)

engine_configuration_channel = EngineConfigurationChannel()
engine_configuration_channel.set_configuration_parameters(time_scale=10.0)

env_parameters_channel = EnvironmentParametersChannel()
env_parameters_channel.set_float_parameter("seed", 5.0)
env = UnityEnvironment(file_name="Training/Maze",
                       base_port=5000, timeout_wait=120,
                       no_graphics=False, seed=0, side_channels=[engine_configuration_channel, env_parameters_channel])

sac_module.set_environment(env)
sac_module.init_network_and_optim()
steps = 0
replay_buffer = Buffer(action_space=sac_module.action_space)

while steps < 1000000:
    buffer, steps_taken = sac_module.generate_trajectories(replay_buffer, 2000)
    print("Steps taken since last update: {}".format(steps_taken))
    update_steps = 0
    while update_steps < steps_taken:
        batch = buffer.sample_batch(batch_size=128)
        sac_module.update(batch)
        update_steps += 1

    sac_module.meta_step += 1
    sac_module.learning_rate_scheduler.step()


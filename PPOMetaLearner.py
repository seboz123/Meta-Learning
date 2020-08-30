import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter
from ReplayBuffer import ReplayBuffer

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

flatten = lambda l: [item for sublist in l for item in sublist]


class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, action_dim, hidden_size: int = 256):
        super(ActorCritic, self).__init__()

        self.device = device
        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.out_layers = [nn.Sequential(nn.Linear(hidden_size, shape).to(self.device), nn.Softmax(dim=-1)) for shape in
                           action_dim]

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def act(self, state: np.ndarray) -> (Categorical, torch.Tensor):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            action_hidden_state = self.action_layer(state)
            action_prob = torch.stack([layer(action_hidden_state) for layer in self.out_layers])
            dist = Categorical(action_prob)
            value = self.value_layer(state)
        return dist, value


class PPO_Meta_Learner:
    """DQN Agent interacting with environment.

    Attribute:
        env (UnityEnvironment.env): UnityEnvironment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
            self,
            device: torch.device,
            gamma: float = 0.99,
            update_period: int = 5000,
            writer: SummaryWriter = None,
            decision_requester: int = 5,
    ):
        self.device = device
        print("Using: " + str(self.device))
        self.writer = writer
        self.meta_step = 0

    def set_environment(self, env):
        env.reset()
        self.env = env
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

    def get_gae(self, rewards, value_estimates, value_next=0.0, gamma=0.99, lambd=0.95):
        """
        Computes generalized advantage estimate for use in updating policy.
        :param rewards: list of rewards for time-steps t to T.
        :param value_next: Value estimate for time-step T+1.
        :param value_estimates: list of value estimates for time-steps t to T.
        :param gamma: Discount factor.
        :param lambd: GAE weighing factor.
        :return: list of advantage estimates for time-steps t to T.
        """
        value_estimates = np.append(value_estimates, value_next)
        delta_t = rewards + gamma * value_estimates[1:] - value_estimates[:-1]
        advantage = self.discount_rewards(r=delta_t, gamma=gamma * lambd)
        return advantage

    def calc_loss(self, buffer: {}, batch_size: int = 256, old_log_probs=None) -> torch.Tensor:
        for element in buffer:
            element.co
        print(buffer)
        create_batches()
        advantages_list = []
        returns_list = []
        epsilon = 0.2
        n_old_log_probs = None
        for i in range(len(buffer['rews'])):
            advantage = self.get_gae(buffer['rews'][i], buffer['values'][i])
            returns = np.add(advantage, buffer['values'][i])
            normalized_return = [(element - returns.mean()) / (returns.std() + 0.0000001) for element in returns]

            returns_list.append(normalized_return)
            advantages_list.append(advantage)
            print(buffer['obs'][i])
            log_probs, action = self.actor_critic.act(buffer['obs'][i][0])
            print(action)
            n_log_probs = - log_probs.log_prob(action)

            if n_old_log_probs is not None:
                ratio = torch.exp(n_old_log_probs - n_log_probs)
                print(ratio)
                pg_loss_clipped = - advantage * ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
                pg_loss_unclipped = ratio * advantage
                policy_loss = - torch.min(pg_loss_unclipped, pg_loss_clipped)

            n_old_log_probs = n_log_probs

        loss = policy_loss + 0.5 * value_loss - decay_beta * test

        return loss, elementwise_loss, indices

    def init_network_and_optim(self, obs_dim, action_dim, hidden_size, learning_rate: float = 0.0003):

        self.actor_critic = ActorCritic(self.device,
                                        obs_dim, action_dim, hidden_size
                                        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

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

    def generate_and_fill_buffer(self, buffer_size, task, time_horizon=256) -> {}:
        env = self.env
        env.reset()

        # Create transition dict
        experiences = {}
        transitions = {}
        rewards = []
        trajectory_lengths = []
        episode_lengths = []

        buffer_length = 0
        first_iteration = True
        buffer_finished = False

        while not buffer_finished:
            if first_iteration:
                # This part is to set the inital actions for the agents
                for brain in env.behavior_specs:
                    decision_steps, terminal_steps = env.get_steps(brain)
                    num_agents = len(decision_steps)
                    episode_step = [1 for _ in range(num_agents)]

                    actions = np.zeros((24, len(self.action_space)))
                    print("Brain :" + str(brain) + " with " + str(num_agents) + " agents detected.")
                    agent_ptr = [0 for _ in range(num_agents)]  # Create pointer for agent transitions

                    for i in range(num_agents):
                        experiences[i] = [[], [], 0, [], False]
                        transitions[i] = {
                            'obs_buf': np.zeros([time_horizon, self.obs_space], dtype=np.float32),
                            'n_obs_buf': np.zeros([time_horizon, self.obs_space], dtype=np.float32),
                            'acts_buf': np.zeros([time_horizon, len(self.action_space)], dtype=np.float32),
                            'act_log_prob_buf': np.zeros([time_horizon, len(self.action_space)],
                                                         dtype=np.float32),
                            'values_buf': np.zeros([time_horizon], dtype=np.float32),
                            'rews_buf': np.zeros([time_horizon], dtype=np.float32),
                            'done_buf': np.zeros([time_horizon], dtype=np.float32)}

                    for agent_id_decisions in decision_steps:
                        init_state = decision_steps[agent_id_decisions].obs
                        init_state = flatten(init_state)
                        dist, value = self.actor_critic.act(np.array(init_state))

                        action = dist.sample()
                        action_log_prob = dist.log_prob(action)
                        action = action.detach().cpu().numpy()
                        actions[agent_id_decisions] = action

                        experiences[agent_id_decisions][0] = init_state
                        transitions[agent_id_decisions]['obs_buf'][agent_ptr[agent_id_decisions]] = init_state
                        transitions[agent_id_decisions]['acts_buf'][agent_ptr[agent_id_decisions]] = action
                        transitions[agent_id_decisions]['act_log_prob_buf'][agent_ptr[agent_id_decisions]] = action_log_prob.detach().cpu().numpy()
                        transitions[agent_id_decisions]['values_buf'][agent_ptr[agent_id_decisions]] = value.detach().cpu().numpy()

                    for agent_id_terminated in terminal_steps:
                        init_state = terminal_steps[agent_ptr[agent_id_terminated]].obs
                        init_state = flatten(init_state)
                        dist, value = self.actor_critic.act(np.array(init_state))

                        action = dist.sample()
                        action_log_prob = dist.log_prob(action)
                        action = action.detach().cpu().numpy()

                        actions[agent_id_terminated] = action
                        experiences[agent_id_terminated][0] = init_state
                        transitions[agent_id_terminated]['obs_buf'][agent_ptr[agent_id_terminated]] = init_state
                        transitions[agent_id_terminated]['acts_buf'][agent_ptr[agent_id_terminated]] = action
                        transitions[agent_id_terminated]['values_buf'][agent_ptr[agent_id_terminated]] = value.detach().cpu().numpy()
                        transitions[agent_id_terminated]['act_log_prob_buf'][0] = action_log_prob.detach().cpu().numpy()

                # Create the buffers
                finished_obs_buf = np.zeros([buffer_size, self.obs_space], dtype=np.float32)
                finished_next_obs_buf = np.zeros([buffer_size, self.obs_space], dtype=np.float32)
                finished_acts_buf = np.zeros([buffer_size, len(self.action_space)], dtype=np.float32)
                finished_log_probs_buf = np.zeros([buffer_size, len(self.action_space)], dtype=np.float32)
                finished_rews_buf = np.zeros([buffer_size], dtype=np.float32)
                finished_values_buf = np.zeros([buffer_size], dtype=np.float32)
                finished_done_buf = np.zeros([buffer_size], dtype=np.float32)
            else:
                for agent in experiences:
                    # Set Observation and Action to the last next_obs and the selected action for the observation
                    experiences[agent][0] = experiences[agent][3]  # obs = next_obs
                    experiences[agent][1] = actions[int(agent)]  # action = action_vector[]

            # Step environment
            next_experiences = self.step_env(self.env, np.array(actions))
            # Create action vector to store actions
            actions = np.zeros((num_agents, len(self.action_space)))

            for agent_id in next_experiences:

                reward = next_experiences[agent_id][1]  # Reward
                next_obs = flatten(next_experiences[agent_id][0])  # Next_obs
                done = next_experiences[agent_id][2]  # Done

                # Store 1-step-experience of every agent_id {agent_id0:[obs,act,rew,n_obs,done] agent_id1: ....}
                experiences[agent_id][2] = reward
                experiences[agent_id][3] = next_obs
                experiences[agent_id][4] = done
                # Store trajectory of every Agent {Agent0:[obs,act,rew,n_obs,done,obs,act,rew,n_obs,done,.... Agent1: ....}
                transitions[agent_id]['rews_buf'][agent_ptr[agent_id]] = reward
                transitions[agent_id]['n_obs_buf'][agent_ptr[agent_id]] = next_obs
                transitions[agent_id]['done_buf'][agent_ptr[agent_id]] = done

                if done or agent_ptr[agent_id] == time_horizon - 1:
                    # If the corresponding agent is done or trajectory is max length
                    if done:
                        episode_lengths.append(episode_step[agent_id])
                        episode_step[agent_id] = 0

                    dist, value = self.actor_critic.act(np.array(next_obs))

                    next_action = dist.sample()
                    action_log_prob = dist.log_prob(next_action)
                    next_action = next_action.detach().cpu().numpy()

                    transitions[agent_id]['act_log_prob_buf'][
                        agent_ptr[agent_id]] = action_log_prob.detach().cpu().numpy()
                    transitions[agent_id]['values_buf'][agent_ptr[agent_id]] = value.detach().cpu().numpy()

                    actions[agent_id] = next_action

                    if agent_ptr[agent_id] + buffer_length >= buffer_size:
                        buffer_finished = True
                        break
                    trajectory_lengths.append(agent_ptr[agent_id] + 1)

                    advantages = self.get_gae(transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1],
                                              transitions[agent_id]['values_buf'][:agent_ptr[agent_id] + 1])

                    for i in range(agent_ptr[agent_id] + 1):  # For every experiences store it in buffer
                        finished_obs_buf[i + buffer_length] = transitions[agent_id]['obs_buf'][i]
                        finished_acts_buf[i + buffer_length] = transitions[agent_id]['acts_buf'][i]
                        finished_rews_buf[i + buffer_length] = transitions[agent_id]['rews_buf'][i]
                        finished_values_buf[i + buffer_length] = transitions[agent_id]['values_buf'][i]

                        finished_next_obs_buf[i + buffer_length] = transitions[agent_id]['n_obs_buf'][i]
                        finished_done_buf[i + buffer_length] = transitions[agent_id]['done_buf'][i]
                        finished_log_probs_buf[i + buffer_length] = transitions[agent_id]['act_log_prob_buf'][i]

                    buffer_length += agent_ptr[agent_id] + 1

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
                    agent_ptr[agent_id] = 0

                else:  # If the corresponding agent is not done, continue
                    episode_step[agent_id] += 1
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs

                    dist, value = self.actor_critic.act(np.array(next_obs))

                    next_action = dist.sample()
                    action_log_prob = dist.log_prob(next_action)
                    next_action = next_action.detach().cpu().numpy()

                    transitions[agent_id]['acts_buf'][agent_ptr[agent_id]] = next_action
                    transitions[agent_id]['values_buf'][agent_ptr[agent_id]] = value.detach().cpu().numpy()

                    transitions[agent_id]['act_log_prob_buf'][
                        agent_ptr[agent_id]] = action_log_prob.detach().cpu().numpy()
                    actions[agent_id] = next_action

            first_iteration = False
        self.writer.add_scalar('Task: ' + str(task) + '/Cumulative Reward', np.mean(rewards), self.meta_step)
        self.writer.add_scalar('Task: ' + str(task) + '/Mean Episode Length', np.mean(episode_lengths),
                               self.meta_step)

        print(episode_lengths, trajectory_lengths, rewards)

        return {'obs': finished_obs_buf[:buffer_length], 'acts': finished_acts_buf[:buffer_length],
                'rews': finished_rews_buf[:buffer_length],
                'n_obs': finished_next_obs_buf[:buffer_length], 'dones': finished_done_buf[:buffer_length],
                'log_probs': finished_log_probs_buf[:buffer_length],
                'values': finished_values_buf[:buffer_length], 'trajectory_lengths': trajectory_lengths}


writer = SummaryWriter("C:/Users/Sebastian/Desktop/RLUnity/Training/results" + r"/Meta_Learning3")

ppo_module = PPO_Meta_Learner('cuda', writer=writer)

engine_configuration_channel = EngineConfigurationChannel()
engine_configuration_channel.set_configuration_parameters(time_scale=10.0)

env_parameters_channel = EnvironmentParametersChannel()
env_parameters_channel.set_float_parameter("seed", 5.0)
env = UnityEnvironment(file_name="C:/Users/Sebastian/Desktop/RLUnity/Training/mMaze/RLProject",
                       base_port=5000, timeout_wait=120,
                       no_graphics=False, seed=0, side_channels=[engine_configuration_channel, env_parameters_channel])

ppo_module.set_environment(env)
ppo_module.init_network_and_optim(ppo_module.obs_space, ppo_module.action_space, 3)
np.set_printoptions(suppress=True, threshold=np.inf)
buffer = ppo_module.generate_and_fill_buffer(4000, 0, 512)
np.set_printoptions(suppress=True, threshold=np.inf)
ppo_module.calc_loss(buffer)

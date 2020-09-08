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

from utils import torch_from_np

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

    def get_dist_and_value(self, obs: torch.FloatTensor) -> (torch.distributions.Categorical, torch.FloatTensor):
        """

        Args:
            obs: Observations
        Returns: dists, values

        """
        action_hidden_state = self.action_layer(obs)
        action_prob = []
        dists = []
        for layer in self.out_layers:
            action_out = layer(action_hidden_state)
            action_prob.append(action_out)
            dist = Categorical(action_out)
            dists.append(dist)

        values = self.value_layer(obs)
        values = values.squeeze()

        return dists, values

    def forward(self, obs: torch.FloatTensor) -> (torch.distributions.Categorical, torch.FloatTensor):
        obs = torch_from_np(obs)
        return self.get_dist_and_value(obs)


    def get_probs_and_entropies(self, acts: torch.FloatTensor, dists: torch.distributions.Categorical):
        log_probs = torch.zeros([acts.shape[0]]).to(self.device)
        entropies = torch.zeros([acts.shape[0]]).to(self.device)
        for i, dist in enumerate(dists):
            log_probs = torch.add(log_probs, dist.log_prob(acts[:, i]))
            entropies = torch.add(entropies, dist.entropy())
        return log_probs, entropies


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
        self.actor_critic: ActorCritic

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

    def evaluate_actions(self, obs: np.ndarray, acts: np.ndarray) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):
        """
        Args:
            obs:
            acts:

        Returns:
        """
        obs = torch_from_np(obs)
        acts = torch_from_np(acts)
        dists, values = self.actor_critic.get_dist_and_value(obs)
        log_probs, entropies = self.actor_critic.get_probs_and_entropies(acts, dists) # Error here
        return log_probs, entropies, values

    def calc_value_loss(self, values: torch.FloatTensor, old_values: torch.FloatTensor, returns: torch.FloatTensor, eps) -> torch.FloatTensor:
        clipped_value_est = old_values + torch.clamp(values - old_values, -1 * eps, eps)
        tmp_loss_1 = torch.pow(returns - values, 2)
        tmp_loss_2 = torch.pow(returns - clipped_value_est, 2)
        value_loss = torch.max(tmp_loss_1, tmp_loss_2)
        value_loss = 0.5 * torch.mean(value_loss)
        return value_loss

    def calc_policy_loss(self, advantages: torch.FloatTensor, log_probs: torch.FloatTensor, old_log_probs: torch.FloatTensor, eps: float = 0.2  ) -> torch.FloatTensor:
        ratio = torch.exp(log_probs - old_log_probs)
        tmp_loss_1 = ratio * advantages
        tmp_loss_2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        policy_loss = - (torch.min(tmp_loss_1, tmp_loss_2)).mean()
        return policy_loss


    def calc_loss(self, buffer: {}, epsilon: float = 0.2, beta: float = 0.001, lambd = 0.99) -> torch.Tensor:

        old_log_probs = np.sum(buffer['log_probs'], axis=1) # Multiply all probs is adding all log_probs
        obs = buffer['obs']
        acts = buffer['acts']
        values = buffer['values']
        rews = buffer['rews']

        advantages = self.get_gae(rews, values, lambd)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 0.000001)

        returns = np.add(normalized_advantages, values)

        old_values = torch_from_np(values)
        old_log_probs = torch_from_np(old_log_probs)
        returns = torch_from_np(returns)
        normalized_advantages = torch_from_np(normalized_advantages)

        log_probs, entropies, values = self.evaluate_actions(obs, acts)
        value_loss = self.calc_value_loss(values, old_values, returns, eps=epsilon)
        policy_loss = self.calc_policy_loss(normalized_advantages, log_probs, old_log_probs, eps=epsilon)
        entropy = entropies.mean()

        kullback_leibler = 0.5 * torch.mean(torch.pow(log_probs-old_log_probs, 2))
        policy_loss = policy_loss
        value_loss = 0.5 * value_loss
        entropy_loss = - beta * entropy

        # self.writer.add_scalar('Task: ' + str(self.task) + '/Entropy ', entropy.item(), self.meta_step)
        # self.writer.add_scalar('Task: ' + str(self.task) + '/Policy Loss ', policy_loss.item(), self.meta_step)
        # self.writer.add_scalar('Task: ' + str(self.task) + '/Value Loss ', value_loss.item(), self.meta_step)
        # self.writer.add_scalar('Task: ' + str(self.task) + '/Approx Kullback-Leibler ', kullback_leibler.item(), self.meta_step)

        # print("Entropy: {:.3f}\nLosses:  Policy: {:.3f}, Value: {:.3f}, Approx KL: {:.3f}".format(entropy.item(), policy_loss.item(), value_loss.item(), kullback_leibler.item()))

        loss = policy_loss + value_loss + entropy_loss

        return loss, policy_loss.item(), value_loss.item(), entropy.item(), kullback_leibler.item()

    def init_network_and_optim(self, hidden_size: int = 256, learning_rate: float = 0.0003):
        obs_dim = self.obs_space
        action_dim = self.action_space
        print("Obs space detected as: " + str(self.obs_space))
        print("Action space detected as: " + str(self.action_space))
        self.actor_critic = ActorCritic(self.device,
                                        obs_dim, action_dim, hidden_size
                                        ).to(self.device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        lambda2 = lambda epoch: 0.95**epoch
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda2)


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

    def split_buffer_into_batches(self, buffer, batch_size: int = 512):
        batches = []
        buffer_length = len(buffer['rews'])
        indx = np.arange(buffer_length)
        np.random.shuffle(indx)
        for key in buffer:
            buffer[key] = buffer[key][indx]

        for size in range(0, buffer_length, batch_size):
            batch = {}
            if size+batch_size <= buffer_length:
                for key in buffer:
                    batch[key] = buffer[key][size:size+batch_size]
            else:
                for key in buffer:
                    batch[key] = buffer[key][size:]

            batches.append(batch)
        return batches

    def generate_and_fill_buffer(self, buffer_size, time_horizon=256) -> {}:
        start_time = time.time()
        env = self.env
        env.reset()

        # Create transition dict
        transitions = {}
        rewards = []
        trajectory_lengths = [] # length of agent trajectories, maxlen = timehorizon/done
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
                    episode_step = [0 for _ in range(num_agents)]
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
                            dists, value = self.actor_critic.forward(np.array(init_state))
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
                            dists, value = self.actor_critic.forward(np.array(init_state))
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

                # Create the buffers
                finished_obs_buf = np.zeros([buffer_size, self.obs_space], dtype=np.float32)
                finished_next_obs_buf = np.zeros([buffer_size, self.obs_space], dtype=np.float32)
                finished_acts_buf = np.zeros([buffer_size, len(self.action_space)], dtype=np.float32)
                finished_log_probs_buf = np.zeros([buffer_size, len(self.action_space)], dtype=np.float32)
                finished_rews_buf = np.zeros([buffer_size], dtype=np.float32)
                finished_values_buf = np.zeros([buffer_size], dtype=np.float32)
                finished_done_buf = np.zeros([buffer_size], dtype=np.float32)
                finished_adv_buf = np.zeros([buffer_size], dtype=np.float32)
                finished_entropies_buf = np.zeros([buffer_size, len(self.action_space)], dtype=np.float32)


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

                episode_step[agent_id] += 1
                if done or agent_ptr[agent_id] == time_horizon - 1:
                    # If the corresponding agent is done or trajectory is max length
                    with torch.no_grad():
                        dists, value = self.actor_critic.forward(np.array(next_obs))
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

                    if agent_ptr[agent_id] + buffer_length >= buffer_size:
                        buffer_finished = True
                        break
                    if done:
                        episode_lengths.append(episode_step[agent_id])
                        episode_step[agent_id] = 0
                        rewards.append(current_added_reward[agent_id])
                        current_added_reward[agent_id] = 0

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
                        finished_adv_buf[i + buffer_length] = advantages[i]
                        finished_entropies_buf[i + buffer_length] = transitions[agent_id]['entropies_buf'][i]

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
                    transitions[agent_id]['entropies_buf'] = np.zeros(
                        [time_horizon, len(self.action_space)],
                        dtype=np.float32)
                    agent_ptr[agent_id] = 0

                else:  # If the corresponding agent is not done, continue
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs

                    with torch.no_grad():
                        dists, value = self.actor_critic.forward(np.array(next_obs))
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

        print("Generated buffer of lenth: {} in {:.3f} secs.".format(buffer_length, time.time()-start_time))

        return {'obs': finished_obs_buf[:buffer_length], 'acts': finished_acts_buf[:buffer_length],
                'rews': finished_rews_buf[:buffer_length], 'advs': finished_adv_buf[:buffer_length],
                'n_obs': finished_next_obs_buf[:buffer_length], 'dones': finished_done_buf[:buffer_length],
                'log_probs': finished_log_probs_buf[:buffer_length], 'entropies': finished_entropies_buf[:buffer_length],
                'values': finished_values_buf[:buffer_length]}, buffer_length

    def train(self, env: UnityEnvironment, max_steps: int = 1000000, buffer_size: int = 10000, time_horizon: int = 512, batch_size: int = 512):
        step = 0
        self.set_environment(env)
        self.init_network_and_optim()
        while step < max_steps:
            print("Current step: " + str(step))
            buffer, buffer_length = self.generate_and_fill_buffer(buffer_size=buffer_size, time_horizon=time_horizon)
            step += buffer_length
            epochs = 3
            p_losses , v_losses, entropies, kls = [], [], [], []
            for i in range(epochs):
                batches = self.split_buffer_into_batches(buffer, batch_size=batch_size)
                for batch in batches:
                    loss, p_loss, v_loss, entropy, kl = self.calc_loss(batch, epsilon=0.2, beta=0.001, lambd=0.99)
                    p_losses.append(p_loss)
                    v_losses.append(v_loss)
                    entropies.append(entropy)
                    kls.append(kl)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            for parameter_group in ppo_module.optimizer.param_groups:
                print("Current learning rate: {}".format(parameter_group['lr']))

            self.meta_step += 1
            self.learning_rate_scheduler.step()
            self.writer.add_scalar('Task: ' + str(self.task) + '/Entropy ', np.mean(entropies), self.meta_step)
            self.writer.add_scalar('Task: ' + str(self.task) + '/Policy Loss ', np.mean(p_losses), self.meta_step)
            self.writer.add_scalar('Task: ' + str(self.task) + '/Value Loss ', np.mean(v_losses), self.meta_step)
            self.writer.add_scalar('Task: ' + str(self.task) + '/Approx Kullback-Leibler ', np.mean(kls), self.meta_step)


writer = SummaryWriter("results")

ppo_module = PPO_Meta_Learner('cpu', writer=writer)

engine_configuration_channel = EngineConfigurationChannel()
engine_configuration_channel.set_configuration_parameters(time_scale=10.0)

env_parameters_channel = EnvironmentParametersChannel()
env_parameters_channel.set_float_parameter("seed", 5.0)
env = UnityEnvironment(file_name="Training/Maze",
                       base_port=5000, timeout_wait=120,
                       no_graphics=False, seed=0, side_channels=[engine_configuration_channel, env_parameters_channel])

ppo_module.train(env, 1000000, buffer_size=3000)


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

from utils import torch_from_np, get_probs_and_entropies
from buffers import PPOBuffer
from models import ActorCriticPolicy
from env_utils import step_env

flatten = lambda l: [item for sublist in l for item in sublist]


class PPO_Meta_Learner:
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

    def init_networks_and_optimizers(self, hidden_size: int, num_hidden_layers: int, max_scheduler_steps: int,
                                     learning_rate: float):
        obs_dim = self.obs_space
        action_dim = self.action_space
        self.policy = ActorCriticPolicy(obs_dim, action_dim, hidden_size, num_hidden_layers, True).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        linear_schedule = lambda epoch: max((1 - epoch / max_scheduler_steps), 1e-6)

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_schedule)

    def get_state_dict(self):
        return self.policy.state_dict()


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

    def evaluate_actions(self, obs: np.ndarray, acts: np.ndarray) -> (
            torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):
        obs = torch_from_np(obs)
        acts = torch_from_np(acts)
        dists, values = self.policy.policy(obs)
        cumulated_log_probs, entropies, all_log_probs = get_probs_and_entropies(acts, dists, self.device)  # Error here
        return cumulated_log_probs, entropies, values

    def calc_value_loss(self, values: torch.FloatTensor, old_values: torch.FloatTensor, returns: torch.FloatTensor,
                        eps) -> torch.FloatTensor:
        clipped_value_est = old_values + torch.clamp(values - old_values, -1 * eps, eps)
        tmp_loss_1 = torch.pow(returns - values, 2)
        tmp_loss_2 = torch.pow(returns - clipped_value_est, 2)
        value_loss = torch.max(tmp_loss_1, tmp_loss_2)
        value_loss = 0.5 * torch.mean(value_loss)
        return value_loss

    def calc_policy_loss(self, advantages: torch.FloatTensor, log_probs: torch.FloatTensor,
                         old_log_probs: torch.FloatTensor, eps: float = 0.2) -> torch.FloatTensor:
        ratio = torch.exp(log_probs - old_log_probs)
        tmp_loss_1 = ratio * advantages
        tmp_loss_2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        policy_loss = - (torch.min(tmp_loss_1, tmp_loss_2)).mean()
        return policy_loss

    def calc_loss(self, buffer: {}, gamma: float, epsilon, beta, lambd) -> torch.Tensor:

        old_log_probs = np.sum(buffer.act_log_probs, axis=1)  # Multiply all probs is adding all log_probs
        obs = buffer.observations
        acts = buffer.actions
        values = buffer.values
        rews = buffer.rewards

        advantages = self.get_gae(rews, values, gamma, lambd)
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

        kullback_leibler = 0.5 * torch.mean(torch.pow(log_probs - old_log_probs, 2))
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

    def generate_and_fill_buffer(self, buffer: PPOBuffer, time_horizon=256) -> {}:
        start_time = time.time()
        env = self.env
        env.reset()

        # Create transition dict
        transitions = {}
        rewards = []
        trajectory_lengths = []  # length of agent trajectories, maxlen = timehorizon/done
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
                            init_tensor = torch_from_np(np.array(init_state), self.device)
                            dists, value = self.policy.policy(init_tensor)
                            action = [dist.sample() for dist in dists]
                            entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                            action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in
                                                zip(dists, action)]
                            action = [action_branch.detach().cpu().numpy() for action_branch in action]
                            actions[agent_id_decisions] = action
                            value = value.detach().cpu().numpy()

                        transitions[agent_id_decisions]['obs_buf'][agent_ptr[agent_id_decisions]] = init_state
                        transitions[agent_id_decisions]['acts_buf'][agent_ptr[agent_id_decisions]] = action
                        transitions[agent_id_decisions]['act_log_prob_buf'][
                            agent_ptr[agent_id_decisions]] = action_log_probs
                        transitions[agent_id_decisions]['values_buf'][agent_ptr[agent_id_decisions]] = value
                        transitions[agent_id_decisions]['entropies_buf'][agent_ptr[agent_id_decisions]] = entropies

                    for agent_id_terminated in terminal_steps:
                        init_state = terminal_steps[agent_id_terminated].obs
                        init_state = flatten(init_state)
                        with torch.no_grad():
                            init_tensor = torch_from_np(np.array(init_state), self.device)
                            dists, value = self.policy.policy(init_tensor)
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
                        transitions[agent_id_terminated]['act_log_prob_buf'][
                            agent_ptr[agent_id_terminated]] = action_log_probs
                        transitions[agent_id_terminated]['entropies_buf'][agent_ptr[agent_id_terminated]] = entropies

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

                episode_step[agent_id] += 1
                if done or agent_ptr[agent_id] == time_horizon - 1:
                    # If the corresponding agent is done or trajectory is max length
                    with torch.no_grad():
                        next_obs_tensor = torch_from_np(np.array(next_obs), self.device)
                        dists, value = self.policy.policy(next_obs_tensor)
                        action = [dist.sample() for dist in dists]
                        entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                        action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in
                                            zip(dists, action)]
                        next_action = [action_branch.detach().cpu().numpy() for action_branch in action]
                        actions[agent_id] = action
                        value = value.detach().cpu().numpy()

                        if agent_ptr[agent_id] == time_horizon - 1:
                            transitions[agent_id]['rews_buf'][agent_ptr[agent_id]] = value

                    transitions[agent_id]['act_log_prob_buf'][agent_ptr[agent_id]] = action_log_probs
                    transitions[agent_id]['values_buf'][agent_ptr[agent_id]] = value
                    transitions[agent_id]['entropies_buf'][agent_ptr[agent_id]] = entropies

                    actions[agent_id] = next_action

                    if agent_ptr[agent_id] + buffer_length >= buffer.max_buffer_size:
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

                    buffer.store(transitions[agent_id]['obs_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['acts_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['n_obs_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['done_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['values_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['entropies_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['act_log_prob_buf'][:agent_ptr[agent_id] + 1],
                                 advantages[:agent_ptr[agent_id] + 1])

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
                        next_obs_tensor = torch_from_np(np.array(next_obs), self.device)
                        dists, value = self.policy.policy(next_obs_tensor)
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

        print("Generated buffer of lenth: {} in {:.3f} secs.".format(buffer_length, time.time() - start_time))

    def train(self, max_steps: int, buffer_size: int, time_horizon: int,
              batch_size: int, gamma: float, beta: float, lambd: float, epsilon: float):
        step = 0
        while step < max_steps:
            print("Current step: " + str(step))
            buffer = PPOBuffer(buffer_size=buffer_size)
            self.generate_and_fill_buffer(buffer=buffer, time_horizon=time_horizon)
            buffer_length = len(buffer)
            step += buffer_length
            epochs = 3
            p_losses, v_losses, entropies, kls = [], [], [], []
            for i in range(epochs):
                batches = buffer.split_into_batches(batch_size=batch_size)
                for batch in batches:
                    loss, p_loss, v_loss, entropy, kl = self.calc_loss(batch, gamma=gamma,epsilon=epsilon, beta=beta, lambd=lambd)
                    p_losses.append(p_loss)
                    v_losses.append(v_loss)
                    entropies.append(entropy)
                    kls.append(kl)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
                    self.optimizer.step()
            for parameter_group in ppo_module.optimizer.param_groups:
                print("Current learning rate: {:.6f}".format(parameter_group['lr']))

            self.meta_step += 1
            self.learning_rate_scheduler.step(epoch=step)
            self.writer.add_scalar('Task: ' + str(self.task) + '/Entropy ', np.mean(entropies), step)
            self.writer.add_scalar('Task: ' + str(self.task) + '/Policy Loss ', np.mean(p_losses), step)
            self.writer.add_scalar('Task: ' + str(self.task) + '/Value Loss ', np.mean(v_losses), step)
            self.writer.add_scalar('Task: ' + str(self.task) + '/Approx Kullback-Leibler ', np.mean(kls), step)


if __name__ == '__main__':
    writer = SummaryWriter("results/ppo_0")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    ppo_module = PPO_Meta_Learner(device, writer=writer)

    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration_parameters(time_scale=10.0)

    env_parameters_channel = EnvironmentParametersChannel()
    env_parameters_channel.set_float_parameter("seed", 5.0)
    env = UnityEnvironment(file_name="Training/Maze.app",
                           base_port=5000, timeout_wait=120,
                           no_graphics=False, seed=0, side_channels=[engine_configuration_channel, env_parameters_channel])

    hyperparameters = {}

    hyperparameters['Algorithm'] = "PPO"

    hyperparameters['max_steps'] = 50000
    hyperparameters['buffer_size'] = 2000  # Replay buffer size
    hyperparameters['learning_rate'] = 0.0001  # Typical range: 0.00001 - 0.001
    hyperparameters['batch_size'] = 512  # Typical range: 32-512
    hyperparameters['hidden_layers'] = 2
    hyperparameters['layer_size'] = 256
    hyperparameters['time_horizon'] = 64
    hyperparameters['gamma'] = 0.99

    hyperparameters['beta'] = 0.005 # Typical range: 0.0001 - 0.01 Strength of entropy regularization -> make sure entropy falls when reward rises, if it drops too quickly, decrease beta
    hyperparameters['epsilon'] = 0.2 # Typical range: 0.1 - 0.3 Clipping factor of PPO -> small increases stability
    hyperparameters['lambd'] = 0.95 # Typical range: 0.9 - 0.95 GAE Parameter
    hyperparameters['num_epochs'] = 3 # Typical range: 3 - 10

    writer.add_text("Hyperparameters", str(hyperparameters))
    print("Started run with following hyperparameters:")
    for key in hyperparameters:
        print("{:<25s} {:<20s}".format(key, str(hyperparameters[key])))

    ppo_module.set_env_and_detect_spaces(env, task=0)
    ppo_module.init_networks_and_optimizers(hidden_size=hyperparameters['layer_size'], num_hidden_layers=hyperparameters['hidden_layers'],
                                            learning_rate=hyperparameters['learning_rate'], max_scheduler_steps=hyperparameters['max_steps'])

    ppo_module.train(max_steps=hyperparameters['max_steps'], buffer_size=hyperparameters['buffer_size'],
                     time_horizon=hyperparameters['time_horizon'], batch_size=hyperparameters['batch_size'],
                     beta=hyperparameters['beta'], gamma=hyperparameters['gamma'], epsilon=hyperparameters['epsilon'],
                     lambd=hyperparameters['lambd'])

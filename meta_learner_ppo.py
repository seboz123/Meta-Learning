import numpy as np
import time

import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from curiosity_module import CuriosityModule

from utils import torch_from_np, get_probs_and_entropies, init_unity_env
from buffers import PPOBuffer
from models import ActorCriticPolicy
from env_utils import step_env

flatten = lambda l: [item for sublist in l for item in sublist]


class PPO_Meta_Learner:
    def __init__(
            self,
            device: str,
            writer: SummaryWriter,
    ):
        self.device = device
        print("Using: " + str(self.device))
        self.writer = writer
        self.meta_step = 0
        self.step = 0
        self.policy: ActorCriticPolicy
        self.enable_curiosity = False
        self.curiosity = None

    def get_default_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['Algorithm'] = "PPO"

        hyperparameters['enable_curiosity'] = True
        hyperparameters['curiosity_lambda'] = 0.1 # Weight factor of extrinsic reward. 0.1 -> 10*Curiosity
        hyperparameters['curiosity_beta'] = 0.2 # Factor for using more of forward loss or more of inverse loss

        hyperparameters['max_steps'] = 1000000
        hyperparameters['learning_rate'] = 0.0001  # Typical range: 0.00001 - 0.001
        hyperparameters['batch_size'] = 512  # Typical range: 32-512
        hyperparameters['hidden_layers'] = 2
        hyperparameters['layer_size'] = 256
        hyperparameters['time_horizon'] = 64
        hyperparameters['gamma'] = 0.99
        hyperparameters['decay_lr'] = True

        hyperparameters['buffer_size'] = 20000  # Replay buffer size
        hyperparameters['beta'] = 0.005  # Typical range: 0.0001 - 0.01 Strength of entropy regularization -> make sure entropy falls when reward rises, if it drops too quickly, decrease beta
        hyperparameters['epsilon'] = 0.2  # Typical range: 0.1 - 0.3 Clipping factor of PPO -> small increases stability
        hyperparameters['lambd'] = 0.95  # Typical range: 0.9 - 0.95 GAE Parameter
        hyperparameters['num_epochs'] = 3  # Typical range: 3 - 10


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
        max_scheduler_steps = hyperparameters['max_steps']
        learning_rate = hyperparameters['learning_rate']

        obs_dim = self.obs_space
        action_dim = self.action_space
        self.policy = ActorCriticPolicy(obs_dim, action_dim, hyperparameters=hyperparameters, shared_actor_critic=True).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        if hyperparameters['enable_curiosity']:
            self.enable_curiosity = True
            self.curiosity = CuriosityModule(obs_size=obs_dim, enc_size=64, hidden_layers=2,hidden_size=128, learning_rate=0.003, device=self.device, action_shape=self.action_space)
            print("Enabled curiosity module")


        linear_schedule = lambda epoch: max((1 - epoch / max_scheduler_steps), 1e-6)

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_schedule)

    def get_networks_and_parameters(self):
        networks_and_parameters = {'networks': [self.policy]}
        if self.curiosity is not None:
            for network in self.curiosity.get_networks():
                networks_and_parameters['networks'].append(network)
        networks_and_parameters['parameters'] = []
        return networks_and_parameters

    def generate_and_fill_buffer(self, buffer: PPOBuffer, hyperparameters: {}) -> {}:
        start_time = time.time()
        env = self.env
        env.reset()

        time_horizon = hyperparameters['time_horizon']

        # Create transition dict
        transitions = {}
        cumulative_rewards = []
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
                            'values_buf': np.zeros([time_horizon, 2], dtype=np.float32),
                            'rews_buf': np.zeros([time_horizon, 2], dtype=np.float32),
                            'done_buf': np.zeros([time_horizon], dtype=np.float32),
                        }

                    for agent_id_decisions in decision_steps:
                        init_state = decision_steps[agent_id_decisions].obs
                        init_state = flatten(init_state)
                        with torch.no_grad():
                            init_tensor = torch_from_np(np.array(init_state), self.device)
                            dists, value, action_torch, curiosity_value = self.policy.policy(init_tensor)
                            entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                            action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in
                                                zip(dists, action_torch)]
                            action = [branch.detach().cpu().numpy() for branch in action_torch]
                            actions[agent_id_decisions] = action
                            value = value.detach().cpu().numpy()
                            curiosity_value = curiosity_value.detach().cpu().numpy()

                        transitions[agent_id_decisions]['obs_buf'][agent_ptr[agent_id_decisions]] = init_state
                        transitions[agent_id_decisions]['acts_buf'][agent_ptr[agent_id_decisions]] = action
                        transitions[agent_id_decisions]['act_log_prob_buf'][
                            agent_ptr[agent_id_decisions]] = action_log_probs
                        transitions[agent_id_decisions]['values_buf'][agent_ptr[agent_id_decisions], 0] = value
                        transitions[agent_id_decisions]['entropies_buf'][agent_ptr[agent_id_decisions]] = entropies
                        transitions[agent_id_decisions]['values_buf'][agent_ptr[agent_id_decisions], 1] = curiosity_value

                    for agent_id_terminated in terminal_steps:
                        init_state = terminal_steps[agent_id_terminated].obs
                        init_state = flatten(init_state)
                        with torch.no_grad():
                            init_tensor = torch_from_np(np.array(init_state), self.device)
                            dists, value, action_torch, curiosity_value = self.policy.policy(init_tensor)
                            entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                            action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in
                                                zip(dists, action_torch)]
                            action = [action_branch.detach().cpu().numpy() for action_branch in action_torch]
                            value = value.detach().cpu().numpy()
                            curiosity_value = curiosity_value.detach().cpu().numpy()

                        actions[agent_id_terminated] = action
                        transitions[agent_id_terminated]['obs_buf'][agent_ptr[agent_id_terminated]] = init_state
                        transitions[agent_id_terminated]['acts_buf'][agent_ptr[agent_id_terminated]] = action
                        transitions[agent_id_terminated]['values_buf'][agent_ptr[agent_id_terminated], 0] = value
                        transitions[agent_id_terminated]['act_log_prob_buf'][
                            agent_ptr[agent_id_terminated]] = action_log_probs
                        transitions[agent_id_terminated]['entropies_buf'][agent_ptr[agent_id_terminated]] = entropies
                        transitions[agent_id_terminated]['values_buf'][agent_ptr[agent_id_terminated], 1] = curiosity_value


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
                transitions[agent_id]['rews_buf'][agent_ptr[agent_id], 0] = reward
                transitions[agent_id]['n_obs_buf'][agent_ptr[agent_id]] = next_obs
                transitions[agent_id]['done_buf'][agent_ptr[agent_id]] = done

                episode_step[agent_id] += 1
                if done or agent_ptr[agent_id] == time_horizon - 1:
                    # If the corresponding agent is done or trajectory is max length
                    with torch.no_grad():
                        next_obs_tensor = torch_from_np(np.array(next_obs), self.device)
                        dists, value, action_torch, curiosity_value = self.policy.policy(next_obs_tensor)
                        entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                        action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in
                                            zip(dists, action_torch)]
                        next_action = [action_branch.detach().cpu().numpy() for action_branch in action_torch]
                        actions[agent_id] = action
                        value = value.detach().cpu().numpy()
                        curiosity_value = curiosity_value.detach().cpu().numpy()

                        if agent_ptr[agent_id] == time_horizon - 1:
                            transitions[agent_id]['rews_buf'][agent_ptr[agent_id], 0] = value
                            transitions[agent_id]['rews_buf'][agent_ptr[agent_id], 1] = value


                    transitions[agent_id]['act_log_prob_buf'][agent_ptr[agent_id]] = action_log_probs
                    transitions[agent_id]['values_buf'][agent_ptr[agent_id], 0] = value
                    transitions[agent_id]['entropies_buf'][agent_ptr[agent_id]] = entropies
                    transitions[agent_id]['values_buf'][agent_ptr[agent_id], 1] = curiosity_value

                    actions[agent_id] = next_action

                    if agent_ptr[agent_id] + buffer_length >= buffer.max_buffer_size:
                        buffer_finished = True
                        break
                    if done:
                        episode_lengths.append(episode_step[agent_id])
                        episode_step[agent_id] = 0
                        cumulative_rewards.append(current_added_reward[agent_id])
                        current_added_reward[agent_id] = 0

                    trajectory_lengths.append(agent_ptr[agent_id] + 1)

                    advantages = np.zeros((agent_ptr[agent_id] + 1, 2))
                    returns = np.zeros((agent_ptr[agent_id] + 1, 2))

                    if self.enable_curiosity:
                        transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1, 1] = self.curiosity.evaluate(
                            transitions[agent_id]['obs_buf'][:agent_ptr[agent_id] + 1],
                            transitions[agent_id]['acts_buf'][:agent_ptr[agent_id] + 1],
                            transitions[agent_id]['n_obs_buf'][:agent_ptr[agent_id] + 1]).detach().cpu().numpy()

                    # Calculate Advantages and Returns (GAE)
                    advantages[:, 0] = self.get_gae(transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1, 0],
                                              transitions[agent_id]['values_buf'][:agent_ptr[agent_id] + 1, 0], gamma=hyperparameters['gamma'],lambd=hyperparameters['lambd'])
                    advantages[:, 0] = (advantages[:, 0] - np.mean(advantages[:, 0])) / (np.std(advantages[:, 0])+ 0.000001)

                    advantages[:, 1] = self.get_gae(transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1, 1],
                                              transitions[agent_id]['values_buf'][:agent_ptr[agent_id] + 1, 1], gamma=hyperparameters['gamma'],lambd=hyperparameters['lambd'])
                    advantages[:, 1] = (advantages[:, 1] - np.mean(advantages[:, 1])) / (np.std(advantages[:, 1])+ 0.000001)

                    returns[:, 0] = advantages[:, 0] + transitions[agent_id]['values_buf'][:agent_ptr[agent_id] + 1, 0]
                    returns[:, 1] = advantages[:, 1] + transitions[agent_id]['values_buf'][:agent_ptr[agent_id] + 1, 1]

                    buffer.store(transitions[agent_id]['obs_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['acts_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['n_obs_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['done_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['values_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['entropies_buf'][:agent_ptr[agent_id] + 1],
                                 transitions[agent_id]['act_log_prob_buf'][:agent_ptr[agent_id] + 1],
                                 advantages[:agent_ptr[agent_id] + 1], returns[:agent_ptr[agent_id] + 1]
                                 )

                    buffer_length += agent_ptr[agent_id] + 1

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
                    transitions[agent_id]['act_log_prob_buf'] = np.zeros(
                        [time_horizon, len(self.action_space)],
                        dtype=np.float32)
                    transitions[agent_id]['entropies_buf'] = np.zeros(
                        [time_horizon, len(self.action_space)],
                        dtype=np.float32)
                    transitions[agent_id]['values_buf'] = np.zeros(
                        [time_horizon, 2],
                        dtype=np.float32)
                    agent_ptr[agent_id] = 0

                else:  # If the corresponding agent is not done, continue
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs

                    with torch.no_grad():
                        next_obs_tensor = torch_from_np(np.array(next_obs), self.device)
                        dists, value, action_torch, curiosity_value = self.policy.policy(next_obs_tensor)
                        entropies = [dist.entropy().detach().cpu().numpy() for dist in dists]
                        action_log_probs = [dist.log_prob(s_action).detach().cpu().numpy() for dist, s_action in
                                            zip(dists, action_torch)]
                        next_action = [action_branch.detach().cpu().numpy() for action_branch in action_torch]
                        actions[agent_id_decisions] = action
                        value = value.detach().cpu().numpy()
                        curiosity_value = curiosity_value.detach().cpu().numpy()

                    transitions[agent_id]['entropies_buf'][agent_ptr[agent_id]] = entropies
                    transitions[agent_id]['acts_buf'][agent_ptr[agent_id]] = next_action
                    transitions[agent_id]['values_buf'][agent_ptr[agent_id], 0] = value
                    transitions[agent_id]['act_log_prob_buf'][agent_ptr[agent_id]] = action_log_probs
                    transitions[agent_id]['values_buf'][agent_ptr[agent_id], 1] = curiosity_value

                    actions[agent_id] = next_action

            first_iteration = False

        print("Mean Cumulative Reward: {} at step {}".format(np.mean(cumulative_rewards), self.step))
        print("Mean Episode Length: {} at step {}".format(np.mean(trajectory_lengths), self.step))

        self.writer.add_scalars('task_' + str(self.task) + r"\PPO Cumulative Reward",
                                {r'\meta_step_' + str(self.meta_step): np.mean(cumulative_rewards)}, self.step)
        self.writer.add_scalars('task_' + str(self.task) + r'\PPO Mean Episode Length',
                                {r'\meta_step_' + str(self.meta_step): np.mean(episode_lengths)}, self.step)

        print("Generated buffer of lenth: {} in {:.3f} secs.".format(buffer_length, time.time() - start_time))

        return np.mean(cumulative_rewards), np.mean(episode_lengths)

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
        obs = torch_from_np(obs, self.device).to(dtype=torch.float32)
        acts = torch_from_np(acts, self.device)
        dists, values, actions, curiosity_values = self.policy.policy(obs)
        cumulated_log_probs, entropies, all_log_probs = get_probs_and_entropies(acts, dists, self.device)  # Error here
        out_value = []
        for value, curiosity_value in zip(values, curiosity_values):
            out_value.append(value)
            out_value.append(curiosity_value)
        values = torch.stack(out_value).view(-1, 2)
        return cumulated_log_probs, entropies, values, dists, actions

    def calc_value_loss(self, values: torch.FloatTensor, old_values: torch.FloatTensor,returns: torch.FloatTensor,
                        eps) -> torch.FloatTensor:
        value_losses = []
        clipped_value_est = old_values[:, 0] + torch.clamp(values[:, 0] - old_values[:, 0], -1 * eps, eps)
        tmp_loss_1 = torch.pow(returns[:, 0] - values[:, 0], 2)
        tmp_loss_2 = torch.pow(returns[:, 0] - clipped_value_est, 2)
        value_loss = torch.max(tmp_loss_1, tmp_loss_2)
        value_losses.append(value_loss)
        curiosity_value_loss = None

        if self.enable_curiosity:
            clipped_value_est = old_values[:, 1] + torch.clamp(values[:, 1] - old_values[:, 1], -1 * eps, eps)
            tmp_loss_1 = torch.pow(returns[:, 1] - values[:, 1], 2)
            tmp_loss_2 = torch.pow(returns[:, 1] - clipped_value_est, 2)
            curiosity_value_loss = torch.max(tmp_loss_1, tmp_loss_2)
            value_losses.append(curiosity_value_loss)
            value_loss = 0.5 * torch.mean(torch.stack(value_losses))

        return value_loss, curiosity_value_loss

    def calc_policy_loss(self, advantages: torch.FloatTensor, log_probs: torch.FloatTensor,
                         old_log_probs: torch.FloatTensor, eps: float = 0.2) -> torch.FloatTensor:
        ratio = torch.exp(log_probs - old_log_probs)
        tmp_loss_1 = ratio * advantages
        tmp_loss_2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        policy_loss = - (torch.min(tmp_loss_1, tmp_loss_2)).mean()
        return policy_loss

    def calc_loss(self, buffer, epsilon, beta, hyperparameters: {}) -> torch.Tensor:
        losses = {}
        old_log_probs = np.sum(buffer.act_log_probs, axis=1)
        obs = buffer.observations
        acts = buffer.actions
        values = buffer.values
        advantages = buffer.advantages[:, 0]
        returns = buffer.returns

        old_values = torch_from_np(values, self.device)
        old_log_probs = torch_from_np(old_log_probs, self.device)
        returns = torch_from_np(returns, self.device)
        advantages = torch_from_np(advantages, self.device)

        log_probs, entropies, values, dists, actions = self.evaluate_actions(obs, acts)

        value_loss, curiosity_value_loss = self.calc_value_loss(values, old_values, returns, eps=epsilon)

        policy_loss = self.calc_policy_loss(advantages, log_probs, old_log_probs, eps=epsilon)
        entropy = entropies.mean()

        kullback_leibler = 0.5 * torch.mean(torch.pow(log_probs - old_log_probs, 2))
        policy_loss = policy_loss
        value_loss = 0.5 * value_loss
        entropy_loss = - beta * entropy

        losses['Policy Loss'] = policy_loss.item()
        losses['Value Loss'] = value_loss.item()
        losses['Entropy'] = entropy.item()
        losses['Approximate Kullback-Leibler'] = kullback_leibler.item()

        loss = value_loss + policy_loss + entropy_loss

        if self.enable_curiosity:
            f_loss, i_loss = self.curiosity.calc_loss_ppo_sac(buffer)
            curiosity_loss = 1 / hyperparameters['curiosity_lambda'] * (hyperparameters['curiosity_beta'] * f_loss + (1-hyperparameters['curiosity_beta']) * i_loss)
            losses['Curiosity Forward Loss'] = f_loss.item()
            losses['Curiosity Inverse Loss'] = i_loss.item()


        return loss,curiosity_loss, losses



    def close_env(self):
        self.env.close()

    def train(self, hyperparameters: dict):

        max_steps = hyperparameters['max_steps']
        batch_size = hyperparameters['batch_size']
        time_horizon = hyperparameters['time_horizon']
        buffer_size = hyperparameters['buffer_size']

        self.step = 0
        mean_rewards = []
        mean_episode_lengths = []
        while self.step < max_steps:
            print("Current PPO Training step: " + str(self.step))

            epsilon = max(hyperparameters['epsilon'] * (1 - self.step/max_steps), 0.1)
            beta = max(hyperparameters['beta'] * (1 - self.step/max_steps), 1e-5)


            for parameter_group in self.optimizer.param_groups:
                print("Current PPO Training learning rate: {:.6f}".format(parameter_group['lr']))

            buffer = PPOBuffer(buffer_size=buffer_size, obs_size=self.obs_space, action_space=self.action_space)
            mean_reward, mean_episode_length = self.generate_and_fill_buffer(hyperparameters=hyperparameters, buffer=buffer)

            mean_rewards.append(mean_reward)
            mean_episode_lengths.append(mean_episode_length)

            buffer_length = len(buffer)
            self.step += buffer_length

            total_losses ,p_losses, v_losses, entropies, kls, curio_f_losses, curio_i_losses = [], [], [], [], [], [], []
            for i in range(hyperparameters['num_epochs']):
                batches = buffer.split_into_batches(batch_size=batch_size)
                for batch in batches:
                    loss, curiosity_loss, losses = self.calc_loss(batch,epsilon=epsilon, beta=beta, hyperparameters=hyperparameters)
                    total_losses.append(loss.item())
                    p_losses.append(losses['Policy Loss'])
                    v_losses.append(losses['Value Loss'])
                    entropies.append(losses['Entropy'])
                    kls.append(losses['Approximate Kullback-Leibler'])
                    if hyperparameters['enable_curiosity']:
                        curio_f_losses.append(losses['Curiosity Forward Loss'])
                        curio_i_losses.append(losses['Curiosity Inverse Loss'])
                        self.curiosity.optimizer.zero_grad()
                        curiosity_loss.backward()
                        for network in self.curiosity.get_networks():
                            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                        self.curiosity.optimizer.step()
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                    self.optimizer.step()

            if(hyperparameters['decay_lr']):
                self.learning_rate_scheduler.step(epoch=self.step)

            print("Total Loss: {}".format(np.mean(total_losses)))
            print("Curiosity Forward Loss: {}".format(np.mean(curio_f_losses)))
            print("Curiosity Inverse Loss: {}".format(np.mean(curio_i_losses)))
            print("Value Loss: {}".format(np.mean(v_losses)))
            print("Policy Loss: {}".format(np.mean(p_losses)))
            print("Entropy Loss: {}".format(np.mean(entropies)))



            if hyperparameters['enable_curiosity']:
                self.writer.add_scalars('task_' + str(self.task) + r"\PPO Curiosity Forward Loss",
                                        {r'\meta_step_' + str(self.meta_step): np.mean(curio_f_losses)}, self.step)
                self.writer.add_scalars('task_' + str(self.task) + r"\PPO Curiosity Inverse Loss",
                                        {r'\meta_step_' + str(self.meta_step): np.mean(curio_i_losses)}, self.step)

            self.writer.add_scalars('task_' + str(self.task) + r"\PPO Entropy Loss",
                                    {r'\meta_step_'+str(self.meta_step): np.mean(entropies)}, self.step)
            self.writer.add_scalars('task_' + str(self.task) + r'\PPO Policy Loss',
                                   {r'\meta_step_' + str(self.meta_step) : np.mean(p_losses)}, self.step)
            self.writer.add_scalars('task_' + str(self.task) + r'\PPO Value Loss',
                                   {r'\meta_step_' + str(self.meta_step) : np.mean(v_losses)}, self.step)
            self.writer.add_scalars(
                'task_' + str(self.task) + r'\PPO Approx Kullback-Leibler',
                {r'\meta_step_' + str(self.meta_step): np.mean(kls)}, self.step)

        return np.mean(mean_rewards), np.mean(mean_episode_lengths)

if __name__ == '__main__':

    run_id = "results/ppo_0"

    writer = SummaryWriter(run_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    ppo_module = PPO_Meta_Learner(device, writer=writer)

    env = init_unity_env("mMaze/RLProject.exe", maze_rows=3, maze_cols=3, maze_seed=0, random_agent=0, random_target=0,
                         agent_x=0, agent_z=0, target_x=0, target_z=2)

    training_parameters = ppo_module.get_default_hyperparameters()

    training_parameters['Algorithm'] = "PPO"
    training_parameters['run_id'] = run_id

    training_parameters['max_steps'] = 5000000
    training_parameters['buffer_size'] = 20000  # Replay buffer size
    training_parameters['learning_rate'] = 0.0001  # Typical range: 0.00001 - 0.001
    training_parameters['batch_size'] = 1024  # Typical range: 32-512
    training_parameters['hidden_layers'] = 2
    training_parameters['layer_size'] = 512
    training_parameters['time_horizon'] = 256
    training_parameters['gamma'] = 0.99
    training_parameters['decay_lr'] = True
    training_parameters['enable_curiosity'] = True

    training_parameters['beta'] = 0.005 # Typical range: 0.0001 - 0.01 Strength of entropy regularization -> make sure entropy falls when reward rises, if it drops too quickly, decrease beta
    training_parameters['epsilon'] = 0.2 # Typical range: 0.1 - 0.3 Clipping factor of PPO -> small increases stability
    training_parameters['lambd'] = 0.95 # Typical range: 0.9 - 0.95 GAE Parameter
    training_parameters['num_epochs'] = 3 # Typical range: 3 - 10

    writer.add_text("training_parameters", str(training_parameters))
    print("Started run with following training_parameters:")
    for key in training_parameters:
        print("{:<25s} {:<20s}".format(key, str(training_parameters[key])))

    ppo_module.set_env_and_detect_spaces(env, task=0)
    ppo_module.init_networks_and_optimizers(training_parameters)

    ppo_module.train(training_parameters)

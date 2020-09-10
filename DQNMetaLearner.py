import numpy as np
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict

from mlagents_envs.environment import UnityEnvironment

from torch.utils.tensorboard import SummaryWriter
from ActionFlattener import ActionFlattener
from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer

def flatten(input):
    return [item for sublist in input for item in sublist]


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())


class Network(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


class DQN_Meta_Learner:
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
            decay_lr: bool = True,
            gamma: float = 0.99,
            update_period: int = 5000,
            # PER parameters
            alpha: float = 0.7,
            beta: float = 0.5,
            prior_eps: float = 1e-6,
            # Categorical DQN parameters

            # N-step Learning
            writer: SummaryWriter = None,
            run_id: str = r"\DQN_Rainbow",
            decision_requester: int = 5,
    ):
        """Initialization.

        Args:
            env (UnityEnvironment.env): UnityEnvironment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        self.task = None
        self.gamma = gamma
        self.decision_requester = decision_requester
        self.update_period = update_period
        self.decay_lr = decay_lr
        self.meta_step = 0
        self.step_counter = 0
        self.alpha = alpha
        # device: cpu / gpu
        self.device = device
        print("Using: " + str(self.device))
        self.writer = writer
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps

    def set_environment(self, env):
        env.reset()
        self.env = env
        result = 0
        for brain_name in self.env.behavior_specs:
            # Set up flattened action space
            branches = self.env.behavior_specs[brain_name].discrete_action_branches
            self.flattener = ActionFlattener(branches)
            self.action_space = self.flattener.action_space
            print("Action space: " + str(self.action_space))
            for shape in self.env.behavior_specs[brain_name].observation_shapes:
                if (len(shape) == 1):
                    result += shape[0]
        self.obs_dim = result
        print("Obs space: " + str(result))
        self.env.reset()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def init_network_and_optim(self, obs_dim, flattener: ActionFlattener, learning_rate=0.001, v_max: int = 13,
                               v_min: int = -13, atom_size: int = 51):
        self.obs_dim = obs_dim
        self.flattener = flattener
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        self.dqn = Network(
            self.obs_dim, self.flattener.action_space.n, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            self.obs_dim, self.flattener.action_space.n, self.atom_size, self.support
        ).to(self.device)
        # Load theta parameters if available
        # if theta:
        #     self.dqn.load_state_dict(theta)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)

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
                                                           terminal_steps[agent_id_terminated].reward,
                                                           not terminal_steps[agent_id_terminated].interrupted]

        return agents_transitions

    def calc_loss(self, memory, memory_n, n_step: int = 1) -> torch.Tensor:
        use_n_step = True if n_step > 1 else False

        samples = memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if use_n_step:
            samples = memory_n.sample_batch_from_idxs(indices)
            gamma = self.gamma ** n_step
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)

            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        return loss, elementwise_loss, indices

    def update_priorities(self, elementwise_loss, indices, memory):
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

    def generate_and_fill_buffer(self, buffer_size, n_step, batch_size, task, max_trajectory_length=600):
        if (buffer_size <= batch_size):
            raise ValueError
        start_time = time.time()
        env = self.env
        env.reset()

        # Create transition dict
        experiences = {}
        transitions = {}

        buffer_length = 0
        first_iteration = True

        rewards = []
        trajectory_lengths = []

        while buffer_length < buffer_size:
            if first_iteration:
                for brain in env.behavior_specs:
                    decision_steps, terminal_steps = env.get_steps(brain)
                    num_agents = len(decision_steps)
                    actions = np.zeros((num_agents, 1))
                    print("Brain :" + str(brain) + " with " + str(num_agents) + " agents detected.")
                    agent_ptr = [0 for _ in range(num_agents)]  # Create pointer for agent transitions

                    for i in range(num_agents):
                        experiences[i] = [[], 0, 0, [], False]
                        transitions[i] = {'obs_buf': np.zeros([max_trajectory_length, self.obs_dim], dtype=np.float32),
                                          'n_obs_buf': np.zeros([max_trajectory_length, self.obs_dim],
                                                                dtype=np.float32),
                                          'acts_buf': np.zeros([max_trajectory_length], dtype=np.float32),
                                          'rews_buf': np.zeros([max_trajectory_length], dtype=np.float32),
                                          'done_buf': np.zeros([max_trajectory_length], dtype=np.float32)}
                        # {0: {obs_buf: ,n_obs_buf: , ....}, 1: {obs_buf: ,n_obs_buf: , ....}, 2: {obs_buf: ,n_obs_buf: , ....}}

                        # step_counter[i] = 0

                    for agent_id_decisions in decision_steps:
                        init_state = decision_steps[agent_id_decisions].obs
                        init_state = flatten(init_state)
                        action = self.select_action(init_state)
                        actions[agent_id_decisions] = action
                        experiences[agent_id_decisions][0] = init_state
                        transitions[agent_id_decisions]['obs_buf'][0] = init_state
                        transitions[agent_id_decisions]['acts_buf'][0] = action

                    for agent_id_terminated in terminal_steps:
                        init_state = terminal_steps[agent_id_terminated].obs
                        init_state = flatten(init_state)
                        action = self.select_action(init_state)
                        actions[agent_id_terminated] = action
                        experiences[agent_id_terminated][0] = init_state
                        transitions[agent_id_terminated]['obs_buf'][0] = init_state
                        transitions[agent_id_terminated]['acts_buf'][0] = action

                obs_buf = np.zeros([buffer_size, self.obs_dim], dtype=np.float32)
                n_obs_buf = np.zeros([buffer_size, self.obs_dim], dtype=np.float32)
                acts_buf = np.zeros([buffer_size], dtype=np.float32)
                rews_buf = np.zeros([buffer_size], dtype=np.float32)
                done_buf = np.zeros([buffer_size], dtype=np.float32)
            else:
                for agent in experiences:
                    # Set Observation and Action to the last next_obs and the selected action for the observation
                    experiences[agent][0] = experiences[agent][3]  # obs = next_obs
                    experiences[agent][1] = actions[int(agent)]  # action = action_vector[]

            # Create Action vector
            action_vector = [self.flattener.lookup_action(int((action))) for action in actions]
            # Step environment
            next_experiences = self.step_env(self.env, np.array(action_vector))
            # Create action vector to store actions
            actions = np.zeros((num_agents, 1))

            for agent_id in next_experiences:
                if buffer_length >= buffer_size:
                    break
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

                if not done:  # If the corresponding agent is not done yet, select and action and continue
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs
                    next_action = int(self.select_action(next_obs))  # Action of next step
                    actions[agent_id] = next_action
                    transitions[agent_id]['acts_buf'][agent_ptr[agent_id]] = next_action
                else:  # If the corresponding agent is done, store the trajectory into obs_buf, rews_buf...
                    next_action = int(self.select_action(next_obs))
                    actions[agent_id] = next_action
                    rewards.append(sum(transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1]))
                    trajectory_lengths.append(agent_ptr[agent_id])
                    for i in range(agent_ptr[agent_id] + 1):  # For every experiences store it in buffer
                        if (i + buffer_length >= buffer_size):
                            break
                        obs_buf[i + buffer_length] = transitions[agent_id]['obs_buf'][i]
                        acts_buf[i + buffer_length] = transitions[agent_id]['acts_buf'][i]
                        rews_buf[i + buffer_length] = transitions[agent_id]['rews_buf'][i]
                        n_obs_buf[i + buffer_length] = transitions[agent_id]['n_obs_buf'][i]
                        done_buf[i + buffer_length] = transitions[agent_id]['done_buf'][i]

                    buffer_length += agent_ptr[agent_id] + 1

                    transitions[agent_id]['obs_buf'] = np.zeros(
                        [max_trajectory_length, self.obs_dim], dtype=np.float32)
                    transitions[agent_id]['acts_buf'] = np.zeros([max_trajectory_length],
                                                                 dtype=np.float32)
                    transitions[agent_id]['rews_buf'] = np.zeros([max_trajectory_length],
                                                                 dtype=np.float32)
                    transitions[agent_id]['n_obs_buf'] = np.zeros(
                        [max_trajectory_length, self.obs_dim], dtype=np.float32)
                    transitions[agent_id]['done_buf'] = np.zeros([max_trajectory_length],
                                                                 dtype=np.float32)
                    agent_ptr[agent_id] = 0
            first_iteration = False

        self.writer.add_scalar('Task: ' + str(task) + '/Cumulative Reward', np.mean(rewards), self.meta_step)
        self.writer.add_scalar('Task: ' + str(task) + '/Mean Episode Length', np.mean(trajectory_lengths),
                               self.meta_step)

        use_n_step = True if n_step > 1 else False

        acion_dim = 1
        memory = PrioritizedReplayBuffer(
            self.obs_dim, buffer_size, acion_dim, batch_size, n_step=n_step, alpha=self.alpha
        )
        memory_n = ReplayBuffer(
            self.obs_dim, buffer_size, acion_dim, batch_size, n_step=n_step, gamma=self.gamma
        )
        for ptr in range(len(rews_buf)):
            transition = [obs_buf[ptr], acts_buf[ptr], rews_buf[ptr], n_obs_buf[ptr], done_buf[ptr]]
            if use_n_step:
                one_step_transition = memory_n.store(*transition)
            else:
                one_step_transition = transition
            if one_step_transition:
                memory.store(*one_step_transition)

        print("Finished generating Buffer with size of: {} in {:.3f}s!".format(len(memory), time.time() - start_time))
        # Memory = PER
        # Memory_n = Replay
        return memory, memory_n, buffer_size

    def generate_trajectories(self, env: UnityEnvironment, num_trajectories: int,
                              max_trajectory_length: int = 600) -> tuple:
        """
        :param env: Unity environment which creates the trajectories
        :param num_trajectories: Number of trajectories to create
        :param max_trajectory_length: Maximum length of a trajectory
        :return: Tuple: (obs_buf, acts_buf, rews_buf, n_obs_buf, done_buf)
        """
        start_time = time.time()
        env.reset()

        # Create transition dict
        experiences = {}
        transitions = {}

        for brain in env.behavior_specs:
            decision_steps, terminal_steps = env.get_steps(brain)
            num_agents = len(decision_steps)
            actions = np.zeros((num_agents, 1))
            print("Brain :" + str(brain) + " with " + str(num_agents) + " agents detected.")
            agent_ptr = [0 for _ in range(num_agents)]  # Create pointer for agent transitions

            for i in range(num_agents):
                experiences[i] = [[], 0, 0, [], False]
                transitions[i] = {'obs_buf': np.zeros([max_trajectory_length, self.obs_dim], dtype=np.float32),
                                  'n_obs_buf': np.zeros([max_trajectory_length, self.obs_dim], dtype=np.float32),
                                  'acts_buf': np.zeros([max_trajectory_length], dtype=np.float32),
                                  'rews_buf': np.zeros([max_trajectory_length], dtype=np.float32),
                                  'done_buf': np.zeros([max_trajectory_length], dtype=np.float32)}

            for agent_id_decisions in decision_steps:
                init_state = decision_steps[agent_id_decisions].obs
                init_state = flatten(init_state)
                action = self.select_action(init_state)
                actions[agent_id_decisions] = action
                experiences[agent_id_decisions][0] = init_state
                transitions[agent_id_decisions]['obs_buf'][0] = init_state
                transitions[agent_id_decisions]['acts_buf'][0] = action

            for agent_id_terminated in terminal_steps:
                init_state = terminal_steps[agent_id_terminated].obs
                init_state = flatten(init_state)
                action = self.select_action(init_state)
                actions[agent_id_terminated] = action
                experiences[agent_id_terminated][0] = init_state
                transitions[agent_id_decisions]['obs_buf'][0] = init_state
                transitions[agent_id_decisions]['acts_buf'][0] = action

        obs_buf = np.zeros([num_trajectories * max_trajectory_length, self.obs_dim], dtype=np.float32)
        n_obs_buf = np.zeros([num_trajectories * max_trajectory_length, self.obs_dim], dtype=np.float32)
        acts_buf = np.zeros([num_trajectories * max_trajectory_length], dtype=np.float32)
        rews_buf = np.zeros([num_trajectories * max_trajectory_length], dtype=np.float32)
        done_buf = np.zeros(num_trajectories * max_trajectory_length, dtype=np.float32)

        finished_trajectories = 0
        first_iteration = True
        trajectory_length = 0

        while finished_trajectories < num_trajectories:

            if not first_iteration:
                for agent in experiences:
                    # Set Observation and Action to the last next_obs and the selected action for the observation
                    experiences[agent][0] = experiences[agent][3]  # obs = next_obs
                    experiences[agent][1] = actions[int(agent)]  # action = action_vector[]

            # Create Action vector
            action_vector = [self.flattener.lookup_action(int((action))) for action in actions]
            # Step environment
            next_experiences = self.step_env(self.env, np.array(action_vector))
            # Create action vector to store actions
            actions = np.zeros((num_agents, 1))

            for agent_id in next_experiences:
                reward = next_experiences[agent_id][1]  # Reward
                next_obs = flatten(next_experiences[agent_id][0])  # Next_obs
                done = next_experiences[agent_id][2]  # Done

                # Store 1-step-experience of every agent_id {agent_id0:[obs,act,rew,n_obs,done] agent_id1: ....}
                experiences[agent_id][2] = reward
                experiences[agent_id][3] = next_obs
                experiences[agent_id][4] = done
                # Store trajectory of every Agent {Agent0:[obs,act,rew,n_obs,done,obs,act,rew,n_obs,done,.... Agent1: .}
                transitions[agent_id]['rews_buf'][agent_ptr[agent_id]] = reward
                transitions[agent_id]['n_obs_buf'][agent_ptr[agent_id]] = next_obs
                transitions[agent_id]['done_buf'][agent_ptr[agent_id]] = done

                if not done:  # If the corresponding agent is not done yet, select and action and continue
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs
                    next_action = int(self.select_action(next_obs))  # Action of next step
                    actions[agent_id] = next_action
                    transitions[agent_id]['acts_buf'][agent_ptr[agent_id]] = next_action
                else:  # If the corresponding agent is done, store the trajectory into obs_buf, rews_buf...
                    finished_trajectories += 1  # 1 trajectory is finished
                    next_action = int(self.select_action(next_obs))
                    actions[agent_id] = next_action
                    for i in range(agent_ptr[agent_id] + 1):
                        obs_buf[i + trajectory_length] = transitions[agent_id]['obs_buf'][i]
                        acts_buf[i + trajectory_length] = transitions[agent_id]['acts_buf'][i]
                        rews_buf[i + trajectory_length] = transitions[agent_id]['rews_buf'][i]
                        n_obs_buf[i + trajectory_length] = transitions[agent_id]['n_obs_buf'][i]
                        done_buf[i + trajectory_length] = transitions[agent_id]['done_buf'][i]
                    trajectory_length += agent_ptr[agent_id] + 1

                    transitions[agent_id]['obs_buf'] = np.zeros([max_trajectory_length, self.obs_dim], dtype=np.float32)
                    transitions[agent_id]['acts_buf'] = np.zeros([max_trajectory_length], dtype=np.float32)
                    transitions[agent_id]['rews_buf'] = np.zeros([max_trajectory_length], dtype=np.float32)
                    transitions[agent_id]['n_obs_buf'] = np.zeros([max_trajectory_length, self.obs_dim],
                                                                  dtype=np.float32)
                    transitions[agent_id]['done_buf'] = np.zeros([max_trajectory_length], dtype=np.float32)
                    agent_ptr[agent_id] = 0
            first_iteration = False

        obs_buf = obs_buf[:trajectory_length]
        acts_buf = acts_buf[:trajectory_length]
        rews_buf = rews_buf[:trajectory_length]
        n_obs_buf = n_obs_buf[:trajectory_length]
        done_buf = done_buf[:trajectory_length]
        print("Finished generating {} trajectories with {} experiences in {:.3f}s!".format(finished_trajectories,
                                                                                           trajectory_length,
                                                                                           time.time() - start_time))

        return (obs_buf, acts_buf, rews_buf, n_obs_buf, done_buf), trajectory_length

    def eval_policy(self, num_trajectories, max_trajectory_length, task):
        print("Evaluating on {} trajectories.".format(num_trajectories))
        env = self.env
        env.reset()
        transitions = {}

        for brain in env.behavior_specs:
            decision_steps, terminal_steps = env.get_steps(brain)
            num_agents = len(decision_steps)
            actions = np.zeros((num_agents, 1))
            agent_ptr = [0 for _ in range(num_agents)]  # Create pointer for agent transitions

            for i in range(num_agents):
                transitions[i] = {'rews_buf': np.zeros([max_trajectory_length], dtype=np.float32)}

            for agent_id_decisions in decision_steps:
                init_state = decision_steps[agent_id_decisions].obs
                init_state = flatten(init_state)
                action = self.select_action(init_state)
                actions[agent_id_decisions] = action

            for agent_id_terminated in terminal_steps:
                init_state = terminal_steps[agent_id_terminated].obs
                init_state = flatten(init_state)
                action = self.select_action(init_state)
                actions[agent_id_terminated] = action

        finished_trajectories = 0
        rewards = []

        while finished_trajectories < num_trajectories:
            # Create Action vector
            action_vector = [self.flattener.lookup_action(int((action))) for action in actions]
            # Step environment
            next_experiences = self.step_env(self.env, np.array(action_vector))
            # Create action vector to store actions
            actions = np.zeros((num_agents, 1))

            for agent_id in next_experiences:
                reward = next_experiences[agent_id][1]  # Reward
                next_obs = flatten(next_experiences[agent_id][0])  # Next_obs
                done = next_experiences[agent_id][2]  # Done
                # Store trajectory of every Agent {Agent0:[obs,act,rew,n_obs,done,obs,act,rew,n_obs,done,.... Agent1: ....}
                transitions[agent_id]['rews_buf'][agent_ptr[agent_id]] = reward

                if not done:  # If the corresponding agent is not done yet, select and action and continue
                    agent_ptr[agent_id] += 1
                    next_action = int(self.select_action(next_obs))  # Action of next step
                    actions[agent_id] = next_action
                else:  # If the corresponding agent is done, store the trajectory into obs_buf, rews_buf...
                    finished_trajectories += 1  # 1 trajectory is finished
                    next_action = int(self.select_action(next_obs))
                    actions[agent_id] = next_action
                    reward = sum(transitions[agent_id]['rews_buf'])
                    transitions[agent_id]['rews_buf'] = np.zeros([max_trajectory_length], dtype=np.float32)
                    agent_ptr[agent_id] = 0
                    rewards.append(reward)

        print("Mean evaluated reward: " + str(np.mean(rewards)) + " for task: " + str(task))
        self.writer.add_scalar('Task: ' + str(task) + '/Cumulative Reward', np.mean(rewards), self.meta_step)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""

        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(len(next_action)), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (len(action) - 1) * self.atom_size, len(action)
                ).long()
                    .unsqueeze(1)
                    .expand(len(action), self.atom_size)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        # print(dist)
        # print(action)
        log_p = torch.log(dist[range(len(action)), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

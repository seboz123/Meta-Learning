import numpy as np
import time

import torch
import torch.optim as optim

from typing import Dict, List

from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from mlagents_envs.environment import UnityEnvironment

from torch.utils.tensorboard import SummaryWriter

from buffers import DQNBuffer, PrioritizedDQNBuffer
from utils import ActionFlattener, torch_from_np
from env_utils import step_env
from models import DeepQNetwork


def flatten(input):
    return [item for sublist in input for item in sublist]


class DQN_Meta_Learner:
    """DQN Agent interacting with environment.
    """

    def __init__(
            self,
            device: str,
            writer: SummaryWriter,

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
        self.device = device
        self.writer = writer

    def init_network_and_optim(self, hidden_size: int, beta: float, prior_eps: float, network_num_hidden_layers: int,
                               max_scheduler_steps: int,v_max: int, v_min: int, atom_size: int, learning_rate: float,
                               gamma: float, alpha: float):
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.alpha = alpha
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        self.beta = beta
        self.prior_eps = prior_eps
        self.gamma = gamma

        self.dqn = DeepQNetwork(in_dim=self.obs_dim, out_dim=self.action_dim.n, hidden_size=hidden_size, num_hidden_layers=network_num_hidden_layers, atom_size=self.atom_size, support=self.support
        ).to(self.device)
        self.dqn_target = DeepQNetwork(in_dim=self.obs_dim, out_dim=self.action_dim.n, hidden_size=hidden_size, num_hidden_layers=network_num_hidden_layers, atom_size=self.atom_size, support=self.support
        ).to(self.device)

        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)

        linear_schedule = lambda epoch: (1 - epoch / max_scheduler_steps)

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_schedule)

    def set_env_and_detect_spaces(self, env, task):
        env.reset()
        self.env = env
        self.task = task
        result = 0
        for brain_name in self.env.behavior_specs:
            # Set up flattened action space
            branches = self.env.behavior_specs[brain_name].discrete_action_branches
            flattener = ActionFlattener(branches)
            self.flattener = flattener
            action_space = flattener.action_space
            for shape in self.env.behavior_specs[brain_name].observation_shapes:
                if (len(shape) == 1):
                    result += shape[0]
        self.obs_dim = result
        self.action_dim = action_space

        print("Space detected successfully.")
        print("Observation space detected as {}\nAction space detected as: {}".format(result, branches))
        print("For DQN, Action space gets reshaped to: {}".format(str(action_space)))

        self.env.reset()

    def get_state_dict(self):
        return [self.dqn.state_dict(), self.dqn_target.state_dict()]

    def select_action(self, state: np.ndarray, epsilon, device) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        with torch.no_grad():
            state_tensor = torch_from_np(state, device)
            selected_action = self.dqn(state_tensor).detach().cpu().numpy()[0]
            if np.random.random(1) > epsilon:
                selected_action = selected_action.argmax()
            else:
                selected_action = np.random.randint(0, len(selected_action))

        return selected_action


    def calc_loss(self, memory, memory_n, batch_size: int,n_step: int) -> [torch.Tensor, torch.Tensor, np.ndarray]:
        use_n_step = True if n_step > 1 else False

        samples = memory.sample_batch(batch_size=batch_size, beta=self.beta)
        weights = torch_from_np(samples["weights"].reshape(-1, 1), self.device)

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

    def generate_and_fill_buffer(self, buffer_size, time_horizon, epsilon,max_trajectory_length=600):
        start_time = time.time()
        env = self.env
        env.reset()

        # Create transition dict
        transitions = {}

        buffer_length = 0
        first_iteration = True

        rewards = []
        trajectory_lengths = []

        buffer_filled = False

        while not buffer_filled:
            if first_iteration:
                for brain in env.behavior_specs:
                    decision_steps, terminal_steps = env.get_steps(brain)
                    num_agents = len(decision_steps)
                    if (num_agents == 0):
                        env.reset()
                        decision_steps, terminal_steps = env.get_steps(brain)
                        num_agents = len(decision_steps)
                    actions = np.zeros((num_agents, 1))
                    print("Brain :" + str(brain) + " with " + str(num_agents) + " agents detected.")
                    agent_ptr = [0 for _ in range(num_agents)]  # Create pointer for agent transitions

                    for i in range(num_agents):
                        transitions[i] = {'obs_buf': np.zeros([max_trajectory_length, self.obs_dim], dtype=np.float32),
                                          'n_obs_buf': np.zeros([max_trajectory_length, self.obs_dim],
                                                                dtype=np.float32),
                                          'acts_buf': np.zeros([max_trajectory_length], dtype=np.float32),
                                          'rews_buf': np.zeros([max_trajectory_length], dtype=np.float32),
                                          'done_buf': np.zeros([max_trajectory_length], dtype=np.float32)}


                    for agent_id_decisions in decision_steps:
                        init_state = decision_steps[agent_id_decisions].obs
                        init_state = flatten(init_state)
                        action = self.select_action(init_state, epsilon, self.device)
                        actions[agent_id_decisions] = action
                        transitions[agent_id_decisions]['obs_buf'][0] = init_state
                        transitions[agent_id_decisions]['acts_buf'][0] = action

                    for agent_id_terminated in terminal_steps:
                        init_state = terminal_steps[agent_id_terminated].obs
                        init_state = flatten(init_state)
                        action = self.select_action(init_state, epsilon,self.device)
                        actions[agent_id_terminated] = action
                        transitions[agent_id_terminated]['obs_buf'][0] = init_state
                        transitions[agent_id_terminated]['acts_buf'][0] = action

                obs_buf = np.zeros([buffer_size, self.obs_dim], dtype=np.float32)
                n_obs_buf = np.zeros([buffer_size, self.obs_dim], dtype=np.float32)
                acts_buf = np.zeros([buffer_size], dtype=np.float32)
                rews_buf = np.zeros([buffer_size], dtype=np.float32)
                done_buf = np.zeros([buffer_size], dtype=np.float32)

            # Create Action vector
            action_vector = [self.flattener.lookup_action(int((action))) for action in actions]
            # Step environment
            next_experiences = step_env(self.env, np.array(action_vector))
            # Create action vector to store actions
            actions = np.zeros((num_agents, 1))

            for agent_id in next_experiences:
                if buffer_filled:
                    break
                reward = next_experiences[agent_id][1]  # Reward
                next_obs = flatten(next_experiences[agent_id][0])  # Next_obs
                done = next_experiences[agent_id][2]  # Done
                # Store trajectory of every Agent {Agent0:[obs,act,rew,n_obs,done,obs,act,rew,n_obs,done,.... Agent1: ....}
                transitions[agent_id]['rews_buf'][agent_ptr[agent_id]] = reward
                transitions[agent_id]['n_obs_buf'][agent_ptr[agent_id]] = next_obs
                transitions[agent_id]['done_buf'][agent_ptr[agent_id]] = done

                if not done:  # If the corresponding agent is not done yet, select and action and continue
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs
                    next_action = int(self.select_action(next_obs, epsilon,self.device))  # Action of next step
                    actions[agent_id] = next_action
                    transitions[agent_id]['acts_buf'][agent_ptr[agent_id]] = next_action
                else:  # If the corresponding agent is done, store the trajectory into obs_buf, rews_buf...
                    next_action = int(self.select_action(next_obs, epsilon,self.device))
                    actions[agent_id] = next_action
                    rewards.append(sum(transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1]))
                    trajectory_lengths.append(agent_ptr[agent_id])
                    for i in range(agent_ptr[agent_id] + 1):  # For every experiences store it in buffer
                        if (i + buffer_length >= buffer_size):
                            buffer_filled = True
                            buffer_filled += i
                            break
                        obs_buf[i + buffer_length] = transitions[agent_id]['obs_buf'][i]
                        acts_buf[i + buffer_length] = transitions[agent_id]['acts_buf'][i]
                        rews_buf[i + buffer_length] = transitions[agent_id]['rews_buf'][i]
                        n_obs_buf[i + buffer_length] = transitions[agent_id]['n_obs_buf'][i]
                        done_buf[i + buffer_length] = transitions[agent_id]['done_buf'][i]

                    if not buffer_filled:
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

        self.writer.add_scalar('Task: ' + str(self.task) + '/Cumulative Reward', np.mean(rewards), self.meta_step)
        self.writer.add_scalar('Task: ' + str(self.task) + '/Mean Episode Length', np.mean(trajectory_lengths),
                               self.meta_step)

        use_n_step = True if time_horizon > 1 else False

        acion_dim = 1
        buffer_start = time.time()
        per_dqn_buffer = PrioritizedDQNBuffer(size=buffer_length, obs_dim=self.obs_dim, action_dim=acion_dim, n_step=time_horizon, alpha=self.alpha
        )

        n_dqn_buffer = DQNBuffer(size=buffer_length,
            obs_dim=self.obs_dim, action_dim=acion_dim, n_step=time_horizon, gamma=self.gamma
        )
        for ptr in range(buffer_length):
            transition = [obs_buf[ptr], acts_buf[ptr], rews_buf[ptr], n_obs_buf[ptr], done_buf[ptr]]
            if use_n_step:
                one_step_transition = n_dqn_buffer.store(*transition)
            else:
                one_step_transition = transition
            if one_step_transition:
                per_dqn_buffer.store(*one_step_transition)
        print("Buffer storing took {}s".format(time.time()-buffer_start))
        print("Finished generating Buffer with size of: {} in {:.3f}s!".format(len(per_dqn_buffer), time.time() - start_time))
        return per_dqn_buffer, n_dqn_buffer, buffer_size

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""

        device = self.device  # for shortening the following lines
        state = torch_from_np(samples["obs"], device)
        next_state = torch_from_np(samples["next_obs"], device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch_from_np(samples["rews"].reshape(-1, 1), device)
        done = torch_from_np(samples["done"].reshape(-1, 1), device)

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
        log_p = torch.log(dist[range(len(action)), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())


    def train(self, max_steps: int, buffer_size: int,batch_size: int, time_horizon: int, learning_rate: float, gamma: float, hidden_size: int,
              hidden_layers: int, v_max: int, v_min: int, atom_size: int, update_period: int, beta: float, alpha: float,
              prior_eps: float, epsilon: float):

        self.init_network_and_optim(hidden_size=hidden_size, network_num_hidden_layers=hidden_layers, learning_rate=learning_rate,
                                    beta=beta, prior_eps=prior_eps, alpha=alpha, gamma=gamma,
                                    v_max=v_max, v_min=v_min, atom_size=atom_size, max_scheduler_steps=100000)
        self.meta_step = 0

        steps = 0
        update_counter = 1
        while steps < max_steps:
            memory, memory_n, buffer_length = self.generate_and_fill_buffer(buffer_size=buffer_size,epsilon=epsilon, time_horizon=time_horizon)
            steps += buffer_length
            print("Generated buffer of size: {}".format(buffer_length))

            ###### Update the model for every step taken ##########
            for epoch in range(10):
                loss, elementwise_loss, indices = self.calc_loss(memory, memory_n, batch_size=batch_size, n_step=time_horizon)

                print("Current Loss {} at step {}".format(loss.item(), steps))
                if steps > update_period * update_counter:
                    self._target_hard_update()
                    update_counter += 1

                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                self.learning_rate_scheduler.step()
                ### Increase beta for PER
                self.beta = self.beta + steps / max_steps * (1.0 - self.beta)

                loss_for_prior = elementwise_loss.detach().cpu().numpy()
                new_priorities = loss_for_prior + prior_eps
                memory.update_priorities(indices, new_priorities)
        self.meta_step += 1


if __name__ == '__main__':

    writer = SummaryWriter("results/dqn_1")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dqn_module = DQN_Meta_Learner(device=device, writer=writer)

    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration_parameters(time_scale=5.0)

    env_parameters_channel = EnvironmentParametersChannel()
    env_parameters_channel.set_float_parameter("seed", 5.0)
    env = UnityEnvironment(file_name="Training/Maze.app",
                           base_port=5000, timeout_wait=120,
                           no_graphics=False, seed=0, side_channels=[engine_configuration_channel, env_parameters_channel])

    ############ Hyperparameters DQN ##############

    max_steps = 1000000
    buffer_size = 2000 # Replay buffer size
    learning_rate = 0.0001 # Typical range: 0.00001 - 0.001
    batch_size = 512 # Typical range: 32-512
    network_num_hidden_layers = 2
    network_layer_size = 256
    time_horizon = 512
    gamma= 0.99

    ### DQN specific

    epsilon = 0.15
    v_max = 13
    v_min = -13
    atom_size = 51
    update_period = 5000
    beta = 0.5
    alpha = 0.7
    prior_eps = 1e-6


    dqn_module.set_env_and_detect_spaces(env, task=0)
    dqn_module.train(max_steps=max_steps, buffer_size=buffer_size, batch_size=batch_size, time_horizon=time_horizon,
                     learning_rate=learning_rate, gamma=gamma, hidden_size=network_layer_size, hidden_layers=network_layer_size,
                     v_max=v_max, v_min=v_min, atom_size=atom_size, update_period=update_period, beta=beta, alpha=alpha,
                     prior_eps=prior_eps, epsilon=epsilon)
import numpy as np
import time

import torch
import torch.optim as optim

from typing import Dict, List

from mlagents_envs.environment import UnityEnvironment

from torch.utils.tensorboard import SummaryWriter

from buffers import DQNBuffer, PrioritizedDQNBuffer
from utils import ActionFlattener, torch_from_np, init_unity_env
from env_utils import step_env
from models import DeepQNetwork


def flatten(input):
    return [item for sublist in input for item in sublist]


class Rainbow_Meta_Learner:
    """DQN Agent interacting with environment.
    """

    def __init__(
            self,
            device: str,
            writer: SummaryWriter,

    ):
        self.device = device
        self.writer = writer

        self.step = 0
        self.meta_step = 0


    def get_default_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['Algorithm'] = "RAINBOW"
        hyperparameters['enable_curiosity'] = True


        hyperparameters['max_steps'] = 1000000
        hyperparameters['learning_rate'] = 0.0001  # Typical range: 0.00001 - 0.001
        hyperparameters['batch_size'] = 512  # Typical range: 32-512
        hyperparameters['hidden_layers'] = 2
        hyperparameters['layer_size'] = 256
        hyperparameters['time_horizon'] = 64
        hyperparameters['gamma'] = 0.99
        hyperparameters['decay_lr'] = True


        hyperparameters['buffer_size'] = 5000  # Replay buffer size
        hyperparameters['epochs'] = 10  # Buffer_length/epochs = num_updates
        hyperparameters['epsilon'] = 0.15  # Percentage to explore epsilon = 0 -> Decaying after half training
        hyperparameters['v_max'] = 13  # Maximum Value of Reward
        hyperparameters['v_min'] = -13  # Minimum Value of Reward
        hyperparameters['atom_size'] = 51  # Atom Size for categorical DQN
        hyperparameters['update_period'] = 50  # Period after which Target Network gets updated
        hyperparameters['beta'] = 0.6  # How much to use importance sampling
        hyperparameters['alpha'] = 0.2  # How much to use prioritization
        hyperparameters['prior_eps'] = 1e-6  # Guarantee to use all experiences

        return hyperparameters

    def init_networks_and_optimizers(self, hyperparameters: dict):
        hidden_size = hyperparameters['layer_size']
        network_num_hidden_layers = hyperparameters['hidden_layers']
        max_scheduler_steps = hyperparameters['max_steps']
        atom_size = hyperparameters['atom_size']
        v_max = hyperparameters['v_max']
        v_min = hyperparameters['v_min']
        prior_eps = hyperparameters['prior_eps']
        alpha = hyperparameters['alpha']
        beta = hyperparameters['beta']
        gamma = hyperparameters['gamma']
        learning_rate = hyperparameters['learning_rate']

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

        linear_schedule = lambda epoch: max((1 - epoch / max_scheduler_steps), 1e-6)

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
        print("For Rainbow the Action space gets reshaped to: {}".format(str(action_space)))

        self.env.reset()

    def get_networks_and_parameters(self):
        networks_and_parameters = {}
        networks_and_parameters['networks'] = [self.dqn, self.dqn_target]
        networks_and_parameters['parameters'] = []
        return networks_and_parameters

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

        cumulative_rewards = []
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
                        action = self.select_action(init_state, epsilon, self.device)
                        actions[agent_id_terminated] = action
                        transitions[agent_id_terminated]['obs_buf'][0] = init_state
                        transitions[agent_id_terminated]['acts_buf'][0] = action

                obs_buf = np.zeros([buffer_size, self.obs_dim], dtype=np.float32)
                n_obs_buf = np.zeros([buffer_size, self.obs_dim], dtype=np.float32)
                acts_buf = np.zeros([buffer_size], dtype=np.float32)
                rews_buf = np.zeros([buffer_size], dtype=np.float32)
                done_buf = np.zeros([buffer_size], dtype=np.float32)

            # Create Action vector
            action_vector = [self.flattener.lookup_action(int(action)) for action in actions]
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
                    cumulative_rewards.append(sum(transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1]))
                    trajectory_lengths.append(agent_ptr[agent_id])
                    for i in range(agent_ptr[agent_id] + 1):  # For every experiences store it in buffer
                        if (agent_ptr[agent_id] + buffer_length >= buffer_size):
                            buffer_filled = True
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

        use_n_step = True if time_horizon > 1 else False

        acion_dim = 1
        buffer_time = time.time()
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

        self.writer.add_scalars('task_' + str(self.task) + r"\Rainbow Cumulative Reward",
                                {r'\meta_step_' + str(self.meta_step): np.mean(cumulative_rewards)}, self.step)
        self.writer.add_scalars('task_' + str(self.task) + r'\Rainbow Mean Episode Length',
                                {r'\meta_step_' + str(self.meta_step): np.mean(trajectory_lengths)}, self.step)

        print("Mean Cumulative Reward: {} at step {}".format(np.mean(cumulative_rewards), self.step))
        print("Mean Episode Lengths: {} at step {}".format(np.mean(trajectory_lengths), self.step))

        print("Finished generating Buffer with size of: {} in {:.3f}s! Storing took {:.3f}s".format(len(per_dqn_buffer), time.time() - start_time, time.time() - buffer_time))
        return per_dqn_buffer, n_dqn_buffer, np.mean(cumulative_rewards), np.mean(trajectory_lengths)

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

    def close_env(self):
        self.env.close()

    def train(self, hyperparameters: dict):
        max_steps = hyperparameters['max_steps']
        batch_size = hyperparameters['batch_size']
        time_horizon = hyperparameters['time_horizon']
        buffer_size = hyperparameters['buffer_size']

        epochs = hyperparameters['epochs']

        update_period = hyperparameters['update_period']
        prior_eps = hyperparameters['prior_eps']
        epsilon = hyperparameters['epsilon']

        update_counter = 1

        mean_rewards = []
        mean_episode_lengths = []

        while self.step < max_steps:
            memory, memory_n, mean_reward, mean_episode_length = self.generate_and_fill_buffer(buffer_size=buffer_size,epsilon=epsilon, time_horizon=time_horizon)
            mean_rewards.append(mean_reward)
            mean_episode_lengths.append(mean_episode_length)
            self.step += len(memory)

            ###### Update the model for every step taken ##########
            losses = []
            for epoch in range(len(memory) // epochs):
                loss, elementwise_loss, indices = self.calc_loss(memory, memory_n, batch_size=batch_size, n_step=time_horizon)

                if update_counter % update_period == 0:
                    self._target_hard_update()
                    update_counter += 1

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 10.0)
                self.optimizer.step()
                losses.append(loss.detach().cpu().numpy())

                loss_for_prior = elementwise_loss.detach().cpu().numpy()
                new_priorities = loss_for_prior + prior_eps
                memory.update_priorities(indices, new_priorities)

                ### Increase beta for PER
            self.beta = self.beta + self.step / max_steps * (1.0 - self.beta)
            epsilon = max((1 - self.step * 2/max_steps) * epsilon , 0)
            if(hyperparameters['decay_lr']):
                self.learning_rate_scheduler.step(epoch=self.step)

            self.writer.add_scalars('task_' + str(self.task) + r"\Rainbow Value Loss",
                                    {r'\meta_step_' + str(self.meta_step): np.mean(losses)}, self.step)

            print("Current Loss: {:.3f} at step {} with learning rate of: {:.6f}".format(np.mean(losses), self.step, self.optimizer.param_groups[0]['lr']))
            print("Current beta: {:.3f}\nCurrent eps: {:.3f}".format(self.beta, epsilon))

        return np.mean(mean_rewards), np.mean(mean_episode_lengths)


if __name__ == '__main__':

    run_id = "results/dqn_0"

    writer = SummaryWriter(run_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    dqn_module = Rainbow_Meta_Learner(device=device, writer=writer)

    env = init_unity_env("mMaze/RLProject.exe", maze_rows=3, maze_cols=3, maze_seed=0, random_agent=0, random_target=0, agent_x=0, agent_z=0, target_x=2, target_z=2)

    ############ Hyperparameters DQN ##############
    training_parameters = dqn_module.get_default_hyperparameters()

    training_parameters['Algorithm'] = "RAINBOW"
    training_parameters['run_id'] = run_id
    training_parameters['enable_curiosity'] = True


    training_parameters['max_steps'] = 1000000
    training_parameters['buffer_size'] = 5000 # Replay buffer size
    training_parameters['learning_rate'] = 0.0001 # Typical range: 0.00001 - 0.001
    training_parameters['batch_size'] = 512 # Typical range: 32-512
    training_parameters['hidden_layers'] = 2
    training_parameters['layer_size'] = 256
    training_parameters['time_horizon'] = 64
    training_parameters['gamma'] = 0.99
    training_parameters['decay_lr'] = True

    ### DQN specific

    training_parameters['epochs'] = 10             # Epochs of Optimizer steps of a finished buffer len(Finished Buffer) / Epochs -> greater means less updates
    training_parameters['epsilon'] = 0.15          # Exploration vs Exploitation Percentage to explore. epsilon = 0 -> No exploring is decaying to zero after half of training steps
    training_parameters['v_max'] = 13              # Maximum Value of Reward
    training_parameters['v_min'] = -13             # Minimum Value of Reward
    training_parameters['atom_size'] = 51          # Atom Size for categorical DQN
    training_parameters['update_period'] = 50      # Period after which Target Network gets updated
    training_parameters['beta'] = 0.6              # How much to use importance sampling
    training_parameters['alpha'] = 0.2             # How much to use prioritization
    training_parameters['prior_eps'] = 1e-6        # Guarantee to use all experiences


    writer.add_text("training_parameters", str(training_parameters))
    print("Started run with following training_parameters:")
    for key in training_parameters:
        print("{:<25s} {:<20s}".format(key, str(training_parameters[key])))

    dqn_module.set_env_and_detect_spaces(env, task=0)

    dqn_module.init_networks_and_optimizers(training_parameters)
    dqn_module.train(training_parameters)
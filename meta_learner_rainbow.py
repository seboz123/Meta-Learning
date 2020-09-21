import numpy as np
import time

import torch
import torch.optim as optim

from typing import Dict, List

from curiosity_module import CuriosityModule
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
            is_meta_learning: bool

    ):
        self.device = device
        self.writer = writer

        self.step = 0
        self.meta_step = 0
        self.is_meta_learning = is_meta_learning
        self.enable_curiosity = False
        self.curiosity = None

    def get_default_hyperparameters(self):
        hyperparameters = {}
        hyperparameters['Algorithm'] = "RAINBOW"
        hyperparameters['logging_period'] = 2000

        hyperparameters['enable_curiosity'] = True
        hyperparameters['curiosity_lambda'] = 10 # Weight factor of extrinsic reward. 0.1 -> 10*Curiosity
        hyperparameters['curiosity_beta'] = 0.2 # Factor for using more of forward loss or more of inverse loss
        hyperparameters['curiosity_enc_size'] = 32 # Encoding size of curiosity_module
        hyperparameters['curiosity_layers'] = 2 # Layers of Curiosity Modules
        hyperparameters['curiosity_units'] = 128 # Number of hidden units for curiosity modules


        hyperparameters['max_steps'] = 1000000
        hyperparameters['learning_rate'] = 0.0003  # Typical range: 0.00001 - 0.001
        hyperparameters['batch_size'] = 512  # Typical range: 32-512
        hyperparameters['hidden_layers'] = 2
        hyperparameters['layer_size'] = 256
        hyperparameters['time_horizon'] = 64
        hyperparameters['gamma'] = 0.99
        hyperparameters['decay_lr'] = True


        hyperparameters['buffer_size'] = 5000  # Replay buffer size
        hyperparameters['steps_per_update'] = 1


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

        self.dqn = DeepQNetwork(in_dim=self.obs_dim, out_dim=self.action_dim.n, hidden_size=hidden_size, num_hidden_layers=network_num_hidden_layers, atom_size=self.atom_size, support=self.support,
                                enable_curiosity=hyperparameters['enable_curiosity']).to(self.device)
        self.dqn_target = DeepQNetwork(in_dim=self.obs_dim, out_dim=self.action_dim.n, hidden_size=hidden_size, num_hidden_layers=network_num_hidden_layers, atom_size=self.atom_size, support=self.support,
                                       enable_curiosity=hyperparameters['enable_curiosity']).to(self.device)

        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)

        linear_schedule = lambda epoch: max((1 - epoch / max_scheduler_steps), 1e-6)

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_schedule)

        if hyperparameters['enable_curiosity']:
            self.enable_curiosity = True
            self.curiosity = CuriosityModule(obs_size=self.obs_dim, enc_size=hyperparameters['curiosity_enc_size'],
                                             hidden_layers=hyperparameters['curiosity_layers'],
                                             hidden_size=hyperparameters['curiosity_units'], learning_rate=0.003,
                                             device=self.device, action_shape=self.branches)
            print("Enabled curiosity module")

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
        self.branches = branches

        print("Space detected successfully.")
        print("Observation space detected as {}\nAction space detected as: {}".format(result, branches))
        print("For Rainbow the Action space gets reshaped to: {}".format(str(action_space)))

        self.env.reset()

    def get_networks_and_parameters(self):
        networks_and_parameters = {}
        networks_and_parameters['networks'] = [self.dqn, self.dqn_target]
        if self.enable_curiosity:
            for network in self.curiosity.get_networks():
                networks_and_parameters['networks'].append(network)
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

        if self.enable_curiosity:
            forward_loss, inverse_loss = self.curiosity.calc_loss_rainbow(samples, self.flattener, self.branches)
        else:
            forward_loss, inverse_loss = None, None

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

        return loss, elementwise_loss, indices, forward_loss, inverse_loss

    def update_priorities(self, elementwise_loss, indices, memory):
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

    def generate_and_fill_buffer(self, replay_buffer, n_replay_buffer, epsilon, max_trajectory_length=600):
        env = self.env

        # Create transition dict
        transitions = {}

        first_iteration = True

        cumulative_rewards = []
        trajectory_lengths = []

        buffer_finished = False
        steps_taken = 0

        while not buffer_finished:
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

            # Create Action vector
            action_vector = np.array([self.flattener.lookup_action(int(action)) for action in actions])
            # Step environment
            next_experiences = step_env(self.env, action_vector)
            # Create action vector to store actions
            actions = np.zeros((num_agents, 1))

            for agent_id in next_experiences:
                reward = next_experiences[agent_id][1]  # Reward
                next_obs = flatten(next_experiences[agent_id][0])  # Next_obs
                done = next_experiences[agent_id][2]  # Done
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

                    obs_buf = transitions[agent_id]['obs_buf'][:agent_ptr[agent_id] +1]
                    acts_buf = transitions[agent_id]['acts_buf'][:agent_ptr[agent_id] +1]
                    rews_buf = transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] +1]
                    n_obs_buf = transitions[agent_id]['n_obs_buf'][:agent_ptr[agent_id] +1]
                    done_buf = transitions[agent_id]['done_buf'][:agent_ptr[agent_id] +1]

                    for i in range(agent_ptr[agent_id] + 1):  # For every experiences store it in buffer
                        transition = [obs_buf[i], acts_buf[i], rews_buf[i], n_obs_buf[i], done_buf[i]]
                        one_step_transition = n_replay_buffer.store(*transition)
                        if one_step_transition:
                            replay_buffer.store(*one_step_transition)

                    steps_taken += agent_ptr[agent_id] + 1

                    if steps_taken > 0.2 * replay_buffer.max_size and len(replay_buffer) == replay_buffer.max_size:
                        print("Finished Buffer")
                        buffer_finished = True
                        break

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



        print("Mean Cumulative Reward: {} at step {}".format(np.mean(cumulative_rewards), self.step))
        print("Mean Episode Lengths: {} at step {}".format(np.mean(trajectory_lengths), self.step))

        return steps_taken, np.mean(cumulative_rewards), np.mean(trajectory_lengths)

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
        self.writer.add_text("training_parameters", str(hyperparameters))
        print("Started run with following hyperparameters:")
        for key in hyperparameters:
            print("{:<25s} {:<20s}".format(key, str(hyperparameters[key])))

        max_steps = hyperparameters['max_steps']
        batch_size = hyperparameters['batch_size']
        time_horizon = hyperparameters['time_horizon']
        buffer_size = hyperparameters['buffer_size']

        gamma = hyperparameters['gamma']
        alpha = hyperparameters['alpha']
        update_period = hyperparameters['update_period']
        prior_eps = hyperparameters['prior_eps']
        epsilon = hyperparameters['epsilon']

        steps_per_update = hyperparameters['steps_per_update']

        mean_rewards = []
        mean_episode_lengths = []
        losses = []

        logging_steps = 0

        buffer = PrioritizedDQNBuffer(size=buffer_size, obs_dim=self.obs_dim, action_dim=1, n_step=time_horizon,
                                      alpha=alpha)
        buffer_n = DQNBuffer(size=buffer_size, obs_dim=self.obs_dim, action_dim=1, n_step=time_horizon, gamma=gamma)

        while self.step < max_steps:

            steps_taken, mean_reward, mean_episode_length = self.generate_and_fill_buffer(replay_buffer=buffer, n_replay_buffer=buffer_n, epsilon=epsilon)
            print("Finished Buffer with {} steps taken".format(steps_taken))

            mean_rewards.append(mean_reward)
            mean_episode_lengths.append(mean_episode_length)
            self.step += steps_taken

            if self.step > logging_steps * hyperparameters['logging_period']:
                if self.is_meta_learning:
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(self.meta_step) + r"/Rainbow Cumulative Reward",
                        np.mean(mean_rewards), self.step)
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(
                            self.meta_step) + r"/Rainbow Mean Episode Lengths",
                        np.mean(mean_episode_lengths), self.step)
                else:
                    self.writer.add_scalar(
                        "Environment/Cumulative Reward", np.mean(mean_rewards), self.step)
                    self.writer.add_scalar("Environment/Episode Length",
                                           np.mean(mean_episode_lengths), self.step)
                mean_rewards.clear()
                mean_episode_lengths.clear()

            ######### Update the model for every step taken ##########

            frame_start = time.time()
            update_steps = 0
            while update_steps * steps_per_update < steps_taken:
                print("Current update step: {} of {} steps".format(update_steps, steps_taken))
                loss, elementwise_loss, indices, f_loss, i_loss = self.calc_loss(buffer, buffer_n, batch_size=batch_size, n_step=time_horizon)

                if self.enable_curiosity:
                    curiosity_loss = hyperparameters['curiosity_lambda'] * (
                            hyperparameters['curiosity_beta'] * f_loss + (
                                1 - hyperparameters['curiosity_beta']) * i_loss)
                    self.curiosity.optimizer.zero_grad()
                    curiosity_loss.backward()
                    for network in self.curiosity.get_networks():
                        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                    self.curiosity.optimizer.step()
                    self.curiosity.learning_rate_scheduler.step()

                if update_steps % update_period == 0:
                    self._target_hard_update()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 10.0)
                self.optimizer.step()
                losses.append(loss.detach().cpu().numpy())

                loss_for_prior = elementwise_loss.detach().cpu().numpy()
                new_priorities = loss_for_prior + prior_eps
                buffer.update_priorities(indices, new_priorities)

                update_steps += 1
            frame_end = time.time()
            print("Current Update rate = {} updates per second".format(steps_taken / (frame_end - frame_start)))
            print("Current Loss: {:.3f} at step {} with learning rate of: {:.6f}".format(np.mean(losses), self.step, self.optimizer.param_groups[0]['lr']))
            print("Current beta: {:.3f}\nCurrent eps: {:.3f}".format(self.beta, epsilon))

            ####### Increase beta for PER ########
            self.beta = self.beta + self.step / max_steps * (1.0 - self.beta)
            ####### Decrease epsilon ########
            epsilon = max((1 - self.step * 2 / max_steps) * epsilon, 0)

            if(hyperparameters['decay_lr']):
                self.learning_rate_scheduler.step(epoch=self.step)

            if self.step > logging_steps * hyperparameters['logging_period']:
                if self.is_meta_learning:
                    self.writer.add_scalar('Meta Learning Parameters/Learning Rate', self.learning_rate_scheduler.get_lr()[0], self.meta_step)
                    self.writer.add_scalar(
                        'Task: ' + str(self.task) + r"/Meta Step: " + str(self.meta_step) + r"/Rainbow Value Loss",
                        np.mean(losses), self.step)
                else:
                    self.writer.add_scalar('Policy/Learning Rate', self.learning_rate_scheduler.get_lr()[0], self.step)
                    self.writer.add_scalar("Losses/Value Loss",
                                           np.mean(losses), self.step)
                losses.clear()
                logging_steps += 1

if __name__ == '__main__':

    run_id = "results/dqn_0"

    writer = SummaryWriter(run_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    dqn_module = Rainbow_Meta_Learner(device=device, writer=writer, is_meta_learning=False)

    env = init_unity_env("mMaze.app", maze_rows=3, maze_cols=3, maze_seed=0, random_agent=0, random_target=0, agent_x=0, agent_z=0, target_x=0, target_z=1)

    ############ Hyperparameters DQN ##############
    training_parameters = dqn_module.get_default_hyperparameters()

    training_parameters['Algorithm'] = "RAINBOW"
    training_parameters['run_id'] = run_id
    training_parameters['enable_curiosity'] = False


    training_parameters['max_steps'] = 1000000
    training_parameters['buffer_size'] = 2000 # Replay buffer size
    training_parameters['steps_per_update'] = 1
    training_parameters['learning_rate'] = 0.0003 # Typical range: 0.00001 - 0.001
    training_parameters['batch_size'] = 512 # Typical range: 32-512
    training_parameters['hidden_layers'] = 2
    training_parameters['layer_size'] = 256
    training_parameters['time_horizon'] = 128
    training_parameters['gamma'] = 0.99
    training_parameters['decay_lr'] = True

    ### DQN specific

    training_parameters['epsilon'] = 0.05          # Exploration vs Exploitation Percentage to explore. epsilon = 0 -> No exploring is decaying to zero after half of training steps
    training_parameters['v_max'] = 13              # Maximum Value of Reward
    training_parameters['v_min'] = -13             # Minimum Value of Reward
    training_parameters['atom_size'] = 51          # Atom Size for categorical DQN
    training_parameters['update_period'] = 50      # Period after which Target Network gets updated
    training_parameters['beta'] = 0.6              # How much to use importance sampling
    training_parameters['alpha'] = 0.2             # How much to use prioritization
    training_parameters['prior_eps'] = 1e-6        # Guarantee to use all experiences

    dqn_module.set_env_and_detect_spaces(env, task=0)

    dqn_module.init_networks_and_optimizers(training_parameters)
    dqn_module.train(training_parameters)
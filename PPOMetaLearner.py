import numpy as np
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from typing import Dict, List

from mlagents_envs.environment import UnityEnvironment

from torch.utils.tensorboard import SummaryWriter
from ActionFlattener import ActionFlattener
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
        self.out_layers = [nn.Sequential(nn.Linear(hidden_size, shape).to(self.device), nn.Softmax()) for shape in action_dim]

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def act(self, state: np.ndarray):
        state = torch.from_numpy(state).float().to(self.device)
        action_hidden_state = self.action_layer(state)
        action_prob = torch.stack([layer(action_hidden_state) for layer in self.out_layers])
        dist = Categorical(action_prob)
        action = dist.sample()
        return action.detach().cpu().numpy()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


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
                if(len(shape) == 1):
                    result += shape[0]
        self.obs_space = result
        self.action_space = branches
        self.env.reset()

    # def select_action(self, state: np.ndarray) -> np.ndarray:
    #     """Select an action from the input state."""
    #     # NoisyNet: no epsilon greedy action selection
    #     selected_action = self.dqn(
    #         torch.FloatTensor(state).to(self.device)
    #     ).argmax()
    #     selected_action = selected_action.detach().cpu().numpy()
    #
    #     return selected_action

    def init_network_and_optim(self, obs_dim, action_dim, hidden_size, learning_rate: float = 0.0003):

        self.actor_critic = ActorCritic(self.device,
            obs_dim, action_dim, hidden_size
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)


    def step_env(self, env: UnityEnvironment, actions: np.array):
        agents_transitions = {}
        for brain in env.behavior_specs:
            actions = np.resize(actions,(len(env.get_steps(brain)[0]), len(env.behavior_specs[brain].discrete_action_branches)))
            self.env.set_actions(brain, actions)
            self.env.step()
            decision_steps, terminal_steps = env.get_steps(brain)

            for agent_id_decisions in decision_steps:
                agents_transitions[agent_id_decisions] = [decision_steps[agent_id_decisions].obs, decision_steps[agent_id_decisions].reward, False]

            for agent_id_terminated in terminal_steps:
                agents_transitions[agent_id_terminated] = [terminal_steps[agent_id_terminated].obs,
                                                               terminal_steps[agent_id_terminated].reward, True]

        return agents_transitions

    def create_replay_buffer(self, trajectories: List, n_step: int, memory_size: int, batch_size: int):
        use_n_step = True if n_step > 1 else False
        obs_buf, acts_buf, rews_buf, n_obs_buf, done_buf = trajectories


        memory_n = ReplayBuffer(
            self.obs_dim, memory_size, batch_size, n_step=n_step, gamma=self.gamma
        )

        for ptr in range(len(acts_buf)):
            transition = []
            transition.append(obs_buf[ptr])
            transition.append(acts_buf[ptr])
            transition.append(rews_buf[ptr])
            transition.append(n_obs_buf[ptr])
            transition.append(done_buf[ptr])
            if use_n_step:
                one_step_transition = memory_n.store(*transition)
            else:
                one_step_transition = transition
            if one_step_transition:
                memory.store(*one_step_transition)

        return memory, memory_n

    def generate_and_fill_buffer(self, buffer_size, n_step, batch_size, task, max_trajectory_length = 600, gamma: float = 0.99):
        if(buffer_size <= batch_size):
            raise ValueError
        start_time = time.time()
        env = self.env
        env.reset()

        # Create transition dict
        experiences = {}
        transitions = {}

        for brain in env.behavior_specs:
            decision_steps, terminal_steps = env.get_steps(brain)
            num_agents = len(decision_steps)
            actions = np.zeros((24,len(self.action_space)))
            print("Brain :" + str(brain) + " with " + str(num_agents) + " agents detected.")
            agent_ptr = [0 for _ in range(num_agents)]  # Create pointer for agent transitions

            for i in range(num_agents):
                experiences[i] = [[], [], 0, [], False]
                transitions[i] = {'obs_buf': np.zeros([max_trajectory_length, self.obs_space], dtype=np.float32),
                                  'n_obs_buf': np.zeros([max_trajectory_length, self.obs_space], dtype=np.float32),
                                  'acts_buf': np.zeros([max_trajectory_length, len(self.action_space)], dtype=np.float32),
                                  'rews_buf': np.zeros([max_trajectory_length], dtype=np.float32),
                                  'done_buf': np.zeros([max_trajectory_length], dtype=np.float32)}
                # {0: {obs_buf: ,n_obs_buf: , ....}, 1: {obs_buf: ,n_obs_buf: , ....}, 2: {obs_buf: ,n_obs_buf: , ....}}

                # step_counter[i] = 0

            for agent_id_decisions in decision_steps:
                init_state = decision_steps[agent_id_decisions].obs
                init_state = flatten(init_state)
                action = self.actor_critic.act(np.array(init_state))
                actions[agent_id_decisions] = action
                experiences[agent_id_decisions][0] = init_state
                transitions[agent_id_decisions]['obs_buf'][0] = init_state
                transitions[agent_id_decisions]['acts_buf'][0] = action

            for agent_id_terminated in terminal_steps:
                init_state = terminal_steps[agent_id_terminated].obs
                init_state = flatten(init_state)
                action = self.actor_critic.act(np.array(init_state))

                # action = self.select_action(init_state)
                actions[agent_id_terminated] = action
                experiences[agent_id_terminated][0] = init_state
                transitions[agent_id_decisions]['obs_buf'][0] = init_state
                transitions[agent_id_decisions]['acts_buf'][0] = action

        obs_buf = np.zeros([buffer_size, self.obs_space], dtype=np.float32)
        n_obs_buf = np.zeros([buffer_size, self.obs_space], dtype=np.float32)
        acts_buf = np.zeros([buffer_size, len(self.action_space)], dtype=np.float32)
        rews_buf = np.zeros([buffer_size], dtype=np.float32)
        done_buf = np.zeros([buffer_size], dtype=np.float32)

        buffer_length = 0
        first_iteration = True

        rewards = []
        trajectory_lengths = []

        while buffer_length < buffer_size:
            if not first_iteration:
                for agent in experiences:
                    # Set Observation and Action to the last next_obs and the selected action for the observation
                    experiences[agent][0] = experiences[agent][3]  # obs = next_obs
                    experiences[agent][1] = actions[int(agent)]  # action = action_vector[]

            # Create Action vector
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

                if not done:  # If the corresponding agent is not done yet, select and action and continue
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs

                    next_action = self.actor_critic.act(np.array(next_obs))

                    actions[agent_id] = next_action
                    # print(agent_ptr[agent_id])
                    # print(transitions[agent_id]['acts_buf'])
                    # print(next_action)
                    # print(transitions[agent_id]['acts_buf'][agent_ptr[agent_id]])
                    transitions[agent_id]['acts_buf'][agent_ptr[agent_id]] = next_action
                else:  # If the corresponding agent is done, store the trajectory into obs_buf, rews_buf...
                    next_action = self.actor_critic.act(np.array(next_obs))
                    actions[agent_id] = next_action
                    rewards.append(sum(transitions[agent_id]['rews_buf'][:agent_ptr[agent_id] + 1]))
                    trajectory_lengths.append(agent_ptr[agent_id])
                    for i in range(agent_ptr[agent_id] + 1): # For every experiences store it in buffer
                        if(i + buffer_length >= buffer_size):
                            break
                        obs_buf[i + buffer_length] = transitions[agent_id]['obs_buf'][i]
                        acts_buf[i + buffer_length] = transitions[agent_id]['acts_buf'][i]
                        rews_buf[i + buffer_length] = transitions[agent_id]['rews_buf'][i]
                        n_obs_buf[i + buffer_length] = transitions[agent_id]['n_obs_buf'][i]
                        done_buf[i + buffer_length] = transitions[agent_id]['done_buf'][i]
                    buffer_length += agent_ptr[agent_id] + 1

                    transitions[agent_id]['obs_buf'] = np.zeros(
                        [max_trajectory_length, self.obs_space], dtype=np.float32)
                    transitions[agent_id]['acts_buf'] = np.zeros([max_trajectory_length],
                                                                 dtype=np.float32)
                    transitions[agent_id]['rews_buf'] = np.zeros([max_trajectory_length],
                                                                 dtype=np.float32)
                    transitions[agent_id]['n_obs_buf'] = np.zeros(
                        [max_trajectory_length, self.obs_space], dtype=np.float32)
                    transitions[agent_id]['done_buf'] = np.zeros([max_trajectory_length],
                                                                 dtype=np.float32)
                    agent_ptr[agent_id] = 0
            first_iteration = False

        self.writer.add_scalar('Task: ' + str(task) + '/Cumulative Reward', np.mean(rewards), self.meta_step)
        self.writer.add_scalar('Task: ' + str(task) + '/Mean Episode Length', np.mean(trajectory_lengths), self.meta_step)

        use_n_step = True if n_step > 1 else False

        memory_n = ReplayBuffer(
            self.obs_space, buffer_size, batch_size, n_step=n_step, gamma=gamma
        )

        for ptr in range(len(acts_buf)):
            transition = []
            transition.append(obs_buf[ptr])
            transition.append(acts_buf[ptr])
            transition.append(rews_buf[ptr])
            transition.append(n_obs_buf[ptr])
            transition.append(done_buf[ptr])
            one_step_transition = memory_n.store(*transition)

        print("Finished generating Buffer with of size: {} in {:.3f}s!".format(len(memory_n), time.time() - start_time))
        return memory_n, buffer_size


    def generate_trajectories(self, env: UnityEnvironment, num_trajectories: int, max_trajectory_length: int = 600) -> tuple:
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
            print("Brain :" + str(brain) +" with " + str(num_agents) + " agents detected.")
            agent_ptr = [0 for _ in range(num_agents)] # Create pointer for agent transitions

            for i in range(num_agents):
                experiences[i] = [[], 0, 0, [], False]
                transitions[i] = {'obs_buf' : np.zeros([max_trajectory_length, self.obs_space], dtype=np.float32), 'n_obs_buf' : np.zeros([max_trajectory_length, self.obs_space], dtype=np.float32),
                    'acts_buf' : np.zeros([max_trajectory_length], dtype=np.float32), 'rews_buf' : np.zeros([max_trajectory_length], dtype=np.float32), 'done_buf' : np.zeros([max_trajectory_length], dtype=np.float32) }


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



        obs_buf = np.zeros([num_trajectories*max_trajectory_length, self.obs_space], dtype=np.float32)
        n_obs_buf = np.zeros([num_trajectories*max_trajectory_length, self.obs_space], dtype=np.float32)
        acts_buf = np.zeros([num_trajectories*max_trajectory_length], dtype=np.float32)
        rews_buf = np.zeros([num_trajectories*max_trajectory_length], dtype=np.float32)
        done_buf = np.zeros(num_trajectories*max_trajectory_length, dtype=np.float32)

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
                # Store trajectory of every Agent {Agent0:[obs,act,rew,n_obs,done,obs,act,rew,n_obs,done,.... Agent1: ....}
                transitions[agent_id]['rews_buf'][agent_ptr[agent_id]] = reward
                transitions[agent_id]['n_obs_buf'][agent_ptr[agent_id]] = next_obs
                transitions[agent_id]['done_buf'][agent_ptr[agent_id]] = done


                if not done:  # If the corresponding agent is not done yet, select and action and continue
                    agent_ptr[agent_id] += 1
                    transitions[agent_id]['obs_buf'][agent_ptr[agent_id]] = next_obs
                    next_action = int(self.select_action(next_obs)) # Action of next step
                    actions[agent_id] = next_action
                    transitions[agent_id]['acts_buf'][agent_ptr[agent_id]] = next_action
                else:                               # If the corresponding agent is done, store the trajectory into obs_buf, rews_buf...
                    finished_trajectories += 1 # 1 trajectory is finished
                    next_action = int(self.select_action(next_obs))
                    actions[agent_id] = next_action
                    for i in range(agent_ptr[agent_id]+1):
                        obs_buf[i+trajectory_length] = transitions[agent_id]['obs_buf'][i]
                        acts_buf[i+trajectory_length] = transitions[agent_id]['acts_buf'][i]
                        rews_buf[i+trajectory_length] = transitions[agent_id]['rews_buf'][i]
                        n_obs_buf[i+trajectory_length] = transitions[agent_id]['n_obs_buf'][i]
                        done_buf[i+trajectory_length] = transitions[agent_id]['done_buf'][i]
                    trajectory_length += agent_ptr[agent_id]+1

                    transitions[agent_id]['obs_buf'] = np.zeros([max_trajectory_length, self.obs_space], dtype=np.float32)
                    transitions[agent_id]['acts_buf'] = np.zeros([max_trajectory_length], dtype=np.float32)
                    transitions[agent_id]['rews_buf'] = np.zeros([max_trajectory_length], dtype=np.float32)
                    transitions[agent_id]['n_obs_buf'] = np.zeros([max_trajectory_length, self.obs_space], dtype=np.float32)
                    transitions[agent_id]['done_buf'] = np.zeros([max_trajectory_length], dtype=np.float32)
                    agent_ptr[agent_id] = 0
            first_iteration = False

        obs_buf = obs_buf[:trajectory_length]
        acts_buf = acts_buf[:trajectory_length]
        rews_buf = rews_buf[:trajectory_length]
        n_obs_buf = n_obs_buf[:trajectory_length]
        done_buf = done_buf[:trajectory_length]
        print("Finished generating {} trajectories with {} experiences in {:.3f}s!".format(finished_trajectories,trajectory_length, time.time()-start_time))

        return (obs_buf, acts_buf, rews_buf, n_obs_buf, done_buf) , trajectory_length


writer = SummaryWriter("C:/Users/Sebastian/Desktop/RLUnity/Training/results" + r"/Meta_Learning3")


ppo_module = PPO_Meta_Learner('cuda', writer=writer)


engine_configuration_channel = EngineConfigurationChannel()
env_parameters_channel = EnvironmentParametersChannel()
env_parameters_channel.set_float_parameter("seed", 5.0)
env = UnityEnvironment(file_name="C:/Users/Sebastian/Desktop/RLUnity/Training/mMaze/RLProject",
                             base_port=5000, timeout_wait=120,
                             no_graphics=False, seed=0, side_channels=[engine_configuration_channel, env_parameters_channel])

ppo_module.set_environment(env)
ppo_module.init_network_and_optim(ppo_module.obs_space,ppo_module.action_space, 3)
ppo_module.generate_and_fill_buffer(1000, 128, 128, 0, 600)
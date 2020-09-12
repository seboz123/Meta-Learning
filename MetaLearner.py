import os
from typing import Dict, List, Tuple
from copy import deepcopy

import time


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from utils import ActionFlattener
from curiosity_module import CuriosityModule
from meta_learner_dqn import DQN_Meta_Learner

# Seed Torch Function
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

# Get inital configs
class MetaLearner():
    def __init__(self, writer: SummaryWriter):



    def meta_learn(self, algorithm: str, num_meta_updates: int, buffer_size: int, max_trajectory_length: int, n_step: int, batch_size: int, epochs: int):
        meta_start = time.time()
        start_time = time.localtime()
        start_time = time.strftime("%H:%M:%S", start_time)

        ("Meta Learning started at: " + start_time)
        print("Starting Meta Learning with following Hyperparameters:")
        print("Meta Steps: " + str(num_meta_updates))
        print("Buffer size: " + str(buffer_size))
        print("Batch size: " + str(batch_size))
        print("Time horizon: " + str(n_step))
        print("Meta Learning-rate: " + str(self.meta_lr))

        training_steps = []
        for task in self.tasks:
            training_steps.append(0)

        for meta_step in range(num_meta_updates):
            print("Meta step: {} of {}".format(meta_step, num_meta_updates))
            # Sample task
            for task in self.tasks:
                task.reset()
            sample_task = lambda: np.random.randint(0,len(self.tasks))
            task_number = sample_task()

            # Set environment according to class
            self.policy.set_environment(self.tasks[task_number])

            # Get inital network parameters
            theta_before = [deepcopy(self.policy.dqn.state_dict()), deepcopy(self.curiosity.forwardModel.state_dict()), deepcopy(self.curiosity.inverseModel.state_dict())]

            # Generate trajectories
            memory, memory_n, buffer_size = self.policy.generate_and_fill_buffer(buffer_size=buffer_size, max_trajectory_length=max_trajectory_length,task=task_number, n_step=n_step, batch_size=batch_size)
            training_steps[task_number] += buffer_size
            # Inner loop updates
            inner_loop_iters = buffer_size // batch_size * epochs
            if (self.enable_curiosity):
                forward_losses, inverse_losses = [], []

            for epoch in range(inner_loop_iters):

                policy_loss, elementwise_loss, indices = self.policy.calc_loss(memory, memory_n, n_step=n_step)
                loss = policy_loss
                if (self.enable_curiosity):
                    forward_loss, inverse_loss = self.curiosity.calc_loss(meta_step, task_number, memory_n, indices) # Add curiosity loss
                    curiosity_loss = 0.2 * forward_loss + 0.8 * inverse_loss
                    forward_losses.append(forward_loss.item())
                    inverse_losses.append(inverse_loss.item())
                    loss += curiosity_loss

                self.policy.dqn.zero_grad()  # reset last gradient
                self.curiosity.optimizer.zero_grad() # reset last gradient of curiosity
                loss.backward() # Calculate new gradients

                self.policy.optimizer.step()  # Perform ADAM inner update of DQN
                self.curiosity.optimizer.step() # Perform ADAM inner update of curiosity
                self.policy.update_priorities(elementwise_loss, indices, memory)

            if (self.enable_curiosity):
                writer.add_scalar('Task: ' + str(task_number) + '/Curiosity Forward Loss', np.mean(forward_losses), meta_step)
                writer.add_scalar('Task: ' + str(task_number) + '/Curiosity Inverse Loss', np.mean(inverse_losses), meta_step)

            writer.add_scalar('Task: ' +str(task_number) + '/Policy Loss', policy_loss, meta_step)
            writer.add_scalar('Task: ' +str(task_number) + '/Curiosity Loss ', curiosity_loss, meta_step)
            writer.add_scalar('Task: ' +str(task_number) + '/Training Steps', training_steps[task_number], meta_step)
            writer.add_scalar('Meta Learning' + '/Meta Learning rate', self.meta_lr, meta_step)


            self.policy.eval_policy(num_trajectories=20, max_trajectory_length=600, task=task_number)
            # Meta update step

            if algorithm == 'reptile':
                # Meta update with SGD, change weights to theta_before weights and perform gradient step
                theta_after = [self.policy.dqn.state_dict(),
                                self.curiosity.forwardModel.state_dict(),
                                self.curiosity.inverseModel.state_dict()]
                self.policy.dqn.load_state_dict(
                    {name: theta_before[0][name] + (theta_after[0][name] - theta_before[0][name]) * self.meta_lr for name in
                     theta_before[0]})  # Perform meta update
                self.curiosity.forwardModel.load_state_dict(
                    {name: theta_before[1][name] + (theta_after[1][name] - theta_before[1][name]) * self.meta_lr for name in
                     theta_before[1]})
                self.curiosity.inverseModel.load_state_dict(
                    {name: theta_before[2][name] + (theta_after[2][name] - theta_before[2][name]) * self.meta_lr for name in
                     theta_before[2]})
                self.policy.dqn_target.load_state_dict(self.policy.dqn.state_dict())  # Reset target network weights
            elif algorithm == 'fomaml':
                # Reset params to old params
                self.policy.dqn.load_state_dict(theta_before)
                # Perform meta learning
                with torch.no_grad():
                    for parameter in self.policy.dqn.parameters():
                        parameter -= self.meta_lr * parameter.grad
                    self.policy.dqn.zero_grad()
            elif algorithm == 'somaml':
                pass
            else:
                print("Please enter a valid algorithm: somaml, fomaml or reptile")
                exit(-1)

            self.policy.meta_step += 1
            print("Meta update performed in: {:0.3f}s".format(time.time()-meta_start))



seed = 777
# np.random.seed(seed)
# random.seed(seed)
# seed_torch(seed)

tasks = []
engine_configuration_channel = EngineConfigurationChannel()
env_parameters_channel = EnvironmentParametersChannel()
env_parameters_channel.set_float_parameter("seed", 5.0)
engine_configuration_channel.set_configuration_parameters(time_scale=10.0)
tasks.append(
            UnityEnvironment(file_name="C:/Users/Sebastian/Desktop/RLUnity/Training/mMaze/RLProject",
                             base_port=5000, timeout_wait=120,
                             no_graphics=False, seed=seed, side_channels=[engine_configuration_channel, env_parameters_channel]))

# Get Cuda Infos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

writer = SummaryWriter("C:/Users/Sebastian/Desktop/RLUnity/Training/results" + r"/Meta_Learning3")


np.set_printoptions(suppress=True, threshold=np.inf)
dqn_module = DQN_Meta_Learner(device=device, writer=writer)
meta_learner = MetaLearner(dqn_module, writer, tasks, enable_curiosity=True, meta_lr=1)
meta_learner.start_meta_learning('reptile', num_meta_updates=100, buffer_size=2000, max_trajectory_length=550, n_step=64, batch_size=512, epochs=3)
#
# writer = SummaryWriter("C:/Users/Sebastian/Desktop/RLUnity/Training/results" + r"/Meta_Learning1")
# dqn_module = DQN_Meta_Learner(device=device, writer=writer)
# meta_learner = MetaLearner(dqn_module, tasks, enable_curiosity=True, meta_lr=1)
# meta_learner.start_meta_learning('reptile', num_meta_updates=100, buffer_size=10000, max_trajectory_length=550, n_step=256, batch_size=1024, epochs=3)

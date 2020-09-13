import os
from typing import Dict, List, Tuple
from copy import deepcopy

import time


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from meta_learner_rainbow import Rainbow_Meta_Learner
from meta_learner_ppo import PPO_Meta_Learner
from meta_learner_sac import SAC_Meta_Learner

from utils import init_unity_env


# Seed Torch Function
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

# Get inital configs
class MetaLearner():
    def __init__(self, writer: SummaryWriter, device: str):
        self.writer = writer
        self.device = device


    def meta_learn(self, algorithm: str, num_meta_updates: int):
        meta_start = time.time()
        start_time = time.localtime()
        start_time = time.strftime("%H:%M:%S", start_time)
        print("Meta Learning started at: " + start_time)

        learner_algorithm = 'rainbow'

        if learner_algorithm == 'ppo':
            learner = PPO_Meta_Learner(writer=self.writer, device=self.device)
        elif learner_algorithm == 'sac':
            learner = SAC_Meta_Learner(writer=self.writer, device=self.device)
        elif learner_algorithm == 'rainbow':
            learner = Rainbow_Meta_Learner(writer=self.writer, device=self.device)

        for meta_step in range(num_meta_updates):
            print("Meta step: {} of {}".format(meta_step, num_meta_updates))

            task = init_unity_env('mMaze/RLProject.exe', maze_seed=0, maze_rows=5, maze_cols=5, random_target=0, random_agent=0, agent_x=0,agent_z=0, target_x=2, target_z=2)


            hyperparameters = learner.get_default_hyperparameters()
            hyperparameters['max_steps'] = 20000

            learner.set_env_and_detect_spaces(task, 0)
            learner.init_networks_and_optimizers(hyperparameters)
            theta_0 = [deepcopy(network.state_dict()) for network in learner.get_networks()]
            learner.train(hyperparameters)
            theta_1 = [deepcopy(network.state_dict()) for network in learner.get_networks()]

            print("Inital network weights:")
            print(theta_0)

            print("After Training network weights")
            print(theta_1)

            if algorithm == 'reptile':
                for network_before, network_after in zip(theta_0, theta_1):
                    network_after.load_state_dict(
                        {name: network_before[name] + (network_after[name] - network_before[name]) * meta_lr for
                         name in network_before
                         }
                    )

            learner.close_env()


            ############## Meta Updates ##################

            # elif algorithm == 'fomaml':
            #     # Reset params to old params
            #
            #     self.policy.dqn.load_state_dict(theta_before)
            #     # Perform meta learning
            #     with torch.no_grad():
            #         for parameter in self.policy.dqn.parameters():
            #             parameter -= self.meta_lr * parameter.grad
            #         self.policy.dqn.zero_grad()
            #
            # self.policy.eval_policy(num_trajectories=20, max_trajectory_length=600, task=task_number)
            # # Meta update step
            #
            # if algorithm == 'reptile':
            #     # Meta update with SGD, change weights to theta_before weights and perform gradient step
            #     theta_after = [self.policy.dqn.state_dict(),
            #                     self.curiosity.forwardModel.state_dict(),
            #                     self.curiosity.inverseModel.state_dict()]
            #     self.policy.dqn.load_state_dict(
            #         {name: theta_before[0][name] + (theta_after[0][name] - theta_before[0][name]) * self.meta_lr for name in
            #          theta_before[0]})  # Perform meta update
            #     self.curiosity.forwardModel.load_state_dict(
            #         {name: theta_before[1][name] + (theta_after[1][name] - theta_before[1][name]) * self.meta_lr for name in
            #          theta_before[1]})
            #     self.curiosity.inverseModel.load_state_dict(
            #         {name: theta_before[2][name] + (theta_after[2][name] - theta_before[2][name]) * self.meta_lr for name in
            #          theta_before[2]})
            #     self.policy.dqn_target.load_state_dict(self.policy.dqn.state_dict())  # Reset target network weights
            # elif algorithm == 'fomaml':
            #     # Reset params to old params
            #     self.policy.dqn.load_state_dict(theta_before)
            #     # Perform meta learning
            #     with torch.no_grad():
            #         for parameter in self.policy.dqn.parameters():
            #             parameter -= self.meta_lr * parameter.grad
            #         self.policy.dqn.zero_grad()
            # elif algorithm == 'somaml':
            #     pass
            # else:
            #     print("Please enter a valid algorithm: somaml, fomaml or reptile")
            #     exit(-1)
            #
            # self.policy.meta_step += 1
            # print("Meta update performed in: {:0.3f}s".format(time.time()-meta_start))


if __name__ == '__main__':

    run_id = "results/meta_0"

    meta_writer = SummaryWriter("results/meta_0")

    # Get Cuda Infos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    meta_learner = MetaLearner(meta_writer, device)

    meta_learner.meta_learn('reptile', 100)



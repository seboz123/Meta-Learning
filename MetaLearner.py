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

    def sample_class(self):
        task = init_unity_env('mMaze/RLProject.exe', maze_seed=0, maze_rows=3, maze_cols=3, random_target=0,
                              random_agent=0, agent_x=0, agent_z=0, target_x=2, target_z=2)

        task_number = 0

        return task, task_number


    def meta_learn(self, algorithm: str,learner, num_meta_updates: int, meta_lr: float):
        start_time = time.localtime()
        start_time = time.strftime("%H:%M:%S", start_time)
        print("Meta Learning started at: " + start_time)



        updated_parameters = None

        hyperparameters = learner.get_default_hyperparameters()
        hyperparameters['buffer_size'] = 20000
        hyperparameters['max_steps'] = 60000
        hyperparameters['decay_lr'] = False
        hyperparameters['learning_rate'] = 0.0003

        self.writer.add_text("Hyperparameters", str(hyperparameters))
        print("Hyperparameters for this run")
        print(str(hyperparameters))

        for meta_step in range(num_meta_updates):
            meta_start = time.time()
            print("Meta step: {} of {}".format(meta_step, num_meta_updates))

            task, task_number = self.sample_class()



            hyperparameters['learning_rate'] = hyperparameters['learning_rate'] * (1 - meta_step/num_meta_updates)


            learner.set_env_and_detect_spaces(task, task_number)
            learner.init_networks_and_optimizers(hyperparameters)

            if updated_parameters is not None:
                print("Setting network parameters to updated parameters!")
                for network, updated_parameters in zip(learner.get_networks_and_parameters()['networks'], updated_parameters):
                    network.load_state_dict(updated_parameters)

            theta0_state_dicts = [deepcopy(network.state_dict()) for network in learner.get_networks_and_parameters()['networks']]
            theta0_parameters = [deepcopy(parameter) for parameter in learner.get_networks_and_parameters()['parameters']]

            print("Inital network weights:")
            for key in theta0_state_dicts[0]:
                print(theta0_state_dicts[0][key])
                break

            mean_reward, mean_episode_length = learner.train(hyperparameters)

            self.writer.add_scalar("task_" + str(task_number) + r"\Overall Mean Cumulative Reward", mean_reward, meta_step)
            self.writer.add_scalar("task_" + str(task_number) + r"\Overall Mean Episode Length", mean_episode_length, meta_step)

            theta1_state_dicts = [deepcopy(network.state_dict()) for network in
                                learner.get_networks_and_parameters()['networks']]
            theta1_parameters = [deepcopy(parameter) for parameter in
                                  learner.get_networks_and_parameters()['parameters']]

            print("After Training network weights")
            for key in theta1_state_dicts[0]:
                print(theta1_state_dicts[0][key])
                break

            if algorithm == 'reptile':
                print("Performing reptile meta update!")
                for network, state_dict_0, state_dict_1 in zip(learner.get_networks_and_parameters()['networks'], theta0_state_dicts, theta1_state_dicts):
                    network.load_state_dict(
                        {name: state_dict_0[name] + (state_dict_1[name] - state_dict_0[name]) * meta_lr for
                         name in state_dict_0
                         }
                    )
            elif algorithm == 'fomaml':
                for network, inital_parameters in zip(learner.get_networks_and_parameters()['networks'],
                                                       theta0_state_dicts):
                    network.load_state_dict(inital_parameters)
                    with torch.no_grad():
                        for parameter in network.parameters():
                            parameter -= meta_lr * parameter.grad
                        network.zero_grad()

            elif algorithm == 'somaml':
                pass
            else:
                print("Please enter a valid algorithm: reptile, fomaml or somaml!")
                exit(-1)

            print("After Update network weights")
            for key in learner.get_networks_and_parameters()['networks'][0].state_dict():
                print(learner.get_networks_and_parameters()['networks'][0].state_dict()[key])
                break

            learner.meta_step += 1

            learner.close_env()
            print("Meta update took {:.3f}s".format(time.time()-meta_start))


if __name__ == '__main__':

    run_id = "results/meta_sac_3"
    learner_algorithm = 'sac'
    meta_learn_algorithm = 'reptile'

    meta_writer = SummaryWriter(run_id)

    # Get Cuda Infos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    meta_learner = MetaLearner(meta_writer, device)



    if learner_algorithm == 'ppo':
        learner = PPO_Meta_Learner(writer=meta_writer, device=device)
    elif learner_algorithm == 'sac':
        learner = SAC_Meta_Learner(writer=meta_writer, device=device)
    elif learner_algorithm == 'rainbow':
        learner = Rainbow_Meta_Learner(writer=meta_writer, device=device)
    else:
        exit(-1)

    meta_learning_rate = 1.0
    print("Meta learning rate set to: " + str(meta_learning_rate))
    meta_learner.meta_learn(meta_learn_algorithm, learner, num_meta_updates=100, meta_lr=meta_learning_rate)



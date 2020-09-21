from itertools import chain
from copy import deepcopy

import time

import numpy as np
import torch
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


    def sample_task_from_distribution(self, task_distribution: list):
        sampled_size = np.where(np.random.multinomial(1, task_distribution) == 1)[0] + 2
        target_x = np.random.randint(0, sampled_size)
        target_z = np.random.randint(0, sampled_size)

        agent_x = np.random.randint(0, sampled_size)
        agent_z = np.random.randint(0, sampled_size)

        if agent_z == agent_x and agent_x == target_x and target_x == target_z:
            task, task_number = self.sample_task_from_distribution(task_distribution)
        else:

            task = init_unity_env('mMaze.app', maze_seed=0, maze_rows=sampled_size, maze_cols=sampled_size, random_target=0,
                              random_agent=0, agent_x=agent_x, agent_z=agent_z, target_x=target_x, target_z=target_z, enable_heatmap=True)

            task_number = 0

        return task, task_number

    def meta_learn(self, algorithm: str, learner, num_meta_updates: int, meta_lr: float):
        start_time = time.localtime()
        start_time = time.strftime("%H:%M:%S", start_time)
        print("Meta Learning started at: " + start_time)

        updated_parameters = None

        hyperparameters = learner.get_default_hyperparameters()
        hyperparameters['buffer_size'] = 5000
        hyperparameters['max_steps'] = 10000
        hyperparameters['time_horizon'] = 512
        hyperparameters['decay_lr'] = False
        hyperparameters['learning_rate'] = 0.0003

        self.writer.add_text("Hyperparameters", str(hyperparameters))
        print("Hyperparameters for this run")
        print(str(hyperparameters))

        task_dist = [0.7, 0.2, 0.1]
        print("Distribution over Tasks: " + str(task_dist))


        for meta_step in range(num_meta_updates):
            meta_start = time.time()
            print("Meta step: {} of {}".format(meta_step, num_meta_updates))

            task, task_number = self.sample_task_from_distribution(task_dist)

            hyperparameters['learning_rate'] = hyperparameters['learning_rate'] * (1 - meta_step / num_meta_updates)

            learner.set_env_and_detect_spaces(task, task_number)
            learner.init_networks_and_optimizers(hyperparameters)

            if updated_parameters is not None:
                print("Setting network parameters to updated parameters!")
                for network, updated_parameters in zip(learner.get_networks_and_parameters()['networks'],
                                                       updated_parameters):
                    network.load_state_dict(updated_parameters)

            if meta_step == 0:
                networks = learner.get_networks_and_parameters()['networks']
                optimizer_params = []
                for network in networks:
                    optimizer_params.extend(network.parameters())
                self.meta_optimizer = optim.Adam(chain(optimizer_params), lr=meta_lr)

            for parameter_group in self.meta_optimizer.param_groups:
                parameter_group['lr'] = hyperparameters['learning_rate']

            networks_before = deepcopy(learner.get_networks_and_parameters()['networks'])
            theta_before_state_dicts = [network.state_dict() for network in networks_before]



            mean_reward, mean_episode_length = learner.train(hyperparameters)

            self.writer.add_scalar("task_" + str(task_number) + r"\Overall Mean Cumulative Reward", mean_reward,
                                   meta_step)
            self.writer.add_scalar("task_" + str(task_number) + r"\Overall Mean Episode Length", mean_episode_length,
                                   meta_step)

            networks_after = deepcopy(learner.get_networks_and_parameters()['networks'])
            theta_after_state_dicts = [network.state_dict() for network in networks_after]

            print("Inital network weights:")
            for key in theta_before_state_dicts[0]:
                print(theta_before_state_dicts[0][key])
                break

            print("After Training network weights")
            for key in theta_after_state_dicts[0]:
                print(theta_after_state_dicts[0][key])
                break

            if algorithm == 'reptile':
                print("Performing reptile meta update!")
                # Set Theta - W as gradient to step with ADAM
                self.meta_optimizer.zero_grad()
                for network_before, network_after in zip(networks_before,
                                                         learner.get_networks_and_parameters()['networks']):
                    for parameter_before, parameter_after in zip(network_before.parameters(),
                                                                 network_after.parameters()):
                        parameter_after.grad = parameter_before - parameter_after
                self.meta_optimizer.step()

            elif algorithm == 'fomaml':
                print("Performing First-Order MAML meta update!")
                for networks_after, network_before in zip(learner.get_networks_and_parameters()['networks'],
                                                      networks_before):
                    network_before.load_state_dict(networks_after.state_dict())
                self.meta_optimizer.step()

            elif algorithm == 'somaml':
                pass
            else:
                print("Please enter a valid algorithm: reptile, fomaml or somaml!")
                exit(-1)

            print("After Update network weights")
            for key in learner.get_networks_and_parameters()['networks'][0].state_dict():
                print(learner.get_networks_and_parameters()['networks'][0].state_dict()[key])
                break

            updated_parameters = theta_after_state_dicts
            learner.meta_step += 1
            learner.close_env()
            print("Meta update took {:.3f}s".format(time.time() - meta_start))


if __name__ == '__main__':

    run_id = "results/meta_sac_0"
    learner_algorithm = 'sac'
    meta_learn_algorithm = 'fomaml'

    meta_writer = SummaryWriter(run_id)
    meta_writer.add_text("Run_ID", run_id)
    print("Current Run ID: " + run_id)

    # Get Cuda Infos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    meta_learner = MetaLearner(meta_writer, device)

    if learner_algorithm == 'ppo':
        learner = PPO_Meta_Learner(writer=meta_writer, device=device, is_meta_learning=True)
    elif learner_algorithm == 'sac':
        learner = SAC_Meta_Learner(writer=meta_writer, device=device, is_meta_learning=True)
    elif learner_algorithm == 'rainbow':
        learner = Rainbow_Meta_Learner(writer=meta_writer, device=device, is_meta_learning=True)
    else:
        exit(-1)

    meta_learning_rate = 0.01
    print("Meta learning rate set to: " + str(meta_learning_rate))
    meta_learner.meta_learn(meta_learn_algorithm, learner, num_meta_updates=100, meta_lr=meta_learning_rate)

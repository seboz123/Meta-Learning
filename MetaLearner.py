from itertools import chain
from copy import deepcopy
import sys

import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from meta_learner_rainbow import Rainbow_Meta_Learner
from mlagents_meta_learner import MLAgentsTrainer

from utils import init_unity_env


# Seed Torch Function
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


# Get inital configs
class MetaLearner:
    def __init__(self, learner, writer: SummaryWriter, run_id: str, device: str):
        self.writer = writer
        self.device = device
        self.run_id = run_id
        self.learner = learner
        seed_torch(0)

    def sample_task_from_distribution(self, learner_type: str):

        difficulties = [2, 2, 4, 4, 6]
        # Maze size, seed
        mazes = [[2, 0], [2, 2], [3, 0], [3, 2], [3, 4]]
        sum = np.sum(difficulties)
        probs_unscaled = []
        for difficulty in difficulties:
            probs_unscaled.append((difficulty / sum) ** -1)
        prob_sum = np.sum(probs_unscaled)
        probs_scaled = []
        for prob in probs_unscaled:
            probs_scaled.append(prob / prob_sum)

        print(probs_scaled)
        sampled_task = np.where(np.random.multinomial(1, probs_scaled) == 1)[0][0]
        print(sampled_task)

        maze = mazes[sampled_task]

        if learner_type == 'rainbow':
            task = init_unity_env('mMaze/RLProject.exe', maze_seed=maze[1], maze_rows=maze[0],
                                  maze_cols=maze[0],
                                  random_target=0, random_agent=0, agent_x=0, agent_z=0,
                                  target_x=maze[1]-1, agent_rot=0,
                                  target_z=maze[1]-1, enable_heatmap=True, enable_sight_cone=True)
            return task, sampled_task
        else:
            self.learner.set_env_parameters(maze_seed=maze[1], maze_rows=maze[0],
                                  maze_cols=maze[0],
                                  random_target=0, random_agent=0, agent_x=0, agent_z=0,
                                  target_x=maze[1]-1, agent_rot=0,
                                  target_z=maze[1]-1, enable_heatmap=True, enable_sight_cone=True)

            return sampled_task

    def print_weights(self, weight: dict):
        for i, key in enumerate(weight):
            if i == 5:
                print(weight[key])
                break

    def meta_learn(self, learner_algorithm: str, meta_algorithm: str, meta_optimizer: str, num_meta_updates: int,
                   meta_lr: float):
        start_time = time.localtime()
        start_time = time.strftime("%H:%M:%S", start_time)
        print("Meta Learning started at: " + start_time)
        # torch.autograd.set_detect_anomaly(True)

        if learner_algorithm == 'rainbow':
            hyperparameters = self.learner.get_default_hyperparameters()
            hyperparameters['buffer_size'] = 20000
            hyperparameters['batch_size'] = 1024

            hyperparameters['max_steps'] = 1000000
            hyperparameters['time_horizon'] = 3
            hyperparameters['decay_lr'] = True
            hyperparameters['learning_rate'] = 0.0003
            hyperparameters['task_distribution'] = [1, 0.0, 0.0]
            hyperparameters['num_meta_updates'] = num_meta_updates
            hyperparameters['Meta Learning Rate'] = meta_lr
            hyperparameters['Meta Algorithm'] = meta_algorithm
            hyperparameters['Meta Optimizer'] = meta_optimizer

            hyperparameters['epsilon'] = 0.2  # Percentage to explore epsilon = 0 -> Decaying after half training
            hyperparameters['v_max'] = 15  # Maximum Value of Reward
            hyperparameters['v_min'] = -15  # Minimum Value of Reward
            hyperparameters['atom_size'] = 51  # Atom Size for categorical DQN
            hyperparameters['update_period'] = 80  # Period after which Target Network gets updated
            hyperparameters['beta'] = 0.4  # How much to use importance sampling
            hyperparameters['alpha'] = 0.2  # How much to use prioritization
            hyperparameters['prior_eps'] = 1e-6  # Guarantee to use all experiences

            hyperparameters['enable_curiosity'] = False
            hyperparameters['curiosity_lambda'] = 10  # Weight factor of extrinsic reward. 0.1 -> 10*Curiosity
            hyperparameters['curiosity_beta'] = 0.2  # Factor for using more of forward loss or more of inverse loss
            hyperparameters['curiosity_enc_size'] = 32  # Encoding size of curiosity_module
            hyperparameters['curiosity_layers'] = 2  # Layers of Curiosity Modules
            hyperparameters['curiosity_units'] = 128  # Number of hidden units for curiosity modules

            hypers_text = [str(key) + ": " + str(hyperparameters[key]) for key in hyperparameters]
            hypers_text = '  \n'.join(hypers_text)
            self.writer.add_text("Meta-Hyperparameters", hypers_text, 0)

            print("Meta-Hyperparameters:")
            for key in hyperparameters:
                print("{:<25s} {:<20s}".format(key, str(hyperparameters[key])))

            for meta_step in range(num_meta_updates):
                meta_start = time.time()
                print("Meta step: {} of {}".format(meta_step, num_meta_updates))

                # task, task_number = self.sample_task_from_distribution(learner_algorithm, hyperparameters['task_distribution'])
                task = init_unity_env('mMaze_dis_ref/RLProject.exe', maze_seed=0, maze_rows=3,
                                      maze_cols=3,
                                      random_target=0, random_agent=0, agent_x=0, agent_z=0,
                                      target_x=2,
                                      target_z=2, enable_heatmap=True, enable_sight_cone=True)
                # task_number = 200
                self.learner.set_env_and_detect_spaces(task, 0)

                if meta_step == 0:
                    self.learner.init_networks_and_optimizers(hyperparameters)
                    self.train_networks = self.learner.get_networks_and_parameters()['networks']
                    optimizer_params = []
                    for network in self.train_networks:
                        optimizer_params.extend(network.parameters())
                    if meta_optimizer == 'adam':
                        self.meta_optimizer = optim.Adam(chain(optimizer_params), lr=meta_lr)
                    elif meta_optimizer == 'sgd':
                        self.meta_optimizer = optim.SGD(chain(optimizer_params), lr=meta_lr)
                    else:
                        print("Enter valid Meta optimizer")
                        exit(-1)
                    # exp_schedule = lambda epoch: max(0.99 ** epoch, 0.001)
                    # self.meta_scheduler = optim.lr_scheduler.LambdaLR(self.meta_optimizer, exp_schedule)
                if hyperparameters['decay_lr']:
                    hyperparameters['learning_rate'] = hyperparameters['learning_rate'] * (
                                1 - meta_step / num_meta_updates)

                for parameter_group in self.meta_optimizer.param_groups:
                    parameter_group['lr'] = hyperparameters['learning_rate']

                # print("Before")
                # print(self.train_networks[0].state_dict()['feature_layer.0.0.weight'])

                networks_before = deepcopy(self.train_networks)

                self.learner.train(run_id, hyperparameters)

                # print(self.learner.get_networks_and_parameters()['networks'][0].state_dict()['feature_layer.0.0.weight'])

                self.meta_optimizer.zero_grad()
                if meta_algorithm == 'reptile':
                    print("Performing reptile meta update!")
                    # Set Theta - W as gradient to step with ADAM
                    for network_before, network_after in zip(networks_before, self.learner.get_networks_and_parameters()['networks']):
                        for parameter_init, parameter_after in zip(network_before.parameters(),
                                                                   network_after.parameters()):
                            parameter_after.grad = parameter_init.data - parameter_after.data
                            parameter_after.data = parameter_init.data
                elif meta_algorithm == 'fomaml':
                    print("Performing First-Order MAML meta update!")
                    for network_before, network_after in zip(networks_before, self.learner.get_networks_and_parameters()['networks']):
                        for parameter_init, parameter_after in zip(network_before.parameters(),
                                                                   network_after.parameters()):
                            parameter_after.data = parameter_init.data

                elif meta_algorithm == 'somaml':
                    print("Performing Second-Order MAML meta update!")
                    for network_before, network_after in zip(networks_before,
                                                             self.learner.get_networks_and_parameters()['networks']):
                        for parameter_init, parameter_after in zip(network_before.parameters(),
                                                                   network_after.parameters()):
                            parameter_after.data = parameter_init.data
                else:
                    print("Please enter a valid meta_algorithm: reptile, fomaml or somaml!")
                    exit(-1)

                self.meta_optimizer.step()
                print("After Meta step network weights")
                for key in self.train_networks[0].state_dict():
                    print(self.train_networks[0].state_dict()[key])
                    break

                self.learner.meta_step += 1
                self.learner.close_env()

                print("Meta update took {:.3f}s".format(time.time() - meta_start))

        elif learner_algorithm == 'ppo' or learner_algorithm == 'sac':

            if learner_algorithm == 'ppo':
                hyperparameters = {'buffer_size': 5000, 'max_steps': 30000, 'time_horizon': 128, 'decay_lr': True,
                                   'learning_rate': 0.0003, 'batch_size': 4096, 'hidden_layers': 2, 'layer_size': 512,
                                   'task_distribution': [0.7, 0.2, 0.1], 'num_meta_updates': num_meta_updates,
                                   'Meta Learning Rate': meta_lr, 'Meta Algorithm': meta_algorithm,
                                   'Meta Optimizer': meta_optimizer}
                self.learner.set_hyperparameters(hyperparameters)
                self.learner.set_env_parameters(maze_rows=3, maze_cols=3, agent_x=0, agent_z=0, target_x=2,
                                                target_z=2, random_agent=0, random_target=0, maze_seed=0,
                                                agent_rot=0, enable_heatmap=True, enable_sight_cone=True)
            elif learner_algorithm == 'sac':
                hyperparameters = {'buffer_size': 2000000, 'max_steps': 3000000, 'time_horizon': 128,
                                   'decay_lr': False, 'learning_rate': 0.0003, 'batch_size': 256, 'hidden_layers': 2,
                                   'layer_size': 512, 'task_distribution': [0.7, 0.2, 0.1],
                                   'num_meta_updates': num_meta_updates,
                                   'Meta Learning Rate': meta_lr, 'Meta Algorithm': meta_algorithm,
                                   'Meta Optimizer': meta_optimizer}
                self.learner.set_hyperparameters(hyperparameters)
                self.learner.set_env_parameters(maze_rows=3, maze_cols=3, agent_x=0, agent_z=0, target_x=0,
                                                target_z=2, random_agent=0, random_target=0, maze_seed=0,
                                                enable_heatmap=True, enable_sight_cone=True, agent_rot=0)
            self.learner.init_optimizer()

            # Set up the meta optimizer
            optimizer_params = []
            for network in self.learner.train_networks:
                optimizer_params.extend(network.parameters())
            if learner_algorithm == 'sac':
                optimizer_params.extend(self.learner.init_params[3])

            if meta_optimizer == 'adam':
                self.meta_optimizer = optim.Adam(chain(optimizer_params), lr=meta_lr)
            elif meta_optimizer == 'sgd':
                self.meta_optimizer = optim.SGD(chain(optimizer_params), lr=meta_lr)

            hypers_text = [str(key) + ": " + str(hyperparameters[key]) for key in hyperparameters]
            hypers_text = '  \n'.join(hypers_text)
            self.writer.add_text("Meta-Hyperparameters", hypers_text, 0)

            print("Meta-Hyperparameters:")
            for key in hyperparameters:
                print("{:<25s} {:<20s}".format(key, str(hyperparameters[key])))

            networks_trained = self.learner.train_networks

            for meta_step in range(num_meta_updates):
                meta_start = time.time()
                print("Meta step: {} of {}".format(meta_step, num_meta_updates))

                # task, task_number = self.sample_task_from_distribution(learner_type, hyperparameters['task_distribution'])
                # self.learner.set_env_parameters(maze_rows=3, maze_cols=3, agent_x=0, agent_z=0, target_x=2,
                #                                 target_z=2, random_agent=0, random_target=0, maze_seed=0,
                #                                 enable_heatmap=True, enable_sight_cone=True)
                task_number = 200
                self.writer.add_text("Meta-Tasks", "Meta-Step:" + str(meta_step) + " Task Number" + str(task_number))
                # Decay The overall learning Rate
                self.learner.options.behaviors["Brain"].hyperparameters.learning_rate = (1 - meta_step / num_meta_updates) * self.learner.init_lr
                networks_before = deepcopy(networks_trained)
                # Train Phase
                networks_trained, parameters_trained, _ = self.learner.train(task_number=task_number,
                                                                                        run_id=self.run_id,
                                                                                        init_networks=networks_before)

                print("Weights before Training: ")
                self.print_weights(networks_before[0].state_dict())
                print("Weights after Training: ")
                self.print_weights(networks_trained[0].state_dict())
                # print("Curiosity before Training:")
                # print(networks_before[3].state_dict()['_state_encoder.linear_encoder.seq_layers.0.weight'])
                # print("Curiosity after Training:")
                # print(networks_trained[3].state_dict()['_state_encoder.linear_encoder.seq_layers.0.weight'])

                self.meta_optimizer.zero_grad()
                # Perform Meta-update
                if meta_algorithm == 'reptile':
                    # Calculate Gradient for Reptile update
                    # grad = theta_1 - theta_0
                    # Set the gradient of init_network to perform Meta-update
                    for network_trained, init_network in zip(networks_trained, self.learner.train_networks):
                        for param_t, param_init in zip(network_trained.parameters(), init_network.parameters()):
                            param_init.grad = param_init.data - param_t.data
                    for param_t, param_init in zip(parameters_trained[-1], self.learner.init_params[-1]):
                        param_init.grad = param_init.data - param_t.data
                if meta_algorithm == 'fomaml':
                    # Get Gradient for First-Order MAML update
                    # grad = grad(Last_update)
                    # Set the gradient of init_network to perform Meta-update
                    for network_parameters, init_network in zip(parameters_trained, self.learner.train_networks):
                        for param_t, param_init in zip(network_parameters, init_network.parameters()):
                            # Set gradients to last gradients of Training
                            param_init.grad = param_t.grad
                    for param_t, param_init in zip(parameters_trained[-1], self.learner.init_params[-1]):
                        param_init.grad = param_t.grad
                if meta_algorithm == 'somaml':
                    hyperparameters = {'buffer_size': 5000, 'max_steps': 15000, 'time_horizon': 128, 'decay_lr': True,
                                       'learning_rate': 0.0003, 'batch_size': 1024, 'hidden_layers': 2,
                                       'layer_size': 512,
                                       'task_distribution': [0.7, 0.2, 0.1], 'num_meta_updates': num_meta_updates,
                                       'Meta Learning Rate': meta_lr, 'Meta Algorithm': meta_algorithm,
                                       'Meta Optimizer': meta_optimizer}
                    self.learner.set_hyperparameters(hyperparameters)

                    for network_t, network_init in zip(networks_trained, self.learner.train_networks):
                        network_init.load_state_dict(network_t.state_dict())

                    init_networks = deepcopy(self.learner.train_networks)
                    networks_trained, parameters_trained, meta_loss = self.learner.train(task_number=task_number,
                                                                                         run_id="test_maml_1",
                                                                                         init_networks=self.learner.train_networks,
                                                                                         meta_eval=True)
                    meta_loss.backward()
                    for i_network, t_network in zip(init_networks, self.learner.train_networks):
                        t_network.load_state_dict(i_network.state_dict())
                    print("Succesful MAML Update!")

                # Step the Meta optimizer
                self.meta_optimizer.step()

                # Load the optimized Network into the next inital network
                for network_t, network_init in zip(networks_trained, self.learner.train_networks):
                    network_t.load_state_dict(network_init.state_dict())

                print("Weights after Meta update: ")
                self.print_weights(networks_trained[0].state_dict())

                self.learner.meta_step += 1
                print("Meta update took {:.3f}s".format(time.time() - meta_start))


if __name__ == '__main__':

    run_id = "Meta_Run_2"
    run_id = sys.argv[1]
    learner_type = 'rainbow'
    learner_type = sys.argv[2]
    meta_learn_algorithm = 'reptile'
    meta_learn_algorithm = sys.argv[3]
    meta_optimizer = 'sgd'
    meta_optimizer = sys.argv[4]
    num_meta_updates = 100
    num_meta_updates = int(sys.argv[5])
    meta_lr = 0.01
    meta_lr = float(sys.argv[6])

    print("Current Run ID: " + run_id)

    # Get Cuda Infos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    learner = None
    if learner_type == 'ppo':
        meta_writer = SummaryWriter("results/" + run_id + "_PPO_INFO")
        learner = MLAgentsTrainer(run_id=run_id, rl_algorithm='ppo')
    elif learner_type == 'sac':
        meta_writer = SummaryWriter("results/" + run_id + "_SAC_INFO")
        learner = MLAgentsTrainer(run_id=run_id, rl_algorithm='sac')
    elif learner_type == 'rainbow':
        meta_writer = SummaryWriter("results/" + run_id + "_RBOW_INFO")
        learner = Rainbow_Meta_Learner(device=device, is_meta_learning=True)
    else:
        exit(-1)

    meta_learner = MetaLearner(learner, meta_writer, run_id, device)
    meta_learner.meta_learn(learner_type, meta_learn_algorithm, meta_optimizer, num_meta_updates=num_meta_updates,
                            meta_lr=meta_lr)

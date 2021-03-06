from itertools import chain
from copy import deepcopy
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from meta_learner_rainbow import RainbowMetaLearner
from mlagents_meta_learner import MLAgentsTrainer

# Seed Torch Function
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


class MetaLearner:
    def __init__(self, learner, writer: SummaryWriter, run_id: str, device: str):
        self.writer = writer
        self.device = device
        self.run_id = run_id
        self.learner = learner
        seed_torch(0)

    # Get Default Hyperparameters for Algorithms
    def get_ppo_hyperparameters(self):
        return {'buffer_size': 60000, 'max_steps': 130000, 'time_horizon': 128, 'decay_lr': False, 'num_epochs': 4,
                'summary_freq': 20000,
                           'learning_rate': 0.0001, 'batch_size': 4096, 'hidden_layers': 2, 'layer_size': 512}

    def get_sac_hyperparameters(self):
        return {'buffer_size': 50000, 'max_steps': 100000, 'time_horizon': 128, 'summary_freq': 20000,
                           'decay_lr': False, 'learning_rate': 0.0003, 'batch_size': 2048, 'hidden_layers': 2,
                           'layer_size': 512}
    # Start Joint Training
    def joint_train(self, num_updates, task_dist):
        # Set Enviornment Parameters ( First to just initialize the optimizers)
        self.learner.set_env_parameters(difficulty=0, maze_rows=3, maze_cols=3, agent_x=0, agent_z=0, target_x=2,
                                        target_z=2, random_agent=0, random_target=0, maze_seed=0, joint_training=False,
                                        agent_rot=0, enable_heatmap=True, enable_sight_cone=True)
        self.learner.init_optimizer()
        hyperparameters = self.get_ppo_hyperparameters()
        hyperparameters['max_steps'] = 130000
        hyperparameters['task_dist'] = task_dist
        self.learner.set_hyperparameters(hyperparameters)
        seeds = [i+100 for i in range(num_meta_updates + 20)]
        networks_trained = self.learner.train_networks

        # Train for number of joint updates
        for step in range(num_updates):
            # Get the difficulty according to task distribution
            difficulty = np.where(np.random.multinomial(1, hyperparameters['task_dist']) == 1)[0][0] + 1
            seed = seeds[step]
            task_number = difficulty * 1000 + seed
            # Set Enviornment Parameters to generate Tasks of sampled Difficulty
            self.learner.set_env_parameters(difficulty=difficulty, maze_rows=3, maze_cols=3, agent_x=0, agent_z=0,
                                            target_x=2,
                                            target_z=2, random_agent=0, random_target=0, maze_seed=seed, agent_rot=0,
                                            joint_training=False,
                                            enable_heatmap=True, enable_sight_cone=True)
            # Start Training
            networks_trained, parameters_trained, _ = self.learner.train(task_number=task_number,
                                                                         run_id="Joint_RUN_"+str(step),
                                                                         init_networks=networks_trained)

        # Evaluate Joint-Training Process
        self.eval(learner_algorithm='ppo', eval_networks=networks_trained)

    # Start Meta-Learning
    def meta_learn(self, learner_algorithm: str, meta_algorithm: str, meta_optimizer: str, num_meta_updates: int,
                   meta_lr: float, meta_batch_size: int, task_dist: list, separate_fomaml: bool):
        assert meta_algorithm == "reptile" or meta_algorithm == 'fomaml' or meta_algorithm == 'joint_training'
        assert learner_algorithm == 'ppo' or learner_algorithm == 'sac' or learner_algorithm == 'rainbow'
        start_time = time.localtime()
        start_time = time.strftime("%H:%M:%S", start_time)
        print("Meta Learning started at: " + start_time)

        # Meta-Learning for Rainbow
        # Implementation unfinished
        # Should be a good start to implement meta-learning with rainbow

        # if learner_algorithm == 'rainbow':
            # hyperparameters = self.learner.get_default_hyperparameters()
            # hyperparameters['buffer_size'] = 20000
            # hyperparameters['batch_size'] = 1024
            # hyperparameters['max_steps'] = 100000
            # hyperparameters['time_horizon'] = 3
            # hyperparameters['decay_lr'] = False
            # hyperparameters['learning_rate'] = 0.0003
            # hyperparameters['num_meta_updates'] = num_meta_updates
            # hyperparameters['Meta Learning Rate'] = meta_lr
            # hyperparameters['Meta Algorithm'] = meta_algorithm
            # hyperparameters['Meta Optimizer'] = meta_optimizer
            # hyperparameters['task_dist'] = task_dist
            # hyperparameters['meta_batch_size'] = meta_batch_size
            # # Save Hyperparameters for Tensorboard
            # hypers_text = [str(key) + ": " + str(hyperparameters[key]) for key in hyperparameters]
            # hypers_text = '  \n'.join(hypers_text)
            # self.writer.add_text("Meta-Hyperparameters", hypers_text, 0)
            #
            # print("Meta-Hyperparameters:")
            # for key in hyperparameters:
            #     print("{:<25s} {:<20s}".format(key, str(hyperparameters[key])))
            #
            # for meta_step in range(num_meta_updates):
            #     meta_start = time.time()
            #     print("Meta step: {} of {}".format(meta_step, num_meta_updates))
            #
            #     difficulty = np.where(np.random.multinomial(1, hyperparameters['task_dist']) == 1)[0][0] + 1
            #     seed = np.random.randint(0, 4)
            #     task_number = difficulty*1000 + seed
            #     # Start environment with sampled difficulty
            #     task = init_unity_env('mMaze_dis_ref/RLProject.exe', difficulty=difficulty, maze_rows=3, maze_cols=3, agent_x=0, agent_z=0, target_x=2,
            #                                     target_z=2, random_agent=0, random_target=0, maze_seed=seed, base_port=np.random.randint(1000, 10000),
            #                                     agent_rot=0, enable_heatmap=True, enable_sight_cone=True)
            #     self.learner.set_env_and_detect_spaces(task, task_number)
            #
            #     # Initialize Optimizers
            #     if meta_step == 0:
            #         self.learner.init_networks_and_optimizers(hyperparameters)
            #         self.train_networks = self.learner.get_networks_and_parameters()['networks']
            #         optimizer_params = []
            #         for network in self.train_networks:
            #             optimizer_params.extend(network.parameters())
            #         if meta_optimizer == 'adam':
            #             self.meta_optimizer = optim.Adam(chain(optimizer_params), lr=meta_lr)
            #         elif meta_optimizer == 'sgd':
            #             self.meta_optimizer = optim.SGD(chain(optimizer_params), lr=meta_lr)
            #         else:
            #             print("Enter valid Meta optimizer")
            #             exit(-1)
            #
            #     networks_before = deepcopy(self.train_networks)
            #     self.learner.train(run_id, hyperparameters)
            #
            #     self.meta_optimizer.zero_grad()
            #     if meta_algorithm == 'reptile':
            #         print("Performing reptile meta update!")
            #         # Set Theta - W as gradient to step with ADAM
            #         for network_before, network_after in zip(networks_before, self.learner.get_networks_and_parameters()['networks']):
            #             for parameter_init, parameter_after in zip(network_before.parameters(),
            #                                                        network_after.parameters()):
            #                 parameter_after.grad = parameter_init.data - parameter_after.data
            #                 parameter_after.data = parameter_init.data
            #     elif meta_algorithm == 'fomaml':
            #         print("Performing First-Order MAML meta update!")
            #         for network_before, network_after in zip(networks_before, self.learner.get_networks_and_parameters()['networks']):
            #             for parameter_init, parameter_after in zip(network_before.parameters(),
            #                                                        network_after.parameters()):
            #                 parameter_after.data = parameter_init.data
            #     else:
            #         print("Please enter a valid meta_algorithm: reptile, fomaml!")
            #         exit(-1)
            #
            #     self.meta_optimizer.step()
            #     self.learner.meta_step += 1
            #     self.learner.close_env()
            #
            #     print("Meta update took {:.3f}s".format(time.time() - meta_start))

        # Meta Learning of PPO or SAC
        if learner_algorithm == 'ppo' or learner_algorithm == 'sac':
            # Load the correct Hyperparameters for training with ppo/sac
            if learner_algorithm == 'ppo':
                # Get the Hyperparameters as a dictonnary
                # Set the RunOptions with set_hyperparameters function
                hyperparameters = self.get_ppo_hyperparameters()
                hyperparameters['num_meta_updates'] = num_meta_updates
                hyperparameters['Meta Learning Rate'] = meta_lr
                hyperparameters['Meta Algorithm'] = meta_algorithm
                hyperparameters['Meta Optimizer'] = meta_optimizer
                hyperparameters['task_dist'] = task_dist
                hyperparameters['meta_batch_size'] = meta_batch_size
                hyperparameters['separate_fomaml'] = separate_fomaml
                self.learner.set_hyperparameters(hyperparameters)

            elif learner_algorithm == 'sac':
                # Get the Hyperparameters as a dictonnary
                # Set the RunOptions with set_hyperparameters function
                hyperparameters = self.get_sac_hyperparameters()
                hyperparameters['num_meta_updates'] = num_meta_updates
                hyperparameters['Meta Learning Rate'] = meta_lr
                hyperparameters['Meta Algorithm'] = meta_algorithm
                hyperparameters['Meta Optimizer'] = meta_optimizer
                hyperparameters['task_dist'] = task_dist
                hyperparameters['meta_batch_size'] = meta_batch_size
                hyperparameters['separate_fomaml'] = separate_fomaml
                self.learner.set_hyperparameters(hyperparameters)

            hypers_text = [str(key) + ": " + str(hyperparameters[key]) for key in hyperparameters]
            hypers_text = '  \n'.join(hypers_text)
            self.writer.add_text("Meta-Hyperparameters", hypers_text, 0)

            # Set the environment parameter with which MLAgents will get launched
            self.learner.set_env_parameters(difficulty=0, maze_rows=3, maze_cols=3, agent_x=0, agent_z=0, target_x=2,
                                                target_z=2, random_agent=0, random_target=0, maze_seed=0,joint_training=False,
                                                agent_rot=0, enable_heatmap=True, enable_sight_cone=True)
            # Init Optimizers, e.g. set the initial networks to access them for later usage
            self.learner.init_optimizer()

            # With the obtained Networks in init_optimizer() we can init ADAM/SGD...
            # Set up the meta optimizer
            optimizer_params = []
            for network in self.learner.train_networks:
                optimizer_params.extend(network.parameters())
            # SAC has learnable Temperature
            if learner_algorithm == 'sac':
                optimizer_params.extend(self.learner.init_params[-1])

            if meta_optimizer == 'adam':
                self.meta_optimizer = optim.Adam(chain(optimizer_params), lr=meta_lr)
            elif meta_optimizer == 'sgd':
                self.meta_optimizer = optim.SGD(chain(optimizer_params), lr=meta_lr)



            print("Meta-Hyperparameters:")
            for key in hyperparameters:
                print("{:<25s} {:<20s}".format(key, str(hyperparameters[key])))

            networks_trained = self.learner.train_networks

            seeds = [i for i in range(num_meta_updates+20)]
            # Outer Loop
            for meta_step in range(num_meta_updates):
                print("Meta step: {} of {}".format(meta_step, num_meta_updates))
                # Sample task according to distribution
                difficulty = np.where(np.random.multinomial(1, hyperparameters['task_dist']) == 1)[0][0]+1
                seed = seeds[meta_step]
                task_number = difficulty*1000 + seed

                self.learner.set_env_parameters(difficulty=difficulty, maze_rows=3, maze_cols=3, agent_x=0, agent_z=0, target_x=2,
                                                target_z=2, random_agent=0, random_target=0, maze_seed=seed, agent_rot=0, joint_training=False,
                                                enable_heatmap=True, enable_sight_cone=True)
                self.writer.add_text("Meta-Tasks", "Meta-Step:" + str(meta_step) + " Task Number: " + str(task_number))

                networks_before = deepcopy(networks_trained)
                networks_before.append(self.learner.init_params[-1])
                # Train Phase

                # Print weights for debugging
                # print("Weights before Training: ")
                # for i, weight in enumerate(networks_trained[0].parameters()):
                #     # print(weight)
                #     if i == 3:
                #         print(weight)
                #         break

                # Inner-Loop Update
                # Detect Batched and Non-Batched Versions of Meta-Learning Algorithm
                if hyperparameters['meta_batch_size'] > 1:
                    networks_batched = []
                    parameters_batched = []
                    for i in range(hyperparameters['meta_batch_size']):
                        difficulty = np.where(np.random.multinomial(1, hyperparameters['task_dist']) == 1)[0][0] + 1
                        seed = seeds[meta_step]+i+num_meta_updates
                        task_number = difficulty * 1000 + seed

                        # Set the Environment Parameters (e.g. Change the difficulty of maze and so on)
                        self.learner.set_env_parameters(difficulty=difficulty, maze_rows=3, maze_cols=3, agent_x=0,
                                                        agent_z=0, target_x=2,
                                                        target_z=2, random_agent=0, random_target=0, maze_seed=seed,
                                                        agent_rot=0, joint_training=False,
                                                        enable_heatmap=True, enable_sight_cone=True)

                        # Start Training of current Task of Meta-Batch
                        networks_trained, parameters_trained, _ = self.learner.train(task_number=task_number,
                                                                                                run_id=self.run_id+"_batch_"+str(i),
                                                                                                init_networks=networks_before)
                        networks_batched.append(networks_trained)
                        parameters_batched.append(parameters_trained)
                else:
                    # If using Meta-Batch size of 1 we don't need Batches
                    networks_trained, parameters_trained, _ = self.learner.train(task_number=task_number,
                                                                                 run_id=self.run_id,
                                                                                 init_networks=networks_before)
                    networks_batched = [networks_trained]
                    parameters_batched = [parameters_trained]

                # Print weights for Debugging
                # print("Weights of Batch-Network:0 after Training: ")
                # for i, weight in enumerate(networks_trained[0].parameters()):
                #     if i == 3:
                #         print(weight)
                #         break

                self.meta_optimizer.zero_grad()
                # Perform Meta-update
                # Meta-Update sets the gradients of the weights to the Meta-Gradient
                if meta_algorithm == 'reptile':
                    # Perform Reptile Meta-Update
                    self.reptile(parameters_batched, hyperparameters)
                if meta_algorithm == 'fomaml':
                    # Perform FOMAML Meta-Update
                    self.fomaml(networks_batched, parameters_batched, hyperparameters)
                # Step the Meta optimizer, with the set Meta-Gradient
                self.meta_optimizer.step()

                # We need to load the now Updated Network weights for the next iteration
                # network_init state_dict contains weights after Meta-Update
                # networks_trained contains weights for next train phase
                for network_t, network_init in zip(networks_trained, self.learner.train_networks):
                    network_t.load_state_dict(network_init.state_dict())

                # Print weights for Debugging
                #
                # print("Weights after Meta update: ")
                # for i, weight in enumerate(networks_trained[0].parameters()):
                #     if i == 3:
                #         print(weight)
                #         break

                self.learner.meta_step += 1

            # After Meta-Run start the Evaluation of the obtained weights
            self.eval(eval_networks=networks_trained,learner_algorithm='ppo')


    def reptile(self, parameters_batched, hyperparameters):
        print("Perfoming reptile update!")
        # Calculate Gradient for Reptile update
        # grad = theta_1 - theta_0
        # Set the gradient of init_network to perform Meta-update

        # We go through all batches
        for i, parameters in enumerate(parameters_batched):
            # We set the gradient with the difference of weights according to reptile update
            if i == 0:

                for network_parameters, init_network in zip(parameters,
                                                            self.learner.train_networks):
                    for param_t, param_init in zip(network_parameters, init_network.parameters()):
                        # Set gradients to last gradients of Training
                        try:
                            param_init.grad = (param_init.data - param_t.data) / hyperparameters['learning_rate']
                        except TypeError:
                            pass
            else:
                # If more than 1 Task per Batch: We accumulate the gradient
                for network_parameters, init_network in zip(parameters,
                                                            self.learner.train_networks):
                    for param_t, param_init in zip(network_parameters, init_network.parameters()):
                        # Set gradients to last gradients of Training
                        try:
                            param_init.grad += (param_init.data - param_t.data) / hyperparameters['learning_rate']
                        except TypeError:
                            pass

        # We need to divide the accumulated Meta-Gradient with the number of Tasks per Meta-Batch
        for network in self.learner.train_networks:
            for param in network.parameters():
                try:
                    param.grad = param.grad / meta_batch_size
                except TypeError:
                    pass

    def fomaml(self, networks_batch, parameters_batched, hyperparameters: {}):
        # Get Gradient for First-Order MAML update
        # grad = grad(Last_update)
        # Set the gradient of init_network to perform Meta-update
        # Choose between separate_fomaml and shared_fomaml
        print("Perfoming fomaml update!")
        if hyperparameters['separate_fomaml']:
            print("Separate Tail FOMAML")

            # We create a new task for separate Tail FOMAML
            # We train on this new Task (Evaluation procedure)
            difficulty = np.where(np.random.multinomial(1, hyperparameters['task_dist']) == 1)[0][0] + 1
            seed = np.random.randint(0, 100)
            task_number = difficulty * 1000 + seed

            self.learner.set_env_parameters(difficulty=difficulty, maze_rows=3, maze_cols=3, agent_x=0, agent_z=0,
                                            target_x=2, target_z=2, random_agent=0, random_target=0, maze_seed=seed,
                                            agent_rot=0, joint_training=False, enable_heatmap=True, enable_sight_cone=True)

            max_steps_temp = hyperparameters['max_steps']
            hyperparameters['max_steps'] = 70000
            self.learner.set_hyperparameters(hyperparameters)

            # Perform FOMAML Update
            # Go through all networks of every task per batch
            for i, trained_networks in enumerate(networks_batch):
                if i == 0:
                    # Perform evaluation on new task with the weights obtained
                    networks_trained, parameters_trained, _ = self.learner.train(task_number=task_number,
                                       run_id=self.run_id+"_batch_"+str(i)+"_fomaml_eval",
                                       init_networks=trained_networks)
                    # Set the gradient to the last gradient of the evaluation task
                    for network_parameters, init_network in zip(parameters_trained, self.learner.train_networks):
                        for param_t, param_init in zip(network_parameters, init_network.parameters()):
                            # Set gradients to last gradients of Training
                            param_init.grad = param_t.grad
                else:
                    # For Meta-Batch Size greater than 1, accumulate Meta-Gradient
                    networks_trained, parameters_trained, _ = self.learner.train(task_number=task_number,
                                       run_id=self.run_id+"_batch_"+str(i)+"_fomaml_eval",
                                       init_networks=trained_networks)
                    for network_parameters, init_network in zip(parameters_trained, self.learner.train_networks):
                        for param_t, param_init in zip(network_parameters, init_network.parameters()):
                            # Set gradients to last gradients of Training
                            try:
                                param_init.grad += param_t.grad
                            except TypeError:
                                pass
            # Divide by the meta_batch_size
            for network in self.learner.train_networks:
                for param in network.parameters():
                    try:
                        param.grad = param.grad / meta_batch_size
                    except TypeError:
                        pass

            hyperparameters['max_steps'] = max_steps_temp
            self.learner.set_hyperparameters(hyperparameters)
        else:
            print("Shared Tail FOMAML")
            # Set the Meta-Gradient to the Last Gradient of the Networks of the Trained Tasks
            counter = 0
            for parameters in parameters_batched:
                if counter == 0:
                    for network_parameters, init_network in zip(parameters,
                                                                self.learner.train_networks):
                        for param_t, param_init in zip(network_parameters, init_network.parameters()):
                            # Set gradients to last gradients of Training
                            param_init.grad = param_t.grad
                else:
                    for network_parameters, init_network in zip(parameters,
                                                                self.learner.train_networks):
                        for param_t, param_init in zip(network_parameters, init_network.parameters()):
                            # Set gradients to last gradients of Training
                            try:
                                param_init.grad += param_t.grad
                            except TypeError:
                                pass
                counter += 1

                for network in self.learner.train_networks:
                    for param in network.parameters():
                        try:
                            param.grad = param.grad / meta_batch_size
                        except TypeError:
                            pass

    def eval(self, eval_networks, learner_algorithm):
        # Start Evaluation
        # Evaluating Finished Meta-Run on pre-defined Tasks
        learner.set_num_envs(learner_algorithm, 1)

        if learner_algorithm == 'ppo':
            # Set Hyperparameters for Evaluation Run
            hyperparameters = self.get_ppo_hyperparameters()
            hyperparameters['max_steps'] = 1000000
            hyperparameters['batch_size'] = 4096
            hyperparameters['buffer_size'] = 60000
            hyperparameters['summary_freq'] = 50000
            self.learner.set_hyperparameters(hyperparameters)
        elif learner_algorithm == 'sac':
            pass
        elif learner_algorithm == 'rainbow':
            pass

        seed = -1
        # Get the Trained Networks and copy them, to use them on each Evaluation Task
        networks_before = deepcopy(eval_networks)
        for difficulty in range(1, 5):
            # Print Weights for Debugging
            # print("Eval weights: ")
            # for i, weight in enumerate(networks_before[0].parameters()):
            #     if i == 3:
            #         print(weight)
            #         break

            # Get a task of the corresponding Difficulty of Evaluation and Perform 3 Runs on it.
            task_number = difficulty * 1000 + seed
            self.learner.set_env_parameters(difficulty=difficulty, maze_rows=3, maze_cols=3, agent_x=0, agent_z=0,
                                            target_x=2, joint_training=False,
                                            target_z=2, random_agent=0, random_target=0, maze_seed=seed, agent_rot=0,
                                            enable_heatmap=True, enable_sight_cone=True)
            # Start 3 Runs because of high Variance
            self.learner.train(task_number=task_number,
                               run_id=self.run_id + "_eval_task_"+str(difficulty)+"_0",
                               init_networks=networks_before)

            self.learner.train(task_number=task_number,
                               run_id=self.run_id + "_eval_task_"+str(difficulty)+"_1",
                               init_networks=networks_before)

            self.learner.train(task_number=task_number,
                               run_id=self.run_id + "_eval_task_"+str(difficulty)+"_2",
                               init_networks=networks_before)


if __name__ == '__main__':

    ### Set Training-Parameters per Command-line
    # Else: Hard-Coding
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
    meta_batch_size = int(sys.argv[7])
    tmp_separate_fomaml = int(sys.argv[8])
    separate_fomaml = True if tmp_separate_fomaml == 1 else False
    task_dist_number = int(sys.argv[9])

    # Predifned Task Distributions for Meta-Learning
    task_dist = [[0.4, 0.3, 0.15, 0.15], [0.8, 0.1, 0.05, 0.05], [0.05, 0.05, 0.1, 0.8],[0.25, 0.25, 0.25, 0.25]]
    print("Current Run ID: " + run_id)

    # Get Cuda Device for Gpu support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    # Initialize new Learner of Type 'ppo', 'sac' or 'rainbow'
    learner = None
    if learner_type == 'ppo':
        meta_writer = SummaryWriter("results/" + run_id + "_INFO_PPO")
        learner = MLAgentsTrainer(run_id=run_id, rl_algorithm='ppo')
    elif learner_type == 'sac':
        meta_writer = SummaryWriter("results/" + run_id + "_INFO_SAC")
        learner = MLAgentsTrainer(run_id=run_id, rl_algorithm='sac')
    elif learner_type == 'rainbow':
        meta_writer = SummaryWriter("results/" + run_id + "_INFO_RBW")
        learner = RainbowMetaLearner(device=device, is_meta_learning=True)
    else:
        exit(-1)

    meta_learner = MetaLearner(learner, meta_writer, run_id, device)

    # Perform Joint-Training or Meta-Learning
    if meta_learn_algorithm == 'joint_train':
        meta_learner.joint_train(num_updates=10, task_dist=task_dist[task_dist_number])
    else:
        meta_learner.meta_learn(learner_type, meta_learn_algorithm, meta_optimizer, num_meta_updates=num_meta_updates,
                             meta_lr=meta_lr, meta_batch_size=meta_batch_size, task_dist=task_dist[task_dist_number], separate_fomaml=separate_fomaml)

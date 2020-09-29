from mlagents.trainers.learn import run_training
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.cli_utils import parser
from mlagents_envs import logging_util
import json
from mlagents import tf_utils
import torch.optim as optim
from itertools import chain
import numpy as np
from mlagents.trainers.settings import EnvironmentParameterSettings, Lesson, ConstantSettings

logger = logging_util.get_logger(__name__)
log_level = logging_util.INFO
tf_utils.set_warnings_enabled(False)
logging_util.set_log_level(log_level)


class MLAgentsTrainer:
    def __init__(self, run_id: str, rl_algorithm: str):
        self.meta_step = 0
        self.run_id = run_id
        self.rl_algorithm = rl_algorithm
        base_port = str(np.random.randint(1000, 10000))
        if rl_algorithm == 'ppo':
            args = parser.parse_args(["C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\tools\\ppo.yaml",
                                      "--env=C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\mMaze\\RLProject.exe",
                                      "--num-envs=4", "--torch", "--run-id=init", "--base-port="+base_port, "--force"])
        elif rl_algorithm == 'sac':
            args = parser.parse_args(["C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\tools\\sac.yaml",
                                      "--env=C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\mMaze\\RLProject.exe",
                                      "--num-envs=4", "--torch", "--run-id=init", "--base-port="+base_port, "--force"])
        options = RunOptions.from_argparse(args)
        self.options = options
        # Get inital Networks and weights to Meta learn #

    def init_optimizer(self):
        max_steps = self.options.behaviors['Brain'].max_steps
        time_horizon = self.options.behaviors['Brain'].time_horizon
        self.init_lr = self.options.behaviors['Brain'].hyperparameters.learning_rate
        self.options.behaviors['Brain'].max_steps = 100
        self.options.behaviors['Brain'].time_horizon = 50
        self.init_networks, self.init_params, _ = run_training(run_seed=0, options=self.options, init_networks=None,
                                                            meta_step=self.meta_step, task_number=0)
        self.options.behaviors['Brain'].max_steps = max_steps
        self.options.behaviors['Brain'].time_horizon = time_horizon
        self.options.checkpoint_settings.force = False

    def set_hyperparameters(self, hyperparameters: {}):
        self.options.behaviors['Brain'].hyperparameters.buffer_size = hyperparameters['buffer_size']
        self.options.behaviors['Brain'].hyperparameters.batch_size = hyperparameters['batch_size']
        self.options.behaviors['Brain'].hyperparameters.learning_rate = hyperparameters['learning_rate']
        # self.options.behaviors['Brain'].hyperparameters.learning_rate_schedule = learning_rate_schedule
        self.options.behaviors['Brain'].time_horizon = hyperparameters['time_horizon']
        self.options.behaviors['Brain'].network_settings.hidden_units = hyperparameters['layer_size']
        self.options.behaviors['Brain'].network_settings.num_layers = hyperparameters['hidden_layers']
        self.options.behaviors['Brain'].max_steps = hyperparameters['max_steps']
        if not hyperparameters['decay_lr']:
            self.options.behaviors['Brain'].hyperparameters.learning_rate_schedule = self.options.behaviors['Brain'].hyperparameters.learning_rate_schedule.CONSTANT
        else:
            self.options.behaviors['Brain'].hyperparameters.learning_rate_schedule = self.options.behaviors['Brain'].hyperparameters.learning_rate_schedule.LINEAR
        if self.rl_algorithm == 'sac':
            self.options.behaviors['Brain'].hyperparameters.buffer_init_steps = 8000
            self.options.behaviors['Brain'].hyperparameters.init_entcoef = 0.3

    def set_env_parameters(self, maze_rows: int, maze_cols: int, agent_x: int, agent_z: int, target_x: int,
                           target_z: int,
                           random_agent: int, random_target: int, maze_seed: int, enable_sight_cone: bool,
                           enable_heatmap: bool):
        self.options.environment_parameters['maze_rows'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(maze_rows)), "maze_rows", completion_criteria=None)])
        self.options.environment_parameters['maze_cols'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(maze_cols)), "maze_cols", completion_criteria=None)])
        self.options.environment_parameters['agent_x'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(agent_x)), "agent_x", completion_criteria=None)])
        self.options.environment_parameters['agent_z'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(agent_z)), "agent_z", completion_criteria=None)])
        self.options.environment_parameters['target_x'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(target_x)), "target_x", completion_criteria=None)])
        self.options.environment_parameters['target_z'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(target_z)), "target_z", completion_criteria=None)])
        self.options.environment_parameters['random_agent'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(random_agent)), "random_agent", completion_criteria=None)])
        self.options.environment_parameters['random_target'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(random_target)), "random_target", completion_criteria=None)])
        self.options.environment_parameters['maze_seed'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(maze_seed)), "maze_seed", completion_criteria=None)])
        if enable_heatmap:
            self.options.environment_parameters['enable_heatmap'] = EnvironmentParameterSettings(
                [Lesson(ConstantSettings(seed=0, value=1.0), "enable_heatmap", completion_criteria=None)])
        else:
            self.options.environment_parameters['enable_heatmap'] = EnvironmentParameterSettings(
                [Lesson(ConstantSettings(seed=0, value=0.0), "enable_heatmap", completion_criteria=None)])
        if enable_sight_cone:
            self.options.environment_parameters['enable_sight_cone'] = EnvironmentParameterSettings(
                [Lesson(ConstantSettings(seed=0, value=1.0), "enable_sight_cone", completion_criteria=None)])
        else:
            self.options.environment_parameters['enable_sight_cone'] = EnvironmentParameterSettings(
                [Lesson(ConstantSettings(seed=0, value=0.0), "enable_sight_cone", completion_criteria=None)])

    def train(self, task_number: int, run_id: str = "ppo_", init_networks=None):
        self.options.checkpoint_settings.run_id = run_id + "_step_"+str(self.meta_step)
        print(self.options.checkpoint_settings.run_id)
        logger.debug("Configuration for this run:")
        logger.debug(json.dumps(self.options.as_dict(), indent=4))
        trained_networks, trained_parameters, losses = run_training(run_seed=0, options=self.options,
                                                            init_networks=init_networks, meta_step=self.meta_step, task_number=task_number)
        return trained_networks, trained_parameters, losses

if __name__ == '__main__':
    ml_trainer = MLAgentsTrainer(run_id="Meta_SAC_", meta_lr=0.8, meta_optimizer='SGD', rl_algorithm='ppo')
    ml_trainer.meta_learn(algorithm='fomaml', num_meta_updates=100)

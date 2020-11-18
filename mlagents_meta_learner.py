from mlagents.trainers.learn import run_training
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.cli_utils import parser
from mlagents_envs import logging_util
import json
from mlagents import tf_utils
import numpy as np
from mlagents.trainers.settings import EnvironmentParameterSettings, Lesson, ConstantSettings

logger = logging_util.get_logger(__name__)
log_level = logging_util.INFO
tf_utils.set_warnings_enabled(False)
logging_util.set_log_level(log_level)

# Meta-Learner Implementation for ML-Agents Toolkit
class MLAgentsTrainer:
    def __init__(self, run_id: str, rl_algorithm: str):
        self.meta_step = 0
        self.run_id = run_id
        self.rl_algorithm = rl_algorithm
        base_port = str(np.random.randint(1000, 10000))
        if rl_algorithm == 'ppo':
            args = parser.parse_args(["C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\tools\\ppo.yaml",
                                      "--env=C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\mMaze_cont_ref\\RLProject.exe",
                                      "--num-envs=5", "--torch", "--run-id=init", "--base-port="+base_port, "--force",
                                      "--force"])
        elif rl_algorithm == 'sac':
            args = parser.parse_args(["C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\tools\\sac.yaml",
                                      "--env=C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\mMaze_cont_ref\\RLProject.exe",
                                      "--num-envs=5", "--run-id=init", "--base-port="+base_port, "--force", "--torch"])
        options = RunOptions.from_argparse(args)
        self.options = options
        # Get inital Networks and weights to Meta learn #

    def set_num_envs(self, rl_algorithm: str, num_envs: int):
        base_port = str(np.random.randint(1000, 10000))
        if rl_algorithm == 'ppo':
            args = parser.parse_args(["C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\tools\\ppo.yaml",
                                      "--env=C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\mMaze_cont_ref\\RLProject.exe",
                                      "--num-envs="+str(num_envs), "--torch", "--run-id=init", "--base-port="+base_port, "--force",
                                      "--force"])
        elif rl_algorithm == 'sac':
            args = parser.parse_args(["C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\tools\\sac.yaml",
                                      "--env=C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\mMaze_cont_ref\\RLProject.exe",
                                      "--num-envs="+str(num_envs), "--run-id=init", "--base-port="+base_port, "--force", "--torch"])
        options = RunOptions.from_argparse(args)
        self.options = options

    def init_optimizer(self):
        # Initialize Optimizers
        max_steps = self.options.behaviors['Brain'].max_steps
        time_horizon = self.options.behaviors['Brain'].time_horizon
        self.init_lr = self.options.behaviors['Brain'].hyperparameters.learning_rate
        self.options.behaviors['Brain'].max_steps = 100
        self.options.behaviors['Brain'].time_horizon = 50
        self.train_networks, self.init_params, _ = run_training(run_seed=0, options=self.options, init_networks=None,
                                                            meta_step=self.meta_step, task_number=0)
        self.options.behaviors['Brain'].max_steps = max_steps
        self.options.behaviors['Brain'].time_horizon = time_horizon
        self.options.checkpoint_settings.force = True

    def set_hyperparameters(self, hyperparameters: {}):
        # Set Hyperprameters
        if self.options.behaviors['Brain'].trainer_type == self.options.behaviors['Brain'].trainer_type.PPO:
            self.options.behaviors['Brain'].hyperparameters.num_epoch = hyperparameters['num_epochs']

        self.options.behaviors['Brain'].hyperparameters.learning_rate = hyperparameters['learning_rate']
        self.options.behaviors['Brain'].hyperparameters.buffer_size = hyperparameters['buffer_size']
        self.options.behaviors['Brain'].hyperparameters.batch_size = hyperparameters['batch_size']
        self.options.behaviors['Brain'].time_horizon = hyperparameters['time_horizon']
        self.options.behaviors['Brain'].network_settings.hidden_units = hyperparameters['layer_size']
        self.options.behaviors['Brain'].network_settings.num_layers = hyperparameters['hidden_layers']
        self.options.behaviors['Brain'].max_steps = hyperparameters['max_steps']
        self.options.behaviors['Brain'].summary_freq = hyperparameters['summary_freq']
        if not hyperparameters['decay_lr']:
            self.options.behaviors['Brain'].hyperparameters.learning_rate_schedule = self.options.behaviors['Brain'].hyperparameters.learning_rate_schedule.CONSTANT
        else:
            self.options.behaviors['Brain'].hyperparameters.learning_rate_schedule = self.options.behaviors['Brain'].hyperparameters.learning_rate_schedule.LINEAR

    def set_env_parameters(self, maze_rows: int, maze_cols: int, agent_x: int, agent_z: int, target_x: int,
                           target_z: int, agent_rot: float, difficulty: int,
                           random_agent: int, random_target: int, maze_seed: int, enable_sight_cone: bool,
                           enable_heatmap: bool, joint_training: bool):
        # Set Env Training Parameters (Maze_size, Position of Agent/Target, etc.)
        self.options.environment_parameters['difficulty'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(difficulty)), "difficulty", completion_criteria=None)])
        self.options.environment_parameters['maze_rows'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(maze_rows)), "maze_rows", completion_criteria=None)])
        self.options.environment_parameters['maze_cols'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(maze_cols)), "maze_cols", completion_criteria=None)])
        self.options.environment_parameters['agent_x'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(agent_x)), "agent_x", completion_criteria=None)])
        self.options.environment_parameters['agent_z'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(agent_z)), "agent_z", completion_criteria=None)])
        self.options.environment_parameters['agent_rot'] = EnvironmentParameterSettings(
            [Lesson(ConstantSettings(seed=0, value=float(agent_rot)), "agent_rot", completion_criteria=None)])
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

        if joint_training:
            self.options.environment_parameters['joint_train'] = EnvironmentParameterSettings(
                [Lesson(ConstantSettings(seed=0, value=1.0), "joint_train", completion_criteria=None)])
        else:
            self.options.environment_parameters['joint_train'] = EnvironmentParameterSettings(
                [Lesson(ConstantSettings(seed=0, value=0.0), "joint_train", completion_criteria=None)])

    def train(self, task_number: int, run_id: str = "ppo_", init_networks=None, meta_eval=False):
        # Start Inner-Loop Deep RL Training
        self.options.checkpoint_settings.run_id = run_id + "_step_"+str(self.meta_step)
        print(self.options.checkpoint_settings.run_id)
        logger.debug("Configuration for this run:")
        logger.debug(json.dumps(self.options.as_dict(), indent=4))
        if self.options.env_settings.seed == -1:
            run_seed = np.random.randint(0, 10000)
        else:
            run_seed = self.options.env_settings.seed
        trained_networks, trained_parameters, meta_loss = run_training(run_seed=run_seed,options=self.options, init_networks=init_networks, meta_step=self.meta_step, task_number=task_number, meta_eval=meta_eval)
        return trained_networks, trained_parameters, meta_loss

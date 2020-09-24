from mlagents.trainers.learn import run_training
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.cli_utils import parser
from mlagents_envs import logging_util
import json
from mlagents import tf_utils
import torch.optim as optim
from itertools import chain
from copy import deepcopy


logger = logging_util.get_logger(__name__)
log_level = logging_util.INFO
tf_utils.set_warnings_enabled(False)
logging_util.set_log_level(log_level)

class MLAgentsTrainer:
    def __init__(self):
        self.meta_step = 0
        args = parser.parse_args(["C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\tools\\sac-7.yaml",
                                  "--env=C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\mMaze\\RLProject.exe",
                                  "--num-envs=4", "--torch", "--run-id=init", "--base-port=3000", "--force"])
        options = RunOptions.from_argparse(args)
        self.options = options
        max_steps = options.behaviors['Brain'].max_steps
        options.behaviors['Brain'].max_steps = 1000
        self.init_networks = run_training(run_seed=0, options=options, init_networks=None, meta_step=self.meta_step)
        options.behaviors['Brain'].max_steps = max_steps
        options.checkpoint_settings.force = False
        optimizer_params = []
        for network in self.init_networks:
            try:
                optimizer_params.extend(network.parameters())
            except:
                pass
        self.meta_optimizer = optim.Adam(chain(optimizer_params), lr=0.01)

    def train(self, init_networks=None):
        self.options.checkpoint_settings.run_id = "SAC_meta_learn_" + str(self.meta_step)
        logger.debug("Configuration for this run:")
        logger.debug(json.dumps(self.options.as_dict(), indent=4))
        final_weights = run_training(run_seed=0, options=self.options, init_networks=init_networks, meta_step=self.meta_step)
        return final_weights


if __name__ == '__main__':

    ml_trainer = MLAgentsTrainer()
    trained_networks = ml_trainer.init_networks
    for i in range(20):

        print("Weights before Training: ")
        print(trained_networks[0].state_dict())
        networks_before = deepcopy(trained_networks)
        networks_trained = ml_trainer.train(init_networks=networks_before)
        print("Weights after Training: ")
        print(trained_networks[0].state_dict())
        for network_trained, network_before in zip(networks_trained, networks_before):
            try:
                for param_t, param_b in zip(network_trained.parameters(), network_before.parameters()):
                    param_t = param_b
                    print(param_t.grad)
            except AttributeError:
                pass
        print("Weights after Meta update: ")
        print(trained_networks[0].state_dict())

        ml_trainer.meta_step += 1

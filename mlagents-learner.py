from mlagents.trainers.learn import run_training
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.cli_utils import StoreConfigFile, DetectDefault, parser

args = parser.parse_args(["C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\tools\\ppo-7.yaml", "--env=C:\\Users\\Sebastian\\Desktop\\RLUnity\\Meta-Learner\\mMaze\\RLProject.exe",
                          "--num-envs=4", "--force", "--torch", "--run-id=ppo_torch_ml"])
options = RunOptions.from_argparse(args)

if __name__ == '__main__':
    final_weights = run_training(run_seed=0, options=options)
    print(final_weights)


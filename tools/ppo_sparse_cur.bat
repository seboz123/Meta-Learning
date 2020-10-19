REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion
REM Start PPO Runs:

REM BATCH

call mlagents-learn tools/ppo_cur_sparse.yaml --run-id=PPO_sparse_0 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=0 --torch --base-port=100
call mlagents-learn tools/ppo_cur_sparse.yaml --run-id=PPO_sparse_1 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=1 --torch --base-port=100
call mlagents-learn tools/ppo_cur_sparse.yaml --run-id=PPO_sparse_2 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=2 --torch --base-port=100

call mlagents-learn tools/ppo_cur0.yaml --run-id=PPO_sparse_cur_0_0 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=0 --torch --base-port=100
call mlagents-learn tools/ppo_cur0.yaml --run-id=PPO_sparse_cur_0_1 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=1 --torch --base-port=100
call mlagents-learn tools/ppo_cur0.yaml --run-id=PPO_sparse_cur_0_2 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=2 --torch --base-port=100

call mlagents-learn tools/ppo_cur1.yaml --run-id=PPO_sparse_cur_1_0 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=0 --torch --base-port=100
call mlagents-learn tools/ppo_cur1.yaml --run-id=PPO_sparse_cur_1_1 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=1 --torch --base-port=100
call mlagents-learn tools/ppo_cur1.yaml --run-id=PPO_sparse_cur_1_2 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=2 --torch --base-port=100

call mlagents-learn tools/ppo_cur2.yaml --run-id=PPO_sparse_cur_2_0 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=0 --torch --base-port=100
call mlagents-learn tools/ppo_cur2.yaml --run-id=PPO_sparse_cur_2_1 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=1 --torch --base-port=100
call mlagents-learn tools/ppo_cur2.yaml --run-id=PPO_sparse_cur_2_2 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --seed=2 --torch --base-port=100
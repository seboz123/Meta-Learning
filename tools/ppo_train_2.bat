REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion
REM Start PPO Runs:

REM BATCH

call mlagents-learn tools/ppo_rew.yaml --run-id=PPO_rew_sparse_0 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0
call mlagents-learn tools/ppo_rew.yaml --run-id=PPO_rew_sparse_1 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1
call mlagents-learn tools/ppo_rew.yaml --run-id=PPO_rew_sparse_2 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2

call mlagents-learn tools/ppo_rew.yaml --run-id=PPO_rew_coll_0 --env=mMaze_rew_coll/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0
call mlagents-learn tools/ppo_rew.yaml --run-id=PPO_rew_coll_1 --env=mMaze_rew_coll/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1
call mlagents-learn tools/ppo_rew.yaml --run-id=PPO_rew_coll_2 --env=mMaze_rew_coll/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2

call mlagents-learn tools/ppo_rew.yaml --run-id=PPO_rew_velo_0 --env=mMaze_rew_velo/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0
call mlagents-learn tools/ppo_rew.yaml --run-id=PPO_rew_velo_1 --env=mMaze_rew_velo/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1
call mlagents-learn tools/ppo_rew.yaml --run-id=PPO_rew_velo_2 --env=mMaze_rew_velo/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2

call mlagents-learn tools/ppo.yaml --run-id=PPO_rew_heatmap_0 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0
call mlagents-learn tools/ppo.yaml --run-id=PPO_rew_heatmap_1 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1
call mlagents-learn tools/ppo.yaml --run-id=PPO_rew_heatmap_2 --env=mMaze_rew_sparse/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2

call mlagents-learn tools/ppo.yaml --run-id=PPO_rew_combi_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0
call mlagents-learn tools/ppo.yaml --run-id=PPO_rew_combi_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1
call mlagents-learn tools/ppo.yaml --run-id=PPO_rew_combi_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2


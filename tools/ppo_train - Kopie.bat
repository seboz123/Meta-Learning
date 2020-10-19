REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion
REM Start PPO Runs:

REM BATCH

call mlagents-learn tools/ppo_cam.yaml --run-id=PPO_cam_0 --env=mMaze_cam/RLProject.exe --num-envs=6 --time-scale=20 --force --seed=0
call mlagents-learn tools/ppo_cam.yaml --run-id=PPO_cam_1 --env=mMaze_cam/RLProject.exe --num-envs=6 --time-scale=20 --force --seed=1
call mlagents-learn tools/ppo_cam.yaml --run-id=PPO_cam_2 --env=mMaze_cam/RLProject.exe --num-envs=6 --time-scale=20 --force --seed=2

call python MetaLearner.py PPO_META_3_rep ppo reptile sgd 100 0.00002 3
call python MetaLearner.py PPO_META_3_fomaml ppo fomaml sgd 100 0.005 3


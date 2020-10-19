REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"


setlocal EnableDelayedExpansion
REM Start PPO Runs:

call mlagents-learn tools/ppo_cam.yaml --run-id=cam_PPO_2 --env=mMaze_cam/RLProject.exe --num-envs=6 --time-scale=20 --force --seed=2


pause
REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion
REM Start SAC Runs:

call mlagents-learn tools/sac/rew_sac_0.yaml --run-id=rew_sac_0_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/rew_sac_0.yaml --run-id=rew_sac_0_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/rew_sac_0.yaml --run-id=rew_sac_0_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

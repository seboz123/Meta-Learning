REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion
REM Start SAC Runs:

REM BATCH

call mlagents-learn tools/sac/batch_sac_0.yaml --run-id=batch_sac_0_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/batch_sac_0.yaml --run-id=batch_sac_0_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/batch_sac_0.yaml --run-id=batch_sac_0_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/batch_sac_1.yaml --run-id=batch_sac_1_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/batch_sac_1.yaml --run-id=batch_sac_1_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/batch_sac_1.yaml --run-id=batch_sac_1_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/batch_sac_2.yaml --run-id=batch_sac_2_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/batch_sac_2.yaml --run-id=batch_sac_2_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/batch_sac_2.yaml --run-id=batch_sac_2_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

REM BUFFER

call mlagents-learn tools/sac/buffer_sac_0.yaml --run-id=buffer_sac_0_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/buffer_sac_0.yaml --run-id=buffer_sac_0_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/buffer_sac_0.yaml --run-id=buffer_sac_0_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/buffer_sac_1.yaml --run-id=buffer_sac_1_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/buffer_sac_1.yaml --run-id=buffer_sac_1_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/buffer_sac_1.yaml --run-id=buffer_sac_1_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/buffer_sac_2.yaml --run-id=buffer_sac_2_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/buffer_sac_2.yaml --run-id=buffer_sac_2_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/buffer_sac_2.yaml --run-id=buffer_sac_2_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/buffer_sac_3.yaml --run-id=buffer_sac_3_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/buffer_sac_3.yaml --run-id=buffer_sac_3_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/buffer_sac_3.yaml --run-id=buffer_sac_3_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

REM INIT

call mlagents-learn tools/sac/init_sac_0.yaml --run-id=init_sac_0_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/init_sac_0.yaml --run-id=init_sac_0_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/init_sac_0.yaml --run-id=init_sac_0_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/init_sac_1.yaml --run-id=init_sac_1_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/init_sac_1.yaml --run-id=init_sac_1_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/init_sac_1.yaml --run-id=init_sac_1_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/init_sac_2.yaml --run-id=init_sac_2_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/init_sac_2.yaml --run-id=init_sac_2_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/init_sac_2.yaml --run-id=init_sac_2_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

REM REW

call mlagents-learn tools/sac/rew_sac_0.yaml --run-id=rew_sac_0_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/rew_sac_0.yaml --run-id=rew_sac_0_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/rew_sac_0.yaml --run-id=rew_sac_0_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/rew_sac_1.yaml --run-id=rew_sac_1_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/rew_sac_1.yaml --run-id=rew_sac_1_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/rew_sac_1.yaml --run-id=rew_sac_1_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/rew_sac_2.yaml --run-id=rew_sac_2_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/rew_sac_2.yaml --run-id=rew_sac_2_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/rew_sac_2.yaml --run-id=rew_sac_2_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch

call mlagents-learn tools/sac/rew_sac_3.yaml --run-id=rew_sac_3_0 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=0 --torch
call mlagents-learn tools/sac/rew_sac_3.yaml --run-id=rew_sac_3_1 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=1 --torch
call mlagents-learn tools/sac/rew_sac_3.yaml --run-id=rew_sac_3_2 --env=mMaze_cont_ref/RLProject.exe --num-envs=6 --time-scale=20 --force --no-graphics --seed=2 --torch
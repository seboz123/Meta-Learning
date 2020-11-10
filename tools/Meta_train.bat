REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion

call python MetaLearner.py PPO_META_100_up_reptile_adam_mlr_00001 ppo reptile adam 100 0.0001 1 0 0
call python MetaLearner.py PPO_META_100_up_reptile_adam_mlr_000001 ppo reptile adam 100 0.00001 1 0 0
call python MetaLearner.py PPO_META_100_up_reptile_adam_mlr_000005 ppo reptile adam 100 0.00005 1 0 0
pause

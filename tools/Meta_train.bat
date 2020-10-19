REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion

call python MetaLearner.py PPO_META_10_updates_3 ppo reptile sgd 10 0.00002 3
call python MetaLearner.py PPO_META_30_updates_3 ppo reptile sgd 30 0.00002 3


pause



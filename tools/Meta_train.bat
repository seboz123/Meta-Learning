REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion

call python MetaLearner.py PPO_META_30_up_fomaml_mlr_001_0 ppo fomaml sgd 30 0.001 1
call python MetaLearner.py PPO_META_30_up_fomaml_mlr_001_1 ppo fomaml sgd 30 0.001 1
call python MetaLearner.py PPO_META_30_up_fomaml_mlr_001_2 ppo fomaml sgd 30 0.001 1

call python MetaLearner.py PPO_META_30_up_fomaml_mlr_003_0 ppo fomaml sgd 30 0.003 1
call python MetaLearner.py PPO_META_30_up_fomaml_mlr_003_1 ppo fomaml sgd 30 0.003 1
call python MetaLearner.py PPO_META_30_up_fomaml_mlr_003_2 ppo fomaml sgd 30 0.003 1

call python MetaLearner.py PPO_META_30_up_fomaml_mlr_005_0 ppo fomaml sgd 30 0.005 1
call python MetaLearner.py PPO_META_30_up_fomaml_mlr_005_1 ppo fomaml sgd 30 0.005 1
call python MetaLearner.py PPO_META_30_up_fomaml_mlr_005_2 ppo fomaml sgd 30 0.005 1





pause



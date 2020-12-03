REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion

call python MetaLearner.py PPO_META_50_up_fomaml_mlr_002 ppo fomaml sgd 50 0.002 1 0 0

pause

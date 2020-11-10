REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion
REM Start RBW Runs:

call python meta_learner_rainbow.py rbw_best_0 1500000 10000 2048 0.2 5 0.3 0.6







REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion
REM Start RBW Runs:

call python meta_learner_rainbow.py rbw_alpha_03_1 10000 2048 0.2 5 0.3 0.4
call python meta_learner_rainbow.py rbw_alpha_03_2 10000 2048 0.2 5 0.3 0.4
call python meta_learner_rainbow.py rbw_alpha_03_3 10000 2048 0.2 5 0.3 0.4

call python meta_learner_rainbow.py rbw_alpha_06_1 10000 2048 0.2 5 0.6 0.4
call python meta_learner_rainbow.py rbw_alpha_06_2 10000 2048 0.2 5 0.6 0.4
call python meta_learner_rainbow.py rbw_alpha_06_3 10000 2048 0.2 5 0.6 0.4

call python meta_learner_rainbow.py rbw_alpha_08_1 10000 2048 0.2 5 0.8 0.4
call python meta_learner_rainbow.py rbw_alpha_08_2 10000 2048 0.2 5 0.8 0.4
call python meta_learner_rainbow.py rbw_alpha_08_3 10000 2048 0.2 5 0.8 0.4





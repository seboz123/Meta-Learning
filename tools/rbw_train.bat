REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion
REM Start RBW Runs:

call python meta_learner_rainbow.py rbw_batch_512_0 500000 10000 512 0.2 5 0.3 0.6
call python meta_learner_rainbow.py rbw_batch_512_1 500000 10000 512 0.2 5 0.3 0.6
call python meta_learner_rainbow.py rbw_batch_512_2 500000 10000 512 0.2 5 0.3 0.6

call python meta_learner_rainbow.py rbw_batch_1024_0 500000 10000 1024 0.2 5 0.3 0.6
call python meta_learner_rainbow.py rbw_batch_1024_1 500000 10000 1024 0.2 5 0.3 0.6
call python meta_learner_rainbow.py rbw_batch_1024_2 500000 10000 1024 0.2 5 0.3 0.6




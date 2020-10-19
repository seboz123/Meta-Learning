REM Start Tensorboard and Conda Environment

call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate mlagents-release-7
cd "C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner"

setlocal EnableDelayedExpansion
REM Start RBW Runs:

call python meta_learner_rainbow.py rbw_10k_0 10000 0.2 3 0.2
call python meta_learner_rainbow.py rbw_10k_1 10000 0.2 3 0.2
call python meta_learner_rainbow.py rbw_10k_2 10000 0.2 3 0.2

call python meta_learner_rainbow.py rbw_40k_0 40000 0.2 3 0.2
call python meta_learner_rainbow.py rbw_40k_1 40000 0.2 3 0.2
call python meta_learner_rainbow.py rbw_40k_2 40000 0.2 3 0.2

call python meta_learner_rainbow.py rbw_eps_0 20000 0.1 3 0.2
call python meta_learner_rainbow.py rbw_eps_1 20000 0.1 3 0.2
call python meta_learner_rainbow.py rbw_eps_2 20000 0.1 3 0.2

call python meta_learner_rainbow.py rbw_nstep_0 20000 0.2 5 0.2
call python meta_learner_rainbow.py rbw_nstep_1 20000 0.2 5 0.2
call python meta_learner_rainbow.py rbw_nstep_2 20000 0.2 5 0.2

call python meta_learner_rainbow.py rbw_alpha_0 20000 0.2 5 0.4
call python meta_learner_rainbow.py rbw_alpha_1 20000 0.2 5 0.4
call python meta_learner_rainbow.py rbw_alpha_2 20000 0.2 5 0.4


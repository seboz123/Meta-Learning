call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate ml-agents-release-6
start tensorboard --logdir="C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner\results" --port=6006
timeout /t 3
start firefox localhost:6006

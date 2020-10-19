call "C:\Users\Sebastian\anaconda3\Scripts\activate.bat"
call conda activate ml-agents-release-6
start tensorboard --logdir="C:\Users\Sebastian\Desktop\RLUnity\Meta-Learner\results_meta" --port=6007
timeout /t 10
start chrome 127.0.0.1:6007

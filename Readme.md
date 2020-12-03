# Meta-Learner

## Unity Modules

This Project contains different Files in Unity.
Open the scene: maze for the implementation of Agent+Maze
Should work by just pressing play.
The scene contains several prefabs of training stages.
Each one can be changed and adapted.
The MainCamera GameObejct contains a script for managing the cameras in Unity. You can press 't' to toggle between BirdsView and ThirdPersonView. You can press '1' and '2' to get to the next and previous cameras in the scene.

Each Prefab-Stage contains a Maze-GameObject. It contains a script to create the Maze. Here, the script does not directly create the maze, but is rather loaded in the NaviAgent.cs Script.

The Most important GameObject is the FindTargetAgent. It contains the Main Functionality. In the NaviAgent.cs Script, the Maze gets loaded before the Epsiodes, Python Parameters get loaded, and the training process is implemented.

The FindTargetAgent also contains a SightCone for detecting the Target and another Camera under the Direction GameObject for managing Third Person View.

## Python Modules

The Main Module of the Python Implementation is the MetaLearner.py file.

In the MetaLearner.py, the meta_learner_rainbow.py or the mlagents_meta_learner.py files can be loaded for performing rainbow or ppo/sac training. The meta_learner_sac.py and meta_learner_ppo.py files are obsolete. They contain former implementations for ppo and sac but are currently not used.

Different Helper Functions are implemented in utils.py

Neural Networks Models for Rainbow (and old PPO/SAC) are implemented in models.py

buffers.py and segment_tree.py contain the Replay Buffer for Rainbow (and buffers.py old versions for PPO/SAC)

curiosity_module.py contains an implementation of curiosity.

The MetaLearner.py file calls the mlagents_meta_learner.py if ppo/sac are used or the meta_learner_rainbow.py if rainbow is used. mlagents_meta_learner.py calls then the learn.py file of the altered MLAgents repository. The learn.py file calls most importantly the trainer_controller.py file and the rl_trainer.py file for training with ppo/sac. 


```python

```

# Multiagent-Quadruped-Environments

Multiagent-Quadruped-Environments(MQE) is a multi-functional and easy-to-use quadruped-simulation environment based on Isaac Gym that supports multi-agent tasks. Currently, MQE supports following features:

* Interaction between multiple quadrupeds and articulated objects.
* Train high-level planning policy only with built-in walk policy.
* Train locomotion policy with pre-implemented locomotion rewards.
* Build your terrain from blocks like LEGO and your task through a wrapper.
* Click-to-use RL pipeline through [OpenRL](https://github.com/OpenRL-Lab/openrl) on pre-defined cooperative and competitive tasks.

## Installation ##
1. Create a new Python virtual env or conda environment with Python 3.6, 3.7, or 3.8 (3.8 recommended)
2. Install PyTorch and Isaac Gym of any version. Make sure Isaac Gym is available by running
    - `cd examples && python 1080_balls_of_solitude.py`https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Move to the directory of this repository and run
   - `pip install -e .`

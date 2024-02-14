# Multiagent-Quadruped-Environments

Multiagent-Quadruped-Environments(MQE) is a multi-functional and easy-to-use quadruped-simulation environment based on Isaac Gym that supports multi-agent tasks. Currently, MQE supports following features:

* Interaction between multiple quadrupeds and articulated objects.
* Train high-level planning policy only with built-in walk policy.
* Build your terrain from blocks like LEGO.
* Pre-defined cooperative and competitive tasks.
* Click-to-use RL pipeline through [OpenRL](https://github.com/OpenRL-Lab/openrl).

## Installation ##
1. Create a new Python virtual env or conda environment with Python 3.6, 3.7, or 3.8 (3.8 recommended)
2. Install PyTorch and Isaac Gym of any version. Make sure Isaac Gym is available by running the example `cd examples && python 1080_balls_of_solitude.py`
    - `pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
5. Install legged_gym
   - `cd ../legged_gym && pip install -e .`

# Multiagent-Quadruped-Environments

A simulation environment supporting creating multi-agent tasks for quadrupeds.

## Installation ##
1. Create a new Python virtual env or conda environment with Python 3.6, 3.7, or 3.8 (3.8 recommended)
2. Install PyTorch 1.10 with cuda-11.3:
    - `pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 (I didn't test the history version) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`
4. Install rsl_rl (PPO implementation)
   - Using the command to direct to the root path of this repository
   - `cd rsl_rl && pip install -e .` 
5. Install legged_gym
   - `cd ../legged_gym && pip install -e .`

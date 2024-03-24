# ![](docs/static/images/sheep.png) Multi-agent Quadruped Environment

Multi-agent Quadruped Environment(MQE) is a multi-functional and easy-to-use quadruped-simulation environment based on Isaac Gym that supports multi-agent tasks. Currently, MQE supports following features:

* Interaction between multiple quadrupeds and articulated objects.
* Train high-level planning policy only with built-in walk policy.
* Build your terrain from blocks like LEGO.
* Click-to-use RL pipeline through [OpenRL](https://github.com/OpenRL-Lab/openrl) on pre-defined cooperative and competitive tasks.

## Useful Links ##

Project Website: https://ziyanx02.github.io/multiagent-quadruped-environment/

Paper:

## Installation ##
1. Create a new Python virtual env or conda environment with Python 3.6, 3.7, or 3.8 (3.8 recommended)
2. Install PyTorch and Isaac Gym.
    - Install appropriate PyTorch version from https://pytorch.org/
    - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
3. Check Isaac Gym is available by running
    - `cd examples && python 1080_balls_of_solitude.py`
4. Install MQE. Move to the directory of this repository and run
    - `pip install -e .`
5. Check MQE is available by running
    - `python ./test.py`

## Code Structure ##

Environment for each task is defined by:
- a class for controlling objects involved in the task. `./mqe/envs/go1/go1.py` is a base class for Unitree Go1 robot with locomotion policy implemented in [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways). `./mqe/envs/npc/` includes several classes created for different interactive objects.
- a wrapper to specify observations, actions, rewards, and infos. `./mqe/envs/wrappers/` includes several wrappers for reference.
- a config file to specify all the configuration about the environment, including configs for simulation, terrain registration, robot assets, etc. Config files use inheritance. `./mqe/envs/configs/` includes the config files of pre-defined tasks. To explore more available configuration, please check config files in `./mqe/envs/base/`, `./mqe/envs/field/` and `./mqe/envs/go1/`, there should be no intersections between these config files for clearance.

Blocks used in terrain registration is defined in `./mqe/utils/terrain/barrier_track.py`.

## Usage ##
1. Try different tasks

    `python ./test.py`

    - Task could be specified in `./test.py`

2. Train using OpenRL

    `python ./openrl_ws/train.py --algo ALGO_NAME --task TASK_NAME`
    - `--num_envs NUM_ENVS` to specify the number of parallel simulated environments
    - `--train_timesteps NUM_STEPS` to specify the number of environment steps during the training
    - `--sim_device SIM_DEVICE` to specify device for simulation
    - `--rl_device RL_DEVICE` to specify device for runing OpenRL
    - `--headless` to render headlessly
    - `--seed RANDOM_SEED` to specify random seed
    - `--config /PATH/TO/CONFIG` to speicy cinfiguration for OpenRL
    - `--use_wandb` to use WanDB
    - `--use_tensorboard` to use TensorBoard

3. Evaluate trained policy

    `python ./openrl_ws/test.py --algo ALGO_NAME --task TASK_NAME --checkpoint /PATH/TO/CHECKPOINT`
    - `--record_video` to record video (frames)
    - `--algo ALGO_NAME` should be specified as well as `--checkpoint`

4. Create new task

## Existing Tasks ##

TBD

## Trouble Shooting ##

1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, it is also possible that you need to do `export LD_LIBRARY_PATH=/PATH/TO/LIBPYTHON/DIRECTORY` / `export LD_LIBRARY_PATH=/PATH/TO/CONDA/envs/YOUR_ENV_NAME/lib`. You can also try: `sudo apt install libpython3.8`.
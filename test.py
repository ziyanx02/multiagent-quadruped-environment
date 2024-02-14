
import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.utils import get_args
import torch

from legged_gym.envs.utils import make_mqe_env


def train(args):

    # task_name = "go1plane"
    task_name = "go1gate"
    # task_name = "go1football"
    # task_name = "go1sheep"
    # task_name = "go1seesaw"
    # task_name = "go1pushbox"

    env, env_cfg = make_mqe_env(task_name, args)
    # env, env_cfg = make_env(Go1, Go1PlaneCfg(), args)
    # env, env_cfg = make_env(Go1, Go1Cfg(), args)
    env.reset()
    obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    num_envs = env.num_envs
    num_actions = env.num_actions
    import time
    while True:
        obs, _, _, _ = env.step(torch.tensor([[[2.0, 0, 0],[1.0, 0, 0]],], dtype=torch.float32, device="cuda").repeat(env.num_envs, 1, 1))
    
if __name__ == '__main__':
    args = get_args()
    train(args)

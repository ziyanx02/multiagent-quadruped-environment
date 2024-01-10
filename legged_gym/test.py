
import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, make_env
import torch

from legged_gym.debugger import break_into_debugger

def train(args):

    from legged_gym.envs.go1.go1 import Go1
    from legged_gym.envs.go1.go1_dualrun_test_config import Go1DualrunTestCfg

    env, env_cfg = make_env(Go1, Go1DualrunTestCfg(), args)
    # env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env.reset()
    obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    num_envs = env.num_envs
    num_actions = env.num_actions
    import time
    while True:
        env.step(torch.randn([num_envs, num_actions], dtype=torch.float32, device="cuda"))


if __name__ == '__main__':
    args = get_args()
    train(args)
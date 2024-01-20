
import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, make_env
import torch

from legged_gym.debugger import break_into_debugger

def train(args):

    from legged_gym.envs.go1.go1 import Go1
    from legged_gym.envs.configs.go1_plane_config import Go1PlaneCfg
    from legged_gym.envs.configs.go1_gate_config import Go1GateCfg
    from legged_gym.envs.wrappers.go1_gate_wrapper import Go1GateWrapper

    env, env_cfg = make_env(Go1, Go1GateCfg(), args)
    env = Go1GateWrapper(env)
    # env, env_cfg = make_env(Go1, Go1PlaneCfg(), args)
    # env, env_cfg = make_env(Go1, Go1Cfg(), args)
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

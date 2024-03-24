
import numpy as np
import os

import isaacgym
from mqe.utils import get_args
import torch

from mqe.envs.utils import make_mqe_env

if __name__ == '__main__':
    args = get_args()

    # task_name = "go1plane"
    # task_name = "go1gate"
    # task_name = "go1football-defender"
    # task_name = "go1sheep-easy"
    task_name = "go1sheep-hard"
    # task_name = "go1seesaw"
    # task_name = "go1pushbox"
    args.headless = False

    env, env_cfg = make_mqe_env(task_name, args)
    env.reset()
    import time
    while True:
        obs, _, _, _ = env.step((1 + torch.randn(1, 2, 3, device="cuda")) * torch.tensor([[[1, 0, 0],[1, 0, 0]],], dtype=torch.float32, device="cuda").repeat(env.num_envs, 1, 1))

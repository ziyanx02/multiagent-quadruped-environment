from typing import Any, Dict, Optional, Union
import isaacgym

import numpy as np
import torch
import gym
from gym import spaces

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils import make_env

def make_mqe_env(name, args=None, env_cfg=None):
    

    from legged_gym.envs.go1.go1 import Go1
    from legged_gym.envs.configs.go1_plane_config import Go1PlaneCfg
    from legged_gym.envs.configs.go1_gate_config import Go1GateCfg
    from legged_gym.envs.wrappers.go1_gate_wrapper import Go1GateWrapper

    env, env_cfg = make_env(Go1, Go1GateCfg(), args)
    env = Go1GateWrapper(env)

    return mqe_openrl_wrapper(env)

class mqe_openrl_wrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.agent_num = self.env.num_agents
        self.parallel_env_num = self.env.num_envs
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        """Reset all environments."""
        obs = self.env.reset()
        return obs.cpu().numpy()

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""
        actions = torch.from_numpy(actions).cuda()

        obs_buff, self._rew, self._resets, self._extras = self.env.step(actions)

        obs = obs_buff.cpu().numpy()
        rewards = self._rew.cpu().numpy()
        dones = self._resets.cpu().numpy().astype(bool)

        infos = []
        for i in range(dones.shape[0]):
            infos.append({})

        return obs, rewards, dones, infos

    def close(self, **kwargs):
        return self.env.close()

    @property
    def use_monitor(self):
        return False

    def batch_rewards(self, buffer):
        return {}
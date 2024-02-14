import gym
from gym import spaces
import numpy
import torch
from copy import copy
from legged_gym.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1SeesawWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(14,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        # self.height_reward_scale = 0

        self.reward_buffer = {
            "height reward": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        return
    
    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)

        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)

        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)

        # height reward
        if self.height_reward_scale != 0:
            height_reward = self.height_reward_scale * base_pos[:, 2].reshape(self.num_envs, -1).sum(dim=1)
            reward[:, 0] += height_reward
            self.reward_buffer["height reward"] += torch.sum(height_reward).cpu()

        reward = reward.repeat(1, self.num_agents)

        return obs, reward, termination, info
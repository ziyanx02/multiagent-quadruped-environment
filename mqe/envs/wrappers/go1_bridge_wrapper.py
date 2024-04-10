import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1BridgeWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(12,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device=self.env.device).repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        # self.success_reward_scale = 0

        self.reward_buffer = {
            "target reward": 0,
            "success reward": 0,
            "punishment": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        base_pos = obs.base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        self.target_pos = torch.flip(base_pos, [1])

    def reset(self):
        obs_buf = self.env.reset()
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        self._init_extras(obs_buf)
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)
        obs[:, 1, 0] = abs(self.target_pos[:, 0, 0] + self.target_pos[:, 1, 0]) - obs[:, 1, 0]
        obs[:, 1, 4] = -obs[:, 1, 4]
        obs[:, 1, 6] = abs(self.target_pos[:, 0, 0] + self.target_pos[:, 1, 0]) - obs[:, 1, 6]
        obs[:, 1, 10] = -obs[:, 1, 10]
        return obs

    def step(self, action):
        action[:, 1, 1:] = -action[:, 1, 1:]
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        self.reward_buffer["step count"] += 1
        
        reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device, dtype=torch.float)

        base_pos = obs_buf.base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])

        if self.success_reward_scale != 0:
            success_reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
            success_reward[base_pos[:, 1, 2] < 0.5] = self.success_reward_scale
            reward[:, 0] += success_reward[:, 0]
            self.reward_buffer["success reward"] += torch.sum(success_reward[:, 0]).cpu()
        
        if self.punishment_scale != 0:
            punishment = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
            punishment[base_pos[:, 0, 2] < 0.5] = self.punishment_scale
            reward[:, 0] -= punishment[:, 0]
            self.reward_buffer["punishment"] += torch.sum(punishment[:, 0]).cpu()
        
        if self.target_reward_scale != 0:
            target_reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
            target_reward[base_pos[:, 0, 0] > self.target_pos[:, 0, 0]] = self.target_reward_scale
            reward[:, 0] += target_reward[:, 0]
            self.reward_buffer["target reward"] += torch.sum(target_reward[:, 0]).cpu()
            
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)
        obs[:, 1, 0] = abs(self.target_pos[:, 0, 0] + self.target_pos[:, 1, 0]) - obs[:, 1, 0]
        obs[:, 1, 4] = -obs[:, 1, 4]
        obs[:, 1, 6] = abs(self.target_pos[:, 0, 0] + self.target_pos[:, 1, 0]) - obs[:, 1, 6]
        obs[:, 1, 10] = -obs[:, 1, 10]

        return obs, reward, termination, info
import gym
from gym import spaces
import numpy
import torch
from copy import copy
from legged_gym.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1WrestlingWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(14,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device=self.env.device).repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        # self.success_reward_scale = 0

        self.reward_buffer = {
            "target reward": 0,
            "success reward": 0,
            # "approach frame punishment": 0,
            "agent distance punishment": 0,
            # "command lin_vel.y punishment": 0,
            # "command value punishment": 0,
            # "lin_vel.x reward": 0,
            "step count": 0
        }
        self.reward_buffer_1 = {
            "target reward": 0,
            "success reward": 0,
            # "approach frame punishment": 0,
            "agent distance punishment": 0,
            # "command lin_vel.y punishment": 0,
            # "command value punishment": 0,
            # "lin_vel.x reward": 0,
            "step count": 0
        }
        

    def _init_extras(self, obs):
        agent_ids = self.env_agent_indices.reshape(-1)
        target_pos = self.base_init_state[agent_ids][:, 2].reshape(self.env.num_envs*self.env.num_agents, -1)
        self.target_pos = target_pos - 0.5

    def reset(self):
        obs_buf = self.env.reset()
        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)
        self._init_extras(obs_buf)

        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        self.reward_buffer["step count"] += 1

        reward = torch.zeros([self.env.num_envs, self.env.num_agents, 1], device=self.env.device, dtype=torch.float)

        if self.success_reward_scale != 0:
            base_pos = obs_buf.base_pos
            success_reward = torch.zeros([self.env.num_envs*self.env.num_agents], device=self.env.device)
            success_reward[base_pos[:, 2] <= self.target_pos[:, 0]] = self.success_reward_scale
            reward += success_reward.reshape([self.env.num_envs, self.env.num_agents, 1])
            self.reward_buffer["success reward"] += torch.sum(success_reward).cpu()
        
        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)

        return obs, reward, termination, info
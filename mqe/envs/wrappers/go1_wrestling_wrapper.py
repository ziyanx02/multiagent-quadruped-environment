import gym
from gym import spaces
import numpy as np
from isaacgym.torch_utils import get_euler_xyz
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1WrestlingWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(12,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device=self.env.device).repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        # self.success_reward_scale = 0

        self.reward_buffer = {
            "punishment": 0,
            "success reward": 0,
            "step count": 0
        }
        

    def _init_extras(self, obs):
        agent_ids = self.env_agent_indices.reshape(-1)
        target_pos = self.base_init_state[agent_ids][:, 2].reshape(self.env.num_envs*self.env.num_agents, -1)
        self.target_pos = target_pos - 0.5

    def reset(self):
        obs_buf = self.env.reset()
        self._init_extras(obs_buf)
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)
        obs[:, 1, 1] = -obs[:, 1, 1]
        obs[:, 1, 4] = -obs[:, 1, 4]
        obs[:, 1, 7] = -obs[:, 1, 7]
        obs[:, 1, 10] = -obs[:, 1, 10]
        return obs

    def step(self, action):
        action[:, 1, 1:] = -action[:, 1, 1:]
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        self.reward_buffer["step count"] += 1

        reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device, dtype=torch.float)

        base_quat = obs_buf.base_quat
        r, p, y = get_euler_xyz(base_quat)
        r[r > np.pi] -= np.pi * 2 # to range (-pi, pi)
        p[p > np.pi] -= np.pi * 2 # to range (-pi, pi)
        r = r.reshape([self.env.num_envs, self.env.num_agents])
        p = p.reshape([self.env.num_envs, self.env.num_agents])

        if self.success_reward_scale != 0:
            success_reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
            p_term_buff = torch.zeros(self.env.num_envs, device=self.env.device)
            r_term_buff = torch.zeros(self.env.num_envs, device=self.env.device)
            p_term_buff[abs(p[:, 1]) > np.pi * 0.9] = 1
            r_term_buff[abs(r[:, 1]) >= np.pi * 0.4] = 1
            success_reward[p_term_buff + r_term_buff > 0, 0] = self.success_reward_scale
            reward[:, 0] += success_reward[:, 0]
            self.reward_buffer["success reward"] += torch.sum(success_reward[:, 0]).cpu()

        if self.punishment_scale != 0:
            punishment_reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
            p_term_buff = torch.zeros(self.env.num_envs, device=self.env.device)
            r_term_buff = torch.zeros(self.env.num_envs, device=self.env.device)
            p_term_buff[abs(p[:, 0]) > np.pi * 0.9] = 1
            r_term_buff[abs(r[:, 0]) >= np.pi * 0.4] = 1
            punishment_reward[p_term_buff + r_term_buff > 0, 0] = self.punishment_scale
            reward[:, 0] -= punishment_reward[:, 0]
            self.reward_buffer["punishment"] += torch.sum(punishment_reward[:, 0]).cpu()
        
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)
        obs[:, 1, 1] = -obs[:, 1, 1]
        obs[:, 1, 4] = -obs[:, 1, 4]
        obs[:, 1, 7] = -obs[:, 1, 7]
        obs[:, 1, 10] = -obs[:, 1, 10]

        return obs, reward.reshape([self.env.num_envs, self.env.num_agents, 1]), termination, info
import gym
from gym import spaces
import numpy
import torch
from copy import copy
from legged_gym.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1TugWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(14,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device=self.env.device).repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        # self.success_reward_scale = 0

        self.reward_buffer = {
            "success reward": 0,
            "step count": 0
        }
        
        self.reward_buffer_1 = {
            "success reward": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        self.target_pos = torch.zeros([self.env.num_envs*self.env.num_npcs, 2], dtype=torch.float, device=self.device, requires_grad=False)
        npc_ids = self.env_npc_indices.reshape(-1)
        target_pos = torch.clone(self.env.base_pos_npc)
        self.target_pos[:, 0] = target_pos[:, 1] - 0.5
        self.target_pos[:, 1] = target_pos[:, 1] + 0.5

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
            base_pos_npc = self.env.base_pos_npc
            success_reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
            success_reward[base_pos_npc[:, 1] >= self.target_pos[:,1], :self.env.num_agents//2] = self.success_reward_scale
            success_reward[base_pos_npc[:, 1] <= self.target_pos[:,0], self.env.num_agents//2:] = self.success_reward_scale
            reward[:, :, 0] += success_reward
            self.reward_buffer["success reward"] += torch.sum(success_reward.reshape([self.env.num_envs*self.env.num_agents])).cpu()

        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)

        return obs, reward, termination, info
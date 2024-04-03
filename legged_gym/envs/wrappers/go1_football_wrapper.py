import gym
from gym import spaces
import numpy
import torch
from copy import copy
from legged_gym.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1FootballWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(18 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device=self.env.device).repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        self.catch_ball_reward_scale = 0

        self.reward_buffer = {
            "catch ball reward": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        self.axis = self.BarrierTrack_kwargs["football"]["block_length"]
        return

    def reset(self):
        obs_buf = self.env.reset()

        self._init_extras(obs_buf)

        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), ball_pos, ball_vel], dim=2)
        obs[:, 1, 2] = self.axis - obs[:, 1, 2]
        obs[:, 1, 6] = -obs[:, 1, 6]
        obs[:, 1, 8] =  self.axis - obs[:, 1, 8]
        obs[:, 1, 11] = -obs[:, 1, 11]
        obs[:, 1, 14] = self.axis - obs[:, 1, 14]
        obs[:, 1, 17] = -obs[:, 1, 17]
        return obs

    def step(self, action):
        action[:, 1, 1:] = -action[:, 1, 1:]
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        reward = torch.zeros([self.env.num_envs*self.env.num_agents], device=self.env.device)
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        base_pos = obs_buf.base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])

        if self.catch_ball_reward_scale != 0:
            catch_ball_reward = torch.zeros([self.env.num_envs*self.num_agents], device=self.env.device)
            dis = torch.zeros([self.num_envs, self.num_agents, 2], dtype=torch.float, device=self.env.device)
            dis[:, :, 0] =  base_pos[:, :, 0] - ball_pos[:, 0]
            dis[:, :, 1] =  base_pos[:, :, 1] - ball_pos[:, 1]
            dis = dis.norm(p=2, dim=-1, keepdim=False).reshape(self.env.num_envs*self.num_agents)
            catch_ball_reward[dis <= 0.5] = self.catch_ball_reward_scale
            reward += catch_ball_reward
            self.reward_buffer["catch ball reward"] += torch.sum(catch_ball_reward.reshape(self.env.num_envs, -1)[:, 0])

        ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), ball_pos, ball_vel], dim=2)
        self.reward_buffer["step count"] += 1
        
        obs[:, 1, 2] = self.axis - obs[:, 1, 2]
        obs[:, 1, 6] = -obs[:, 1, 6]
        obs[:, 1, 8] =  self.axis - obs[:, 1, 8]
        obs[:, 1, 11] = -obs[:, 1, 11]
        obs[:, 1, 14] = self.axis - obs[:, 1, 14]
        obs[:, 1, 17] = -obs[:, 1, 17]

        reward = reward.reshape([self.num_envs, self.num_agents, 1])
        reward[:, 0, 0] -= reward[:, 1, 0]

        return obs, reward, termination, info
    
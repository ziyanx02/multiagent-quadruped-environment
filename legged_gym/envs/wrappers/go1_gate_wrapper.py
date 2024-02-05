import gym
from gym import spaces
import numpy
import torch
from copy import copy

class Go1GateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = self.env.num_envs
        self.num_agents = self.env.num_agents
        self.BarrierTrack_kwargs = env.cfg.terrain.BarrierTrack_kwargs
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(16,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[1, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        self.success_reward_scale = 10
        self.lin_vel_x_reward_scale = 0
        self.approach_frame_punishment_scale = -0.0
        self.agent_distance_punishment = -0.0
        self.lin_vel_y_punishment = -0.0

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self.gate_pos = obs_buf.env_info["gate_deviation"]
            self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
            self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
            self.frame_left = self.gate_pos.reshape(-1, 2)
            self.frame_right = self.gate_pos.reshape(-1, 2)
            self.frame_left[:, 1] += self.BarrierTrack_kwargs["gate"]["width"] / 2
            self.frame_right[:, 1] -= self.BarrierTrack_kwargs["gate"]["width"] / 2
            self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1]), self.gate_pos], dim=2)
        # obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)

        return obs

    def step(self, action):
        action[:, :, 1] = 0
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self.gate_pos = obs_buf.env_info["gate_deviation"]
            self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
            self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
            self.frame_left = self.gate_pos.reshape(-1, 2)
            self.frame_right = self.gate_pos.reshape(-1, 2)
            self.frame_left[:, 1] += self.BarrierTrack_kwargs["gate"]["width"] / 2
            self.frame_right[:, 1] -= self.BarrierTrack_kwargs["gate"]["width"] / 2
            self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1]), self.gate_pos], dim=2)
        # obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)

        # success reward
        success_reward = torch.zeros([self.env.num_envs * self.env.num_agents], device="cuda")
        success_reward[base_pos[:, 0] > self.gate_distance + 0.25] = self.success_reward_scale
        reward = success_reward.reshape([self.env.num_envs, self.env.num_agents])

        # approach frame punishment
        dis_to_left_frame = ((base_pos[:, :2] - self.frame_left) ** 2).sum(dim=1).reshape(self.num_envs, -1)
        dis_to_right_frame = ((base_pos[:, :2] - self.frame_right) ** 2).sum(dim=1).reshape(self.num_envs, -1)

        reward[dis_to_left_frame < 0.04] += self.approach_frame_punishment_scale / dis_to_left_frame[dis_to_left_frame < 0.04]
        reward[dis_to_right_frame < 0.04] += self.approach_frame_punishment_scale / dis_to_right_frame[dis_to_right_frame < 0.04]

        # agent distance punishment

        agent_dis = (base_pos[:, :2] - torch.flip(base_pos[:, :2].reshape(self.num_envs, self.num_agents, 2), dims=[1,]).reshape(-1, 2)) ** 2
        agent_dis = agent_dis.sum(dim=1).reshape(self.num_envs, -1)
        reward[agent_dis < 0.25] += self.agent_distance_punishment  / agent_dis[agent_dis < 0.25]

        # command lin_vel.y punishment

        reward += self.lin_vel_y_punishment * action[:, :, 1] ** 2

        # lin_vel.x reward

        reward += self.lin_vel_x_reward_scale * obs_buf.lin_vel[:, 0].reshape(self.num_envs, self.num_agents)

        return obs, reward, termination, info
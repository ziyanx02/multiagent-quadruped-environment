import gym
from gym import spaces
import numpy
import torch
from copy import copy
from legged_gym.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1FootballAgainstWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.num_agents = self.env.num_agents

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(18 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        self.obs_ids = torch.eye(self.num_agents, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1).reshape(self.num_envs, self.num_agents, -1)

        # for hard setting of reward scales (not recommended)
        
        # self.target_reward_scale = 1
        # self.success_reward_scale = 0
        # self.lin_vel_x_reward_scale = 0
        # self.approach_frame_punishment_scale = 0
        # self.agent_distance_punishment_scale = 0
        # self.lin_vel_y_punishment_scale = 0
        # self.command_value_punishment_scale = 0

        self.reward_buffer = {
            "goal reward": 0,
            "ball gate distance reward": 0,
            "goal punishment": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        self.gate_pos = obs.env_info["door_pos"].reshape(self.num_envs, 1, -1).repeat(1, 2, 1)
        self.gate_pos[:, :, 1] = 0
        self.gate_pos[:, 0, 0] = self.BarrierTrack_kwargs["football"]["block_length"] - self.gate_pos[:, 0, 0]
        self.axis = self.BarrierTrack_kwargs["football"]["block_length"]
        return

    def reset(self):
        obs_buf = self.env.reset()
        self._init_extras(obs_buf)
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, 4, 1)
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 4, 1)
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.num_envs, self.num_agents, -1])
        base_flip_info = torch.cat([torch.flip(base_info[:, :self.num_agents // 2, :], [1]), torch.flip(base_info[:, self.num_agents // 2:, :], [1])], dim=1)
        obs = torch.cat([self.obs_ids, base_info, base_flip_info, ball_pos, ball_vel], dim=2)
        obs[:, 2:, 4] = self.axis - obs[:, 2:, 4]
        obs[:, 2:, 8] = -obs[:, 2:, 8]
        obs[:, 2:, 10] =  self.axis - obs[:, 2:, 10]
        obs[:, 2:, 14] = -obs[:, 2:, 14]
        obs[:, 2:, 16] = self.axis - obs[:, 2:, 16]
        obs[:, 2:, 19] = -obs[:, 2:, 19]
        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        action[:, 2:, 1:] = -action[:, 2:, 1:]
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        reward = torch.zeros([self.env.num_envs, 4], device=self.env.device)
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins

        if self.goal_reward_scale != 0:
            goal_reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)
            goal_reward[ball_pos[:, 0] > self.gate_pos[:, 0, 0], 0] = self.goal_reward_scale
            reward[:, :2] += goal_reward.repeat(1, 2)
            self.reward_buffer["goal reward"] += torch.sum(goal_reward[:, 0]).cpu()
        
        if self.ball_gate_distance_reward_scale != 0:
            ball_gate_distance = torch.norm(ball_pos[:, :2] - self.gate_pos[:, 0, :], dim=1, keepdim=True)
            ball_gate_distance_reward = self.ball_gate_distance_reward_scale * torch.exp(-ball_gate_distance / 3)
            reward[:, :2] += ball_gate_distance_reward.repeat(1, 2)
            self.reward_buffer["ball gate distance reward"] += torch.sum(ball_gate_distance_reward).cpu()
        
        if self.goal_punishment_scale != 0:
            goal_punishment = torch.zeros([self.env.num_envs, 1], device=self.env.device)
            goal_punishment[ball_pos[:, 0] < self.gate_pos[:, 1, 0], 0] = self.goal_reward_scale
            reward[:, :2] -= goal_punishment.repeat(1, 2)
            self.reward_buffer["goal punishment"] += torch.sum(goal_punishment[:, 0]).cpu()

        ball_pos = ball_pos.unsqueeze(1).repeat(1, 4, 1)
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 4, 1)
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.num_envs, self.num_agents, -1])
        base_flip_info = torch.cat([torch.flip(base_info[:, :self.num_agents // 2, :], [1]), torch.flip(base_info[:, self.num_agents // 2:, :], [1])], dim=1)
        obs = torch.cat([self.obs_ids, base_info, base_flip_info, ball_pos, ball_vel], dim=2)
        self.reward_buffer["step count"] += 1
        obs[:, 2:, 4] = self.axis - obs[:, 2:, 4]
        obs[:, 2:, 8] = -obs[:, 2:, 8]
        obs[:, 2:, 10] =  self.axis - obs[:, 2:, 10]
        obs[:, 2:, 14] = -obs[:, 2:, 14]
        obs[:, 2:, 16] = self.axis - obs[:, 2:, 16]
        obs[:, 2:, 19] = -obs[:, 2:, 19]
        
        return obs, reward.reshape(self.num_envs, 4, 1), termination, info
    
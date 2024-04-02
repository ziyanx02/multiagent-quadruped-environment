import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1FootballDefenderWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.num_agents = 2

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(18 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, 2, 1)

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
            "step count": 0
        }

    def _init_extras(self, obs):
        return

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)

        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), ball_pos, ball_vel], dim=2)

        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)

        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), ball_pos, ball_vel], dim=2)

        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)

        if self.goal_reward_scale != 0:

            goal_reward = reward.clone()
            goal_reward[ball_pos[:, 0, 0] > self.gate_pos[:, 0], 0] = self.goal_reward_scale
            reward += goal_reward
            self.reward_buffer["goal reward"] += torch.sum(goal_reward).cpu()

        if self.ball_gate_distance_reward_scale != 0:
            
            ball_gate_distance = torch.norm(ball_pos[:, 0, :2] - self.gate_pos[:, :2], dim=1, keepdim=True)
            ball_gate_distance_reward = self.ball_gate_distance_reward_scale * torch.exp(-ball_gate_distance / 3)
            reward += ball_gate_distance_reward
            self.reward_buffer["ball gate distance reward"] += torch.sum(ball_gate_distance_reward).cpu()

        return obs, reward.repeat(1, 2), termination, info
    
class Go1FootballGameWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

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
            "step count": 0
        }

    def _init_extras(self, obs):
        return

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)

        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]

        return None

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)

        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]

        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)

        return None, reward.repeat(1, 4), termination, info
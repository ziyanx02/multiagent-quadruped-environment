import gym
from gym import spaces
import numpy
import torch
from copy import copy
from legged_gym.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1SheepWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(14 + 2 * self.cfg.env.num_npcs + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        # self.success_reward_scale = 0

        self.reward_buffer = {
            "success reward": 0,
            "contact punishment": 0,
            "sheep movement reward": 0,
            "mixed sheep reward": 0,
            "sheep pos var punishment": 0,
            "step count": 0
        }

    def _init_extras(self, obs):

        gate_pos = obs.env_info["gate_deviation"]
        gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["plane"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        self.gate_pos = gate_pos.unsqueeze(1)
        self.gate_distance = gate_pos[:, 0].unsqueeze(1).repeat(1, self.num_npcs)

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        sheep_pos = self.root_states_npc[:, :3].reshape(self.num_envs, -1, 3) - self.npc_env_origins
        sheep_pos_flatten = sheep_pos[..., :2].reshape(self.num_envs, 1, -1).repeat(1, self.num_agents, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), self.gate_pos.repeat(1, self.num_agents, 1), sheep_pos_flatten], dim=2)

        self.last_sheep_pos_avg = None

        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        sheep_pos = self.root_states_npc[:, :3].reshape(self.num_envs, -1, 3) - self.npc_env_origins
        sheep_pos_flatten = sheep_pos[..., :2].reshape(self.num_envs, 1, -1).repeat(1, self.num_agents, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), self.gate_pos.repeat(1, self.num_agents, 1), sheep_pos_flatten], dim=2)

        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)

        # success reward
        if self.success_reward_scale != 0:
            success = (sheep_pos[:, :, 0] - self.gate_distance) > 0
            success_reward = success.sum(dim=1)
            reward[:, 0] = success_reward
            self.reward_buffer["success reward"] += torch.sum(success_reward).cpu()
            
        # contact punishment
        if self.contact_punishment_scale != 0:
            collide_reward = self.contact_punishment_scale * self.env.collide_buf
            reward += collide_reward.unsqueeze(1)
            self.reward_buffer["contact punishment"] += torch.sum(collide_reward).cpu()
        
        # sheep movement reward
        if self.sheep_movement_reward_scale != 0:

            if self.last_sheep_pos_avg != None:
                x_movement = (self.sheep_pos_avg - self.last_sheep_pos_avg)[:, 0]
                x_movement[self.delayed_reset_buf] = 0
                sheep_movement_reward = self.sheep_movement_reward_scale * x_movement
                reward[:, 0] += sheep_movement_reward
                self.reward_buffer["sheep movement reward"] += torch.sum(sheep_movement_reward).cpu()

            self.last_sheep_pos_avg = copy(self.sheep_pos_avg)

        # mixed sheep reward
        if self.mixed_sheep_reward_scale != 0:

            mixed_sheep_reward = torch.zeros(*sheep_pos.shape[:-1], device=self.env.device)
            distance_to_gate = torch.norm(sheep_pos[..., :-1] - self.gate_pos.repeat(1, self.num_npcs, 1), dim=-1)
            mixed_sheep_reward = torch.exp(- distance_to_gate / 2) * self.mixed_sheep_reward_scale
            mixed_sheep_reward[sheep_pos[..., 0] >= self.gate_distance] = self.mixed_sheep_reward_scale
            reward[:, 0] += mixed_sheep_reward.sum(dim=-1)
            self.reward_buffer["mixed sheep reward"] += torch.sum(mixed_sheep_reward).cpu()

        # sheep pos var punishment
        if self.sheep_pos_var_exp_punishment_scale != 0 or self.sheep_pos_var_lin_punishment_scale != 0:

            sheep_pos_var_punishment = self.sheep_pos_var_lin_punishment_scale * (self.sheep_pos_var - 1) + self.sheep_pos_var_exp_punishment_scale * torch.exp(self.sheep_pos_var / 2 - 1)
            reward[:, 0] += sheep_pos_var_punishment
            self.reward_buffer["sheep pos var punishment"] += torch.sum(sheep_pos_var_punishment).cpu()

        reward = reward.repeat(1, self.num_agents)

        self.delayed_reset_buf = copy(self.env.reset_ids)

        return obs, reward, termination, info
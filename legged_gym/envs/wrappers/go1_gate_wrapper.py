import gym
from gym import spaces
import numpy
import torch

class Go1GateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = self.env.num_envs
        self.num_agents = self.env.num_agents
        BarrierTrack_kwargs = env.cfg.terrain.BarrierTrack_kwargs
        self.gate_distance = BarrierTrack_kwargs["init"]["block_length"] \
                            + BarrierTrack_kwargs["gate"]["block_length"] / 2 \
                            + BarrierTrack_kwargs["gate"]["offset"][0]
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(14,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

    def step(self, action):
        obs, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        base_pos = obs.base_pos
        base_quat = obs.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)

        reward = torch.zeros([self.env.num_envs * self.env.num_agents], device="cuda")
        reward[base_pos[:, 0] > self.gate_distance + 0.5] = 1
        reward = reward.reshape([self.env.num_envs, self.env.num_agents])

        return obs, reward, termination, info
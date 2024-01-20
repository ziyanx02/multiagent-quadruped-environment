import gym
import torch

class Go1GateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        BarrierTrack_kwargs = env.cfg.terrain.BarrierTrack_kwargs
        self.gate_distance = BarrierTrack_kwargs["init"]["block_length"] \
                            + BarrierTrack_kwargs["gate"]["block_length"] / 2 \
                            + BarrierTrack_kwargs["gate"]["offset"][0]

    def step(self, action):
        obs, _, termination, info = self.env.step(action)

        base_pos = obs.base_pos
        base_quat = obs.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, torch.flip(base_info, [1])], dim=2)

        reward = torch.zeros([self.env.num_envs * self.env.num_agents], device="cuda")
        reward[base_pos[:, 0] > self.gate_distance + 0.5] = 1
        reward = reward.reshape([self.env.num_envs, self.env.num_agents])

        return obs, reward, termination, info
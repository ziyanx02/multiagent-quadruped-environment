from isaacgym import gymtorch
import gym
from gym import spaces
import numpy
import torch
from copy import copy
from legged_gym.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1TugWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device=self.env.device).repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        # self.success_reward_scale = 0

        self.reward_buffer = {
            "success reward": 0,
            "pos reward": 0,
            # "vel reward": 0,
            "step count": 0,
            "npc pos": 0,
            "punishment": 0,
            "total reward": 0,
            "pos": 0 ,
            "pos_y": 0,
            "opponet pos": 0,
            "opponet pos_y": 0,
        }
        self.reset_dic = torch.zeros([self.env.num_envs], dtype=torch.float, device=self.device, requires_grad=False)

    def _init_extras(self, obs):
        self.last_dis = torch.clone(obs.base_pos.reshape([self.env.num_envs, self.env.num_agents, -1]))
        self.last_npc_pos = torch.clone(self.env.dof_state_npc[:, :, :1])
        self.env_step = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def reset(self):
        obs_buf = self.env.reset()
        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        self._init_extras(obs_buf)
        obs = torch.cat([base_info, self.env.dof_state_npc.repeat(1, self.env.num_agents, 1), self.last_npc_pos.repeat(1, self.env.num_agents, 1)], dim=2)
        self.env_step[:] += 1
        

        return obs

    def step(self, action):
        
        action[:, 1, 1:] = -action[:, 1, 1:]
        if any(self.reset_dic[:] > 0):
            self.reset_dic[self.reset_dic[:] > 0] -= 1
            self.env.dof_state_npc[self.reset_dic[:] > 0, 0, 0] = 0.
            self.env.dof_state_npc[self.reset_dic[:] > 0, 0, 1] = 0.
            npc_indices = self.npc_indices.reshape(-1)
            self.env.gym.set_dof_state_tensor_indexed(self.env.sim,
                                                gymtorch.unwrap_tensor(self.env.all_dof_states),
                                                gymtorch.unwrap_tensor(npc_indices), len(npc_indices))
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        self.reset_dic[self.env.reset_ids] = 3
        self.reward_buffer["step count"] += 1
        base_pos = obs_buf.base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])

        reward = torch.zeros([self.env.num_envs, self.env.num_agents, 1], device=self.env.device, dtype=torch.float)
        
        dof_state_pos_npc = self.env.dof_state_npc[:, 0, 0]
        dof_state_quat_npc = self.env.dof_state_npc[:, 0, 1]
        pos_reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
        success_reward = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
        punishment = torch.zeros([self.env.num_envs, self.env.num_agents], device=self.env.device)
        
        for i in range(self.env.num_envs):
                
            if dof_state_pos_npc[i] > 0:
                if self.last_npc_pos[i, 0, 0] < dof_state_pos_npc[i]:
                    success_reward[i, 0] +=  self.success_reward_scale * abs(dof_state_pos_npc[i])
                else:
                    success_reward[i, 0] += self.success_reward_scale * abs(dof_state_pos_npc[i]) / 2
            if dof_state_pos_npc[i] < 0:
                if self.last_npc_pos[i, 0, 0] >= dof_state_pos_npc[i]:
                    punishment[i, 0] +=  self.punishment_reward_scale * abs(dof_state_pos_npc[i])
                else:
                    punishment[i, 0] += self.punishment_reward_scale * abs(dof_state_pos_npc[i]) / 2

            if not i in self.env.reset_ids:
                last_dis = torch.clone(self.last_dis[i, 0, :2])
                last_dis[0] -= 2.2
                last_dis[1] -= dof_state_pos_npc[i] 
                dis = torch.clone(base_pos[i, 0, :2])
                dis[0] -= 2.2
                dis[1] -= dof_state_pos_npc[i]
                if base_pos[i, 0, 0] > 2.0 and base_pos[i, 0, 0] < 2.4:
                    if abs(dis[1]) < abs(last_dis[1]) and abs(dis[1]) > 1.5:
                        pos_reward[i, 0] += pow(0.5, abs(base_pos[i, 0, 1] - dof_state_pos_npc[i]) * self.env_step[i]) * self.pos_reward_scale
                    else: punishment[i, 0] += (pow(2, abs(base_pos[i, 0, 1] - dof_state_pos_npc[i])) - 1)
                else:
                    if abs(dis[0]) < abs(last_dis[0]):
                        pos_reward[i, 0] += pow(0.5, abs(base_pos[i, 0, 0] - 2.2) * self.env_step[i]) * self.pos_reward_scale
                    else: punishment[i, 0] += (pow(2, abs(base_pos[i, 0, 0] - 2.2)) - 1)

        self.last_dis = torch.clone(base_pos)
        self.last_npc_pos = torch.clone(self.env.dof_state_npc[:, :, :1])
        reward[:, :, 0] += pos_reward
        reward[:, :, 0] += success_reward
        reward[:, :, 0] -= punishment

        self.reward_buffer["npc pos"] += torch.sum(dof_state_pos_npc[:]).cpu()
        self.reward_buffer["success reward"] += torch.sum(success_reward[:, 0]).cpu()
        self.reward_buffer["pos reward"] += torch.sum(pos_reward[:, 0]).cpu()
        self.reward_buffer["punishment"] += torch.sum(punishment[:, 0]).cpu()
        self.reward_buffer["total reward"] = self.reward_buffer["total reward"] + torch.sum(pos_reward[:, 0]).cpu() + torch.sum(success_reward[:, 0]).cpu() - torch.sum(punishment[:, 0]).cpu()
        self.reward_buffer["pos"] += torch.sum(base_pos[:, 0, 0]).cpu()
        self.reward_buffer["pos_y"] += torch.sum(base_pos[:, 0, 1]).cpu()
        self.reward_buffer["opponet pos"] += torch.sum(base_pos[:, 1, 0]).cpu()
        self.reward_buffer["opponet pos_y"] += torch.sum(base_pos[:, 1, 1]).cpu()


        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_info = torch.cat([base_pos, base_quat], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([base_info, self.env.dof_state_npc.repeat(1, self.env.num_agents, 1), self.last_npc_pos.repeat(1, self.env.num_agents, 1)], dim=2)
        obs[:, 1, 1] = -obs[:, 1, 1]
        obs[:, 1, 4] = -obs[:, 1, 4]
        obs[:, 1, -1] = -obs[:, 1, -1]
        self.env_step[:] += 1
        self.env_step[self.env.reset_ids] = 1
        
        return obs, reward, termination, info
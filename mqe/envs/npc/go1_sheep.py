
from mqe import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from mqe import LEGGED_GYM_ROOT_DIR

from mqe.envs.go1.go1 import Go1

def relative_pos_to_dv(relative_pos):
    dis = torch.norm(relative_pos ** 2, dim=2)
    dv = relative_pos / (dis ** 1.4).unsqueeze(-1).repeat(1, 1, 3)
    dv[dis > 9, ...] = 0
    return dv

class Go1Sheep(Go1):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):

        self.npc_collision = True
        self.fix_npc_base_link = False
        self.npc_gravity = True
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # dv = randomness + scale * fn(relative distance) 
        self.sheep_movement_scale = self.cfg.asset.sheep_movement_scale
        self.sheep_movement_randomness = self.cfg.asset.sheep_movement_randomness
        self.sheep_movement_range = self.cfg.asset.sheep_movement_range

    def _step_npc(self):

        dog_pos = self.root_states[:, :3].reshape(self.num_envs, -1, 3)
        sheep_pos = self.root_states_npc[:, :3].reshape(self.num_envs, -1, 3)
        sheep_avg_pos = torch.mean(sheep_pos, dim=1, keepdim=True).repeat(1, self.cfg.env.num_npcs, 1)
        self.sheep_pos_avg = sheep_avg_pos[:, 0, :2]
        self.sheep_pos_var = torch.var(sheep_pos, dim=1, unbiased=False)[..., :2].sum(dim=-1)

        dv = self.sheep_movement_randomness * torch.randn_like(sheep_pos, device=self.device) * 2

        if self.cfg.env.num_npcs != 1:
            relative_pos = sheep_avg_pos - sheep_pos
            dv += self.sheep_movement_randomness * relative_pos / torch.norm(relative_pos, p=2, dim=2, keepdim=True).repeat(1, 1, 3) / 1.5

        for i in range(self.num_agents):

            relative_pos = sheep_pos - dog_pos[:, i : i+1, :].repeat(1, self.num_npcs, 1)
            dv += self.sheep_movement_scale * relative_pos_to_dv(relative_pos)

        dv[:, :, 2] = 0

        npc_indices = self.npc_indices.reshape(-1)
        npc_indices_long = npc_indices.long()
        self.all_root_states[npc_indices_long, 7:10] += dv.reshape(-1, 3)
        self.all_root_states[npc_indices_long, 7:9] = torch.clip(self.all_root_states[npc_indices_long, 7:9], -2, 2)
        self.all_root_states[npc_indices_long, 2] = torch.clip(self.all_root_states[npc_indices_long, 2], 0, 0.3)
        self.all_root_states[npc_indices_long, 3:5] = 0
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(npc_indices), len(npc_indices))

    def _prepare_npc(self):
        
        #creat npc asset
        asset_path_npc = self.cfg.asset.file_npc.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root_npc = os.path.dirname(asset_path_npc)
        asset_file_npc = os.path.basename(asset_path_npc)
        asset_options_npc = gymapi.AssetOptions()
        asset_options_npc.fix_base_link = self.fix_npc_base_link
        asset_options_npc.disable_gravity = not self.npc_gravity
        self.asset_npc = self.gym.load_asset(self.sim, asset_root_npc, asset_file_npc, asset_options_npc)

        #init npc state
        num_rows = self.cfg.asset.num_rows
        num_cols = self.cfg.asset.num_cols
        dis_sheep = self.cfg.asset.dis_sheep

        sheep_origin = np.array([self.cfg.terrain.BarrierTrack_kwargs["init"]["block_length"] + self.cfg.terrain.BarrierTrack_kwargs["plane"]["block_length"] / 2
                                 - num_rows // 2 * dis_sheep[0],
                                 - (num_cols // 2) * dis_sheep[1], 0.3])
        pos = sheep_origin.copy()
        rot = np.array([0.0, 0.0, 0.0, 1.0])
        lin_vel = np.array([0., 0., 0.])
        ang_vel = np.array([0., 0., 0.])

        self.start_pose_npc = gymapi.Transform()
        self.start_pose_npc.p = gymapi.Vec3(*pos)

        init_state_list_npc = []
        for i in range(num_rows):
            for j in range(num_cols):
                
                init_state = np.concatenate((
                    pos,
                    rot + np.random.randn(4) * np.array([0, 0, np.pi, 1]),
                    lin_vel,
                    ang_vel
                ))
                init_state = to_torch(init_state, device=self.device, requires_grad=False)
                init_state_list_npc.append(init_state)

                pos[1] += dis_sheep[1]

            pos[0] += dis_sheep[0]
            pos[1] = sheep_origin[1]

        self.base_init_state_npc = torch.stack(init_state_list_npc, dim=0).repeat(self.num_envs, 1)
        
    def _create_npc(self, env_handle, env_id):

        npc_handles = []
        for i in range(self.num_npcs):
            npc_handle = self.gym.create_actor(env_handle, self.asset_npc, self.start_pose_npc, self.cfg.asset.name_npc, env_id, not self.npc_collision)
            npc_handles.append(npc_handle)
    
        self.npc_env_origins = self.env_origins.unsqueeze(1).repeat(1, self.cfg.env.num_npcs, 1)

        return npc_handles

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.envs.go1.go1 import Go1

def relative_pos_to_dv(relative_pos):
    dv = relative_pos / (torch.sum(relative_pos ** 2, dim=2) ** 0.8).unsqueeze(-1).repeat(1, 1, 3)
    return dv

class Go1Sheep(Go1):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):

        self.npc_collision = True
        self.fix_npc_base_link = False
        self.npc_gravity = True
        
        # dv = randomness + scale * fn(relative distance) 
        self.sheep_movement_scale = 0.2
        self.sheep_movement_randomness = 0.1
        self.sheep_movement_range = [2.0, 2.0, 0]

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _step_npc(self):

        dog_pos = self.root_states[:, :3].reshape(self.num_envs, -1, 3)
        sheep_pos = self.root_states_npc[:, :3].reshape(self.num_envs, -1, 3)

        dv = self.sheep_movement_randomness * torch.randn_like(sheep_pos, device=self.device)

        for i in range(self.num_agents):

            relative_pos = sheep_pos - dog_pos[:, i : i+1, :].repeat(1, self.num_npcs, 1)
            dv += self.sheep_movement_scale * relative_pos_to_dv(relative_pos)

        dv[:, :, 2] = 0

        npc_indices = self.npc_indices.reshape(-1)
        self.all_root_states[npc_indices, 7:10] += dv.reshape(-1, 3)
        self.all_root_states[npc_indices, 7:9] = torch.clip(self.all_root_states[npc_indices, 7:9], -2, 2)
        self.all_root_states[npc_indices, 2] = torch.clip(self.all_root_states[npc_indices, 2], 0, 0.5)
        self.all_root_states[npc_indices, 3:5] = 0
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
        num_rols = 5
        num_cols = 5
        dis_sheep = (0.8, 0.8)

        sheep_origin = np.array([4.0 - num_rols // 2 * dis_sheep[0], - (num_cols // 2) * dis_sheep[1], 0.3])
        pos = sheep_origin.copy()
        rot = np.array([0.0, 0.0, 0.0, 1.0])
        lin_vel = np.array([0., 0., 0.])
        ang_vel = np.array([0., 0., 0.])

        self.start_pose_npc = gymapi.Transform()
        self.start_pose_npc.p = gymapi.Vec3(*pos)

        init_state_list_npc = []
        for i in range(num_rols):
            for j in range(num_cols):
                
                init_state = np.concatenate((
                    pos + np.random.randn(3) * np.array([0.2, 0.2, 0]),
                    rot + np.random.randn(4) * np.array([0, 0, np.pi, 1]),
                    lin_vel,
                    ang_vel
                ))
                init_state = to_torch(init_state, device=self.device, requires_grad=False)
                init_state_list_npc.append(init_state)

                pos[1] += dis_sheep[1]

            pos[0] += dis_sheep[0]
            pos[1] = sheep_origin[1]

        # self.start_pose_npc = gymapi.Transform()
        # for idx, init_state_npc  in enumerate(self.init_state_npc):
        #     base_init_state_list_npc = [0.0, .0, 0.3] + init_state_npc.rot + init_state_npc.lin_vel + init_state_npc.ang_vel
        #     base_init_state_npc = to_torch(base_init_state_list_npc, device=self.device, requires_grad=False)
        #     init_state_list_npc.append(base_init_state_npc)
        #     if idx == 0:
        #         self.start_pose_npc.p = gymapi.Vec3(*base_init_state_npc[:3])
        self.base_init_state_npc = torch.stack(init_state_list_npc, dim=0).repeat(self.num_envs, 1)
        
    def _create_npc(self, env_handle, env_id):

        npc_handles = []
        for i in range(self.num_npcs):
            npc_handle = self.gym.create_actor(env_handle, self.asset_npc, self.start_pose_npc, self.cfg.asset.name_npc, env_id, not self.npc_collision)
            npc_handles.append(npc_handle)
        return npc_handles

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.envs.go1.go1 import Go1

class Go1Sheep(Go1):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):

        self.npc_collision = True
        self.fix_npc_base_link = False
        self.npc_gravity = True

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _step_npc(self):
        
        actor_ids_int32 = self.actor_indices.view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))

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
        dis_sheep = (1.0, 1.0)

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
            npc_handle = self.gym.create_actor(env_handle, self.asset_npc, self.start_pose_npc, self.cfg.asset.name_npc, env_id, self.npc_collision, 0)
            npc_handles.append(npc_handle)
        return npc_handles
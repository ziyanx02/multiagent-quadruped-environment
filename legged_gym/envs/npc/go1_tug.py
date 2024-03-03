
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.envs.go1.go1 import Go1

class Go1Tug(Go1):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):

        self.npc_collision = getattr(cfg.asset, "npc_collision", True)
        self.fix_npc_base_link = getattr(cfg.asset, "fix_npc_base_link", False)
        self.npc_gravity = getattr(cfg.asset, "npc_gravity", True)
    
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _step_npc(self):

        dog_pos = self.root_states[:, :3].reshape(self.num_envs, -1, 3)
        object_pos = self.root_states_npc[:, :3].reshape(self.num_envs, -1, 3)

        object_pos[:, :, 2] = 0

        npc_indices = self.npc_indices.reshape(-1)
        self.all_root_states[npc_indices, 2] = 0.5
        self.all_root_states[npc_indices, 3:5] = 0
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(npc_indices), len(npc_indices))

    def _prepare_npc(self):
        
        #creat npc asset
        self.init_state_npc = getattr(self.cfg.init_state, "init_states_npc")
        if hasattr(self.cfg.init_state, "default_npc_joint_angles"):
            self.default_dof_pos_npc = torch.tensor(self.cfg.asset.default_npc_joint_angles, dtype=torch.float, device=self.device, requires_grad=False).reshape(1, -1)
        else:
            self.default_dof_pos_npc = torch.zeros(self.num_actions_npc, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
        
        #creat npc asset
        asset_path_npc = self.cfg.asset.file_npc.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root_npc = os.path.dirname(asset_path_npc)
        asset_file_npc = os.path.basename(asset_path_npc)
        asset_options_npc = gymapi.AssetOptions()
        asset_options_npc.fix_base_link = self.fix_npc_base_link
        asset_options_npc.disable_gravity = not self.npc_gravity
        self.asset_npc = self.gym.load_asset(self.sim, asset_root_npc, asset_file_npc, asset_options_npc)

        #init npc state
        init_state_list_npc = []
        self.start_pose_npc = gymapi.Transform()
        for idx, init_state_npc  in enumerate(self.init_state_npc):
            base_init_state_list_npc = init_state_npc.pos + init_state_npc.rot + init_state_npc.lin_vel + init_state_npc.ang_vel
            base_init_state_npc = to_torch(base_init_state_list_npc, device=self.device, requires_grad=False)
            init_state_list_npc.append(base_init_state_npc)
            if idx == 0:
                self.start_pose_npc.p = gymapi.Vec3(*base_init_state_npc[:3])
        self.base_init_state_npc = torch.stack(init_state_list_npc, dim=0).repeat(self.num_envs, 1)
        
    def _create_npc(self, env_handle, env_id):

        npc_handles = []
        for i in range(self.num_npcs):
            npc_handle = self.gym.create_actor(env_handle, self.asset_npc, self.start_pose_npc, self.cfg.asset.name_npc, env_id, not self.npc_collision)
            npc_handles.append(npc_handle)
        return npc_handles
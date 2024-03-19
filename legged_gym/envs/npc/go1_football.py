
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.envs.go1.go1 import Go1
import random

class Go1Football(Go1):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):

        self.npc_collision = getattr(cfg.asset, "npc_collision", True)
        self.fix_npc_base_link = getattr(cfg.asset, "fix_npc_base_link", False)
        self.npc_gravity = getattr(cfg.asset, "npc_gravity", True)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.ball_flag = torch.zeros(self.num_envs, self.num_agents, dtype=torch.float, device=self.device, requires_grad=False)

    def _step_npc(self):
        return

    def _prepare_npc(self):
    
        self.init_state_npc = getattr(self.cfg.init_state, "init_states_npc")
        if hasattr(self.cfg.init_state, "default_npc_joint_angles"):
            self.default_dof_pos_npc = torch.tensor(self.cfg.init_state.default_npc_joint_angles, dtype=torch.float, device=self.device, requires_grad=False).reshape(1, -1)
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
        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        agent_ids = self.env_agent_indices[env_ids].reshape(-1)
        npc_ids = self.env_npc_indices[env_ids].reshape(-1)
        self.root_states[agent_ids] = self.base_init_state[agent_ids]
        self.root_states[agent_ids, :3] += self.agent_origins[env_ids].reshape(-1, 3)
        
        self.root_states_npc[npc_ids] = self.base_init_state_npc[npc_ids]
        self.root_states_npc[npc_ids, :3] += self.env_origins[env_ids].unsqueeze(1).repeat(1, self.num_npcs, 1).reshape(-1, 3)
        for i in env_ids:
            a = random.randint(0,1)
            self.ball_flag[i, a] = 1
            self.root_states_npc[i, 0] += 3.5 if a == 1 else 0
                

        if self.custom_origins:
            if getattr(self.cfg.domain_rand, "init_base_pos_range", None) is not None:
                self.root_states[agent_ids, 0:1] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["x"], (len(agent_ids), 1), device=self.device)
                self.root_states[agent_ids, 1:2] += torch_rand_float(*self.cfg.domain_rand.init_base_pos_range["y"], (len(agent_ids), 1), device=self.device)

            if getattr(self.cfg.domain_rand, "init_npc_base_pos_range", None) is not None:
                self.root_states_npc[npc_ids, 0:1] += torch_rand_float(*self.cfg.domain_rand.init_npc_base_pos_range["x"], (len(npc_ids), 1), device=self.device)
                self.root_states_npc[npc_ids, 1:2] += torch_rand_float(*self.cfg.domain_rand.init_npc_base_pos_range["y"], (len(npc_ids), 1), device=self.device)

            if getattr(self.cfg.domain_rand, "init_npc_base_rpy_range", None) is not None:
                self.root_states_npc[npc_ids, 3:7] = quat_from_euler_xyz(torch_rand_float(*self.cfg.domain_rand.init_npc_base_rpy_range["r"], (len(npc_ids), 1), device=self.device),
                                                                         torch_rand_float(*self.cfg.domain_rand.init_npc_base_rpy_range["p"], (len(npc_ids), 1), device=self.device),
                                                                         torch_rand_float(*self.cfg.domain_rand.init_npc_base_rpy_range["y"], (len(npc_ids), 1), device=self.device)).squeeze()

        # base velocities
        if getattr(self.cfg.domain_rand, "init_base_vel_range", None) is None:
            base_vel_range = (-0.5, 0.5)
        else:
            base_vel_range = self.cfg.domain_rand.init_base_vel_range
        self.root_states[agent_ids, 7:13] = torch_rand_float(
            *base_vel_range,
            (len(agent_ids), 6),
            device=self.device, 
        ) # [7:10]: lin vel, [10:13]: ang vel
        agent_indices_long = self.agent_indices[env_ids].reshape(-1).long()
        npc_indices_long = self.npc_indices[env_ids].reshape(-1).long()
        self.all_root_states[agent_indices_long] = self.root_states[agent_ids]
        self.all_root_states[npc_indices_long] = self.root_states_npc[npc_ids]
        actor_ids_int32 = self.actor_indices[env_ids].view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.all_root_states),
                                                     gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
        
    def _create_npc(self, env_handle, env_id):

        npc_handles = []
        for i in range(self.num_npcs):
            pos = self.env_origins[i].clone()
            self.start_pose_npc.p = gymapi.Vec3(*pos)
            npc_handle = self.gym.create_actor(env_handle, self.asset_npc, self.start_pose_npc, self.cfg.asset.name_npc, env_id, not self.npc_collision, 0)
            npc_handles.append(npc_handle)
        return npc_handles
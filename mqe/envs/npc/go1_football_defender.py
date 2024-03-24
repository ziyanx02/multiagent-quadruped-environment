
from mqe import LEGGED_GYM_ROOT_DIR
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from mqe import LEGGED_GYM_ROOT_DIR

from mqe.envs.go1.go1 import Go1
from copy import copy

class Go1FootballDefender(Go1):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):

        self.npc_collision = True
        self.fix_npc_base_link = False
        self.npc_gravity = True
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def step(self, action):
        
        defender_action = self._get_defender_action()

        if self.cfg.control.control_type == "C":
            action = torch.cat([action.reshape(self.num_envs, -1, 3), defender_action.unsqueeze(1)], dim=1).reshape(-1, 3)
            action = self.preprocess_action(action)
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(action, -clip_actions, clip_actions).reshape(self.num_envs, -1).to(self.device)
            # action = torch.zeros([self.num_envs, 12], device = "cuda")
        else:
            raise NotImplementedError
            actions = action.reshape(self.num_envs, -1)
            self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            torques = torch.cat((self.torques, torch.zeros((self.num_envs, self.num_actions_npc), dtype=torch.long, device=self.device)), dim=1) if self.num_actions_npc != 0 else self.torques

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
            self.post_decimation_step(dec_i)

        self.post_physics_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def _get_defender_action(self):

        defender_pos = self.root_states[:, :3].reshape(self.num_envs, -1, 3)[:, 2, :].clone()
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3).clone()

        if not hasattr(self, "gate_pos"):
            self.gate_pos = self.env_origins.clone()
            self.gate_pos[:, 0] += self.cfg.terrain.BarrierTrack_kwargs["init"]["block_length"] + self.cfg.terrain.BarrierTrack_kwargs["plane"]["block_length"]

        target_pos = (0.6 * ball_pos + 0.4 * self.gate_pos)

        defender_yaw = self.obs_buf.base_rpy.reshape(self.num_envs, self.num_agents, 3)[:, 2, 2]
        defender_yaw_to_gate = torch.pi + torch.atan((self.gate_pos - defender_pos)[:, 1] / (self.gate_pos - defender_pos)[:, 0])

        yaw_command = (defender_yaw_to_gate - defender_yaw).clip(-0.3, 0.3) / 0.3

        target_distance_to_gate = torch.norm((target_pos - self.gate_pos)[:, :2], dim=1)
        distance_to_gate = torch.norm((defender_pos - self.gate_pos)[:, :2], dim=1)

        x_command = (target_distance_to_gate - distance_to_gate).clip(-0.5, 0.5)
        y_command = - (self.gate_pos[:, 1] + (target_pos[:, 1] - self.gate_pos[:, 1]) * (defender_pos[:, 0] - self.gate_pos[:, 0]) / (target_pos[:, 0] - self.gate_pos[:, 0]) - defender_pos[:, 1]).clip(-0.5, 0.5)
        
        defender_command = torch.stack([x_command, y_command, yaw_command], dim=1)

        return defender_command

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
        
    def _create_npc(self, env_handle, env_id):

        npc_handles = []
        for i in range(self.num_npcs):
            pos = self.env_origins[env_id].clone()
            self.start_pose_npc.p = gymapi.Vec3(*pos)
            npc_handle = self.gym.create_actor(env_handle, self.asset_npc, self.start_pose_npc, self.cfg.asset.name_npc, env_id, not self.npc_collision, 0)
            npc_handles.append(npc_handle)
        return npc_handles
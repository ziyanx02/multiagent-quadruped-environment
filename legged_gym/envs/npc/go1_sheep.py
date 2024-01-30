
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from copy import copy
import random

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict

from legged_gym.envs.go1.go1 import Go1

class Go1Sheep(Go1):

    def _prepare_npc(self):
    
        #creat npc asset
        asset_path_npc = self.cfg.asset.file_npc.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root_npc = os.path.dirname(asset_path_npc)
        asset_file_npc = os.path.basename(asset_path_npc)
        asset_options_npc = gymapi.AssetOptions()
        asset_options_npc.fix_base_link = self.type_npc
        asset_options_npc.disable_gravity = self.type_npc
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
        
    def _create_npc(self, env_handle, i):

        npc_handles = []
        for i in range(self.num_npcs):
            pos = self.env_origins[i].clone()
            self.start_pose_npc.p = gymapi.Vec3(*pos)
            npc_handle = self.gym.create_actor(env_handle, self.asset_npc, self.start_pose_npc, self.cfg.asset.name_npc, i, self.cfg.asset.self_collisions, 0)
            npc_handles.append(npc_handle)
        return npc_handles
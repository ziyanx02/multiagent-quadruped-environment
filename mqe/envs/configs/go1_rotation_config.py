import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1RotationCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1rotationCfg"
        num_envs = 1 # 4096
        num_agents = 2
        num_npcs = 1
        num_actions_npc = 1
        episode_length_s = 5 # episode length in seconds
    
    class asset(Go1Cfg.asset):

        terminate_after_contacts_on = []
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/rotation_door.urdf"
        name_npc = "rotation"
        npc_collision = True
        fix_npc_base_link = True

    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "wall",
                "gate",
                "wall",
            ],
            randomize_obstacle_order = False,
            # wall_thickness= 0.2,
            track_width = 3.5,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 0,
                room_size = (0.0, 0.0),
                border_width = 0.00,
                offset = (0, 0),
            ),
            gate = dict(
                block_length = 5.0,
                width = 2.0,
                depth = 0.1, # size along the forward axis
                offset = (0, 0),
                random = (0, 0),
            ),
            rotation = dict(
                block_length = 5,
                depth = 0.1, # size along the forward axis
                offset = (0, 0),
                wide_px = (0.84,0.2)
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.85,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False,
       ))
        
        x_limits = [5.0,]
        y_limits = [-1.5, 1.5]

    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [0.5, -1.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [0.5, 1.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            init_state_class(
                pos = [2.59, -0.01, 0.04],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class control(Go1Cfg.control):
        control_type = 'C'

    class termination(Go1Cfg.termination):

        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        init_base_pos_range = None
        init_npc_base_pos_range = None

    class rewards(Go1Cfg.rewards):
        class scales:

            punishment_scale = 1
            success_reward_scale = 10
            distance_reward_scale = 1

    class viewer(Go1Cfg.viewer):
        pos = [12., 20., 20.]  # [m]
        lookat = [13., 20., 0.]  # [m]

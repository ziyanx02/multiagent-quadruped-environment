import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1WrestlingCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1wrestling"
        num_envs = 1 # 4096
        num_agents = 2
        env_type = 1
        num_npcs = 1
        episode_length_s = 15 # episode length in seconds

    class asset(Go1Cfg.asset):
        terminate_after_contacts_on = []
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/wrestling_field/urdf/wrestling.urdf"
        name_npc = "wrestling"
        fix_npc_base_link = True

    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "plane",
            ],
            randomize_obstacle_order = False,
            # wall_thickness= 0.2,
            track_width = 6,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 0.0,
                room_size = (0.0, 0.0),
                border_width = 0.00,
                offset = (0, 0),
            ),
            wall = dict(
                block_length = 0.1
            ),
            plane = dict(
                block_length = 7,
            ),
            wall_height= 0.001,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False
       ))

        TerrainPerlin_kwargs = merge_dict(Go1Cfg.terrain.TerrainPerlin_kwargs, dict(
            zScale = [0.05, 0.1],
       ))
    
    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [3.1, 1., 0.74],
                rot = [0.0, 0.0, -1.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [3.1, -1.0, 0.74],
                rot = [0.0, 0.0, 1.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            init_state_class(
                pos = [3.1, 0.0, 0.0],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
                
            )
        ]

    class control(Go1Cfg.control):
        control_type = 'C'

    class termination(Go1Cfg.termination):

        termination_terms = [
            "roll",
            "pitch",
            "z_low",
        ]
        z_low_kwargs = dict(
            threshold= 0.3, # [m]
        )

    class domain_rand(Go1Cfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False # use for non-virtual training
        init_dof_pos_ratio_range = None
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )
        init_npc_base_pos_range = None
    class rewards(Go1Cfg.rewards):
        class scales:

            punishment_scale = 1
            success_reward_scale = 10
            
    class viewer(Go1Cfg.viewer):
        pos = [0., 3., 5.]  # [m]
        lookat = [4., 3., 0.]  # [m]

import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1TugCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1tug"
        num_envs = 1 # 4096
        num_agents = 2
        num_npcs = 1
        num_actions_npc = 1
        env_type = 1
        episode_length_s = 15 # episode length in seconds

    class asset(Go1Cfg.asset):

        terminate_after_contacts_on = []
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/cylinder.urdf"
        name_npc = "circular"
        fix_npc_base_link = True

    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "wall",
                "plane",
                "wall",
            ],
            randomize_obstacle_order = False,
            # wall_thickness= 0.2,
            track_width = 6.,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 0.0,
                room_size = (.0, 0.0),
                border_width = 0.00,
                offset = (0, 0),
            ),
            plane = dict(
                block_length = 3.0,
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 1.0,
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
                pos = [1.6, 2.5, 0.34],
                rot = [0.0, 0.0, -1.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [1.6, -2.5, 0.34],
                rot = [0.0, 0.0, 1., 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            )
        ]
        init_states_npc = [
            init_state_class(
                pos = [1.6, 0.0, 0.0],
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
            "z_high",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False # use for non-virtual training
        init_dof_pos_ratio_range = None
        init_base_pos_range = dict(
            x= [-1.0, 1.0],
            y= [-0.0, 0.0],
        )
        init_npc_base_pos_range = None

    class rewards(Go1Cfg.rewards):
        class scales:

            success_reward_scale = 10
            punishment_reward_scale = 10
            pos_reward_scale = 2
            pos_punishment_scale = 2
        
    class viewer(Go1Cfg.viewer):
        pos = [0., 11., 5.]  # [m]
        lookat = [4., 11., 0.]  # [m]

import numpy as np
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go1.go1 import Go1Cfg

class Go1FootballCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1foolball"
        num_envs = 1 # 4096
        num_agents = 2
        num_npcs = 1
        env_type = 1
        obs_components = [
            "proprioception", # 48
            # "height_measurements", # 187
            "base_pose",
            "robot_config",
            # "engaging_block",
            # "sidewall_distance",
            # "forward_depth",
        ]
        episode_length_s = 5 # episode length in seconds
    
    class asset(Go1Cfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/ball.urdf"
        name_npc = "circular"

    class terrain(Go1Cfg.terrain):

        # mesh_type = "plane"
        # selected = False
        mesh_type = "trimesh"
        selected = "BarrierTrack"
        num_rows = 1 # 20
        num_cols = 1 # 50
        max_init_terrain_level = 2
        border_size = 1
        slope_treshold = 20.
        curriculum = False

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "football"
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
            football = dict(
                block_length = 8.0,
                height = 4.0,
                width = 2.0,
                deepth = 1.0,
                offset = (0, 0),
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
            vel = False         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [1.5, 0.0, 0.34],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [6.5, .0, 0.34],
                rot = [0.0, 0.0, 1., 0.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            )
        ]
        init_states_npc = [
            init_state_class(
                pos = [2.2, 0.0, 0.5],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
                
            )
        ]

    class control(Go1Cfg.control):
        control_type = 'C'

    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            # "roll",
            # "pitch",
            # "z_low",
            # "z_high",
            # "out_of_track",
        ]

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

            catch_ball_reward_scale = 1
        
    class viewer(Go1Cfg.viewer):
        pos = [0., 11., 5.]  # [m]
        lookat = [4., 11., 0.]  # [m]

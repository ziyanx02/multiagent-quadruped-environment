import numpy as np
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go1.go1 import Go1Cfg

class Go1GateCfg(Go1Cfg):

    class env(Go1Cfg.env):
        num_envs = 1 # 4096
        num_agents = 2
        obs_components = [
            "proprioception", # 48
            # "height_measurements", # 187
            "base_pose",
            "robot_config",
            # "engaging_block",
            # "sidewall_distance",
            # "forward_depth",
        ]

    #### uncomment this to train non-virtual terrain
    # class sensor(A1FieldCfg.sensor):
    #     class proprioception(A1FieldCfg.sensor.proprioception):
    #         delay_action_obs = True
    #         latency_range = [0.04-0.0025, 0.04+0.0075]
    #### uncomment the above to train non-virtual terrain

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
                "gate",
                "wall",
            ],
            randomize_obstacle_order = False,
            # wall_thickness= 0.2,
            track_width = 2.6,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 2.0,
                room_size = (1.0, 0.8),
                border_width = 0.00,
                offset = (0, 0),
            ),
            gate = dict(
                block_length = 3.0,
                width = 0.6,
                depth = 0.1, # size along the forward axis
                offset = (0, 0),
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False
       ))

        TerrainPerlin_kwargs = merge_dict(Go1Cfg.terrain.TerrainPerlin_kwargs, dict(
            zScale = [0.05, 0.1],
       ))
    
    class commands(Go1Cfg.commands):
        class ranges(Go1Cfg.commands.ranges):
            lin_vel_x = [0.3, 0.6]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]

    class control(Go1Cfg.control):
        control_type = 'C'

    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            "out_of_track",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False # use for non-virtual training
        init_dof_pos_ratio_range = None
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )

    class rewards(Go1Cfg.rewards):
        class scales:
            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -1e-5
            alive = 2.
            # penetrate_depth = -3e-3
            # penetrate_volume = -3e-3
            # exceed_dof_pos_limits = -1e-1
            # exceed_torque_limits_i = -2e-1

    class viewer(Go1Cfg.viewer):
        pos = [0., 11., 5.]  # [m]
        lookat = [4., 11., 0.]  # [m]

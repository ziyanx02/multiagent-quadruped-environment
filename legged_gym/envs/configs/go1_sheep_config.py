import numpy as np
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go1.go1 import Go1Cfg

class SingleSheepCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1sheep"
        num_envs = 1 # 4096
        num_agents = 2
        num_npcs = 1
        episode_length_s = 15
    
    class asset(Go1Cfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/sheep.urdf"
        name_npc = "sheep"

        num_rows = 1
        num_cols = 1
        dis_sheep = (1.5, 1.5)

        sheep_movement_scale = 0.2
        sheep_movement_randomness = 0.0
        sheep_movement_range = [2.0, 2.0, 0]

    
    class terrain(Go1Cfg.terrain):

        num_rows = 2 # 20
        num_cols = 1 # 50

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "plane",
                "gate",
                "plane",
                "wall",
            ],
            randomize_obstacle_order = False,
            # wall_thickness= 0.2,
            track_width = 4.,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 1.5,
                room_size = (1.0, 1.95),
                border_width = 0.00,
                offset = (0.5, 0),
            ),
            gate = dict(
                block_length = 1.0,
                width = 1.,
                depth = 0.1, # size along the forward axis
                offset = (0, 0),
                random = (0, 0.5)
            ),
            plane = dict(
                block_length = 3.0,
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False
       ))

    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [0.0, 0.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [0.0, 0.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )
        init_npc_base_pos_range = dict(
            x= [-0.3, 0.3],
            y= [-0.3, 0.3],
        )


    class rewards(Go1Cfg.rewards):
        class scales:
            success_reward_scale = 1
            contact_punishment_scale = -0
            sheep_movement_reward_scale = 2
            # sheep_variance_reward_scale = 1
            # tracking_ang_vel = 0.05
            # world_vel_l2norm = -1.
            # legs_energy_substeps = -1e-5
            # alive = 2.
            # penetrate_depth = -3e-3
            # penetrate_volume = -3e-3
            # exceed_dof_pos_limits = -1e-1
            # exceed_torque_limits_i = -2e-1

    class viewer(Go1Cfg.viewer):
        pos = [0., 3., 5.]  # [m]
        lookat = [4., 3., 0.]  # [m]

class NineSheepCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1sheep"
        num_envs = 4 # 4096
        num_agents = 2
        num_npcs = 9
        episode_length_s = 15
    
    class asset(Go1Cfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/sheep.urdf"
        name_npc = "sheep"

        num_rows = 3
        num_cols = 3
        dis_sheep = (1.5, 1.5)

        sheep_movement_scale = 0.2
        sheep_movement_randomness = 0.1
        sheep_movement_range = [2.0, 2.0, 0]
    
    class terrain(Go1Cfg.terrain):

        num_rows = 2 # 20
        num_cols = 1 # 50

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "plane",
                "gate",
                "plane",
                "wall",
            ],
            randomize_obstacle_order = False,
            # wall_thickness= 0.2,
            track_width = 6.,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 2,
                room_size = (1.0, 1.95),
                border_width = 0.00,
                offset = (0.5, 0),
            ),
            gate = dict(
                block_length = 1.0,
                width = 1.5,
                depth = 0.1, # size along the forward axis
                offset = (0, 0),
                random = (0, 0)
            ),
            plane = dict(
                block_length = 6.0,
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False
       ))

    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [0.0, 0.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [0.0, 0.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )
        init_npc_base_pos_range = dict(
            x= [-0.3, 0.3],
            y= [-0.3, 0.3],
        )


    class rewards(Go1Cfg.rewards):
        class scales:
            success_reward_scale = 1
            contact_punishment_scale = -0
            sheep_movement_reward_scale = 20
            # sheep_variance_reward_scale = 1
            # tracking_ang_vel = 0.05
            # world_vel_l2norm = -1.
            # legs_energy_substeps = -1e-5
            # alive = 2.
            # penetrate_depth = -3e-3
            # penetrate_volume = -3e-3
            # exceed_dof_pos_limits = -1e-1
            # exceed_torque_limits_i = -2e-1

    class viewer(Go1Cfg.viewer):
        pos = [0., 3., 5.]  # [m]
        lookat = [4., 3., 0.]  # [m]

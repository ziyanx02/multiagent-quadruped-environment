import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1FootballDefenderCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1football"
        num_envs = 1
        num_agents = 3
        num_npcs = 1
        episode_length_s = 20
    
    class asset(Go1Cfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/ball.urdf"
        name_npc = "ball"
        terminate_after_contacts_on = []
        npc_collision = True
        fix_npc_base_link = False
        npc_gravity = True
    
    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1
 
        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "gate",
                "plane",
                "gate",
                "wall",
            ],
            # wall_thickness= 0.2,
            track_width = 9.0,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 1.0,
                room_size = (0, 3.0),
                border_width = 0.00,
                offset = (0.5, 0),
            ),
            plane = dict(
                block_length = 10.0,
            ),
            gate = dict(
                block_length = 1.0,
                width = 2.0,
                depth = 1.0, # size along the forward axis
                offset = (0, 0),
                random = (0, 0.),
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 1.0,
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
                pos = [3.0, 1.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [3.0, 2.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [9.0, -3.0, 0.42],
                rot = [0.0, 0.0, 1.0, 0.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            init_state_class(
                pos = [5.0, -2.1, 0.3],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class control(Go1Cfg.control):
        control_type = 'C'

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

    class rewards(Go1Cfg.rewards):
        class scales:
            goal_reward_scale = 10
            ball_gate_distance_reward_scale = 3
            # tracking_ang_vel = 0.05
            # world_vel_l2norm = -1.
            # legs_energy_substeps = -1e-5
            # alive = 2.
            # penetrate_depth = -3e-3
            # penetrate_volume = -3e-3
            # exceed_dof_pos_limits = -1e-1
            # exceed_torque_limits_i = -2e-1

    class viewer(Go1Cfg.viewer):
        pos = [2., 2., 2.]  # [m]
        lookat = [6., 5., 0.]  # [m]

class Go1Football1vs1Cfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1football"
        num_envs = 1
        num_agents = 2
        num_npcs = 1
        episode_length_s = 1
    
    class asset(Go1Cfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/ball.urdf"
        name_npc = "ball"
        terminate_after_contacts_on = []
        npc_collision = True
        fix_npc_base_link = False
        npc_gravity = True
    
    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1
 
        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "gate",
                "plane",
                "gate",
                "wall",
            ],
            # wall_thickness= 0.2,
            track_width = 9.0,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 1.0,
                room_size = (0.0, 0.0),
                border_width = 0.00,
                offset = (0.5, 0),
            ),
            plane = dict(
                block_length = 10.0,
            ),
            gate = dict(
                block_length = 1.0,
                width = 2.0,
                depth = 1.0, # size along the forward axis
                offset = (0, 0),
                random = (0, 0.),
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 1.0,
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
                pos = [3.0, 0.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [9.0, 0.0, 0.42],
                rot = [0.0, 0.0, 1.0, 0.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            init_state_class(
                pos = [7.0, 0.0, 0.2],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
    class control(Go1Cfg.control):
        control_type = 'C'

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

    class rewards(Go1Cfg.rewards):
        class scales:
            goal_reward_scale = 1

    class viewer(Go1Cfg.viewer):
        pos = [2., 2., 2.]  # [m]
        lookat = [6., 5., 0.]  # [m]

class Go1Football2vs2Cfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1football"
        num_envs = 1
        num_agents = 4
        num_npcs = 1
        episode_length_s = 20
    
    class asset(Go1Cfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/ball.urdf"
        name_npc = "ball"
        terminate_after_contacts_on = []
        npc_collision = True
        fix_npc_base_link = False
        npc_gravity = True
    
    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1
 
        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "gate",
                "plane",
                "gate",
                "wall",
            ],
            # wall_thickness= 0.2,
            track_width = 9.0,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 1.0,
                room_size = (0, 0),
                border_width = 0.00,
                offset = (0.5, 0),
            ),
            plane = dict(
                block_length = 10.0,
            ),
            gate = dict(
                block_length = 1.0,
                width = 2.0,
                depth = 1.0, # size along the forward axis
                offset = (0, 0),
                random = (0, 0.),
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 1.0,
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
                pos = [3.0, 2.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [3.0, -2.0, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [9.0, 2.0, 0.42],
                rot = [0.0, 0.0, 1.0, 0.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [9.0, -2.0, 0.42],
                rot = [0.0, 0.0, 1.0, 0.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            init_state_class(
                pos = [7.0, 0.0, 0.2],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class control(Go1Cfg.control):
        control_type = 'C'

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

    class rewards(Go1Cfg.rewards):
        class scales:
            goal_reward_scale = 1

    class viewer(Go1Cfg.viewer):
        pos = [2., 2., 2.]  # [m]
        lookat = [6., 5., 0.]  # [m]

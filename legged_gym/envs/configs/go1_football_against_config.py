import numpy as np
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go1.go1 import Go1Cfg

class Go1FootballAgainstCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1football"
        num_envs = 2 # 4096
        num_agents = 4
        num_npcs = 1
        episode_length_s = 5
    
    class asset(Go1Cfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/ball.urdf"
        name_npc = "ball"
        terminate_after_contacts_on = []
        npc_collision = True
        fix_npc_base_link = False
        npc_gravity = True
    
    class terrain(Go1Cfg.terrain):

        num_rows = 1 # 20
        num_cols = 1 # 50
 
        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "football"
            ],
            randomize_obstacle_order = False,
            # wall_thickness= 0.2,
            track_width = 9.0,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 0.0,
                room_size = (0.0, 0.0),
                border_width = 0.00,
                offset = (0, 0),
            ),
            football = dict(
                block_length = 12.0,
                height = 4.0,
                width = 2.0,
                deepth = 1.0,
                offset = (0, 0),
            ),
            wall_height= 2.0,
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
                pos = [6 , 0.0, 0.3],
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
            goal_punishment_scale = 0
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
        lookat = [4., 2., 0.]  # [m]

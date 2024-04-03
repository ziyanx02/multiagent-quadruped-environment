import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1DoorCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1door"
        num_envs = 1
        num_agents = 2
        num_npcs = 1
        num_actions_npc = 1
        episode_length_s = 15 # episode length in seconds
    
    class asset(Go1Cfg.asset):
        terminate_after_contacts_on = []
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/door.urdf"
        name_npc = "seesaw"
        npc_collision = True
        fix_npc_base_link = True
        npc_gravity = True
    
    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "gate",
                "plane",
                "wall",
            ],
            # wall_thickness= 0.2,
            track_width = 3.0,
            # track_block_length = 2., # the x-axis distance from the env origin point
            init = dict(
                block_length = 2.0,
                room_size = (1.0, 1.5),
                border_width = 0.00,
                offset = (0, 0),
            ),
            gate = dict(
                block_length = 4.0,
                width = 1.1,
                depth = 0.1, # size along the forward axis
                offset = (0, 0),
                random = (0.0, 0.0),
            ),
            plane = dict(
                block_length = 1.0,
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False,
       ))

    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [0.0, 0.5, 0.42],
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
        init_states_npc = [
            init_state_class(
                pos = [2.5, -0.5, 1.05],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        default_npc_joint_angles = [0.0]

    class control(Go1Cfg.control):
        control_type = 'C'

        class default_command(Go1Cfg.control.default_command):

            gait = "pacing"

    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )
        init_npc_base_pos_range = None

    class obs(Go1Cfg.obs):

        class cfgs(Go1Cfg.obs.cfgs):

            env_info = False

    class rewards(Go1Cfg.rewards):
        class scales:
            
            height_reward_scale = 1
            success_reward_scale = 10
            contact_punishment_scale = -2
            agent_distance_punishment_scale = -0.25
            x_movement_reward_scale = 5
            fall_punishment_scale = -2
            y_punishment_scale = -0.5
            # tracking_ang_vel = 0.05
            # world_vel_l2norm = -1.
            # legs_energy_substeps = -1e-5
            # alive = 2.
            # penetrate_depth = -3e-3
            # penetrate_volume = -3e-3
            # exceed_dof_pos_limits = -1e-1
            # exceed_torque_limits_i = -2e-1

    class viewer(Go1Cfg.viewer):
        pos = [0., -2., 4.]  # [m]
        lookat = [4., 2., 0.]  # [m]

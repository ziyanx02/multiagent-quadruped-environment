import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1GateCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1gate"
        num_envs = 1
        num_agents = 2
        episode_length_s = 10 # episode length in seconds

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
                block_length = 3.0,
                width = 0.6,
                depth = 0.1, # size along the forward axis
                offset = (0, 0),
                random = (0.5, 0.5),
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
            # init_state_class(
            #     pos = [0.0, 0.5, 0.42],
            #     rot = [0.0, 0.0, 0.0, 1.0],
            #     lin_vel = [0.0, 0.0, 0.0],
            #     ang_vel = [0.0, 0.0, 0.0],
            # ),
            # init_state_class(
            #     pos = [0.0, 1.0, 0.42],
            #     rot = [0.0, 0.0, 0.0, 1.0],
            #     lin_vel = [0.0, 0.0, 0.0],
            #     ang_vel = [0.0, 0.0, 0.0],
            # ),
            # init_state_class(
            #     pos = [2.8, -0.3, 0.42],
            #     rot = [0.0, 0.0, 0.0, 1.0],
            #     lin_vel = [0.0, 0.0, 0.0],
            #     ang_vel = [0.0, 0.0, 0.0],
            # ),
            # init_state_class(
            #     pos = [3.6, -1, 0.42],
            #     rot = [0.0, 0.0, 0.0, 1.0],
            #     lin_vel = [0.0, 0.0, 0.0],
            #     ang_vel = [0.0, 0.0, 0.0],
            # ),
        ]

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
        ]

    class domain_rand(Go1Cfg.domain_rand):
        init_base_pos_range = None
 
    class rewards(Go1Cfg.rewards):
        class scales:

            target_reward_scale = 1
            success_reward_scale = 5
            lin_vel_x_reward_scale = 0
            approach_frame_punishment_scale = 0
            agent_distance_punishment_scale = -0.025
            contact_punishment_scale = -2
            lin_vel_y_punishment_scale = 0
            command_value_punishment_scale = 0

    class viewer(Go1Cfg.viewer):
        pos = [-2., 2.5, 4.]  # [m]
        lookat = [4., 2.5, 0.]  # [m]

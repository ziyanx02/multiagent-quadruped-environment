
from mqe.envs.base.legged_robot_config import LeggedRobotCfg

class LeggedRobotFieldCfg( LeggedRobotCfg ):

    class terrain( LeggedRobotCfg.terrain ):
        
        num_rows = 20
        num_cols = 50
        selected = "BarrierTrack"
        max_init_terrain_level = 0 # for climb, leap finetune
        border_size = 1
        slope_treshold = 100.

        curriculum = False # for tilt, crawl, climb, leap
        # curriculum = False # for walk
        horizontal_scale = 0.025 # [m]
        pad_unavailable_info = True

        BarrierTrack_kwargs = dict(
            options = [
                "init",
                "gate",
                "wall",
                "plane",
            ],
            wall_thickness= 0.04,
            track_width = 2.,
            # track_block_length = 2., # the x-axis distance from the env origin point
            wall = dict(
                block_length = 3.0,
            ),
            plane = dict(
                block_length = 3.0,
            ),
            init = dict(
                block_length = 3.0,
                room_size = (1.0, 1.0),
                border_width = 0.00,
                offset = (0, 0),
            ),
            gate = dict(
                block_length = 1.6,
                width = 0.5,
                depth = 0.1, # size along the forward axis
                offset = (0.4, 0),
                random = (0., 0.),
            ),
            wall_height= 0.5,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False,
            border_perlin_noise= False,
            border_height= 0., # for climb, crawl, tilt, walk
            # border_height= -0.5, # for leap
            # virtual_terrain= True, # for tilt
            engaging_next_threshold= 1.2,
            curriculum_perlin= False,
            # no_perlin_threshold= 0.05, # for leap
            # no_perlin_threshold= 0.06, # for climb
            )

        TerrainPerlin_kwargs = dict(
            # zScale= 0.1, # for crawl
            zScale= 0.12, # for tilt
            # zScale= [0.05, 0.1], # for climb
            # zScale= [0.04, 0.1], # for leap
            # zScale= [0.1, 0.15], # for walk
            frequency= 10,
        )

    class sensor:
        class forward_camera:
            resolution = [16, 16]
            position = [0.26, 0., 0.03] # position in base_link
            rotation = [0., 0., 0.] # ZYX Euler angle in base_link
    
        class proprioception:
            delay_action_obs = False
            latency_range = [0.0, 0.0]
            latency_resample_time = 2.0 # [s]

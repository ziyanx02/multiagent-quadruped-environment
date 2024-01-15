
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class LeggedRobotFieldCfg( LeggedRobotCfg ):

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "trimesh" # Don't change
        num_rows = 20 # 20
        num_cols = 50 # 50
        selected = "BarrierTrack" # "BarrierTrack" or "TerrainPerlin", "TerrainPerlin" can be used for training a walk policy.
        max_init_terrain_level = 0
        border_size = 5
        slope_treshold = 20.

        curriculum = False # for walk
        horizontal_scale = 0.025 # [m]
        # vertical_scale = 1. # [m] does not change the value in hightfield
        pad_unavailable_info = True

        BarrierTrack_kwargs = dict(
            options= [
                # "climb",
                # "crawl",
                # "tilt",
                # "leap",
            ], # each race track will permute all the options
            track_width= 1.6,
            track_block_length= 2., # the x-axis distance from the env origin point
            wall_thickness= (0.04, 0.2), # [m]
            wall_height= -0.05,
            climb= dict(
                height= (0.2, 0.6),
                depth= (0.1, 0.8), # size along the forward axis
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                climb_down_prob= 0.0,
            ),
            crawl= dict(
                height= (0.25, 0.5),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            tilt= dict(
                width= (0.24, 0.32),
                depth= (0.4, 1.), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.5,
            ),
            leap= dict(
                length= (0.2, 1.0),
                depth= (0.4, 0.8),
                height= 0.2,
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            curriculum_perlin= False,
            no_perlin_threshold= 0.0,
        )

        TerrainPerlin_kwargs = dict(
            zScale= [0.1, 0.15],
            # zScale= 0.1, # Use a constant zScale for training a walk policy
            frequency= 10,
        )
    
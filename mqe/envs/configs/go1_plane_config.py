
from mqe.envs.go1.go1_config import Go1Cfg

class Go1PlaneCfg(Go1Cfg):
    class env(Go1Cfg.env):
        env_name = "go1plane"
        use_lin_vel = True
        num_envs = 25
        use_lin_vel = True
        num_actions = 12
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        # recording cfgs
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1

    class terrain:
        mesh_type = "plane"
        selected = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        x_init_range = 1.
        y_init_range = 1.
        yaw_init_range = 0.
        x_init_offset = 0.
        y_init_offset = 0.

    # class obs(Go1Cfg):
    #     pass

from legged_gym.envs.go1.go1_config import Go1Cfg

class Go1PlaneCfg(Go1Cfg):
    class env(Go1Cfg.env):
        env_name = "go1plane"
        use_lin_vel = True
        num_envs = 25
        num_observations = 235
        use_lin_vel = True
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
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
        # mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # horizontal_scale = 0.1 # [m]
        # vertical_scale = 0.005 # [m]
        # border_size = 0 # [m]
        # curriculum = True
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
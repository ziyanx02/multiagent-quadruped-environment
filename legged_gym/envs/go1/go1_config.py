# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.field.legged_robot_field_config import LeggedRobotFieldCfg

class Go1Cfg(LeggedRobotFieldCfg):

    class env(LeggedRobotFieldCfg.env):
        use_lin_vel = True
        num_envs = 256
        num_observations = 235
        use_lin_vel = True
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 5 # episode length in seconds


        # recording cfgs
        recording_width_px = 360
        recording_height_px = 240
        recording_mode = "COLOR"
        num_recording_envs = 1

    class asset:

        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
        name = "go1"
        foot_name = "foot"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = ["base", "thigh"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class init_state(LeggedRobotFieldCfg.init_state):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FR_hip_joint': -0.1 ,  # [rad]
            'FL_hip_joint': 0.1,   # [rad]
            'RR_hip_joint': -0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class normalization(LeggedRobotFieldCfg.normalization):
        clip_actions = 10.

    class control(LeggedRobotFieldCfg.control):
        control_type = 'C' # P: position, V: velocity, T: torques, C: command
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        # action_scale = [0.2, 0.4, 0.4] * 4 # for walk
        action_scale = 0.25
        # for climb, leap
        torque_limits = [20., 20., 25.] * 4
        computer_clip_torque = True
        motor_clip_torque = False
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5

        locomotion_policy_dir = "./legged_gym/utils/locomotion_checkpoints/walk_these_ways"
        actuator_network_path = "./resources/actuator_nets"

        class default_command:

            lin_vel_x = 1.0
            lin_vel_y = -0.0
            ang_vel = -0.0
            body_height = 0.0
            gait_freq = 3.0
            gait = "trotting"
            # gait = "pacing"
            footswing_height = 0.08
            body_pitch = 0.0
            body_roll = 0.0
            stance_width = 0.25
            stance_length = 0.428
            aux_reward = 0.0
        
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            body_height = 2.0
            gait_phase = 1.0
            gait_freq = 1.0
            footswing_height = 0.15
            body_pitch = 0.3
            body_roll = 0.3
            aux_reward = 1.0
            compliance = 1.0
            stance_width = 1.0
            stance_length = 1.0

    class command:
            
        gaits = {"pronking": [0, 0, 0],
                "trotting": [0.5, 0, 0],
                "bounding": [0, 0.5, 0],
                "pacing": [0, 0, 0.5]}

        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error

        class cfg:
            vel = False         # lin_vel, ang_vel
            body_height = False
            body_pose = False   # body_pitch, body_roll
            gait_freq = False
            gait = False
            footswing_height = False
            stance_width = False
            stance_length = False
            aux_reward = False

        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class termination:
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
        ]

        roll_kwargs = dict(
            threshold= 0.8, # [rad] # for tilt
        )
        pitch_kwargs = dict(
            threshold= 1.6,
        )
        z_low_kwargs = dict(
            threshold= 0.08, # [m]
        )
        z_high_kwargs = dict(
            threshold= 1.5, # [m]
        )
        out_of_track_kwargs = dict(
            threshold= 1., # [m]
        )

        # check_obstacle_conditioned_threshold = True
        # timeout_at_border = True
        # timeout_at_finished = True

    class domain_rand(LeggedRobotFieldCfg.domain_rand):
        randomize_com = False
        class com_range:
            x = [-0.05, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]
        
        randomize_motor = False
        leg_motor_strength_range = [0.9, 1.1]

        randomize_base_mass = False
        added_mass_range = [-1.0, 3.0]

        randomize_friction = False
        friction_range = [0.05, 4.5]

        randomize_lag_timesteps = False
        lag_timesteps = 6

        init_base_pos_range = dict(
            x= [0.1, 0.1],
            y= [-0.1, 0.1],
        )

        init_dof_pos_ratio_range = [0.7, 1.3]

        init_npc_base_pos_range = dict(
            x= [-0.2, 0.2],
            y= [-0.2, 0.2],
        )

        push_robots = False

    class obs:

        class cfgs:
            base_pos = True
            base_quat = True
            dof_pos = True
            dof_vel = True
            lin_vel = True
            ang_vel = True
            projected_gravity = True
            base_rpy = True
            contact_states = False
            command = True
            height_command = False
            gait_commands = False
            timing_parameter = False
            clock_inputs = False
            last_action = True
            last_last_action = True
            imu = False

            depth_image = False
            rgb_image = False
            env_info = True
        
        class scales:
            base_pos = 1.0
            base_quat = 1.0
            segmentation_image = 1.0
            rgb_image = 1.0
            depth_image = 1.0

        def keys(self):
            key_dict = dir(self.cfgs)
            key_list = []
            for key in key_dict:
                if getattr(self.cfgs, key) == True and key:
                    key_list.append(key)
            return key_list

    class privileged_obs:
        
        class cfgs:
            pass
        
        def keys(self):
            key_dict = dir(self.cfgs)
            print("===Available Obsevations===")
            for key in key_dict:
                if getattr(self.cfgs, key) == True and key:
                    print(key, end=" ")
            print("\n===========================")

    class rewards(LeggedRobotFieldCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales(LeggedRobotFieldCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0

    class viewer(LeggedRobotFieldCfg.viewer):
        pos = [0., 11., 5.]  # [m]
        lookat = [4., 11., 0.]  # [m]

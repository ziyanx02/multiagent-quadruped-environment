import numpy as np
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from copy import copy

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.field.legged_robot_field import LeggedRobotField
from legged_gym.envs.go1.go1_config import Go1Cfg
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict

class Go1(LeggedRobotField):
    def __init__(self, cfg: Go1Cfg, sim_params, physics_engine, sim_device, headless):

        self.cfg = cfg
        self.env_name = cfg.env.env_name
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        self.obs_buf = copy(self.cfg.obs)
        self.privileged_obs_buf = copy(self.cfg.privileged_obs)

        self.last_locomotion_action = torch.zeros(self.num_envs * self.num_agents, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_two_locomotion_action = torch.zeros(self.num_envs * self.num_agents, 12, dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.control.control_type == "C":
            self._prepare_locomotion_policy()

    def step(self, action):
        
        if self.cfg.control.control_type == "C":
            action = torch.clip(action, -1, 1)
            action = self.preprocess_action(action)
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(action, -clip_actions, clip_actions).reshape(self.num_envs, -1).to(self.device)
            # action = torch.zeros([self.num_envs, 12], device = "cuda")
        else:
            actions = action.reshape(self.num_envs, -1)
            self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            torques = torch.cat((self.torques, torch.zeros((self.num_envs, self.num_actions_npc), dtype=torch.long, device=self.device)), dim=1) if self.num_actions_npc != 0 else self.torques

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            
            self.post_decimation_step(dec_i)

        self.post_physics_step()
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def preprocess_action(self, actions):

        if self.cfg.command.cfg.vel:
            self.locomotion_obs[:, 3:5] = actions[:, self.vel_idx : self.vel_idx + 2] * self.cfg.control.obs_scales.lin_vel
            self.locomotion_obs[:, 5] = actions[:, self.vel_idx + 2] * self.cfg.control.obs_scales.ang_vel
        
        if self.cfg.command.cfg.body_height:
            self.locomotion_obs[:, 6] = actions[:, self.body_height_idx] * self.cfg.control.obs_scales.body_height
        
        if self.cfg.command.cfg.gait_freq:
            self.locomotion_obs[:, 7] = actions[:, self.gait_freq_idx] * self.cfg.control.obs_scales.gait_freq

        if self.cfg.command.cfg.gait:
            raise NotImplementedError
            
        if self.cfg.command.cfg.footswing_height:
            self.locomotion_obs[:, 12] = actions[:, self.footswing_height_idx] * self.cfg.control.obs_scales.footswing_height

        if self.cfg.command.cfg.body_pose:
            self.locomotion_obs[:, 13] = actions[:, self.body_pose_idx] * self.cfg.control.obs_scales.body_pitch
            self.locomotion_obs[:, 14] = actions[:, self.body_pose_idx+1] * self.cfg.control.obs_scales.body_roll

        if self.cfg.command.cfg.stance_width:
            self.locomotion_obs[:, 15] = actions[:, self.stance_width_idx] * self.cfg.control.obs_scales.stance_width

        if self.cfg.command.cfg.stance_length:
            self.locomotion_obs[:, 16] = actions[:, self.stance_length_idx] * self.cfg.control.obs_scales.stance_length

        if self.cfg.command.cfg.aux_reward:
            self.locomotion_obs[:, 17] = actions[:, self.aux_reward_idx] * self.cfg.control.obs_scales.aux_reward

        self.locomotion_obs[:, 0 : 3] = self.obs_buf.projected_gravity
        self.locomotion_obs[:, 18 : 30] = self.obs_buf.dof_pos
        self.locomotion_obs[:, 30 : 42] = self.obs_buf.dof_vel
        self.locomotion_obs[:, 42 : 54] = self.last_locomotion_action
        self.locomotion_obs[:, 54 : 66] = self.last_two_locomotion_action
        self.locomotion_obs[:, 66 : 70] = self.obs_buf.clock_inputs
        obs = self.locomotion_obs
        self.history_locomotion_obs = torch.cat((self.history_locomotion_obs[:, 70:], self.locomotion_obs), dim=-1)

        locomotion_action = self.locomotion_policy(self.history_locomotion_obs)

        self.last_two_locomotion_action = self.last_locomotion_action
        self.last_locomotion_action = locomotion_action
        return locomotion_action

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        self.reset_ids = env_ids
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)
        
        self._fill_extras(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # self._resample_commands(env_ids)
        self._reset_buffers(env_ids)

        self.store_recording(env_ids)
    
    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)
        agent_ids = self.env_agent_indices[env_ids].reshape(-1)
        self.gait_indices[agent_ids] = 0
        self.history_locomotion_obs[agent_ids] = 0

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf
    
    def compute_observations(self):
        """ Computes observations
        """
        assert self.dof_pos.shape[1] % self.num_agents == 0, "DOF number is not compatible with agent number"
        dof_num = self.dof_pos.shape[1] // self.num_agents
        dof_pos = (self.dof_pos - self.default_dof_pos).reshape(-1, dof_num)
        dof_vel = self.dof_vel.reshape(-1, dof_num)

        if self.cfg.obs.cfgs.base_pos:
            self.obs_buf.base_pos = (self.base_pos - self.env_origins_repeat) * self.cfg.obs.scales.base_pos

        if self.cfg.obs.cfgs.base_quat:
            self.obs_buf.base_quat = self.base_quat * self.cfg.obs.scales.base_quat

        if self.cfg.obs.cfgs.dof_pos or self.cfg.control.control_type == "C":
            self.obs_buf.dof_pos = dof_pos * self.obs_scales.dof_pos

        if self.cfg.obs.cfgs.dof_vel or self.cfg.control.control_type == "C":
            self.obs_buf.dof_vel = dof_vel * self.obs_scales.dof_vel

        if self.cfg.obs.cfgs.lin_vel or self.cfg.control.control_type == "C":
            self.obs_buf.lin_vel = self.base_lin_vel * self.obs_scales.lin_vel

        if self.cfg.obs.cfgs.ang_vel or self.cfg.control.control_type == "C":
            self.obs_buf.ang_vel = self.base_ang_vel * self.obs_scales.ang_vel

        if self.cfg.obs.cfgs.last_action or self.cfg.control.control_type == "C":
            self.obs_buf.last_action = self.actions.reshape(-1, dof_num)

        if self.cfg.obs.cfgs.last_last_action or self.cfg.control.control_type == "C":
            self.obs_buf.last_last_action = self.last_actions.reshape(-1, dof_num)

        if self.cfg.obs.cfgs.projected_gravity or self.cfg.control.control_type == "C":
            self.obs_buf.projected_gravity = copy(self.projected_gravity)
        
        if self.cfg.obs.cfgs.clock_inputs or self.cfg.control.control_type == "C":
            assert self.cfg.control.control_type == "C", "To active clock_inputs, control_type should be set to \"C\" instead of \"{}\"".format(self.cfg.control.control_type)
            self.obs_buf.clock_inputs = copy(self.clock_inputs)
        
        if self.cfg.obs.cfgs.base_rpy:
            self.obs_buf.base_rpy = torch.stack(get_euler_xyz(self.base_quat), dim=1)
        
        if self.cfg.obs.cfgs.env_info and hasattr(self, "env_info"):
            self.obs_buf.env_info = self.env_info

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
        
        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs * self.num_agents, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        # env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # self._resample_commands(env_ids)
        self._step_contact_targets()

        # if self.cfg.commands.heading_command:
        #     forward = quat_apply(self.base_quat, self.forward_vec)
        #     heading = torch.atan2(forward[:, 1], forward[:, 0])
        #     self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        # if self.cfg.terrain.measure_heights:
        #     self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _step_contact_targets(self):
        if self.cfg.obs.cfgs.clock_inputs or self.cfg.control.control_type == "C":
            frequencies = self.locomotion_obs[:, 7]
            phases = self.locomotion_obs[:, 8]
            offsets = self.locomotion_obs[:, 9]
            bounds = self.locomotion_obs[:, 10]
            durations = self.locomotion_obs[:, 11]
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + bounds,
                            self.gait_indices + phases]

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

            # if self.cfg.commands.durations_warp_clock_inputs:

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        #     # von mises distribution
        #     kappa = self.cfg.rewards.kappa_gait_probs
        #     smoothing_cdf_start = torch.distributions.normal.Normal(0,
        #                                                             kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        #     smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
        #             1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
        #                                smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
        #                                        1 - smoothing_cdf_start(
        #                                    torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        #     smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
        #             1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
        #                                smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
        #                                        1 - smoothing_cdf_start(
        #                                    torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        #     smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
        #             1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
        #                                smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
        #                                        1 - smoothing_cdf_start(
        #                                    torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        #     smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
        #             1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
        #                                smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
        #                                        1 - smoothing_cdf_start(
        #                                    torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        #     self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        #     self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        #     self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        #     self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        # if self.cfg.commands.num_commands > 9:
        #     self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        if isinstance(self.cfg.control.action_scale, (tuple, list)):
            self.cfg.control.action_scale = torch.tensor(self.cfg.control.action_scale, device= self.sim_device)
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled = actions_scaled.reshape(-1, 12)
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction
        actions_scaled = actions_scaled.reshape(self.num_envs, -1)
        control_type = self.cfg.control.control_type

        if control_type == "C" or control_type == "control_net":
            
            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
                self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
            else:
                self.joint_pos_target = actions_scaled + self.default_dof_pos

            self.joint_pos_err = (self.dof_pos - self.joint_pos_target).reshape([-1, 12]) # + self.motor_offsets
            self.joint_vel = self.dof_vel.reshape([-1, 12])
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
            # torques = torques * self.motor_strengths
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return super()._compute_torques(actions)

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        ### get gym GPU state tensors ###
        super()._init_buffers()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps + 1)]

        if self.cfg.control.control_type == "actuator_net" or self.cfg.control.control_type == "C":

            actuator_network = torch.jit.load(self.cfg.control.actuator_network_path + "/unitree_go1.pt").to(self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):

                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                with torch.no_grad():
                    torques = actuator_network(xs.view(self.num_envs * self.num_agents * 12, 6))
                return torques.view(self.num_envs, self.num_actuated_dof)

            self.actuator_network = eval_actuator_network

            self.joint_pos_err_last_last = torch.zeros((self.num_envs * self.num_agents, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs * self.num_agents, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs * self.num_agents, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs * self.num_agents, 12), device=self.device)

    def _prepare_locomotion_policy(self):
        # currently only support walk_these_ways
        assert self.cfg.control.locomotion_policy_dir != None, "No locomotion policy provided."

        locomotion_obs = self._fill_command_obs()
        self.locomotion_obs = locomotion_obs.repeat([self.num_envs * self.num_agents, 1])
        self.history_locomotion_obs = torch.zeros(self.num_envs * self.num_agents, 2100, dtype=torch.float, device=self.device, requires_grad=False)
        
        body = torch.jit.load(self.cfg.control.locomotion_policy_dir + '/body_latest.jit')
        adaptation_module = torch.jit.load(self.cfg.control.locomotion_policy_dir + '/adaptation_module_latest.jit')

        def policy(obs, info={}):
            with torch.no_grad():
                
                latent = adaptation_module.forward(obs.to('cpu'))
                action = body.forward(torch.cat((obs.to('cpu'), latent), dim=-1))

            info['latent'] = latent
            return action
        
        self.locomotion_policy = policy

    def _fill_command_obs(self):
        """
        fill command in locomotion observation with default command
        """

        idx = 0
        locomotion_obs = torch.zeros(1, 70, dtype=torch.float, device=self.device, requires_grad=False)

        if not self.cfg.command.cfg.vel:
            locomotion_obs[0, 3] = self.cfg.control.default_command.lin_vel_x * self.cfg.control.obs_scales.lin_vel
            locomotion_obs[0, 4] = self.cfg.control.default_command.lin_vel_y * self.cfg.control.obs_scales.lin_vel
            locomotion_obs[0, 5] = self.cfg.control.default_command.ang_vel * self.cfg.control.obs_scales.ang_vel
        else:
            self.vel_idx = idx
            idx += 3
        
        if not self.cfg.command.cfg.body_height:
            locomotion_obs[0, 6] = self.cfg.control.default_command.body_height * self.cfg.control.obs_scales.body_height
        else:
            self.body_height_idx = idx
            idx += 1
        
        if not self.cfg.command.cfg.gait_freq:
            locomotion_obs[0, 7] = self.cfg.control.default_command.gait_freq * self.cfg.control.obs_scales.gait_freq
        else:
            self.gait_freq_idx = idx
            idx += 1

        if not self.cfg.command.cfg.gait:
            locomotion_obs[0, 8] = self.cfg.command.gaits[self.cfg.control.default_command.gait][0] * self.cfg.control.obs_scales.gait_phase
            locomotion_obs[0, 9] = self.cfg.command.gaits[self.cfg.control.default_command.gait][1] * self.cfg.control.obs_scales.gait_phase
            locomotion_obs[0, 10] = self.cfg.command.gaits[self.cfg.control.default_command.gait][2] * self.cfg.control.obs_scales.gait_phase
            locomotion_obs[0, 11] = 0.5 * self.cfg.control.obs_scales.gait_phase
        else:
            self.gait_idx = idx
            idx += 4
            
        if not self.cfg.command.cfg.footswing_height:
            locomotion_obs[0, 12] = self.cfg.control.default_command.footswing_height * self.cfg.control.obs_scales.footswing_height
        else:
            self.footswing_height_idx = idx
            idx += 1

        if not self.cfg.command.cfg.body_pose:
            locomotion_obs[0, 13] = self.cfg.control.default_command.body_pitch * self.cfg.control.obs_scales.body_pitch
            locomotion_obs[0, 14] = self.cfg.control.default_command.body_roll * self.cfg.control.obs_scales.body_roll
        else:
            self.body_pose_idx = idx
            idx += 2

        if not self.cfg.command.cfg.stance_width:
            locomotion_obs[0, 15] = self.cfg.control.default_command.stance_width * self.cfg.control.obs_scales.stance_width
        else:
            self.stance_width_idx = idx
            idx += 1

        if not self.cfg.command.cfg.stance_length:
            locomotion_obs[0, 16] = self.cfg.control.default_command.stance_length * self.cfg.control.obs_scales.stance_length
        else:
            self.stance_length_idx = idx
            idx += 1

        if not self.cfg.command.cfg.aux_reward:
            locomotion_obs[0, 17] = self.cfg.control.default_command.aux_reward * self.cfg.control.obs_scales.aux_reward
        else:
            self.aux_reward_idx = idx
            idx += 1

        return locomotion_obs

    def _init_custom_buffers__(self):

        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]

        # if self.initial_dynamics_dict is not None:
        #     for k, v in self.initial_dynamics_dict.items():
        #         if k in dynamics_params:
        #             setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs * self.num_agents, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs * self.num_agents, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs * self.num_agents, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs * self.num_agents, 4, dtype=torch.float, device=self.device, requires_grad=False)

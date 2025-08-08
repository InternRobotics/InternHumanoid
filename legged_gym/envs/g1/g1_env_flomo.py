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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import Terrain
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_yaw_quat_from_quat
from legged_gym.utils.helpers import class_to_dict
import threading
import time
from numbers import Number

import matplotlib.pyplot as plt
import pickle as pkl


def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x.unsqueeze(1), pitch_y.unsqueeze(1), yaw_z.unsqueeze(1)


class Flomo(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.fix_waist = self.cfg.control.fix_waist
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_actions = cfg.env.num_dofs
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.critic_history_length = self.cfg.env.num_critic_history
        self.num_height_dim = self.cfg.env.num_height_dim
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length
        self.critic_proprioceptive_obs_length = self.num_one_step_privileged_obs * self.critic_history_length
        self.actor_use_height = True if self.num_obs > self.actor_proprioceptive_obs_length else False
        self.num_lower_dof = self.cfg.env.num_actions
        if self.fix_waist:
            self.lower_dof_indices = torch.arange(0, self.num_lower_dof, device=self.device, dtype=torch.long)
            self.upper_dof_indices = torch.arange(
                self.num_lower_dof, self.cfg.env.num_dofs, device=self.device, dtype=torch.long
            )
            assert (
                self.cfg.env.num_actions == 12 and self.cfg.env.num_dofs == 27
            ), "Fixing waist is only supported for 12 lower body actions and 27 total dofs"
        else:
            self.lower_dof_indices = torch.cat(
                (
                    torch.arange(0, 12, device=self.device, dtype=torch.long),
                    torch.arange(13, 15, device=self.device, dtype=torch.long),
                )
            )
            self.upper_dof_indices = torch.cat(
                (
                    torch.arange(12, 13, device=self.device, dtype=torch.long),
                    torch.arange(15, self.cfg.env.num_dofs, device=self.device, dtype=torch.long),
                )
            )
            assert (
                self.cfg.env.num_actions == 14 and self.cfg.env.num_dofs == 29
            ), "Fixing waist is only supported for 14 (lower body actions + 2 dof waist actions) and 29 total dofs"
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_lower_dof, device=self.device, requires_grad=False)
        )
        return obs, privileged_obs

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        # !!! WARNING: in fix waist == False case, # actions = 14
        # !!! WARNING: [..., 12: waist yaw, 13: waist roll, 14: waist pitch, ...], where yaw is controlled by the user input, and roll & pitch are controlled by the policy. MUST be very careful with the order of actions!!!
        clip_actions = self.cfg.normalization.clip_actions

        if self.common_step_counter % self.cfg.domain_rand.upper_interval == 0:
            # (NOTE) implementation of upper-body curriculum
            self.random_upper_ratio = min(self.action_curriculum_ratio, 1.0)
            uu = torch.rand(self.num_envs, self.num_actions - self.num_lower_dof, device=self.device)
            self.random_upper_ratio = (
                -1.0
                / (20 * (1 - self.random_upper_ratio * 0.99))
                * torch.log(1 - uu + uu * np.exp(-20 * (1 - self.random_upper_ratio * 0.99)))
            )
            self.random_joint_ratio = self.random_upper_ratio * torch.rand(
                self.num_envs, self.num_actions - self.num_lower_dof
            ).to(self.device)
            if self.cfg.commands.heading_command:
                command_ratio = self.command_ratio[:, [0]].max(dim=-1).values  # consider only x vel
            else:
                command_ratio = self.command_ratio[:, [0]].max(dim=-1).values  # consider only x vel
            self.random_joint_ratio = self.random_joint_ratio * (1.0 - command_ratio.unsqueeze(1)) * 1.0
            rand_pos = torch.rand(self.num_envs, self.num_actions - self.num_lower_dof, device=self.device) - 0.5

            self.random_upper_actions = (
                (self.action_min[:, self.upper_dof_indices] * (rand_pos >= 0))
                + (self.action_max[:, self.upper_dof_indices] * (rand_pos < 0))
            ) * self.random_joint_ratio

            self.delta_upper_actions = (self.random_upper_actions - self.current_upper_actions) / (
                self.cfg.domain_rand.upper_interval
            )
        self.current_upper_actions += self.delta_upper_actions

        concat_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        concat_actions[:, self.lower_dof_indices] = actions
        concat_actions[:, self.upper_dof_indices] = self.current_upper_actions
        actions = concat_actions

        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.origin_actions[:] = self.actions[:]
        self.delayed_actions = (
            self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        )
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)

        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(
                self.cfg.domain_rand.joint_injection_range[0],
                self.cfg.domain_rand.joint_injection_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):

            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # upper-body with position control; lower-body with force control;
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            termination_ids,
            termination_priveleged_obs,
        )

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.torso_quat = self.rigid_body_states[:, self.torso_body_index, 3:7]
        relative_quat = quat_mul(
            quat_conjugate(get_yaw_quat_from_quat(self.base_quat)), self.torso_quat
        )  # handle torso rotation
        self.roll, self.pitch, self.yaw = euler_from_quaternion(relative_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.torso_ang_vel[:] = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.rigid_body_states[:, self.torso_imu_index, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt

        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        feet_height, feet_height_var, feet_height_raw = self._get_feet_heights()

        self.feet_max_height = torch.maximum(self.feet_max_height, feet_height)

        # compute joint power
        joint_power = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((self.joint_powers[:, 1:], joint_power), dim=1)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt
        self.feet_max_height *= ~self.contact_filt

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10.0, dim=1
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.gravity_termination_buf = torch.any(
            torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1
        )

        _, _, feet_height_raw = self._get_feet_heights()

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf
        self.reset_buf |= torch.any(feet_height_raw < -0.15, dim=1)

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)
        # update action curriculum for specific dofs
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_action_curriculum(env_ids)

        self.refresh_actor_rigid_shape_props(env_ids)

        # reset robot states
        self._reset_root_states(env_ids)
        self._reset_dofs(env_ids)

        # resample commands
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.joint_powers[env_ids] = 0.0
        self.random_upper_actions[env_ids] = 0.0
        self.current_upper_actions[env_ids] = 0.0
        self.delta_upper_actions[env_ids] = 0.0
        reset_roll, reset_pitch, reset_yaw = euler_from_quaternion(self.base_quat[env_ids])
        self.roll[env_ids] = reset_roll
        self.pitch[env_ids] = reset_pitch
        self.yaw[env_ids] = reset_yaw
        self.reset_buf[env_ids] = 1
        self.base_err_int[env_ids, :] = 0.0

        self.last_heights[env_ids, :] = (
            self.cfg.rewards.base_height_target - 1.0
        ) * self.obs_scales.height_measurements

        # reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kp_range[0],
                self.cfg.domain_rand.kp_range[1],
                (len(env_ids), self.num_actions),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kd_range[0],
                self.cfg.domain_rand.kd_range[1],
                (len(env_ids), self.num_actions),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(
                self.cfg.domain_rand.actuation_offset_range[0],
                self.cfg.domain_rand.actuation_offset_range[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)

        if self.add_noise:
            self.height_noise_offset[env_ids] = torch_rand_float(
                self.cfg.noise.height_measurements.offset_range[0],
                self.cfg.noise.height_measurements.offset_range[1],
                (len(env_ids), 1),
                device=self.device,
            )

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(
                self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt
            )
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if torch.isnan(rew).any():
                import ipdb

                ipdb.set_trace()
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """Computes observations"""
        imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.imu_index, 3:7], self.rigid_body_states[:, self.imu_index, 10:13]
        )
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index, 3:7], self.gravity_vec)
        current_obs = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,
                self.commands[:, 4].unsqueeze(1),
                imu_ang_vel * self.obs_scales.ang_vel,
                imu_projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions[:, self.lower_dof_indices],
            ),
            dim=-1,
        )
        current_actor_obs = torch.clone(current_obs)
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[
                0 : (10 + 2 * self.num_actions + self.num_lower_dof)
            ]
        self.obs_buf = torch.cat(
            (
                self.obs_buf[:, self.num_one_step_obs : self.actor_proprioceptive_obs_length],
                current_actor_obs[:, : self.num_one_step_obs],
            ),
            dim=-1,
        )
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        self.privileged_obs_buf = torch.cat(
            (
                self.privileged_obs_buf[:, self.num_one_step_privileged_obs : self.critic_proprioceptive_obs_length],
                current_critic_obs,
            ),
            dim=-1,
        )

        if (
            self.cfg.terrain.measure_heights
            and self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]
            and self.num_height_dim > 0
        ):
            heights = (
                torch.clip(self.root_states[:, 2].unsqueeze(1) - 1.0 - self.measured_heights, -1, 1.0)
                * self.obs_scales.height_measurements
            )
            # self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
            if self.add_noise:
                heights = self._add_height_noise(heights)
            # print("heights[0].view(11, 11)", heights[0].view(15, 15))
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
            # self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

    def compute_termination_observations(self, env_ids):
        """Computes observations"""
        imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.imu_index, 3:7], self.rigid_body_states[:, self.imu_index, 10:13]
        )
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index, 3:7], self.gravity_vec)
        current_obs = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,
                self.commands[:, 4].unsqueeze(1),
                imu_ang_vel * self.obs_scales.ang_vel,
                imu_projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions[:, self.lower_dof_indices],
            ),
            dim=-1,
        )

        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[
                0 : (10 + 2 * self.num_actions + self.num_lower_dof)
            ]
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        return torch.cat(
            (
                self.privileged_obs_buf[:, self.num_one_step_privileged_obs : self.critic_proprioceptive_obs_length],
                current_critic_obs,
            ),
            dim=-1,
        )[env_ids]

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def create_cameras(self):
        """Creates camera for each robot"""
        self.camera_params = gymapi.CameraProperties()
        self.camera_params.width = self.cfg.camera.width
        self.camera_params.height = self.cfg.camera.height
        self.camera_params.horizontal_fov = self.cfg.camera.horizontal_fov
        self.camera_params.enable_tensors = True
        self.cameras = []
        for env_handle in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_params)
            torso_handle = self.gym.get_actor_rigid_body_handle(env_handle, 0, self.torso_index)
            camera_offset = gymapi.Vec3(self.cfg.camera.offset[0], self.cfg.camera.offset[1], self.cfg.camera.offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 1, 0),
                np.deg2rad(self.cfg.camera.angle_randomization * (2 * np.random.random() - 1) + self.cfg.camera.angle),
            )
            self.gym.attach_camera_to_body(
                camera_handle,
                env_handle,
                torso_handle,
                gymapi.Transform(camera_offset, camera_rotation),
                gymapi.FOLLOW_TRANSFORM,
            )
            self.cameras.append(camera_handle)

    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        new_images = torch.stack(self.cam_tensors)
        new_images = torch.nan_to_num(new_images, neginf=0)
        new_images = torch.clamp(new_images, min=-self.cfg.camera.far, max=-self.cfg.camera.near)
        # new_images = new_images[:, 4:-4, :-2] # crop the image
        self.last_visual_obs_buf = torch.clone(self.visual_obs_buf)
        self.visual_obs_buf = new_images.view(self.num_envs, -1)

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(
                    friction_range[0], friction_range[1], (self.num_envs, 1), device=self.device
                )

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(
                    restitution_range[0], restitution_range[1], (self.num_envs, 1), device=self.device
                )

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(
                self.cfg.domain_rand.friction_range[0],
                self.cfg.domain_rand.friction_range[1],
                (len(env_ids), 1),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(
                self.cfg.domain_rand.restitution_range[0],
                self.cfg.domain_rand.restitution_range[1],
                (len(env_ids), 1),
                device=self.device,
            )

        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.hard_dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_body_index].mass = (
                self.default_rigid_body_mass[self.torso_body_index] + self.payload[env_id, 0]
            )
            props[self.left_hand_index].mass = (
                self.default_rigid_body_mass[self.left_hand_index] + self.hand_payload[env_id, 0]
            )
            props[self.right_hand_index].mass = (
                self.default_rigid_body_mass[self.right_hand_index] + self.hand_payload[env_id, 1]
            )

            if self.backpack_payload is not None:
                props[self.backpack_index].mass = (
                    self.default_rigid_body_mass[self.backpack_index] + self.backpack_payload[env_id, 0]
                )

        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(
                self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2]
            )
        if self.cfg.domain_rand.randomize_body_displacement:
            props[self.torso_body_index].com = self.default_body_com + gymapi.Vec3(
                self.body_displacement[env_id, 0], self.body_displacement[env_id, 1], self.body_displacement[env_id, 2]
            )

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)

        if self.cfg.terrain.measure_heights and self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def discretize_speed(self, speeds: torch.Tensor, K: float) -> torch.Tensor:
        half = K * 0.5
        return torch.floor((speeds + half) / K) * K

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        set_x = torch.rand(len(env_ids), 1).to(self.device)
        is_height = set_x < 1 / 3
        is_vel = set_x >= 1 / 3
        self.commands[env_ids, 0] = (
            torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1],
                (len(env_ids), 1),
                device=self.device,
            )
            * is_vel
        ).squeeze(1)
        self.command_ratio[env_ids, 0] = (self.commands[env_ids, 0] / self.command_ranges["lin_vel_x"][1]) * (
            self.commands[env_ids, 0] > 0
        ) + (self.commands[env_ids, 0] / self.command_ranges["lin_vel_x"][0]) * (self.commands[env_ids, 0] < 0)
        self.commands[env_ids, 1] = (
            torch_rand_float(
                self.command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][1],
                (len(env_ids), 1),
                device=self.device,
            )
            * (is_vel * (self.commands[env_ids, 0] < 1.2).unsqueeze(1))  # ignore y vel when x vel is large
        ).squeeze(1)
        self.command_ratio[env_ids, 1] = (self.commands[env_ids, 1] / self.command_ranges["lin_vel_y"][1]) * (
            self.commands[env_ids, 1] > 0
        ) + (self.commands[env_ids, 1] / self.command_ranges["lin_vel_y"][0]) * (self.commands[env_ids, 1] < 0)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = (
                torch_rand_float(
                    self.command_ranges["heading"][0],
                    self.command_ranges["heading"][1],
                    (len(env_ids), 1),
                    device=self.device,
                )
                * is_vel
            ).squeeze(1)
            self.command_ratio[env_ids, 3] = (self.commands[env_ids, 3] / self.command_ranges["heading"][1]) * (
                self.commands[env_ids, 3] > 0
            ) + (self.commands[env_ids, 3] / self.command_ranges["heading"][0]) * (self.commands[env_ids, 3] < 0)
            self.commands[env_ids, 4] = (
                torch_rand_float(
                    self.command_ranges["height"][0],
                    self.command_ranges["height"][1],
                    (len(env_ids), 1),
                    device=self.device,
                )
                * is_height
            ).squeeze(
                1
            ) + self.cfg.rewards.base_height_target  # height
            self.command_ratio[env_ids, 4] = (self.commands[env_ids, 4] / self.command_ranges["height"][1]) * (
                self.commands[env_ids, 4] > 0
            ) + (self.commands[env_ids, 4] / self.command_ranges["height"][0]) * (self.commands[env_ids, 4] < 0)
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)
        else:
            self.commands[env_ids, 2] = (
                torch_rand_float(
                    self.command_ranges["ang_vel_yaw"][0],
                    self.command_ranges["ang_vel_yaw"][1],
                    (len(env_ids), 1),
                    device=self.device,
                )
                * is_vel
            ).squeeze(1)
            self.command_ratio[env_ids, 2] = (self.commands[env_ids, 2] / self.command_ranges["ang_vel_yaw"][1]) * (
                self.commands[env_ids, 2] > 0
            ) + (self.commands[env_ids, 2] / self.command_ranges["ang_vel_yaw"][0]) * (self.commands[env_ids, 2] < 0)
            self.commands[env_ids, 4] = (
                torch_rand_float(
                    self.command_ranges["height"][0],
                    self.command_ranges["height"][1],
                    (len(env_ids), 1),
                    device=self.device,
                )
                * is_height
            ).squeeze(
                1
            ) + self.cfg.rewards.base_height_target  # height
            self.command_ratio[env_ids, 4] = (self.commands[env_ids, 4] / self.command_ranges["height"][1]) * (
                self.commands[env_ids, 4] > 0
            ) + (self.commands[env_ids, 4] / self.command_ranges["height"][0]) * (self.commands[env_ids, 4] < 0)

            self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.1).unsqueeze(1)

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_pos + actions_scaled
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos)
                - self.d_gains * self.Kd_factors * self.dof_vel
            )
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type == "M":
            torques = (
                self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos)
                - self.d_gains * self.Kd_factors * self.dof_vel
            )

            torques = torques + self.actuation_offset + self.joint_injection
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            control = torch.zeros_like(torques)
            control[..., self.lower_dof_indices] = torques[..., self.lower_dof_indices]
            control[..., self.upper_dof_indices] = self.joint_pos_target[..., self.upper_dof_indices]
            return control

        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(
                self.cfg.domain_rand.initial_joint_pos_scale[0],
                self.cfg.domain_rand.initial_joint_pos_scale[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
            init_dos_pos += torch_rand_float(
                self.cfg.domain_rand.initial_joint_pos_offset[0],
                self.cfg.domain_rand.initial_joint_pos_offset[1],
                (len(env_ids), self.num_dof),
                device=self.device,
            )
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch.ones((len(env_ids), self.num_dof), device=self.device)

        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_pos.contiguous()),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
            self.root_states[env_ids, 2:3] += self._get_init_heights(env_ids)
            self.root_states[env_ids, 2:3] += torch_rand_float(0.03, 0.05, (len(env_ids), 1), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y

        if hasattr(self.cfg.domain_rand, "max_push_vel_z") and self.cfg.domain_rand.max_push_vel_z > 0.0:
            max_vel_z = self.cfg.domain_rand.max_push_vel_z
            self.root_states[:, [9]] += torch_rand_float(  # note we use `add` instead of `assign`
                -max_vel_z, max_vel_z, (self.num_envs, 1), device=self.device
            )  # ang vel z
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 75% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_x_vel"]
        ) and (
            torch.mean(self.episode_sums["tracking_y_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_y_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.0
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.2, 0.0, self.cfg.commands.max_curriculum
            )

    def update_action_curriculum(self, env_ids):
        """Implements a curriculum of increasing action range

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if (
            torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_x_vel"]
        ):
            self.action_curriculum_ratio += 0.05
            self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)

        if self.cfg.terrain.hard_terrain:
            move_up_by_one = (distance >= self.terrain.env_length - 0.5) * ~self.border_buf[env_ids]
        else:
            move_up_by_one = (distance >= self.terrain.env_length - 0.5) * (
                self.invalid_foothold_times[env_ids] / self.contact_times[env_ids] < 0.1
            )
        self.move_up_counter[env_ids] = (self.move_up_counter[env_ids] + move_up_by_one) * move_up_by_one
        move_up = self.move_up_counter[env_ids] >= 3
        self.move_up_counter[env_ids] *= ~move_up
        move_down = 0

        self.terrain_levels[env_ids] = self.terrain_levels[env_ids] + 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)

        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(10 + 2 * self.num_actions + self.num_lower_dof, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:4] = 0.0  # commands
        noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity * noise_level
        noise_vec[10 : (10 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(10 + self.num_actions) : (10 + 2 * self.num_actions)] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[(10 + 2 * self.num_actions) : (10 + 2 * self.num_actions + self.num_lower_dof)] = (
            0.0  # previous actions
        )
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.torso_quat = self.rigid_body_states[:, self.torso_body_index, 3:7]
        relative_quat = quat_mul(quat_conjugate(get_yaw_quat_from_quat(self.base_quat)), self.torso_quat)
        self.roll, self.pitch, self.yaw = euler_from_quaternion(relative_quat)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.origin_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.command_ratio = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.feet_max_height = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.first_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.torso_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.rigid_body_states[:, self.torso_imu_index, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.border_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.move_up_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.base_err_int = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )  # store integral of base position error

        self.last_heights = torch.zeros(
            self.num_envs,
            len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_heights[:, :] = (self.cfg.rewards.base_height_target - 1.0) * self.obs_scales.height_measurements

        if self.cfg.terrain.measure_heights and self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(
                f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}"
            )
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.action_max = (
            self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos
        ) / self.cfg.control.action_scale
        self.action_min = (
            self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos
        ) / self.cfg.control.action_scale
        self.action_curriculum_ratio = self.cfg.domain_rand.init_upper_ratio
        self.target_heights = torch.ones((self.num_envs), device=self.device) * self.cfg.rewards.base_height_target
        print(f"Action min: {self.action_min}")
        print(f"Action max: {self.action_max}")

        self.random_upper_actions = torch.zeros(
            (self.num_envs, self.num_actions - self.num_lower_dof), device=self.device
        )
        self.current_upper_actions = torch.zeros(
            (self.num_envs, self.num_actions - self.num_lower_dof), device=self.device
        )
        self.delta_upper_actions = torch.zeros((self.num_envs, 1), device=self.device)
        # randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.Kd_factors = torch.ones(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.joint_injection = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actuation_offset = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.height_noise_offset = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )

        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(
                self.cfg.domain_rand.kp_range[0],
                self.cfg.domain_rand.kp_range[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(
                self.cfg.domain_rand.kd_range[0],
                self.cfg.domain_rand.kd_range[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(
                self.cfg.domain_rand.joint_injection_range[0],
                self.cfg.domain_rand.joint_injection_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(
                self.cfg.domain_rand.actuation_offset_range[0],
                self.cfg.domain_rand.actuation_offset_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            ) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(
                self.cfg.domain_rand.payload_mass_range[0],
                self.cfg.domain_rand.payload_mass_range[1],
                (self.num_envs, 1),
                device=self.device,
            )
            self.hand_payload = torch_rand_float(
                self.cfg.domain_rand.hand_payload_mass_range[0],
                self.cfg.domain_rand.hand_payload_mass_range[1],
                (self.num_envs, 2),
                device=self.device,
            )

            if self.backpack_payload is not None:
                self.backpack_payload = torch_rand_float(
                    self.cfg.domain_rand.backpack_payload_mass_range[0],
                    self.cfg.domain_rand.backpack_payload_mass_range[1],
                    (self.num_envs, 1),
                    device=self.device,
                )

        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(
                self.cfg.domain_rand.com_displacement_range[0],
                self.cfg.domain_rand.com_displacement_range[1],
                (self.num_envs, 3),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(
                self.cfg.domain_rand.body_displacement_range[0],
                self.cfg.domain_rand.body_displacement_range[1],
                (self.num_envs, 3),
                device=self.device,
            )

        # store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
        )

        # joint powers
        self.joint_powers = torch.zeros(
            self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            if callable(self.reward_scales[key]) or not isinstance(self.reward_scales[key], Number):
                continue
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if callable(self.reward_scales[name]) or not isinstance(self.reward_scales[name], Number):
                continue
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def _create_trimesh(self):
        """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg."""
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size  # TODO: what is border size?
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim, self.terrain.vertices.flatten(order="C"), self.terrain.triangles.flatten(order="C"), tm_params
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]
            ).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()

        if not self.cfg.terrain.hard_terrain:
            points[:, :, 0] += self.terrain.st_distance  # transform to the source terrain observation

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_init_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(len(env_ids), self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        y = torch.tensor([-0.25, -0.15, 0.0, 0.15, 0.25], device=self.device, requires_grad=False)
        x = torch.tensor([-0.25, -0.15, 0.0, 0.15, 0.25], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        num_height_points = grid_x.numel()
        sample_points = torch.zeros(self.num_envs, num_height_points, 3, device=self.device, requires_grad=False)
        sample_points[:, :, 0] = grid_x.flatten()
        sample_points[:, :, 1] = grid_y.flatten()

        if env_ids is not None:
            points = sample_points[env_ids] + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = sample_points + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()

        if not self.cfg.terrain.hard_terrain:
            points[:, :, 0] += self.terrain.st_distance  # transform to the source terrain observation

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return torch.max(heights.view(len(env_ids), -1) * self.terrain.cfg.vertical_scale, dim=-1, keepdim=True).values

    def _add_height_noise(self, heights):
        # extend noise
        heights = heights.reshape(
            self.num_envs, len(self.cfg.terrain.measured_points_x), len(self.cfg.terrain.measured_points_y)
        )
        mean_height = heights[heights < 0].mean()
        valid_index = (heights < 0).float()
        kernel = (
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        valid_neighbor = torch.nn.functional.conv2d(valid_index.unsqueeze(1), kernel, padding=1).squeeze(1)
        random_tensor = torch.rand_like(heights)
        valid_neighbor = (
            (valid_neighbor > 0) & (heights >= 0) & (random_tensor < self.cfg.noise.height_measurements.extend_prob)
        )
        extend_noise = torch.normal(
            mean=mean_height + 0.1, std=0.03, size=heights[valid_neighbor].shape, device=self.device
        )
        heights[valid_neighbor] = extend_noise
        heights = heights.reshape(self.num_envs, -1)

        # vertical noise
        noise = torch_rand_float(
            -self.cfg.noise.height_measurements.vertical_scale,
            self.cfg.noise.height_measurements.vertical_scale,
            heights.shape,
            device=self.device,
        )
        offset = self.height_noise_offset
        heights = heights + noise * self.cfg.noise.noise_level + offset

        # repeat
        if torch.rand(1).item() < self.cfg.noise.height_measurements.map_repeat_prob:
            heights = self.last_heights
        self.last_heights = heights

        heights = torch.where((heights > 0.5), 1.0, heights)
        return heights

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        if self.cfg.terrain.mesh_type != "trimesh" or not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = (
                quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            )
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.use_physx_armature = self.cfg.asset.use_physx_armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dof = len(self.dof_names)

        assert self.fix_waist or ("waist_roll_joint" in self.dof_names and "waist_pitch_joint" in self.dof_names), (
            "The robot asset must have 'waist_roll_joint' and 'waist_pitch_joint' DOFs, "
            "or the cfg.control.fix_waist must be set to True."
        )

        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        left_foot_names = [s for s in self.body_names if self.cfg.asset.left_foot_name in s]
        right_foot_names = [s for s in self.body_names if self.cfg.asset.right_foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.default_rigid_body_mass = torch.zeros(
            self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False
        )

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []

        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.hand_payload = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        if "backpack_link" in self.body_names:
            self.backpack_payload = torch.zeros(
                self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False
            )
        else:
            self.backpack_payload = None

        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(
                self.cfg.domain_rand.payload_mass_range[0],
                self.cfg.domain_rand.payload_mass_range[1],
                (self.num_envs, 1),
                device=self.device,
            )
            self.hand_payload = torch_rand_float(
                self.cfg.domain_rand.hand_payload_mass_range[0],
                self.cfg.domain_rand.hand_payload_mass_range[1],
                (self.num_envs, 2),
                device=self.device,
            )

            if self.backpack_payload is not None:
                self.backpack_payload = torch_rand_float(
                    self.cfg.domain_rand.backpack_payload_mass_range[0],
                    self.cfg.domain_rand.backpack_payload_mass_range[1],
                    (self.num_envs, 1),
                    device=self.device,
                )

        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(
                self.cfg.domain_rand.com_displacement_range[0],
                self.cfg.domain_rand.com_displacement_range[1],
                (self.num_envs, 3),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(
                self.cfg.domain_rand.body_displacement_range[0],
                self.cfg.domain_rand.body_displacement_range[1],
                (self.num_envs, 3),
                device=self.device,
            )

        self.torso_body_index = self.body_names.index("torso_link")
        self.left_hand_index = self.body_names.index("left_hand_palm_link")
        self.right_hand_index = self.body_names.index("right_hand_palm_link")

        if "backpack_link" in self.body_names:
            self.backpack_index = self.body_names.index("backpack_link")
        else:
            self.backpack_index = None

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0
            )
            dof_props = self._process_dof_props(dof_props_asset, i)

            if self.fix_waist:
                dof_props["driveMode"][12:].fill(gymapi.DOF_MODE_POS)
                dof_props["stiffness"][12:] = [
                    200.0,
                    40.0,
                    40.0,
                    25.0,
                    25.0,
                    10.0,
                    10.0,
                    10.0,
                    40.0,
                    40.0,
                    25.0,
                    25.0,
                    10.0,
                    10.0,
                    10.0,
                ]
                dof_props["damping"][12:] = [
                    5.0000,
                    1.0000,
                    1.0000,
                    1.0000,
                    1.0000,
                    0.2500,
                    0.2500,
                    0.2500,
                    1.0000,
                    1.0000,
                    1.0000,
                    1.0000,
                    0.2500,
                    0.2500,
                    0.2500,
                ]
            else:
                # handle waist yaw joint (user input)
                # waist roll and pitch joints are policy controlled
                dof_props["driveMode"][12] = gymapi.DOF_MODE_POS
                dof_props["stiffness"][12] = 200.0
                dof_props["damping"][12] = 5.0
                # handle other user input joints
                dof_props["driveMode"][15:] = gymapi.DOF_MODE_POS
                dof_props["stiffness"][15:] = [
                    40.0,
                    40.0,
                    25.0,
                    25.0,
                    10.0,
                    10.0,
                    10.0,
                    40.0,
                    40.0,
                    25.0,
                    25.0,
                    10.0,
                    10.0,
                    10.0,
                ]
                dof_props["damping"][15:] = [
                    1.0000,
                    1.0000,
                    1.0000,
                    1.0000,
                    0.2500,
                    0.2500,
                    0.2500,
                    1.0000,
                    1.0000,
                    1.0000,
                    1.0000,
                    0.2500,
                    0.2500,
                    0.2500,
                ]

            armature = []
            for dof_idx in range(self.num_dof):
                name = self.dof_names[dof_idx]
                armature.append(self.cfg.init_state.joint_armature[name])
            dof_props["armature"] = armature

            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                self.default_body_com = copy.deepcopy(body_props[self.torso_body_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass

            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], knee_names[i]
            )

        self.left_foot_indices = torch.zeros(
            len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], left_foot_names[i]
            )

        self.right_foot_indices = torch.zeros(
            len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], right_foot_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        self.left_leg_joint_indices = torch.zeros(
            len(self.cfg.asset.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_leg_joints[i])

        self.right_leg_joint_indices = torch.zeros(
            len(self.cfg.asset.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_leg_joints[i])

        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))

        self.left_hip_joint_indices = torch.zeros(
            len(self.cfg.asset.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_joints[i])

        self.right_hip_joint_indices = torch.zeros(
            len(self.cfg.asset.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_joints[i])

        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))

        self.hip_pitch_joint_indices = torch.zeros(
            len(self.cfg.asset.hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.hip_pitch_joints)):
            self.hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.hip_pitch_joints[i])

        self.hip_roll_yaw_joint_indices = torch.zeros(
            len(self.cfg.asset.hip_roll_yaw_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.hip_roll_yaw_joints)):
            self.hip_roll_yaw_joint_indices[i] = self.dof_names.index(self.cfg.asset.hip_roll_yaw_joints[i])

        if self.cfg.rewards.scales.deviation_arm_joint != 0:
            self.left_arm_joint_indices = torch.zeros(
                len(self.cfg.asset.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False
            )
            for i in range(len(self.cfg.asset.left_arm_joints)):
                self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_arm_joints[i])

            self.right_arm_joint_indices = torch.zeros(
                len(self.cfg.asset.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False
            )
            for i in range(len(self.cfg.asset.right_arm_joints)):
                self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_arm_joints[i])

            self.arm_joint_indices = torch.cat((self.left_arm_joint_indices, self.right_arm_joint_indices))

        if self.cfg.rewards.scales.deviation_waist_joint != 0:
            self.waist_joint_indices = torch.zeros(
                len(self.cfg.asset.waist_joints), dtype=torch.long, device=self.device, requires_grad=False
            )
            for i in range(len(self.cfg.asset.waist_joints)):
                self.waist_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_joints[i])

        self.ankle_joint_indices = torch.zeros(
            len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])

        hip_yaw_joint_names = ["left_hip_yaw_joint", "right_hip_yaw_joint"]
        self.hip_yaw_joint_indices = torch.zeros(
            len(hip_yaw_joint_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(hip_yaw_joint_names)):
            self.hip_yaw_joint_indices[i] = self.dof_names.index(hip_yaw_joint_names[i])

        hip_roll_joint_names = ["left_hip_roll_joint", "right_hip_roll_joint"]
        self.hip_roll_joint_indices = torch.zeros(
            len(hip_roll_joint_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(hip_roll_joint_names)):
            self.hip_roll_joint_indices[i] = self.dof_names.index(hip_roll_joint_names[i])

        self.knee_joint_indices = torch.zeros(
            len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])

        self.upper_body_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.cfg.asset.upper_body_link
        )
        self.imu_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.cfg.asset.imu_link
        )
        self.torso_imu_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.cfg.asset.imu_torso
        )

        self.torso_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "torso_link")

        self.left_wrist_handle = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "left_hand_palm_link"
        )
        self.right_wrist_handle = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "right_hand_palm_link"
        )

        if not self.fix_waist:
            self.waist_active_joint_indices = torch.zeros(2, dtype=torch.long, device=self.device, requires_grad=False)
            waist_joints = ["waist_roll_joint", "waist_pitch_joint"]
            for i in range(len(waist_joints)):
                self.waist_active_joint_indices[i] = self.dof_names.index(waist_joints[i])

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.max_terrain_level
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.upper_interval = np.ceil(self.cfg.domain_rand.upper_interval_s / self.dt)

    def _get_feet_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        if self.cfg.terrain.mesh_type == "plane":
            left_foot_height = torch.mean(left_foot_pos[:, :, 2], dim=-1, keepdim=True)
            left_foot_height_var = torch.var(left_foot_pos[:, :, 2], dim=-1, keepdim=True)
            right_foot_height = torch.mean(right_foot_pos[:, :, 2], dim=-1, keepdim=True)
            right_foot_height_var = torch.var(right_foot_pos[:, :, 2], dim=-1, keepdim=True)
            return (
                torch.cat((left_foot_height, right_foot_height), dim=-1),
                torch.cat((left_foot_height_var, right_foot_height_var), dim=-1),
                torch.cat((left_foot_height, right_foot_height), dim=-1),
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            left_points = left_foot_pos[env_ids].clone()
            right_points = right_foot_pos[env_ids].clone()
        else:
            left_points = left_foot_pos.clone()
            right_points = right_foot_pos.clone()

        left_points += self.terrain.cfg.border_size
        right_points += self.terrain.cfg.border_size
        left_points = (left_points / self.terrain.cfg.horizontal_scale).long()
        right_points = (right_points / self.terrain.cfg.horizontal_scale).long()
        left_px = left_points[:, :, 0].view(-1)
        right_px = right_points[:, :, 0].view(-1)
        left_py = left_points[:, :, 1].view(-1)
        right_py = right_points[:, :, 1].view(-1)
        left_px = torch.clip(left_px, 0, self.height_samples.shape[0] - 2)
        right_px = torch.clip(right_px, 0, self.height_samples.shape[0] - 2)
        left_py = torch.clip(left_py, 0, self.height_samples.shape[1] - 2)
        right_py = torch.clip(right_py, 0, self.height_samples.shape[1] - 2)

        left_heights1 = self.height_samples[left_px, left_py]
        left_heights2 = self.height_samples[left_px + 1, left_py]
        left_heights3 = self.height_samples[left_px, left_py + 1]
        left_heights = torch.min(left_heights1, left_heights2)
        left_heights = torch.min(left_heights, left_heights3)
        left_heights = left_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        left_foot_heights = left_foot_pos[:, :, 2] - left_heights

        right_heights1 = self.height_samples[right_px, right_py]
        right_heights2 = self.height_samples[right_px + 1, right_py]
        right_heights3 = self.height_samples[right_px, right_py + 1]
        right_heights = torch.min(right_heights1, right_heights2)
        right_heights = torch.min(right_heights, right_heights3)
        right_heights = right_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        right_foot_heights = right_foot_pos[:, :, 2] - right_heights

        feet_heights = torch.cat(
            (torch.mean(left_foot_heights, dim=-1, keepdim=True), torch.mean(right_foot_heights, dim=-1, keepdim=True)),
            dim=-1,
        )
        feet_heights_var = torch.cat(
            (torch.var(left_foot_heights, dim=-1, keepdim=True), torch.var(right_foot_heights, dim=-1, keepdim=True)),
            dim=-1,
        )

        return torch.clip(feet_heights, min=0.0), feet_heights_var, feet_heights

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1:2] - self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_regu_y_vel(self):
        return torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square((self.commands[:, 2] - self.torso_ang_vel[:, 2]) / torch.pi)
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel_int(self):
        err = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        self.base_err_int[:, 2] = (1 - self.cfg.rewards.beta_int) * self.base_err_int[
            :, 2
        ] + self.cfg.rewards.beta_int * err * self.dt
        return torch.exp(-self.base_err_int[:, 2] / self.cfg.rewards.tracking_sigma_int)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) * (torch.norm(self.commands[:, :3], dim=1) > 0.1)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_tracking_base_height(self):
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        height_error = torch.abs(base_height - self.commands[:, 4] + self.cfg.asset.ankle_sole_distance)
        return torch.exp(-height_error * 10)

    def _reward_step_on_stair(self):
        left_points = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_points = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        left_points += self.terrain.cfg.border_size
        right_points += self.terrain.cfg.border_size
        left_points = (left_points / self.terrain.cfg.horizontal_scale).long()
        right_points = (right_points / self.terrain.cfg.horizontal_scale).long()
        left_px = left_points[:, :, 0].view(-1)
        right_px = right_points[:, :, 0].view(-1)
        left_py = left_points[:, :, 1].view(-1)
        right_py = right_points[:, :, 1].view(-1)
        left_px = torch.clip(left_px, 0, self.height_samples.shape[0] - 2)
        right_px = torch.clip(right_px, 0, self.height_samples.shape[0] - 2)
        left_py = torch.clip(left_py, 0, self.height_samples.shape[1] - 2)
        right_py = torch.clip(right_py, 0, self.height_samples.shape[1] - 2)

        left_heights = self.height_samples[left_px, left_py].view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        left_var = torch.var(left_heights, dim=-1)
        right_heights = (
            self.height_samples[right_px, right_py].view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        )
        right_var = torch.var(right_heights, dim=-1)
        feet_heights_var = left_var + right_var
        feet_contact = torch.sum(self.contact_forces[:, self.feet_indices, 2] < 0.5, dim=-1)

        return torch.exp(-100 * feet_contact * feet_heights_var) - 1

    def _reward_deviation_all_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)

    def _reward_deviation_arm_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.arm_joint_indices], dim=-1)

    def _reward_deviation_leg_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.leg_joint_indices], dim=-1)

    def _reward_deviation_hip_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_joint_indices], dim=-1) * (
            torch.norm(self.commands[:, :3], dim=1) > 0.1
        ) + 2 * torch.sum(
            torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_roll_yaw_joint_indices], dim=-1
        ) * (
            torch.norm(self.commands[:, :3], dim=1) <= 0.1
        )

    def _reward_deviation_waist_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.waist_joint_indices], dim=-1)

    def _reward_deviation_ankle_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.ankle_joint_indices], dim=-1) * (
            torch.norm(self.commands[:, :3], dim=1) > 0.1
        )

    def _reward_deviation_knee_joint(self):
        height_error = (
            self.root_states[:, 2] - self.commands[:, 4]
        )  # / (torch.abs(self.root_states[:, 2] - self.commands[:, 4]) + 1e-6)
        knee_action_min = (
            self.default_dof_pos[:, self.knee_joint_indices]
            + self.cfg.control.action_scale * self.action_min[:, self.knee_joint_indices]
        )
        knee_action_max = (
            self.default_dof_pos[:, self.knee_joint_indices]
            + self.cfg.control.action_scale * self.action_max[:, self.knee_joint_indices]
        )
        joint_deviation = (self.dof_pos[:, self.knee_joint_indices] - knee_action_min) / (
            knee_action_max - knee_action_min
        )  # always positive
        # height_error < 0, dof pos lower; else, dof pos larger
        return torch.sum(torch.abs((joint_deviation - 0.5) * height_error.unsqueeze(-1)), dim=-1)

    def _reward_feet_distance(self):
        feet_distance = torch.norm(self.feet_pos[:, 0, :2] - self.feet_pos[:, 1, :2], dim=1, p=2)
        return torch.clip(feet_distance - self.cfg.rewards.least_feet_distance, max=0) + torch.clip(
            -feet_distance + self.cfg.rewards.most_feet_distance, max=0
        )

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0])[:, : self.num_actions].clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1])[:, : self.num_actions].clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * self.first_contacts, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1  # no reward for zero command
        return rew_airTime

    def _reward_feet_clearance(self):
        cur_feetvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        feetvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            feetvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_feetvel_translated[:, i, :])
        feet_height, feet_height_var, _ = self._get_feet_heights()
        height_error = torch.square(feet_height - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        feet_leteral_vel = torch.sqrt(torch.sum(torch.square(feetvel_in_body_frame[:, :, :2]), dim=2)).view(
            self.num_envs, -1
        )
        return torch.sum(height_error * feet_leteral_vel, dim=1) * torch.clip(
            (torch.norm(self.commands[:, :3], dim=1) - 0.1) / 0.2, min=0.0, max=1.0
        )

    def _reward_feet_distance_lateral(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        return torch.clamp(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0) + torch.clamp(
            -foot_leteral_dis + self.cfg.rewards.most_feet_distance_lateral, max=0
        )

    def _reward_knee_distance_lateral(self):
        cur_knee_pos_translated = self.rigid_body_states[:, self.knee_indices, :3].clone() - self.root_states[
            :, 0:3
        ].unsqueeze(1)
        knee_pos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            knee_pos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_translated[:, i, :])
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - knee_pos_in_body_frame[:, 2, 1]) + torch.abs(
            knee_pos_in_body_frame[:, 1, 1] - knee_pos_in_body_frame[:, 3, 1]
        )
        return torch.clamp(knee_lateral_dis - self.cfg.rewards.least_knee_distance_lateral * 2, max=0) + torch.clamp(
            -knee_lateral_dis + self.cfg.rewards.most_knee_distance_lateral * 2, max=0
        )

    def _reward_feet_ground_parallel(self):
        feet_heights, feet_heights_var, _ = self._get_feet_heights()
        continue_contact = (self.feet_air_time >= 3 * self.dt) * self.contact_filt
        return torch.sum(feet_heights_var * continue_contact, dim=1)

    def _reward_feet_parallel(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices[0:3], :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices[0:3], :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=2)
        feet_distances_var = torch.var(feet_distances, dim=1)
        return feet_distances_var

    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(
            torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1
        )

    def _reward_joint_power(self):
        # Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(
            torch.sum(torch.square(self.commands[:, 0:2]), dim=-1) + 0.2 * torch.square(self.commands[:, 2]), min=0.1
        )

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 3 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square((self.torques / self.p_gains.unsqueeze(0))[:, self.lower_dof_indices]), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, self.lower_dof_indices]), dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit)[
                :, self.lower_dof_indices
            ].clip(min=0.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit)[
                :, self.lower_dof_indices
            ].clip(min=0.0),
            dim=1,
        )

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.5
        single_contact = torch.sum(1.0 * contacts, dim=1) == 1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(
            torch.max(rew_no_fly, 1.0 * (torch.norm(self.commands[:, :3], dim=1) < 0.1)),
            (self.base_lin_vel[:, 0] > 1.2) * 1.0,
        )  # full reward for zero command
        return rew_no_fly

    def _reward_joint_tracking_error(self):
        return torch.sum(
            torch.square(self.joint_pos_target[:, self.lower_dof_indices] - self.dof_pos[:, self.lower_dof_indices]),
            dim=-1,
        )

    def _reward_feet_slip(self):
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        return torch.sum(torch.norm(self.feet_vel[:, :, :2], dim=2) * contact, dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_contact_momentum(self):
        # encourage soft contacts
        feet_contact_momentum_z = torch.clip(self.feet_vel[:, :, 2], max=0) * torch.clip(
            self.contact_forces[:, self.feet_indices, 2] - 50, min=0
        )
        return torch.sum(feet_contact_momentum_z, dim=1)

    def _reward_action_vanish(self):
        upper_error = torch.clip(
            self.origin_actions[:, self.lower_dof_indices] - self.action_max[:, self.lower_dof_indices], min=0
        )
        lower_error = torch.clip(
            self.action_min[:, self.lower_dof_indices] - self.origin_actions[:, self.lower_dof_indices], min=0
        )
        return torch.sum(upper_error + lower_error, dim=-1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] < 0.1, dim=-1)  # no contact
        error_sim = contacts
        return error_sim * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)

        c_update = (lin_mismatch + ang_mismatch) / 2.0

        return c_update

    def _reward_penalize_pelvis_ang_vel(self):
        ang_vel = self.torso_ang_vel.clone()
        ang_vel[:, 2] = ang_vel[:, 2] - self.commands[:, 2]
        return torch.sum(torch.square(ang_vel), dim=-1) * (torch.norm(self.commands[:, :3], dim=1) > 0.1)

    def _reward_waist_action(self):
        if self.fix_waist:
            return torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # Penalize waist action
        waist_action = self.dof_pos[:, self.waist_active_joint_indices]
        return torch.exp(-30 * waist_action.abs()[:, 0]) + torch.exp(-30 * waist_action.abs()[:, 1])

    def _reward_hip_roll_yaw_action(self):

        hip_roll_yaw_action = self.dof_pos[:, self.hip_roll_yaw_joint_indices]
        penalize = torch.sum(
            torch.square(hip_roll_yaw_action - self.default_dof_pos[:, self.hip_roll_yaw_joint_indices]),
            dim=-1,
            keepdim=True,
        )
        factor = (torch.clip(self.commands[:, 0] - 1.0, min=0.0, max=1.0)).unsqueeze(-1)
        return (penalize * factor).squeeze(-1)

    def _reward_imu_stand_still(self):
        """
        Penalizes the robot's IMU readings when the robot is expected to be stationary.
        This function checks if the robot is not moving and applies a penalty based on the IMU readings.
        """
        imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.imu_index, 3:7], self.rigid_body_states[:, self.imu_index, 10:13]
        )
        torso_imu_ang_vel = quat_rotate_inverse(
            self.rigid_body_states[:, self.torso_imu_index, 3:7], self.rigid_body_states[:, self.torso_imu_index, 10:13]
        )
        still_condition = torch.norm(self.commands[:, :3], dim=1) < 0.1
        imu_ang_vel = torch.mean(torch.square(imu_ang_vel), dim=1)
        torso_imu_ang_vel = torch.mean(torch.square(torso_imu_ang_vel), dim=1)

        return still_condition * (
            imu_ang_vel + torso_imu_ang_vel
        )  # penalize both IMU and torso IMU angular velocities when stationary

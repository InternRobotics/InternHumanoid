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
from isaacgym import gymtorch, gymapi, gymutil

import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

from legged_gym.utils.geometry import WireframeSphereGeometry
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.motion_lib import (
    MotionLib, 
    MotionLoader, 
    load_imitation_dataset, 
    filter_and_fix_motion,
    )
from legged_gym.utils.math import (
    get_axis_params,
    quat_rotate,
    quat_rotate_inverse,
    quat_mul,
    quat_mul_inverse,
    quat_mul_yaw,
    quat_mul_yaw_inverse,
    quat_rotate_yaw,
    quat_rotate_yaw_inverse,
    quat_to_rot6d,
    quat_to_euler_xyz,
    quat_to_angle_axis,
    euler_xyz_to_quat,
    torch_rand_float,
    torch_rand_like_float,
    )


class G1Imitation:
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file, calls create_sim()
            (which creates, simulation, terrain and environments),
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
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless
        self.height_samples = None
        self.test_mode = False
        self.motion_mode = False
        self.debug_viz = True
        self._parse_cfg()
        self.gym = gymapi.acquire_gym()

        # env device is GPU only if sim is on GPU and use_gpu_pipeline = True, 
        # otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless:self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.extras = {}
        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.enable_viewer_sync = True
        self.viewer = None
        
        self._init_buffers()
        self._prepare_rewards()
        
        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "free_cam")
            for i in range(9):
                self.gym.subscribe_viewer_keyboard_event(
                    self.viewer, getattr(gymapi, "KEY_" + str(i)), "lookat" + str(i))
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "prev_env_id")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "next_env_id")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause")
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
            
        self.free_cam = False
        self.lookat_id = 0
        self.lookat_vec = torch.tensor([0.0, 2.0, 1.0], device=self.device)
            
    def get_observations(self):
        return self.obs_buf, self.critic_obs_buf
    
    def get_commands(self):
        return self.commands_buf, self.critic_commands_buf
    
    def get_policy_states(self):
        return self.states_buf, self.next_states_buf

    def get_task_info(self):
        completion, success_rate = self.motions.get_imitation_info()
        return {
            "total_completion": completion.item(),
            "total_success_rate": success_rate.item(),
        }
        
    def reset(self):
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids)
        # render twice to make sure the viewer is synced
        self.render();self.render()
        self.compute_observations()
        return self.get_observations()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """        
        # step physics and render each frame
        self.render()
        self.pre_physics_step(actions)        
        for _ in range(self.cfg.control.decimation):
            if self.motion_mode:
                self.cfg.domain_rand.randomize_initial_joint_position = False
                self._reset_robot_states(torch.arange(self.num_envs, device=self.device))
            else:
                self.torques = self._compute_torques(self.actions[:, -1]).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu": self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        
        return self.get_observations() + (self.rew_buf, self.reset_buf.float(), self.extras,)

    def pre_physics_step(self, actions):
        self.states_buf = self.compute_policy_states()
        
        clip_action = self.cfg.control.clip_action
        actions = torch.clip(actions.to(self.device), min=-clip_action, max=clip_action)
        self.actions[:, -1] = 0.0
        self.actions[:, -1, self.action_indices] = actions
        
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self._refresh_tensor_state()
        self._setup_tensor_state()
        self._setup_motion_state()
        self._post_physics_step_callback()
        
        # compute next states before checking termination
        self.next_states_buf = self.compute_policy_states()
        
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.motion_time += self.dt
        self.feet_air_time += self.dt
        
        # compute observations, rewards, resets, ...
        self.check_deviation()
        self.check_termination()
        self.compute_rewards()
        
        env_ids = self.reset_buf.nonzero().flatten()
        self.reset_idx(env_ids)
        
        self.compute_motions()
        self.compute_commands()
        self.compute_observations()
        self._post_states_step_callback()

    def check_deviation(self):
        """ Check if environments occurred deviation
        """
        deviation_distance = self.cfg.termination.deviation_distance
        
        diff_local_body_pos = self.motion_local_body_pos - self.local_body_pos
        diff_body_dist = torch.norm(diff_local_body_pos, dim=2)
        self.deviation = diff_body_dist.amax(dim=1) > deviation_distance
        
        deviation_dist = torch.clip(diff_body_dist - deviation_distance, min=0.0)
        self.deviation_distance = torch.sum(deviation_dist, dim=1)
        
        diff_body_pos = self.motion_body_pos - self.body_pos
        mix_diff_body_pos = torch.cat([
            diff_local_body_pos[..., 0:2], diff_body_pos[..., 2:3]], dim=2)
        max_diff_body_dist = torch.amax(torch.norm(mix_diff_body_pos, dim=2), dim=1)
        self.max_distance = torch.maximum(max_diff_body_dist, self.max_distance)
        
    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_height = self.cfg.termination.termination_height
        
        diff_body_pos = self.motion_body_pos - self.body_pos
        diff_body_height = torch.abs(
            diff_body_pos[:, self.end_effector_indices, 2])
        tracking_fail = diff_body_height.amax(dim=1) > termination_height

        if self.test_mode:
            tracking_fail = torch.zeros_like(tracking_fail)
        fallen_down = torch.zeros_like(tracking_fail)
        if len(self.terminate_contact_indices) > 0:
            fallen_down |= torch.any(self.terminate_contacts, dim=1)
        
        if self.cfg.tracking_reference.resample_motion and not self.test_mode:
            resampling = self.episode_length_buf > self.max_episode_length
        else:
            resampling = torch.ones_like(self.motion_success)
        self.time_out_buf = self.motion_success & resampling
        self.reset_buf = fallen_down | tracking_fail | self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments. Calls self._reset_robot_states(env_ids),
            and logs episode info Resets some buffers
        Args: env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0: return
        
        # fill extras
        self.extras["episode"], self.extras["episode_metrics"] = {}, {}
        for key in self.episode_sums.keys():
            episode_length = self.episode_length_buf[env_ids].clip(min=1)
            episode_sum = self.episode_sums[key][env_ids] / (episode_length * self.dt)
            self.extras["episode"]["rew_" + key] = torch.mean(episode_sum)            
            episode_metric = self.episode_metrics[key][env_ids] / episode_length
            self.extras["episode_metrics"][key] = torch.mean(episode_metric)
            self.episode_sums[key][env_ids], self.episode_metrics[key][env_ids] = 0.0, 0.0
        
        motion_time = self.motion_time[:, self.motion_pivot_id] - self.motion_init_time
        motion_total_time = self.motions.get_motion_time(self.motion_ids[:, 0])
        motion_total_time = torch.clip(motion_total_time - self.motion_init_time, min=1e-3)
        self.extras["completions"] = motion_time / motion_total_time
        self.extras["time_outs"] = self.time_out_buf
        
        self._reset_randomizations(env_ids)
        self._reset_env_origins(env_ids)
        self._reset_motions(env_ids)
        self._reset_robot_states(env_ids)
            
        # reset buffers        
        self.actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.external_forces[env_ids] = 0.0
        self.external_torques[env_ids] = 0.0
        
        self.max_distance[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.feet_contacts[env_ids] = False
        
        self.reset_buf[env_ids] = True
        self.episode_length_buf[env_ids] = 0
        self.motion_init_length[env_ids] = 0
        
    def compute_rewards(self):
        """ Compute rewards Calls each reward function which 
            had a non-zero scale (processed in self._prepare_rewards())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        termination_reward = 0.0
        momentum = self.cfg.rewards.momentum
        compute_ema = lambda x, y, eps=1e-6: max(eps, (1 - momentum) * x + momentum * y)
        
        for i, function in enumerate(self.reward_functions):
            name = self.reward_names[i]
            reward, metric = function()
            self.episode_metrics[name] += metric
            
            if "tracking" in name:
                original_sigma = self.tracking_sigma[name]
                current_sigma = compute_ema(original_sigma, torch.mean(metric).item())
                self.tracking_sigma[name] = min(current_sigma, original_sigma)
            elif "penalty" in name:
                original_penalty = self.penalty_coefficient[name]
                current_penalty = compute_ema(original_penalty, torch.mean(metric).item())
                self.penalty_coefficient[name] = min(current_penalty, original_penalty)

            if "termination" in name:
                scale = self.reward_scales[name] / self.dt
                termination_reward = scale * reward
                self.episode_sums[name] += termination_reward
            else:
                scale_reward = reward * self.reward_scales[name]
                self.rew_buf += scale_reward
                self.episode_sums[name] += scale_reward
        
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf = torch.clip(self.rew_buf, min=0.0)
        self.rew_buf += termination_reward
    
    def compute_motions(self):
        motion_dict = self.motions.get_motion_states(self.motion_ids, self.motion_time)
        for key in motion_dict.keys(): self.motion_dict[key] = motion_dict[key]
        
    def compute_observations(self):
        obs_scales = self.cfg.observation.obs_scales
        noise_scales = self.cfg.observation.noise_scales
        concat = lambda xs: torch.cat([x.view(self.num_envs, -1) for x in xs], dim=-1)
        
        # robot observations
        robot_base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_lin_vel)
        robot_base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_ang_vel)
        
        robot_local_body_pos = self.body_pos - self.base_pos[:, None]
        robot_body_pos = quat_rotate_inverse(self.base_quat[:, None], robot_local_body_pos)
        robot_end_effector_pos = robot_body_pos[:, self.end_effector_indices]
        
        robot_dof_pos = (self.dof_pos - self.default_dof_pos)[:, self.action_indices]
        robot_dof_vel = self.dof_vel[:, self.action_indices]
        robot_actions = self.actions[:, -1, self.action_indices]
        
        # compute observation noise
        compute_noise = lambda x, scale: torch_rand_like_float(
            -1.0, 1.0, x) * (scale if self.cfg.observation.add_noise else 0.0)
        
        base_ang_vel_noise = compute_noise(robot_base_ang_vel, noise_scales.base_ang_vel)
        gravity_noise = compute_noise(self.projected_gravity, noise_scales.gravity)
        end_effector_pos_noise = compute_noise(robot_end_effector_pos, noise_scales.body_pos)
        dof_pos_noise = compute_noise(robot_dof_pos, noise_scales.dof_pos)
        dof_vel_noise = compute_noise(robot_dof_vel, noise_scales.dof_vel)
        
        # compute proprioceptions
        self.obs_buf = concat([                
            robot_base_ang_vel * obs_scales.base_ang_vel + base_ang_vel_noise, 
            self.projected_gravity * obs_scales.gravity + gravity_noise,
            robot_end_effector_pos * obs_scales.body_pos + end_effector_pos_noise,
            robot_dof_pos * obs_scales.dof_pos + dof_pos_noise, 
            robot_dof_vel * obs_scales.dof_vel + dof_vel_noise,
            robot_actions,
            ])
        
        self.critic_obs_buf = concat([
            robot_base_lin_vel * obs_scales.base_lin_vel,
            robot_base_ang_vel * obs_scales.base_ang_vel,
            self.projected_gravity * obs_scales.gravity,
            robot_end_effector_pos * obs_scales.body_pos,
            robot_dof_pos * obs_scales.dof_pos, 
            robot_dof_vel * obs_scales.dof_vel,
            robot_actions,
            ])
        
        clip_obs = self.cfg.observation.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, min=-clip_obs, max=clip_obs)
        self.critic_obs_buf = torch.clip(self.critic_obs_buf, min=-clip_obs, max=clip_obs)
        
    def compute_commands(self):
        clip_obs = self.cfg.observation.clip_observations
        obs_scales = self.cfg.observation.obs_scales
        concat = lambda xs: torch.cat([x.view(self.num_envs, -1) for x in xs], dim=-1)
            
        # motion observations
        motion_pivot_quat = self.motion_dict["base_quat"][:, self.motion_pivot_id, None]
        
        motion_base_height = self.motion_dict["base_pos"][..., 2]
        motion_base_quat = quat_mul_yaw(self.base_quat_offset[:, None], self.motion_dict["base_quat"])
        motion_base_lin_vel = quat_rotate_yaw_inverse(motion_pivot_quat, self.motion_dict["base_lin_vel"])
        motion_base_ang_vel = quat_rotate_yaw_inverse(motion_pivot_quat, self.motion_dict["base_ang_vel"])
        motion_dof_pos = (self.motion_dict["dof_pos"] - self.default_dof_pos)[..., self.motion_dof_indices]
        motion_body_pos = quat_rotate_yaw_inverse(
            self.motion_dict["base_quat"][..., None, :], 
            self.motion_dict["body_pos"] - self.motion_dict["base_pos"][..., None, :])
        motion_body_pos[..., 2] = self.motion_dict["body_pos"][..., 2]
        motion_end_effector_pos = motion_body_pos[..., self.end_effector_indices, :]
        
        # residual observations
        residual_dof_pos = (self.motion_dof_pos - self.dof_pos)[:, self.motion_dof_indices]
        residual_body_pos = self.motion_local_body_pos - self.local_body_pos
        
        # convert quat to rotation
        motion_base_rotation = quat_to_rot6d(motion_base_quat)
        
        # compute commands
        commands_buf = concat([
            motion_base_rotation * obs_scales.rotation,
            motion_base_height * obs_scales.base_pos,
            motion_base_lin_vel * obs_scales.base_lin_vel,
            motion_base_ang_vel * obs_scales.base_ang_vel,
            motion_dof_pos * obs_scales.dof_pos,
            motion_end_effector_pos * obs_scales.body_pos,
            ])
        self.commands_buf = torch.clip(commands_buf, min=-clip_obs, max=clip_obs)
        
        critic_commands_buf = concat([
            motion_base_rotation * obs_scales.rotation,
            motion_base_height * obs_scales.base_pos,
            motion_base_lin_vel * obs_scales.base_lin_vel,
            motion_base_ang_vel * obs_scales.base_ang_vel,
            motion_dof_pos * obs_scales.dof_pos,
            motion_body_pos * obs_scales.body_pos,
            
            residual_dof_pos * obs_scales.dof_pos,
            residual_body_pos * obs_scales.body_pos,
            ])
        self.critic_commands_buf = torch.clip(critic_commands_buf, min=-clip_obs, max=clip_obs)
        
    def compute_policy_states(self):
        obs_scales = self.cfg.observation.obs_scales
        concat = lambda xs: torch.cat([x.view(self.num_envs, -1) for x in xs], dim=-1)
        
        base_height = self.base_pos[:, 2] - self.measured_heights.mean(dim=1)
        base_rotation = quat_to_rot6d(self.base_quat)
        dof_pos = (self.dof_pos - self.default_dof_pos)[:, self.motion_dof_indices]
        dof_vel = self.dof_vel[:, self.motion_dof_indices]
        end_effector_pos = self.local_body_pos[:, self.end_effector_indices]
            
        return concat([
            base_height * obs_scales.base_pos,
            base_rotation * obs_scales.rotation,
            self.local_base_lin_vel * obs_scales.base_lin_vel, 
            self.local_base_ang_vel * obs_scales.base_ang_vel, 
            dof_pos * obs_scales.dof_pos,
            dof_vel * obs_scales.dof_vel,
            end_effector_pos * obs_scales.body_pos,])
    
    def compute_expert_states(self, motion_dict):
        assert motion_dict["base_pos"].shape[1] == 2
        batch_size = motion_dict["base_pos"].shape[0]
        device = motion_dict["base_pos"].device
        obs_scales = self.cfg.observation.obs_scales
        concat = lambda xs: torch.cat([x.view(batch_size, -1) for x in xs], dim=-1)
                
        base_rpy_offset = torch.zeros(batch_size, 3, dtype=torch.float, device=device)
        base_rpy_offset[:, 2] = torch_rand_float(-np.pi, np.pi, (batch_size,), device=device)
        base_quat_offset = euler_xyz_to_quat(base_rpy_offset)
        
        base_height = motion_dict["base_pos"][..., 2]
        base_quat = quat_mul(base_quat_offset[:, None], motion_dict["base_quat"])
        base_rotation = quat_to_rot6d(base_quat)
        
        base_lin_vel = quat_rotate_yaw_inverse(motion_dict["base_quat"], motion_dict["base_lin_vel"])
        base_ang_vel = quat_rotate_yaw_inverse(motion_dict["base_quat"], motion_dict["base_ang_vel"])
        
        motion_dof_indices = self.motion_dof_indices.to(device)
        end_effector_indices = self.end_effector_indices.to(device)
        
        dof_pos = (motion_dict["dof_pos"] - self.default_dof_pos.to(device))[..., motion_dof_indices]
        dof_vel = motion_dict["dof_vel"][..., motion_dof_indices]
        body_pos_wrt_base = motion_dict["body_pos"] - motion_dict["base_pos"][..., None, :]
        body_pos = quat_rotate_yaw_inverse(motion_dict["base_quat"][..., None, :], body_pos_wrt_base)
        end_effector_pos = body_pos[..., end_effector_indices, :]
        
        expert_states = concat([
            base_height[:, 0] * obs_scales.base_pos,
            base_rotation[:, 0] * obs_scales.rotation,
            base_lin_vel[:, 0] * obs_scales.base_lin_vel, 
            base_ang_vel[:, 0] * obs_scales.base_ang_vel, 
            dof_pos[:, 0] * obs_scales.dof_pos,
            dof_vel[:, 0] * obs_scales.dof_vel,
            end_effector_pos[:, 0] * obs_scales.body_pos,])
        
        next_expert_states = concat([
            base_height[:, 1] * obs_scales.base_pos,
            base_rotation[:, 1] * obs_scales.rotation,
            base_lin_vel[:, 1] * obs_scales.base_lin_vel, 
            base_ang_vel[:, 1] * obs_scales.base_ang_vel, 
            dof_pos[:, 1] * obs_scales.dof_pos,
            dof_vel[:, 1] * obs_scales.dof_vel,
            end_effector_pos[:, 1] * obs_scales.body_pos,])
        
        assert expert_states.shape == (batch_size, self.num_states)        
        return expert_states, next_expert_states
    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if self.cfg.terrain.mesh_type == "plane":
            self._create_ground_plane()
        elif self.cfg.terrain.mesh_type == "heightfield":
            self._create_heightfield()
        elif self.cfg.terrain.mesh_type == "trimesh":
            self._create_trimesh()
        elif self.cfg.terrain.mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def lookat(self, env_id):
        look_at_pos = self.root_states[env_id, 0:3].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)
    
    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
                
            if not self.free_cam:
                self.lookat(self.lookat_id)
                
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                
                if not self.free_cam:
                    for i in range(9):
                        if evt.action == "lookat" + str(i) and evt.value > 0:
                            self.lookat(i)
                            self.lookat_id = i
                    if evt.action == "prev_env_id" and evt.value > 0:
                        self.lookat_id = (self.lookat_id-1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_env_id" and evt.value > 0:
                        self.lookat_id = (self.lookat_id+1) % self.num_envs
                        self.lookat(self.lookat_id)

                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    if self.free_cam:
                        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
                
                if evt.action == "pause" and evt.value > 0:
                    self.pause = True
                    while self.pause:
                        time.sleep(0.1)
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        for evt in self.gym.query_viewer_action_events(self.viewer):
                            if evt.action == "pause" and evt.value > 0:
                                self.pause = False
                        if self.gym.query_viewer_has_closed(self.viewer):
                            sys.exit()

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    #------------- Callbacks --------------
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
            for i in range(len(props)):            
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) * 0.5
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        
        joint_friction = 1.0
        joint_damping = 1.0
        joint_armature = 0.0
        
        if self.cfg.domain_rand.randomize_joint_friction:
            rng = self.cfg.domain_rand.joint_friction_range
            joint_friction = np.random.uniform(rng[0], rng[1])
        
        if self.cfg.domain_rand.randomize_joint_damping:
            rng = self.cfg.domain_rand.joint_damping_range
            joint_damping = np.random.uniform(rng[0], rng[1])
        
        if self.cfg.domain_rand.randomize_joint_armature:
            rng = self.cfg.domain_rand.joint_armature_range
            joint_armature = np.random.uniform(rng[0], rng[1])
        
        for j in range(len(props)):
            props["friction"][j] *= joint_friction
            props["damping"][j] *= joint_damping
            props["armature"][j] = joint_armature
        
        return props

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
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
                rng, num_buckets = self.cfg.domain_rand.friction_range, 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs,))
                friction_buckets = torch_rand_float(rng[0], rng[1], (num_buckets,), device="cpu")
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                
        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                # prepare restitution randomization
                rng, num_buckets = self.cfg.domain_rand.restitution_range, 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs,))
                restitution_buckets = torch_rand_float(rng[0], rng[1], (num_buckets,), device="cpu")
                self.restitution_coeffs = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
                
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            self.default_body_com_x = np.array([p.com.x for p in props])
            self.default_body_com_y = np.array([p.com.y for p in props])
            self.default_body_com_z = np.array([p.com.z for p in props])
            self.default_body_mass = np.array([p.mass for p in props])
            self.total_mass = self.default_body_mass.sum()
            print(f"Total mass {self.total_mass:.3f} (before randomization)")
            
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_payload:
            payload_id = self.payload_indices[0]
            rng = self.cfg.domain_rand.added_payload_range
            payload_mass = np.random.uniform(rng[0] * self.total_mass, rng[1] * self.total_mass)
            props[payload_id].mass = np.maximum(self.default_body_mass[payload_id] + payload_mass, 1e-3)

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                props[i].mass = self.default_body_mass[i] * np.random.uniform(rng[0], rng[1])
        if self.cfg.domain_rand.randomize_link_com:
            rng = self.cfg.domain_rand.link_com_range
            for i in range(len(props)):
                props[i].com.x = self.default_body_com_x[i] + np.random.uniform(rng[0], rng[1])
                props[i].com.y = self.default_body_com_y[i] + np.random.uniform(rng[0], rng[1])
                props[i].com.z = self.default_body_com_z[i] + np.random.uniform(rng[0], rng[1])
                
        if env_id == 0:
            self.min_total_mass = sum([p.mass for p in props])
        else:
            self.min_total_mass = min(
                self.min_total_mass, sum([p.mass for p in props]))
        if env_id == self.num_envs - 1:
            print(f"Total mass {self.min_total_mass:.3f} (after randomization)")
        return props
    
    def _post_physics_step_callback(self):
        if self.cfg.domain_rand.randomize_joint_injection:
            self._apply_joint_injections()
            
        if self.cfg.domain_rand.push_robot_base:
            self._apply_push_robot_bases()
            
        if self.cfg.domain_rand.push_robot_body:
            self._apply_push_robot_bodys()
        
        self.motion_success = self.motions.check_success(
            self.motion_ids[:, 0],
            self.motion_time[:, self.motion_pivot_id])

        if self.cfg.tracking_reference.resample_motion and not self.test_mode:
            env_ids = self.motion_success.nonzero().flatten()
            if len(env_ids) == 0: return

            self._reset_motions(env_ids, resample_mode=True)
            self.max_distance[env_ids] = 0.0
            self.motion_init_length[env_ids] = self.episode_length_buf[env_ids]
            
    def _post_states_step_callback(self):
        self.last_dof_vel[:] = self.dof_vel.clone()
        self.actions[:, :-1] = self.actions[:, 1:].clone()
        self.feet_contacts[:, :-1] = self.feet_contacts[:, 1:].clone()
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity 
            targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, 
            even if some DOFs are not actuated.
        Args: actions (torch.Tensor): Actions
        Returns: [torch.Tensor]: Torques sent to the simulation
        """
        # PD controller
        control_type = self.cfg.control.control_type
        if control_type == "P":
            p_gains = self.p_gains * self.kp_factors
            d_gains = self.d_gains * self.kd_factors
            action_targets = actions * self.action_scales + self.default_dof_pos
            torques = p_gains * (action_targets - self.dof_pos) - d_gains * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        return torques + self.actuation_offset + self.joint_injection
    
    def _reset_motions(self, env_ids, resample_mode=False):
        pivot_motion_time=self.motion_time[env_ids, self.motion_pivot_id],
        self.motions.update_imitation_info(
            motion_ids=self.motion_ids[env_ids, 0],
            init_time=self.motion_init_time[env_ids],
            motion_time=pivot_motion_time,
            success=self.motion_success[env_ids],
            max_distance=self.max_distance[env_ids],
            )
        
        if self.cfg.tracking_reference.randomize_motion_speed:
            speed_rng = self.cfg.tracking_reference.motion_speed_range
            random_timestep_sequences = self.timestep_sequences * \
                torch_rand_float(speed_rng[0], speed_rng[1], (len(env_ids), 1), device=self.device)
        else:
            random_timestep_sequences = self.timestep_sequences

        motion_ids = self.motions.sample_motions(len(env_ids), uniform=self.test_mode)
        motion_ids = motion_ids[:, None].repeat(1, self.motion_horizon)
        motion_init_time = self.motions.sample_init_time(motion_ids[:, 0])
        if self.test_mode: motion_init_time *= 0.0        
        motion_time = motion_init_time[:, None] + random_timestep_sequences
            
        motion_dict = self.motions.get_motion_states(motion_ids, motion_time)
        for key in motion_dict.keys(): self.motion_dict[key][env_ids] = motion_dict[key]
        
        self.motion_ids[env_ids] = motion_ids
        self.motion_init_time[env_ids] = motion_init_time
        self.motion_time[env_ids] = motion_time + self.dt
        
        if resample_mode:
            self.base_rpy_offset[env_ids, 2] = quat_to_euler_xyz(self.base_quat[env_ids])[:, 2]
            self.base_quat_offset[env_ids] = euler_xyz_to_quat(self.base_rpy_offset[env_ids])
            self.base_pos_offset[env_ids, 0:2] = self.base_pos[env_ids, 0:2]
        else:
            self.base_rpy_offset[env_ids, 2] = torch_rand_float(-np.pi, np.pi, (len(env_ids),), device=self.device)
            self.base_quat_offset[env_ids] = euler_xyz_to_quat(self.base_rpy_offset[env_ids])
            self.base_pos_offset[env_ids, 0:2] = self.env_origins[env_ids, 0:2] + self.base_init_state[0:2]
                  
    def _reset_env_origins(self, env_ids):
        """ Reset environment origins.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.terrain_levels[env_ids] = torch.randint(0, self.cfg.terrain.num_rows, (len(env_ids),), device=self.device)
            self.terrain_types[env_ids] = torch.randint(0, self.cfg.terrain.num_cols, (len(env_ids),), device=self.device)
            self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            self.env_origins[env_ids, 0:2] += torch_rand_float(-2.0, 2.0, (len(env_ids), 2), device=self.device)

    def _reset_robot_states(self, env_ids):
        """ Resets ROOT states, DOF position and velocities of selected environmments
        
        Args:
            env_ids (List[int]): Environemnt ids
        """        
        self.root_states[env_ids] = self.base_init_state
        if self.motion_mode:
            self.root_states[env_ids, 2] -= self.base_init_state[2]
        self.root_states[env_ids, 0:3] += self.env_origins[env_ids]
        self.root_states[env_ids, 0:3] += quat_rotate_yaw(
            self.base_quat_offset[env_ids], self.motion_dict["base_pos"][env_ids, self.motion_pivot_id])
        self.root_states[env_ids, 3:7] = quat_mul_yaw(
            self.base_quat_offset[env_ids], self.motion_dict["base_quat"][env_ids, self.motion_pivot_id])
        
        # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 7:10] = quat_rotate_yaw(
            self.base_quat_offset[env_ids], self.motion_dict["base_lin_vel"][env_ids, self.motion_pivot_id])
        self.root_states[env_ids, 10:13] = quat_rotate_yaw(
            self.base_quat_offset[env_ids], self.motion_dict["base_ang_vel"][env_ids, self.motion_pivot_id])
        self.root_states[env_ids, 7:13] += torch_rand_float(-0.2, 0.2, (len(env_ids), 6), device=self.device)
        
        self.dof_states[env_ids, :, 0] = self.motion_dict["dof_pos"][env_ids, self.motion_pivot_id]
        self.dof_states[env_ids, :, 1] = 0.0
        # self.dof_states[env_ids, :, 1] = self.motion_dict["dof_vel"][env_ids, self.motion_pivot_id]
        
        if self.cfg.domain_rand.randomize_initial_joint_position:
            rng = self.cfg.domain_rand.initial_joint_offset_range
            joint_offset = torch_rand_float(rng[0], rng[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_states[env_ids, :, 0] = torch.clip(self.dof_states[env_ids, :, 0] + joint_offset,
                                                        min=self.dof_pos_limits[:, 0], max=self.dof_pos_limits[:, 1])
        
        self.body_pos[env_ids] = self.base_init_state[0:3] + self.env_origins[env_ids, None]
        self.body_pos[env_ids] += quat_rotate_yaw(
            self.base_quat_offset[env_ids, None], self.motion_dict["body_pos"][env_ids, self.motion_pivot_id])
        self.body_quat[env_ids] = quat_mul_yaw(
            self.base_quat_offset[env_ids, None], self.motion_dict["body_quat"][env_ids, self.motion_pivot_id])
        self.body_lin_vel[env_ids] = self.root_states[env_ids, None, 7:10]
        self.body_ang_vel[env_ids] = self.root_states[env_ids, None, 10:13]
        # self.body_lin_vel[env_ids] = quat_rotate_yaw(
        #     self.base_quat_offset[env_ids, None], self.motion_dict["body_lin_vel"][env_ids, self.motion_pivot_id])
        # self.body_ang_vel[env_ids] = quat_rotate_yaw(
        #     self.base_quat_offset[env_ids, None], self.motion_dict["body_ang_vel"][env_ids, self.motion_pivot_id])
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_states), 
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def _reset_randomizations(self, env_ids):
        # randomize kp, kd, motor strength
        if self.cfg.domain_rand.randomize_pd_gain:
            kp_rng, kd_rng = self.cfg.domain_rand.kp_range, self.cfg.domain_rand.kd_range
            self.kp_factors[env_ids] = torch_rand_float(
                kp_rng[0], kp_rng[1], (len(env_ids), self.num_dof), device=self.device)
            self.kd_factors[env_ids] = torch_rand_float(
                kd_rng[0], kd_rng[1], (len(env_ids), self.num_dof), device=self.device)

        if self.cfg.domain_rand.randomize_actuation_offset:
            rng = self.cfg.domain_rand.actuation_offset_range
            random_scale = torch_rand_float(rng[0], rng[1], (len(env_ids), self.num_dof), device=self.device)
            self.actuation_offset[env_ids] = random_scale * self.torque_limits

    def _apply_joint_injections(self):
        rng = self.cfg.domain_rand.joint_injection_range
        random_scale = torch_rand_float(rng[0], rng[1], (self.num_envs, self.num_dof), device=self.device)
        self.joint_injection = random_scale * self.torque_limits

    def _apply_push_robot_bases(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.common_step_counter % self.random_push_base_interval == 0:
            max_lin_vel = self.cfg.domain_rand.push_robot_base_max_lin_vel
            self.root_states[:, 7:9] += torch_rand_float(-max_lin_vel, max_lin_vel, (self.num_envs, 2), device=self.device)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.random_push_base_interval = np.random.randint(self.push_robot_base_interval[0], self.push_robot_base_interval[1])

    def _apply_push_robot_bodys(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        random_push_body_total_interval = self.random_push_body_interval + self.random_push_body_duration
        if self.common_step_counter % random_push_body_total_interval == 0:
            max_force = self.cfg.domain_rand.push_robot_body_max_force
            max_torque = self.cfg.domain_rand.push_robot_body_max_torque
            self.external_forces[:, self.perturb_body_indices] = torch_rand_float(
                -max_force, max_force, (self.num_envs, len(self.perturb_body_indices), 3), device=self.device)
            self.external_torques[:, self.perturb_body_indices] = torch_rand_float(
                -max_torque, max_torque, (self.num_envs, len(self.perturb_body_indices), 3), device=self.device)
            self.gym.apply_rigid_body_force_tensors(self.sim,
                gymtorch.unwrap_tensor(self.external_forces), gymtorch.unwrap_tensor(self.external_torques))
        elif self.common_step_counter % random_push_body_total_interval < self.random_push_body_duration:
            self.gym.apply_rigid_body_force_tensors(self.sim,
                gymtorch.unwrap_tensor(self.external_forces), gymtorch.unwrap_tensor(self.external_torques))
        elif self.common_step_counter % random_push_body_total_interval == self.random_push_body_duration:
            self.random_push_body_interval = np.random.randint(self.push_robot_body_interval[0], self.push_robot_body_interval[1])
            self.random_push_body_duration = np.random.randint(self.push_robot_body_duration[0], self.push_robot_body_duration[1])

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._refresh_tensor_state()

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.dof_states = gymtorch.wrap_tensor(dof_state).view(self.num_envs, self.num_dof, 2)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        
        num_feet = len(self.feet_contact_indices)
        self.feet_contacts = torch.zeros(self.num_envs, 2, num_feet, dtype=torch.bool, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device)

        self.deviation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.deviation_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        gravity_vec = get_axis_params(-1.0, self.up_axis_idx)
        self.gravity_vec = torch.tensor(gravity_vec, dtype=torch.float, device=self.device)
        self.gravity_vec = self.gravity_vec.repeat(self.num_envs, 1)
        self.height_points = self._init_height_points()
        
        self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, dtype=torch.float, device=self.device)
        self.base_rpy_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.base_quat_offset = euler_xyz_to_quat(self.base_rpy_offset)
        self.base_pos_offset = torch.cat([self.env_origins[:, 0:2] + self.base_init_state[0:2], 
                                          self.measured_heights.mean(dim=1, keepdim=True)], dim=1)
        self._setup_tensor_state()
        
        # pushing robot setting
        self.random_push_base_interval = np.random.randint(self.push_robot_base_interval[0], self.push_robot_base_interval[1])
        self.random_push_body_duration = np.random.randint(self.push_robot_body_duration[0], self.push_robot_body_duration[1])
        self.random_push_body_interval = np.random.randint(self.push_robot_body_interval[0], self.push_robot_body_interval[1])
        self.external_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.external_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)

        # initialize some data used later on
        self.common_step_counter, self.extras = 0, {}
        self.actions = torch.zeros(self.num_envs, 3, self.num_dof, dtype=torch.float, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.action_scales = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        
        self.max_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # joint positions offsets and PD gains
        zero_dof_names = []
        action_indices = []
        smooth_action_indices = []
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        for i, name in enumerate(self.dof_names):
            if name in self.cfg.init_state.default_joint_angles.keys():
                self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            else: zero_dof_names.append(name)
            found = False
            for cfg_name in self.cfg.control.stiffness.keys():
                if cfg_name in name:
                    found = True
                    self.p_gains[i] = self.cfg.control.stiffness[cfg_name]
                    self.d_gains[i] = self.p_gains[i] * self.cfg.control.default_damping_ratio
                    self.action_scales[i] = self.cfg.control.action_scale
                    self.action_scales[i] *= torch.abs(self.torque_limits[i] / self.p_gains[i])
                    
                    for act_name in self.cfg.control.action_names:
                        if act_name in name:
                            action_indices.append(i)
                            break
                        
                    for act_name in self.cfg.control.smooth_action_names:
                        if act_name in name:
                            smooth_action_indices.append(i)
                            break
            for cfg_name in self.cfg.control.damping.keys():
                if cfg_name in name:
                    self.d_gains[i] = self.cfg.control.damping[cfg_name]
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                print(f"PD gains of joint {name} were not defined, setting them to zero.")
        
        self.num_actions = len(action_indices)
        self.action_indices = torch.tensor(action_indices, dtype=torch.long, device=self.device)
        self.smooth_action_indices = torch.tensor(smooth_action_indices, dtype=torch.long, device=self.device)
        print(f"Setting default joint position of joint {zero_dof_names} to zero.")
        
        # randomize kp, kd, motor strength
        if self.cfg.domain_rand.randomize_pd_gain:
            kp_rng, kd_rng = self.cfg.domain_rand.kp_range, self.cfg.domain_rand.kd_range
            self.kp_factors = torch_rand_float(kp_rng[0], kp_rng[1], (self.num_envs, self.num_dof), device=self.device)
            self.kd_factors = torch_rand_float(kd_rng[0], kd_rng[1], (self.num_envs, self.num_dof), device=self.device)
        else:
            self.kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
            self.kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)

        if self.cfg.domain_rand.randomize_joint_injection:
            rng = self.cfg.domain_rand.joint_injection_range
            random_scale = torch_rand_float(rng[0], rng[1], (self.num_envs, self.num_dof), device=self.device)
            self.joint_injection = random_scale * self.torque_limits
        else:
            self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        
        if self.cfg.domain_rand.randomize_actuation_offset:
            rng = self.cfg.domain_rand.actuation_offset_range
            random_scale = torch_rand_float(rng[0], rng[1], (self.num_envs, self.num_dof), device=self.device)
            self.actuation_offset = random_scale * self.torque_limits
        else:
            self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)

        self._init_observations()
        self._init_motion_references()
        
    def _init_observations(self):
        """ Initialize torch tensors which will contain observations and commands
        """
        motion_horizon = self.cfg.env.motion_horizon
        num_joint = len(self.motion_dof_indices)
        num_keyframe = len(self.keyframe_indices)
        num_end_effector = len(self.end_effector_indices)
        
        self.num_obs = 3 + 3 + num_end_effector * 3 + self.num_actions * 3
        self.num_critic_obs = 3 + 3 + 3 + num_end_effector * 3 + self.num_actions * 3
        
        one_step_command = 1 + 6 + 3 + 3 + num_joint + num_end_effector * 3
        one_step_critic_command = 1 + 6 + 3 + 3 + num_joint + num_keyframe * 3
        self.num_commands = motion_horizon * one_step_command
        self.num_critic_commands = motion_horizon * one_step_critic_command + num_joint + num_keyframe * 3
        
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_critic_obs, dtype=torch.float, device=self.device)
        self.commands_buf = torch.zeros(self.num_envs, self.num_commands, dtype=torch.float, device=self.device)
        self.critic_commands_buf = torch.zeros(self.num_envs, self.num_critic_commands, dtype=torch.float, device=self.device)
        
        # for discriminator
        self.num_states = 1 + 6 + 3 + 3 + num_joint * 2 + num_end_effector * 3
        self.states_buf = torch.zeros(self.num_envs, self.num_states, dtype=torch.float, device=self.device)
        self.next_states_buf = torch.zeros(self.num_envs, self.num_states, dtype=torch.float, device=self.device)

    def _init_motion_references(self):
        """ Initialize motion library and imitation schedule
        """
        self.motion_pivot_id = 0
        self.motion_horizon = self.cfg.env.motion_horizon
        self.motion_time_horizon = (self.motion_horizon - 1) * self.cfg.env.motion_dt
        
        folder_path = os.path.join(self.cfg.tracking_reference.prefix, self.cfg.tracking_reference.folder)
        mapping_path = os.path.join(folder_path, self.cfg.tracking_reference.joint_mapping)
        dataset, data_names, mapping = load_imitation_dataset(folder_path, mapping_path)
        filter_cfg = self.cfg.tracking_reference.motion_filter
        dataset, data_names = filter_and_fix_motion(
            dataset, data_names,
            min_time=filter_cfg.min_time,
            link_height_offset=filter_cfg.link_height_offset,
            floor_height_offset=filter_cfg.floor_height_offset,
            base_height_range=filter_cfg.base_height_range,
            base_roll_range=filter_cfg.base_roll_range,
            base_pitch_range=filter_cfg.base_pitch_range,
            )
        self.motions = MotionLib(dataset, data_names, mapping, self.dof_names, self.keyframe_names, self.device)
        self.motion_ids = self.motions.sample_motions(self.num_envs, uniform=self.test_mode)
        self.motion_ids = self.motion_ids[:, None].repeat(1, self.motion_horizon)
        
        self.timestep_sequences = torch.linspace(0, self.motion_time_horizon,
                                                 self.motion_horizon,
                                                 dtype=torch.float, device=self.device)
        self.motion_init_length = self.episode_length_buf.clone()
        
        if self.cfg.tracking_reference.randomize_motion_speed:
            speed_rng = self.cfg.tracking_reference.motion_speed_range
            random_timestep_sequences = self.timestep_sequences * \
                torch_rand_float(speed_rng[0], speed_rng[1], (self.num_envs, 1), device=self.device)
        else:
            random_timestep_sequences = self.timestep_sequences
        
        self.motion_init_time = self.motions.sample_init_time(self.motion_ids[:, 0])
        if self.test_mode: self.motion_init_time *= 0.0
        self.motion_time = self.motion_init_time[:, None] + random_timestep_sequences
        self.motion_dict = self.motions.get_motion_states(self.motion_ids, self.motion_time)
        self.motion_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)        
        self._setup_motion_state()
        
    def init_expert_dataloader(self):
        folder_path = os.path.join(self.cfg.skill_demonstration.prefix, self.cfg.skill_demonstration.folder)
        mapping_path = os.path.join(folder_path, self.cfg.skill_demonstration.joint_mapping)
        dataset, data_names, mapping = load_imitation_dataset(folder_path, mapping_path)
        expert_dataloader = MotionLoader(dataset, data_names, mapping, 
                                         self.dof_names, self.keyframe_names, self.dt, self.device)
        expert_dataloader.load_states_function(self.compute_expert_states)
        return expert_dataloader
 
    def _refresh_tensor_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
    def _setup_tensor_state(self):
        # global states
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = self.root_states[:, 7:10]
        self.base_ang_vel = self.root_states[:, 10:13]
        
        self.body_pos = self.rigid_body_states[:, self.keyframe_indices, 0:3]
        self.body_quat = self.rigid_body_states[:, self.keyframe_indices, 3:7]
        self.body_lin_vel = self.rigid_body_states[:, self.keyframe_indices, 7:10]
        self.body_ang_vel = self.rigid_body_states[:, self.keyframe_indices, 10:13]
        
        self.dof_pos = self.dof_states[..., 0]
        self.dof_vel = self.dof_states[..., 1]
        
        # local states
        self.local_base_lin_vel = quat_rotate_yaw_inverse(self.base_quat, self.base_lin_vel)
        self.local_base_ang_vel = quat_rotate_yaw_inverse(self.base_quat, self.base_ang_vel)
        self.local_body_pos = quat_rotate_yaw_inverse(self.base_quat[:, None], 
                                                      self.body_pos - self.base_pos[:, None])
        self.local_body_quat = quat_mul_yaw_inverse(self.base_quat[:, None], self.body_quat)
        self.local_body_lin_vel = quat_rotate_yaw_inverse(self.base_quat[:, None], self.body_lin_vel)
        self.local_body_ang_vel = quat_rotate_yaw_inverse(self.base_quat[:, None], self.body_ang_vel)
        
        self.measured_heights = self._get_heights()
        self.base_pos_offset[:, 2] = self.measured_heights.mean(dim=1)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.feet_contacts[:, 1] = torch.abs(self.contact_forces[:, self.feet_contact_indices, 2]) > 1.0
        self.terminate_contacts = torch.norm(self.contact_forces[:, self.terminate_contact_indices], dim=2) > 1.0
        self.feet_first_contact = (self.feet_air_time > 0.0) & torch.any(self.feet_contacts, dim=1)

    def _setup_motion_state(self):
        motion_base_pos = self.motion_dict["base_pos"][:, self.motion_pivot_id]
        motion_base_quat = self.motion_dict["base_quat"][:, self.motion_pivot_id]
        motion_base_lin_vel = self.motion_dict["base_lin_vel"][:, self.motion_pivot_id]
        motion_base_ang_vel = self.motion_dict["base_ang_vel"][:, self.motion_pivot_id]
        
        # global states
        self.motion_base_pos = quat_rotate_yaw(self.base_quat_offset, motion_base_pos) + self.base_pos_offset
        self.motion_base_quat = quat_mul_yaw(self.base_quat_offset, motion_base_quat)
        self.motion_base_lin_vel = quat_rotate_yaw(self.base_quat_offset, motion_base_lin_vel)
        self.motion_base_ang_vel = quat_rotate_yaw(self.base_quat_offset, motion_base_ang_vel)
        
        motion_body_pos = self.motion_dict["body_pos"][:, self.motion_pivot_id]
        motion_body_quat = self.motion_dict["body_quat"][:, self.motion_pivot_id]
        motion_body_lin_vel = self.motion_dict["body_lin_vel"][:, self.motion_pivot_id]
        motion_body_ang_vel = self.motion_dict["body_ang_vel"][:, self.motion_pivot_id]
        
        base_pos_offset = self.base_pos_offset[:, None]
        base_quat_offset = self.base_quat_offset[:, None]
        self.motion_body_pos = quat_rotate_yaw(base_quat_offset, motion_body_pos) + base_pos_offset
        self.motion_body_quat = quat_mul_yaw(base_quat_offset, motion_body_quat)
        self.motion_body_lin_vel = quat_rotate_yaw(base_quat_offset, motion_body_lin_vel)
        self.motion_body_ang_vel = quat_rotate_yaw(base_quat_offset, motion_body_ang_vel)
        
        # local states
        self.motion_local_base_lin_vel = quat_rotate_yaw_inverse(motion_base_quat, motion_base_lin_vel)
        self.motion_local_base_ang_vel = quat_rotate_yaw_inverse(motion_base_quat, motion_base_ang_vel)
        self.motion_local_body_pos = quat_rotate_yaw_inverse(motion_base_quat[:, None],
                                                             motion_body_pos - motion_base_pos[:, None])
        self.motion_local_body_quat = quat_mul_yaw_inverse(motion_base_quat[:, None], motion_body_quat)
        self.motion_local_body_lin_vel = quat_rotate_yaw_inverse(motion_base_quat[:, None], motion_body_lin_vel)
        self.motion_local_body_ang_vel = quat_rotate_yaw_inverse(motion_base_quat[:, None], motion_body_ang_vel)
        
        self.motion_dof_pos = self.motion_dict["dof_pos"][:, self.motion_pivot_id]
        self.motion_dof_vel = self.motion_dict["dof_vel"][:, self.motion_pivot_id]
        
    def _prepare_rewards(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt        
        for name in list(self.reward_scales.keys()):
            scale = self.reward_scales[name]
            if scale == 0.0:
                self.reward_scales.pop(name)
            else:
                self.reward_scales[name] *= self.dt
        
        # prepare list of functions
        self.tracking_sigma, self.penalty_coefficient = {}, {}
        self.reward_functions, self.reward_names = [], []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))
            
            if "tracking" in name:
                self.tracking_sigma[name] = self.cfg.rewards.init_sigma
            elif "penalty" in name:
                self.penalty_coefficient[name] = self.cfg.rewards.init_coefficient
        
        # reward episode sums
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_sums = {name: torch.zeros_like(self.rew_buf) for name in self.reward_scales.keys()}
        self.episode_metrics = {name: torch.zeros_like(self.rew_buf) for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
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
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order="C"), self.terrain.triangles.flatten(order="C"), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
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

        self.keyframe_names = [s for s in self.body_names if self.cfg.asset.keyframe_name in s]
        self.body_link_names = [s for s in self.body_names if not s in self.keyframe_names]
        print(f"Reduced body names: {self.body_link_names}")        
        get_body_index = lambda n: self.gym.find_asset_rigid_body_index(robot_asset, n)

        trunk_indices, marker_indices, mobile_indices, end_effector_indices = [], [], [], []
        for i, keyframe_name in enumerate(self.keyframe_names):
            add_keyframe_index = lambda ids, names: ids + [i for name in names if name in keyframe_name]
            trunk_indices = add_keyframe_index(trunk_indices, self.cfg.asset.trunk_names)
            marker_indices = add_keyframe_index(marker_indices, self.cfg.asset.marker_names)
            mobile_indices = add_keyframe_index(mobile_indices, self.cfg.asset.mobile_names)
            end_effector_indices = add_keyframe_index(end_effector_indices, self.cfg.asset.end_effector_names)
        self.trunk_indices = torch.tensor(trunk_indices, dtype=torch.long, device=self.device)
        self.marker_indices = torch.tensor(marker_indices, dtype=torch.long, device=self.device)
        self.mobile_indices = torch.tensor(mobile_indices, dtype=torch.long, device=self.device)
        self.end_effector_indices = torch.tensor(end_effector_indices, dtype=torch.long, device=self.device)

        robot_body_names = [name for name in self.body_names if not self.cfg.asset.keyframe_name in name]
        self.payload_names = [s for s in robot_body_names if self.cfg.asset.payload_name in s]
        self.payload_indices = torch.tensor([get_body_index(n) for n in self.payload_names], dtype=torch.long, device=self.device)
        
        perturb_body_indices = []
        for body_name in self.cfg.asset.perturb_body_names:
            perturb_body_indices += [get_body_index(n) for n in robot_body_names if body_name in n]
        self.perturb_body_indices = torch.tensor(perturb_body_indices, dtype=torch.long, device=self.device)
        
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = torch.tensor(base_init_state_list, dtype=torch.float, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs, self.actor_handles = [], []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone();start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, 
                start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle);self.actor_handles.append(actor_handle)
        
        self.feet_contact_names, self.feet_keyframe_names = [], []
        for key, value in self.cfg.asset.feet_contact_names.items():
            self.feet_contact_names += [s for s in robot_body_names if value in s]
            self.feet_keyframe_names += [s for s in self.keyframe_names if key in s]
            
        self.feet_contact_indices = torch.tensor([get_body_index(n) for n in self.feet_contact_names], dtype=torch.long, device=self.device)
        self.keyframe_indices = torch.tensor([get_body_index(n) for n in self.keyframe_names], dtype=torch.long, device=self.device)
        assert len(self.feet_contact_names) == len(self.feet_keyframe_names)
        
        self.feet_indices = []
        for i, keyframe_name in enumerate(self.keyframe_names):
            add_keyframe_index = lambda ids, names: ids + [i for name in names if name in keyframe_name]
            self.feet_indices = add_keyframe_index(self.feet_indices, self.feet_keyframe_names)
        self.feet_indices = torch.tensor(self.feet_indices, dtype=torch.long, device=self.device)
        assert len(self.feet_indices) == 2, "The number of feet should be 2!!!"

        self.penalised_contact_names, self.terminate_contact_names = [], []
        for name in self.cfg.asset.penalised_contacts_on:
            self.penalised_contact_names += [body for body in robot_body_names if name in body]
        for name in self.cfg.asset.terminate_after_contacts_on:
            self.terminate_contact_names += [body for body in robot_body_names if name in body]
        penalised_contact_indices = [get_body_index(n) for n in self.penalised_contact_names]
        terminate_contact_indices = [get_body_index(n) for n in self.terminate_contact_names]
        self.penalised_contact_indices = torch.tensor(penalised_contact_indices, dtype=torch.long, device=self.device)
        self.terminate_contact_indices = torch.tensor(terminate_contact_indices, dtype=torch.long, device=self.device)

        motion_dof_indices = []
        for name in self.cfg.asset.motion_dof_names:
            motion_dof_indices += [i for i, joint in enumerate(self.dof_names) if name in joint]
        motion_dof_indices = list(set(motion_dof_indices))
        self.motion_dof_indices = torch.tensor(motion_dof_indices, dtype=torch.long, device=self.device)
        
    def _get_env_origins(self):
        """ Sets environment origins.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.terrain_levels = torch.randint(0, self.cfg.terrain.num_rows, (self.num_envs,), device=self.device)
            self.terrain_types = torch.randint(0, self.cfg.terrain.num_cols, (self.num_envs,), device=self.device)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).float()
            self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.env_origins[:, 0:2] += torch_rand_float(-2.0, 2.0, (self.num_envs, 2), device=self.device)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length = np.ceil(np.array(self.cfg.env.max_episode_length_sec) / self.dt)
        self.push_robot_base_interval = np.ceil(np.array(self.cfg.domain_rand.push_robot_base_interval_sec) / self.dt)
        self.push_robot_body_duration = np.ceil(np.array(self.cfg.domain_rand.push_robot_body_duration_sec) / self.dt)
        self.push_robot_body_interval = np.ceil(np.array(self.cfg.domain_rand.push_robot_body_interval_sec) / self.dt)
        self.push_robot_body_interval += self.push_robot_body_duration
        
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws target body position and orientation
        """
        self.gym.clear_lines(self.viewer)
        keyframe_sphere = WireframeSphereGeometry(0.03, 16, 16, None, color=(0, 1, 1))
                
        motion_local_body_pos = quat_rotate_yaw(self.base_quat[:, None], self.motion_local_body_pos)
        motion_local_body_pos += self.base_pos[:, None]
        for i in range(self.num_envs):
            motion_body_pos = motion_local_body_pos[i].cpu().numpy()
            for j in range(len(self.keyframe_indices)):
                x, y, z = motion_body_pos[j, 0], motion_body_pos[j, 1], motion_body_pos[j, 2]
                target_sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(keyframe_sphere, self.gym, self.viewer, self.envs[i], target_sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, self.num_height_points, dtype=torch.float, device=self.device)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids is None:
            points = quat_rotate_yaw(self.base_quat[:, None], self.height_points)
        else:
            points = quat_rotate_yaw(self.base_quat[env_ids, None], self.height_points[env_ids])
            
        points += self.root_states[:, None, 0:3]
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        
        heights = torch.amin(torch.stack([heights1, heights2, heights3], dim=-1), dim=-1)
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_tracking_joint_pos(self):
        diff_dof_pos = (self.dof_pos - self.motion_dof_pos)[:, self.motion_dof_indices]
        diff_dof_pos = torch.norm(diff_dof_pos, dim=1)
        coefficient = -1.0 / self.tracking_sigma["tracking_joint_pos"]
        reward = torch.exp(coefficient * diff_dof_pos)
        return reward, diff_dof_pos

    def _reward_tracking_joint_vel(self):
        diff_dof_vel = (self.dof_vel - self.motion_dof_vel)[:, self.motion_dof_indices]
        diff_dof_vel = torch.norm(diff_dof_vel, dim=1)
        coefficient = -1.0 / self.tracking_sigma["tracking_joint_vel"]
        reward = torch.exp(coefficient * diff_dof_vel)
        return reward, diff_dof_vel

    def _reward_tracking_trunk_lin_vel(self):
        diff_body_lin_vel = self.local_body_lin_vel - self.motion_local_body_lin_vel
        diff_trunk_lin_vel = diff_body_lin_vel[:, self.trunk_indices]
        diff_trunk_lin_vel = torch.norm(diff_trunk_lin_vel, dim=(1, 2))
        coefficient = -1.0 / self.tracking_sigma["tracking_trunk_lin_vel"]
        reward = torch.exp(coefficient * diff_trunk_lin_vel)
        return reward, diff_trunk_lin_vel

    def _reward_tracking_trunk_ang_vel(self):
        diff_body_ang_vel = self.local_body_ang_vel - self.motion_local_body_ang_vel
        diff_trunk_ang_vel = diff_body_ang_vel[:, self.trunk_indices]
        diff_trunk_ang_vel = torch.norm(diff_trunk_ang_vel, dim=(1, 2))
        coefficient = -1.0 / self.tracking_sigma["tracking_trunk_ang_vel"]
        reward = torch.exp(coefficient * diff_trunk_ang_vel)
        return reward, diff_trunk_ang_vel
    
    def _reward_tracking_trunk_vel_dir(self):
        robot_trunk_lin_vel = self.local_body_lin_vel[:, self.trunk_indices]
        robot_trunk_ang_vel = self.local_body_ang_vel[:, self.trunk_indices]

        motion_trunk_lin_vel = self.motion_local_body_lin_vel[:, self.trunk_indices]
        motion_trunk_ang_vel = self.motion_local_body_ang_vel[:, self.trunk_indices]
        
        robot_trunk_vel_dir = torch.cat([
            F.normalize(robot_trunk_lin_vel, dim=2),
            F.normalize(robot_trunk_ang_vel, dim=2),
            ], dim=2)
        motion_trunk_vel_dir = torch.cat([
            F.normalize(motion_trunk_lin_vel, dim=2),
            F.normalize(motion_trunk_ang_vel, dim=2),
            ], dim=2)
        diff_vel_direction = torch.norm(1.0 - F.cosine_similarity(
            robot_trunk_vel_dir, motion_trunk_vel_dir, dim=2), dim=1)
        coefficient = -1.0 / self.tracking_sigma["tracking_trunk_vel_dir"]
        reward = torch.exp(coefficient * diff_vel_direction)
        return reward, diff_vel_direction

    def _reward_tracking_marker_pos(self):
        diff_local_body_pos = self.local_body_pos - self.motion_local_body_pos
        diff_marker_pos = diff_local_body_pos[:, self.marker_indices]
        diff_marker_dist = torch.norm(diff_marker_pos, dim=(1, 2))
        coefficient = -1.0 / self.tracking_sigma["tracking_marker_pos"]
        reward = torch.exp(coefficient * diff_marker_dist)
        return reward, diff_marker_dist
    
    def _reward_tracking_feet_height(self):
        diff_local_body_pos = self.motion_local_body_pos - self.local_body_pos
        diff_feet_height = diff_local_body_pos[:, self.feet_indices, 2]
        diff_feet_height = torch.norm(torch.where(
            torch.any(self.feet_contacts, dim=1),
            diff_feet_height,
            diff_feet_height + self.cfg.rewards.feet_height_target), dim=1)
        coefficient = -1.0 / self.tracking_sigma["tracking_feet_height"]
        reward = torch.exp(coefficient * diff_feet_height)
        return reward, diff_feet_height

    def _reward_tracking_keyframe_height(self):
        diff_body_pos = self.body_pos - self.motion_body_pos
        diff_keyframe_height = diff_body_pos[:, self.end_effector_indices, 2]
        diff_keyframe_height = torch.amax(torch.abs(diff_keyframe_height), dim=1)
        coefficient = -1.0 / self.tracking_sigma["tracking_keyframe_height"]
        reward = torch.exp(coefficient * diff_keyframe_height)
        return reward, diff_keyframe_height

    def _reward_penalty_action_rates(self):
        actions = self.actions[..., self.action_indices]
        action_rates = torch.norm(actions[:, 2] - actions[:, 1], dim=1)
        coefficient = self.penalty_coefficient["penalty_action_rates"]
        return action_rates / coefficient, action_rates
    
    def _reward_penalty_action_smoothness(self):
        actions = self.actions[..., self.smooth_action_indices]
        action_smoothness = torch.norm(
            actions[:, 2] - 2.0 * actions[:, 1] + actions[:, 0], dim=1)
        coefficient = self.penalty_coefficient["penalty_action_smoothness"]
        return action_smoothness / coefficient, action_smoothness
    
    def _reward_penalty_torques(self):
        soft_limits = self.torque_limits * self.cfg.rewards.soft_torques_limit
        torques = (self.torques / soft_limits)[:, self.action_indices]
        torques = torch.norm(torques, dim=1)
        coefficient = self.penalty_coefficient["penalty_torques"]
        return torques / coefficient, torques

    def _reward_penalty_dof_acc(self):
        dof_acc = (self.dof_vel - self.last_dof_vel)[:, self.action_indices]
        dof_acc = torch.norm(dof_acc, dim=1)
        coefficient = self.penalty_coefficient["penalty_dof_acc"]
        return dof_acc / coefficient, dof_acc

    def _reward_feet_slippage(self):
        feet_lin_vel_xy = self.rigid_body_states[:, self.feet_contact_indices, 7:9]
        feet_lin_vel_xy = torch.norm(feet_lin_vel_xy, dim=2)
        feet_contact_masks = torch.any(self.feet_contacts, dim=1).float()
        feet_slippage = torch.sum(feet_lin_vel_xy * feet_contact_masks, dim=1)
        return feet_slippage, feet_slippage
    
    def _reward_feet_stumble(self):
        threshold = 5.0 * torch.abs(self.contact_forces[:, self.feet_contact_indices, 2])
        contact_forces = torch.norm(self.contact_forces[:, self.feet_contact_indices, 0:2], dim=2)
        feet_stumble = torch.any(contact_forces > threshold, dim=1).float()
        return feet_stumble, feet_stumble

    def _reward_feet_air_time(self):
        air_time_target = self.cfg.rewards.feet_air_time_target
        feet_air_time = torch.clip(1.0 - self.feet_air_time / air_time_target, min=0.0)
        feet_air_time = torch.sum(torch.where(self.feet_first_contact, feet_air_time, 0.0), dim=1)
        feet_on_fly = torch.all(~self.feet_contacts, dim=1)
        self.feet_air_time = torch.where(feet_on_fly, self.feet_air_time, 0.0)
        return feet_air_time, feet_air_time
    
    def _reward_feet_distance(self):
        xy_dist_target = self.cfg.rewards.feet_xy_dist_target
        feet_pos = self.local_body_pos[:, self.feet_indices]
        feet_distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)
        feet_distance = torch.clip(1.0 - feet_distance / xy_dist_target, min=0.0)
        return feet_distance, feet_distance

    def _reward_deviation(self):
        return self.deviation_distance, self.deviation_distance

    def _reward_alive(self):
        alive = 1.0 - (self.reset_buf | self.deviation).float()
        return alive, alive

    def _reward_termination(self):
        termination = (~self.time_out_buf & self.reset_buf).float()
        return termination, termination

    def _reward_collisions(self):
        collision_force = self.contact_forces[:, self.penalised_contact_indices]
        collision = torch.sum((torch.norm(collision_force, dim=2) > 1.0).float(), dim=1)
        return collision, collision
    
    def _reward_action_limits(self):
        out_of_lower_limits = (self.dof_pos <= self.dof_pos_limits[:, 0]).float()
        out_of_upper_limits = (self.dof_pos >= self.dof_pos_limits[:, 1]).float()
        soft_limits = self.torque_limits * self.cfg.rewards.soft_torques_limit
        torques = self.torques / soft_limits
        out_of_limits = torch.clip(-1.0 * out_of_lower_limits * torques, min=0.0)
        out_of_limits += torch.clip(1.0 * out_of_upper_limits * torques, min=0.0)
        out_of_limits = torch.sum(out_of_limits[:, self.action_indices], dim=1)
        return out_of_limits, out_of_limits

    def _reward_torque_limits(self):
        soft_limits = self.torque_limits * self.cfg.rewards.soft_torques_limit
        out_of_limits = torch.clip(torch.abs(self.torques) - soft_limits, min=0.0)
        out_of_limits = out_of_limits / soft_limits    
        out_of_limits = torch.sum(out_of_limits[:, self.action_indices], dim=1)
        return out_of_limits, out_of_limits

    def _reward_dof_pos_limits(self):
        out_of_limits = torch.clip(self.dof_pos_limits[:, 0] - self.dof_pos, min=0.0)
        out_of_limits += torch.clip(self.dof_pos - self.dof_pos_limits[:, 1], min=0.0)
        out_of_limits = torch.sum(out_of_limits[:, self.action_indices], dim=1)
        return out_of_limits, out_of_limits

    def _reward_dof_vel_limits(self):
        soft_limits = self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
        out_of_limits = torch.clip(torch.abs(self.dof_vel) - soft_limits, min=0.0)
        out_of_limits = torch.sum(out_of_limits[:, self.action_indices], dim=1)
        return out_of_limits, out_of_limits
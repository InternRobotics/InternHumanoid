import os
import math
import torch
import random
import pickle
import datetime
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from .math import (
    normalize,
    euler_xyz_to_quat,
    quat_mul,
    quat_conjugate,
    quat_mul_yaw_inverse,
    quat_to_angle_axis,
    quat_to_euler_xyz,
)


def load_imitation_dataset(folder, mapping="joint_id.txt", suffix=".pkl"):
    filenames = [name for name in os.listdir(folder) if name[-len(suffix):] == suffix]

    dataset = {}
    print("Loading motion dataset...")
    for filename in tqdm(filenames):
        try:
            file = open(os.path.join(folder, filename), 'rb')
            data = pickle.load(file)
            for key, value in data.items():
                dataset[key] = value
        except:
            print(f"{filename} load failed!!!")
            continue
    
    dataset_names = [k for k in dataset.keys()]
    dataset_list = [v for v in dataset.values()]
        
    lines = open(mapping).readlines()
    lines = [line[:-1].split(" ") for line in lines]
    joint_id_dict = {k: int(v) for v, k in lines}
    return dataset_list, dataset_names, joint_id_dict


def filter_and_fix_motion(
    datasets, 
    data_names,
    min_time,
    link_height_offset,
    floor_height_offset,
    base_height_range=None,
    base_roll_range=None,
    base_pitch_range=None,
    ):
    legal_datasets, legal_names, total_length, total_time = [], [], 0, 0.0
    
    print("Fixing heights...")
    for data, name in zip(datasets, data_names):
        data["base_position"][:, 2] += floor_height_offset
        for link_name in data["link_position"].keys():
            data["link_position"][link_name][:, 2] += floor_height_offset
            data["link_position"][link_name][:, 2] += link_height_offset

    print("Fixing NaN values...")
    for data, name in zip(datasets, data_names):
        num_frame = data["base_orientation"].shape[0]
        
        for n in data["link_orientation"].keys():
            nan_ids = np.nonzero(np.isnan(data["link_orientation"][n]))
            temp_ids, coor_ids = nan_ids

            if len(temp_ids) > 0:
                print(f"Nan values found in {name}: {temp_ids}")

            for i in range(len(temp_ids)):
                curr_ids = temp_ids[i].item()
                orientation = data["link_orientation"][n][:, coor_ids[i]]
                
                if curr_ids == 0:
                    prev_ids = post_ids = 1
                elif curr_ids == num_frame - 1:
                    prev_ids = post_ids = num_frame - 2
                else:
                    prev_ids, post_ids = curr_ids - 1, curr_ids + 1
                    
                fill_value = (orientation[prev_ids] + orientation[post_ids]) * 0.5
                data["link_orientation"][n][temp_ids[i], coor_ids[i]] = fill_value

    print("Filtering motion dataset...")
    for data, name in zip(datasets, data_names):
        base_position = torch.tensor(data["base_position"], dtype=torch.float)
        base_orientation = torch.tensor(data["base_orientation"], dtype=torch.float)
        
        last_ids = torch.tensor([base_position.shape[0]], dtype=torch.long)
        
        min_height_ids, max_height_ids = last_ids, last_ids
        min_base_roll_ids, max_base_roll_ids = last_ids, last_ids
        min_base_pitch_ids, max_base_pitch_ids = last_ids, last_ids
        
        if base_height_range is not None:
            min_height_ids = (base_position[:, 2] < min(base_height_range)).nonzero().flatten()
            max_height_ids = (base_position[:, 2] > max(base_height_range)).nonzero().flatten()
            
        if base_roll_range is not None or base_pitch_range is not None:
            motion_base_quat = euler_xyz_to_quat(base_orientation)
            motion_base_rpy = quat_to_euler_xyz(quat_mul_yaw_inverse(motion_base_quat, motion_base_quat))
            
            if base_roll_range is not None:
                min_base_roll_ids = (motion_base_rpy[:, 0] < min(base_roll_range)).nonzero().flatten()
                max_base_roll_ids = (motion_base_rpy[:, 0] > max(base_roll_range)).nonzero().flatten()

            if base_pitch_range is not None:
                min_base_pitch_ids = (motion_base_rpy[:, 1] < min(base_pitch_range)).nonzero().flatten()
                max_base_pitch_ids = (motion_base_rpy[:, 1] > max(base_pitch_range)).nonzero().flatten()

        illegal_id_list = torch.cat([min_height_ids, min_base_roll_ids, min_base_pitch_ids, 
                                     max_height_ids, max_base_roll_ids, max_base_pitch_ids,], dim=0)
        
        if len(illegal_id_list) > 0:
            first_illegal_ids = torch.amin(illegal_id_list).item()
            if first_illegal_ids > max(math.ceil(min_time * data["framerate"]), 3):
                data["base_position"] = data["base_position"][1:first_illegal_ids]
                data["base_orientation"] = data["base_orientation"][1:first_illegal_ids]
                data["joint_position"] = data["joint_position"][1:first_illegal_ids]
                    
                for n in data["link_position"].keys():
                    data["link_position"][n] = data["link_position"][n][1:first_illegal_ids]
                    data["link_orientation"][n] = data["link_orientation"][n][1:first_illegal_ids]
                
                legal_datasets += [data]
                legal_names += [name]
                total_length += first_illegal_ids - 1
                total_time += (first_illegal_ids - 1) / data["framerate"]
        else:
            if len(data["base_position"]) > max(math.ceil(min_time * data["framerate"]), 3):
                data["base_position"] = data["base_position"][1:]
                data["base_orientation"] = data["base_orientation"][1:]
                data["joint_position"] = data["joint_position"][1:]
                        
                for n in data["link_position"].keys():
                    data["link_position"][n] = data["link_position"][n][1:]
                    data["link_orientation"][n] = data["link_orientation"][n][1:]
                
                legal_datasets += [data]
                legal_names += [name]
                total_length += data["base_position"].shape[0]
                total_time += data["base_position"].shape[0] / data["framerate"]
    
    print("Number of legal motion dataset: ", len(legal_datasets))
    print("Total frame number: ", total_length)
    print("Total time: ", str(datetime.timedelta(seconds=total_time)))
    return legal_datasets, legal_names
        
        
class MotionLib:
    def __init__(self, datasets, data_names, mapping, dof_names, body_names, device="cpu"):
        self.device = device
        self.data_names = data_names
        
        length = [data["base_position"].shape[0] - 1 for data in datasets]
        self.length = torch.tensor(length, dtype=torch.long, device=self.device)
        self.num_motion, self.total_length = self.length.shape[0], self.length.sum()
        
        self.fps = datasets[0]["framerate"]
        self.dt = 1.0 / self.fps
        
        self.end_indices = torch.cumsum(self.length, dim=0)
        self.start_indices = torch.nn.functional.pad(self.end_indices, (1, -1), "constant", 0)
        
        self.base_rpy = torch.zeros(self.total_length, 3, dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros(self.total_length, 3, dtype=torch.float, device=self.device)
        self.base_lin_vel = torch.zeros(self.total_length, 3, dtype=torch.float, device=self.device)
        self.base_ang_vel = torch.zeros(self.total_length, 3, dtype=torch.float, device=self.device)
        self.dof_pos = torch.zeros(self.total_length, len(dof_names), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros(self.total_length, len(dof_names), dtype=torch.float, device=self.device)
        self.body_pos = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=self.device)
        self.body_rpy = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=self.device)
        self.body_lin_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=self.device)
        self.body_ang_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=self.device)
        
        def compute_linear_velocity(xyz):
            return (xyz[1:] - xyz[:-1]) * self.fps
        
        def compute_angular_velocity(rpy):
            quat = euler_xyz_to_quat(rpy)
            diff_quat = quat_mul(quat[1:], quat_conjugate(quat[:-1]))
            angle, axis = quat_to_angle_axis(diff_quat)
            return axis * angle[..., None] * self.fps

        def compute_joint_velocity(q):
            return (q[1:] - q[:-1]) * self.fps
        
        print(f"Moving motion dataset to {self.device}...")
        for i, data in enumerate(tqdm(datasets)):
            start, end = self.start_indices[i], self.end_indices[i]
            base_pos = torch.tensor(data["base_position"], dtype=torch.float, device=self.device)
            base_rpy = torch.tensor(data["base_orientation"], dtype=torch.float, device=self.device)
                        
            self.base_pos[start:end] = base_pos[:self.length[i]]
            self.base_rpy[start:end] = base_rpy[:self.length[i]]
            self.base_lin_vel[start:end] = compute_linear_velocity(base_pos)[:self.length[i]]
            self.base_ang_vel[start:end] = compute_angular_velocity(base_rpy)[:self.length[i]]
            
            dof_pos = torch.tensor(data["joint_position"], dtype=torch.float, device=self.device)
            dof_vel = compute_joint_velocity(dof_pos)
            for j, name in enumerate(dof_names):
                if name in mapping.keys():
                    self.dof_pos[start:end, j] = dof_pos[:self.length[i], mapping[name]]
                    self.dof_vel[start:end, j] = dof_vel[:self.length[i], mapping[name]]

            for k, name in enumerate(body_names):
                body_pos = torch.tensor(data["link_position"][name], dtype=torch.float, device=self.device)
                body_rpy = torch.tensor(data["link_orientation"][name], dtype=torch.float, device=self.device)
                
                self.body_pos[start:end, k] = body_pos[:self.length[i]]
                self.body_rpy[start:end, k] = body_rpy[:self.length[i]]
                self.body_lin_vel[start:end, k] = compute_linear_velocity(body_pos)[:self.length[i]]
                self.body_ang_vel[start:end, k] = compute_angular_velocity(body_rpy)[:self.length[i]]
        
        # flush original datasets
        del datasets
        
        # hyper parameters
        self.default_distance = 1.5 # m
        self.info_decay_rate = 0.9 # decay rate of imitation info
        self.sampling_ratio = 0.5 # ratio between failure and distance
        self.init_epsilon = 0.05 # initial exploration rate
        self.uniform_init_epsilon = 0.5 # uniform exploration rate

        # priority scheduler
        self.random_start_times = torch.clone((self.length - 2) * self.dt)
        self.num_failure = torch.zeros(self.num_motion, dtype=torch.float, device=self.device)
        self.motion_score = torch.zeros(self.num_motion, dtype=torch.float, device=self.device)
        self.motion_score += self.default_distance
        
        self.temp_failure = torch.zeros(self.num_motion, dtype=torch.float, device=self.device)
        self.temp_score = torch.zeros(self.num_motion, dtype=torch.float, device=self.device)

    def check_success(self, motion_ids, motion_times):
        motion_step = torch.ceil(motion_times * self.fps)
        return motion_step >= (self.length[motion_ids] - 1)
    
    def sample_motions(self, num_samples, uniform=False):
        if uniform:
            return torch.randint(self.num_motion, (num_samples,), device=self.device)
        
        max_failure = torch.amax(torch.clip(self.num_failure, min=1e-4), dim=0)
        failure_score = self.num_failure / max_failure * self.sampling_ratio
        
        max_distance = torch.amax(torch.clip(self.motion_score, min=1e-4), dim=0)
        distance_score = self.motion_score / max_distance * (1.0 - self.sampling_ratio)
        
        sampling_weight = torch.clip(failure_score + distance_score, min=1e-4)
        return torch.multinomial(sampling_weight.to(self.device), num_samples=num_samples, replacement=True)
    
    def sample_init_time(self, motion_ids, uniform=False):
        phase = torch.rand(motion_ids.shape, dtype=torch.float, device=self.device)
        phase = torch.clip(phase + (phase - 1) * self.init_epsilon, min=0.0, max=1.0)
        
        motion_total_time = self.get_motion_time(motion_ids)
        if uniform:
            return phase * motion_total_time
        else:
            return phase * torch.where(
                torch.rand_like(phase) < self.uniform_init_epsilon,
                motion_total_time,
                self.random_start_times[motion_ids],
            )

    def update_imitation_info(self, motion_ids, init_time, motion_time, success, max_distance):
        success_ids = success.nonzero().flatten()
        if len(success_ids) > 0:
            self.random_start_times.scatter_reduce_(
                dim=0, index=motion_ids[success_ids],
                src=init_time[success_ids],
                reduce="amin", include_self=True,)
        
        unique_motion_ids = torch.unique(motion_ids)
        self.temp_failure.scatter_reduce_(
            dim=0, index=motion_ids,
            src=(~success).float(),
            reduce="mean", include_self=False,
            )
        
        motion_time = self.get_motion_time(motion_ids) - init_time
        completion = (motion_time - init_time) / torch.clip(motion_time, min=1e-3)
        motion_score = completion * max_distance + (1.0 - completion) * self.default_distance
        self.temp_score.scatter_reduce_(
            dim=0, index=motion_ids,
            src=motion_score,
            reduce="mean", include_self=False,
            )
        
        self.num_failure[unique_motion_ids] *= self.info_decay_rate
        new_failure = self.temp_failure[unique_motion_ids]
        new_failure *= (1.0 - self.info_decay_rate)
        self.num_failure[unique_motion_ids] += new_failure
        
        self.motion_score[unique_motion_ids] *= self.info_decay_rate
        new_distance = self.temp_score[unique_motion_ids]
        new_distance *= (1.0 - self.info_decay_rate)
        self.motion_score[unique_motion_ids] += new_distance

    def get_imitation_info(self):
        length, completion = self.length - 1, self.get_completion()
        completion = torch.sum(completion * length) / torch.sum(length)
        success_rate = (self.random_start_times < self.dt).float()
        success_rate = torch.sum(success_rate / self.num_motion)
        return completion, success_rate

    def get_motion_time(self, motion_ids):
        return (self.length[motion_ids] - 1) * self.dt
    
    def get_completion(self, motion_ids=None):
        motion_time = (self.length - 1) * self.dt
        completion = (motion_time - self.random_start_times) / motion_time
        return completion if motion_ids is None else completion[motion_ids]
        
    def get_motion_states(self, motion_ids, motion_times):
        motion_start_ids = self.start_indices[motion_ids]
        
        frames = torch.clip(motion_times * self.fps, 
            min=torch.zeros_like(motion_times), max=self.length[motion_ids] - 2)
        floors = torch.floor(frames).long()
        blend_motion_lerp = lambda x: self.calc_lerp(
            x, motion_start_ids + floors, frames - floors)
        blend_motion_elerp = lambda x: self.calc_elerp(
            x, motion_start_ids + floors, frames - floors)
        
        return dict(
            base_pos=blend_motion_lerp(self.base_pos), 
            base_quat=blend_motion_elerp(self.base_rpy),
            base_lin_vel=blend_motion_lerp(self.base_lin_vel),
            base_ang_vel=blend_motion_lerp(self.base_ang_vel),
            
            dof_pos=blend_motion_lerp(self.dof_pos), 
            dof_vel=blend_motion_lerp(self.dof_vel),
            
            body_pos=blend_motion_lerp(self.body_pos), 
            body_quat=blend_motion_elerp(self.body_rpy),
            body_lin_vel=blend_motion_lerp(self.body_lin_vel), 
            body_ang_vel=blend_motion_lerp(self.body_ang_vel),
            )
    
    @staticmethod
    def calc_lerp(motion, frame, t):
        motion0, motion1 = motion[frame], motion[frame + 1]
        shape = torch.Size(t.shape + (1,) * (motion0.dim() - t.dim()))
        return (1 - t).view(shape) * motion0 + t.view(shape) * motion1
    
    @staticmethod
    def calc_elerp(motion, frame, t):
        motion0, motion1 = motion[frame], motion[frame + 1]
        shape = torch.Size(t.shape + (1,) * (motion0.dim() - t.dim()))
        diff_motion = motion1 - motion0
        diff_motion = (diff_motion + math.pi) % (2 * math.pi) - math.pi
        return euler_xyz_to_quat(motion0 + t.view(shape) * diff_motion)

    @staticmethod
    def calc_slerp(motion, frame, t):
        motion0 = euler_xyz_to_quat(motion[frame])
        motion1 = euler_xyz_to_quat(motion[frame + 1])
        
        shape = torch.Size(t.shape + (1,) * (motion0.dim() - t.dim()))
        w0, w1 = (1 - t).view(shape), t.view(shape)
        
        cosine = F.cosine_similarity(motion0, motion1, dim=-1)
        cosine = cosine[..., None]
        motion1 = torch.where(cosine < 0.0, -motion1, motion1)
        
        cosine = torch.abs(cosine)
        theta = torch.clip(torch.acos(cosine), min=1e-3)
        sin_theta_recip = 1.0 / torch.sin(theta)

        s0 = torch.sin(theta * w0) * sin_theta_recip
        s1 = torch.sin(theta * w1) * sin_theta_recip
        return normalize(torch.where(cosine > 0.9995,
            w0 * motion0 + w1 * motion1, s0 * motion0 + s1 * motion1))
        

class MotionLoader:
    def __init__(self, datasets, data_names, mapping, dof_names, body_names, sim_dt, device="cpu"):
        self.device = device
        self.data_names = data_names
        self.function = None  # initialize function later
        
        origin_fps = datasets[0]["framerate"]
        origin_length = torch.tensor([data["base_position"].shape[0] - 1 for data in datasets], dtype=torch.long)
        self.length = torch.floor(origin_length / (origin_fps * sim_dt)).long()
        
        self.num_motion = self.length.shape[0]
        self.total_length = self.length.sum()
        
        self.end_indices = torch.cumsum(self.length, dim=0)
        self.start_indices = torch.nn.functional.pad(self.end_indices, (1, -1), "constant", 0)
        
        self.base_pos = torch.zeros(self.total_length, 3, dtype=torch.float)
        self.base_quat = torch.zeros(self.total_length, 4, dtype=torch.float)
        self.base_lin_vel = torch.zeros(self.total_length, 3, dtype=torch.float)
        self.base_ang_vel = torch.zeros(self.total_length, 3, dtype=torch.float)
        self.dof_pos = torch.zeros(self.total_length, len(dof_names), dtype=torch.float)
        self.dof_vel = torch.zeros(self.total_length, len(dof_names), dtype=torch.float)
        self.body_pos = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float)
        self.body_quat = torch.zeros(self.total_length, len(body_names), 4, dtype=torch.float)
        self.body_lin_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float)
        self.body_ang_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float)
        
        def compute_linear_velocity(xyz):
            return (xyz[1:] - xyz[:-1]) * origin_fps
        
        def compute_angular_velocity(rpy):
            quat = euler_xyz_to_quat(rpy)
            diff_quat = quat_mul(quat[1:], quat_conjugate(quat[:-1]))
            angle, axis = quat_to_angle_axis(diff_quat)
            return axis * angle[..., None] * origin_fps
        
        def compute_joint_velocity(q):
            return (q[1:] - q[:-1]) * origin_fps
        
        print(f"Moving motion dataset to {self.device}...")
        for i, data in enumerate(tqdm(datasets)):
            start, end = self.start_indices[i], self.end_indices[i]
            motion_times = torch.arange(0, self.length[i], dtype=torch.float) * sim_dt
            timesteps = torch.minimum(motion_times * origin_fps, origin_length[i] - 2)
            timesteps_floor = torch.floor(timesteps).long()
            
            blend_motion_lerp = lambda x: self.calc_lerp(x, timesteps_floor, timesteps - timesteps_floor)
            blend_motion_elerp = lambda x: self.calc_elerp(x, timesteps_floor, timesteps - timesteps_floor)
            
            base_pos = torch.tensor(data["base_position"], dtype=torch.float)
            base_rpy = torch.tensor(data["base_orientation"], dtype=torch.float)
            base_lin_vel = compute_linear_velocity(base_pos)
            base_ang_vel = compute_angular_velocity(base_rpy)
            
            self.base_pos[start:end] = blend_motion_lerp(base_pos)[:self.length[i]]
            self.base_quat[start:end] = blend_motion_elerp(base_rpy)[:self.length[i]]
            self.base_lin_vel[start:end] = blend_motion_lerp(base_lin_vel)[:self.length[i]]
            self.base_ang_vel[start:end] = blend_motion_lerp(base_ang_vel)[:self.length[i]]
            
            dof_pos = torch.tensor(data["joint_position"], dtype=torch.float)
            dof_vel = compute_joint_velocity(dof_pos)
            for j, name in enumerate(dof_names):
                if name in mapping.keys():
                    self.dof_pos[start:end, j] = blend_motion_lerp(dof_pos)[:self.length[i], mapping[name]]
                    self.dof_vel[start:end, j] = blend_motion_lerp(dof_vel)[:self.length[i], mapping[name]]

            for k, name in enumerate(body_names):
                body_pos = torch.tensor(data["link_position"][name], dtype=torch.float)
                body_rpy = torch.tensor(data["link_orientation"][name], dtype=torch.float)
                body_lin_vel = compute_linear_velocity(body_pos)
                body_ang_vel = compute_angular_velocity(body_rpy)
                
                self.body_pos[start:end, k] = blend_motion_lerp(body_pos)[:self.length[i]]
                self.body_quat[start:end, k] = blend_motion_elerp(body_rpy)[:self.length[i]]
                self.body_lin_vel[start:end, k] = blend_motion_lerp(body_lin_vel)[:self.length[i]]
                self.body_ang_vel[start:end, k] = blend_motion_lerp(body_ang_vel)[:self.length[i]]
        
        # flush
        del datasets
        
    def load_states_function(self, function):
        self.function = function

    @staticmethod
    def calc_lerp(motion, frame, t):
        motion0, motion1 = motion[frame], motion[frame + 1]
        shape = torch.Size(t.shape + (1,) * (motion0.dim() - t.dim()))
        return (1 - t).view(shape) * motion0 + t.view(shape) * motion1

    @staticmethod
    def calc_elerp(motion, frame, t):
        motion0, motion1 = motion[frame], motion[frame + 1]
        shape = torch.Size(t.shape + (1,) * (motion0.dim() - t.dim()))
        diff_motion = motion1 - motion0
        diff_motion = (diff_motion + math.pi) % (2 * math.pi) - math.pi
        return euler_xyz_to_quat(motion0 + t.view(shape) * diff_motion)

    @staticmethod
    def calc_slerp(motion, frame, t):
        motion0 = euler_xyz_to_quat(motion[frame])
        motion1 = euler_xyz_to_quat(motion[frame + 1])
        
        shape = torch.Size(t.shape + (1,) * (motion0.dim() - t.dim()))
        w0, w1 = (1 - t).view(shape), t.view(shape)
        
        cosine = F.cosine_similarity(motion0, motion1, dim=-1)
        cosine = cosine[..., None]
        motion1 = torch.where(cosine < 0.0, -motion1, motion1)
        
        cosine = torch.abs(cosine)
        theta = torch.clip(torch.acos(cosine), min=1e-3)
        sin_theta_recip = 1.0 / torch.sin(theta)

        s0 = torch.sin(theta * w0) * sin_theta_recip
        s1 = torch.sin(theta * w1) * sin_theta_recip
        return normalize(torch.where(cosine > 0.9995,
            w0 * motion0 + w1 * motion1, s0 * motion0 + s1 * motion1))
        
    def get_motion_states(self, motion_frames):                
        return dict(
            base_pos=self.base_pos[motion_frames], 
            base_quat=self.base_quat[motion_frames],
            base_lin_vel=self.base_lin_vel[motion_frames],
            base_ang_vel=self.base_ang_vel[motion_frames],
            
            dof_pos=self.dof_pos[motion_frames], 
            dof_vel=self.dof_vel[motion_frames],
            
            body_pos=self.body_pos[motion_frames], 
            body_quat=self.body_quat[motion_frames],
            body_lin_vel=self.body_lin_vel[motion_frames], 
            body_ang_vel=self.body_ang_vel[motion_frames],
            )

    def mini_batch_generator(self, num_mini_batches, num_epochs=8, mini_batch_size=512):
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                motion_ids_batch = torch.randint(self.num_motion, (mini_batch_size,))
                motion_phase_batch = torch.rand(motion_ids_batch.shape, dtype=torch.float)
                motion_indices = torch.floor(motion_phase_batch * (self.length[motion_ids_batch] - 2)).long()
                motion_indices_batch = torch.stack([motion_indices, motion_indices + 1], dim=1)
                state_batch, next_state_batch = self.function(self.get_motion_states(motion_indices_batch))
                yield state_batch.to(self.device), next_state_batch.to(self.device)

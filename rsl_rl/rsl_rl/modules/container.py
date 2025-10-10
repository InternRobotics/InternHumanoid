#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import LSTMCell, GRUCell
from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent


class Container(nn.Module):
    is_recurrent = False
    
    def __init__(self, actor_critic: ActorCritic, num_envs):
        super().__init__()
        self.noise_mode = actor_critic.noise_mode
        self.actor = nn.Sequential(
            actor_critic.actor,
            actor_critic.actor_head,
            )
        self.normalizer1 = actor_critic.normalizers["actor"]
        self.normalizer2 = actor_critic.normalizers["command"]
        
    def forward(self, observations, commands):
        concate_observations = torch.cat([
            self.normalizer1(observations),
            self.normalizer2(commands),], dim=-1)
        
        action_mean = self.actor(concate_observations)
        if self.noise_mode == "policy":
            return torch.chunk(action_mean, 2, dim=-1)[0]
        else:
            return action_mean
    

class RecurrentContainer(nn.Module):
    is_recurrent = True
    
    def __init__(self, actor_critic: ActorCriticRecurrent, num_envs):
        super().__init__()
        self.noise_mode = actor_critic.noise_mode
        self.actor = nn.Sequential(
            actor_critic.actor,
            actor_critic.actor_head,
            )
        self.memory = actor_critic.memory_a
        self.is_lstm = isinstance(self.memory, LSTMCell)
        
        self.normalizer0 = actor_critic.normalizers["memory"]
        self.normalizer1 = actor_critic.normalizers["actor"]
        self.normalizer2 = actor_critic.normalizers["command"]
        
        hidden_dim = self.memory.hidden_dim
        self.register_buffer("hidd", torch.zeros(num_envs, hidden_dim, dtype=torch.float))
        self.register_buffer("cell", torch.zeros(num_envs, hidden_dim, dtype=torch.float))
    
    @torch.jit.export
    def reset(self, num_batch, masks):
        self.hidd[:num_batch][masks] = 0.0
        self.cell[:num_batch][masks] = 0.0
        
    def forward(self, observations, commands, resets):
        num_batch = observations.shape[0]
        self.reset(num_batch, resets)
        
        if self.is_lstm:
            hidden_states = (
                self.hidd[:num_batch], 
                self.cell[:num_batch],
                )
        else:
            hidden_states = self.hidd[:num_batch]
        
        memories, next_hidden_states = self.memory(
            self.normalizer0(observations), hidden_states)
        
        if self.is_lstm:
            self.hidd[:num_batch] = next_hidden_states[0]
            self.cell[:num_batch] = next_hidden_states[1]
        else:
            self.hidd[:num_batch] = next_hidden_states
        
        concate_observations = torch.cat([
            self.normalizer1(memories),
            self.normalizer2(commands),], dim=-1)
        
        action_mean = self.actor(concate_observations)
        if self.noise_mode == "policy":
            return torch.chunk(action_mean, 2, dim=-1)[0]
        else:
            return action_mean

class HIMContainer(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = actor_critic.actor
        self.estimator = actor_critic.estimator.encoder

    def forward(self, obs_history):
        parts = self.estimator(obs_history)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, -82:], vel, z), dim=1))
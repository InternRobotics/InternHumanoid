#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
from functools import reduce
from torch.distributions import Normal
from .normalization import Normalization
from .network import (
    HyperEmbedder, 
    HyperLERPBlock, 
    HyperPolicyHead, 
    HyperValueHead,
    )


class ActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_commands,
        num_critic_commands,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        init_std=1.0,
        noise_mode="constant",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        super().__init__()        
        mlp_input_dim_a = num_actor_obs + num_commands
        mlp_input_dim_c = num_critic_obs + num_critic_commands
        
        self.init_std = init_std
        self.noise_mode = noise_mode
        
        # Action noise
        zero_actions = torch.zeros(num_actions)
        if self.noise_mode == "constant":
            self.register_buffer("std", init_std + zero_actions)
        elif self.noise_mode == "learnable":
            self.std = nn.Parameter(init_std + zero_actions)
        elif self.noise_mode == "policy":
            self.std = None
            num_actions *= 2   # mean and std
        else:
            raise ValueError(f"Invalid noise mode: {self.noise_mode}")

        # Normalizer
        self.normalizers = nn.ModuleDict({
            "actor": Normalization(shape=[num_actor_obs]),
            "critic": Normalization(shape=[num_critic_obs]),
            "command": Normalization(shape=[num_commands]),
            "critic_command": Normalization(shape=[num_critic_commands]),
        })
        
        # Policy
        num_blocks = len(actor_hidden_dims)
        actor_hidden_dim = reduce(math.gcd, actor_hidden_dims)
        actor_layers = [
            HyperEmbedder(
                mlp_input_dim_a, actor_hidden_dim,
                init=math.sqrt(2.0 / actor_hidden_dim),
                scale=math.sqrt(2.0 / actor_hidden_dim),
            )]
        for index in range(len(actor_hidden_dims) - 1):
            dimensions = math.sqrt(actor_hidden_dims[index])
            actor_layers += [
                HyperLERPBlock(
                    actor_hidden_dim,
                    actor_hidden_dims[index] // actor_hidden_dim,
                    scaler_init=math.sqrt(2.0 / actor_hidden_dim),
                    scaler_scale=math.sqrt(2.0 / actor_hidden_dim),
                    alpha_init=1.0 / num_blocks,
                    alpha_scale=1.0 / dimensions,
                    )]
        self.actor = nn.Sequential(*actor_layers)
        self.actor_head = HyperPolicyHead(actor_hidden_dim, num_actions)

        # Value function
        num_blocks = len(critic_hidden_dims)
        critic_hidden_dim = reduce(math.gcd, critic_hidden_dims)
        critic_layers = [
            HyperEmbedder(
                mlp_input_dim_c, critic_hidden_dim,
                init=math.sqrt(2.0 / critic_hidden_dim),
                scale=math.sqrt(2.0 / critic_hidden_dim),
            )]
        for index in range(len(critic_hidden_dims) - 1):
            dimensions = math.sqrt(critic_hidden_dims[index])
            critic_layers += [
                HyperLERPBlock(
                    critic_hidden_dim,
                    critic_hidden_dims[index] // critic_hidden_dim,
                    scaler_init=math.sqrt(2.0 / critic_hidden_dim),
                    scaler_scale=math.sqrt(2.0 / critic_hidden_dim),
                    alpha_init=1.0 / num_blocks,
                    alpha_scale=1.0 / dimensions,
                    )]
        self.critic = nn.Sequential(*critic_layers)
        self.critic_head = HyperValueHead(critic_hidden_dim)
        
        self.distribution = None
        
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        return
    
    def get_hidden_states(self):
        return

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, concate_observations):
        outputs = self.actor(concate_observations)
        if self.noise_mode == "policy":
            outputs = self.actor_head(outputs)
            mean, log_std = torch.chunk(outputs, 2, dim=-1)
            std = torch.clip(log_std, min=-5.0, max=1.0).exp()
        else:
            mean = self.actor_head(outputs)
            std = torch.abs(self.std.expand_as(mean))
            std = torch.clip(std, min=1e-3)
        self.distribution = Normal(mean, std)
        
    def act(self, observations, commands, **kwargs):
        observations = self.normalizers["actor"](observations)
        commands = self.normalizers["command"](commands)
        concate_observations = torch.cat([observations, commands], dim=-1)
        self.update_distribution(concate_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return torch.sum(self.distribution.log_prob(actions), dim=-1)

    def act_inference(self, observations, commands, **kwargs):
        observations = self.normalizers["actor"](observations)
        commands = self.normalizers["command"](commands)
        concate_observations = torch.cat([observations, commands], dim=-1)
        outputs = self.actor_head(self.actor(concate_observations))
        
        if self.noise_mode == "policy":
            action_mean = torch.chunk(outputs, 2, dim=-1)[0]
        else:
            action_mean = outputs
        return action_mean, None

    def evaluate(self, critic_observations, critic_commands, **kwargs):
        observations = self.normalizers["critic"](critic_observations)
        commands = self.normalizers["critic_command"](critic_commands)
        concate_observations = torch.cat([observations, commands], dim=-1)
        return self.critic_head(self.critic(concate_observations))
        
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "mish":
        return nn.Mish()
    else:
        print("invalid activation function!")
        return None

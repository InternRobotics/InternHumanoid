#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from .actor_critic import get_activation
from .normalization import Normalization


class Discriminator(nn.Module):
    def __init__(
        self, 
        num_states,
        disc_hidden_dims=[512, 256],
        gan_type="lsgan",
        activation="elu",
        task_reward_lerp=0.5,
        style_reward_scale=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "Discriminator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
            
        super().__init__()
        self.gan_type = gan_type.lower()
        if self.gan_type == "lsgan":
            self.loss_function = nn.MSELoss(reduce="mean")
        elif self.gan_type == "bcegan":
            self.loss_function = nn.BCEWithLogitsLoss(reduce="mean")
        elif self.gan_type == "wasserstein":
            self.loss_function = lambda src, target: torch.mean(src * target)
        else:
            raise ValueError(f"Invalid discriminator type: {self.gan_type}")
        
        activation = get_activation(activation)
        self.task_reward_lerp = task_reward_lerp
        self.style_reward_scale = style_reward_scale
        if self.gan_type == "wasserstein":
            self.reward_normalizer = Normalization(shape=[1])
        else:
            self.reward_normalizer = nn.Identity()
        
        # Discriminator
        disc_layers = [nn.Linear(num_states * 2, disc_hidden_dims[0]), activation]
        for index in range(len(disc_hidden_dims) - 1):
            disc_layers += [
                nn.Linear(disc_hidden_dims[index], disc_hidden_dims[index + 1]), activation]
        self.trunk = nn.Sequential(*disc_layers)
        self.output = nn.Linear(disc_hidden_dims[-1], 1)

    def forward(self, states, next_states):
        concate_states = torch.cat([states, next_states], dim=-1)
        logits = self.output(self.trunk(concate_states))
        return logits

    def compute_policy_loss(self, states, next_states):
        logits = self.forward(states, next_states)
        if self.gan_type == "lsgan":
            return self.loss_function(logits, -1.0 * torch.ones_like(logits))
        elif self.gan_type == "bcegan":
            return self.loss_function(logits, torch.zeros_like(logits))
        elif self.gan_type == "wasserstein":
            return self.loss_function(logits, torch.ones_like(logits))
        else:
            raise ValueError(f"Invalid discriminator type: {self.gan_type}")

    def compute_expert_loss(self, states, next_states):
        logits = self.forward(states, next_states)
        if self.gan_type == "lsgan":
            return self.loss_function(logits, torch.ones_like(logits))
        elif self.gan_type == "bcegan":
            return self.loss_function(logits, torch.ones_like(logits))
        elif self.gan_type == "wasserstein":
            return self.loss_function(logits, -1.0 * torch.ones_like(logits))
        else:
            raise ValueError(f"Invalid discriminator type: {self.gan_type}")
    
    def compute_grad_penalty(self, states, next_states):
        concate_states = torch.cat([states, next_states], dim=-1)
        concate_states.requires_grad = True

        logits = self.output(self.trunk(concate_states))
        ones = torch.ones_like(logits)
        grad = autograd.grad(
            outputs=logits, inputs=concate_states,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True,)[0]
        
        return torch.mean(torch.square(grad).sum(dim=-1))

    @torch.no_grad()
    def compute_prediction(self, states, next_states):
        self.eval()
        logits = self.forward(states, next_states)
        logits = self.reward_normalizer(logits)
        if self.gan_type == "lsgan":
            return logits
        if self.gan_type == "bcegan":
            return F.sigmoid(logits)
        else:
            raise ValueError(f"Invalid discriminator type: {self.gan_type}")

    @torch.no_grad()
    def compute_mixed_reward(self, states, next_states, task_reward):
        self.eval()
        logits = self.forward(states, next_states)
        logits = self.reward_normalizer(logits)
        if self.gan_type == "lsgan":
            style_reward = torch.clip(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0)
        elif self.gan_type == "bcegan":
            style_reward = -1.0 * torch.log(torch.clip(1.0 - F.sigmoid(logits), min=1e-4))
        elif self.gan_type == "wasserstein":
            self.reward_normalizer.update(logits)
        else:
            raise ValueError(f"Invalid discriminator type: {self.gan_type}")
        style_reward = self.style_reward_scale * style_reward.squeeze()
        return self.compute_lerp_reward(style_reward, task_reward)
    
    def compute_lerp_reward(self, style_reward, task_reward):
        return (1.0 - self.task_reward_lerp) * style_reward + self.task_reward_lerp * task_reward
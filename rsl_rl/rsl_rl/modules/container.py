#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from .estimator import EstimatorRecurrent
from .network import LSTMCell, GRUCell, MinRNNCell
from .actor_critic import ActorCritic
from .actor_critic_latent import ActorCriticLatent
from .actor_critic_recurrent import ActorCriticRecurrent


class Container(nn.Module):
    estimator: Union[EstimatorRecurrent, None]
    actor_critic: Union[ActorCritic, ActorCriticRecurrent]
    
    def __init__(self, actor_critic, estimator=None):
        super().__init__()
        self.estimator = estimator
        self.actor_critic = actor_critic
        
    def reset_hidden_states(self, inputs):
        hidden_e, hidden_a = (), ()
        if self.estimator is not None:
            rnn_module = self.estimator.memory_e
            hidden_e = get_rnn_initial_states(rnn_module, inputs)
        if self.actor_critic.is_recurrent:
            rnn_module = self.actor_critic.memory_a
            hidden_a = get_rnn_initial_states(rnn_module, inputs)
        return hidden_e, hidden_a

    def forward(
        self,
        observations,
        commands,
        hidden_e=None, 
        hidden_a=None,
        ):
        actor_obs = observations
        if self.estimator is not None:
            estimations, hidden_e = self.estimator.predict_inference(observations, hidden_states=hidden_e)
            actor_obs = torch.cat([estimations, observations], dim=-1)
        action_mean, hidden_a = self.actor_critic.act_inference(actor_obs, commands, hidden_states=hidden_a)
        return action_mean, hidden_e, hidden_a
    

class LatentContainer(nn.Module):
    estimator: Union[EstimatorRecurrent, None]
    actor_critic: ActorCriticLatent
    
    def __init__(self, actor_critic, estimator=None):
        super().__init__()
        self.latents = None
        self.estimator = estimator
        self.actor_critic = actor_critic
            
    def reset_hidden_states(self, inputs):
        hidden_e, hidden_a = (), ()
        if self.estimator is not None:
            rnn_module = self.estimator.memory_e
            hidden_e = get_rnn_initial_states(rnn_module, inputs)
        if self.actor_critic.is_recurrent:
            rnn_module = self.actor_critic.memory_a
            hidden_a = get_rnn_initial_states(rnn_module, inputs)
        return hidden_e, hidden_a
            
    def forward(
        self,
        observations,
        commands,
        update_sign,
        hidden_e=None, 
        hidden_a=None,
        generation=False,
        ):
        if generation and self.latents is not None:
            prior_latents = self.actor_critic.predict_prior_inference(observations, noise_scale=0.01)
            self.latents = torch.where(update_sign.view(-1, 1), prior_latents, self.latents)
        else:
            self.latents = self.actor_critic.encode_inference(commands)
            
        actor_obs = observations
        if self.estimator is not None:
            estimations, hidden_e = self.estimator.predict_inference(observations, hidden_states=hidden_e)
            actor_obs = torch.cat([estimations, observations], dim=-1)        
        action_mean, hidden_a = self.actor_critic.act_inference(actor_obs, self.latents, hidden_states=hidden_a)
        return action_mean, hidden_e, hidden_a

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
    

def get_rnn_initial_states(rnn_module, input):
    if isinstance(rnn_module, LSTMCell):
        h_zeros = torch.zeros(
            input.shape[0],
            rnn_module.hidden_dim,
            dtype=input.dtype,
            device=input.device,)
        c_zeros = torch.zeros(
            input.shape[0],
            rnn_module.hidden_dim,
            dtype=input.dtype,
            device=input.device,)
        return h_zeros, c_zeros
    elif isinstance(rnn_module, (GRUCell, MinRNNCell)):
        h_zeros = torch.zeros(
            input.shape[0],
            rnn_module.hidden_dim,
            dtype=input.dtype,
            device=input.device,)
        return h_zeros
    else:
        raise NotImplementedError
    
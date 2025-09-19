#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from typing import Union
from .actor_critic import ActorCritic
from .normalization import Normalization
from .network import GRUCell, LSTMCell


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_commands,
        num_critic_commands,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        rnn_type="gru",
        rnn_hidden_dim=256,
        init_std=1.0,
        noise_mode="constant",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
            
        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_commands=num_commands,
            num_critic_commands=num_critic_commands,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            init_std=init_std,
            noise_mode=noise_mode,
        )
        rnn_input_dim_a = num_actor_obs
        rnn_input_dim_c = num_critic_obs
        
        self.normalizers.update({
            "memory": Normalization(shape=[num_actor_obs]),
            "critic_memory": Normalization(shape=[num_critic_obs]),
            })
        
        self.rnn_type, self.rnn_hidden_dim = rnn_type, rnn_hidden_dim
        if self.rnn_type.lower() == "gru":
            rnn_class = GRUCell
        elif self.rnn_type.lower() == "lstm":
            rnn_class = LSTMCell
        else:
            raise ValueError(f"Invalid RNN type: {self.rnn_type.lower()}")

        self.memory_a = rnn_class(rnn_input_dim_a, rnn_hidden_dim)
        self.memory_c = rnn_class(rnn_input_dim_c, rnn_hidden_dim)
        self.hidden_states_a, self.hidden_states_c = None, None

    def reset(self, dones=None):
        self.reset_actor(dones)
        self.reset_critic(dones)
        
    def reset_actor(self, dones=None):
        assert dones is not None
        if isinstance(self.hidden_states_a, tuple):
            self.hidden_states_a[0][dones] = 0.0
            self.hidden_states_a[1][dones] = 0.0
        else:
            self.hidden_states_a[dones] = 0.0

    def reset_critic(self, dones=None):
        assert dones is not None
        if isinstance(self.hidden_states_c, tuple):
            self.hidden_states_c[0][dones] = 0.0
            self.hidden_states_c[1][dones] = 0.0
        else:
            self.hidden_states_c[dones] = 0.0

    def get_hidden_states(self):
        if self.hidden_states_a is None or self.hidden_states_c is None:
            return self.hidden_states_a, self.hidden_states_c
        
        if isinstance(self.hidden_states_a, tuple):
            hidden_states_a = (self.hidden_states_a[0], self.hidden_states_a[1])
            hidden_states_c = (self.hidden_states_c[0], self.hidden_states_c[1])
            return hidden_states_a, hidden_states_c
        else:
            return self.hidden_states_a, self.hidden_states_c

    def act(self, observations, commands, hidden_states=None, **kwargs):
        memories, _ = self.act_memory(observations, hidden_states)
        return super().act(memories, commands)

    def act_inference(self, observations, commands, hidden_states=None, **kwargs):
        memories, next_hidden_states = self.act_memory(observations, hidden_states)
        return super().act_inference(memories, commands)[0], next_hidden_states

    def evaluate(self, critic_observations, critic_commands, hidden_states=None, **kwargs):
        memories, _ = self.evaluate_memory(critic_observations, hidden_states)
        return super().evaluate(memories, critic_commands)
    
    def act_memory(self, observations, hidden_states=None):
        observations = self.normalizers["memory"](observations)
        if hidden_states is None:
            current_hidden_state = self.hidden_states_a
        else:
            current_hidden_state = hidden_states
        output, next_hidden_states = self.memory_a(observations, current_hidden_state)
        if hidden_states is None:
            self.hidden_states_a = next_hidden_states
        return output, next_hidden_states
    
    def evaluate_memory(self, critic_observations, hidden_states=None):
        critic_observations = self.normalizers["critic_memory"](critic_observations)
        if hidden_states is None:
            current_hidden_state = self.hidden_states_c
        else:
            current_hidden_state = hidden_states
        output, next_hidden_states = self.memory_c(critic_observations, current_hidden_state)
        if hidden_states is None:
            self.hidden_states_c = next_hidden_states
        return output, next_hidden_states
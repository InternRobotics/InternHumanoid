#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import torch


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.commands = None
            self.critic_commands = None
            
            self.next_observations = None
            self.next_critic_observations = None
            self.next_commands = None
            self.next_critic_commands = None
            
            self.prior_states = None
            self.hidden_states = None
            
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions,
        obs_shape, 
        critic_obs_shape,
        command_shape,
        critic_command_shape,
        actions_shape,
        device="cpu",
    ):
        self.observations = torch.zeros(num_transitions, num_envs, *obs_shape, dtype=torch.float, device=device)
        self.critic_observations = torch.zeros(num_transitions, num_envs, *critic_obs_shape, dtype=torch.float, device=device)
        self.next_observations = torch.zeros(num_transitions, num_envs, *obs_shape, dtype=torch.float, device=device)
        self.next_critic_observations = torch.zeros(num_transitions, num_envs, *critic_obs_shape, dtype=torch.float, device=device)
        
        self.commands = torch.zeros(num_transitions, num_envs, *command_shape, dtype=torch.float, device=device)
        self.critic_commands = torch.zeros(num_transitions, num_envs, *critic_command_shape, dtype=torch.float, device=device)
        self.next_commands = torch.zeros(num_transitions, num_envs, *command_shape, dtype=torch.float, device=device)
        self.next_critic_commands = torch.zeros(num_transitions, num_envs, *critic_command_shape, dtype=torch.float, device=device)
        
        self.rewards = torch.zeros(num_transitions, num_envs, 1, dtype=torch.float, device=device)
        self.actions = torch.zeros(num_transitions, num_envs, *actions_shape, dtype=torch.float, device=device)
        self.dones = torch.zeros(num_transitions, num_envs, 1, dtype=torch.bool, device=device)

        self.values = torch.zeros(num_transitions, num_envs, 1, dtype=torch.float, device=device)
        self.returns = torch.zeros(num_transitions, num_envs, 1, dtype=torch.float, device=device)
        self.advantages = torch.zeros(num_transitions, num_envs, 1, dtype=torch.float, device=device)
        
        self.actions_log_prob = torch.zeros(num_transitions, num_envs, 1, dtype=torch.float, device=device)
        self.mu = torch.zeros(num_transitions, num_envs, *actions_shape, dtype=torch.float, device=device)
        self.sigma = torch.zeros(num_transitions, num_envs, *actions_shape, dtype=torch.float, device=device)
        
        # For RNN Actor Critic
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        
        # For RNN Estimator
        self.saved_hidden_states_p = None
        
        self.step = 0
        self.device = device
        self.num_envs = num_envs
        self.num_transitions = num_transitions

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.critic_observations[self.step].copy_(transition.critic_observations)        
        self.commands[self.step].copy_(transition.commands)
        self.critic_commands[self.step].copy_(transition.critic_commands)

        if isinstance(transition.next_observations, torch.Tensor):
            self.next_observations[self.step].copy_(transition.next_observations)
            self.next_critic_observations[self.step].copy_(transition.next_critic_observations)
            self.next_commands[self.step].copy_(transition.next_commands)
            self.next_critic_commands[self.step].copy_(transition.next_critic_commands)

        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values.view(-1, 1))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        
        # For RNN networks
        self._save_prior_states(transition.prior_states)
        self._save_hidden_states(transition.hidden_states)
        
        # Increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states[0] is None: return
        hidden_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hidden_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.num_transitions, *hid.shape, dtype=torch.float, device=self.device) for hid in hidden_a
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.num_transitions, *hid.shape, dtype=torch.float, device=self.device) for hid in hidden_c
            ]
        # copy the states
        for i, (hid_a, hid_c) in enumerate(zip(hidden_a, hidden_c)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a)
            self.saved_hidden_states_c[i][self.step].copy_(hid_c)

    def _save_prior_states(self, hidden_states):
        if hidden_states is None: return
        hidden_p = hidden_states if isinstance(hidden_states, tuple) else (hidden_states,)
        # initialize if needed
        if self.saved_hidden_states_p is None:
            self.saved_hidden_states_p = [
                torch.zeros(self.num_transitions, *hid.shape, dtype=torch.float, device=self.device) for hid in hidden_p
            ]
        # copy the states
        for i, hid_p in enumerate(hidden_p):
            self.saved_hidden_states_p[i][self.step].copy_(hid_p)

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions)):
            if step == self.num_transitions - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            not_done = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + not_done * gamma * next_values - self.values[step]
            advantage = delta + not_done * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        action_noise = torch.mean(self.sigma)
        action_mean = torch.flatten(self.mu, 2, -1) * (1.0 - self.dones.float())
        action_rate = torch.norm(action_mean[1:] - action_mean[:-1], dim=2).mean()
        action_smoothness = torch.norm(
            action_mean[2:] - 2.0 * action_mean[1:-1] + action_mean[:-2], dim=2).mean()
        return action_noise.item(), action_rate.item(), action_smoothness.item()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        total_batch_size = self.num_envs * self.num_transitions
        mini_batch_size = total_batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)
        
        observations = self.observations.flatten(0, 1)
        critic_observations = self.critic_observations.flatten(0, 1)
        commands = self.commands.flatten(0, 1)
        critic_commands = self.critic_commands.flatten(0, 1)
        
        next_observations = self.next_observations.flatten(0, 1)
        next_critic_observations = self.next_critic_observations.flatten(0, 1)
        next_commands = self.next_commands.flatten(0, 1)
        next_critic_commands = self.next_critic_commands.flatten(0, 1)
        
        if self.saved_hidden_states_a is not None:
            hidden_states_a = [hidden_states.flatten(0, 1) for hidden_states in self.saved_hidden_states_a]
            hidden_states_c = [hidden_states.flatten(0, 1) for hidden_states in self.saved_hidden_states_c]
        if self.saved_hidden_states_p is not None:
            hidden_states_p = [hidden_states.flatten(0, 1) for hidden_states in self.saved_hidden_states_p]
                
        not_dones = 1.0 - self.dones.float().flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_obs_batch = critic_observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                next_critic_obs_batch = next_critic_observations[batch_idx]
                
                commands_batch = commands[batch_idx]
                critic_commands_batch = critic_commands[batch_idx]
                next_commands_batch = next_commands[batch_idx]
                next_critic_commands_batch = next_critic_commands[batch_idx]
                                    
                cont_batch = not_dones[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                
                hidden_a_batch, hidden_c_batch = None, None
                if self.saved_hidden_states_a is not None:
                    hidden_a_batch = [hidden_batch[batch_idx] for hidden_batch in hidden_states_a]
                    hidden_c_batch = [hidden_batch[batch_idx] for hidden_batch in hidden_states_c]
                    
                    if len(hidden_a_batch) == 1:
                        hidden_a_batch = hidden_a_batch[0]
                        hidden_c_batch = hidden_c_batch[0]
                    else:
                        hidden_a_batch = tuple(hidden_a_batch)
                        hidden_c_batch = tuple(hidden_c_batch)
                    
                hidden_p_batch = None
                if self.saved_hidden_states_p is not None:
                    hidden_p_batch = [hidden_batch[batch_idx] for hidden_batch in hidden_states_p]
                    
                    if len(hidden_p_batch) == 1:
                        hidden_p_batch = hidden_p_batch[0]
                    else:
                        hidden_p_batch = tuple(hidden_p_batch)
                
                yield (
                    obs_batch,
                    critic_obs_batch, 
                    next_obs_batch,
                    next_critic_obs_batch,
                    commands_batch,
                    critic_commands_batch, 
                    next_commands_batch,
                    next_critic_commands_batch,
                    cont_batch,
                    actions_batch, 
                    target_values_batch,
                    advantages_batch,
                    returns_batch, 
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hidden_a_batch,
                    hidden_c_batch,
                    hidden_p_batch,
                    (i == num_mini_batches - 1),
                )
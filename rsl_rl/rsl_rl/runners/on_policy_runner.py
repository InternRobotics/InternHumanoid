#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import copy
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from tqdm import tqdm
from typing import Union

import rsl_rl
from rsl_rl.algorithms import PPO, PPO_DISC, PPO_LATENT, HIMPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import Discriminator, Container, LatentContainer
from rsl_rl.modules import ActorCritic, ActorCriticLatent, ActorCriticRecurrent
from rsl_rl.utils import store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.alg_cfg = train_cfg["algorithm"]
        self.runner_cfg = train_cfg["runner"]
        self.device = device
        self.env = env
        
        self.save_interval = self.runner_cfg["save_interval"]
        self.log_interval = self.runner_cfg["log_interval"]
        self.num_warmup_steps = self.runner_cfg["num_warmup_steps"]
        self.num_steps_per_env = self.runner_cfg["num_steps_per_env"]
        
        policy_class_name = self.runner_cfg["policy_class_name"]
        alg_class_name = self.runner_cfg["algorithm_class_name"]
        
        actor_critic: Union[ActorCritic, ActorCriticLatent, ActorCriticRecurrent] = \
            eval(policy_class_name)(
            self.env.num_obs, 
            self.env.num_critic_obs,
            self.env.num_commands,
            self.env.num_critic_commands,
            self.env.num_actions, **train_cfg["policy"]).to(self.device)
            
        if alg_class_name == "PPO":
            self.alg: PPO = eval(alg_class_name)(
                actor_critic, device=self.device, **self.alg_cfg)
            self.alg.init_storage(
                num_envs=self.env.num_envs,
                num_transitions=self.num_steps_per_env, 
                obs_shape=[self.env.num_obs], 
                critic_obs_shape=[self.env.num_critic_obs],
                command_shape=[self.env.num_commands],
                critic_command_shape=[self.env.num_critic_commands],
                actions_shape=[self.env.num_actions],
                )
        elif alg_class_name == "PPO_LATENT":
            self.alg: PPO_LATENT = eval(alg_class_name)(
                actor_critic, device=self.device, **self.alg_cfg)
            self.alg.init_storage(
                num_envs=self.env.num_envs,
                num_transitions=self.num_steps_per_env,
                obs_shape=[self.env.num_obs], 
                critic_obs_shape=[self.env.num_critic_obs],
                command_shape=[self.env.num_commands],
                critic_command_shape=[self.env.num_critic_commands],
                actions_shape=[self.env.num_actions],
                )
        elif alg_class_name == "PPO_DISC":
            discriminator = Discriminator(self.env.num_states, **train_cfg["discriminator"])            
            self.alg: PPO_DISC = eval(alg_class_name)(
                actor_critic, discriminator, device=self.device, **self.alg_cfg)
            self.alg.init_storage(
                num_envs=self.env.num_envs,
                num_transitions=self.num_steps_per_env,
                obs_shape=[self.env.num_obs], 
                critic_obs_shape=[self.env.num_critic_obs],
                command_shape=[self.env.num_commands],
                critic_command_shape=[self.env.num_critic_commands],
                state_shape=[self.env.num_states], 
                actions_shape=[self.env.num_actions],
                )
        else:
            raise ValueError(f"Invalid algorithm class name: {alg_class_name}")
        
        if alg_class_name == "PPO_LATENT":
            self.alg.load_extra_components(
                self.env.sample_overlap_commands,
                self.env.sample_random_commands,
                self.env.compute_dtw_distances,
                ) 
        elif alg_class_name == "PPO_DISC":
            self.alg.load_extra_components(self.env.init_expert_dataloader())
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def warmup(self):
        with torch.inference_mode():
            self.alg.test_mode()
            obs, critic_obs = self.env.reset()
            commands, critic_commands = self.env.get_commands()
            obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
            commands, critic_commands = commands.to(self.device), critic_commands.to(self.device)
            
            for i in tqdm(range(self.num_warmup_steps)):
                actions = self.alg.act(obs, critic_obs, commands, critic_commands)
                obs, critic_obs, rewards, dones, infos = self.env.step(actions)
                commands, critic_commands = self.env.get_commands()
                obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                commands, critic_commands = commands.to(self.device), critic_commands.to(self.device)

    def learn(self, num_learning_iterations: int):
        # initialize writer
        if self.log_dir is not None and self.writer is None and self.env.headless:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.runner_cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()
            
            if self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.runner_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        self.warmup()
        self.train_mode()
        with torch.inference_mode():
            obs, critic_obs = self.env.reset()
            commands, critic_commands = self.env.get_commands()
            obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
            commands, critic_commands = commands.to(self.device), critic_commands.to(self.device)
            
            if isinstance(self.alg, PPO_DISC):
                policy_states, next_policy_states = self.env.get_policy_states()
                policy_states, next_policy_states = policy_states.to(self.device), next_policy_states.to(self.device)
            
        ep_infos, ep_metrics = [], []
        rewbuffer, lenbuffer, sucbuffer, combuffer = (deque(maxlen=100) for _ in range(4))
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                self.alg.test_mode()
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, commands, critic_commands)
                    obs, critic_obs, rewards, dones, infos = self.env.step(actions)
                    commands, critic_commands = self.env.get_commands()
                    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    commands, critic_commands = commands.to(self.device), critic_commands.to(self.device)
                    self.alg.process_post_act(obs, critic_obs, commands, critic_commands)
                    
                    if isinstance(self.alg, PPO_DISC):
                        rewards = self.alg.compute_rewards(policy_states, next_policy_states, rewards)
                        policy_states, next_policy_states = self.env.get_policy_states()
                        policy_states, next_policy_states = policy_states.to(self.device), next_policy_states.to(self.device)
                    
                    self.alg.process_env_step(rewards, dones, infos)

                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    
                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos: ep_infos.append(infos["episode"])
                        if "episode_metrics" in infos: ep_metrics.append(infos["episode_metrics"])

                    new_ids = (dones > 0.0).nonzero().flatten()
                    if len(new_ids) > 0:
                        if self.log_dir is not None:
                            tolist = lambda x: x.cpu().numpy().tolist()
                            rewbuffer.extend(tolist(cur_reward_sum[new_ids]))
                            lenbuffer.extend(tolist(cur_episode_length[new_ids]))
                            sucbuffer.extend(tolist(infos["time_outs"].float()[new_ids]))
                            combuffer.extend(tolist(infos["completions"].float()[new_ids]))
                        
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, critic_commands)
            
            total_completion, total_success_rate = self.env.get_task_info()
            mean_noise_std, mean_action_rate, mean_action_smoothness = self.alg.storage.get_statistics()
            
            if isinstance(self.alg, PPO):
                mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_kl_divergence = self.alg.update()
            elif isinstance(self.alg, PPO_LATENT):
                mean_value_loss, mean_surrogate_loss, mean_estimator_loss, \
                    mean_kl_divergence, mean_overlap_loss, mean_triplet_loss = self.alg.update()
            elif isinstance(self.alg, PPO_DISC):
                mean_value_loss, mean_surrogate_loss, mean_estimator_loss, \
                    mean_kl_divergence, mean_discriminator_loss, mean_prediction = self.alg.update()    
            else:
                raise NotImplementedError
            
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and it % self.log_interval == 0:
                self.log(locals())
            if it % self.save_interval == 0 and self.env.headless:
                ckpt_path = os.path.join(self.log_dir, f"model_{it}.pt")
                self.save(ckpt_path)
                
            ep_infos.clear()
            ep_metrics.clear()
            
            if it == start_iter and self.env.headless:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb"] and git_file_paths:
                    for path in git_file_paths: self.writer.save_file(path)

        if self.env.headless:
            ckpt_path = os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
            self.save(ckpt_path)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if self.env.headless:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                    
        if locs["ep_metrics"]:
            for key in locs["ep_metrics"][0]:
                metrictensor = torch.tensor([], device=self.device)
                for ep_metric in locs["ep_metrics"]:
                    # handle scalar and zero dimensional tensor metrics
                    if key not in ep_metric:
                        continue
                    if not isinstance(ep_metric[key], torch.Tensor):
                        ep_metric[key] = torch.Tensor([ep_metric[key]])
                    if len(ep_metric[key].shape) == 0:
                        ep_metric[key] = ep_metric[key].unsqueeze(0)
                    metrictensor = torch.cat((metrictensor, ep_metric[key].to(self.device)))
                value = torch.mean(metrictensor)
                # log to logger and terminal
                if self.env.headless:
                    self.writer.add_scalar("Metric/" + key, value, locs["it"])
                ep_string += f"""{f'Mean metric {key}:':>{pad}} {value:.4f}\n"""
                
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        if self.env.headless:
            self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
            self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
            self.writer.add_scalar("Loss/estimation", locs["mean_estimator_loss"], locs["it"])
            self.writer.add_scalar("Loss/kl_divergence", locs["mean_kl_divergence"], locs["it"])
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
            
            if isinstance(self.alg, PPO_LATENT):
                self.writer.add_scalar("Loss/overlaps", locs["mean_overlap_loss"], locs["it"])
                self.writer.add_scalar("Loss/triplets", locs["mean_triplet_loss"], locs["it"])
            elif isinstance(self.alg, PPO_DISC):
                self.writer.add_scalar("Loss/discriminator_loss", locs["mean_discriminator_loss"], locs["it"])
                self.writer.add_scalar("Loss/mean_prediction", locs["mean_prediction"], locs["it"])
            
            self.writer.add_scalar("Policy/mean_noise_std", locs["mean_noise_std"], locs["it"])
            self.writer.add_scalar("Policy/mean_action_rate", locs["mean_action_rate"], locs["it"])
            self.writer.add_scalar("Policy/mean_action_smoothness", locs["mean_action_smoothness"], locs["it"])
            
            self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
            self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
            self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
            if len(locs["lenbuffer"]) > 0:
                self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
                self.writer.add_scalar("Train/mean_completion", statistics.mean(locs["combuffer"]), locs["it"])
                self.writer.add_scalar("Train/mean_successful_rate", statistics.mean(locs["sucbuffer"]), locs["it"])
            self.writer.add_scalar("Train/total_completion", locs["total_completion"], locs["it"])
            self.writer.add_scalar("Train/total_success_rate", locs["total_success_rate"], locs["it"])
        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["lenbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'KL Divergence:':>{pad}} {locs['mean_kl_divergence']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {locs["mean_noise_std"]:.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Mean successful rate:':>{pad}} {statistics.mean(locs['sucbuffer']):.2f}\n"""
                f"""{'Mean completion:':>{pad}} {statistics.mean(locs['combuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {locs["mean_noise_std"]:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {}
        saved_dict["model_state_dict"] = self.alg.actor_critic.state_dict()
        if isinstance(self.alg, PPO_DISC):
            saved_dict["discriminator_state_dict"] = self.alg.discriminator.state_dict()
        saved_dict["optimizer_state_dict"] = self.alg.optimizer.state_dict()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["wandb"] and self.env.headless:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if isinstance(self.alg, PPO_DISC):
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = lambda x, y: self.alg.actor_critic.act_inference(x, y)[0]
        return policy

    def export_model(self):
        self.eval_mode()        
        if isinstance(self.alg, (PPO, PPO_DISC)):
            actor_critic = Container(copy.deepcopy(self.alg.actor_critic).to("cpu"))
        elif isinstance(self.alg, PPO_LATENT):
            actor_critic = LatentContainer(copy.deepcopy(self.alg.actor_critic).to("cpu"))
        else:
            raise NotImplementedError
        
        return actor_critic

    def train_mode(self):
        self.alg.train_mode()

    def eval_mode(self):
        self.alg.test_mode()
            
    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

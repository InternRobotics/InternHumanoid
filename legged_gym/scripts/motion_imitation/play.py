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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
import torch
import pickle
import hydra
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, export_jit_to_onnx, load_onnx_policy
from hydra.utils import get_class
from easydict import EasyDict
from omegaconf import OmegaConf, DictConfig

@hydra.main(version_base=None,config_path="../../config", config_name="base")
def play(cfg: OmegaConf):

    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    args = get_args(cfg.args)
    cfg.pop('args')
    
    env_cfg, train_cfg = EasyDict(), EasyDict()
    for key in cfg.keys():
        train_cfg.update(cfg[key]) if key == 'algo' else env_cfg.update(cfg[key])

    train_cfg.runner.resume = True
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 8)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.termination.termination_time = 100.0 # sec

    # prepare environment
    env = task_registry.make_env(name=env_cfg.name, args=args, env_cfg=env_cfg)[0]
    obs, critic_obs = env.reset()
    commands, critic_commands = env.get_commands()
    runner_class = get_class(f"rsl_rl.runners.{train_cfg.runner_class_name}")
    runner, train_cfg = task_registry.make_runner(
        env=env, runner_class=runner_class, name=env_cfg.name, args=args)

    policy = runner.export_model().to(env.device)
    dof_pos_info = {name: [] for name in env.dof_names}
    dof_target_info = {name: [] for name in env.dof_names}
    dof_reference_info = {name: [] for name in env.dof_names}

    hidden_e, hidden_a = policy.reset_hidden_states(obs)
    with torch.inference_mode():
        for i in range(1000):
            obs, critic_obs = env.get_observations()
            commands, critic_commands = env.get_commands()
            
            # action, hidden_e, hidden_a = policy(
            #     obs, commands, env.get_update_sign(), hidden_e, hidden_a, True)
            action, hidden_e, hidden_a = policy(
                obs, commands, hidden_e, hidden_a)
            env.step(action)

            env_dof_targets = env.actions[:, -1] * env.cfg.control.action_scale + env.default_dof_pos
            for i, name in enumerate(env.dof_names):
                dof_pos_info[name] += [env.dof_pos[0, i].item()]
                dof_target_info[name] += [env_dof_targets[0, i].item()]
                dof_reference_info[name] += [env.motion_dof_pos[0, i].item()]

        pickle.dump({
            "joint position": dof_pos_info,
            "joint target": dof_target_info,
            "joint reference": dof_reference_info,
            }, open(f"{LEGGED_GYM_ROOT_DIR}/logs/record.pkl", "wb"))

if __name__ == '__main__':
    play()

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
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, export_jit_to_onnx, load_onnx_policy
from rsl_rl.runners import OnPolicyRunner


def export(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    train_cfg.runner.resume = True

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 8)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.randomize_friction = False

    # prepare environment
    env = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)[0]
    obs, critic_obs = env.get_observations()
    
    runner, train_cfg = task_registry.make_runner(
        env=env, runner_class=OnPolicyRunner, name=args.task, args=args)
    model_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/export_model")
    if not os.path.exists(model_path): os.mkdir(model_path)
    ckpt_path = os.path.join(model_path, f"{train_cfg.runner.run_name}.onnx")
    export_jit_to_onnx(runner.export_model(), ckpt_path, obs)
    
    pickle.dump({
        "FUTURE_DT": env.cfg.env.future_dt,
        "FUTURE_TIMESTEPS": env.cfg.env.future_timesteps,
        "HISTORY_TIMESTEPS": env.cfg.env.history_timesteps,
        
        "STIFFNESS": env.cfg.control.stiffness,
        "DAMPING": env.cfg.control.damping,
        "ACTION SCALE": env.cfg.control.action_scale,
        "DEFAULT JOINT ANGLES": env.cfg.init_state.default_joint_angles,
        
        "DOF NAMES": env.dof_names,
        "KEYFRAME NAMES": env.keyframe_names,
        "ACTION INDICES": env.action_indices.cpu().numpy().tolist(),
        "MOTION DOF INDICES": env.motion_dof_indices.cpu().numpy().tolist(),
        "DEFAULT DOF POS": env.default_dof_pos.cpu().numpy().tolist(),
        "TORQUE LIMITS": env.torque_limits.cpu().numpy().tolist(),
        "END EFFECTOR INDICES": env.endeffector_indices.cpu().numpy().tolist(),
    }, open(ckpt_path.split(".")[0] + ".pkl", 'wb'))
        

if __name__ == '__main__':
    args = get_args()
    args.headless = True
    export(args)

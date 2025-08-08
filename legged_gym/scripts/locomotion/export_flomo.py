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
import hydra
from easydict import EasyDict
from hydra.utils import get_class
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def export(cfg: OmegaConf):

    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    args = get_args(cfg.args)
    cfg.pop("args")
    env_cfg, train_cfg = EasyDict(), EasyDict()
    for key in cfg.keys():
        train_cfg.update(cfg[key]) if key == "algo" else env_cfg.update(cfg[key])

    train_cfg.runner.resume = True
    train_cfg.headless = True

    # override some parameters for testing
    args.headless = True
    args.num_envs = min(args.num_envs, 1)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.randomize_friction = False

    # prepare environment
    env = task_registry.make_env(name=env_cfg.name, args=args, env_cfg=env_cfg)[0]
    obs = env.get_observations()

    runner_class = get_class(f"rsl_rl.runners.{train_cfg.runner_class_name}")

    runner, train_cfg = task_registry.make_runner(
        env=env, runner_class=runner_class, name=env_cfg.name, args=args, train_cfg=train_cfg
    )
    model_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/export_model")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    ckpt_path = os.path.join(model_path, f"{train_cfg.runner.experiment_name}.onnx")
    export_jit_to_onnx(runner.export_model(), ckpt_path, obs[[0]])

    pickle.dump(
        {
            "STIFFNESS": env.cfg.control.stiffness,
            "DAMPING": env.cfg.control.damping,
            "DEFAULT JOINT ANGLES": env.cfg.init_state.default_joint_angles,
            "ACTION SCALE": env.cfg.control.action_scale,
            "NUM ACTIONS": env.cfg.env.num_actions,
            "NUM OBS": env.cfg.env.num_observations,
            "LEN HISTORY": env.cfg.env.num_actor_history,
            "DOF NAMES": env.dof_names,
        },
        open(ckpt_path.split(".")[0] + ".pkl", "wb"),
    )
    print(f"Exported ONNX model to {ckpt_path} and metadata to {ckpt_path.split('.')[0] + '.pkl'}")


if __name__ == "__main__":

    export()

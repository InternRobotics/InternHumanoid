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

import onnxruntime as ort

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, Logger
import hydra
import numpy as np
import torch
from easydict import EasyDict
from hydra.utils import get_class
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Tuple, List
from dataclasses import dataclass
from pynput import keyboard


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


# ? >>> Control
@dataclass
class ControlState:
    # current values
    x_vel: float = 0.0
    y_vel: float = 0.0
    yaw_vel: float = 0.0
    height: float = 0.74
    exit_flag: bool = False

    # limits
    range_map: Dict[str, Tuple[float, float]] = None
    # step sizes
    step_map: Dict[str, float] = None
    # key bindings: key -> (field, +delta) ;
    keymap: Dict[str, Tuple[str, float]] = None

    def update(self, pressed: Dict[str, bool]) -> None:
        for key, is_down in pressed.items():
            if not is_down:
                continue
            if key in self.keymap:
                field, delta = self.keymap[key]
                step = self.step_map.get(field, 0.0) * (1 if delta > 0 else -1)
                new_val = getattr(self, field) + step
                lo, hi = self.range_map[field]
                setattr(self, field, clamp(new_val, lo, hi))
            elif key == "q":
                self.exit_flag = True

        def zero_if_both_up(neg_key: str, pos_key: str, field: str):
            if not pressed.get(neg_key, False) and not pressed.get(pos_key, False):
                setattr(self, field, 0.0)

        zero_if_both_up("s", "w", "x_vel")
        zero_if_both_up("d", "a", "y_vel")
        zero_if_both_up("r", "e", "yaw_vel")


class KeyboardHandler:
    def __init__(self, keys: List[str]):
        self.pressed = {k: False for k in keys}

    def on_press(self, key):
        try:
            c = key.char
            if c in self.pressed:
                self.pressed[c] = True
        except Exception:
            pass  # ignore special keys

    def on_release(self, key):
        try:
            c = key.char
            if c in self.pressed:
                self.pressed[c] = False
        except Exception:
            pass


# ? <<< Control


def load_policy():
    body = torch.jit.load("", map_location="cuda:0")

    def policy(obs):
        action = body.forward(obs)
        return action

    return policy


def load_onnx_policy():
    model = ort.InferenceSession("")

    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")

    return run_inference


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def play(cfg: OmegaConf):

    # --- Keyboard / Control ---
    control_range = {
        "x_vel": (-0.8, 1.0),
        "y_vel": (-0.5, 0.5),
        "yaw_vel": (-0.8, 0.8),
        "height": (0.34, 0.74),
    }
    control = ControlState(
        range_map=control_range,
        step_map={"x_vel": 0.1, "y_vel": 0.1, "yaw_vel": 0.1, "height": 0.01},
        keymap={
            # move
            "w": ("x_vel", +1),
            "s": ("x_vel", -1),
            "a": ("y_vel", +1),
            "d": ("y_vel", -1),
            "e": ("yaw_vel", +1),
            "r": ("yaw_vel", -1),
            # height
            "x": ("height", +1),
            "z": ("height", -1),
            # "q" handled in update() to exit
        },
    )
    kb = KeyboardHandler(keys=list(control.keymap.keys()) + ["q"])
    listener = keyboard.Listener(on_press=kb.on_press, on_release=kb.on_release)
    listener.daemon = True
    listener.start()

    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    args = get_args(cfg.args)
    cfg.pop("args")

    env_cfg, train_cfg = EasyDict(), EasyDict()
    for key in cfg.keys():
        train_cfg.update(cfg[key]) if key == "algo" else env_cfg.update(cfg[key])

    args.headless = False
    args.num_envs = min(args.num_envs, 8)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 8)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.asset.self_collisions = 0
    env_cfg.env.upper_teleop = False
    # prepare environment
    env, env_cfg = task_registry.make_env(name=env_cfg.name, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = 0
    env.commands[:, 1] = 0
    env.commands[:, 2] = 0
    env.commands[:, 4] = 0.74
    env.action_curriculum_ratio = 1.0
    obs = env.get_observations()
    # load policy
    runner_class = get_class(f"rsl_rl.runners.{train_cfg.runner_class_name}")
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_runner(
        env=env, runner_class=runner_class, name=env_cfg.name, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)  # Use this to load from trained pt file

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    while not control.exit_flag:

        control.update(kb.pressed)

        env.action_curriculum_ratio = 1.0
        actions = policy(obs.detach())

        env.commands[:, 0] = control.x_vel
        env.commands[:, 1] = control.y_vel
        env.commands[:, 2] = control.yaw_vel
        env.commands[:, 4] = control.height

        env.command_ratio[:, 0] = (env.commands[:, 0] / control_range["x_vel"][1]) * (env.commands[:, 0] > 0) + (
            env.commands[:, 0] / control_range["x_vel"][0]
        ) * (env.commands[:, 0] < 0)

        obs, _, _, _, _, _, _ = env.step(actions.detach())
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    play()

#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .ppo_disc import PPO_DISC
from .ppo_latent import PPO_LATENT
from .him_ppo import HIMPPO

__all__ = ["PPO", "PPO_DISC", "PPO_LATENT", "HIMPPO"]

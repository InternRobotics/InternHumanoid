#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_latent import ActorCriticLatent
from .actor_critic_recurrent import ActorCriticRecurrent

from .network import HyperEmbedder, HyperLERPBlock
from .network import HyperPolicyHead, HyperValueHead

from .estimator import EstimatorRecurrent
from .discriminator import Discriminator
from .normalization import Normalization
from .container import Container, LatentContainer, HIMContainer

from .him_actor_critic import HIMActorCritic
from .him_estimator import HIMEstimator

__all__ = [
    "ActorCritic", "ActorCriticLatent", "ActorCriticRecurrent",
    "HyperEmbedder", "HyperLERPBlock", "HyperPolicyHead", "HyperValueHead",
    "EstimatorRecurrent", "Discriminator", "Normalization", "Container", "LatentContainer",
    "HIMActorCritic", "HIMEstimator", "HIMContainer"
    ]

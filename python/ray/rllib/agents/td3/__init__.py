from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents.td3.td3 import TD3Trainer, DEFAULT_CONFIG
from ray.rllib.utils import renamed_class

TD3Agent = renamed_class(TD3Trainer)

__all__ = [
    "TD3Agent", "TD3Trainer", "DEFAULT_CONFIG"
]

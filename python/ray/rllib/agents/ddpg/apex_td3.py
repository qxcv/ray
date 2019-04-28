from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
from ray.rllib.utils.annotations import override
from ray.rllib.utils import merge_dicts

APEX_TD3_DEFAULT_CONFIG = merge_dicts(
    TD3_DEFAULT_CONFIG,
    {
        "optimizer_class": "AsyncReplayOptimizer",
        "optimizer": merge_dicts(
            TD3_DEFAULT_CONFIG["optimizer"], {
                "max_weight_sync_delay": 400,
                "num_replay_buffer_shards": 4,
                "debug": False
            }),
        # empirically I haven't found that k-step returns help much on Mujoco
        # envs, so disabling for simplicity
        "n_step": 1,
        "num_gpus": 0,
        "num_workers": 32,
        # 10x size of normal TD3 buffer, since replacement is super super fast
        # on most envs
        "buffer_size": 10000000,
        # this doesn't really matter any more because I'm doing a target update
        # on every policy update
        # TODO: figure out whether once-per-timestep is even a good idea,
        # especially in the distributed SGD setting
        "target_network_update_freq": int(1e10),
        # 50,000 timesteps per epoch, with 10 epochs of pure exploration at the
        # beginning
        "timesteps_per_iteration": 30000,
        "learning_starts": 300000,
        "pure_exploration_steps": 300000,
        "train_batch_size": 256,
        "sample_batch_size": 20,
        "per_worker_exploration": False,
        "worker_side_prioritization": True,
        "min_iter_time_s": 30,
    },
)


class ApexTD3Trainer(TD3Trainer):
    """TD3 variant that uses the Ape-X distributed policy optimizer. Like
    APEX_DDPG, it's configured for for a large single node (32 cores). For
    running in a large cluster, increase the `num_workers` config var. """

    _name = "APEX_TD3"
    _default_config = APEX_TD3_DEFAULT_CONFIG

    @override(TD3Trainer)
    def update_target_if_needed(self):
        # Ape-X updates based on num steps trained, not sampled
        if self.optimizer.num_steps_trained - self.last_target_update_ts > \
                self.config["target_network_update_freq"]:
            self.local_evaluator.foreach_trainable_policy(lambda p, _: p.
                                                          update_target())
            self.last_target_update_ts = self.optimizer.num_steps_trained
            self.num_target_updates += 1

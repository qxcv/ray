from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.td3.td3_torch_policy_graph import TD3TorchPolicyGraph
from ray.rllib.utils.annotations import override
from ray.rllib.utils.schedules import PiecewiseSchedule, step_interpolation

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # see https://spinningup.openai.com/en/latest/algorithms/td3.html for docs

    # TD3 reference impl does not use any of the following things, but base DQN
    # trainer does
    "grad_norm_clipping": None,
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,
    "prioritized_replay_eps": 1e-6,
    "beta_annealing_fraction": 0.2,
    "final_prioritized_replay_beta": 0.4,
    "schedule_max_timesteps": 0,
    "compress_observations": False,
    "evaluation_interval": None,
    "optimizer_class": "SyncReplayOptimizer",
    "min_iter_time_s": 1,
    "per_worker_exploration": False,
    # TO ADD: "num_workers": 0

    "steps_per_epoch": 5000,  # FIXME what does this map to in rllib?
    "sample_batch_size": 5,  # ??? is this the right way to implement steps_per_epoch?
    "timesteps_per_iteration": 5000,  # ??? same here, I think this controls how many env steps we have to wait for before optimising, but I'm not certain

    "epochs": 100,  # FIXME too

    # equivalent to replay_size in Spinning Up
    "buffer_size": 1000000,

    "max_ep_len": 1000,  # FIXME this will be called something else

    "gamma": 0.99,  # FIXME: probably called something else

    # equivalent to batch_size in Spinning Up
    "train_batch_size": 100,
    # number of random actions to take at start of training; equivalent to
    # start_steps in Spinning Up (TODO: figure out how to alias this to
    # learning_starts)
    "random_explore_steps": 10000,
    "learning_starts": 10000,
    # coefficient used for Polyak averaging of target q & pi networks;
    # corresponds to 1-tau in the original TD3 paper
    "polyak": 0.995,
    # learning rate for policy
    "pi_lr": 0.001,
    # learning rate for Q-functions
    "q_lr": 0.001,
    # stddev of Gaussian train time noise (set this to 0 when testing, e.g.
    # with the `rllib rollout` command)
    "act_noise": 0.1,
    # noise added to target policy to compute Q-values for target
    "target_noise": 0.2,
    # clipping for target_noise; called noise_clip in Spinning Up
    "target_noise_clip": 0.5,
    # we update q-function policy_delay times more often than policy (2x in the
    # TD3 paper)
    "policy_delay": 2,
    # TODO: add architecture options for actor and critic
})
# __sphinx_doc_end__
# yapf: enable


# TODO: don't have this inherit from DQNTrainer, because I have no fucking idea
# what DQNTrainer actually does. Instead, inherit from Trainer directly.
class TD3Trainer(DQNTrainer):
    """TD3 implementation in PyTorch (maybe TF in future)."""
    _name = "TD3"
    _default_config = DEFAULT_CONFIG
    _policy_graph = TD3TorchPolicyGraph

    @override(DQNTrainer)
    def update_target_if_needed(self):
        # XXX: this is really pointless; I should always be updating the policy
        # no matter what, so there's no need to separate
        # update_target_if_needed() out from _train().
        self.local_evaluator.foreach_trainable_policy(lambda p, _: p.
                                                      update_target())
        self.last_target_update_ts = self.global_timestep
        self.num_target_updates += 1

    @override(DQNTrainer)
    def _make_exploration_schedule(self, worker_index):
        # Like Spinning Up, we do pure exploration for
        # self.config["start_steps"] but does no exploration at all thereafter.
        # TODO: make sure we don't update the policy in that time, either
        return PiecewiseSchedule(endpoints=[
            (0, 1.0), (self.config["random_explore_steps"], 0.0)
        ],
                                 interpolation=step_interpolation,
                                 outside_value=0.0)

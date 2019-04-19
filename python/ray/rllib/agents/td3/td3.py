from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

from ray.rllib import optimizers
from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.agents.td3.td3_torch_policy_graph import TD3TorchPolicyGraph
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.utils.annotations import override
from ray.rllib.utils.schedules import PiecewiseSchedule, step_interpolation

logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # see https://spinningup.openai.com/en/latest/algorithms/td3.html for docs

    # TODO: add architecture options for actor and critic

    # TD3 reference impl does not use any of the following things, but base DQN
    # trainer does
    "grad_norm_clipping": None,
    "compress_observations": True,
    "optimizer_class": "SyncReplayOptimizer",
    "prioritized_replay": False,
    "min_iter_time_s": 1,
    "num_workers": 0,

    "evaluation_interval": 10,
    "evaluation_num_episodes": 5,

    # "steps_per_epoch": 5000,  # FIXME what does this map to in rllib?
    "sample_batch_size": 1,  # ??? is this the right way to implement steps_per_epoch?
    "timesteps_per_iteration": 1000,  # ??? same here, I think this controls how many env steps we have to wait for before optimising, but I'm not certain

    "epochs": 100,  # FIXME too

    # equivalent to replay_size in Spinning Up
    "buffer_size": 1000000,

    # "max_ep_len": 1000,  # FIXME this will be called something else

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

    # disable all action noise in compute_actions (good for test rollouts)
    "test_mode": False,
})
# __sphinx_doc_end__
# yapf: enable

OPTIMIZER_SHARED_CONFIGS = [
    "buffer_size",
    "sample_batch_size",
    "train_batch_size",
    "learning_starts",
    "prioritized_replay",
]


class TD3Trainer(Trainer):
    """TD3 implementation in PyTorch (maybe TF in future)."""
    _name = "TD3"
    _default_config = DEFAULT_CONFIG
    _policy_graph = TD3TorchPolicyGraph
    _optimizer_shared_configs = OPTIMIZER_SHARED_CONFIGS

    def _evaluate(self):
        logger.info("Evaluating current policy for {} episodes".format(
            self.config["evaluation_num_episodes"]))
        self.evaluation_ev.restore(self.local_evaluator.save())
        self.evaluation_ev.foreach_policy(lambda p, _: p.set_test_mode(True))
        for _ in range(self.config["evaluation_num_episodes"]):
            self.evaluation_ev.sample()
        metrics = collect_metrics(self.evaluation_ev)
        self.evaluation_ev.foreach_policy(lambda p, _: p.set_test_mode(False))
        return {"evaluation": metrics}

    # HACK HACK HACK copy-pasting DQNTrainer shit and removing the stuff I
    # don't need

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)

        # Update effective batch size to include n-step
        adjusted_batch_size = max(config["sample_batch_size"],
                                  config.get("n_step", 1))
        config["sample_batch_size"] = adjusted_batch_size

        # Like Spinning Up, we do pure exploration for
        # self.config["start_steps"] but does no exploration at all thereafter.
        # TODO: make sure we don't update the policy in that time, either
        self.exploration0 = PiecewiseSchedule(endpoints=[
            (0, 1.0), (self.config["random_explore_steps"], 0.0)
        ],
                                              interpolation=step_interpolation,
                                              outside_value=0.0)
        self.explorations = [
            self._make_exploration_schedule(i)
            for i in range(config["num_workers"])
        ]

        for k in self._optimizer_shared_configs:
            if k not in config["optimizer"]:
                config["optimizer"][k] = config[k]

        self.local_evaluator = self.make_local_evaluator(
            env_creator, self._policy_graph)

        if config["evaluation_interval"]:
            self.evaluation_ev = self.make_local_evaluator(
                env_creator,
                self._policy_graph,
                extra_config={
                    "batch_mode": "complete_episodes",
                    "batch_steps": 1,
                })
            self.evaluation_metrics = self._evaluate()

        def create_remote_evaluators():
            return self.make_remote_evaluators(env_creator, self._policy_graph,
                                               config["num_workers"])

        assert config["optimizer_class"] != "AsyncReplayOptimizer"
        self.remote_evaluators = create_remote_evaluators()

        self.optimizer = getattr(optimizers, config["optimizer_class"])(
            self.local_evaluator, self.remote_evaluators, config["optimizer"])
        # Create the remote evaluators *after* the replay actors
        if self.remote_evaluators is None:
            self.remote_evaluators = create_remote_evaluators()
            self.optimizer._set_evaluators(self.remote_evaluators)

        self.last_target_update_ts = 0
        self.num_target_updates = 0

    @override(Trainer)
    def _train(self):
        start_timestep = self.global_timestep

        # Update worker explorations
        exp_vals = [self.exploration0.value(self.global_timestep)]
        self.local_evaluator.foreach_trainable_policy(lambda p, _: p.
                                                      set_epsilon(exp_vals[0]))
        for i, e in enumerate(self.remote_evaluators):
            exp_val = self.explorations[i].value(self.global_timestep)
            e.foreach_trainable_policy.remote(lambda p, _: p.set_epsilon(
                exp_val))
            exp_vals.append(exp_val)

        # Do optimization steps
        start = time.time()
        while (self.global_timestep - start_timestep <
               self.config["timesteps_per_iteration"]
               ) or time.time() - start < self.config["min_iter_time_s"]:
            # I believe this scatters weights to workers, pulls new experience
            # from them, and updates weights
            self.optimizer.step()

            def target_updater(p, _):
                p.update_target()

            self.local_evaluator.foreach_trainable_policy(target_updater)
            self.last_target_update_ts = self.global_timestep
            self.num_target_updates += 1

        result = self.collect_metrics()

        result.update(timesteps_this_iter=self.global_timestep -
                      start_timestep,
                      info=dict(
                          {
                              "min_exploration": min(exp_vals),
                              "max_exploration": max(exp_vals),
                              "num_target_updates": self.num_target_updates,
                          }, **self.optimizer.stats()))

        if self.config["evaluation_interval"]:
            if self.iteration % self.config["evaluation_interval"] == 0:
                self.evaluation_metrics = self._evaluate()
            result.update(self.evaluation_metrics)

        return result

    @property
    def global_timestep(self):
        return self.optimizer.num_steps_sampled

    def __getstate__(self):
        state = Trainer.__getstate__(self)
        state.update({
            "num_target_updates": self.num_target_updates,
            "last_target_update_ts": self.last_target_update_ts,
        })
        return state

    def __setstate__(self, state):
        Trainer.__setstate__(self, state)
        self.num_target_updates = state.get("num_target_updates", 0)
        self.last_target_update_ts = state.get("last_target_update_ts", 0)

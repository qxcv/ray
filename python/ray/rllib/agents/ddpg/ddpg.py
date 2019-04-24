from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.ddpg.ddpg_policy_graph import DDPGPolicyGraph
from ray.rllib.utils.annotations import override
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # === Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC) tricks ===
    # TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
    # In addition to settings below, you can use "exploration_noise_type" and
    # "exploration_gauss_act_noise" to get IID Gaussian exploration noise
    # instead of OU exploration noise.
    # twin Q-net
    "twin_q": False,
    # delayed policy update
    "policy_delay": 1,
    # target policy smoothing
    # (this also replaces OU exploration noise with IID Gaussian exploration
    # noise, for now)
    "smooth_target_policy": False,
    # gaussian stddev of target action noise for smoothing
    "target_noise": 0.2,
    # target noise limit (bound)
    "target_noise_clip": 0.5,

    # === Evaluation ===
    # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    "evaluation_interval": None,
    # Number of episodes to run per evaluation period.
    "evaluation_num_episodes": 10,

    # === Model ===
    # Postprocess the policy network model output with these hidden layers
    "actor_hiddens": [64, 64],
    # Hidden layers activation of the policy network
    "actor_hidden_activation": "relu",
    # Postprocess the critic network model output with these hidden layers
    "critic_hiddens": [64, 64],
    # Hidden layers activation of the critic network
    "critic_hidden_activation": "relu",
    # N-step Q learning
    "n_step": 1,

    # === Exploration ===
    # Turns on annealing schedule for exploration noise. Exploration is
    # annealed from 1.0 to exploration_final_eps over schedule_max_timesteps
    # scaled by exploration_fraction. Original DDPG and TD3 papers do not
    # anneal noise, so this is False by default.
    "exploration_should_anneal": False,
    # Max num timesteps for annealing schedules.
    "schedule_max_timesteps": 100000,
    # Number of env steps to optimize for before returning
    "timesteps_per_iteration": 1000,
    # Fraction of entire training period over which the exploration rate is
    # annealed
    "exploration_fraction": 0.1,
    # Final scaling multiplier for action noise (initial is 1.0)
    "exploration_final_scale": 0.02,
    # valid values: "ou" (time-correlated, like original DDPG paper),
    # "gaussian" (IID, like TD3 paper). Currently this is ignored (FIXME); use
    # smooth_target policy instead.
    "exploration_noise_type": "ou",
    # OU-noise scale; this can be used to scale down magnitude of OU noise
    # before adding to actions (requires "exploration_noise_type" to be "ou")
    "exploration_ou_noise_scale": 0.1,
    # theta for OU
    "exploration_ou_theta": 0.15,
    # sigma for OU
    "exploration_ou_sigma": 0.2,
    # gaussian stddev of act noise for exploration (requires
    # "exploration_noise_type" to be "gaussian")
    "exploration_gaussian_sigma": 0.1,
    # If True parameter space noise will be used for exploration
    # See https://blog.openai.com/better-exploration-with-parameter-noise/
    "parameter_noise": False,
    # Until this many timesteps have elapsed, the agent's policy will be
    # ignored & it will instead take uniform random actions. Can be used in
    # conjunction with learning_starts (which controls when the first
    # optimization step happens) to decrease dependence of exploration &
    # optimization on initial policy parameters. Note that this will be
    # disabled when the action noise scale is set to 0 (e.g during evaluation).
    "pure_exploration_steps": 1000,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": 50000,
    # If True prioritized replay buffer will be used.
    "prioritized_replay": True,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,
    # Whether to LZ4 compress observations
    "compress_observations": False,

    # === Optimization ===
    # Learning rate for the critic (Q-function) optimizer.
    "critic_lr": 1e-3,
    # Learning rate for the actor (policy) optimizer.
    "actor_lr": 1e-3,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 0,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 0.002,
    # If True, use huber loss instead of squared loss for critic network
    # Conventionally, no need to clip gradients if using a huber loss
    "use_huber": False,
    # Threshold of a huber loss
    "huber_threshold": 1.0,
    # Weights for L2 regularization
    "l2_reg": 1e-6,
    # If not None, clip gradients during optimization at this value
    "grad_norm_clipping": None,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1500,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    "sample_batch_size": 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 256,

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you're using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Optimizer class to use.
    "optimizer_class": "SyncReplayOptimizer",
    # Whether to use a distribution of epsilons across workers for exploration.
    "per_worker_exploration": False,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,
})
# __sphinx_doc_end__
# yapf: enable


class DDPGTrainer(DQNTrainer):
    """DDPG implementation in TensorFlow."""
    _name = "DDPG"
    _default_config = DEFAULT_CONFIG
    _policy_graph = DDPGPolicyGraph

    @override(DQNTrainer)
    def _train(self):
        pure_expl_steps = self.config["pure_exploration_steps"]
        if pure_expl_steps:
            # tell workers whether they should do pure exploration
            only_explore = self.global_timestep < pure_expl_steps
            self.local_evaluator.foreach_trainable_policy(
                lambda p, _: p.set_pure_exploration_phase(only_explore))
            for e in self.remote_evaluators:
                e.foreach_trainable_policy.remote(
                    lambda p, _: p.set_pure_exploration_phase(only_explore))
        return super(DDPGTrainer, self)._train()

    @override(DQNTrainer)
    def _make_exploration_schedule(self, worker_index):
        # Override DQN's schedule to take into account
        # `exploration_ou_noise_scale`
        if self.config["per_worker_exploration"]:
            assert self.config["num_workers"] > 1, \
                "This requires multiple workers"
            if worker_index >= 0:
                # TODO: document magic constants (0.4, 7)
                max_index = float(self.config["num_workers"] - 1)
                exponent = 1 + worker_index / max_index * 7
                return ConstantSchedule(0.4**exponent)
            else:
                # local ev should have zero exploration so that eval rollouts
                # run properly
                return ConstantSchedule(0.0)
        elif self.config["exploration_should_anneal"]:
            return LinearSchedule(
                schedule_timesteps=int(self.config["exploration_fraction"] *
                                       self.config["schedule_max_timesteps"]),
                initial_p=1.0,
                final_p=self.config["exploration_final_scale"])
        else:
            # *always* add exploration noise
            return ConstantSchedule(1.0)

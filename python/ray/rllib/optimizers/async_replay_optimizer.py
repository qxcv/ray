"""Implements Distributed Prioritized Experience Replay.

https://arxiv.org/abs/1803.00933"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import time
import threading

import numpy as np
from six.moves import queue

import ray
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.evaluation.policy_evaluator import PolicyEvaluator
from ray.rllib.evaluation.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.optimizers.replay_buffer import PrioritizedReplayBuffer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.actors import TaskPool, create_colocated
from ray.rllib.utils.memory import ray_get_and_free
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.window_stat import WindowStat

SAMPLE_QUEUE_DEPTH = 2
REPLAY_QUEUE_DEPTH = 4
LEARNER_QUEUE_MAX_SIZE = 16


class AsyncReplayOptimizer(PolicyOptimizer):
    """Main event loop of the Ape-X optimizer (async sampling with replay).

    This class coordinates the data transfers between the learner thread,
    remote evaluators (Ape-X actors), and replay buffer actors.

    This has two modes of operation:
        - normal replay: replays independent samples.
        - batch replay: simplified mode where entire sample batches are
            replayed. This supports RNNs, but not prioritization.

    This optimizer requires that policy evaluators return an additional
    "td_error" array in the info return of compute_gradients(). This error
    term will be used for sample prioritization."""

    def __init__(self,
                 local_evaluator,
                 remote_evaluators,
                 learning_starts=1000,
                 buffer_size=10000,
                 prioritized_replay=True,
                 prioritized_replay_alpha=0.6,
                 prioritized_replay_beta=0.4,
                 prioritized_replay_eps=1e-6,
                 train_batch_size=512,
                 sample_batch_size=50,
                 num_replay_buffer_shards=1,
                 max_weight_sync_delay=400,
                 debug=False,
                 batch_replay=False):
        PolicyOptimizer.__init__(self, local_evaluator, remote_evaluators)

        self.debug = debug
        self.batch_replay = batch_replay
        self.replay_starts = learning_starts
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_eps = prioritized_replay_eps
        self.max_weight_sync_delay = max_weight_sync_delay
        self.train_batch_size = train_batch_size

        if self.batch_replay:
            replay_cls = BatchReplayActor
        else:
            replay_cls = ReplayActor
        # XXX: this is going to be a bottleneck if I try to scale out
        # optimisers.
        self.replay_actors = create_colocated(replay_cls, [
            num_replay_buffer_shards,
            learning_starts,
            buffer_size,
            train_batch_size,
            prioritized_replay_alpha,
            prioritized_replay_beta,
            prioritized_replay_eps,
        ], num_replay_buffer_shards)

        # TODO: replace this with an actual actor so that it can run
        # concurrently
        # self.learner = LearnerThread(self.local_evaluator)
        # self.learner.start()
        self.param_server = ApexParameterServer.remote()

        self.last_num_steps_trained_time = time.time()
        self.last_num_steps_trained = self.num_steps_trained

        # Stats
        self.timers = {
            k: TimerStat()
            for k in [
                "put_weights", "get_samples", "sample_processing",
                "replay_processing", "update_priorities", "train", "sample"
            ]
        }
        self.num_weight_syncs = 0
        self.num_samples_dropped = 0
        self.learning_started = False

        # Number of worker steps since the last weight update
        self.steps_since_update = {}

        # Otherwise kick of replay tasks for local gradient updates
        self.replay_tasks = TaskPool()
        for ra in self.replay_actors:
            for _ in range(REPLAY_QUEUE_DEPTH):
                self.replay_tasks.add(ra, ra.replay.remote())

        # Kick off async background sampling
        self.sample_tasks = TaskPool()
        print("NOT calling set_evalutors")
        self.remote_evaluators = None
        # assert self.remote_evaluators is None, \
        #     "this should be set later on; my hacks only work with DQNTrainer"
        self.remote_opt_evaluator = None
        self.learner_stats =None

    @override(PolicyOptimizer)
    def step(self):
        assert len(self.remote_evaluators) > 0
        start = time.time()
        sample_timesteps, new_train_timesteps, stats = self._step()
        time_delta = time.time() - start
        self.timers["sample"].push(time_delta)
        self.timers["sample"].push_units_processed(sample_timesteps)
        if new_train_timesteps > 0:
            self.learning_started = True
        if self.learning_started \
           and new_train_timesteps > self.num_steps_trained:
            train_td = time.time() - self.last_num_steps_trained_time
            self.timers["train"].push(train_td)
            train_step_d = new_train_timesteps - self.num_steps_trained
            self.timers["train"].push_units_processed(train_step_d)
            self.last_num_steps_trained_time = time.time()
            self.learner_stats = stats
        self.num_steps_sampled += sample_timesteps
        self.num_steps_trained = new_train_timesteps

    @override(PolicyOptimizer)
    def stop(self):
        for r in self.replay_actors:
            r.__ray_terminate__.remote()
        # self.learner.stopped = True

    @override(PolicyOptimizer)
    def reset(self, remote_evaluators):
        self.remote_evaluators = remote_evaluators
        self.sample_tasks.reset_evaluators(remote_evaluators)

    @override(PolicyOptimizer)
    def stats(self):
        replay_stats = ray_get_and_free(self.replay_actors[0].stats.remote(
            self.debug))
        timing = {
            "{}_time_ms".format(k): round(1000 * self.timers[k].mean, 3)
            for k in self.timers
        }
        # timing["learner_grad_time_ms"] = round(
        #     1000 * self.learner.grad_timer.mean, 3)
        # timing["learner_dequeue_time_ms"] = round(
        #     1000 * self.learner.queue_timer.mean, 3)
        stats = {
            # XXX: sample_throughput is complete garbage, and train_throughput
            # WAS complete garbage before I changed it. Before, step() was just
            # timing how long it took to *talk to the actors for one
            # iteration*, and using that as the denominator in the throughput
            # computation. Trust the num_steps_trained numbers over the
            # _throughput numbers!
            "sample_throughput": round(self.timers["sample"].mean_throughput,
                                       3),
            "train_throughput": round(self.timers["train"].mean_throughput, 3),
            "num_weight_syncs": self.num_weight_syncs,
            "num_samples_dropped": self.num_samples_dropped,
            # "learner_queue": self.learner.learner_queue_size.stats(),
            "replay_shard_0": replay_stats,
            "async_time_sample_processing": round(
                self.timers["sample_processing"].mean, 3),
            "async_time_replay_processing": round(
                self.timers["replay_processing"].mean, 3),
            "async_time_update_priorities": round(
                self.timers["update_priorities"].mean, 3),
        }
        debug_stats = {
            "timing_breakdown": timing,
            "pending_sample_tasks": self.sample_tasks.count,
            "pending_replay_tasks": self.replay_tasks.count,
        }
        if self.debug:
            stats.update(debug_stats)
        if self.learner_stats:
            stats["learner"] = self.learner_stats
        return dict(PolicyOptimizer.stats(self), **stats)

    # For https://github.com/ray-project/ray/issues/2541 only
    def _set_evaluators(self, remote_evaluators, remote_opt_evaluator):
        print("calling set_evaluators()")
        self.remote_evaluators = remote_evaluators
        weights = self.local_evaluator.get_weights()
        for ev in self.remote_evaluators:
            ev.set_weights.remote(weights)
            self.steps_since_update[ev] = 0
            for _ in range(SAMPLE_QUEUE_DEPTH):
                self.sample_tasks.add(ev, ev.sample_with_count.remote())
        self.remote_opt_evaluator = remote_opt_evaluator
        self.remote_opt_evaluator.do_apex_setup.remote(
            weights, self.replay_actors, self.train_batch_size,
            self.param_server)
        self.remote_opt_evaluator.train_forever.remote()

    def _step(self):
        sample_timesteps = 0
        weights = None

        with self.timers["sample_processing"]:
            completed = list(self.sample_tasks.completed())
            counts = ray_get_and_free([c[1][1] for c in completed])
            for i, (ev, (sample_batch, count)) in enumerate(completed):
                sample_timesteps += counts[i]

                # HACK THIS IS WRONG! The replay actors should send their data
                # *straight to a random shard*. It doesn't make sense to
                # introduce a bottleneck here. Fix this later.
                # Send the data to the replay buffer
                random.choice(
                    self.replay_actors).add_batch.remote(sample_batch)

                # Update weights if needed
                self.steps_since_update[ev] = counts[i]
                if self.steps_since_update[ev] >= self.max_weight_sync_delay:
                    # Note that it's important to pull new weights once
                    # updated to avoid excessive correlation between actors
                    if weights is None or self.learner.weights_updated:
                        self.learner.weights_updated = False
                        with self.timers["put_weights"]:
                            weights = ray.put(
                                self.local_evaluator.get_weights())
                    ev.set_weights.remote(weights)
                    self.num_weight_syncs += 1
                    self.steps_since_update[ev] = 0

                # Kick off another sample request
                self.sample_tasks.add(ev, ev.sample_with_count.remote())

        # we *always* pull weights from param server on every iteration; that
        # helps ensure that we don't get too out of date (also the amount of
        # compute done by this actor doesn't matter that much, since the actual
        # learner is independent)
        new_weights_h = self.param_server.get_weights.remote()
        new_weights, new_train_timesteps, stats \
            = ray_get_and_free(new_weights_h)
        if new_weights is not None:
            self.local_evaluator.set_weights(new_weights)

        return sample_timesteps, new_train_timesteps, stats


@ray.remote(num_cpus=0)
class ReplayActor(object):
    """A replay buffer shard.

    Ray actors are single-threaded, so for scalability multiple replay actors
    may be created to increase parallelism."""

    def __init__(self, num_shards, learning_starts, buffer_size,
                 train_batch_size, prioritized_replay_alpha,
                 prioritized_replay_beta, prioritized_replay_eps):
        self.replay_starts = learning_starts // num_shards
        self.buffer_size = buffer_size // num_shards
        self.train_batch_size = train_batch_size
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_eps = prioritized_replay_eps

        def new_buffer():
            return PrioritizedReplayBuffer(self.buffer_size,
                                           alpha=prioritized_replay_alpha)

        self.replay_buffers = collections.defaultdict(new_buffer)

        # Metrics
        self.add_batch_timer = TimerStat()
        self.replay_timer = TimerStat()
        self.update_priorities_timer = TimerStat()
        self.num_added = 0

    def get_host(self):
        return os.uname()[1]

    def add_batch(self, batch):
        # Handle everything as if multiagent
        if isinstance(batch, SampleBatch):
            batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
        with self.add_batch_timer:
            for policy_id, s in batch.policy_batches.items():
                for row in s.rows():
                    self.replay_buffers[policy_id].add(
                        row["obs"], row["actions"], row["rewards"],
                        row["new_obs"], row["dones"], row["weights"])
        self.num_added += batch.count

    def replay(self, force=False, batch_size=None, uniform=False):
        if self.num_added < self.replay_starts and not force:
            return None

        if not batch_size:
            batch_size = self.train_batch_size

        with self.replay_timer:
            samples = {}
            for policy_id, replay_buffer in self.replay_buffers.items():
                if uniform:
                    (obses_t, actions, rewards, obses_tp1,
                     dones) = replay_buffer.sample_uniform(batch_size)
                    samples[policy_id] = SampleBatch({
                        "obs": obses_t,
                        "actions": actions,
                        "rewards": rewards,
                        "new_obs": obses_tp1,
                        "dones": dones,
                    })
                else:
                    (obses_t, actions, rewards, obses_tp1, dones, weights,
                     batch_indexes) = replay_buffer.sample(
                         batch_size, beta=self.prioritized_replay_beta)
                    samples[policy_id] = SampleBatch({
                        "obs": obses_t,
                        "actions": actions,
                        "rewards": rewards,
                        "new_obs": obses_tp1,
                        "dones": dones,
                        "weights": weights,
                        "batch_indexes": batch_indexes
                    })
            return MultiAgentBatch(samples, batch_size)

    def update_priorities(self, prio_dict):
        with self.update_priorities_timer:
            for policy_id, (batch_indexes, td_errors) in prio_dict.items():
                new_priorities = (np.abs(td_errors) +
                                  self.prioritized_replay_eps)
                self.replay_buffers[policy_id].update_priorities(
                    batch_indexes, new_priorities)

    def stats(self, debug=False):
        stat = {
            "add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
            "replay_time_ms": round(1000 * self.replay_timer.mean, 3),
            "update_priorities_time_ms": round(
                1000 * self.update_priorities_timer.mean, 3),
        }
        for policy_id, replay_buffer in self.replay_buffers.items():
            stat.update({
                "policy_{}".format(policy_id): replay_buffer.stats(debug=debug)
            })
        return stat


# note: we set num_cpus=0 to avoid failing to create replay actors when
# resources are fragmented. This isn't ideal.
@ray.remote(num_cpus=0)
class BatchReplayActor(object):
    """The batch replay version of the replay actor.

    This allows for RNN models, but ignores prioritization params.
    """

    def __init__(self, num_shards, learning_starts, buffer_size,
                 train_batch_size, prioritized_replay_alpha,
                 prioritized_replay_beta, prioritized_replay_eps):
        self.replay_starts = learning_starts // num_shards
        self.buffer_size = buffer_size // num_shards
        self.train_batch_size = train_batch_size
        self.buffer = []

        # Metrics
        self.num_added = 0
        self.cur_size = 0

    def get_host(self):
        return os.uname()[1]

    def add_batch(self, batch):
        # Handle everything as if multiagent
        if isinstance(batch, SampleBatch):
            batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
        self.buffer.append(batch)
        self.cur_size += batch.count
        self.num_added += batch.count
        while self.cur_size > self.buffer_size:
            self.cur_size -= self.buffer.pop(0).count

    def replay(self, force=False):
        if self.num_added < self.replay_starts and not force:
            return None
        return random.choice(self.buffer)

    def update_priorities(self, prio_dict):
        pass

    def stats(self, debug=False):
        stat = {
            "cur_size": self.cur_size,
            "num_added": self.num_added,
        }
        return stat


class ReplayActorSampler(object):
    """Simple class to keep pulling experience batches out of some replay
    actors. This is not meant to be an actor, and is NOT THREAD-SAFE OR
    MULTIPROCESSING-SAFE! You're meant to have one of these per training
    process/thread."""

    def __init__(self, replay_actors, batch_size=None, uniform=True,
                 force=True):
        self.replay_actors = replay_actors
        self.batch_size = batch_size
        # we lazily create replay_tasks so that they get spawned *after*
        # training starts; otherwise the first few sampled batches are empty
        # b/c there's nothing the replay buffer :(
        self.replay_tasks = None
        self.uniform = uniform
        self.force = force

    def _add_replay_task(self, ra):
        if self.replay_tasks is None:
            self._make_replay_tasks()
        self.replay_tasks.add(
            ra,
            ra.replay.remote(force=self.force,
                             batch_size=self.batch_size,
                             uniform=self.uniform))

    def _make_replay_tasks(self):
        if self.replay_tasks is not None:
            return
        self.replay_tasks = TaskPool()
        for ra in self.replay_actors:
            for _ in range(REPLAY_QUEUE_DEPTH):
                self._add_replay_task(ra)

    def get_batch(self):
        if self.replay_tasks is None:
            self._make_replay_tasks()
        ra_replay = list(self.replay_tasks.completed(
            blocking_wait=True, num_returns=1, block_timeout=900.0))
        assert len(ra_replay) >= 1, \
            "didn't get result in 900s; probably timed out"
        ra_replay, = ra_replay
        ra, replay = ra_replay
        self._add_replay_task(ra)
        # TODO: figure out whether this is the best way to maximise throughput;
        # should I be running this in a BG thread or something?
        replay = ray_get_and_free(replay)
        return replay, ra


@ray.remote(num_cpus=0)
class ApexParameterServer(object):
    """Used for storing parameters set by ApexLearnerEvaluator."""
    def __init__(self):
        self.weights = None
        self.stats = None
        self.last_ts = 0

    def set_weights(self, weights, ts, stats):
        self.last_ts = ts
        self.weights = weights
        self.stats = stats

    def get_weights(self, last_ts=None):
        if last_ts is None or last_ts < self.latest_ts:
            return self.weights, self.last_ts, self.stats
        # return nothing if no new weights have been pushed
        return None, self.last_ts, self.stats


class ApexLearnerEvaluator(PolicyEvaluator):
    """Having a thread is stupid, perf dies due to GIL contention. Run this in
    an actor instead using LearnerEvaluator.as_remote(). This is a subclass of
    PolicyEvaluator so that policy evaluation & learning can take place in the
    same worker (rather than needing separate workers & tasks to coordinate
    with the PolicyEvaluator)."""

    def _init(self):
        self.__sample_timer = TimerStat()
        self.__replay_timer = TimerStat()
        self.__grad_timer = TimerStat()
        self.__stats = {}
        self.__train_timesteps = 0
        self.__last_push_h = None

    def do_apex_setup(self, weights, replay_workers, batch_size, param_server):
        self.set_weights(weights)
        self.__replay_workers = replay_workers
        self.__batch_size = batch_size
        self.__sample_batcher = ReplayActorSampler(
            replay_workers, force=False, uniform=False)
        self.__param_server = param_server
        self.__last_push = 0
        self._push_weights()

    def _push_weights(self):
        if self.__last_push_h is not None:
            # wait for last push to complete
            ray.wait([self.__last_push_h])
        self.__last_push_h = self.__param_server.set_weights.remote(
            self.get_weights(), self.__train_timesteps, self.__stats)
        self.__last_push = self.__train_timesteps

    def train_forever(self):
        """Train until the actor gets shut down."""
        while True:
            samples, ra = self.__sample_batcher.get_batch()
            if samples is not None:
                prio_dict = {}
                # with self.grad_timer:
                grad_out = self.learn_on_batch(samples)
                for pid, info in grad_out.items():
                    prio_dict[pid] = (
                        samples.policy_batches[pid].data.get(
                            "batch_indexes"),
                        info.get("td_error"))
                    self.__stats[pid] = get_learner_stats(info)
                ra.update_priorities.remote(prio_dict)
                self.__train_timesteps += samples.total()
                # TODO: make this push interval configurable
                if self.__train_timesteps - self.__last_push >= 400:
                    self._push_weights()

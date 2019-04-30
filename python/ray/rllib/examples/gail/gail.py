"""A distributed implementation of GAIL using the TD3 optimizer with APE-X."""

import collections
import os
import time

# TODO: add a check to make sure these are at correct value; warn or error out
# if not
# os.environ["MKL_NUM_THREADS"] = str(1)
# os.environ["OMP_NUM_THREADS"] = str(1)

import gym  # noqa: E402

import numpy as np  # noqa: E402

import ray  # noqa: E402
from ray import tune  # noqa: E402
from ray.experimental import named_actors  # noqa: E402
from ray.rllib.agents.ddpg import ApexTD3Trainer, TD3Trainer  # noqa: E402
from ray.rllib.evaluation.metrics import collect_metrics  # noqa: E402
from ray.rllib.evaluation.sample_batch import SampleBatch  # noqa: E402
from ray.rllib.offline.json_writer import JsonWriter  # noqa: E402
from ray.rllib.offline.json_reader import JsonReader  # noqa: E402
from ray.rllib.optimizers.async_replay_optimizer \
    import REPLAY_QUEUE_DEPTH  # noqa: E402
from ray.rllib.models import ModelCatalog  # noqa: E402
from ray.rllib.utils.memory import ray_get_and_free  # noqa: E402
from ray.rllib.utils.actors import TaskPool  # noqa: E402
from ray.tune.logger import CSVLogger, pretty_print  # noqa: E402
from ray.tune.util import merge_dicts  # noqa: E402

from sacred import Experiment  # noqa: E402
from sacred.observers import FileStorageObserver  # noqa: E402

import tensorflow as tf  # noqa: E402

ex = Experiment('gail')


@ray.remote(num_cpus=1)
class DiscriminatorActor(object):
    def __init__(self, env_name, disc_config, expert_config, td3_conf,
                 tf_par_args, param_server, replay_actors):
        env = gym.make(env_name)
        self.discriminator = Discriminator(disc_config, env.observation_space,
                                           env.action_space)
        del env
        self.disc_updates = 0
        self.disc_samples_seen = 0
        self.min_iter_time_s = td3_conf["min_iter_time_s"]
        self.min_iter_steps = disc_config["updates_per_epoch"]
        self.rew_update_period = disc_config["sync_at_least_every"]
        self._wait_handle = None

        self.half_batch_size = max(1, disc_config["batch_size"] // 2)
        dataset_size = self.half_batch_size
        demos = load_latest_demos(expert_config["expert_dir"])
        self.ds_range = np.arange(dataset_size)
        self.real_obs_ds = demos['obs']
        self.real_act_ds = demos['actions']
        self.param_server = param_server

        sess_conf = tf.ConfigProto(**tf_par_args)
        self.sess = tf.Session(config=sess_conf)
        self.sess.run(tf.global_variables_initializer())
        self.reward_vars = ray.experimental.tf_utils.TensorFlowVariables(
            self.discriminator.reward, self.sess)

        self.replay_sampler = LocalReplaySampler(
            replay_actors,
            # use 1/2 of real batch size b/c when training discrim we mix with
            # and other 1/2 expert samples
            batch_size=self.half_batch_size)

    def train_epoch(self):
        """Train for at least num_steps or until min_time has elapsed,
        whichever happens last."""
        disc_loss = disc_acc = 0.0
        steps = 0
        start_time = time.time()
        min_steps = self.min_iter_steps
        min_time = self.min_iter_time_s
        while steps < min_steps or time.time() - start_time < min_time:
            if (steps + 1) % self.rew_update_period == 0:
                self.push_weights()
            disc_loss, disc_acc = self._train_step()
            steps += 1
        self.push_weights()
        return {
            "disc_loss": disc_loss,
            "disc_acc": disc_acc,
            "disc_updates": self.disc_updates,
            "disc_samples_seen": self.disc_samples_seen,
        }

    def get_reward_weights(self):
        return self.reward_vars.get_weights()

    def push_weights(self):
        if self._wait_handle is not None:
            # wait for previous request to finish
            ray_get_and_free(self._wait_handle)
            self._wait_handle = None
        # launch a new push asynchronously
        weights = self.get_reward_weights()
        self._wait_handle = self.param_server.set_weights.remote(weights)

    def _train_step(self):
        ma_batch = self.replay_sampler.get_batch()
        if not ma_batch.policy_batches:
            # sometimes this happens at the beginning of training b/c
            # there are not enough samples in the replay buffer
            return 0.0, 0.0
        fake_batch = ma_batch.policy_batches["default_policy"]
        fake_obs_batch = fake_batch['obs']
        fake_act_batch = fake_batch['actions']
        real_batch_inds = np.random.choice(self.ds_range,
                                           size=(self.half_batch_size, ))
        label_batch = np.zeros((2 * self.half_batch_size, ))
        label_batch[self.half_batch_size:] = 1.0
        obs_batch = np.concatenate(
            [fake_obs_batch, self.real_obs_ds[real_batch_inds]], axis=0)
        act_batch = np.concatenate(
            [fake_act_batch, self.real_act_ds[real_batch_inds]], axis=0)

        # update the discriminator
        self.disc_updates += 1
        self.disc_samples_seen += len(label_batch)
        _, disc_loss, disc_acc = self.sess.run(
            [
                self.discriminator.update_op, self.discriminator.loss,
                self.discriminator.accuracy
            ],
            feed_dict={
                self.discriminator.obs_t: obs_batch,
                self.discriminator.act_t: act_batch,
                self.discriminator.is_real_t: label_batch,
                self.discriminator.is_training: True,
            })
        return disc_loss, disc_acc


@ray.remote(num_cpus=0)
class DiscriminatorParameterServer(object):
    """Stores parameters for the discriminator in a commonly-accessible
    place."""

    def __init__(self):
        self.weights = None

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        # might be None initially
        return self.weights


@ray.remote(num_cpus=1)
class TrainerActor(object):
    def __init__(self, env_name, td3_conf):
        self.trainer = ApexTD3Trainer(env=env_name, config=td3_conf)

    def get_replay_actors(self):
        return self.trainer.optimizer.replay_actors

    def train_epoch(self):
        return self.trainer.train()


class LocalReplaySampler(object):
    """Simple class to keep pulling experience batches out of some replay
    actors. NOT THREAD-SAFE OR MULTIPROCESSING-SAFE! You're meant to have one
    of these per discriminator training process/thread."""

    def __init__(self, replay_actors, batch_size):
        self.replay_actors = replay_actors
        self.batch_size = batch_size
        # we lazily create replay_tasks so that they get spawned *after*
        # training starts; otherwise the first few sampled batches are empty
        # b/c there's nothing the replay buffer :(
        self.replay_tasks = None
        # queued batches
        self.batch_queue = collections.deque()

    def _add_replay_task(self, ra):
        if self.replay_tasks is None:
            self._make_replay_tasks()
        self.replay_tasks.add(
            ra,
            ra.replay.remote(force=True,
                             batch_size=self.batch_size,
                             uniform=True))

    def _make_replay_tasks(self):
        if self.replay_tasks is None:
            self.replay_tasks = TaskPool()
            for ra in self.replay_actors:
                for _ in range(REPLAY_QUEUE_DEPTH):
                    self._add_replay_task(ra)

    def _refresh(self, blocking=False):
        # FIXME: this code is partially copied from AsyncReplayOptimizer; I
        # should unify the two to de-duplicate (maybe use this class in both
        # cases)
        if self.replay_tasks is None:
            self._make_replay_tasks()
        for ra, replay in self.replay_tasks.completed(blocking_wait=blocking):
            self._add_replay_task(ra)
            samples = ray_get_and_free(replay)
            # Defensive copy against plasma crashes, see #2610 #3452
            self.batch_queue.append((ra, samples and samples.copy()))

    def get_batch(self):
        if len(self.batch_queue) == 0:
            self._refresh(blocking=True)
            result = self.batch_queue.popleft()
        else:
            result = self.batch_queue.popleft()
            self._refresh(blocking=False)
        _, batch = result
        return batch


class Discriminator(object):
    def __init__(self, discrim_config, obs_space, act_space):
        assert isinstance(obs_space, gym.spaces.Box) \
            and 1 == len(obs_space.shape)
        assert isinstance(act_space, gym.spaces.Box) \
            and 1 == len(act_space.shape)
        obs_dim, = obs_space.shape
        act_dim, = act_space.shape
        self.discrim_config = discrim_config
        self.obs_space = obs_space
        self.act_space = act_space
        self.model_options = discrim_config['model']
        self.obs_t = tf.placeholder(tf.float32, (None, obs_dim), 'obs_t')
        self.act_t = tf.placeholder(tf.float32, (None, act_dim), 'act_t')
        # 1 = real, 0 = fake
        self.is_real_t = tf.placeholder(tf.float32, (None, ), 'is_real_t')
        self.is_training = tf.placeholder(tf.bool, (), 'is_training')
        # TODO: make sure we get the right scope so that we can use TFVariables
        # between policy & discriminator
        with tf.variable_scope('default_policy/reward_net') as scope:
            self.logits = self._make_discrim_logits(self.obs_t, self.act_t)
            discrim_train_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        # Softplus of negative logits is equivalent to -log(D(u))
        self.reward = tf.nn.softplus(-self.logits)
        # We assume that batches are balanced; in that case, the below
        # expression should give a reasonably stable estimate of
        # E_{novice}[log(D(s,a))] + E_{expert}[log(1-D(s,a))].
        # self.loss = tf.reduce_mean(
        #     -(1 - self.is_real_t) * tf.nn.softplus(-self.logits) -
        #     self.is_real_t * tf.nn.softplus(self.logits))
        # sigmoid = tf.nn.sigmoid(self.logits)
        # self.loss = tf.reduce_mean((1 - self.is_real_t) * tf.log(sigmoid) +
        #                            self.is_real_t * tf.log(1 - sigmoid))
        # XXX: I think this is wrong for GANs; need to check the DAC
        # implementation, and maybe the original JSD-GAN paper (which had
        # something non-obvious to say about exactly this thing).
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.is_real_t,
                                                    logits=self.logits))
        reg = tf.contrib.layers.l1_regularizer(1e-2)
        self.loss += sum(reg(v) for v in discrim_train_vars)
        real_labels = self.is_real_t > 0.5
        out_labels = self.logits > 0
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(real_labels, out_labels), tf.float32))
        self.optimiser = tf.train.AdamOptimizer(
            learning_rate=discrim_config['lr'])
        self.update_op = self.optimiser.minimize(self.loss,
                                                 var_list=discrim_train_vars)

    def _make_discrim_logits(self, in_state, in_action):
        in_data = tf.concat([in_state, in_action], axis=1)
        in_dict = {
            "obs": in_data,
            "is_training": self.is_training,
        }
        model_out = ModelCatalog.get_model(input_dict=in_dict,
                                           obs_space=self.obs_space,
                                           action_space=self.act_space,
                                           num_outputs=1,
                                           options=self.model_options)
        out_value = tf.squeeze(model_out.outputs, axis=-1)
        return out_value


def load_latest_demos(demo_dir):
    demo_paths = []
    for file_name in os.listdir(demo_dir):
        if os.path.splitext(file_name)[-1].lower() == '.json':
            demo_path = os.path.join(demo_dir, file_name)
            demo_paths.append(demo_path)
    # choose the latest one just by sorting names
    demo_path = sorted(demo_paths)[-1]
    # now read samples back out
    # this api is so fucked
    demo_reader = JsonReader(demo_path)
    demos = SampleBatch.concat_samples(list(demo_reader))
    return demos


@ex.config
def cfg():
    env_name = 'InvertedPendulum-v2'  # noqa: F841
    max_time_s = 3600  # noqa: F841
    # TensorFlow configurations
    tf_configs = {  # noqa: F841
        "discrim": {
            "gpu_num": None,
            "tf_threads": None,
        },
        "td3_local_learner": {
            "gpu_num": None,
            "tf_threads": None,
        },
        "td3_defaults": {
            # these should generally not use GPU, and should only need one
            # thread each (they don't do much computation)
            "gpu_num": None,
            "tf_threads": 1,
        }
    }
    discrim_config = {  # noqa: F841
        "lr": 1e-3,
        "batch_size": 512,
        "updates_per_epoch": 50,
        # update shared weights at least once every time we take this many
        # steps
        "sync_at_least_every": 25,
        "model": {
            "fcnet_hiddens": [256, 128],
            "fcnet_activation": "relu"
        }
    }
    # TODO: figure out how to put a unique expt ID in here (maybe Sacred can
    # help?)
    data_dir = 'data'  # noqa: F841
    expert_config = {  # noqa: F841
        "train_timesteps": 50000,
        "expert_subdir": 'demos',
        "output_timesteps": 10000,
    }
    td3_conf = {  # noqa: F841
        "evaluation_interval": 10,
        "evaluation_num_episodes": 5,
        "timesteps_per_iteration": 5000,
        "min_iter_time_s": 30,
        "train_batch_size": 256,
    }


@ex.config_hook
def fix_cfg_relpaths(config, command_name, logger):
    full_data_dir = os.path.join(os.getcwd(), config["data_dir"],
                                 config["env_name"])
    return {
        "full_data_dir": full_data_dir,
        "expert_config": {
            "expert_dir": os.path.join(
                full_data_dir, config["expert_config"]["expert_subdir"])
        }
    }


def make_tf_config_args(tf_threads=None, gpu_num=None):
    """Make the appropriate kwargs """
    cproto_args = {}
    if tf_threads is not None:
        cproto_args["inter_op_parallelism_threads"] = tf_threads
        cproto_args["intra_op_parallelism_threads"] = tf_threads
    if gpu_num is None:
        cproto_args['gpu_options'] = dict(visible_device_list="")
    else:
        cproto_args['gpu_options'] = dict(visible_devices=str(gpu_num),
                                          allow_growth=True)
    return cproto_args


@ex.main
def main(env_name, discrim_config, expert_config, full_data_dir, tf_configs,
         td3_conf, max_time_s, _config, _run):
    """Run GAIL on a given environment. Assumes that you've already used
    train_expert to collect expert demos for the environment."""
    out_dir = os.path.join(full_data_dir, "gail_run_%s" % _run._id)
    os.makedirs(out_dir, exist_ok=True)
    stat_logger = CSVLogger(config=_config, logdir=out_dir)
    ray.init()
    # algo config dict
    td3_base_conf = {
        "reward_model": discrim_config["model"],
        # this is used by the local evaluator, which (in the async replay
        # buffer code) is responsible for making actual optimiser steps
        "local_evaluator_tf_session_args": make_tf_config_args(
            **tf_configs["td3_local_learner"]),
        # this is used by all the rollout evaluators & anything else that gets
        # instantiated
        "tf_session_args": make_tf_config_args(**tf_configs["td3_defaults"]),
        # we manage GPUs ourselves
        "num_gpus": 0,
    }
    td3_conf = merge_dicts(td3_base_conf, td3_conf)
    disc_param_server = DiscriminatorParameterServer.remote()
    named_actors.register_actor("global_reward_params", disc_param_server)
    trainer_actor = TrainerActor.remote(env_name, td3_conf)
    replay_actors_handle = trainer_actor.get_replay_actors.remote()
    discrim_actor = DiscriminatorActor.remote(
        env_name, discrim_config, expert_config, td3_conf,
        make_tf_config_args(**tf_configs["discrim"]), disc_param_server,
        replay_actors_handle)

    itr = 0
    start_time = time.time()
    while time.time() - start_time < max_time_s:
        # train for a little while
        trainer_handle = trainer_actor.train_epoch.remote()
        discrim_handle = discrim_actor.train_epoch.remote()
        trainer_result, discrim_result = ray_get_and_free(
            [trainer_handle, discrim_handle])
        full_result = merge_dicts(trainer_result, discrim_result)
        del full_result["config"]

        # end of epoch, print stats
        print('Epoch %d:' % itr)
        print("  {}".format(pretty_print(full_result).replace("\n", "\n  ")))

        # also store in file
        stat_logger.on_result(full_result)
        # TODO: do this less often b/c each call has O(T) cost (it actually
        # copies the whole stats file)
        _run.add_artifact(stat_logger._file.name)

        # next epoch!
        itr += 1

    ray.shutdown()


@ex.command
def train_expert(env_name, expert_config, _run, _config):
    """Train a policy using the true reward function for the given
    environment."""
    expert_dir = expert_config["expert_dir"]
    ray.init()
    os.makedirs(expert_dir, exist_ok=True)
    opt_dict = {
        "env": env_name,
        "evaluation_interval": 10,
        "evaluation_num_episodes": 10,
    }
    tune_rv = tune.run(
        TD3Trainer,
        stop={"timesteps_total": expert_config["train_timesteps"]},
        config=opt_dict,
        checkpoint_at_end=True)
    # now we run the trained policy for out_steps interactions & save the
    # result somewhere
    checkpoint_path = tune_rv[-1]._checkpoint.value
    assert checkpoint_path is not None, "fuck"
    exec_dict = {
        "exploration_gaussian_sigma": 0.0,
        "exploration_ou_noise_scale": 0.0,
        "pure_exploration_steps": 0,
        "parameter_noise": False,
    }
    run_trainer = TD3Trainer(env=env_name, config=exec_dict)
    run_trainer.restore(checkpoint_path)
    writer = JsonWriter(expert_dir)
    steps_sampled = 0
    out_steps = expert_config["output_timesteps"]
    while steps_sampled < out_steps:
        sample = run_trainer.local_evaluator.sample()
        writer.write(sample)
        batch_size = sample[sample.CUR_OBS].shape[0]
        steps_sampled = steps_sampled + batch_size
    metrics = collect_metrics(run_trainer.local_evaluator)
    print("Done. Write %d interactions to %s" % (steps_sampled, writer.path))
    print("Eval metrics:")
    # copied from Tune's trial.py thing
    print("  {}".format(pretty_print(metrics).replace("\n", "\n  ")))


def _init():
    # TODO get the FSObserver output path from config
    observer = FileStorageObserver.create("data/sacred-runs/")
    ex.observers.append(observer)
    ex.run_commandline()


if __name__ == '__main__':
    _init()

#!/usr/bin/env python3
"""Comparison of distributed SGD on a single node with native TF optimization.
Want to see which is faster for a given level of concurrency."""

import argparse
import csv
import os
import subprocess
import sys
import time

import gym
import numpy as np
import tensorflow as tf

import ray
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.utils.memory import ray_get_and_free

# 8 will probably favour the local approach
# other values to try: 1, 16, 32
ENV_NAME = 'HalfCheetah-v2'
DEMO_DIR = "./data/HalfCheetah-v2/demos/"


class ExperimentDiscriminator(object):
    # slightly stripped-down version of discriminator class from gail.py
    def __init__(self, obs_space, act_space, disc_config):
        obs_dim, = obs_space.shape
        act_dim, = act_space.shape
        self.discrim_config = disc_config
        self.obs_space = obs_space
        self.act_space = act_space
        self.model_options = self.discrim_config['model']
        self.obs_t = tf.placeholder(tf.float32, (None, obs_dim), 'obs_t')
        self.act_t = tf.placeholder(tf.float32, (None, act_dim), 'act_t')
        # 1 = real, 0 = fake
        self.is_real_t = tf.placeholder(tf.float32, (None, ), 'is_real_t')
        self.is_training = tf.constant(True, name='is_training')
        with tf.variable_scope('reward_net') as scope:
            self.logits = self._make_discrim_logits(self.obs_t, self.act_t)
            self.discrim_train_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.is_real_t,
                                                    logits=self.logits))
        reg = tf.contrib.layers.l1_regularizer(1e-2)
        self.loss += sum(reg(v) for v in self.discrim_train_vars)
        real_labels = self.is_real_t > 0.5
        out_labels = self.logits > 0
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(real_labels, out_labels), tf.float32))
        self.optimiser = tf.train.AdamOptimizer(
            learning_rate=self.discrim_config['lr'])
        self.grads_vars = self.optimiser.compute_gradients(
            self.loss, var_list=self.discrim_train_vars)
        self.grads = [grad for grad, var in self.grads_vars]
        self.grads_ph = [
            tf.placeholder("float32", shape=var.get_shape())
            for grad, var in self.grads_vars
        ]
        self.grads_vars_ph = zip(self.grads_ph, self.discrim_train_vars)
        self.apply_op = self.optimiser.apply_gradients(self.grads_vars_ph)
        # one-step op that does both of the above
        self.update_op = self.optimiser.minimize(
            self.loss, var_list=self.discrim_train_vars)
        self.weight_phs = [
            tf.placeholder(tf.float32, shape=var.get_shape())
            for var in self.discrim_train_vars
        ]
        self.assign_weight_op = tf.group([
            tf.assign(w, w_ph)
            for w, w_ph in zip(self.discrim_train_vars, self.weight_phs)
        ])

    def apply_gradients(self, grads, sess):
        # convenience fn to apply gradients, copied from Ray parameter server
        # example
        feed_dict = dict(zip(self.grads_ph, grads))
        sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

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


@ray.remote(num_cpus=1)
class DistSGDActor():
    def __init__(self, good_demos, bad_demos, cores, obs_space, act_space):
        # assume there's no latency to get data
        # TODO: make more realistic experiment in which I pull data out of a
        # single replay shard
        self.good_demos = good_demos
        self.bad_demos = bad_demos
        self.sess, self.disc = make_sess_disc(obs_space, act_space, cores)

    def get_grads(self, minibatch_size):
        obs_batch, act_batch, label_batch = get_obs_act_label_batch(
            self.good_demos, self.bad_demos, minibatch_size)
        grads, loss, acc = self.sess.run(
            [self.disc.grads, self.disc.loss, self.disc.accuracy],
            feed_dict={
                self.disc.obs_t: obs_batch,
                self.disc.act_t: act_batch,
                self.disc.is_real_t: label_batch,
                self.disc.is_training: True,
            })
        return grads, loss, acc

    def push_weights(self, weights):
        # apply weights to this actor
        feed_dict = dict(zip(self.disc.weight_phs, weights))
        self.sess.run(self.disc.assign_weight_op, feed_dict=feed_dict)


def load_demo(demo_path):
    demo_reader = JsonReader(demo_path)
    demos = SampleBatch.concat_samples(list(demo_reader))
    return demos


def make_sess_disc(obs_space, act_space, cores, disc_config):
    discrim = ExperimentDiscriminator(obs_space, act_space, disc_config)
    tf_par_args = {
        "inter_op_parallelism_threads": cores,
        "intra_op_parallelism_threads": cores
    }
    sess_conf = tf.ConfigProto(**tf_par_args)
    sess = tf.Session(config=sess_conf)
    sess.run(tf.global_variables_initializer())
    return sess, discrim


def _select_from_batch(all_data, num_to_select):
    obs = all_data['obs']
    acts = all_data['actions']
    n = len(obs)
    inds = np.random.choice(np.arange(n), size=(num_to_select, ))
    return obs[inds], acts[inds]


def get_obs_act_label_batch(good_demos, bad_demos, batch_size):
    half_batch = max(1, batch_size // 2)
    full_batch = half_batch * 2
    good_obs, good_acts = _select_from_batch(good_demos, half_batch)
    bad_obs, bad_acts = _select_from_batch(bad_demos, half_batch)
    labels = np.zeros((full_batch, ))
    labels[:half_batch] = 1.0
    full_obs = np.concatenate((good_obs, bad_obs), axis=0)
    full_acts = np.concatenate((good_acts, bad_acts), axis=0)
    return full_obs, full_acts, labels


def do_dist_sgd(args, obs_space, act_space, disc_config):
    ray.init(num_cpus=args.cores)
    samples_seen = 0.0
    batch_size = disc_config["batch_size"]
    itr = 0
    elapsed = 0.0
    start = time.time()
    single_batch_size = max(1, batch_size // args.cores)
    good_demos_handle = ray.put(args.good_demos)
    bad_demos_handle = ray.put(args.bad_demos)
    workers = [
        DistSGDActor.remote(good_demos_handle, bad_demos_handle, 1, obs_space,
                            act_space) for i in range(args.cores)
    ]
    # we actually just use this for Adam updates
    master_sess, master_disc = make_sess_disc(obs_space, act_space, args.cores,
                                              disc_config)
    while elapsed < args.train_time_s:
        grads_losses_accs = ray_get_and_free(
            [w.get_grads.remote(single_batch_size) for w in workers])
        all_grads, losses, accs = zip(*grads_losses_accs)
        grads_mean = [np.mean(gs, axis=0) for gs in zip(*all_grads)]
        master_sess.run(master_disc.apply_op,
                        feed_dict=dict(zip(master_disc.grads_ph, grads_mean)))
        new_weights = master_sess.run(master_disc.discrim_train_vars)
        new_weights_h = ray.put(new_weights)
        ray_get_and_free(
            [w.push_weights.remote(new_weights_h) for w in workers])
        ray_get_and_free(new_weights_h)
        samples_seen += len(workers) * 2 * max(1, single_batch_size // 2)
        itr += 1
        disc_loss = np.mean(losses)
        disc_acc = np.mean(accs)
        if (itr % 1000) == 0:
            print("iteration %d, loss %.3g, acc %.3g" %
                  (itr, disc_loss, disc_acc))
        if itr < 10:
            # warm start
            start = time.time()
        elapsed = time.time() - start
    print(
        "dist sgd processed %.2f samples/s; final loss %.3g; final acc %.3g" %
        (samples_seen / elapsed, disc_loss, disc_acc))
    print_done_info(args, master_sess, samples_seen, elapsed, disc_loss,
                    disc_acc)


def do_single_sgd(args, obs_space, act_space, disc_config):
    sess, disc = make_sess_disc(obs_space, act_space, args.cores, disc_config)
    samples_seen = 0.0
    batch_size = disc_config["batch_size"]
    itr = 0
    elapsed = 0.0
    fp_times = []
    while elapsed < args.train_time_s:
        obs_batch, act_batch, label_batch = get_obs_act_label_batch(
            args.good_demos, args.bad_demos, batch_size)
        fp_start = time.time()
        _, disc_loss, disc_acc = sess.run(
            [disc.update_op, disc.loss, disc.accuracy],
            feed_dict={
                disc.obs_t: obs_batch,
                disc.act_t: act_batch,
                disc.is_real_t: label_batch,
                disc.is_training: True,
            })
        fp_elapsed = time.time() - fp_start
        fp_times.append(fp_elapsed)
        samples_seen += len(obs_batch)
        itr += 1
        if (itr % 1000) == 0:
            print("iteration %d, loss %.3g, acc %.3g (mean fp time %.3g)" %
                  (itr, disc_loss, disc_acc, np.mean(fp_times)))
            fp_times = fp_times[-10:]
        if itr < 10:
            # warm start
            start = time.time()
        elapsed = time.time() - start
    print(
        "single sgd processed %.2f samples/s; final loss %.3g; final acc %.3g"
        % (samples_seen / elapsed, disc_loss, disc_acc))
    # also print in CSV format
    print_done_info(args, sess, samples_seen, elapsed, disc_loss, disc_acc)


def print_done_info(args, sess, samples_seen, elapsed, disc_loss, disc_acc):
    num_gpus = sum(dev.device_type == 'GPU' for dev in sess.list_devices())
    info = [
        # comment to stop yapf reformatting
        ("type", args.expt),
        ("hiddens", args.hiddens),
        ("batch_size", args.batch_size),
        ("cores", args.cores),
        ("gpus", num_gpus),
        ("samples_per_s", samples_seen / elapsed),
        ("disc_final_loss", disc_loss),
        ("disc_final_acc", disc_acc),
        ("train_time", args.train_time_s)
    ]
    print("-- CSV --")
    headers = [k for k, v in info]
    # using csv module so we get escaping
    write_header = not os.path.exists(args.csv_out)
    with open(args.csv_out, 'a') as fp:
        writer = csv.DictWriter(fp, headers)
        if write_header:
            writer.writeheader()
        writer.writerow(dict(info))


def main(args):
    if 'TASKSET_CORES' not in os.environ:
        args.cores = max(args.cores, 1)
        core_list = ",".join(map(str, range(args.cores)))
        os.environ['TASKSET_CORES'] = core_list
        os.environ['MKL_NUM_THREADS'] = str(args.cores)
        os.environ['OMP_NUM_THREADS'] = str(args.cores)
        ts_args = ["/usr/bin/taskset", "-c", core_list, "python3", *sys.argv]
        print("No TASKSET_CORES, running taskset with args", ts_args)
        time.sleep(0.1)
        # I tried using execv instead but it broke taskset (some inscrutable
        # message about not being able to set affinity when I only gave it core
        # 0)
        return subprocess.call(ts_args)

    print("Running", args.expt)
    disc_config = {
        "lr": 1e-3,
        "batch_size": args.batch_size,
        "model": {
            "fcnet_hiddens": list(map(int, args.hiddens.split(','))),
            "fcnet_activation": "relu"
        }
    }
    env = gym.make(ENV_NAME)
    act_space = env.action_space
    obs_space = env.observation_space
    del env
    if args.expt == 'single':
        do_single_sgd(args, obs_space, act_space, disc_config)
    elif args.expt == 'dist':
        do_dist_sgd(args, obs_space, act_space, disc_config)
    else:
        assert False, "idk about %s" % (args.expt, )
    return 0


parser = argparse.ArgumentParser()
parser.add_argument('--expt', choices=('single', 'dist'), default='single')
parser.add_argument('--good-demos', default='hc-good.json', type=load_demo)
parser.add_argument('--bad-demos', default='hc-bad.json', type=load_demo)
parser.add_argument('--csv-out', default='all-small-expts.csv')
parser.add_argument('--cores', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--hiddens', default='128,128', type=str)
# TODO: raise this up to something more realistic
parser.add_argument('--train-time-s', default=60, type=float)

if __name__ == '__main__':
    exit(main(parser.parse_args()))

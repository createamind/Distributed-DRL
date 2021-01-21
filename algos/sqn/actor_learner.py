from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numbers import Number
import pickle

import time
import datetime
import ray
import ray.experimental.tf_utils

import core
from core import get_vars


class Learner(object):
    def __init__(self, opt, job):
        self.opt = opt
        with tf.Graph().as_default():
            tf.set_random_seed(opt.seed)
            np.random.seed(opt.seed)

            # Inputs to computation graph
            self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(opt.obs_dim, None, opt.obs_dim, None, None)

            # Main outputs from computation graph
            with tf.variable_scope('main'):
                self.mu, self.pi, entropy_x2, q1, q2, q1_mu, q2_mu = core.q_function(self.x_ph, self.x2_ph, opt.alpha,
                                                                           opt.hidden_size, opt.act_dim)

            # Target value network
            with tf.variable_scope('target'):
                mu_, pi_, entropy_x2_, q1_, q2_, q1_mu_, q2_mu_ = core.q_function(self.x2_ph, self.x2_ph, opt.alpha,
                                                                                  opt.hidden_size, opt.act_dim)

            # Count variables
            var_counts = tuple(core.count_vars(scope) for scope in ['main'])
            print('\nNumber of parameters: total: %d\n' % var_counts)

            a_one_hot = tf.one_hot(tf.cast(self.a_ph, tf.int32), depth=opt.act_dim)
            q1_a = tf.reduce_sum(q1 * a_one_hot, axis=1)
            q2_a = tf.reduce_sum(q2 * a_one_hot, axis=1)

            # Min Double-Q:
            min_q_target = tf.minimum(q1_mu_, q2_mu_)

            # Bellman backup for Q functions
            v_backup = tf.stop_gradient(min_q_target - opt.alpha * entropy_x2)
            q_backup = self.r_ph + opt.gamma * (1 - self.d_ph) * v_backup

            # q losses
            # q_loss = 0.5 * tf.reduce_mean((q_backup - q_value) ** 2)
            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_a) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_a) ** 2)
            q_loss = q1_loss + q2_loss

            # Value train op
            # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
            value_optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr)
            value_params = get_vars('main/q')
            train_value_op = value_optimizer.minimize(q_loss, var_list=value_params)

            # Polyak averaging for target variables
            # (control flow because sess.run otherwise evaluates in nondeterministic order)
            with tf.control_dependencies([train_value_op]):
                target_update = tf.group([tf.assign(v_targ, opt.polyak * v_targ + (1 - opt.polyak) * v_main)
                                          for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

            # All ops to call during one training step
            self.step_ops = [q_loss, q1, q2, train_value_op, target_update]

            # Initializing targets to match main variables
            self.target_init = tf.group([tf.assign(v_targ, v_main)
                                    for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

            if job == "learner":
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = opt.gpu_fraction
                config.inter_op_parallelism_threads = 1
                config.intra_op_parallelism_threads = 1
                self.sess = tf.Session(config=config)
            else:
                self.sess = tf.Session(
                    config=tf.ConfigProto(
                        # device_count={'GPU': 0},
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1))

            self.sess.run(tf.global_variables_initializer())

            if job == "learner":
                # Set up summary Ops
                self.train_ops, self.train_vars = self.build_summaries()
                self.writer = tf.summary.FileWriter(
                    opt.summary_dir + "/" + "^^^^^^^^^^" + str(datetime.datetime.now()) + opt.env_name + "-" +
                    opt.exp_name + "-workers_num:" + str(opt.num_workers) + "%" + str(opt.a_l_ratio), self.sess.graph)

            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                q_loss, self.sess)

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))
        self.sess.run(self.target_init)

    def get_weights(self):
        weights = self.variables.get_weights()
        keys = [key for key in list(weights.keys()) if "main" in key]
        values = [weights[key] for key in keys]
        return keys, values

    def train(self, batch, cnt):

        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done'],
                     }

        outs = self.sess.run(self.step_ops, feed_dict)
        if cnt % 500 == 0:
            summary_str = self.sess.run(self.train_ops, feed_dict={
                self.train_vars[0]: outs[0],
                self.train_vars[1]: np.mean(outs[1])
            })

            self.writer.add_summary(summary_str, cnt)
            self.writer.flush()

    def compute_gradients(self, x, y):
        pass

    def apply_gradients(self, gradients):
        pass

    # Tensorflow Summary Ops
    def build_summaries(self):
        train_summaries = []
        LossQ = tf.Variable(0.)
        train_summaries.append(tf.summary.scalar("LossQ", LossQ))
        QVals = tf.Variable(0.)
        train_summaries.append(tf.summary.scalar("QVals", QVals))
        train_ops = tf.summary.merge(train_summaries)
        train_vars = [LossQ, QVals]

        return train_ops, train_vars


class Actor(object):
    def __init__(self, opt, job):
        self.opt = opt
        with tf.Graph().as_default():
            tf.set_random_seed(opt.seed)
            np.random.seed(opt.seed)

            # Inputs to computation graph
            self.x_ph, self.a_ph, self.x2_ph, = core.placeholders(opt.obs_dim, None, opt.obs_dim)

            # Main outputs from computation graph
            with tf.variable_scope('main'):
                self.mu, self.pi, entropy_x2, q1, q2, q1_mu, q2_mu = core.q_function(self.x_ph, self.x2_ph, opt.alpha,
                                                                           opt.hidden_size, opt.act_dim)

            # Set up summary Ops
            self.test_ops, self.test_vars = self.build_summaries()

            self.sess = tf.Session(
                config=tf.ConfigProto(
                    device_count={'GPU': 0},
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1))

            self.sess.run(tf.global_variables_initializer())

            if job == "test":
                self.writer = tf.summary.FileWriter(
                    opt.summary_dir + "/" + str(datetime.datetime.now()) + "-" + opt.env_name + "-" + opt.exp_name +
                    "-workers_num:" + str(opt.num_workers) + "%" + str(opt.a_l_ratio), self.sess.graph)

            variables_all = tf.contrib.framework.get_variables_to_restore()
            variables_bn = [v for v in variables_all if 'moving_mean' in v.name or 'moving_variance' in v.name]

            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                self.mu, self.sess, input_variables=variables_bn)

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variables.get_weights()
        keys = [key for key in list(weights.keys()) if "main" in key]
        values = [weights[key] for key in keys]
        return keys, values

    def get_action(self, o, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: np.expand_dims(o, axis=0)})[0]

    # Tensorflow Summary Ops
    def build_summaries(self):
        test_summaries = []
        episode_reward = tf.Variable(0.)
        episode_score = tf.Variable(0.)
        a_l_ratio = tf.Variable(0.)
        update_frequency = tf.Variable(0.)
        test_summaries.append(tf.summary.scalar("Reward", episode_reward))
        test_summaries.append(tf.summary.scalar("score", episode_score))
        test_summaries.append(tf.summary.scalar("a_l_ratio", a_l_ratio))
        test_summaries.append(tf.summary.scalar("update_frequency", update_frequency))
        test_ops = tf.summary.merge(test_summaries)
        test_vars = [episode_reward, episode_score, a_l_ratio, update_frequency]

        return test_ops, test_vars

    def write_tb(self, ave_test_reward, ave_score, alratio, update_frequency, last_learner_step):
        summary_str = self.sess.run(self.test_ops, feed_dict={
            self.test_vars[0]: ave_test_reward,
            self.test_vars[1]: ave_score,
            self.test_vars[2]: alratio,
            self.test_vars[3]: update_frequency
        })

        self.writer.add_summary(summary_str, last_learner_step)
        self.writer.flush()

    def test(self, test_env, n=10):

        test_rets = []
        scores = []

        for _ in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0

            while True:
                a = self.get_action(o, deterministic=True)
                # Step the env
                o, r, d, _ = test_env.step(a)

                ep_ret += r
                ep_len += 1

                if d:
                    test_rets.append(ep_ret)
                    scores.append(test_env.rewards[0])
                    # print('test_ep_len:', ep_len, 'test_ep_ret:', ep_ret)
                    break
        return np.mean(test_rets), np.mean(scores)




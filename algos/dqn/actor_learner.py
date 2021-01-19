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
                self.q, self.q_x2 = core.q_function(self.x_ph, self.x2_ph, opt.hidden_size, opt.act_dim)

            # Target value network
            with tf.variable_scope('target'):
                self.q_next, _ = core.q_function(self.x_ph, self.x2_ph, opt.hidden_size, opt.act_dim)

            # Count variables
            var_counts = tuple(core.count_vars(scope) for scope in ['main'])
            print('\nNumber of parameters: total: %d\n' % var_counts)

            a_one_hot = tf.one_hot(tf.cast(self.a_ph, tf.int32), depth=opt.act_dim)
            q_value = tf.reduce_sum(self.q * a_one_hot, axis=1)

            # DDQN
            x2_a = tf.argmax(self.q_x2, axis=1)
            x2_a_one_hot = tf.one_hot(tf.cast(x2_a, tf.int32), depth=opt.act_dim)
            q_target = tf.reduce_sum(self.q_next * x2_a_one_hot, axis=1)

            # DQN
            # q_target = tf.reduce_max(q_next, axis=1)

            # Bellman backup for Q functions, using Clipped Double-Q targets
            q_backup = tf.stop_gradient(self.r_ph + opt.gamma * (1 - self.d_ph) * q_target)

            # q losses
            q_loss = 0.5 * tf.reduce_mean((q_backup - q_value) ** 2)

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
            self.step_ops = [q_loss, self.q, train_value_op, target_update]

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
                self.q, _ = core.q_function(self.x_ph, self.x2_ph, opt.hidden_size, opt.act_dim)

            self.sess = tf.Session(
                config=tf.ConfigProto(
                    # device_count={'GPU': 0},
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1))

            self.sess.run(tf.global_variables_initializer())

            variables_all = tf.contrib.framework.get_variables_to_restore()
            variables_bn = [v for v in variables_all if 'moving_mean' in v.name or 'moving_variance' in v.name]

            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                self.q, self.sess, input_variables=variables_bn)

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variables.get_weights()
        keys = [key for key in list(weights.keys()) if "main" in key]
        values = [weights[key] for key in keys]
        return keys, values

    def get_action(self, o):
        if np.random.uniform() < 0.97:
            o = o[np.newaxis, :]
            actions_value = self.sess.run(self.q, feed_dict={self.x_ph: o})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.opt.act_dim)
        return action

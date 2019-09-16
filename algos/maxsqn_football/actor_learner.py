from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numbers import Number
import gym

import gfootball.env as football_env

import datetime
import time
import ray
import ray.experimental.tf_utils

import core
from core import get_vars
from core import actor_critic


class Learner(object):
    def __init__(self, opt, job):
        self.opt = opt
        with tf.Graph().as_default():
            tf.set_random_seed(opt.seed)
            np.random.seed(opt.seed)

            # Inputs to computation graph
            self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = \
                core.placeholders_from_space(opt.obs_space, opt.act_space, opt.obs_space, None, None)

            # ------
            # TODO BUG: TypeError: can't pickle _thread.RLock objects
            if opt.alpha == 'auto':
                log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
                alpha_v = tf.exp(log_alpha)
            else:
                alpha_v = opt.alpha
            # ------

            # Main outputs from computation graph
            with tf.variable_scope('main'):
                mu, pi, logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi, q1_mu, q2_mu \
                    = actor_critic(self.x_ph, self.x2_ph, self.a_ph, alpha_v,
                                   action_space=opt.ac_kwargs['action_space'], model=opt.model)

            # Target value network
            with tf.variable_scope('target'):
                _, _, logp_pi_, _,  _, _,q1_pi_, q2_pi_,q1_mu_, q2_mu_= \
                    actor_critic(self.x2_ph, self.x2_ph, self.a_ph, alpha_v,
                                 action_space=opt.ac_kwargs['action_space'], model=opt.model)

            # Count variables
            var_counts = tuple(core.count_vars(scope) for scope in
                               ['main/pi', 'main/q1', 'main/q2', 'main'])
            print(('\nNumber of parameters: \t pi: %d, \t' + 'q1: %d, \t q2: %d, \t total: %d\n') % var_counts)

            # ------
            if isinstance(alpha_v, tf.Tensor):
                alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi_ + opt.target_entropy))

                alpha_optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, name='alpha_optimizer')
                train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])
            # ------

            # Min Double-Q:
            # min_q_pi = tf.minimum(q1_mu_, q2_mu_)
            min_q_pi = tf.minimum(q1_pi_, q2_pi_)

            # Targets for Q and V regression
            v_backup = tf.stop_gradient(min_q_pi - alpha_v * logp_pi2)
            q_backup = self.r_ph + opt.gamma * (1 - self.d_ph) * v_backup

            # Soft actor-critic losses
            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
            self.value_loss = q1_loss + q2_loss

            value_optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr)
            value_params = get_vars('main/q')

            train_value_op = value_optimizer.minimize(self.value_loss, var_list=value_params)

            # Polyak averaging for target variables
            # (control flow because sess.run otherwise evaluates in nondeterministic order)
            with tf.control_dependencies([train_value_op]):
                target_update = tf.group([tf.assign(v_targ, opt.polyak * v_targ + (1 - opt.polyak) * v_main)
                                          for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

            # All ops to call during one training step
            if isinstance(alpha_v, Number):
                self.step_ops = [q1_loss, q2_loss, q1, q2, logp_pi_, tf.identity(alpha_v),
                                 train_value_op, target_update]
            else:
                self.step_ops = [q1_loss, q2_loss, q1, q2, logp_pi_, alpha_v,
                                 train_value_op, target_update, train_alpha_op]

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
                        device_count={'GPU': 0},
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
                self.value_loss, self.sess)

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

        summary_str = self.sess.run(self.train_ops, feed_dict={
            self.train_vars[0]: outs[0],
            self.train_vars[1]: outs[1],
            self.train_vars[2]: np.mean(outs[2]),
            self.train_vars[3]: np.mean(outs[3]),
            self.train_vars[4]: np.mean(outs[4]),
            self.train_vars[5]: outs[5],
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
        LossQ1 = tf.Variable(0.)
        train_summaries.append(tf.summary.scalar("LossQ1", LossQ1))
        LossQ2 = tf.Variable(0.)
        train_summaries.append(tf.summary.scalar("LossQ2", LossQ2))
        Q1Vals = tf.Variable(0.)
        train_summaries.append(tf.summary.scalar("Q1Vals", Q1Vals))
        Q2Vals = tf.Variable(0.)
        train_summaries.append(tf.summary.scalar("Q2Vals", Q2Vals))
        LogPi = tf.Variable(0.)
        train_summaries.append(tf.summary.scalar("LogPi", LogPi))
        Alpha = tf.Variable(0.)
        train_summaries.append(tf.summary.scalar("Alpha", Alpha))

        train_ops = tf.summary.merge(train_summaries)
        train_vars = [LossQ1, LossQ2, Q1Vals, Q2Vals, LogPi, Alpha]
        # train_vars = [LossQ1, LossQ2, Alpha]

        return train_ops, train_vars


class Actor(object):
    def __init__(self, opt, job):
        self.opt = opt
        with tf.Graph().as_default():
            tf.set_random_seed(opt.seed)
            np.random.seed(opt.seed)

            # Inputs to computation graph
            self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = \
                core.placeholders_from_space(opt.obs_space, opt.act_space, opt.obs_space, None, None)

            # ------
            if opt.alpha == 'auto':
                log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
                alpha_v = tf.exp(log_alpha)
            else:
                alpha_v = opt.alpha
            # ------

            # Main outputs from computation graph
            with tf.variable_scope('main'):
                self.mu, self.pi, logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi, q1_mu, q2_mu \
                    = actor_critic(self.x_ph, self.x2_ph, self.a_ph, alpha_v,
                                   action_space=opt.ac_kwargs['action_space'], model=opt.model)

            # Set up summary Ops
            self.test_ops, self.test_vars = self.build_summaries()

            self.sess = tf.Session(
                config=tf.ConfigProto(
                    device_count={'GPU': 0},
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1))

            self.sess.run(tf.global_variables_initializer())

            if job == "main":
                self.writer = tf.summary.FileWriter(
                    opt.summary_dir + "/" + str(datetime.datetime.now()) + "-" + opt.env_name + "-" + opt.exp_name +
                    "-workers_num:" + str(opt.num_workers) + "%" + str(opt.a_l_ratio), self.sess.graph)

            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                self.pi, self.sess)

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

    def test(self, test_env, replay_buffer, n=100):
        rew = []
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == self.opt.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            rew.append(ep_ret)

        sample_times, _, _, _ = ray.get(replay_buffer.get_counts.remote())
        summary_str = self.sess.run(self.test_ops, feed_dict={
            self.test_vars[0]: sum(rew)/n
        })

        self.writer.add_summary(summary_str, sample_times)
        self.writer.flush()
        return sum(rew)/n

    # Tensorflow Summary Ops
    def build_summaries(self):
        test_summaries = []
        episode_reward = tf.Variable(0.)
        test_summaries.append(tf.summary.scalar("Reward", episode_reward))

        test_ops = tf.summary.merge(test_summaries)
        test_vars = [episode_reward]

        return test_ops, test_vars

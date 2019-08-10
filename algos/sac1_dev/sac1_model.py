from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numbers import Number
import gym
import datetime
import time
import ray
import ray.experimental.tf_utils

import core
from core import get_vars
from core import mlp_actor_critic as actor_critic


class Sac1(object):
    def __init__(self, opt, job):
        self.opt = opt
        with tf.Graph().as_default():
            tf.set_random_seed(opt.seed)
            np.random.seed(opt.seed)

            # Inputs to computation graph
            self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = \
                core.placeholders(opt.obs_dim, opt.act_dim, opt.obs_dim, None, None)

            # Main outputs from computation graph
            with tf.variable_scope('main'):
                self.mu, self.pi, logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi = \
                    actor_critic(self.x_ph, self.x2_ph, self.a_ph, action_space=opt.ac_kwargs["action_space"])

            # Target value network
            with tf.variable_scope('target'):
                _, _, logp_pi_, _, _, _, q1_pi_, q2_pi_ = \
                    actor_critic(self.x2_ph, self.x2_ph, self.a_ph, action_space=opt.ac_kwargs["action_space"])

            # Count variables
            var_counts = tuple(core.count_vars(scope) for scope in
                               ['main/pi', 'main/q1', 'main/q2', 'main'])
            print(('\nNumber of parameters: \t pi: %d, \t' + 'q1: %d, \t q2: %d, \t total: %d\n')%var_counts)

        ######
            if opt.alpha == 'auto':
                target_entropy = (-np.prod(opt.action_space.shape))

                log_alpha = tf.get_variable( 'log_alpha', dtype=tf.float32, initializer=0.0)
                alpha = tf.exp(log_alpha)

                alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + target_entropy))

                alpha_optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, name='alpha_optimizer')
                train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])
        ######

            # Min Double-Q:
            min_q_pi = tf.minimum(q1_pi_, q2_pi_)

            # Targets for Q and V regression
            v_backup = tf.stop_gradient(min_q_pi - opt.alpha * logp_pi2)
            q_backup = self.r_ph + opt.gamma*(1-self.d_ph)*v_backup

            # Soft actor-critic losses
            pi_loss = tf.reduce_mean(opt.alpha * logp_pi - q1_pi)
            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
            self.value_loss = q1_loss + q2_loss

            # Policy train op
            # (has to be separate from value train op, because q1_pi appears in pi_loss)
            pi_optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr)
            train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

            # Value train op
            # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
            value_optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr)
            value_params = get_vars('main/q')
            with tf.control_dependencies([train_pi_op]):
                train_value_op = value_optimizer.minimize(self.value_loss, var_list=value_params)

            # Polyak averaging for target variables
            # (control flow because sess.run otherwise evaluates in nondeterministic order)
            with tf.control_dependencies([train_value_op]):
                self.target_update = tf.group([tf.assign(v_targ, opt.polyak*v_targ + (1-opt.polyak)*v_main)
                                          for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

            # TODO
            # self.grads = self.optimizer.compute_gradients(self.cross_entropy)
            # self.grads_placeholder = [(tf.placeholder(
            #     "float", shape=grad[1].get_shape()), grad[1])
            #     for grad in self.grads]

            # All ops to call during one training step
            if isinstance(opt.alpha, Number):
                self.step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, tf.identity(opt.alpha),
                        train_pi_op, train_value_op, self.target_update]
            else:
                self.step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, opt.alpha,
                        train_pi_op, train_value_op, self.target_update, train_alpha_op]

            # Set up summary Ops
            self.test_ops, self.test_vars = self.build_summaries()

            # Initializing targets to match main variables
            self.target_init = tf.group([tf.assign(v_targ, v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
            if job == 'learner':
                self.sess = tf.Session(
                    config=tf.ConfigProto(
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1))
            else:
                self.sess = tf.Session(
                    config=tf.ConfigProto(
                        device_count={'GPU': 0},
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1))
            self.sess.run(tf.global_variables_initializer())

            if job == "main":
                self.writer = tf.summary.FileWriter(
                    opt.summary_dir + "/" + str(datetime.datetime.now()) + "-" + opt.env_name + "-workers_num:" + str(
                        opt.num_workers), self.sess.graph)

            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                self.value_loss, self.sess)

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))
        # TODO self.sess.run(self.target_update) which way to update target parameters
        # self.sess.run(self.target_update)
        self.sess.run(self.target_init)

    def get_weights(self):
        weights = self.variables.get_weights()
        keys = [key for key in list(weights.keys()) if "main" in key]
        values = [weights[key] for key in keys]
        return keys, values

    def get_action(self, o, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: o.reshape(1, -1)})[0]

    def compute_gradients(self, x, y):
        pass

    def apply_gradients(self, gradients):
        pass

    def parameter_update(self, batch):
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done'],
                     }
        self.sess.run(self.step_ops, feed_dict)

    def test_agent(self, start_time, replay_buffer, n=25):
        test_env = gym.make(self.opt.env_name)
        rew = []
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == self.opt.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            rew.append(ep_ret)

        sample_times, _, _ = ray.get(replay_buffer.get_counts.remote())
        summary_str = self.sess.run(self.test_ops, feed_dict={
            self.test_vars[0]: sum(rew)/25,
            self.test_vars[1]: sample_times
        })
        current_time = time.time()
        self.writer.add_summary(summary_str, current_time-start_time)
        self.writer.flush()
        return sum(rew)/25

    # Tensorflow Summary Ops
    def build_summaries(self):
        test_summaries = []
        episode_reward = tf.Variable(0.)
        sample_times = tf.Variable(0.)
        test_summaries.append(tf.summary.scalar("Reward", episode_reward))
        test_summaries.append(tf.summary.scalar("Sample_times", sample_times))

        test_ops = tf.summary.merge(test_summaries)
        test_vars = [episode_reward, sample_times]

        return test_ops, test_vars

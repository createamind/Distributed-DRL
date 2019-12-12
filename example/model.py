import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger
from core import mlp_actor_critic as actor_critic
import ray.experimental.tf_utils


class Model(object):

    def __init__(self, args):

        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(args.obs_dim, args.act_dim,
                                                                                   args.obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(self.x_ph, self.a_ph, **args.ac_kwargs)

        # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, v_targ = actor_critic(self.x2_ph, self.a_ph, **args.ac_kwargs)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in
                           ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t' + 'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n') % var_counts)

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(self.r_ph + args.gamma * (1 - self.d_ph) * v_targ)
        v_backup = tf.stop_gradient(min_q_pi - args.alpha * logp_pi)

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(args.alpha * logp_pi - q1_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - v) ** 2)
        self.value_loss = q1_loss + q2_loss + v_loss

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        value_params = get_vars('main/q') + get_vars('main/v')
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(self.value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, args.polyak * v_targ + (1 - args.polyak) * v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        self.step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi,
                         train_pi_op, train_value_op, target_update]

        # Initializing targets to match main variables
        self.target_init = tf.group([tf.assign(v_targ, v_main)
                                     for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
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

    def get_action(self, o, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: o.reshape(1, -1)})[0]

    def train(self, replay_buffer, args):

        batch = ray.get(replay_buffer.sample_batch.remote(args.batch_size))
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done'],
                     }
        outs = self.sess.run(self.step_ops, feed_dict)
        # logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
        #              LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
        #              VVals=outs[6], LogPi=outs[7])

    def test_agent(self, test_env, args, n=10):
        test_ret = []
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == args.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            test_ret.append(ep_ret)
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        ave_ret = sum(test_ret)/len(test_ret)
        return ave_ret

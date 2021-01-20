import numpy as np
import tensorflow as tf

EPS = 1e-8


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


"""
Actor-Critics
"""


def q_function(x, x2, hidden_sizes, act_dim, activation=tf.nn.relu, output_activation=None):

    vf_mlp = lambda x: mlp(x, list(hidden_sizes) + [act_dim], activation, None)
    # Q
    q_tp = tf.make_template('q1', vf_mlp, create_scope_now_=True)

    q = q_tp(x)
    q_x2 = q_tp(x2)

    return q, q_x2

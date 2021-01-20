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


def softmax_policy(alpha, q, act_dim):

    pi_log = tf.nn.log_softmax(q/alpha, axis=1)
    mu = tf.argmax(pi_log, axis=1)

    # tf.random.multinomial( logits, num_samples, seed=None, name=None, output_dtype=None )
    # logits: 2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :] represents the unnormalized log-probabilities for all classes.
    # num_samples: 0-D. Number of independent samples to draw for each row slice.
    pi = tf.squeeze(tf.random.multinomial(pi_log, 1), axis=1)

    # logp_pi = tf.reduce_sum(tf.one_hot(mu, depth=act_dim) * pi_log, axis=1)  # use max Q(s,a)
    # logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * pi_log, axis=1)
    logp_pi = tf.reduce_sum(tf.exp(pi_log)*pi_log, axis=1)                     # exact entropy

    return mu, pi, logp_pi


"""
Actor-Critics
"""


def q_function(x, x2, alpha, hidden_sizes, act_dim, activation=tf.nn.relu,
                     output_activation=None, policy=softmax_policy, action_space=None):

    vf_mlp = lambda x: mlp(x, list(hidden_sizes) + [act_dim], activation, None)

    # Q1
    q1_tp = tf.make_template('q1', vf_mlp, create_scope_now_=True)

    q1 = q1_tp(x)

    # policy
    mu, pi, entropy = policy(alpha, q1, act_dim)
    q1_mu = tf.reduce_sum(q1 * tf.one_hot(mu, depth=act_dim), axis=1)

    q1_x2 = q1_tp(x2)

    # policy
    mu_x2, pi_x2, entropy_x2 = policy(alpha, q1_x2, act_dim)

    # Q2
    q2_tp = tf.make_template('q2', vf_mlp, create_scope_now_=True)
    q2 = q2_tp(x)

    # policy
    mu2, pi2, entropy2 = policy(alpha, q2, act_dim)
    q2_mu = tf.reduce_sum(q2 * tf.one_hot(mu2, depth=act_dim), axis=1)

    return mu, pi, entropy_x2, q1, q2, q1_mu, q2_mu

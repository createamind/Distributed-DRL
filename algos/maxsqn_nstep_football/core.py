import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
from baselines.a2c.utils import fc, conv, conv_to_fc

EPS = 1e-8


def placeholder(dim=None):
    if dim is None:
        return tf.placeholder(dtype=tf.float32, shape=(None,))
    elif len(dim) == 0:
        return tf.placeholder(dtype=tf.int32, shape=((None,) + dim))       # for Discrete
    else:
        return tf.placeholder(dtype=tf.float32, shape=((None,) + dim))     # for Box


def placeholders(*args):
    return [placeholder(dim) for dim in args]


initializer_kernel = tf.variance_scaling_initializer(2.0)
# Parameter Regularization
regularizer_l2 = tf.contrib.layers.l2_regularizer


# Batch Normalization
def dense_batch_relu(inputs, units, activation, phase, coefficent_regularizer):
    x = tf.layers.dense(inputs, units, activation=activation,
                         kernel_regularizer=regularizer_l2(coefficent_regularizer), bias_regularizer=regularizer_l2(coefficent_regularizer),
                         kernel_initializer=initializer_kernel)
    x = tf.contrib.layers.batch_norm(x,
                                      center=True, scale=True,
                                      is_training=phase, fused=False)
    if activation:
        x = activation(x)
    return x


def mlp(x, hidden_sizes=(32,), activation=None, output_activation=None, use_bn=False, phase=True, coefficent_regularizer=0.0):
    # MLP with batch normalization
    if use_bn:
        for h in hidden_sizes[:-1]:
            x = dense_batch_relu(x, units=h, activation=activation, phase=phase, coefficent_regularizer=coefficent_regularizer)
        return dense_batch_relu(x, units=hidden_sizes[-1], activation=output_activation, phase=phase, coefficent_regularizer=coefficent_regularizer)
    # Vanilla MLP
    else:
        for h in hidden_sizes[:-1]:
            # x = tf.layers.dense(x, units=h, activation=activation)
            x = tf.layers.dense(x, units=h, activation=activation,
                                kernel_regularizer=regularizer_l2(coefficent_regularizer),
                                bias_regularizer=regularizer_l2(coefficent_regularizer),
                                kernel_initializer=initializer_kernel)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation,
                               kernel_regularizer=regularizer_l2(coefficent_regularizer),
                               bias_regularizer=regularizer_l2(coefficent_regularizer),
                               kernel_initializer=initializer_kernel)


def nature_cnn(unscaled_images,  **conv_kwargs):
    """
    CNN from Nature paper.
    """
    # hidden_sizes = (32,),
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    h4 = activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))
    return fc(h4, 'fc2', nh=21)


def _get_filter_config(shape):
    shape = list(shape)

    filters = [
        [32, [8, 8], 4],
        [64, [4, 4], 2],
        [64, [3, 3], 1],
    ]
    if len(shape) == 3:
        return filters
    else:
        raise ValueError(
            "input shape do not match conv_filters!"
        )


def _build_layers_v2(input_dict, num_outputs, options):
    inputs = input_dict["obs"]
    # filters = options.get("conv_filters")
    filters = _get_filter_config(inputs.shape.as_list()[1:])

    with tf.name_scope("Conv_net"):
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            inputs = tf.layers.conv2d(
                inputs,
                out_size,
                kernel,
                stride,
                activation=tf.nn.relu,
                padding="VALID",
                name="conv{}".format(i))
        out_size, kernel, stride = filters[-1]

        conv3 = tf.layers.conv2d(
            inputs,
            out_size,
            kernel,
            stride,
            activation=tf.nn.relu,
            padding="valid",
            name="conv3")

    conv3_flat = tf.layers.flatten(conv3)

    with tf.name_scope("fc_net"):
        # label = "fcn{}".format(i)
        fcn4 = tf.layers.dense(
            conv3_flat,
            512,
            kernel_initializer=normc_initializer(1.0),
            activation=tf.nn.relu,
            name="fcn4v")
        fcnv = tf.layers.dense(
            fcn4,
            units=1,
            kernel_initializer=normc_initializer(1.0),
            activation=None,
            name="fcnv")
        fcna = tf.layers.dense(
            fcn4,
            units=num_outputs,
            kernel_initializer=normc_initializer(1.0),
            activation=None,
            name="fcna")

        q_values = fcnv + tf.subtract(fcna, tf.reduce_mean(fcna, axis=1, keepdims=True))
    return q_values


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


"""
Policies
"""


def softmax_policy(alpha, v_x, act_dim):

    pi_log = tf.nn.log_softmax(v_x/alpha, axis=1)
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


def actor_critic(x, x2,  a, alpha, hidden_sizes, activation=tf.nn.relu,
                 output_activation=None,
                 use_bn=False, phase=True, coefficent_regularizer=0.0,
                 policy=softmax_policy, action_space=None, model="mlp"):

    if x.shape[1] == 128:                # for Breakout-ram-v4
        x = (x - 128.0) / 128.0          # x: shape(?,128)

    act_dim = action_space.n
    a_one_hot = tf.one_hot(a, depth=act_dim)      # shape(?,4)
    #vfs
    if model == "mlp":
        vf_model = lambda x: mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation, use_bn=use_bn, phase=phase, coefficent_regularizer=coefficent_regularizer)     # return: shape(?,4)
    else:
        vf_model = lambda x: nature_cnn(x)
    # Q1

    ################# Q1

    q1_tp = tf.make_template('q1', vf_model, create_scope_now_=True)

    v1_x = q1_tp(x)

    # policy
    mu, pi, logp_pi = policy(alpha, v1_x, act_dim)

    mu_one_hot = tf.one_hot(mu, depth=act_dim)
    pi_one_hot = tf.one_hot(pi, depth=act_dim)

    q1 = tf.reduce_sum(v1_x * a_one_hot, axis=1)

    q1_mu = tf.reduce_sum(v1_x * mu_one_hot, axis=1)  # use max Q(s,a)
    q1_pi = tf.reduce_sum(v1_x * pi_one_hot, axis=1)

    v1_x2 = q1_tp(x2)

    # policy
    mu_x2, pi_x2, logp_pi_x2 = policy(alpha, v1_x2, act_dim)

    # Q2

    ################# Q2

    q2_tp = tf.make_template('q2', vf_model, create_scope_now_=True)

    v2_x = q2_tp(x)

    mu2, pi2, logp_pi2 = policy(alpha, v2_x, act_dim)

    mu2_one_hot = tf.one_hot(mu2, depth=act_dim)

    q2 = tf.reduce_sum(v2_x * a_one_hot, axis=1)

    q2_mu = tf.reduce_sum(v2_x * mu2_one_hot, axis=1)  # use max Q(s,a)
    q2_pi = tf.reduce_sum(v2_x * pi_one_hot, axis=1)

    # shape(?,)
    return mu, pi, logp_pi, logp_pi_x2, q1, q2, q1_pi, q2_pi, q1_mu, q2_mu



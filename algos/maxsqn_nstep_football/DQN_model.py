import ray

from ray.rllib.models.model import Model
# from ray.rllib.models.misc import get_activation_fn, flatten
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf
from ray.rllib.models.misc import normc_initializer, get_activation_fn
tf = try_import_tf()


class DuelingDQN(Model):
    """Example of a custom model.
    This model just delegates to the built-in fcnet.
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
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
            # with tf.name_scope("fc_net"):
            #     # label = "fcn{}".format(i)
            #     fcnv4 = tf.layers.dense(
            #         conv3_flat,
            #         512,
            #         kernel_initializer=normc_initializer(1.0),
            #         activation=tf.nn.relu,
            #         name="fcn4v")
            #     fcnv = tf.layers.dense(
            #         fcnv4,
            #         units=1,
            #         kernel_initializer=normc_initializer(1.0),
            #         activation=None,
            #         name="fcnv")
            #     fcna4 = tf.layers.dense(
            #         conv3_flat,
            #         512,
            #         kernel_initializer=normc_initializer(1.0),
            #         activation=tf.nn.relu,
            #         name="fcn4v")
            #     fcna = tf.layers.dense(
            #         fcna4,
            #         units=num_outputs,
            #         kernel_initializer=normc_initializer(1.0),
            #         activation=None,
            #         name="fcna")
            q_values = fcnv + tf.subtract(fcna, tf.reduce_mean(fcna, axis=1, keepdims=True))
            # output = tf.argmax(q_values, 1)

        return q_values, fcn4


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

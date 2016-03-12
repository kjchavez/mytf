import mytensorflow as mytf

import tensorflow as tf
from .utils import _variable_on_device, _add_l2_reg
import operator

# TODO(kjchavez): Consider letting the input be None, in which case we simply
# put in a named placeholder.


class ConvLayer(object):
    """ A single convolutional layer for 2D data. """
    def __init__(self, name, input, depth,
                 kernel_size=(3, 3),
                 stride=[1, 1, 1, 1],
                 padding='SAME',
                 device=mytf.DEFAULT_DEVICE,
                 W_init=tf.truncated_normal_initializer(stddev=1e-4),
                 activation_fn=tf.nn.relu,
                 b_init=tf.constant_initializer(0.0)):

        # If stride is specified just as a single integer, we assume it refers
        # to the spatial axes.
        if isinstance(stride, int):
            stride = [1, stride, stride, 1]

        with tf.variable_scope(name) as scope:
            shape = list(kernel_size) + [input.get_shape()[3].value, depth]
            self.kernel = _variable_on_device('W', device, shape, W_init)
            conv = tf.nn.conv2d(input, self.kernel, stride, padding='SAME')
            self.biases = _variable_on_device('b', device, [depth], b_init)
            bias = tf.nn.bias_add(conv, self.biases)
            self.activations = activation_fn(bias, name=scope.name)

            # Add summaries of activations and sparsity. Note, these will be
            # added to the default summary key and accessible through
            # tf.merge_all_summaries().
            tf.histogram_summary(name + '/activations', self.activations)
            tf.scalar_summary(name + '/sparsity',
                              tf.nn.zero_fraction(self.activations))

    def get_train_feed_dict(self):
        return {}

    def get_eval_feed_dict(self):
        return {}


def conv_layer(name, input, depth, **kwargs):
    return ConvLayer(name, input, kwargs)


def conv_layer_with_l2_reg(name, input, depth, weight, **kwargs):
    conv = ConvLayer(name, input, depth, **kwargs)
    _add_l2_reg(conv.kernel, weight)
    return conv


# TODO(kjchavez): use fan-in/fan-out to choose a reasonable initialization
# strategy by default.
class FullyConnectedLayer(object):
    def __init__(self, name, input, output_dim,
                 activation_fn=tf.nn.relu,
                 keep_prob=0.5,
                 W_init=tf.truncated_normal_initializer(stddev=1e-4),
                 b_init=tf.constant_initializer(0.1),
                 device=mytf.DEFAULT_DEVICE):

        assert keep_prob > 0 and keep_prob <= 1
        self.keep_prob_value = keep_prob

        with tf.variable_scope(name) as scope:
            # Move everything into depth so we can perform a single matrix
            # multiply.
            input_dim = reduce(operator.mul, input.get_shape()[1:],
                               tf.Dimension(1)).value
            input = tf.reshape(input, [input.get_shape()[0].value, input_dim])

            self.weights = _variable_on_device('weights', device,
                                               [input_dim, output_dim], W_init)
            self.biases = _variable_on_device('biases', device,
                                              [output_dim], b_init)
            activations = activation_fn(tf.matmul(input, self.weights) +
                                        self.biases, name=scope.name)

            # Add a placeholder for the keep probability so we can turn on/off
            # during training/testing.
            self.keep_prob = tf.placeholder(tf.float32)
            self.output = tf.nn.dropout(activations, keep_prob=keep_prob)

            # Add summaries of activations and sparsity. Note, these will be
            # added to the default summary key and accessible through
            # tf.merge_all_summaries().
            tf.histogram_summary(name + '/activations', activations)
            tf.scalar_summary(name + '/sparsity',
                              tf.nn.zero_fraction(activations))

    def get_train_feed_dict(self):
        return {self.keep_prob: self.keep_prob_value}

    def get_eval_feed_dict(self):
        return {self.keep_prob: 1}


def fc_layer(name, input, output_dim, **kwargs):
    return FullyConnectedLayer(name, input, output_dim, **kwargs)


def fc_layer_with_l2_reg(name, input, output_dim, weight, **kwargs):
    fc = FullyConnectedLayer(name, input, output_dim, **kwargs)
    _add_l2_reg(fc.weights, weight)
    return fc

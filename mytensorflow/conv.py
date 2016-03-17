import mytensorflow as mytf
import math
import tensorflow as tf
from .utils import _variable_on_device, _add_l2_reg
import operator

class ConvLayer(object):
    """ A single convolutional layer for 2D data. """
    def __init__(self, name, input_depth, output_depth,
                 kernel_size=(3, 3),
                 stride=[1, 1, 1, 1],
                 padding='SAME',
                 device=mytf.DEFAULT_DEVICE,
                 activation_fn=tf.nn.relu,
                 W_init=None,
                 b_init=None):

        # If stride is specified just as a single integer, we assume it refers
        # to the spatial axes.
        if isinstance(stride, int):
            stride = [1, stride, stride, 1]

        self.activation_fn = activation_fn
        self.padding = padding
        self.stride = stride
        self.depth = output_depth

        if W_init is None:
            N = input_depth * kernel_size[0] * kernel_size[1]
            W_init = tf.truncated_normal_initializer(stddev=math.sqrt(2.0/N))

        if b_init is None:
            b_init = tf.constant_initializer(0.1)

        with tf.variable_scope(name) as scope:
            shape = list(kernel_size) + [input_depth, output_depth]
            self.kernel = _variable_on_device('W', device, shape, W_init)
            self.biases = _variable_on_device('b', device, [output_depth], b_init)

    def transform(self, X, summarize=False):
        conv = tf.nn.conv2d(X, self.kernel, self.stride, padding=self.padding)
        bias = tf.nn.bias_add(conv, self.biases)
        activations = self.activation_fn(bias)

        if summarize:
            # Add summaries of activations and sparsity. Note, these will be
            # added to the default summary key and accessible through
            # tf.merge_all_summaries().
            tf.histogram_summary(self.name + "_" + X.name + '/activations', activations)
            tf.scalar_summary(self.name + "_" + X.name + '/sparsity',
                              tf.nn.zero_fraction(activations))

        return activations


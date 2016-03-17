import math
import operator
import tensorflow as tf
import mytensorflow as mytf
from .utils import _variable_on_device

class FullyConnectedLayer(object):
    def __init__(self, name, input_dim, output_dim,
                 activation_fn=tf.nn.relu,
                 keep_prob=0.5,
                 W_init=None,
                 b_init=None,
                 device=mytf.DEFAULT_DEVICE):

        assert keep_prob > 0 and keep_prob <= 1
        if isinstance(input_dim, (list, tuple)):
            input_dim = reduce(operator.mul, input_dim, 1)

        self.keep_prob_value = keep_prob
        self.input_dim = input_dim
        self.name = name
        self.activation_fn = activation_fn

        if W_init is None:
            # Use reasonable default initialization strategy.
            W_init = tf.truncated_normal_initializer(
                stddev=math.sqrt(2.0/input_dim))

        if b_init is None:
            b_init = tf.constant_initializer(0.0)

        with tf.variable_scope(name) as scope:
            self.weights = _variable_on_device('weights', device,
                                               [input_dim, output_dim], W_init)
            self.biases = _variable_on_device('biases', device,
                                              [output_dim], b_init)
            # Add a placeholder for the keep probability so we can turn on/off
            # during training/testing.
            self.keep_prob = tf.placeholder(tf.float32)
            tf.add_to_collection(mytf.TRAIN_FEED_DICT_FN,
                                 self.get_train_feed_dict)
            tf.add_to_collection(mytf.EVAL_FEED_DICT_FN,
                                 self.get_eval_feed_dict)


    def transform(self, X, summarize=False):
        # Move everything into depth so we can perform a single matrix
        # multiply.
        with tf.name_scope(self.name) as scope:
            input = tf.reshape(X, [X.get_shape()[0].value, self.input_dim])
            activations = self.activation_fn(tf.matmul(input, self.weights) +
                                             self.biases)
            output = tf.nn.dropout(activations, keep_prob=self.keep_prob)

        if summarize:
            # Add summaries of activations and sparsity. Note, these will be
            # added to the default summary key and accessible through
            # tf.merge_all_summaries().
            tf.histogram_summary(self.name + "_" + X.name + '/activations', activations)
            tf.scalar_summary(self.name + "_" + X.name + '/sparsity',
                              tf.nn.zero_fraction(activations))

        return output


    def get_train_feed_dict(self):
        return {self.keep_prob: self.keep_prob_value}

    def get_eval_feed_dict(self):
        return {self.keep_prob: 1}

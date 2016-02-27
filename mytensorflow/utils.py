""" Shared utilities for creating models. """
import tensorflow as tf


def _variable_on_device(name, device, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device(device):
        var = tf.get_variable(name, shape, initializer=initializer)

    return var


def _add_l2_reg(var, weight):
    l2_reg = tf.mul(tf.nn.l2_loss(var), weight, name="l2_loss")
    tf.add_to_collection('losses', l2_reg)


def _add_l1_reg(var, weight):
    l1_reg = tf.mul(tf.nn.l1_loss(var), weight, name="l1_loss")
    tf.add_to_collection('losses', l1_reg)


def get_train_feed_dict(modules):
    feed_dict = {}
    for module in modules:
        feed_dict.update(module.get_train_feed_dict())

    return feed_dict


def get_eval_feed_dict(modules):
    feed_dict = {}
    for module in modules:
        feed_dict.update(module.get_eval_feed_dict())

    return feed_dict

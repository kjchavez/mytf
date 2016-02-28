""" Shared utilities for creating models. """
import tensorflow as tf
import mytensorflow as mytf


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


def encode_one_hot(labels, num_classes):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)
    return onehot_labels


def _get_train_feed_dict(modules):
    feed_dict = {}
    for module in modules:
        feed_dict.update(module.get_train_feed_dict())

    return feed_dict


def _get_eval_feed_dict(modules):
    feed_dict = {}
    for module in modules:
        feed_dict.update(module.get_eval_feed_dict())

    return feed_dict


def _add_loss_summaries(total_loss):
    """Add summaries for losses.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(
                        mytf.MOVING_AVERAGE_DECAY_FOR_LOSS, name='avg')

    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for i, l in enumerate(losses + [total_loss]):
        # Name each loss as '(raw)' and name the moving average version of the
        # loss as the original loss name.
        tf.scalar_summary(l.op.name + str(i) + ' (raw)', l)
        tf.scalar_summary(l.op.name + str(i), loss_averages.average(l))

    return loss_averages_op

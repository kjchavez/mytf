"""
Examples of using mytensorflow to build models.
"""
import mytensorflow as mytf
from mytensorflow.fully_connected import FullyConnectedLayer
import tensorflow as tf

# Two layer neural network.
BATCH_SIZE = 64
dataset = mytf.source.TFRecordSource('dataset', 'housemates.metadata',
                                     BATCH_SIZE)

# Roughly zero mean:
fc1 = FullyConnectedLayer('fc1', dataset.img_shape, 256, keep_prob=1)
fc2 = FullyConnectedLayer('fc2', 256, dataset.num_classes, keep_prob=1)

def inference(data, summarize=False):
    """ Produces logits. """
    X = (1.0 / 255.0) * (data - 127.0)
    h = fc1.transform(X, summarize=summarize)
    return fc2.transform(h, summarize=summarize)

def loss(logits, dense_labels):
    """ Computes cross entropy loss. """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        logits, dense_labels,
                        name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss

def true_count(logits, labels):
    """ Determines accuracy of prediction. """
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    return count

logits = inference(dataset.train_data, summarize=True)
total_loss = loss(logits, dataset.train_dense_labels)

true_count = true_count(inference(dataset.test_data),
                        dataset.test_dense_labels)

def evaluate():
    """ Should only be called inside a tf.Session().
    Returns a formatted string to be displayed as evaluation results.
    """
    num_correct = 0
    total = 0
    for i in xrange(dataset.num_test_examples // dataset.batch_size):
        num_correct += true_count.eval(feed_dict=mytf.utils._get_eval_feed_dict())
        total += dataset.batch_size

    return "Accuracy on validation set: %0.3f%%" % (100*float(num_correct) /
                                                     total)

train_op = mytf.train.get_sgd_train_op(total_loss, lr=1e-3)
mytf.train.train(train_op, evaluate, total_loss, 1e3)

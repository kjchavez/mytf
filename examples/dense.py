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

# Notice that for any sort of Softmax Classification task, the only thing that
# changes is how you produce the logits.

def loss(logits, dense_labels):
    """ Computes cross entropy loss. """
    # Loss (total_loss = cross_entropy_)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        logits, dense_labels,
                        name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return total_loss

def accuracy(logits, labels):
    """ Determines accuracy of prediction. """
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

logits = inference(dataset.train_data, summarize=True)
total_loss = loss(logits, dataset.train_dense_labels)

test_accuracy = accuracy(inference(dataset.test_data),
                         dataset.test_dense_labels)

print ('Training on small dataset. Should heavily overfit.')
t = mytf.trainer.SGDTrainer(total_loss, test_accuracy, dataset, [fc1, fc2], 1e-3)
t.run(10000, print_every_n=100, summarize_every_n=100, eval_every_n=1000)


